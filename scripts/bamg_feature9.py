"""Opt-in nine-channel operator features; no legacy numerical module is changed.

row9: log relative diagonal followed by the eight signed A_ij / d_i entries.
symmetric9: same log channel, followed by A_ij / sqrt(d_i d_j).
Here d_i=max(A_ii, 1e-14); positive diagonals are required. Geometry and support
are supplied separately. The original A, P constraints and Galerkin products
are untouched. Both SciPy and SparseTensor use the same differentiable path.
"""
from __future__ import annotations

import scipy.sparse as sp
import torch
from torch import nn

from adaptive_mg.grid import as_shape
from adaptive_mg.transfer import OFFSETS_9
from adaptive_mg.v67.autograd_sparse import SparseTensor
from adaptive_mg.v67 import bamg_transfer as legacy


FEATURE_VERSION = 'bamg-feature9-v1'
FEATURE_MODES = ('row9', 'symmetric9')
OFFSETS_8 = tuple(offset for offset in OFFSETS_9 if offset != (0, 0))
DIAGONAL_FLOOR = 1e-14
LOG_EPSILON = 1e-12


def validate_mode(feature_mode):
    if feature_mode not in FEATURE_MODES:
        raise ValueError('feature_mode must be row9 or symmetric9')
    return feature_mode


def direct_features9(a, shape, *, feature_mode, device='cpu', dtype=torch.float64):
    """Return (1,9,nx,ny), retaining signed off-diagonals and autograd.

    Positive diagonal validation is not an SPD certificate. The full sparse
    operator remains authoritative; like the original stencil input, this is
    not lossless if A contains entries beyond the eight nearest neighbours.
    """
    validate_mode(feature_mode)
    nx, ny = as_shape(shape)
    if not isinstance(a, SparseTensor):
        if not sp.issparse(a):
            raise TypeError('features require a SciPy sparse matrix or SparseTensor')
        a = SparseTensor.from_scipy(a)
    if a.shape != (nx * ny, nx * ny):
        raise ValueError('operator/grid shape mismatch')
    values = a.values.to(device=device, dtype=dtype)
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError('operator features require finite floating-point values')
    rows = torch.tensor(a.rows, dtype=torch.long, device=device)
    cols = torch.tensor(a.cols, dtype=torch.long, device=device)
    selected = rows == cols
    diagonal = torch.zeros(nx * ny, dtype=dtype, device=device).index_add(
        0, rows[selected], values[selected])
    if not bool((diagonal > 0).all()):
        raise ValueError('feature9 requires a strictly positive diagonal at every node')
    diagonal = diagonal.clamp_min(DIAGONAL_FLOOR)
    median = torch.quantile(diagonal, .5).clamp_min(DIAGONAL_FLOOR)
    channels = [torch.log(diagonal / median + LOG_EPSILON)]
    di, dj = cols // ny - rows // ny, cols % ny - rows % ny
    root = diagonal.sqrt() if feature_mode == 'symmetric9' else None
    for oi, oj in OFFSETS_8:
        keep = (di == oi) & (dj == oj)
        if root is None:
            normalized = values[keep] / diagonal[rows[keep]]
        else:
            # Avoid forming d_i*d_j, which can overflow for finite SPD input.
            normalized = (values[keep] / root[rows[keep]]) / root[cols[keep]]
        channels.append(torch.zeros(nx * ny, dtype=dtype, device=device).index_add(
            0, rows[keep], normalized))
    result = torch.stack(channels).reshape(1, 9, nx, ny)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Nonfinite feature9 normalization')
    return result


class Feature9WeightCNN(legacy.WeightCNN):
    """WeightCNN with nine matrix channels and a self-describing checkpoint spec.

    The geometry (3), early support mask (16), hidden width/depth, activation,
    sixteen output slots, and signed constraint layer retain their old roles.
    """
    def __init__(self, hidden=8, depth=5, activation='relu', geometry=True,
                 mask_mode='early', *, feature_mode):
        nn.Module.__init__(self)
        self.feature_mode = validate_mode(feature_mode)
        self.hidden = legacy._positive_integer(hidden, 'hidden')
        self.depth = legacy._positive_integer(depth, 'depth')
        self.activation, self.geometry = activation, bool(geometry)
        if mask_mode not in ('early', 'late', 'none'):
            raise ValueError('mask_mode must be early, late, or none')
        self.mask_mode = mask_mode
        inputs = 9 + (3 if self.geometry else 0) + (16 if mask_mode == 'early' else 0)
        self.backbone = legacy._backbone(inputs, self.hidden, self.depth, activation)
        self.head = nn.Conv2d(self.hidden + (16 if mask_mode == 'late' else 0), 16, 1)

    @property
    def spec(self):
        return {**super().spec, 'feature_mode': self.feature_mode}

    def encode(self, features, mask=None):
        if features.ndim != 4 or features.shape[1] != 9:
            raise ValueError('feature9 matrix input must have shape (B,9,nx,ny)')
        matrix = (torch.cat((features, legacy._geometry_channels(features)), dim=1)
                  if self.geometry else features)
        if self.mask_mode == 'early':
            matrix = torch.cat((matrix, legacy._spatial_mask(mask, features)), dim=1)
        return self.backbone(matrix)


def transfer_legacy_weights(model, legacy_model):
    """Warm initialization, not an equivalent conversion or optimizer resume.

    Drop only input column 1 (the normalized centre) of the first convolution.
    The separate log-diagonal, geometry and mask kernels retain their order.
    Dropping a padded constant channel changes the function, especially at
    boundaries. Symmetric normalization changes the input semantics as well.
    """
    if not isinstance(model, Feature9WeightCNN):
        raise TypeError('destination must be Feature9WeightCNN')
    expected = dict(model.spec)
    expected.pop('feature_mode')
    if legacy_model.spec != expected:
        raise ValueError('Legacy architecture differs beyond the input feature change')
    state = {key: value.detach().clone() for key, value in legacy_model.state_dict().items()}
    first = state['backbone.0.weight']
    keep = [index for index in range(first.shape[1]) if index != 1]
    state['backbone.0.weight'] = first[:, keep].clone()
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Initialization contains nonfinite parameters')
    model.load_state_dict(state, strict=True)
