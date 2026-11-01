"""Direct interpolation for the fixed-support, two-CNN BAMG experiment.

Only bounded candidate slots are represented. Geometry determines boundary
row sums and injection; bilinear interpolation values are never a decoder
prior. Structural support is explicit and survives zero-valued coefficients.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from ..grid import as_shape, coarsening_ratios, next_shape
from ..transfer import OFFSETS_9, TRANSFER_SLOT_OFFSETS, TransferPattern
from .autograd_sparse import SparseTensor

FEATURE_VERSION = "bamg-stencil10-v1"
CANDIDATE_VERSION = "bamg-local16-v1"
CONSTRAINT_VERSION = "bamg-geometry-signed-l1-v1"


def _immutable(a):
    a = np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@lru_cache(maxsize=16)
def _pattern(fine, coarse):
    nx, ny = fine
    cx, cy = coarse
    if min(nx, ny, cx, cy) < 1:
        raise ValueError("grid dimensions must be positive")
    rx, ry = coarsening_ratios(fine, coarse)
    if (nx + 1) != rx * (cx + 1) or (ny + 1) != ry * (cy + 1):
        raise ValueError("fine and coarse grid coordinates are not nested")
    ii = np.repeat(np.arange(1, nx + 1), ny)
    jj = np.tile(np.arange(1, ny + 1), nx)
    offsets = np.asarray(TRANSFER_SLOT_OFFSETS, dtype=np.int64)
    ci = ii[:, None] // rx + offsets[None, :, 0]
    cj = jj[:, None] // ry + offsets[None, :, 1]
    valid = (ci >= 1) & (ci <= cx) & (cj >= 1) & (cj <= cy)
    columns = np.where(valid, (ci - 1) * cy + cj - 1, -1)
    # This compatibility field deliberately contains no bilinear weights.
    # The new path accesses columns/shape only, and has its own row targets.
    return TransferPattern(fine, coarse, _immutable(columns),
                           _immutable(np.zeros(columns.shape, np.float64)))


def direct_pattern(shape, coarse_shape=None):
    """Build all valid 4x4 candidate slots without constructing a base P."""
    fine = as_shape(shape)
    coarse = (next_shape(fine, "full", 3) if coarse_shape is None
              else as_shape(coarse_shape))
    return _pattern(fine, coarse)


def geometry_constraints(pattern):
    """Return ``(row_sum, injection_slot, valid)`` from grid coordinates.

    The eliminated homogeneous-Dirichlet boundary has half mass in a boundary
    direction (a quarter at corners under full coarsening). Injection slots
    are -1 for non-injection rows. No interpolation weight tensor is read.
    """
    nx, ny = pattern.fine_shape
    cx, cy = pattern.coarse_shape
    rx, ry = coarsening_ratios(pattern.fine_shape, pattern.coarse_shape)
    columns = np.asarray(pattern.columns)
    if columns.shape != (nx * ny, len(TRANSFER_SLOT_OFFSETS)):
        raise ValueError("expected sixteen candidate slots per fine row")
    valid = columns >= 0
    if np.any(columns[valid] >= cx * cy):
        raise ValueError("candidate column lies outside the coarse grid")
    ii = np.repeat(np.arange(1, nx + 1), ny)
    jj = np.tile(np.arange(1, ny + 1), nx)
    xi, yj = ii / rx, jj / ry
    tx = np.clip(np.minimum(np.minimum(1., xi), cx + 1 - xi), 0., 1.)
    ty = np.clip(np.minimum(np.minimum(1., yj), cy + 1 - yj), 0., 1.)
    targets = tx * ty
    is_injection = (ii % rx == 0) & (jj % ry == 0)
    coincident_column = (ii // rx - 1) * cy + jj // ry - 1
    match = (columns == coincident_column[:, None]) & valid
    if np.any(match[is_injection].sum(1) != 1):
        raise ValueError("missing or duplicate identity-injection candidate")
    injection = np.full(nx * ny, -1, np.int64)
    injection[is_injection] = match[is_injection].argmax(1)
    return targets, injection, valid


def row_slots(output, pattern):
    """Convert one problem's CNN output to the stable ``(N,16)`` order."""
    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(np.asarray(output))
    n, k = pattern.columns.shape
    if output.shape == (n, k):
        return output
    if output.shape == (1, k, *pattern.fine_shape):
        return output[0].permute(1, 2, 0).reshape(n, k)
    if output.shape == (k, *pattern.fine_shape):
        return output.permute(1, 2, 0).reshape(n, k)
    raise ValueError("candidate tensor must be (N,16) or (1,16,nx,ny)")


def _mask(mask, pattern, device):
    tensor = row_slots(mask, pattern)
    if tensor.requires_grad:
        raise ValueError("support mask must be frozen")
    if tensor.dtype != torch.bool:
        if not bool(((tensor == 0) | (tensor == 1)).all()):
            raise ValueError("support mask must contain only zero or one")
    result = tensor.to(device=device, dtype=torch.bool)
    _, injection, valid = geometry_constraints(pattern)
    valid_t = torch.tensor(valid, device=device)
    if bool((result & ~valid_t).any()):
        raise ValueError("support mask selects invalid candidates")
    rows = np.flatnonzero(injection >= 0)
    if len(rows):
        ri = torch.tensor(rows, device=device)
        ci = torch.tensor(injection[rows], device=device)
        if not bool(result[ri, ci].all()) or not bool((result[ri].sum(1) == 1).all()):
            raise ValueError("support mask must preserve identity injection")
    return result


def select_mask(scores, pattern, k=4):
    """Hard valid top-k, with exact injection and coarse-column tie breaking.

    This produces a bool tensor and deliberately supplies no surrogate gradient.
    Supervised selection learns logits directly; stage 2 reuses saved masks.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= 16:
        raise ValueError("k must be an integer in [1,16]")
    scores = row_slots(scores, pattern)
    if not scores.is_floating_point():
        scores = scores.to(torch.float64)
    _, injection, valid = geometry_constraints(pattern)
    valid_t = torch.tensor(valid, device=scores.device)
    if not bool(torch.isfinite(scores[valid_t]).all()):
        raise ValueError("valid selection scores must be finite")
    cols = torch.tensor(np.asarray(pattern.columns), device=scores.device)
    # Secondary key first, followed by a stable score sort.
    column_order = torch.argsort(torch.where(valid_t, cols, pattern.n_coarse_unknowns),
                                dim=1, stable=True)
    safe_scores = torch.where(valid_t, scores, -torch.inf)
    ordered = safe_scores.gather(1, column_order)
    score_order = torch.argsort(ordered, dim=1, descending=True, stable=True)
    winners = column_order.gather(1, score_order[:, :int(k)])
    selected = torch.zeros_like(valid_t).scatter(1, winners, True) & valid_t
    rows = np.flatnonzero(injection >= 0)
    if len(rows):
        ri = torch.tensor(rows, device=scores.device)
        ci = torch.tensor(injection[rows], device=scores.device)
        selected[ri] = False
        selected[ri, ci] = True
    if bool((selected.sum(1) == 0).any()):
        raise ValueError("no valid coarse candidate for a fine row")
    return selected


def fixed_bilinear_mask(pattern, *, device="cpu"):
    """Geometry-only support for the fixed-support/direct-weight ablation.

    Select the enclosing coarse cell corners (one point at a coincident coarse
    coordinate), dropping eliminated Dirichlet boundary points. This supplies
    only structural locations: direct signed values still come from the Weight
    CNN and are never initialized by or added to bilinear interpolation values.
    """
    nx, ny = pattern.fine_shape
    _, cy = pattern.coarse_shape
    rx, ry = coarsening_ratios(pattern.fine_shape, pattern.coarse_shape)
    ii = np.repeat(np.arange(1, nx + 1), ny)
    jj = np.tile(np.arange(1, ny + 1), nx)
    columns = np.asarray(pattern.columns)
    ci, cj = columns // cy + 1, columns % cy + 1
    lower_i, upper_i = ii // rx, (ii + rx - 1) // rx
    lower_j, upper_j = jj // ry, (jj + ry - 1) // ry
    selected = ((columns >= 0)
                & ((ci == lower_i[:, None]) | (ci == upper_i[:, None]))
                & ((cj == lower_j[:, None]) | (cj == upper_j[:, None])))
    if np.any(selected.sum(1) == 0):
        raise ValueError("fixed bilinear support contains an empty row")
    return _mask(torch.tensor(selected, device=device), pattern, device)


def project_weights(raw, mask, pattern, l1_cap=8.):
    """Differentiable signed feasibility retraction on an explicit support.

    First impose the row sum, then radially contract its zero-sum component
    about a uniform feasible row until its L1 norm reaches the cap. The exact
    radial boundary uses sorted prefixes, so autograd includes the derivative
    of the contraction factor (unlike detached bisection). This is not claimed
    to be a Euclidean projection. No bilinear weights or positivity constraint
    enter the construction. It is differentiable away from ordinary active-set
    boundaries/ties, which are excluded from smooth finite-difference checks.
    """
    raw = row_slots(raw, pattern)
    if not raw.is_floating_point() or not bool(torch.isfinite(raw).all()):
        raise ValueError("raw weights must be finite floating-point values")
    if not np.isfinite(l1_cap) or l1_cap <= 0:
        raise ValueError("l1_cap must be finite and positive")
    selected = _mask(mask, pattern, raw.device)
    targets, injection, _ = geometry_constraints(pattern)
    target = torch.tensor(targets, dtype=raw.dtype, device=raw.device)
    if bool((target.abs() > l1_cap).any()):
        raise ValueError("L1 cap is incompatible with row-sum/injection constraints")
    count = selected.sum(1)
    if bool(((count == 0) & (target != 0)).any()):
        raise ValueError("empty support cannot reproduce the row target")
    degree = count.clamp_min(1).to(raw.dtype)
    zeroed = torch.where(selected, raw, 0.)
    direction = torch.where(selected, zeroed - zeroed.sum(1, keepdim=True) / degree[:, None], 0.)
    uniform = target / degree
    particular = selected.to(raw.dtype) * uniform[:, None]

    # For t>=0 and p_i=t/d, ||p+alpha*v||_1 <= cap iff for every
    # sorted prefix with negative sum: alpha <= (cap-t+2*q*p)/(-2*sum(v)).
    sorted_v = torch.sort(torch.where(selected, direction, torch.inf), dim=1).values
    ranks = torch.arange(1, raw.shape[1] + 1, dtype=raw.dtype, device=raw.device)[None, :]
    prefix_valid = ranks <= count[:, None]
    sorted_v = torch.where(prefix_valid, sorted_v, 0.)
    cumulative = sorted_v.cumsum(1)
    active = prefix_valid & (cumulative < 0.)
    denominator = torch.where(active, -2. * cumulative, 1.)
    limits = (l1_cap - target[:, None] + 2. * ranks * uniform[:, None]) / denominator
    limits = torch.where(active, limits, torch.inf)
    alpha = limits.amin(1).clamp(min=0., max=1.)
    out = particular + alpha[:, None] * direction
    # Restore the tiny summation roundoff without changing structural support.
    out = out + selected.to(raw.dtype) * ((target - out.sum(1)) / degree)[:, None]
    fixed = torch.tensor(injection >= 0, device=raw.device)
    injection_values = torch.zeros_like(out)
    rows = np.flatnonzero(injection >= 0)
    if len(rows):
        injection_values[torch.tensor(rows, device=raw.device),
                         torch.tensor(injection[rows], device=raw.device)] = 1.
    return torch.where(fixed[:, None], injection_values, out)


def _selected_entries(pattern, weights, mask):
    weights = row_slots(weights, pattern)
    selected = _mask(mask, pattern, weights.device)
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("interpolation weights must be finite")
    if bool((weights[~selected] != 0).any()):
        raise ValueError("interpolation weights outside the frozen mask must be zero")
    chosen = selected.detach().cpu().numpy()
    rows = np.broadcast_to(np.arange(pattern.n_fine_unknowns)[:, None], chosen.shape)[chosen]
    cols = np.asarray(pattern.columns)[chosen]
    return rows, cols, weights[selected]


def scipy_from_weights(pattern, weights, mask):
    """Assemble only selected coordinates, retaining explicit numerical zeros."""
    rows, cols, values = _selected_entries(pattern, weights, mask)
    values = values.detach().cpu().double().numpy()
    matrix = sp.coo_matrix((values, (rows, cols)),
                          shape=(pattern.n_fine_unknowns, pattern.n_coarse_unknowns)).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def sparse_from_weights(pattern, weights, mask):
    """Graph-preserving CPU FP64 assembly for the sparse autograd backend."""
    rows, cols, values = _selected_entries(pattern, weights, mask)
    return SparseTensor.from_coo(rows, cols, values.to(device="cpu", dtype=torch.float64),
                                 (pattern.n_fine_unknowns, pattern.n_coarse_unknowns))


def direct_features(a, shape, device="cpu", dtype=torch.float64):
    """Ten actual-operator features with identical runtime/autograd arithmetic.

    Includes log diagonal scale and nine normalized local stencil entries.
    These are not a lossless representation of a coarse operator with fill.
    """
    nx, ny = as_shape(shape)
    if not isinstance(a, SparseTensor):
        if not sp.issparse(a):
            raise TypeError("features require a SciPy sparse matrix or SparseTensor")
        a = SparseTensor.from_scipy(a)
    if a.shape != (nx * ny, nx * ny):
        raise ValueError("operator/grid shape mismatch")
    values = a.values.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("operator contains nonfinite values")
    rows = torch.tensor(a.rows, device=device, dtype=torch.long)
    cols = torch.tensor(a.cols, device=device, dtype=torch.long)
    diagonal_sel = rows == cols
    diagonal = torch.zeros(nx * ny, dtype=dtype, device=device).index_add(
        0, rows[diagonal_sel], values[diagonal_sel]).abs().clamp_min(1e-14)
    median = torch.quantile(diagonal, .5).clamp_min(1e-14)
    channels = [torch.log(diagonal / median + 1e-12)]
    di, dj = cols // ny - rows // ny, cols % ny - rows % ny
    for oi, oj in OFFSETS_9:
        keep = (di == oi) & (dj == oj)
        channels.append(torch.zeros(nx * ny, dtype=dtype, device=device).index_add(
            0, rows[keep], values[keep] / diagonal[rows[keep]]))
    return torch.stack(channels).reshape(1, 10, nx, ny)


def _activation(name):
    choices = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
               "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}
    if name not in choices:
        raise ValueError(f"unsupported activation {name!r}")
    return choices[name]()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _geometry_channels(features):
    nx, ny = features.shape[-2:]
    pattern = direct_pattern((nx, ny))
    target, _, _ = geometry_constraints(pattern)
    rx, ry = coarsening_ratios(pattern.fine_shape, pattern.coarse_shape)
    ii = torch.arange(1, nx + 1, device=features.device, dtype=features.dtype)
    jj = torch.arange(1, ny + 1, device=features.device, dtype=features.dtype)
    phase_x = (ii.remainder(rx) / rx)[:, None].expand(nx, ny)
    phase_y = (jj.remainder(ry) / ry)[None, :].expand(nx, ny)
    sums = torch.tensor(target, device=features.device, dtype=features.dtype).reshape(nx, ny)
    geometry = torch.stack((phase_x, phase_y, sums)).unsqueeze(0)
    return geometry.expand(features.shape[0], -1, -1, -1)


def _matrix_input(features, geometry):
    if features.ndim != 4 or features.shape[1] != 10:
        raise ValueError("matrix features must have shape (B,10,nx,ny)")
    return torch.cat((features, _geometry_channels(features)), dim=1) if geometry else features


def _spatial_mask(mask, features):
    if mask is None:
        raise ValueError("this Weight CNN requires an explicit support mask")
    mask = torch.as_tensor(mask, device=features.device)
    if mask.requires_grad:
        raise ValueError("Weight CNN support input must be frozen")
    b, _, nx, ny = features.shape
    if mask.shape == (nx * ny, 16) and b == 1:
        mask = mask.reshape(nx, ny, 16).permute(2, 0, 1).unsqueeze(0)
    elif mask.shape == (b, nx * ny, 16):
        mask = mask.reshape(b, nx, ny, 16).permute(0, 3, 1, 2)
    elif mask.shape != (b, 16, nx, ny):
        raise ValueError("support mask shape does not match the matrix features")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("Weight CNN support input must be binary")
    return mask.to(dtype=features.dtype)


def _backbone(inputs, hidden, depth, activation):
    layers = []
    for _ in range(depth):
        layers.extend((nn.Conv2d(inputs, hidden, 3, padding=1), _activation(activation)))
        inputs = hidden
    return nn.Sequential(*layers)


class MaskCNN(nn.Module):
    """Independent selection logits; ``depth`` counts 3x3 convolutions."""
    def __init__(self, hidden=8, depth=5, activation="relu", geometry=True):
        super().__init__()
        self.hidden = _positive_integer(hidden, "hidden")
        self.depth = _positive_integer(depth, "depth")
        self.activation = activation
        self.geometry = bool(geometry)
        self.backbone = _backbone(13 if self.geometry else 10, self.hidden,
                                  self.depth, activation)
        self.head = nn.Conv2d(self.hidden, 16, 1)

    @property
    def spec(self):
        return dict(kind="mask", hidden=self.hidden, depth=self.depth,
                    activation=self.activation, geometry=self.geometry)

    def forward(self, features):
        return self.head(self.backbone(_matrix_input(features, self.geometry)))


class WeightCNN(nn.Module):
    """Independent direct signed weights conditioned on the selected support.

    ``early`` is the main model. ``late`` exposes independent matrix feature
    extraction via encode(), followed by a mask-conditioned decode(). ``none``
    is an explicitly different architecture for parallelism ablations.
    """
    def __init__(self, hidden=8, depth=5, activation="relu", geometry=True,
                 mask_mode="early"):
        super().__init__()
        self.hidden = _positive_integer(hidden, "hidden")
        self.depth = _positive_integer(depth, "depth")
        self.activation = activation
        self.geometry = bool(geometry)
        if mask_mode not in ("early", "late", "none"):
            raise ValueError("mask_mode must be early, late, or none")
        self.mask_mode = mask_mode
        inputs = (13 if self.geometry else 10) + (16 if mask_mode == "early" else 0)
        self.backbone = _backbone(inputs, self.hidden, self.depth, activation)
        self.head = nn.Conv2d(self.hidden + (16 if mask_mode == "late" else 0), 16, 1)

    @property
    def spec(self):
        return dict(kind="weight", hidden=self.hidden, depth=self.depth,
                    activation=self.activation, geometry=self.geometry,
                    mask_mode=self.mask_mode)

    def encode(self, features, mask=None):
        matrix = _matrix_input(features, self.geometry)
        if self.mask_mode == "early":
            matrix = torch.cat((matrix, _spatial_mask(mask, features)), dim=1)
        return self.backbone(matrix)

    def decode(self, encoded, mask=None):
        if self.mask_mode == "late":
            encoded = torch.cat((encoded, _spatial_mask(mask, encoded)), dim=1)
        return self.head(encoded)

    def forward(self, features, mask=None):
        return self.decode(self.encode(features, mask), mask)


def model_from_spec(spec):
    """Rebuild architecture without unpickling a Python model object."""
    values = dict(spec)
    kind = values.pop("kind", None)
    if kind == "mask":
        return MaskCNN(**values)
    if kind == "weight":
        return WeightCNN(**values)
    raise ValueError("model spec must declare kind='mask' or kind='weight'")
