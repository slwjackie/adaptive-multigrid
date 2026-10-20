"""Level-conditioned, operator-only line smoothers with cached LDL factors.

The zero head uses an inexpensive tridiagonal LDL initialization, not a neural
prediction of an exact sparse LU.  For wider line blocks it is a *declared band
approximation*.  Every sweep still updates the residual using the full actual
operator.  Positive pivots make each generated line matrix SPD; they do not
certify convergence of the complete multigrid iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import math

import numpy as np
import torch
from torch import nn

from ..models import make_spatial_body
from ..smoothers import _zebra_order


class LineSmootherCNN(nn.Module):
    training_only = False

    def __init__(self, hidden=8, body_kind='residual5', coefficient_scale=.25,
                 pivot_scale=.5, pivot_floor=1e-12, version=1):
        super().__init__()
        if version != 1 or type(hidden) is not int or hidden < 1:
            raise ValueError('Invalid line smoother version/width')
        if not (0 < coefficient_scale <= 2 and 0 < pivot_scale <= 2
                and 0 < pivot_floor < 1):
            raise ValueError('Invalid line factor bounds')
        self.hidden, self.body_kind = hidden, body_kind
        self.coefficient_scale = float(coefficient_scale)
        self.pivot_scale, self.pivot_floor = float(pivot_scale), float(pivot_floor)
        self.body = make_spatial_body(body_kind, 12, hidden)
        self.head = nn.Conv2d(hidden, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def line_smoother_spec(self):
        return dict(hidden=self.hidden, body_kind=self.body_kind,
                    coefficient_scale=self.coefficient_scale,
                    pivot_scale=self.pivot_scale, pivot_floor=self.pivot_floor,
                    version=1)

    def forward(self, features, index=0, total_levels=2):
        if features.ndim != 4 or features.shape[1] != 10:
            raise ValueError('Line generator requires 10 operator feature channels')
        if type(index) is not int or type(total_levels) is not int or not 0 <= index < total_levels-1:
            raise ValueError('Line generator is defined only on nonterminal levels')
        size = math.log2(max(features.shape[-2:]) + 1.) / 8.
        position = index / (total_levels - 1)
        extra = features.new_empty((features.shape[0], 2, *features.shape[-2:]))
        extra[:, 0] = position
        extra[:, 1] = size
        return self.head(self.body(torch.cat((features, extra), dim=1)))

    def coefficients(self, a, shape, features, index=0, total_levels=2):
        raw = self(features, index, total_levels)
        return factors_from_raw(a, shape, raw,
            coefficient_scale=self.coefficient_scale, pivot_scale=self.pivot_scale,
            pivot_floor=self.pivot_floor)


def _entries(a):
    if hasattr(a, 'pattern') and hasattr(a, 'values'):
        return a.rows, a.cols, a.values.to('cpu', dtype=torch.float64)
    a = a.tocsr(copy=False)
    return (np.repeat(np.arange(a.shape[0]), np.diff(a.indptr)), a.indices,
            torch.as_tensor(a.data.copy(), dtype=torch.float64))


def factors_from_raw(a, shape, raw, *, coefficient_scale=.25, pivot_scale=.5,
                     pivot_floor=1e-12):
    """Differentiable actual-A initialization followed by bounded CNN changes.

    x arrays have shape (ny,nx), y arrays (nx,ny).  All factor arithmetic and
    applications are FP64 even when generation uses FP32 convolutions.
    """
    nx, ny = map(int, shape)
    if tuple(raw.shape) != (1, 4, nx, ny):
        raise ValueError('Line head must return (1,4,nx,ny)')
    rr, cc, values = _entries(a)
    fields = []
    for mask in (rr == cc, rr - cc == ny,
                 ((rr - cc == 1) & (rr // ny == cc // ny))):
        rows = torch.as_tensor(rr[mask], dtype=torch.long)
        selected = values[torch.as_tensor(mask)]
        fields.append(values.new_zeros(nx*ny).index_add(0, rows, selected).reshape(nx, ny))
    diagonal, sub_x, sub_y = fields
    if not bool(torch.isfinite(values).all()) or not bool((diagonal > 0).all()):
        raise ValueError('Line factors require finite positive-diagonal A')
    raw = raw[0].to('cpu', dtype=torch.float64)
    result = {}
    for direction, base_d, base_e, dl, dd in (
        ('x', diagonal.T, sub_x.T, raw[0].T, raw[1].T),
        ('y', diagonal, sub_y, raw[2], raw[3])):
        pivots, lowers = [], []
        for i in range(base_d.shape[1]):
            if i:
                lower = base_e[:, i] / pivots[-1]
                pivot = base_d[:, i] - base_e[:, i] * lower
            else:
                lower = torch.zeros_like(base_d[:, 0])
                pivot = base_d[:, 0]
            # For an SPD tridiagonal block this clamp is inactive. A wider
            # block's tridiagonal truncation need not itself be SPD.
            pivots.append(torch.maximum(pivot, pivot_floor*base_d[:, i]))
            lowers.append(lower)
        lower = torch.stack(lowers, dim=1)
        pivot = torch.stack(pivots, dim=1)
        free = torch.ones_like(lower)
        free[:, 0] = 0.
        lower = lower + coefficient_scale * torch.tanh(dl) * free
        pivot = pivot * torch.exp(pivot_scale * torch.tanh(dd))
        if not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(pivot).all()):
            raise ValueError('Nonfinite generated line factors')
        result[direction] = (lower, pivot)
    return result


def line_indices(shape, direction):
    nx, ny = shape
    grid = np.arange(nx*ny, dtype=np.int64).reshape(nx, ny)
    return grid.T if direction == 'x' else grid


def line_groups(a, shape, direction):
    """Return (lines,length) index arrays in original zebra sweep order.

    Torch sparse graphs inspect symbolic support, so even a numerical zero
    with a nonzero derivative cannot incorrectly enable independent batching.
    """
    lines = line_indices(shape, direction)
    if hasattr(a, 'pattern'):
        rows, cols = a.rows, a.cols
    else:
        coo = a.tocoo(copy=False)
        nonzero = coo.data != 0
        rows, cols = coo.row[nonzero], coo.col[nonzero]
    ids = np.empty(shape[0]*shape[1], dtype=np.int64)
    ids[lines] = np.arange(len(lines))[:, None]
    coupled = np.any((ids[rows] != ids[cols]) & (ids[rows] % 2 == ids[cols] % 2))
    if coupled:
        return [lines[j:j+1] for j in _zebra_order(len(lines), False)]
    return [lines[c::2] for c in (0, 1) if len(lines[c::2])]


def _solve_torch(lower, pivot, residual):
    # residual: (lines,length,probes), with an explicit singleton probe axis.
    y = [residual[:, 0]]
    for i in range(1, residual.shape[1]):
        y.append(residual[:, i] - lower[:, i, None]*y[-1])
    z = torch.stack(y, dim=1) / pivot[:, :, None]
    x = [z[:, -1]]
    for i in range(residual.shape[1]-2, -1, -1):
        x.append(z[:, i] - lower[:, i+1, None]*x[-1])
    return torch.stack(list(reversed(x)), dim=1)


def apply_line_smoother_torch(a, shape, factors, residual, reverse=False):
    """Apply x/y line sweeps, preserving gradients through factors AND A."""
    vector = residual.ndim == 1
    if residual.ndim not in (1, 2):
        raise ValueError('Residual must be (N,) or (N,probes)')
    r = residual[:, None] if vector else residual
    correction, current = torch.zeros_like(r), r
    for direction in (('y', 'x') if reverse else ('x', 'y')):
        groups = line_groups(a, shape, direction)
        lower, pivot = factors[direction]
        for indices in (reversed(groups) if reverse else groups):
            rows = torch.as_tensor(indices.ravel(), dtype=torch.long)
            ids = indices[:, 0] % shape[1] if direction == 'x' else indices[:, 0] // shape[1]
            ids = torch.as_tensor(ids, dtype=torch.long)
            values = _solve_torch(lower[ids], pivot[ids], current[rows].reshape(*indices.shape, r.shape[1]))
            update = torch.zeros_like(r).index_add(0, rows, values.reshape(-1, r.shape[1]))
            correction = correction + update
            current = current - a.apply(update)
    return correction[:, 0] if vector else correction


@dataclass
class LineSmootherBank:
    a: object
    shape: tuple
    factors: dict
    gain: float = 1.
    is_line_smoother: bool = True

    def __post_init__(self):
        self.groups = {}
        for direction in ('x', 'y'):
            self.groups[direction] = []
            for indices in line_groups(self.a, self.shape, direction):
                rows = indices.ravel()
                ids = indices[:, 0] % self.shape[1] if direction == 'x' else indices[:, 0] // self.shape[1]
                lower, pivot = self.factors[direction]
                self.groups[direction].append((indices, lower[ids], pivot[ids], self.a[:, rows].tocsr()))

    def apply(self, residual, stats, reverse=False):
        start = perf_counter()
        r = np.asarray(residual, dtype=np.float64)
        if r.ndim != 1 or r.size != self.a.shape[0]:
            raise ValueError('Runtime line residual must have shape (N,)')
        correction, current = np.zeros_like(r), r.copy()
        work = 0
        for direction in (('y', 'x') if reverse else ('x', 'y')):
            groups = self.groups[direction]
            for indices, lower, pivot, columns in (reversed(groups) if reverse else groups):
                y = current[indices].copy()
                for i in range(1, y.shape[1]):
                    y[:, i] -= lower[:, i] * y[:, i-1]
                y /= pivot
                for i in range(y.shape[1]-2, -1, -1):
                    y[:, i] -= lower[:, i+1] * y[:, i+1]
                correction[indices] += y
                current -= np.asarray(columns @ y.ravel()).ravel()
                work += 2*columns.nnz + 5*indices.size
                stats.replaced_line_solves += len(indices)
        result = self.gain * correction
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite learned line correction')
        stats.work_flops += work
        stats.neural_apply_calls += 1
        stats.neural_apply_seconds += perf_counter()-start
        return result
