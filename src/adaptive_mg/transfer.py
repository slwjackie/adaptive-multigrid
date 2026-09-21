"""Classical, operator-dependent, and learned structured-grid transfers.

The learned decoder predicts corrections on an expanded 4x4 coarse-neighbour
support (up to 16 candidates per fine node).  Restriction remains tied by
``R=P.T`` and coarse operators use the Galerkin product ``P.T @ A @ P``.
Candidate channels are fixed row-major signed offsets ``(-1,0,1,2)^2`` from
the floor coarse anchor; boundary-invalid slots retain column ``-1`` and
weight ``0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import scipy.sparse as sp
import torch

from .grid import (
    CoarseningKind,
    GridShape,
    as_shape,
    coarsening_ratios,
    next_shape,
    unknowns,
    validate_root_shape,
)

TransferBaseline = Literal["bilinear", "operator"]
OFFSETS_9: tuple[tuple[int, int], ...] = (
    (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)
N_MATRIX_FEATURES = 1 + len(OFFSETS_9)
# A slot is identified by its signed coarse-grid offset from
# ``floor(fine_global_index / coarsening_ratio)``.  Keeping this order fixed is
# important: a TransferNet output channel must have the same geometric meaning
# at odd/even fine nodes and at interior/boundary nodes.
TRANSFER_AXIS_OFFSETS: tuple[int, int, int, int] = (-1, 0, 1, 2)
TRANSFER_SLOT_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (di, dj) for di in TRANSFER_AXIS_OFFSETS for dj in TRANSFER_AXIS_OFFSETS
)
MAX_TRANSFER_CANDIDATES = len(TRANSFER_SLOT_OFFSETS)
# A full-coarsening hierarchy from 1023 down to the default 3-point terminal
# grid has eight non-terminal patterns.  Keeping one such hierarchy hot avoids
# unbounded process-wide retention when experiments sweep many grid shapes.
TRANSFER_PATTERN_CACHE_MAXSIZE = 8


@dataclass(frozen=True)
class TransferPattern:
    fine_shape: GridShape
    coarse_shape: GridShape
    columns: np.ndarray
    bilinear_weights: np.ndarray

    @property
    def n_fine_unknowns(self) -> int:
        return unknowns(self.fine_shape)

    @property
    def n_coarse_unknowns(self) -> int:
        return unknowns(self.coarse_shape)

    @property
    def n_candidates(self) -> int:
        return int(self.columns.shape[1])

    # Compatibility properties for older square-grid callers.
    @property
    def n_fine(self) -> int:
        if self.fine_shape[0] != self.fine_shape[1]:
            raise ValueError("n_fine is only defined for square grids")
        return self.fine_shape[0]

    @property
    def n_coarse(self) -> int:
        if self.coarse_shape[0] != self.coarse_shape[1]:
            raise ValueError("n_coarse is only defined for square grids")
        return self.coarse_shape[0]

    @property
    def base_weights(self) -> np.ndarray:
        return self.bilinear_weights


def _immutable_array(values: np.ndarray) -> np.ndarray:
    """Return a C-contiguous array backed by immutable bytes.

    Merely clearing NumPy's ``WRITEABLE`` flag is reversible for an owning
    array.  Cached transfer patterns are shared by all callers, so their arrays
    use an immutable backing buffer as well as a read-only view.
    """

    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _transfer_column_dtype(coarse_shape: GridShape) -> np.dtype:
    """Return the narrowest signed dtype that can also represent ``-1``."""

    # The largest valid flattened column is n_coarse_unknowns - 1.  Perform
    # the size calculation with Python integers so the safety check itself
    # cannot overflow.
    n_coarse_unknowns = int(coarse_shape[0]) * int(coarse_shape[1])
    if n_coarse_unknowns <= int(np.iinfo(np.int32).max) + 1:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def is_multigrid_size(n: int, coarsest_n: int = 3) -> bool:
    try:
        validate_root_shape(n, coarsest_n)
    except ValueError:
        return False
    return True


def validate_multigrid_size(n: int, coarsest_n: int = 3) -> None:
    try:
        validate_root_shape(n, coarsest_n)
    except ValueError as exc:
        raise ValueError(
            "nested structured MG requires n=2**L-1 and an exactly reachable "
            f"coarsest_n=2**l-1; got n={n}, coarsest_n={coarsest_n}"
        ) from exc


def coarse_size(n_fine: int) -> int:
    if n_fine < 3 or n_fine % 2 == 0:
        raise ValueError("nested geometric MG requires odd n_fine >= 3")
    return (n_fine - 1) // 2


@lru_cache(maxsize=TRANSFER_PATTERN_CACHE_MAXSIZE)
def _build_transfer_pattern_cached(
    fine_shape: GridShape,
    coarse_shape: GridShape,
) -> TransferPattern:
    nx, ny = fine_shape
    cx, cy = coarse_shape
    rx, ry = coarsening_ratios(fine_shape, coarse_shape)
    offsets = np.asarray(TRANSFER_AXIS_OFFSETS, dtype=np.int64)

    # One-dimensional candidate coordinates have shape ``(n_fine, 4)``.  The
    # Cartesian product below is entirely vectorized and retains all sixteen
    # slots even when a candidate falls outside the coarse unknown grid.
    fine_i = np.arange(1, nx + 1, dtype=np.int64)
    fine_j = np.arange(1, ny + 1, dtype=np.int64)
    coarse_i = fine_i[:, None] // rx + offsets[None, :]
    coarse_j = fine_j[:, None] // ry + offsets[None, :]
    valid_i = (coarse_i >= 1) & (coarse_i <= cx)
    valid_j = (coarse_j >= 1) & (coarse_j <= cy)

    candidate_i = coarse_i[:, None, :, None]
    candidate_j = coarse_j[None, :, None, :]
    valid = valid_i[:, None, :, None] & valid_j[None, :, None, :]
    columns_4d = (candidate_i - 1) * cy + (candidate_j - 1)
    columns = np.where(valid, columns_4d, -1).reshape(
        nx * ny, MAX_TRANSFER_CANDIDATES
    ).astype(_transfer_column_dtype(coarse_shape), copy=False)

    # Hat-function weights reproduce injection for ratio one, injection at
    # even nodes for ratio two, and half weights at odd nodes.  Masking invalid
    # candidates preserves the homogeneous Dirichlet boundary contribution.
    center_i = fine_i.astype(np.float64) / float(rx)
    center_j = fine_j.astype(np.float64) / float(ry)
    weight_i = np.maximum(
        1.0 - np.abs(coarse_i.astype(np.float64) - center_i[:, None]), 0.0
    )
    weight_j = np.maximum(
        1.0 - np.abs(coarse_j.astype(np.float64) - center_j[:, None]), 0.0
    )
    weight_i = np.where(valid_i, weight_i, 0.0)
    weight_j = np.where(valid_j, weight_j, 0.0)
    base = (
        weight_i[:, None, :, None] * weight_j[None, :, None, :]
    ).reshape(nx * ny, MAX_TRANSFER_CANDIDATES)

    columns = _immutable_array(columns)
    base = _immutable_array(base)
    return TransferPattern(fine_shape, coarse_shape, columns, base)


def build_transfer_pattern(
    fine: int | GridShape,
    coarse: GridShape | None = None,
    *,
    coarsening: CoarseningKind = "full",
    coarsest_n: int = 3,
    level_index: int = 0,
) -> TransferPattern:
    fine_shape = as_shape(fine)
    coarse_shape = (
        next_shape(fine_shape, coarsening, coarsest_n, level_index=level_index)
        if coarse is None
        else as_shape(coarse)
    )
    return _build_transfer_pattern_cached(fine_shape, coarse_shape)


def transfer_pattern_cache_info() -> object:
    """Return the standard LRU cache statistics for transfer patterns."""

    return _build_transfer_pattern_cached.cache_info()


def clear_transfer_pattern_cache() -> None:
    """Release references owned by the process-wide transfer-pattern cache.

    Patterns already installed in live hierarchies remain valid and are freed
    normally with those hierarchies.
    """

    _build_transfer_pattern_cached.cache_clear()


def scipy_prolongation_from_weights(pattern: TransferPattern, weights: np.ndarray) -> sp.csr_matrix:
    if weights.shape != pattern.bilinear_weights.shape:
        raise ValueError("weights shape mismatch")
    k = pattern.n_candidates
    rows = np.repeat(np.arange(pattern.n_fine_unknowns, dtype=np.int64), k)
    cols = pattern.columns.reshape(-1)
    vals = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = (cols >= 0) & (vals != 0.0) & np.isfinite(vals)
    p = sp.coo_matrix(
        (vals[valid], (rows[valid], cols[valid])),
        shape=(pattern.n_fine_unknowns, pattern.n_coarse_unknowns),
        dtype=np.float64,
    ).tocsr()
    p.sum_duplicates()
    p.eliminate_zeros()
    return p


def weights_from_sparse_matrix(pattern, p):
    """Read only bounded candidate slots from a sparse transfer operator."""
    columns=pattern.columns
    rows=np.broadcast_to(np.arange(columns.shape[0])[:,None],columns.shape)
    valid=columns>=0
    result=np.zeros(columns.shape,dtype=np.float64)
    result[valid]=np.asarray(p.tocsr()[rows[valid],columns[valid]]).ravel()
    return result


def classical_prolongation(
    fine: int | GridShape,
    coarse: GridShape | None = None,
    *,
    coarsening: CoarseningKind = "full",
    coarsest_n: int = 3,
    level_index: int = 0,
) -> sp.csr_matrix:
    pattern = build_transfer_pattern(
        fine, coarse, coarsening=coarsening, coarsest_n=coarsest_n, level_index=level_index
    )
    return scipy_prolongation_from_weights(pattern, pattern.bilinear_weights)


def _drop_sparse_relative(matrix: sp.csr_matrix, tolerance: float) -> sp.csr_matrix:
    if tolerance <= 0.0 or matrix.nnz == 0:
        return matrix
    matrix = matrix.tocsr(copy=True)
    counts = np.diff(matrix.indptr)
    row_ids = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), counts)
    row_max = np.zeros(matrix.shape[0], dtype=np.float64)
    np.maximum.at(row_max, row_ids, np.abs(matrix.data))
    thresholds = tolerance * np.maximum(row_max[row_ids], 1.0e-30)
    matrix.data[np.abs(matrix.data) < thresholds] = 0.0
    matrix.eliminate_zeros()
    return matrix


def _coarse_fine_indices(pattern: TransferPattern) -> np.ndarray:
    nx, ny = pattern.fine_shape
    cx, cy = pattern.coarse_shape
    rx, ry = coarsening_ratios(pattern.fine_shape, pattern.coarse_shape)
    ii = np.arange(1, cx + 1, dtype=np.int64) * rx
    jj = np.arange(1, cy + 1, dtype=np.int64) * ry
    gi, gj = np.meshgrid(ii, jj, indexing="ij")
    return ((gi - 1) * ny + (gj - 1)).reshape(-1)


def coarse_fine_indices(pattern: TransferPattern) -> np.ndarray:
    """Fine-grid vector indices corresponding to geometric coarse nodes."""

    return _coarse_fine_indices(pattern).copy()


def _gather_sparse_candidates(
    matrix: sp.csr_matrix,
    row_positions: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Vectorized sparse lookup for pairs ``(row_positions, columns)``."""

    result = np.zeros(columns.shape, dtype=np.float64)
    valid = (row_positions >= 0) & (columns >= 0)
    if not np.any(valid) or matrix.nnz == 0:
        return result
    coo = matrix.tocoo(copy=False)
    key_scale = matrix.shape[1]
    sparse_keys = coo.row.astype(np.int64) * key_scale + coo.col.astype(np.int64)
    order = np.argsort(sparse_keys)
    sparse_keys = sparse_keys[order]
    sparse_values = coo.data[order]
    query = row_positions[valid].astype(np.int64) * key_scale + columns[valid].astype(np.int64)
    pos = np.searchsorted(sparse_keys, query)
    hit = pos < sparse_keys.size
    hit_indices = np.flatnonzero(hit)
    hit[hit_indices] &= sparse_keys[pos[hit_indices]] == query[hit_indices]
    values = np.zeros(query.size, dtype=np.float64)
    values[hit] = sparse_values[pos[hit]]
    result[valid] = values
    return result


def _topk_rows(values: np.ndarray, valid: np.ndarray, topk: int | None) -> np.ndarray:
    if topk is None or topk <= 0 or topk >= values.shape[1]:
        return values
    masked = np.where(valid, np.abs(values), -np.inf)
    keep = np.zeros_like(valid)
    indices = np.argpartition(masked, -topk, axis=1)[:, -topk:]
    np.put_along_axis(keep, indices, True, axis=1)
    return np.where(keep & valid, values, 0.0)


def operator_dependent_weights(
    a: sp.csr_matrix,
    fine: int | GridShape,
    *,
    coarse: GridShape | None = None,
    coarsening: CoarseningKind = "full",
    coarsest_n: int = 3,
    level_index: int = 0,
    sweeps: int = 3,
    omega: float = 0.70,
    blend: float = 0.75,
    drop_tolerance: float = 1.0e-4,
    max_weight: float = 4.0,
    candidate_topk: int | None = 8,
) -> np.ndarray:
    """Vectorized approximate-ideal interpolation on the expanded support."""

    fine_shape = as_shape(fine)
    if a.shape != (unknowns(fine_shape), unknowns(fine_shape)):
        raise ValueError("matrix shape does not match fine grid")
    if sweeps < 0:
        raise ValueError("sweeps must be nonnegative")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must lie in [0,1]")
    pattern = build_transfer_pattern(
        fine_shape,
        coarse,
        coarsening=coarsening,
        coarsest_n=coarsest_n,
        level_index=level_index,
    )
    if sweeps == 0 or blend == 0.0:
        return pattern.bilinear_weights.copy()

    coarse_fine = _coarse_fine_indices(pattern)
    all_indices = np.arange(pattern.n_fine_unknowns, dtype=np.int64)
    is_coarse = np.zeros(pattern.n_fine_unknowns, dtype=bool)
    is_coarse[coarse_fine] = True
    fine_indices = all_indices[~is_coarse]

    a = a.tocsr()
    aff = a[fine_indices][:, fine_indices].tocsr()
    afc = a[fine_indices][:, coarse_fine].tocsr()
    diagonal = np.maximum(np.abs(aff.diagonal()), 1.0e-14)
    dinv = sp.diags(omega / diagonal, format="csr")
    x = sp.csr_matrix(afc.shape, dtype=np.float64)
    for _ in range(sweeps):
        x = (x + dinv @ (-afc - aff @ x)).tocsr()
        x = _drop_sparse_relative(x, drop_tolerance)

    f_position = np.full(pattern.n_fine_unknowns, -1, dtype=np.int64)
    f_position[fine_indices] = np.arange(fine_indices.size, dtype=np.int64)
    row_positions = np.repeat(f_position[:, None], pattern.n_candidates, axis=1)
    raw = _gather_sparse_candidates(x, row_positions, pattern.columns)
    valid = pattern.columns >= 0
    raw = _topk_rows(raw, valid, candidate_topk)

    target_sum = pattern.bilinear_weights.sum(axis=1, keepdims=True)
    raw_sum = raw.sum(axis=1, keepdims=True)
    usable = (
        (~is_coarse)[:, None]
        & np.isfinite(raw_sum)
        & (np.abs(raw_sum) > 1.0e-12)
    )
    normalized = np.zeros_like(raw)
    normalized[usable[:, 0]] = (
        raw[usable[:, 0]] * target_sum[usable[:, 0]] / raw_sum[usable[:, 0]]
    )
    row_bound = max_weight * np.maximum(np.abs(target_sum[:, 0]), 1.0e-12)
    usable[:, 0] &= np.max(np.abs(normalized), axis=1) <= row_bound

    weights = pattern.bilinear_weights.copy()
    rows = usable[:, 0]
    weights[rows] = (1.0 - blend) * weights[rows] + blend * normalized[rows]
    weights[~valid] = 0.0
    # Preserve each classical row sum exactly, including boundary rows.
    drift = target_sum - weights.sum(axis=1, keepdims=True)
    counts = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    weights += np.where(valid, drift / counts, 0.0)
    return weights


def baseline_weights(
    a: sp.csr_matrix,
    fine: int | GridShape,
    baseline: TransferBaseline,
    *,
    coarse: GridShape | None = None,
    coarsening: CoarseningKind = "full",
    coarsest_n: int = 3,
    level_index: int = 0,
    operator_sweeps: int = 3,
    operator_omega: float = 0.70,
    operator_blend: float = 0.75,
    operator_drop_tolerance: float = 1.0e-4,
    operator_candidate_topk: int | None = 8,
) -> np.ndarray:
    pattern = build_transfer_pattern(
        fine,
        coarse,
        coarsening=coarsening,
        coarsest_n=coarsest_n,
        level_index=level_index,
    )
    if baseline == "bilinear":
        return pattern.bilinear_weights.copy()
    if baseline == "operator":
        return operator_dependent_weights(
            a,
            fine,
            coarse=pattern.coarse_shape,
            sweeps=operator_sweeps,
            omega=operator_omega,
            blend=operator_blend,
            drop_tolerance=operator_drop_tolerance,
            candidate_topk=operator_candidate_topk,
        )
    raise ValueError(f"unknown transfer baseline: {baseline}")


def baseline_prolongation(
    a: sp.csr_matrix,
    fine: int | GridShape,
    baseline: TransferBaseline,
    **kwargs: object,
) -> sp.csr_matrix:
    coarse = kwargs.get("coarse")
    coarsening = kwargs.get("coarsening", "full")
    coarsest_n = int(kwargs.get("coarsest_n", 3))
    level_index = int(kwargs.get("level_index", 0))
    pattern = build_transfer_pattern(
        fine,
        coarse if isinstance(coarse, tuple) else None,
        coarsening=coarsening,  # type: ignore[arg-type]
        coarsest_n=coarsest_n,
        level_index=level_index,
    )
    return scipy_prolongation_from_weights(
        pattern,
        baseline_weights(a, fine, baseline, **kwargs),
    )


def matrix_feature_array(a: sp.csr_matrix, shape: int | GridShape) -> np.ndarray:
    """Log diagonal and normalized local 9-point stencil channels."""

    nx, ny = as_shape(shape)
    if a.shape != (nx * ny, nx * ny):
        raise ValueError("matrix shape does not match grid shape")
    a = a.tocsr(copy=True)
    a.sum_duplicates()
    diagonal = np.maximum(np.abs(a.diagonal()), 1.0e-14).reshape(nx, ny)
    median = max(float(np.median(diagonal)), 1.0e-14)
    features = np.zeros((N_MATRIX_FEATURES, nx, ny), dtype=np.float32)
    features[0] = np.log(diagonal / median + 1.0e-12).astype(np.float32)
    coo = a.tocoo(copy=False)
    ri, rj = np.divmod(coo.row, ny)
    ci, cj = np.divmod(coo.col, ny)
    di, dj = ci - ri, cj - rj
    norm = diagonal[ri, rj]
    for channel, (oi, oj) in enumerate(OFFSETS_9, start=1):
        mask = (di == oi) & (dj == oj)
        if np.any(mask):
            features[channel, ri[mask], rj[mask]] = (
                coo.data[mask] / norm[mask]
            ).astype(np.float32)
    return features


def matrix_features_torch(
    a: sp.csr_matrix,
    shape: int | GridShape,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.from_numpy(matrix_feature_array(a, shape)).unsqueeze(0).to(device=device)


def weights_from_deltas_torch(
    pattern: TransferPattern,
    deltas: torch.Tensor,
    baseline: torch.Tensor | np.ndarray | None = None,
    gate: torch.Tensor | None = None,
    *, support: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    nx, ny = pattern.fine_shape
    k = pattern.n_candidates
    if tuple(deltas.shape) != (1, k, nx, ny):
        raise ValueError(f"deltas must have shape (1,{k},{nx},{ny})")
    device, dtype = deltas.device, deltas.dtype
    if baseline is None:
        base = torch.as_tensor(
            np.array(pattern.bilinear_weights, copy=True), dtype=dtype, device=device
        )
    else:
        base = torch.as_tensor(baseline, dtype=dtype, device=device)
    if tuple(base.shape) != tuple(pattern.bilinear_weights.shape):
        raise ValueError("baseline weight shape mismatch")
    valid = torch.as_tensor(
        np.array(pattern.columns >= 0, copy=True), dtype=torch.bool, device=device
    )
    if support is not None:
        allowed=torch.as_tensor(support,dtype=torch.bool,device=device)
        if allowed.shape!=valid.shape:raise ValueError('support shape mismatch')
        valid=valid & allowed
    logits = deltas.permute(0, 2, 3, 1).reshape(-1, k)
    target_sum = base.sum(dim=1, keepdim=True)
    count = valid.sum(dim=1, keepdim=True).clamp(min=1)
    # A small uniform prior lets the network activate expanded-support entries
    # while the reference subtraction guarantees exact zero-init equivalence.
    prior_floor = 0.05 * torch.abs(target_sum) / count
    prior = torch.where(valid, torch.abs(base) + prior_floor, torch.zeros_like(base))
    log_prior = torch.log(torch.clamp(prior, min=1.0e-12))
    masked_logits = torch.where(valid, logits + log_prior, torch.full_like(logits, -1.0e9))
    masked_reference = torch.where(valid, log_prior, torch.full_like(logits, -1.0e9))
    target = torch.softmax(masked_logits, dim=1)
    reference = torch.softmax(masked_reference, dim=1)
    signed_sum = torch.sign(torch.where(target_sum == 0.0, torch.ones_like(target_sum), target_sum))
    learned = base + signed_sum * torch.abs(target_sum) * (target - reference)
    learned = torch.where(valid, learned, torch.zeros_like(learned))
    drift = target_sum - learned.sum(dim=1, keepdim=True)
    learned = learned + torch.where(valid, drift / count, torch.zeros_like(learned))
    # Preserve exact injection at geometric coarse nodes.  This guarantees
    # column independence/full rank for every neural output and prevents an
    # OOD decoder from corrupting the coarse degrees of freedom themselves.
    coarse_rows_np = np.zeros(pattern.n_fine_unknowns, dtype=bool)
    coarse_rows_np[_coarse_fine_indices(pattern)] = True
    coarse_rows = torch.as_tensor(coarse_rows_np, device=device, dtype=torch.bool).unsqueeze(1)
    learned = torch.where(coarse_rows, base, learned)
    if gate is None:
        return learned
    if tuple(gate.shape) != (1, 1, nx, ny):
        raise ValueError(f"gate must have shape (1,1,{nx},{ny})")
    g = gate.reshape(-1, 1).clamp(0.0, 1.0)
    return base + g * (learned - base)


def sparse_prolongation_torch(pattern: TransferPattern, weights: torch.Tensor) -> torch.Tensor:
    if tuple(weights.shape) != tuple(pattern.bilinear_weights.shape):
        raise ValueError("weights shape mismatch")
    k = pattern.n_candidates
    rows = torch.arange(
        pattern.n_fine_unknowns, device=weights.device, dtype=torch.long
    ).repeat_interleave(k)
    cols = torch.as_tensor(
        np.array(pattern.columns.reshape(-1), copy=True),
        device=weights.device,
        dtype=torch.long,
    )
    vals = weights.reshape(-1)
    valid = cols >= 0
    indices = torch.stack((rows[valid], cols[valid]), dim=0)
    return torch.sparse_coo_tensor(
        indices,
        vals[valid],
        size=(pattern.n_fine_unknowns, pattern.n_coarse_unknowns),
        device=weights.device,
        dtype=weights.dtype,
    ).coalesce()


def dense_prolongation_torch(pattern: TransferPattern, weights: torch.Tensor) -> torch.Tensor:
    """Differentiable dense assembly without sparse-to-dense warning paths."""

    if tuple(weights.shape) != tuple(pattern.bilinear_weights.shape):
        raise ValueError("weights shape mismatch")
    k = pattern.n_candidates
    rows = torch.arange(
        pattern.n_fine_unknowns, device=weights.device, dtype=torch.long
    ).repeat_interleave(k)
    cols = torch.as_tensor(
        np.array(pattern.columns.reshape(-1), copy=True),
        device=weights.device,
        dtype=torch.long,
    )
    vals = weights.reshape(-1)
    valid = cols >= 0
    dense = torch.zeros(
        (pattern.n_fine_unknowns, pattern.n_coarse_unknowns),
        device=weights.device,
        dtype=weights.dtype,
    )
    return dense.index_put((rows[valid], cols[valid]), vals[valid], accumulate=True)


def learned_prolongation(
    a: sp.csr_matrix,
    shape: int | GridShape,
    transfer_net: torch.nn.Module,
    *,
    baseline: TransferBaseline = "bilinear",
    coarse: GridShape | None = None,
    coarsening: CoarseningKind = "full",
    coarsest_n: int = 3,
    level_index: int = 0,
    gate: np.ndarray | torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    **baseline_kwargs: object,
) -> sp.csr_matrix:
    pattern = build_transfer_pattern(
        shape,
        coarse,
        coarsening=coarsening,
        coarsest_n=coarsest_n,
        level_index=level_index,
    )
    base = baseline_weights(
        a,
        shape,
        baseline,
        coarse=pattern.coarse_shape,
        **baseline_kwargs,
    )
    transfer_net = transfer_net.to(device).eval()
    with torch.no_grad():
        deltas = transfer_net(matrix_features_torch(a, shape, device=device))
        if gate is None:
            gate_t = None
        elif isinstance(gate, torch.Tensor):
            gate_t = gate.to(device=device, dtype=deltas.dtype)
        else:
            gate_t = torch.as_tensor(gate, device=device, dtype=deltas.dtype)
        weights = weights_from_deltas_torch(pattern, deltas, base, gate_t)
    return scipy_prolongation_from_weights(pattern, weights.cpu().numpy())


def galerkin_coarse_operator(a: sp.csr_matrix, p: sp.csr_matrix) -> sp.csr_matrix:
    ac = (p.T @ (a @ p)).tocsr()
    ac = (0.5 * (ac + ac.T)).tocsr()
    ac.eliminate_zeros()
    return ac
