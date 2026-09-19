"""Sparse differentiable full-V-cycle operations for neural-MG training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .detector_features import detector_features_torch
from .grid import GridShape, as_shape, next_shape, terminal, unknowns
from .models import NeuralComponents
from .smoothers import SmootherKind
from .strategy import MGStrategy
from .transfer import (
    TransferBaseline,
    TransferPattern,
    baseline_weights,
    build_transfer_pattern,
    coarse_fine_indices,
    matrix_feature_array,
    sparse_prolongation_torch,
    weights_from_deltas_torch,
)

Mode = Literal["classical", "nn_smoother", "learned_transfer", "combined", "adaptive"]


@dataclass
class TorchTransferData:
    pattern: TransferPattern
    baseline_weights: torch.Tensor
    p_baseline: torch.Tensor


@dataclass
class TorchProblem:
    a: torch.Tensor
    features: torch.Tensor
    diag: torch.Tensor
    shape: GridShape
    scipy_a: sp.csr_matrix | None = None
    transfer_cache: dict[tuple[object, ...], TorchTransferData] = field(default_factory=dict)
    line_block_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    @property
    def n(self) -> int:
        if self.shape[0] != self.shape[1]:
            raise ValueError("n is only defined for square training grids")
        return self.shape[0]

    @property
    def device(self) -> torch.device:
        return self.a.device

    @property
    def dtype(self) -> torch.dtype:
        return self.a.dtype


def scipy_to_torch_sparse(
    matrix: sp.spmatrix,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    coo = matrix.tocoo(copy=False)
    indices = torch.as_tensor(
        np.vstack((coo.row, coo.col)), device=device, dtype=torch.long
    )
    values = torch.as_tensor(coo.data, device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device, dtype=dtype).coalesce()


def torch_sparse_to_scipy(matrix: torch.Tensor) -> sp.csr_matrix:
    matrix = matrix.coalesce()
    indices = matrix.indices().detach().cpu().numpy()
    values = matrix.values().detach().cpu().numpy().astype(np.float64)
    return sp.coo_matrix((values, (indices[0], indices[1])), shape=matrix.shape).tocsr()


def _sparse_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.coalesce()
    indices, values = matrix.indices(), matrix.values()
    mask = indices[0] == indices[1]
    diagonal = torch.zeros(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    diagonal = diagonal.index_put((indices[0, mask],), values[mask], accumulate=True)
    return diagonal


def _matrix_features_from_torch_sparse(matrix: torch.Tensor, shape: GridShape) -> torch.Tensor:
    """Detached local stencil features for a dynamic Galerkin operator."""

    nx, ny = shape
    matrix = matrix.coalesce()
    with torch.no_grad():
        indices = matrix.indices()
        values = matrix.values().detach()
        diagonal = torch.clamp(torch.abs(_sparse_diagonal(matrix).detach()), min=1.0e-14)
        diagonal_grid = diagonal.reshape(nx, ny)
        median = torch.clamp(torch.median(diagonal_grid), min=1.0e-14)
        features = torch.zeros(
            (1, 10, nx, ny), device=matrix.device, dtype=matrix.dtype
        )
        features[0, 0] = torch.log(diagonal_grid / median + 1.0e-12)
        ri = torch.div(indices[0], ny, rounding_mode="floor")
        rj = indices[0] % ny
        ci = torch.div(indices[1], ny, rounding_mode="floor")
        cj = indices[1] % ny
        di, dj = ci - ri, cj - rj
        offsets = (
            (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        )
        norm = diagonal[indices[0]]
        for channel, (oi, oj) in enumerate(offsets, start=1):
            mask = (di == oi) & (dj == oj)
            if bool(mask.any()):
                features[0, channel] = features[0, channel].index_put(
                    (ri[mask], rj[mask]), values[mask] / norm[mask], accumulate=True
                )
    return features


def make_torch_problem(
    a: sp.csr_matrix,
    shape: int | GridShape,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    **_: object,
) -> TorchProblem:
    grid_shape = as_shape(shape)
    if a.shape != (unknowns(grid_shape), unknowns(grid_shape)):
        raise ValueError("matrix shape does not match grid")
    sparse = scipy_to_torch_sparse(a, device=device, dtype=dtype)
    features = torch.as_tensor(
        matrix_feature_array(a, grid_shape), device=device, dtype=dtype
    ).unsqueeze(0)
    diag = torch.as_tensor(a.diagonal(), device=device, dtype=dtype)
    return TorchProblem(sparse, features, diag, grid_shape, a.tocsr())


def make_child_problem(a: torch.Tensor, shape: GridShape) -> TorchProblem:
    a = a.coalesce()
    return TorchProblem(
        a=a,
        features=_matrix_features_from_torch_sparse(a, shape),
        diag=_sparse_diagonal(a),
        shape=shape,
        scipy_a=None,
    )


def vector_to_grid(x: torch.Tensor, shape: int | GridShape) -> torch.Tensor:
    nx, ny = as_shape(shape)
    return x.reshape(1, 1, nx, ny)


def grid_to_vector(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def apply_a(problem: TorchProblem, x: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(problem.a, x.reshape(-1, 1)).reshape(-1)


def apply_a_matrix(problem: TorchProblem, x: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(problem.a, x)


def _zebra_order(number_of_lines: int, reverse: bool) -> list[int]:
    order = list(range(0, number_of_lines, 2)) + list(range(1, number_of_lines, 2))
    return list(reversed(order)) if reverse else order


def _line_indices(shape: GridShape, direction: Literal["x", "y"], line: int) -> np.ndarray:
    nx, ny = shape
    if direction == "x":
        return np.arange(nx, dtype=np.int64) * ny + line
    return line * ny + np.arange(ny, dtype=np.int64)


def _line_count(shape: GridShape, direction: Literal["x", "y"]) -> int:
    return shape[1] if direction == "x" else shape[0]


def _line_block(
    problem: TorchProblem,
    direction: Literal["x", "y"],
    line: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (direction, line)
    if key in problem.line_block_cache:
        return problem.line_block_cache[key]
    indices_np = _line_indices(problem.shape, direction, line)
    indices = torch.as_tensor(indices_np, device=problem.device, dtype=torch.long)
    if problem.scipy_a is not None:
        # Root operators come from SciPy.  Only the one-dimensional line block
        # is materialized; the full PDE matrix remains sparse.
        block_np = problem.scipy_a[indices_np][:, indices_np].toarray()
        block = torch.as_tensor(block_np, device=problem.device, dtype=problem.dtype)
        problem.line_block_cache[key] = (indices, block)
        return indices, block

    # Dynamic Galerkin operators depend on the learned transfer.  Extract the
    # requested line directly from sparse COO values so gradients can flow
    # through A_H without ever densifying the full coarse matrix.
    sparse = problem.a.coalesce()
    sparse_indices = sparse.indices()
    sparse_values = sparse.values()
    nx, ny = problem.shape
    if direction == "x":
        row_on_line = sparse_indices[0] % ny == line
        col_on_line = sparse_indices[1] % ny == line
        mask = row_on_line & col_on_line
        local_rows = torch.div(sparse_indices[0, mask], ny, rounding_mode="floor")
        local_cols = torch.div(sparse_indices[1, mask], ny, rounding_mode="floor")
        line_size = nx
    else:
        row_on_line = torch.div(sparse_indices[0], ny, rounding_mode="floor") == line
        col_on_line = torch.div(sparse_indices[1], ny, rounding_mode="floor") == line
        mask = row_on_line & col_on_line
        local_rows = sparse_indices[0, mask] % ny
        local_cols = sparse_indices[1, mask] % ny
        line_size = ny
    block = torch.zeros(
        (line_size, line_size), device=problem.device, dtype=problem.dtype
    ).index_put(
        (local_rows, local_cols), sparse_values[mask], accumulate=True
    )
    problem.line_block_cache[key] = (indices, block)
    return indices, block


def _line_direction_correction(
    problem: TorchProblem,
    correction: torch.Tensor,
    current: torch.Tensor,
    direction: Literal["x", "y"],
    *,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    for line in _zebra_order(_line_count(problem.shape, direction), reverse):
        indices, block = _line_block(problem, direction, line)
        increment = torch.linalg.solve(block, current.index_select(0, indices))
        update = torch.zeros_like(correction).index_copy(0, indices, increment)
        correction = correction + update
        current = current - apply_a(problem, update)
    return correction, current


def classical_correction(
    problem: TorchProblem,
    residual: torch.Tensor,
    *,
    smoother: SmootherKind = "line_alt",
    omega: float = 0.72,
    reverse: bool = False,
) -> torch.Tensor:
    if smoother == "jacobi":
        return omega * residual / torch.clamp(torch.abs(problem.diag), min=1.0e-12)
    correction = torch.zeros_like(residual)
    current = residual
    if smoother == "line_x":
        return _line_direction_correction(
            problem, correction, current, "x", reverse=reverse
        )[0]
    if smoother == "line_y":
        return _line_direction_correction(
            problem, correction, current, "y", reverse=reverse
        )[0]
    if smoother == "line_alt":
        directions = ("y", "x") if reverse else ("x", "y")
        for direction in directions:
            correction, current = _line_direction_correction(
                problem, correction, current, direction, reverse=reverse
            )
        return correction
    raise ValueError(f"unknown smoother: {smoother}")


def neural_delta(
    problem: TorchProblem,
    residual: torch.Tensor,
    components: NeuralComponents,
) -> torch.Tensor:
    normalized = vector_to_grid(residual, problem.shape) / torch.clamp(
        torch.abs(problem.diag.reshape(1, 1, *problem.shape)), min=1.0e-12
    )
    return grid_to_vector(components.smoother(problem.features, normalized))


def local_energy(x_grid: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    return F.avg_pool2d(
        x_grid.square(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2
    )


def detector_feature_tensor(
    problem: TorchProblem,
    residual: torch.Tensor,
    *,
    smoother: SmootherKind = "line_alt",
    omega: float = 0.72,
    reverse: bool = False,
    classical: torch.Tensor | None = None,
) -> torch.Tensor:
    base = (
        classical
        if classical is not None
        else classical_correction(
            problem, residual, smoother=smoother, omega=omega, reverse=reverse
        )
    )
    post = residual - apply_a(problem, base)
    return detector_features_torch(
        problem.features, problem.diag, residual, post, problem.shape
    )


def detector_gate(
    problem: TorchProblem,
    residual: torch.Tensor,
    components: NeuralComponents,
    *,
    smoother: SmootherKind = "line_alt",
    omega: float = 0.72,
    reverse: bool = False,
    classical: torch.Tensor | None = None,
    hard: bool = False,
    threshold: float = 0.5,
) -> torch.Tensor:
    gate = components.detector(
        detector_feature_tensor(
            problem,
            residual,
            smoother=smoother,
            omega=omega,
            reverse=reverse,
            classical=classical,
        )
    )
    return (gate >= threshold).to(gate.dtype) if hard else gate


def _scipy_matrix(problem: TorchProblem) -> sp.csr_matrix:
    return problem.scipy_a if problem.scipy_a is not None else torch_sparse_to_scipy(problem.a)



def _drop_sparse_relative_torch(matrix: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Row-relative sparse drop without CPU transfer or Python row loops."""

    matrix = matrix.coalesce()
    if tolerance <= 0.0 or matrix._nnz() == 0:
        return matrix
    indices, values = matrix.indices(), matrix.values()
    row_max = torch.zeros(
        matrix.shape[0], device=values.device, dtype=values.dtype
    )
    row_max.scatter_reduce_(
        0, indices[0], torch.abs(values), reduce="amax", include_self=True
    )
    keep = torch.abs(values) >= tolerance * torch.clamp(
        row_max.index_select(0, indices[0]), min=1.0e-30
    )
    return torch.sparse_coo_tensor(
        indices[:, keep], values[keep], matrix.shape,
        device=values.device, dtype=values.dtype,
    ).coalesce()


def _sparse_submatrix_maps(
    matrix: torch.Tensor,
    pattern: TransferPattern,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``A_FF``, ``A_FC``, fine globals, and coarse-row mask."""

    matrix = matrix.coalesce()
    total = pattern.n_fine_unknowns
    coarse_np = coarse_fine_indices(pattern)
    coarse_global = torch.as_tensor(
        coarse_np, device=matrix.device, dtype=torch.long
    )
    is_coarse = torch.zeros(total, device=matrix.device, dtype=torch.bool)
    is_coarse[coarse_global] = True
    fine_global = torch.nonzero(~is_coarse, as_tuple=False).reshape(-1)
    fpos = torch.full((total,), -1, device=matrix.device, dtype=torch.long)
    cpos = torch.full((total,), -1, device=matrix.device, dtype=torch.long)
    fpos[fine_global] = torch.arange(
        fine_global.numel(), device=matrix.device, dtype=torch.long
    )
    cpos[coarse_global] = torch.arange(
        coarse_global.numel(), device=matrix.device, dtype=torch.long
    )
    rows, cols = matrix.indices()
    values = matrix.values().detach()
    rf, cf = fpos[rows], fpos[cols]
    rc = cpos[cols]
    aff_mask = (rf >= 0) & (cf >= 0)
    afc_mask = (rf >= 0) & (rc >= 0)
    aff = torch.sparse_coo_tensor(
        torch.stack((rf[aff_mask], cf[aff_mask])),
        values[aff_mask],
        (fine_global.numel(), fine_global.numel()),
        device=matrix.device,
        dtype=matrix.dtype,
    ).coalesce()
    afc = torch.sparse_coo_tensor(
        torch.stack((rf[afc_mask], rc[afc_mask])),
        values[afc_mask],
        (fine_global.numel(), coarse_global.numel()),
        device=matrix.device,
        dtype=matrix.dtype,
    ).coalesce()
    return aff, afc, fine_global, is_coarse


def _gather_sparse_candidates_torch(
    matrix: torch.Tensor,
    fine_position: torch.Tensor,
    pattern: TransferPattern,
) -> torch.Tensor:
    """Vectorized sparse lookup on the candidate support, entirely on device."""

    matrix = matrix.coalesce()
    columns = torch.as_tensor(
        np.array(pattern.columns, copy=True), device=matrix.device, dtype=torch.long
    )
    row_positions = fine_position.reshape(-1, 1).expand_as(columns)
    valid = (row_positions >= 0) & (columns >= 0)
    output = torch.zeros(columns.shape, device=matrix.device, dtype=matrix.dtype)
    if matrix._nnz() == 0 or not bool(valid.any()):
        return output
    key_scale = matrix.shape[1]
    keys = matrix.indices()[0] * key_scale + matrix.indices()[1]
    order = torch.argsort(keys)
    keys = keys.index_select(0, order)
    values = matrix.values().index_select(0, order)
    queries = row_positions[valid] * key_scale + columns[valid]
    positions = torch.searchsorted(keys, queries)
    hit = positions < keys.numel()
    safe_positions = torch.clamp(positions, max=max(keys.numel() - 1, 0))
    hit = hit & (keys.index_select(0, safe_positions) == queries)
    gathered = torch.zeros(queries.numel(), device=matrix.device, dtype=matrix.dtype)
    if bool(hit.any()):
        gathered[hit] = values.index_select(0, positions[hit])
    output[valid] = gathered
    return output


def operator_dependent_weights_torch(
    problem: TorchProblem,
    pattern: TransferPattern,
    *,
    sweeps: int = 3,
    omega: float = 0.70,
    blend: float = 0.75,
    drop_tolerance: float = 1.0e-4,
    max_weight: float = 4.0,
    candidate_topk: int | None = 8,
) -> torch.Tensor:
    """GPU-native sparse approximate-ideal interpolation baseline.

    This mirrors :func:`adaptive_mg.transfer.operator_dependent_weights` but
    keeps dynamic Galerkin operators on the current Torch device.  The
    baseline is deliberately detached: it is a classical reference, while the
    trainable gradient flows through the learned correction and Galerkin cycle.
    """

    base = torch.as_tensor(
        np.array(pattern.bilinear_weights, copy=True),
        device=problem.device,
        dtype=problem.dtype,
    )
    if sweeps <= 0 or blend <= 0.0:
        return base
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        aff, afc, fine_global, is_coarse = _sparse_submatrix_maps(
            problem.a.detach(), pattern
        )
        x = torch.sparse_coo_tensor(
            torch.empty((2, 0), device=problem.device, dtype=torch.long),
            torch.empty((0,), device=problem.device, dtype=problem.dtype),
            afc.shape,
            device=problem.device,
            dtype=problem.dtype,
        ).coalesce()
        diagonal = torch.clamp(
            torch.abs(problem.diag.detach().index_select(0, fine_global)), min=1.0e-14
        )
        for _ in range(sweeps):
            product = torch.sparse.mm(aff, x).coalesce()
            residual = ((-afc) + (-product)).coalesce()
            r_indices, r_values = residual.indices(), residual.values()
            update_values = (
                omega
                * r_values
                / diagonal.index_select(0, r_indices[0])
            )
            update = torch.sparse_coo_tensor(
                r_indices, update_values, residual.shape,
                device=problem.device, dtype=problem.dtype,
            ).coalesce()
            x = _drop_sparse_relative_torch((x + update).coalesce(), drop_tolerance)

        fine_position = torch.full(
            (pattern.n_fine_unknowns,), -1, device=problem.device, dtype=torch.long
        )
        fine_position[fine_global] = torch.arange(
            fine_global.numel(), device=problem.device, dtype=torch.long
        )
        raw = _gather_sparse_candidates_torch(x, fine_position, pattern)
        valid = torch.as_tensor(
            np.array(pattern.columns >= 0, copy=True),
            device=problem.device,
            dtype=torch.bool,
        )
        if candidate_topk is not None and 0 < candidate_topk < raw.shape[1]:
            masked = torch.where(valid, torch.abs(raw), torch.full_like(raw, -torch.inf))
            indices = torch.topk(masked, candidate_topk, dim=1).indices
            keep = torch.zeros_like(valid)
            keep.scatter_(1, indices, True)
            raw = torch.where(keep & valid, raw, torch.zeros_like(raw))

        target_sum = base.sum(dim=1, keepdim=True)
        raw_sum = raw.sum(dim=1, keepdim=True)
        usable = (
            (~is_coarse).reshape(-1, 1)
            & torch.isfinite(raw_sum)
            & (torch.abs(raw_sum) > 1.0e-12)
        )
        normalized = torch.where(
            usable,
            raw * target_sum / torch.where(usable, raw_sum, torch.ones_like(raw_sum)),
            torch.zeros_like(raw),
        )
        row_bound = max_weight * torch.clamp(torch.abs(target_sum[:, 0]), min=1.0e-12)
        usable = usable & (
            torch.max(torch.abs(normalized), dim=1, keepdim=True).values
            <= row_bound.reshape(-1, 1)
        )
        weights = torch.where(
            usable,
            (1.0 - blend) * base + blend * normalized,
            base,
        )
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        drift = target_sum - weights.sum(dim=1, keepdim=True)
        counts = valid.sum(dim=1, keepdim=True).clamp(min=1)
        weights = weights + torch.where(
            valid, drift / counts.to(weights.dtype), torch.zeros_like(weights)
        )
        return weights


def transfer_data(
    problem: TorchProblem,
    strategy: MGStrategy,
    *,
    coarsest_n: int,
    level_index: int,
    operator_sweeps: int = 3,
    operator_omega: float = 0.70,
    operator_blend: float = 0.75,
    operator_drop_tolerance: float = 1.0e-4,
    operator_candidate_topk: int | None = 8,
) -> TorchTransferData:
    coarse_shape = next_shape(
        problem.shape,
        strategy.coarsening,
        coarsest_n,
        level_index=level_index,
    )
    key = (
        strategy.transfer,
        strategy.coarsening,
        coarse_shape,
        operator_sweeps,
        operator_omega,
        operator_blend,
        operator_drop_tolerance,
        operator_candidate_topk,
    )
    if key in problem.transfer_cache:
        return problem.transfer_cache[key]
    pattern = build_transfer_pattern(problem.shape, coarse_shape)
    if strategy.transfer == "operator" and (
        problem.scipy_a is None or problem.device.type == "cuda"
    ):
        base = operator_dependent_weights_torch(
            problem,
            pattern,
            sweeps=operator_sweeps,
            omega=operator_omega,
            blend=operator_blend,
            drop_tolerance=operator_drop_tolerance,
            candidate_topk=operator_candidate_topk,
        )
    else:
        base_np = baseline_weights(
            _scipy_matrix(problem),
            problem.shape,
            strategy.transfer,
            coarse=coarse_shape,
            operator_sweeps=operator_sweeps,
            operator_omega=operator_omega,
            operator_blend=operator_blend,
            operator_drop_tolerance=operator_drop_tolerance,
            operator_candidate_topk=operator_candidate_topk,
        )
        base = torch.as_tensor(base_np, device=problem.device, dtype=problem.dtype)
    data = TorchTransferData(pattern, base, sparse_prolongation_torch(pattern, base))
    problem.transfer_cache[key] = data
    return data


def _smooth_once(
    problem: TorchProblem,
    x: torch.Tensor,
    b: torch.Tensor,
    components: NeuralComponents,
    mode: Mode,
    strategy: MGStrategy,
    *,
    omega: float,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    residual = b - apply_a(problem, x)
    base = classical_correction(
        problem,
        residual,
        smoother=strategy.smoother,
        omega=omega,
        reverse=reverse,
    )
    if mode in {"classical", "learned_transfer"}:
        return x + base, None
    if mode == "adaptive":
        gate = detector_gate(
            problem,
            residual,
            components,
            smoother=strategy.smoother,
            omega=omega,
            reverse=reverse,
            classical=base,
        )
        delta = neural_delta(problem, residual, components)
        return x + base + gate.reshape(-1) * delta, gate
    delta = neural_delta(problem, residual, components)
    if mode in {"nn_smoother", "combined"}:
        return x + base + delta, None
    raise ValueError(f"unknown mode: {mode}")


def _prolongation(
    problem: TorchProblem,
    residual: torch.Tensor,
    components: NeuralComponents,
    mode: Mode,
    strategy: MGStrategy,
    *,
    coarsest_n: int,
    level_index: int,
    omega: float,
    reuse_gate: torch.Tensor | None,
    operator_sweeps: int,
    operator_omega: float,
    operator_blend: float,
    operator_drop_tolerance: float,
    operator_candidate_topk: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None, GridShape]:
    data = transfer_data(
        problem,
        strategy,
        coarsest_n=coarsest_n,
        level_index=level_index,
        operator_sweeps=operator_sweeps,
        operator_omega=operator_omega,
        operator_blend=operator_blend,
        operator_drop_tolerance=operator_drop_tolerance,
        operator_candidate_topk=operator_candidate_topk,
    )
    if mode in {"classical", "nn_smoother"}:
        return data.p_baseline, None, data.pattern.coarse_shape
    deltas = components.transfer(problem.features)
    if mode in {"learned_transfer", "combined"}:
        weights = weights_from_deltas_torch(
            data.pattern, deltas, data.baseline_weights
        )
        return sparse_prolongation_torch(data.pattern, weights), None, data.pattern.coarse_shape
    if mode == "adaptive":
        gate = reuse_gate
        if gate is None:
            probe = classical_correction(
                problem,
                residual,
                smoother=strategy.smoother,
                omega=omega,
                reverse=False,
            )
            gate = detector_gate(
                problem,
                residual,
                components,
                smoother=strategy.smoother,
                omega=omega,
                reverse=False,
                classical=probe,
            )
        weights = weights_from_deltas_torch(
            data.pattern, deltas, data.baseline_weights, gate
        )
        return sparse_prolongation_torch(data.pattern, weights), gate, data.pattern.coarse_shape
    raise ValueError(f"unknown mode: {mode}")


def _galerkin_torch(problem: TorchProblem, p: torch.Tensor) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        ap = torch.sparse.mm(problem.a, p)
        ac = torch.sparse.mm(p.transpose(0, 1), ap).coalesce()
    ac_t = ac.transpose(0, 1).coalesce()
    # Match the production Galerkin hierarchy exactly.  Earlier training code
    # added a levelwise diagonal shift, which created a small but systematic
    # train/inference operator mismatch.  Rank loss is now treated as a real
    # invalid learned transfer and handled by validation rollback.
    return (0.5 * (ac + ac_t)).coalesce()


def _restrict(p: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(p.transpose(0, 1), vector.reshape(-1, 1)).reshape(-1)


def _prolong(p: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(p, vector.reshape(-1, 1)).reshape(-1)


def full_v_cycle(
    problem: TorchProblem,
    x: torch.Tensor,
    b: torch.Tensor,
    components: NeuralComponents,
    *,
    mode: Mode,
    strategy: MGStrategy,
    coarsest_n: int = 3,
    omega: float = 0.72,
    pre_steps: int = 1,
    post_steps: int = 1,
    level_index: int = 0,
    max_levels: int | None = None,
    operator_sweeps: int = 3,
    operator_omega: float = 0.70,
    operator_blend: float = 0.75,
    operator_drop_tolerance: float = 1.0e-4,
    operator_candidate_topk: int | None = 8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply one differentiable full V-cycle using shared networks at every level."""

    if terminal(problem.shape, coarsest_n) or (max_levels is not None and max_levels <= 0):
        dense = problem.a.to_dense()
        return torch.linalg.solve(dense, b), {
            "gate_mean": torch.zeros((), device=x.device, dtype=x.dtype)
        }

    gates: list[torch.Tensor] = []
    last_gate: torch.Tensor | None = None
    for _ in range(pre_steps):
        x, gate = _smooth_once(
            problem,
            x,
            b,
            components,
            mode,
            strategy,
            omega=omega,
            reverse=False,
        )
        if gate is not None:
            gates.append(gate)
            last_gate = gate

    residual = b - apply_a(problem, x)
    p, gate, coarse_shape = _prolongation(
        problem,
        residual,
        components,
        mode,
        strategy,
        coarsest_n=coarsest_n,
        level_index=level_index,
        omega=omega,
        reuse_gate=last_gate,
        operator_sweeps=operator_sweeps,
        operator_omega=operator_omega,
        operator_blend=operator_blend,
        operator_drop_tolerance=operator_drop_tolerance,
        operator_candidate_topk=operator_candidate_topk,
    )
    if gate is not None:
        gates.append(gate)
    ac = _galerkin_torch(problem, p)
    rc = _restrict(p, residual)
    child = make_child_problem(ac, coarse_shape)
    if max_levels is not None and max_levels == 1:
        ec = torch.linalg.solve(ac.to_dense(), rc)
    else:
        ec, child_diagnostics = full_v_cycle(
            child,
            torch.zeros_like(rc),
            rc,
            components,
            mode=mode,
            strategy=strategy,
            coarsest_n=coarsest_n,
            omega=omega,
            pre_steps=pre_steps,
            post_steps=post_steps,
            level_index=level_index + 1,
            max_levels=None if max_levels is None else max_levels - 1,
            operator_sweeps=operator_sweeps,
            operator_omega=operator_omega,
            operator_blend=operator_blend,
            operator_drop_tolerance=operator_drop_tolerance,
            operator_candidate_topk=operator_candidate_topk,
        )
        gates.append(child_diagnostics["gate_mean"].reshape(1, 1, 1, 1))
    x = x + _prolong(p, ec)

    for _ in range(post_steps):
        x, gate = _smooth_once(
            problem,
            x,
            b,
            components,
            mode,
            strategy,
            omega=omega,
            reverse=True,
        )
        if gate is not None:
            gates.append(gate)

    gate_mean = (
        torch.mean(torch.stack([item.mean() for item in gates]))
        if gates
        else torch.zeros((), device=x.device, dtype=x.dtype)
    )
    return x, {"gate_mean": gate_mean, "p": p, "ac": ac}


def two_grid_cycle(
    problem: TorchProblem,
    x: torch.Tensor,
    b: torch.Tensor,
    components: NeuralComponents,
    *,
    mode: Mode,
    smoother: SmootherKind = "line_alt",
    transfer: TransferBaseline = "bilinear",
    omega: float = 0.72,
    pre_steps: int = 1,
    post_steps: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    strategy = MGStrategy("two_grid", smoother, transfer, "full")
    return full_v_cycle(
        problem,
        x,
        b,
        components,
        mode=mode,
        strategy=strategy,
        omega=omega,
        pre_steps=pre_steps,
        post_steps=post_steps,
        max_levels=1,
    )


def residual_ratio(
    problem: TorchProblem,
    x_before: torch.Tensor,
    x_after: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    r0 = b - apply_a(problem, x_before)
    r1 = b - apply_a(problem, x_after)
    return torch.linalg.vector_norm(r1) / (torch.linalg.vector_norm(r0) + 1.0e-12)


def energy_norm(problem: TorchProblem, error: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(torch.dot(error, apply_a(problem, error)), min=1.0e-24))


def multi_cycle_energy_contraction(
    problem: TorchProblem,
    error: torch.Tensor,
    components: NeuralComponents,
    *,
    mode: Mode,
    strategy: MGStrategy,
    cycles: int = 2,
    coarsest_n: int = 3,
    omega: float = 0.72,
    pre_steps: int = 1,
    post_steps: int = 1,
    max_levels: int | None = None,
    operator_sweeps: int = 3,
    operator_omega: float = 0.70,
    operator_blend: float = 0.75,
    operator_drop_tolerance: float = 1.0e-4,
    operator_candidate_topk: int | None = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Geometric-mean A-norm contraction over consecutive full V-cycles."""

    if cycles < 1:
        raise ValueError("cycles must be at least one")
    b = torch.zeros_like(error)
    initial = energy_norm(problem, error)
    x = error
    gates: list[torch.Tensor] = []
    for _ in range(cycles):
        x, diagnostics = full_v_cycle(
            problem,
            x,
            b,
            components,
            mode=mode,
            strategy=strategy,
            coarsest_n=coarsest_n,
            omega=omega,
            pre_steps=pre_steps,
            post_steps=post_steps,
            max_levels=max_levels,
            operator_sweeps=operator_sweeps,
            operator_omega=operator_omega,
            operator_blend=operator_blend,
            operator_drop_tolerance=operator_drop_tolerance,
            operator_candidate_topk=operator_candidate_topk,
        )
        gates.append(diagnostics["gate_mean"])
    final = energy_norm(problem, x)
    contraction = torch.pow(final / (initial + 1.0e-12), 1.0 / cycles)
    return contraction, torch.mean(torch.stack(gates)), x


def sample_error(
    shape: int | GridShape,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    smooth_steps: int = 0,
    problem: TorchProblem | None = None,
    smoother: SmootherKind = "line_alt",
    omega: float = 0.72,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    grid_shape = as_shape(shape)
    error = torch.randn(
        unknowns(grid_shape), device=device, dtype=dtype, generator=generator
    )
    error = error / (torch.linalg.vector_norm(error) + 1.0e-12)
    if smooth_steps > 0:
        if problem is None:
            raise ValueError("problem is required when smooth_steps > 0")
        for step in range(smooth_steps):
            residual = -apply_a(problem, error)
            error = error + classical_correction(
                problem,
                residual,
                smoother=smoother,
                omega=omega,
                reverse=bool(step % 2),
            )
        error = error / (torch.linalg.vector_norm(error) + 1.0e-12)
    return error
