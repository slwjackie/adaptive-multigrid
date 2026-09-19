"""Sparse differentiable multilevel V/W-cycle operations for neural-MG training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .numeric_utils import clamp_abs_diagonal_torch
from .grid import GridShape, as_shape, next_shape, terminal, unknowns
CycleKind = Literal["v", "w"]
from .smoothers import LineDirection, SmootherKind
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
NeuralApplication = Literal["replace", "augment", "jacobi_augment"]
CycleDiagnostics = dict[str, torch.Tensor | None]
MAX_DENSE_COARSE_UNKNOWNS = 4096


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
    line_block_cache: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    chebyshev_lambda_max: torch.Tensor | None = None

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
) -> TorchProblem:
    grid_shape = as_shape(shape)
    if a.shape != (unknowns(grid_shape), unknowns(grid_shape)):
        raise ValueError("matrix shape does not match grid")
    sparse = scipy_to_torch_sparse(a, device=device, dtype=dtype)
    features = torch.as_tensor(
        matrix_feature_array(a, grid_shape), device=device, dtype=dtype
    ).unsqueeze(0)
    diag = torch.as_tensor(a.diagonal(), device=device, dtype=dtype)
    return TorchProblem(
        sparse,
        features,
        diag,
        grid_shape,
        a.tocsr(),
    )


def make_child_problem(
    a: torch.Tensor,
    shape: GridShape,
) -> TorchProblem:
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


def _line_offsets(shape: GridShape, direction: LineDirection) -> tuple[int, ...]:
    nx, ny = shape
    if direction == "x":
        return tuple(range(ny))
    if direction == "y":
        return tuple(range(nx))
    if direction == "diag45":
        return tuple(range(-(ny - 1), nx))
    raise ValueError(f"unknown line direction: {direction}")


def _line_indices(shape: GridShape, direction: LineDirection, line: int) -> np.ndarray:
    nx, ny = shape
    if direction == "x":
        return np.arange(nx, dtype=np.int64) * ny + line
    if direction == "y":
        return line * ny + np.arange(ny, dtype=np.int64)
    if direction == "diag45":
        offset = _line_offsets(shape, direction)[line]
        return np.asarray(
            [i * ny + (i - offset) for i in range(nx) if 0 <= i - offset < ny],
            dtype=np.int64,
        )
    raise ValueError(f"unknown line direction: {direction}")


def _line_count(shape: GridShape, direction: LineDirection) -> int:
    return len(_line_offsets(shape, direction))


def _line_block(
    problem: TorchProblem,
    direction: LineDirection,
    line: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Include the tensor identity and mutation version so a future reusable
    # TorchProblem cannot return a block from a stale learned Galerkin graph.
    key = (id(problem.a), int(problem.a._version), direction, line)
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
    total = problem.a.shape[0]
    positions = torch.full(
        (total,), -1, device=problem.device, dtype=torch.long
    ).index_copy(
        0,
        indices,
        torch.arange(indices.numel(), device=problem.device, dtype=torch.long),
    )
    local_rows_all = positions.index_select(0, sparse_indices[0])
    local_cols_all = positions.index_select(0, sparse_indices[1])
    mask = (local_rows_all >= 0) & (local_cols_all >= 0)
    local_rows = local_rows_all[mask]
    local_cols = local_cols_all[mask]
    line_size = int(indices.numel())
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
    direction: LineDirection,
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


def _chebyshev_upper_bound(problem: TorchProblem) -> torch.Tensor:
    if problem.chebyshev_lambda_max is not None:
        return problem.chebyshev_lambda_max
    sparse = problem.a.coalesce()
    rows = sparse.indices()[0]
    values = torch.abs(sparse.values().detach())
    row_sum = torch.zeros(
        problem.a.shape[0], device=problem.device, dtype=problem.dtype
    ).index_add(0, rows, values)
    diagonal = clamp_abs_diagonal_torch(problem.diag.detach())
    problem.chebyshev_lambda_max = 1.05 * torch.max(row_sum / diagonal).clamp(min=1.0e-6)
    return problem.chebyshev_lambda_max


def _chebyshev_correction(
    problem: TorchProblem,
    residual: torch.Tensor,
    *,
    degree: int,
    lower_fraction: float,
) -> torch.Tensor:
    if degree < 1:
        raise ValueError("chebyshev degree must be positive")
    if not 0.0 < lower_fraction < 1.0:
        raise ValueError("chebyshev lower fraction must lie in (0,1)")
    lambda_max = _chebyshev_upper_bound(problem)
    lambda_min = lower_fraction * lambda_max
    center = 0.5 * (lambda_max + lambda_min)
    radius = 0.5 * (lambda_max - lambda_min)
    roots = [
        center
        - radius
        * torch.cos(
            torch.as_tensor(
                (2 * k - 1) * np.pi / (2 * degree),
                device=problem.device,
                dtype=problem.dtype,
            )
        )
        for k in range(1, degree + 1)
    ]
    roots = sorted(roots, key=lambda item: float(item.detach().cpu()), reverse=True)
    correction = torch.zeros_like(residual)
    current = residual
    diagonal = clamp_abs_diagonal_torch(problem.diag)
    for root in roots:
        increment = current / (diagonal * torch.clamp(root, min=1.0e-12))
        correction = correction + increment
        current = current - apply_a(problem, increment)
    return correction


def classical_correction(
    problem: TorchProblem,
    residual: torch.Tensor,
    *,
    smoother: SmootherKind = "line_alt",
    omega: float = 0.72,
    chebyshev_degree: int = 2,
    chebyshev_lower_fraction: float = 0.15,
    reverse: bool = False,
) -> torch.Tensor:
    if smoother == "jacobi":
        return omega * residual / clamp_abs_diagonal_torch(problem.diag)
    if smoother == "chebyshev":
        return _chebyshev_correction(
            problem,
            residual,
            degree=chebyshev_degree,
            lower_fraction=chebyshev_lower_fraction,
        )
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
    if smoother == "line_diag45":
        return _line_direction_correction(
            problem, correction, current, "diag45", reverse=reverse
        )[0]
    raise ValueError(f"unknown smoother: {smoother}")

