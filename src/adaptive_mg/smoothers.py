"""Classical point, Chebyshev, and direction-aware line smoothers."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .numeric_utils import clamp_abs_diagonal_numpy
from .grid import GridShape, as_shape

SmootherKind = Literal[
    "jacobi",
    "chebyshev",
    "line_x",
    "line_y",
    "line_alt",
    "line_diag45",
]
LineDirection = Literal["x", "y", "diag45"]


class WorkRecorder(Protocol):
    def add_sparse_flops(self, flops: float, *, safety: bool = False) -> None: ...


@dataclass
class _LineBlock:
    indices: np.ndarray
    solve: object
    columns: sp.csr_matrix
    block_nnz: int


@dataclass
class LineSmootherCache:
    a: sp.csr_matrix
    shape: int | GridShape
    blocks: dict[str, list[_LineBlock]] = field(default_factory=dict)
    chebyshev_lambda_max: float | None = None
    build_seconds: float = 0.0
    factorization_count: int = 0

    def __post_init__(self) -> None:
        self.shape = as_shape(self.shape)
        if self.a.shape != (self.shape[0] * self.shape[1],) * 2:
            raise ValueError("matrix shape does not match line-smoother grid")

    def _line_sets(self, direction: LineDirection) -> list[np.ndarray]:
        nx, ny = self.shape
        if direction == "x":
            return [np.arange(nx, dtype=np.int64) * ny + j for j in range(ny)]
        if direction == "y":
            return [i * ny + np.arange(ny, dtype=np.int64) for i in range(nx)]
        if direction == "diag45":
            lines: list[np.ndarray] = []
            # +45-degree lines are level sets i-j=c.  The ordering starts at
            # the lower-left short diagonal and proceeds to the upper-right.
            for offset in range(-(ny - 1), nx):
                indices = [
                    i * ny + (i - offset)
                    for i in range(nx)
                    if 0 <= i - offset < ny
                ]
                if indices:
                    lines.append(np.asarray(indices, dtype=np.int64))
            return lines
        raise ValueError(f"unknown direction: {direction}")

    def get(self, direction: LineDirection) -> list[_LineBlock]:
        if direction in self.blocks:
            return self.blocks[direction]
        build_start = perf_counter()
        lines = self._line_sets(direction)
        result: list[_LineBlock] = []
        for indices in lines:
            block = self.a[indices][:, indices].tocsc()
            lu = spla.splu(block)
            result.append(
                _LineBlock(
                    indices=indices,
                    solve=lu.solve,
                    columns=self.a[:, indices].tocsr(),
                    block_nnz=block.nnz,
                )
            )
        self.blocks[direction] = result
        self.build_seconds += perf_counter() - build_start
        self.factorization_count += len(result)
        return result

    def spectral_upper_bound(self) -> float:
        """Gershgorin upper bound for eigenvalues of D^-1 A."""

        if self.chebyshev_lambda_max is None:
            diagonal = clamp_abs_diagonal_numpy(self.a.diagonal())
            row_sum = np.asarray(np.abs(self.a).sum(axis=1)).reshape(-1)
            self.chebyshev_lambda_max = max(
                1.05 * float(np.max(row_sum / diagonal)), 1.0e-6
            )
        return self.chebyshev_lambda_max


def _zebra_order(number_of_lines: int, reverse: bool) -> list[int]:
    order = list(range(0, number_of_lines, 2)) + list(range(1, number_of_lines, 2))
    return list(reversed(order)) if reverse else order


def _directional(
    correction: np.ndarray,
    current: np.ndarray,
    blocks: list[_LineBlock],
    *,
    reverse: bool,
    work: WorkRecorder | None,
    safety: bool,
) -> tuple[np.ndarray, np.ndarray]:
    for line in _zebra_order(len(blocks), reverse):
        block = blocks[line]
        increment = np.asarray(block.solve(current[block.indices]), dtype=np.float64)
        correction[block.indices] += increment
        current = current - np.asarray(block.columns @ increment, dtype=np.float64).reshape(-1)
        if work is not None:
            m = block.indices.size
            work.add_sparse_flops(
                2.0 * block.columns.nnz + max(2.0 * block.block_nnz, 2.0 * m),
                safety=safety,
            )
    return correction, current


def _chebyshev_correction(
    a: sp.csr_matrix,
    residual: np.ndarray,
    cache: LineSmootherCache,
    *,
    degree: int,
    lower_fraction: float,
    work: WorkRecorder | None,
    safety: bool,
) -> np.ndarray:
    """Apply a fixed-degree Chebyshev-Jacobi polynomial smoother.

    The polynomial roots cover the upper spectral interval of ``D^-1 A``;
    ordering them from large to small eigenvalues keeps intermediate updates
    conservative while preserving the final Chebyshev polynomial.
    """

    if degree < 1:
        raise ValueError("chebyshev degree must be positive")
    if not 0.0 < lower_fraction < 1.0:
        raise ValueError("chebyshev lower fraction must lie in (0,1)")
    diagonal = clamp_abs_diagonal_numpy(a.diagonal())
    lambda_max = cache.spectral_upper_bound()
    lambda_min = lower_fraction * lambda_max
    center = 0.5 * (lambda_max + lambda_min)
    radius = 0.5 * (lambda_max - lambda_min)
    roots = [
        center - radius * np.cos((2 * k - 1) * np.pi / (2 * degree))
        for k in range(1, degree + 1)
    ]
    correction = np.zeros_like(residual)
    current = residual.copy()
    for root in sorted(roots, reverse=True):
        increment = (1.0 / max(float(root), 1.0e-14)) * current / diagonal
        correction += increment
        current -= np.asarray(a @ increment, dtype=np.float64).reshape(-1)
        if work is not None:
            work.add_sparse_flops(2.0 * residual.size + 2.0 * a.nnz, safety=safety)
    return correction


def classical_smoothing_correction(
    a: sp.csr_matrix,
    shape: int | GridShape,
    residual: np.ndarray,
    smoother: SmootherKind,
    *,
    omega: float = 0.72,
    chebyshev_degree: int = 2,
    chebyshev_lower_fraction: float = 0.15,
    cache: LineSmootherCache | None = None,
    reverse: bool = False,
    work: WorkRecorder | None = None,
    safety: bool = False,
) -> np.ndarray:
    """Return one point/line relaxation correction for a supplied residual."""

    grid_shape = as_shape(shape)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if residual.size != grid_shape[0] * grid_shape[1]:
        raise ValueError("residual size does not match grid shape")
    if smoother == "jacobi":
        diagonal = clamp_abs_diagonal_numpy(a.diagonal())
        if work is not None:
            work.add_sparse_flops(2.0 * residual.size, safety=safety)
        return omega * residual / diagonal

    cache = LineSmootherCache(a, grid_shape) if cache is None else cache
    if smoother == "chebyshev":
        return _chebyshev_correction(
            a,
            residual,
            cache,
            degree=chebyshev_degree,
            lower_fraction=chebyshev_lower_fraction,
            work=work,
            safety=safety,
        )
    correction = np.zeros_like(residual)
    current = residual.copy()
    if smoother == "line_x":
        return _directional(
            correction, current, cache.get("x"), reverse=reverse, work=work, safety=safety
        )[0]
    if smoother == "line_y":
        return _directional(
            correction, current, cache.get("y"), reverse=reverse, work=work, safety=safety
        )[0]
    if smoother == "line_alt":
        directions = ("y", "x") if reverse else ("x", "y")
        for direction in directions:
            correction, current = _directional(
                correction,
                current,
                cache.get(direction),  # type: ignore[arg-type]
                reverse=reverse,
                work=work,
                safety=safety,
            )
        return correction
    if smoother == "line_diag45":
        return _directional(
            correction,
            current,
            cache.get("diag45"),
            reverse=reverse,
            work=work,
            safety=safety,
        )[0]
    raise ValueError(f"unknown smoother: {smoother}")
