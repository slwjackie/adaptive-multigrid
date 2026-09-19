"""Classical point and zebra line smoothers with cached line factorizations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import GridShape, as_shape

SmootherKind = Literal["jacobi", "line_x", "line_y", "line_alt"]


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

    def __post_init__(self) -> None:
        self.shape = as_shape(self.shape)
        if self.a.shape != (self.shape[0] * self.shape[1],) * 2:
            raise ValueError("matrix shape does not match line-smoother grid")

    def get(self, direction: Literal["x", "y"]) -> list[_LineBlock]:
        if direction in self.blocks:
            return self.blocks[direction]
        nx, ny = self.shape
        if direction == "x":
            lines = [np.arange(nx, dtype=np.int64) * ny + j for j in range(ny)]
        elif direction == "y":
            lines = [i * ny + np.arange(ny, dtype=np.int64) for i in range(nx)]
        else:
            raise ValueError(f"unknown direction: {direction}")
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
        return result


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


def classical_smoothing_correction(
    a: sp.csr_matrix,
    shape: int | GridShape,
    residual: np.ndarray,
    smoother: SmootherKind,
    *,
    omega: float = 0.72,
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
        diagonal = np.maximum(np.abs(a.diagonal()), 1.0e-14)
        if work is not None:
            work.add_sparse_flops(2.0 * residual.size, safety=safety)
        return omega * residual / diagonal

    cache = LineSmootherCache(a, grid_shape) if cache is None else cache
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
    raise ValueError(f"unknown smoother: {smoother}")
