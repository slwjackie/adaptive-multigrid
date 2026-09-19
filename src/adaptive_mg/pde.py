"""Structured-grid P1 finite-element model for anisotropic elliptic diffusion.

The physical model is

    -div(K(x, y) grad u) = f  in (0, 1)^2,
    u = 0                       on the boundary.

The square grid is logically structured.  Each square is split into two P1
triangles, which gives an SPD stiffness matrix for every symmetric positive
definite diffusion tensor K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import scipy.sparse as sp

Pattern = Literal["uniform", "local_patch", "checkerboard", "channel"]


@dataclass(frozen=True)
class DiffusionCase:
    """Parameters defining one heterogeneous/anisotropic diffusion problem.

    Parameters
    ----------
    n:
        Number of interior nodes in each coordinate direction.  Multigrid
        examples normally use ``n = 2**L - 1``.
    epsilon:
        Ratio of the weak to strong eigenvalue of the anisotropic tensor.
        ``epsilon=1`` is isotropic; smaller values are harder for point MG.
    angle_deg:
        Rotation of the principal diffusion direction relative to the grid.
    contrast:
        Multiplicative coefficient contrast used by non-uniform patterns.
    pattern:
        ``uniform``, ``local_patch``, ``checkerboard``, or ``channel``.
    patch_bounds:
        Rectangle ``(xmin, xmax, ymin, ymax)`` used by ``local_patch``.
    checker_cells:
        Number of checkerboard blocks per coordinate direction.
    channel_width:
        Half-width of the diagonal high-conductivity channel.
    """

    n: int = 31
    epsilon: float = 1.0e-2
    angle_deg: float = 45.0
    contrast: float = 1.0e3
    pattern: Pattern = "uniform"
    patch_bounds: tuple[float, float, float, float] = (0.30, 0.70, 0.30, 0.70)
    checker_cells: int = 4
    channel_width: float = 0.08

    def validate(self) -> None:
        if self.n < 3:
            raise ValueError("n must be at least 3")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.contrast <= 0.0:
            raise ValueError("contrast must be positive")
        if self.pattern not in {"uniform", "local_patch", "checkerboard", "channel"}:
            raise ValueError(f"unsupported pattern: {self.pattern}")


def _rotation(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _base_tensor(epsilon: float, angle_deg: float) -> np.ndarray:
    r = _rotation(angle_deg)
    return r @ np.diag([1.0, epsilon]) @ r.T


def diffusion_tensor(case: DiffusionCase, x: float, y: float) -> np.ndarray:
    """Evaluate the SPD diffusion tensor at a physical point."""

    anisotropic = _base_tensor(case.epsilon, case.angle_deg)
    identity = np.eye(2, dtype=np.float64)

    if case.pattern == "uniform":
        return anisotropic

    if case.pattern == "local_patch":
        xmin, xmax, ymin, ymax = case.patch_bounds
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return case.contrast * anisotropic
        return identity

    if case.pattern == "checkerboard":
        ix = min(int(x * case.checker_cells), case.checker_cells - 1)
        iy = min(int(y * case.checker_cells), case.checker_cells - 1)
        if (ix + iy) % 2 == 0:
            return case.contrast * anisotropic
        return identity

    if case.pattern == "channel":
        # A rotated difficult region embedded in an otherwise easy operator.
        if abs(y - x) <= case.channel_width:
            return case.contrast * anisotropic
        return identity

    raise AssertionError("case.validate() should reject unknown patterns")


def _triangle_stiffness(coords: np.ndarray, tensor: np.ndarray) -> np.ndarray:
    """Return the 3x3 P1 element stiffness matrix."""

    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]
    twice_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * abs(twice_area)
    if area <= 0.0:
        raise ValueError("degenerate triangle")

    # Gradients of barycentric basis functions.  The signed denominator is
    # required so the formula works for either orientation.
    grads = np.array(
        [
            [y1 - y2, x2 - x1],
            [y2 - y0, x0 - x2],
            [y0 - y1, x1 - x0],
        ],
        dtype=np.float64,
    ) / twice_area
    return area * (grads @ tensor @ grads.T)


def _interior_index(global_i: int, global_j: int, n: int) -> int | None:
    """Map a full-grid node to an interior vector index, or return None."""

    if 1 <= global_i <= n and 1 <= global_j <= n:
        return (global_i - 1) * n + (global_j - 1)
    return None


def assemble_stiffness(case: DiffusionCase) -> sp.csr_matrix:
    """Assemble the sparse SPD stiffness matrix."""

    case.validate()
    n = case.n
    h = 1.0 / (n + 1)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # Full-grid node numbers run from 0 through n+1 in each direction.
    for i in range(n + 1):
        for j in range(n + 1):
            square = [
                (i, j),
                (i + 1, j),
                (i + 1, j + 1),
                (i, j + 1),
            ]
            # A fixed diagonal keeps the grid logically structured.
            triangles = ((square[0], square[1], square[2]), (square[0], square[2], square[3]))
            for tri in triangles:
                coords = np.array([(gi * h, gj * h) for gi, gj in tri], dtype=np.float64)
                centroid = coords.mean(axis=0)
                tensor = diffusion_tensor(case, float(centroid[0]), float(centroid[1]))
                local = _triangle_stiffness(coords, tensor)
                mapped = [_interior_index(gi, gj, n) for gi, gj in tri]
                for a, row in enumerate(mapped):
                    if row is None:
                        continue
                    for b, col in enumerate(mapped):
                        if col is None:
                            continue
                        rows.append(row)
                        cols.append(col)
                        vals.append(float(local[a, b]))

    a = sp.coo_matrix((vals, (rows, cols)), shape=(n * n, n * n), dtype=np.float64).tocsr()
    a.sum_duplicates()
    # Remove tiny asymmetry from floating-point accumulation.
    a = (0.5 * (a + a.T)).tocsr()
    return a


def manufactured_solution(n: int) -> np.ndarray:
    """Smooth zero-boundary discrete target used by examples and tests."""

    coords = np.arange(1, n + 1, dtype=np.float64) / (n + 1)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    return (np.sin(np.pi * xx) * np.sin(np.pi * yy)).reshape(-1)


def assemble_problem(case: DiffusionCase) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Return ``(A, b, u_exact)`` for a manufactured discrete problem."""

    a = assemble_stiffness(case)
    u_exact = manufactured_solution(case.n)
    b = np.asarray(a @ u_exact, dtype=np.float64)
    return a, b, u_exact


def case_suite(n: int = 31) -> list[tuple[str, DiffusionCase]]:
    """Small benchmark suite spanning easy and hard MG regimes."""

    return [
        ("poisson", DiffusionCase(n=n, epsilon=1.0, angle_deg=0.0, contrast=1.0, pattern="uniform")),
        ("aligned_anisotropy", DiffusionCase(n=n, epsilon=1.0e-3, angle_deg=0.0, contrast=1.0, pattern="uniform")),
        ("rotated_anisotropy", DiffusionCase(n=n, epsilon=1.0e-3, angle_deg=45.0, contrast=1.0, pattern="uniform")),
        ("local_patch", DiffusionCase(n=n, epsilon=1.0e-3, angle_deg=45.0, contrast=1.0e3, pattern="local_patch")),
        ("checkerboard", DiffusionCase(n=n, epsilon=1.0e-2, angle_deg=30.0, contrast=1.0e4, pattern="checkerboard")),
        ("channel", DiffusionCase(n=n, epsilon=1.0e-3, angle_deg=45.0, contrast=1.0e3, pattern="channel")),
    ]
