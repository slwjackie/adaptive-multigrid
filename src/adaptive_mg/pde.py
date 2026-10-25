"""Structured-grid P1/Q1 finite elements for anisotropic elliptic diffusion.

The physical model is

    -div(K(x, y) grad u) = f  in (0, 1)^2,
    u = 0                       on the boundary.

The default splits each square into two P1 triangles.  The optional Q1 method
uses four bilinear shape functions on each square and 2x2 Gauss quadrature.
Both eliminate zero-Dirichlet boundary nodes and preserve the same ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import scipy.sparse as sp

Pattern = Literal["uniform", "local_patch", "checkerboard", "channel"]
Discretization = Literal["p1", "q1"]
TensorField = Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]


def validate_discretization(value: str) -> Discretization:
    """Validate a discretization selector without silently changing its case."""
    if not isinstance(value, str) or value not in ("p1", "q1"):
        raise ValueError(f"unsupported discretization: {value!r}")
    return value


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
    discretization:
        ``p1`` (default): two centroid-integrated triangles per square.
        ``q1``: one bilinear square with 2x2 Gauss quadrature.
    """

    n: int = 31
    epsilon: float = 1.0e-2
    angle_deg: float = 45.0
    contrast: float = 1.0e3
    pattern: Pattern = "uniform"
    patch_bounds: tuple[float, float, float, float] = (0.30, 0.70, 0.30, 0.70)
    checker_cells: int = 4
    channel_width: float = 0.08
    discretization: Discretization = "p1"

    def validate(self) -> None:
        if self.n < 3:
            raise ValueError("n must be at least 3")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.contrast <= 0.0:
            raise ValueError("contrast must be positive")
        if self.pattern not in {"uniform", "local_patch", "checkerboard", "channel"}:
            raise ValueError(f"unsupported pattern: {self.pattern}")
        validate_discretization(self.discretization)


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


def assemble_stiffness_reference(case: DiffusionCase) -> sp.csr_matrix:
    """Assemble the sparse SPD stiffness matrix."""

    case.validate()
    if case.discretization == "q1":
        return assemble_q1_stiffness_reference(
            case.n, lambda x, y: diffusion_tensor(case, x, y))
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


def assemble_stiffness(case: DiffusionCase) -> sp.csr_matrix:
    """Assemble the selected structured-grid stiffness matrix.

    P1 retains the original centroid-integrated, vectorized implementation.
    Q1 evaluates the same coefficient field at four Gauss points per square.
    """
    case.validate();n=case.n
    if case.discretization == "q1":
        return assemble_q1_stiffness(n, lambda x, y: _diffusion_tensor_array(case, x, y))
    ii,jj=np.meshgrid(np.arange(n+1),np.arange(n+1),indexing="ij")
    ii=ii.ravel();jj=jj.ravel();count=len(ii)
    all_rows=[];all_cols=[];all_vals=[]
    templates=(
        (np.array([[0,0],[1,0],[1,1]]),np.array([[-1,0],[1,-1],[0,1]],dtype=float)),
        (np.array([[0,0],[1,1],[0,1]]),np.array([[0,-1],[1,0],[-1,1]],dtype=float)))
    anis=_base_tensor(case.epsilon,case.angle_deg)
    for offsets,grads in templates:
        x=(ii+offsets[:,0].mean())/(n+1);y=(jj+offsets[:,1].mean())/(n+1)
        if case.pattern=="uniform":
            tensors=np.broadcast_to(anis,(count,2,2))
        else:
            if case.pattern=="local_patch":
                xmin,xmax,ymin,ymax=case.patch_bounds
                hard=(x>=xmin)&(x<=xmax)&(y>=ymin)&(y<=ymax)
            elif case.pattern=="checkerboard":
                hard=((np.minimum((x*case.checker_cells).astype(int),case.checker_cells-1)+np.minimum((y*case.checker_cells).astype(int),case.checker_cells-1))%2)==0
            else:hard=np.abs(y-x)<=case.channel_width
            tensors=np.where(hard[:,None,None],case.contrast*anis,np.eye(2))
        local=.5*np.einsum("ik,nkl,jl->nij",grads,tensors,grads)
        gi=ii[:,None]+offsets[None,:,0];gj=jj[:,None]+offsets[None,:,1]
        valid=(gi>=1)&(gi<=n)&(gj>=1)&(gj<=n)
        idx=(gi-1)*n+(gj-1)
        mask=valid[:,:,None]&valid[:,None,:]
        rr=np.broadcast_to(idx[:,:,None],(count,3,3));cc=np.broadcast_to(idx[:,None,:],(count,3,3))
        all_rows.append(rr[mask]);all_cols.append(cc[mask]);all_vals.append(local[mask])
    a=sp.coo_matrix((np.concatenate(all_vals),(np.concatenate(all_rows),np.concatenate(all_cols))),shape=(n*n,n*n)).tocsr()
    a.sum_duplicates();a=(.5*(a+a.T)).tocsr();a.eliminate_zeros()
    return a


def _diffusion_tensor_array(case: DiffusionCase, x, y) -> np.ndarray:
    """Vectorized counterpart of ``diffusion_tensor`` with identical regions."""
    x, y = np.broadcast_arrays(x, y)
    anisotropic = _base_tensor(case.epsilon, case.angle_deg)
    if case.pattern == "uniform":
        return np.broadcast_to(anisotropic, x.shape + (2, 2))
    if case.pattern == "local_patch":
        xmin, xmax, ymin, ymax = case.patch_bounds
        hard = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    elif case.pattern == "checkerboard":
        ix = np.minimum((x * case.checker_cells).astype(int), case.checker_cells - 1)
        iy = np.minimum((y * case.checker_cells).astype(int), case.checker_cells - 1)
        hard = (ix + iy) % 2 == 0
    else:
        hard = np.abs(y - x) <= case.channel_width
    return np.where(hard[..., None, None], case.contrast * anisotropic, np.eye(2))


def _validate_q1_size(n: int) -> None:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")


def assemble_q1_stiffness(n: int, tensor_field: TensorField) -> sp.csr_matrix:
    """Assemble Q1 stiffness on the unit square with homogeneous Dirichlet data.

    ``tensor_field(x, y)`` receives arrays of cell quadrature coordinates and
    returns symmetric positive-definite tensors of shape ``(..., 2, 2)``;
    a constant ``(2, 2)`` tensor may also be returned.  The caller owns the SPD
    contract.  Local node order is bottom-left, bottom-right, top-right,
    top-left; global interior index is ``(i - 1) * n + (j - 1)``.

    For reference coordinates (xi, eta) in [0,1]^2, physical gradients scale
    by 1/h and area weights by h**2.  These cancel on this square mesh, so the
    formula below uses reference gradients and Gauss weights 1/4 directly.
    Quadrature is exact for cellwise constant K; interfaces cutting a cell
    and general variable coefficients are sampled approximately.
    """
    _validate_q1_size(n)
    ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    count = len(ii)
    local = np.zeros((count, 4, 4), dtype=np.float64)
    gauss = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))
    for xi in gauss:
        for eta in gauss:
            grads = np.array([
                [-(1.0 - eta), -(1.0 - xi)],
                [1.0 - eta, -xi],
                [eta, xi],
                [-eta, 1.0 - xi],
            ])
            tensors = np.broadcast_to(
                np.asarray(tensor_field((ii + xi) / (n + 1),
                                        (jj + eta) / (n + 1)), dtype=np.float64),
                (count, 2, 2))
            local += 0.25 * np.einsum("ik,nkl,jl->nij", grads, tensors, grads)
    offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    gi = ii[:, None] + offsets[None, :, 0]
    gj = jj[:, None] + offsets[None, :, 1]
    valid = (gi >= 1) & (gi <= n) & (gj >= 1) & (gj <= n)
    idx = (gi - 1) * n + (gj - 1)
    mask = valid[:, :, None] & valid[:, None, :]
    rows = np.broadcast_to(idx[:, :, None], (count, 4, 4))[mask]
    cols = np.broadcast_to(idx[:, None, :], (count, 4, 4))[mask]
    a = sp.coo_matrix((local[mask], (rows, cols)), shape=(n * n, n * n)).tocsr()
    a.sum_duplicates()
    a = (0.5 * (a + a.T)).tocsr()
    a.eliminate_zeros()
    return a


def assemble_q1_stiffness_reference(n: int, tensor_field: TensorField) -> sp.csr_matrix:
    """Independent scalar Q1 reference using [-1,1]^2 and an explicit Jacobian.

    ``tensor_field`` must also accept scalar coordinates.  This implementation
    is intended for small-grid parity checks, not production data generation.
    """
    _validate_q1_size(n)
    h = 1.0 / (n + 1)
    jacobian = np.diag([h / 2.0, h / 2.0])
    inverse_jacobian = np.linalg.inv(jacobian)
    determinant = np.linalg.det(jacobian)
    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    rows, cols, values = [], [], []
    for i in range(n + 1):
        for j in range(n + 1):
            local = np.zeros((4, 4))
            for xi in (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)):
                for eta in (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)):
                    gradients = np.column_stack((
                        signs[:, 0] * (1.0 + signs[:, 1] * eta),
                        signs[:, 1] * (1.0 + signs[:, 0] * xi))) / 4.0
                    gradients = gradients @ inverse_jacobian
                    tensor = tensor_field((i + (xi + 1.0) / 2.0) * h,
                                          (j + (eta + 1.0) / 2.0) * h)
                    local += determinant * (gradients @ tensor @ gradients.T)
            nodes = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            mapped = [_interior_index(gi, gj, n) for gi, gj in nodes]
            for a, row in enumerate(mapped):
                for b, col in enumerate(mapped):
                    if row is not None and col is not None:
                        rows.append(row)
                        cols.append(col)
                        values.append(local[a, b])
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(n * n, n * n)).tocsr()
    matrix.sum_duplicates()
    return (0.5 * (matrix + matrix.T)).tocsr()


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
