"""Classical AMG decisions, kept outside the differentiable weight predictor.

The interpolation support is the *actual* PyAMG classical interpolation pattern,
not all pairs in a two-hop ball. ``max_hops`` is a checked upper bound. The node
selection, symbolic support, and reference row targets are frozen for a forward
and backward pass; another matrix can have a different structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp


class CoarseningError(ValueError):
    """A valid, strictly smaller interpolation structure could not be made."""


def checked_matrix(a):
    if not sp.issparse(a):
        raise TypeError("A must be a sparse matrix")
    a = sp.csr_matrix(a, dtype=np.float64, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    if a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("A must be nonempty and square")
    if not np.isfinite(a.data).all() or not (a.diagonal() > 0).all():
        raise ValueError("A requires finite values and a positive diagonal")
    diff = a - a.T
    scale = max(float(np.max(np.abs(a.data))), 1e-300)
    if diff.nnz and np.max(np.abs(diff.data)) > 1e-12 * scale:
        raise ValueError("neural AMG requires symmetric A")
    # Positive diagonal and symmetry are necessary, not an SPD certificate.
    return a


@dataclass(frozen=True)
class Structure:
    coarse: np.ndarray
    reference: sp.csr_matrix
    rows: np.ndarray = field(init=False)
    cols: np.ndarray = field(init=False)
    row_targets: np.ndarray = field(init=False)
    shape: tuple = field(init=False)

    def __post_init__(self):
        coarse = np.asarray(self.coarse, dtype=np.int64).copy()
        p = sp.csr_matrix(self.reference, dtype=np.float64, copy=True)
        p.sum_duplicates()
        p.sort_indices()
        n, nc = p.shape
        if coarse.ndim != 1 or len(coarse) != nc or not 0 < nc < n:
            raise CoarseningError("Expected one distinct coarse node per P column and 0 < Nc < N")
        if np.any(coarse < 0) or np.any(coarse >= n) or len(np.unique(coarse)) != nc:
            raise CoarseningError("Invalid coarse node indices")
        if not np.isfinite(p.data).all() or np.any(np.diff(p.indptr) == 0):
            raise CoarseningError("Interpolation has nonfinite values or an orphan row")
        injection = p[coarse] - sp.eye(nc, format="csr")
        if injection.nnz and np.max(np.abs(injection.data)) > 1e-12:
            raise CoarseningError("Classical interpolation must have P[C,:] = I")
        # Keep C rows structurally exact: constraint projection cannot introduce
        # off-diagonal entries there, even zero-valued ones.
        if not np.all(np.diff(p.indptr)[coarse] == 1):
            raise CoarseningError("Coarse rows must contain only the injection entry")
        for name, value in (("coarse", coarse), ("reference", p),
                            ("rows", np.repeat(np.arange(n), np.diff(p.indptr))),
                            ("cols", p.indices.copy()),
                            ("row_targets", np.asarray(p.sum(axis=1)).ravel()),
                            ("shape", p.shape)):
            object.__setattr__(self, name, value)


def validate_support_hops(a, structure, max_hops=2):
    """Verify reachability by bounded adjacency expansion, without dense N² data."""
    if max_hops not in (1, 2):
        raise ValueError("This experiment fixes interpolation support to at most 1 or 2 hops")
    if a.shape != (structure.shape[0],) * 2:
        raise ValueError("A and interpolation dimensions differ")
    a = sp.csr_matrix(a)
    for i in range(a.shape[0]):
        start, stop = structure.reference.indptr[i:i + 2]
        candidates = set(structure.coarse[structure.cols[start:stop]].tolist())
        reached, frontier = {i}, {i}
        for _ in range(max_hops):
            new = set()
            for j in frontier:
                new.update(a.indices[a.indptr[j]:a.indptr[j + 1]].tolist())
            frontier = new - reached
            reached.update(new)
            if candidates <= reached:
                break
        if not candidates <= reached:
            raise CoarseningError(f"Interpolation row {i} exceeds {max_hops}-hop support")


def build_structure(a, max_hops=2):
    """Use Ruge–Stüben C/F splitting and classical interpolation from PyAMG.

    Classical strength and its threshold are explicit reproducibility choices;
    this baseline is not claimed optimal for general anisotropic SPD matrices.
    """
    import pyamg

    a = checked_matrix(a)
    hierarchy = pyamg.ruge_stuben_solver(
        a, strength=("classical", {"theta": 0.25}),
        CF=("RS", {"second_pass": True}), interpolation="classical",
        max_levels=2, max_coarse=1, keep=True)
    level = hierarchy.levels[0]
    if not hasattr(level, "P"):
        raise CoarseningError("Classical AMG did not produce a smaller coarse level")
    structure = Structure(np.flatnonzero(level.splitting), level.P)
    validate_support_hops(a, structure, max_hops)
    return structure
