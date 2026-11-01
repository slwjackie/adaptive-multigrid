"""Fixed-coarsening bootstrap AMG labels for direct, sparse prolongation.

The teacher is a numerical algorithm, not another neural network.  Each setup
pass fits relaxed test vectors, constructs an actual Galerkin hierarchy, and
then evaluates that hierarchy on independent TRAIN probes.  Subsequent passes
retain the initial vectors and add generalized multilevel eigenvectors and,
optionally, errors left by the current hierarchy.  A one-pass run is explicitly
reported as relaxed-vector LS rather than bootstrap AMG.

Only local least-squares, small Ritz spaces, and the terminal eigenproblem are
dense.  Fine-grid operators and the Galerkin mass matrices remain sparse.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import time
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..config import MGConfig
from ..grid import as_shape, next_shape, terminal, validate_root_shape
from ..smoothers import LineSmootherCache, classical_smoothing_correction
from ..strategy import get_strategy
from ..transfer import TransferPattern, coarse_fine_indices


@dataclass(frozen=True)
class TeacherConfig:
    """Budgets are offline setup costs; two passes include initial construction.

    Unit-norm vectors and weights summing to one make ``mu=1e-6`` a small,
    explicit starting regularization, independent of vector count.  This is an
    experimental default, not an optimal or scale-invariant choice.
    """

    vectors: int = 16
    sweeps: tuple[int, ...] = (2, 4, 8)
    setup_passes: int = 2
    mu: float = 1e-6
    k: int = 4
    l1_cap: float = 8.0
    seed: int = 1
    selection_vectors: int = 16
    audit_vectors: int = 16
    selection_cycles: int = 2
    eigen_count: int = 4
    eigen_refine: int = 2
    adaptive_vectors: int = 4
    adaptive_cycles: int = 1
    augment_vectors: int = 4
    max_vector_retries: int = 1
    rank_tolerance: float = 1e-10
    condition_limit: float = 1e10
    weighting: str = "uniform"
    energy_weight_cap: float = 100.0
    max_operator_complexity: float = 30.0
    max_mass_nnz: int = 10_000_000

    def __post_init__(self):
        object.__setattr__(self, "sweeps", tuple(self.sweeps))
        positive = ("vectors", "setup_passes", "k", "selection_vectors", "audit_vectors",
                    "selection_cycles", "adaptive_cycles", "max_mass_nnz")
        nonnegative = ("seed", "eigen_count", "eigen_refine", "adaptive_vectors",
                       "augment_vectors", "max_vector_retries")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < (1 if name in positive else 0):
                raise ValueError(f"invalid {name}")
        if self.k > 16:
            raise ValueError("k cannot exceed the 16 local candidates")
        if not self.sweeps or any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                                  for x in self.sweeps):
            raise ValueError("sweeps must be nonempty nonnegative integers")
        for name in ("mu", "l1_cap", "rank_tolerance", "condition_limit",
                     "energy_weight_cap", "max_operator_complexity"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.mu < 0 or self.l1_cap < 1:
            raise ValueError("mu must be nonnegative and l1_cap at least one")
        if not 0 < self.rank_tolerance < 1 or self.condition_limit < 1:
            raise ValueError("invalid rank/condition thresholds")
        if self.weighting not in ("uniform", "energy") or self.energy_weight_cap < 1:
            raise ValueError("invalid test-vector weighting")
        if self.max_operator_complexity < 1:
            raise ValueError("max_operator_complexity must be at least one")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TeacherConfig":
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TeacherLevel:
    A: sp.csr_matrix
    shape: tuple[int, int]
    H: sp.csr_matrix
    pattern: TransferPattern | None = None
    mask: np.ndarray | None = None
    weights: np.ndarray | None = None
    P: sp.csr_matrix | None = None
    vectors: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    cache: LineSmootherCache | None = field(default=None, repr=False)
    lu: Any = field(default=None, repr=False)

    @property
    def mass(self):
        """Compatibility alias; mass H is distinct from the connection mask."""
        return self.H

    def __getstate__(self):
        state = dict(self.__dict__)
        # SuperLU and cached factors are runtime state, not serialized labels.
        state["cache"] = None
        state["lu"] = None
        return state


@dataclass
class TeacherResult:
    levels: list[TeacherLevel]
    diagnostics: dict[str, Any]
    setup_seconds: float
    config: TeacherConfig
    mg_config: MGConfig


def matrix_digest(A: sp.spmatrix) -> str:
    a = A.tocsr(copy=True)
    a.sum_duplicates()
    a.sort_indices()
    digest = hashlib.sha256()
    for data in (np.asarray(a.shape, dtype=np.int64), a.indptr.astype(np.int64),
                 a.indices.astype(np.int64), a.data.astype(np.float64)):
        digest.update(data.tobytes())
    return digest.hexdigest()


def _checked_matrix(A, shape=None):
    a = sp.csr_matrix(A, dtype=np.float64, copy=True)
    a.sum_duplicates()
    a.sort_indices()
    if a.shape[0] != a.shape[1] or a.shape[0] == 0 or not np.isfinite(a.data).all():
        raise ValueError("A must be a finite, nonempty square sparse matrix")
    if shape is not None and a.shape[0] != int(np.prod(shape)):
        raise ValueError("matrix and grid shape do not match")
    defect = spla.norm(a - a.T) / max(spla.norm(a), np.finfo(float).tiny)
    if defect > 1e-11 or np.any(a.diagonal() <= 0):
        raise ValueError("teacher requires a symmetric SPD operator")
    return a


def _normalise(vectors):
    z = np.asarray(vectors, dtype=np.float64)
    if z.ndim != 2 or not z.shape[1] or not np.isfinite(z).all():
        raise ValueError("test vectors must be a finite, nonempty column matrix")
    norms = np.linalg.norm(z, axis=0)
    if np.any(norms < 1e-100):
        raise ValueError("test vector collapsed to numerical zero")
    return z / norms


def _correction(a, shape, r, cache, mg, reverse=False):
    return classical_smoothing_correction(
        a, shape, r, "line_alt", omega=mg.jacobi_omega,
        chebyshev_degree=mg.chebyshev_degree,
        chebyshev_lower_fraction=mg.chebyshev_lower_fraction,
        cache=cache, reverse=reverse)


def relaxed_vectors(A, shape, count, sweeps=(2, 4, 8), *, seed=1, mg_config=None,
                    cache=None):
    """Independent initial errors, divided across the requested sweep budgets.

    At most ``n`` initial vectors are generated on a grid with ``n`` unknowns.
    Collapsed vectors are regenerated with a bounded budget, never silently
    removed.  Applying relaxation to A z = 0 requires no exact physical solve.
    """
    mg = mg_config or MGConfig(mode="classical")
    shape = as_shape(shape)
    a = _checked_matrix(A, shape)
    requested = int(count)
    count = min(requested, a.shape[0])
    if count < 1 or not sweeps or any(s < 0 for s in sweeps):
        raise ValueError("invalid vector/sweep budget")
    cache = cache or LineSmootherCache(a, shape)
    rng = np.random.default_rng(seed)
    z = np.empty((a.shape[0], count))
    used = []
    regenerated = 0
    for j in range(count):
        budget = int(sweeps[j % len(sweeps)])
        for attempt in range(4):
            v = rng.standard_normal(a.shape[0])
            for _ in range(budget):
                v += _correction(a, shape, -(a @ v), cache, mg)
                norm = np.linalg.norm(v)
                if not np.isfinite(norm) or norm < 1e-100:
                    break
                # Rescaling preserves direction and avoids underflow when a
                # strong line smoother annihilates most of an initial error.
                v /= norm
            norm = np.linalg.norm(v)
            if np.isfinite(norm) and norm >= 1e-100:
                z[:, j] = v / norm
                break
            regenerated += 1
        else:
            raise FloatingPointError(f"test vector {j} collapsed after four attempts")
        used.append(budget)
    return z, {"seed": int(seed), "requested": requested, "actual": count,
               "sweeps_per_vector": used, "regenerated": regenerated}


def _vector_weights(A, z, config):
    if config.weighting == "uniform":
        return np.full(z.shape[1], 1.0 / z.shape[1])
    energy = np.einsum("ij,ij->j", z, A @ z)
    if np.any(energy <= 0) or not np.isfinite(energy).all():
        raise ValueError("positive test-vector energies required for energy weighting")
    # Capping the ratio to the smallest weight bounds dominance by one mode.
    inverse = 1.0 / np.maximum(energy, np.finfo(float).tiny)
    inverse = np.minimum(inverse, inverse.min() * config.energy_weight_cap)
    return inverse / inverse.sum()


def _row_fit(B, y, vector_weights, target, config):
    """Equality-constrained absolute ridge, with optional signed L1 bound."""
    d = B.shape[1]
    if not d:
        raise ValueError("a row cannot have empty support")
    p0 = np.full(d, target / d)
    null = la.null_space(np.ones((1, d))) if d > 1 else np.empty((d, 0))
    sqrtw = np.sqrt(vector_weights)
    design = sqrtw[:, None] * B
    rhs = sqrtw * y
    free = design @ null
    singular = la.svdvals(free) if d > 1 else np.empty(0)
    # Compare with the original design scale as well: projection of identical
    # columns into the equality nullspace is roundoff, not recovered rank.
    threshold = config.rank_tolerance * max(float(la.norm(design, 2)),
                                            singular[0] if len(singular) else 0.0)
    rank = int(np.count_nonzero(singular > max(threshold, np.finfo(float).tiny)))
    full_rank = rank == d - 1
    condition = (float(singular[0] / singular[-1])
                 if len(singular) and full_rank else (1.0 if d == 1 else math.inf))
    if d > 1:
        # Explicitly penalize p0 + N q, not q relative to an arbitrary base P.
        augmented = np.vstack((free, math.sqrt(config.mu) * null))
        augmented_rhs = np.concatenate((rhs - design @ p0,
                                        -math.sqrt(config.mu) * p0))
        u, augmented_singular, vh = la.svd(augmented, full_matrices=False)
        keep = augmented_singular > config.rank_tolerance * max(
            float(la.norm(design, 2)), float(augmented_singular[0]), np.finfo(float).tiny)
        q = vh[keep].T @ ((u[:, keep].T @ augmented_rhs) / augmented_singular[keep])
        p = p0 + null @ q
        regularized_condition = (float(augmented_singular[0] / augmented_singular[-1])
                                 if augmented_singular[-1] > np.finfo(float).tiny else math.inf)
    else:
        p = p0
        regularized_condition = 1.0
    clipped = False
    if np.sum(np.abs(p)) > config.l1_cap + 1e-10:
        # Convex QP in p = u-v, u,v >= 0.  Unlike naive clipping, this maintains
        # the equality and minimizes the actual stated constrained objective.
        def objective(uv):
            value = uv[:d] - uv[d:]
            residual = design @ value - rhs
            grad = 2 * (design.T @ residual + config.mu * value)
            return (float(residual @ residual + config.mu * (value @ value)),
                    np.concatenate((grad, -grad)))

        signs = np.concatenate((np.ones(d), -np.ones(d)))
        feasible = np.concatenate((np.maximum(p0, 0), np.maximum(-p0, 0)))
        fit = opt.minimize(objective, feasible, jac=True, method="SLSQP",
                           bounds=[(0, None)] * (2 * d), constraints=[
                               {"type": "eq", "fun": lambda x: signs @ x - target,
                                "jac": lambda x: signs},
                               {"type": "ineq", "fun": lambda x: config.l1_cap - x.sum(),
                                "jac": lambda x: -np.ones_like(x)}],
                           options={"ftol": 1e-12, "maxiter": 200})
        if not fit.success:
            raise RuntimeError(f"constrained local LS failed: {fit.message}")
        p = fit.x[:d] - fit.x[d:]
        clipped = True
    if (not np.isfinite(p).all() or abs(p.sum() - target) > 1e-8
            or np.abs(p).sum() > config.l1_cap + 1e-8):
        raise FloatingPointError("local LS violated equality, finiteness, or L1 constraints")
    error = design @ p - rhs
    return p, {"objective": float(error @ error + config.mu * (p @ p)),
               "fit_error": float(error @ error), "free_dimension": d - 1,
               "free_rank": rank, "rank_deficient": not full_rank,
               # None denotes an infinite/undefined condition number while
               # preserving strict JSON output; rank_deficient is explicit.
               "free_condition": condition if math.isfinite(condition) else None,
               "free_smallest_singular": float(singular[-1]) if len(singular) else None,
               "effective_mu": config.mu, "l1_active": clipped,
               "regularized_condition": (regularized_condition if math.isfinite(regularized_condition)
                                          else None),
               "solution_status": ("regularized_rank_deficient" if config.mu > 0 else
                                   "minimum_norm_rank_deficient") if not full_rank else "full_rank"}


def local_fit(A, pattern, vectors, *, config=None, fixed_mask=None, heldout_vectors=None):
    """Fit weights and optional greedy support on a fixed nested grid.

    Returns ``(weights[N,16], mask[N,16], diagnostics)``.  Greedy selection is
    bounded by at most 16 candidates per row and is not a globally optimal
    support search.  Fixed-mask refitting preserves selected zero-valued slots.
    """
    from .bamg_transfer import geometry_constraints

    config = config or TeacherConfig()
    a = _checked_matrix(A, pattern.fine_shape)
    z = _normalise(vectors)
    if z.shape[0] != a.shape[0]:
        raise ValueError("test-vector shape does not match A")
    targets, injection, valid = geometry_constraints(pattern)
    columns = np.asarray(pattern.columns)
    valid = np.asarray(valid, dtype=bool)
    targets, injection = np.asarray(targets), np.asarray(injection)
    coarse_indices = coarse_fine_indices(pattern)
    zc = z[coarse_indices]
    vector_weights = _vector_weights(a, z, config)
    weights = np.zeros(columns.shape, dtype=np.float64)
    mask = np.zeros(columns.shape, dtype=bool)
    if fixed_mask is not None:
        fixed_mask = np.asarray(fixed_mask, dtype=bool)
        if fixed_mask.shape != columns.shape or np.any(fixed_mask & ~valid):
            raise ValueError("fixed mask has wrong shape or invalid candidates")
        degrees = np.minimum(config.k, valid.sum(axis=1))
        degrees[injection >= 0] = 1
        if not np.array_equal(fixed_mask.sum(axis=1), degrees):
            raise ValueError("fixed mask must obey the configured row cardinalities")
    rows = []
    for i in range(a.shape[0]):
        if injection[i] >= 0:
            slot = int(injection[i])
            if fixed_mask is not None and not fixed_mask[i, slot]:
                raise ValueError("fixed mask must preserve coarse-node injection")
            mask[i, slot] = True
            weights[i, slot] = 1.0
            rows.append({"objective": config.mu, "fit_error": 0.0, "free_dimension": 0,
                         "free_rank": 0, "rank_deficient": False, "free_condition": 1.0,
                         "free_smallest_singular": None, "effective_mu": config.mu,
                         "l1_active": False, "regularized_condition": 1.0,
                         "solution_status": "injection"})
            continue
        slots = sorted(np.flatnonzero(valid[i]).tolist(), key=lambda s: int(columns[i, s]))
        if not slots:
            raise ValueError(f"fine row {i} has no valid coarse candidates")
        if fixed_mask is not None:
            chosen = np.flatnonzero(fixed_mask[i]).tolist()
        else:
            chosen = []
            for _ in range(min(config.k, len(slots))):
                best = None
                for slot in slots:
                    if slot in chosen:
                        continue
                    candidate = chosen + [slot]
                    _, info = _row_fit(zc[columns[i, candidate]].T, z[i],
                                       vector_weights, targets[i], config)
                    score = (info["objective"], int(columns[i, slot]))
                    if best is None or score < best[0]:
                        best = (score, slot)
                chosen.append(best[1])
        p, info = _row_fit(zc[columns[i, chosen]].T, z[i], vector_weights,
                           targets[i], config)
        mask[i, chosen] = True
        weights[i, chosen] = p
        rows.append(info)
    # Elementwise assembly, including structural zeros; no P != 0 mask rebuild.
    predicted = np.zeros_like(z)
    for slot in range(columns.shape[1]):
        predicted += weights[:, slot, None] * zc[np.maximum(columns[:, slot], 0)]
    diag = {
        "search": "fixed_mask_refit" if fixed_mask is not None else "bounded_greedy_forward",
        "rows": rows, "vector_count": int(z.shape[1]), "vector_weights": vector_weights.tolist(),
        "rank_deficient_rows": sum(r["rank_deficient"] for r in rows),
        "ill_conditioned_rows": sum(r["rank_deficient"] or r["free_condition"] > config.condition_limit
                                     for r in rows),
        "max_free_condition": (None if any(r["rank_deficient"] for r in rows) else
                               max(r["free_condition"] for r in rows)),
        "weight_frobenius_norm": float(np.linalg.norm(weights)),
        "maximum_row_l1": float(np.max(np.abs(weights).sum(axis=1))),
        "equality_defect": float(np.max(np.abs(weights.sum(axis=1) - targets))),
        "fit_error": float(np.sum((z - predicted) ** 2 * vector_weights)),
        "objective": sum(r["objective"] for r in rows),
        "l1_active_rows": sum(r["l1_active"] for r in rows),
        "candidate_columns": columns.tolist(), "row_targets": targets.tolist(),
        "injection_slots": injection.tolist(), "structural_nnz": int(mask.sum()),
    }
    if heldout_vectors is not None:
        heldout = _normalise(heldout_vectors)
        if heldout.shape[0] != a.shape[0]:
            raise ValueError("held-out vectors have wrong size")
        hc = heldout[coarse_indices]
        approximation = np.zeros_like(heldout)
        for slot in range(columns.shape[1]):
            approximation += weights[:, slot, None] * hc[np.maximum(columns[:, slot], 0)]
        diag["heldout_fit_error"] = float(np.mean(np.sum((heldout - approximation) ** 2, axis=0)))
    return weights, mask, diag


def _runtime_prepare(level):
    if level.P is None:
        if level.lu is None:
            level.lu = spla.splu(level.A.tocsc())
    elif level.cache is None:
        level.cache = LineSmootherCache(level.A, level.shape)
        level.cache.prepare("x")
        level.cache.prepare("y")


def teacher_cycle(result_or_levels, x, b, mg_config=None):
    """A raw classical V-cycle over the teacher's actual P/A/factor hierarchy."""
    if isinstance(result_or_levels, TeacherResult):
        levels = result_or_levels.levels
        mg = mg_config or result_or_levels.mg_config
    else:
        levels = result_or_levels
        mg = mg_config or MGConfig(mode="classical")
    if not levels:
        raise ValueError("empty hierarchy")
    def cycle(index, value, rhs):
        level = levels[index]
        _runtime_prepare(level)
        if level.P is None:
            return level.lu.solve(rhs)
        for _ in range(mg.pre_steps):
            value = value + _correction(level.A, level.shape, rhs - level.A @ value,
                                         level.cache, mg)
        residual = rhs - level.A @ value
        coarse_rhs = level.P.T @ residual
        correction = cycle(index + 1, np.zeros_like(coarse_rhs), coarse_rhs)
        value = value + level.P @ correction
        for _ in range(mg.post_steps):
            value = value + _correction(level.A, level.shape, rhs - level.A @ value,
                                         level.cache, mg, reverse=True)
        return value
    x = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != levels[0].A.shape[0] or b.size != x.size:
        raise ValueError("cycle vector shape mismatch")
    value = cycle(0, x, b)
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite raw V-cycle")
    return value


def _mass_orthonormal(vectors, H, tolerance):
    gram = vectors.T @ (H @ vectors)
    values, rotation = la.eigh(0.5 * (gram + gram.T))
    keep = values > max(float(values.max()) * tolerance, np.finfo(float).tiny)
    if not np.any(keep):
        raise FloatingPointError("generalized eigenvectors lost mass rank")
    return (vectors @ rotation[:, keep]) / np.sqrt(values[keep])[None, :]


def _ritz(A, H, vectors, count, tolerance):
    q = _mass_orthonormal(vectors, H, tolerance)
    small = q.T @ (A @ q)
    values, rotation = la.eigh(0.5 * (small + small.T))
    count = min(count, q.shape[1])
    return values[:count], q @ rotation[:, :count]


def _eigen_residual(A, H, vectors, values):
    av, hv = A @ vectors, H @ vectors
    residual = av - hv * values[None, :]
    scale = np.linalg.norm(av, axis=0) + np.abs(values) * np.linalg.norm(hv, axis=0)
    return residual, np.linalg.norm(residual, axis=0) / np.maximum(scale, np.finfo(float).tiny)


def _refine_eigen(A, H, shape, vectors, config, mg, cache=None):
    """Preconditioned residual subspace refinement of A v = lambda H v."""
    values, v = _ritz(A, H, vectors, min(config.eigen_count, vectors.shape[1]),
                      config.rank_tolerance)
    residual, errors = _eigen_residual(A, H, v, values)
    before = errors.copy()
    cache = cache or LineSmootherCache(A, shape)
    accepted = 0
    for _ in range(config.eigen_refine):
        directions = np.column_stack([
            _correction(A, shape, -residual[:, j], cache, mg)
            for j in range(v.shape[1])])
        next_values, candidate = _ritz(A, H, np.column_stack((v, directions)), v.shape[1],
                                       config.rank_tolerance)
        next_residual, next_errors = _eigen_residual(A, H, candidate, next_values)
        # An enlarged Ritz space lowers Ritz energies; accept only if its
        # generalized eigen-residual also improves, rather than assuming so.
        if np.linalg.norm(next_errors) <= np.linalg.norm(errors) * (1 + 1e-10):
            values, v, residual, errors = next_values, candidate, next_residual, next_errors
            accepted += 1
        else:
            break
    return v, {"eigenvalues": values.tolist(), "residual_before": before.tolist(),
               "residual_after": errors.tolist(), "accepted_refinements": accepted,
               "mass_orthogonality_defect": float(np.linalg.norm(v.T @ (H @ v) - np.eye(v.shape[1])))}


def generalized_enrichment(levels, *, config=None, mg_config=None):
    """Solve the terminal generalized eigenproblem and prolong/refine upward."""
    config = config or TeacherConfig()
    mg = mg_config or MGConfig(mode="classical")
    if config.eigen_count == 0:
        return {}, {"enabled": False, "reason": "eigen_count=0: adaptive LS variant"}
    last = levels[-1]
    count = min(config.eigen_count, last.A.shape[0])
    # This is the only operator-to-dense conversion: the tiny terminal grid.
    values, v = la.eigh(last.A.toarray(), last.H.toarray(), subset_by_index=(0, count - 1))
    if values[0] <= 0:
        raise ValueError("terminal generalized eigenproblem is not SPD")
    enriched = {len(levels) - 1: v}
    reports = {str(len(levels) - 1): {"eigenvalues": values.tolist(),
        "mass_orthogonality_defect": float(np.linalg.norm(v.T @ (last.H @ v) - np.eye(count)))}}
    for index in reversed(range(len(levels) - 1)):
        level = levels[index]
        v = level.P @ v
        v, report = _refine_eigen(level.A, level.H, level.shape, v, config, mg, level.cache)
        enriched[index] = v
        reports[str(index)] = report
    return enriched, {"enabled": True, "kind": "generalized_multilevel_eigenvectors",
                       "count": count, "levels": reports}


def _galerkin(operator, P):
    value = (P.T @ operator @ P).tocsr()
    value.sum_duplicates()
    # Remove roundoff asymmetry, never sparsify actual off-stencil couplings.
    value = (0.5 * (value + value.T)).tocsr()
    value.sort_indices()
    return value


def _build_pass(A, shape, config, mg, pass_index, enrichment, adaptive):
    from .bamg_transfer import direct_pattern, scipy_from_weights

    levels = []
    current, H, current_shape = A, sp.eye(A.shape[0], format="csr"), shape
    index = 0
    while True:
        level = TeacherLevel(current, current_shape, H)
        levels.append(level)
        level.diagnostics = {"level": index, "A_digest": matrix_digest(current),
                             "H_digest": matrix_digest(H), "A_nnz": current.nnz,
                             "H_nnz": H.nnz, "setup_pass": pass_index}
        if terminal(current_shape, mg.coarsest_n):
            _runtime_prepare(level)
            # Coarsest SPD check remains small; no dense fine-grid conversion.
            la.cholesky(current.toarray(), lower=True)
            return levels
        coarse_shape = next_shape(current_shape, "full", mg.coarsest_n, level_index=index)
        pattern = direct_pattern(current_shape, coarse_shape)
        level.pattern = pattern
        level.cache = LineSmootherCache(current, current_shape)
        fit_seed = config.seed + 1009 * index
        z, vector_report = relaxed_vectors(current, current_shape, config.vectors, config.sweeps,
                                           seed=fit_seed, mg_config=mg, cache=level.cache)
        pieces = [z]
        eigen_report = None
        if index in enrichment:
            # Coarse A/H may change during rebuilding.  Use the old mode only as
            # an initial vector and refine against the newly built operators.
            eigen, eigen_report = _refine_eigen(current, H, current_shape, enrichment[index],
                                                config, mg, level.cache)
            pieces.append(_normalise(eigen))
        if index in adaptive:
            hard = adaptive[index].copy()
            for j in range(hard.shape[1]):
                hard[:, j] += _correction(current, current_shape, -(current @ hard[:, j]),
                                           level.cache, mg)
            pieces.append(_normalise(hard))
        z = np.column_stack(pieces)
        audit_seed = config.seed + 2_000_003 + 1009 * index
        heldout, audit_report = relaxed_vectors(current, current_shape, config.audit_vectors,
                                                config.sweeps, seed=audit_seed,
                                                mg_config=mg, cache=level.cache)
        retries = 0
        while True:
            weights, mask, fit_report = local_fit(current, pattern, z, config=config,
                                                  heldout_vectors=heldout)
            if (not fit_report["ill_conditioned_rows"] or retries >= config.max_vector_retries
                    or config.augment_vectors == 0):
                break
            retries += 1
            extra, _ = relaxed_vectors(current, current_shape, config.augment_vectors,
                                        config.sweeps, seed=fit_seed + 10_007 * retries,
                                        mg_config=mg, cache=level.cache)
            z = np.column_stack((z, extra))
        P = scipy_from_weights(pattern, weights, mask)
        level.weights, level.mask, level.P, level.vectors = weights, mask, P, z
        level.diagnostics.update({"fit": fit_report, "initial_vectors": vector_report,
                                  "audit_vectors": audit_report, "vector_retries": retries,
                                  "unresolved_free_rank_rows": fit_report["rank_deficient_rows"],
                                  "unresolved_free_condition_rows": fit_report["ill_conditioned_rows"],
                                  "recovery_policy": "add independent vectors and reselect support; "
                                      "retain and report any residual deficiency with SVD minimum-norm/ridge and L1 cap",
                                  "actual_vector_count": z.shape[1],
                                  "retained_initial_count": pieces[0].shape[1],
                                  "eigen_count": pieces[1].shape[1] if index in enrichment else 0,
                                  "adaptive_count": adaptive[index].shape[1] if index in adaptive else 0,
                                  "refinement_on_rebuilt_operator": eigen_report})
        current, H = _galerkin(current, P), _galerkin(H, P)
        complexity = (sum(l.A.nnz for l in levels) + current.nnz) / A.nnz
        if complexity > config.max_operator_complexity or H.nnz > config.max_mass_nnz:
            raise MemoryError("teacher hierarchy exceeded configured sparse complexity budget")
        current_shape = coarse_shape
        index += 1


def _probe_contraction(levels, count, seed, cycles, mg):
    rng = np.random.default_rng(seed)
    trajectories = []
    for _ in range(count):
        rhs = rng.standard_normal(levels[0].A.shape[0])
        rhs /= np.linalg.norm(rhs)
        x = np.zeros_like(rhs)
        norms = [1.0]
        for _ in range(cycles):
            x = teacher_cycle(levels, x, rhs, mg)
            norm = float(np.linalg.norm(rhs - levels[0].A @ x))
            if not math.isfinite(norm) or norm > mg.divergence_factor:
                raise FloatingPointError("teacher raw V-cycle diverged on a retained probe")
            norms.append(norm)
        trajectories.append(norms)
    final = np.asarray(trajectories)[:, -1]
    return {"seed": seed, "count": count, "cycles": cycles,
            "geometric_mean_ratio": float(np.exp(np.log(np.maximum(final, 1e-300)).mean())),
            "worst_ratio": float(final.max()), "residual_trajectories": trajectories}


def build_teacher(A, shape, *, config=None, mg_config=None):
    """Build/select a teacher using TRAIN-only probes and retain failed passes.

    The independent audit is reported but never used to select a pass.  All
    candidates are assessed with the same selection seeds and raw V-cycles;
    bootstrap therefore need not improve the first hierarchy to be valid.
    """
    started = time.perf_counter()
    config = config or TeacherConfig()
    mg = mg_config or MGConfig(mode="classical")
    strategy = get_strategy(mg.strategy_name)
    if strategy.smoother != "line_alt" or strategy.coarsening != "full":
        raise ValueError("BAMG teacher requires classical line_alt and full coarsening")
    shape = validate_root_shape(shape, mg.coarsest_n)
    a = _checked_matrix(A, shape)
    enrichment, adaptive = {}, {}
    reports = []
    best_levels, best_score, best_pass = None, math.inf, None
    selection_seed, audit_seed = config.seed + 1_000_003, config.seed + 2_000_003
    for pass_index in range(config.setup_passes):
        pass_started = time.perf_counter()
        report = {"pass": pass_index, "initial_construction": pass_index == 0}
        try:
            levels = _build_pass(a, shape, config, mg, pass_index, enrichment, adaptive)
            selection = _probe_contraction(levels, config.selection_vectors, selection_seed,
                                            config.selection_cycles, mg)
            try:
                audit = _probe_contraction(levels, config.audit_vectors, audit_seed,
                                            config.selection_cycles, mg)
            except (ValueError, RuntimeError, FloatingPointError, la.LinAlgError) as error:
                # Independent audit failures are retained, never used as an
                # additional pass-selection criterion or silently removed.
                audit = {"seed": audit_seed, "count": config.audit_vectors,
                         "failed": True, "failure": f"{type(error).__name__}: {error}"}
            report.update({"valid": True, "selection": selection, "audit": audit,
                           "operator_complexity": sum(l.A.nnz for l in levels) / a.nnz,
                           "A_digests": [l.diagnostics["A_digest"] for l in levels],
                           "level_fit_diagnostics": [l.diagnostics for l in levels],
                           "structural_P_nnz": [int(l.mask.sum()) for l in levels[:-1]]})
            score = selection["geometric_mean_ratio"]
            if best_levels is None or score < best_score:
                best_levels, best_score, best_pass = levels, score, pass_index
            if pass_index + 1 < config.setup_passes:
                enrichment, eigen_report = generalized_enrichment(levels, config=config, mg_config=mg)
                report["enrichment"] = eigen_report
                adaptive = {}
                if config.adaptive_vectors:
                    for index, level in enumerate(levels[:-1]):
                        z, _ = relaxed_vectors(level.A, level.shape, config.adaptive_vectors,
                                                config.sweeps, seed=config.seed + 3_000_007 +
                                                pass_index * 10_007 + index * 1009,
                                                mg_config=mg, cache=level.cache)
                        for j in range(z.shape[1]):
                            for _ in range(config.adaptive_cycles):
                                z[:, j] = teacher_cycle(levels[index:], z[:, j],
                                                        np.zeros(level.A.shape[0]), mg)
                        adaptive[index] = _normalise(z)
                report["adaptive_enrichment_counts"] = {
                    str(index): z.shape[1] for index, z in adaptive.items()}
        except (ValueError, RuntimeError, FloatingPointError, la.LinAlgError, MemoryError) as error:
            # A failed round remains visible, even when an earlier valid round
            # can be returned.  Never silently remove a failed row or case.
            if report.get("valid"):
                report["enrichment_failure"] = f"{type(error).__name__}: {error}"
            else:
                report.update({"valid": False, "failure": f"{type(error).__name__}: {error}"})
            if best_levels is None:
                raise RuntimeError(f"BAMG teacher initial construction failed: {error}") from error
            enrichment, adaptive = {}, {}
        report["setup_seconds"] = time.perf_counter() - pass_started
        reports.append(report)
    elapsed = time.perf_counter() - started
    used_eigen = any(any(d.get("eigen_count", 0) for d in r.get("level_fit_diagnostics", []))
                     for r in reports if r.get("valid"))
    method = ("one_pass_relaxed_vector_ls" if config.setup_passes == 1 else
              "fixed_coarsening_bootstrap_amg" if used_eigen else "adaptive_ls_variant")
    diagnostics = {
        "method": method, "full_paper_reproduction": False,
        "source_version": 1, "A_digest": matrix_digest(a),
        "config": config.to_dict(), "mg_config": mg.to_dict(),
        "setup_passes_requested": config.setup_passes, "setup_passes_attempted": len(reports),
        "selected_pass": best_pass, "selection_score": best_score,
        "selection_seed": selection_seed, "audit_seed": audit_seed,
        "seed_roles": "fit, selection, and audit are distinct TRAIN-only probes",
        "passes": reports, "failed_passes": sum(not r["valid"] for r in reports),
        "enrichment_failures": sum("enrichment_failure" in r for r in reports),
        "audit_failures": sum(r.get("audit", {}).get("failed", False) for r in reports),
        "failure_rate": sum(not r["valid"] for r in reports) / len(reports),
        "setup_seconds": elapsed,
    }
    return TeacherResult(best_levels, diagnostics, elapsed, config, mg)
