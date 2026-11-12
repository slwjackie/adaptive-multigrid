"""Sparse, fixed-support BAMG + energy teachers for algebraic coarsening.

The reference is the *selected AMG interpolation*, not geometric bilinear P.
Vector banks, row targets, and reference normalizers are immutable during one
optimization. The energy uses full A, including fill outside any local stencil.
The caller supplies an SPD problem; symmetry and positive diagonals alone are
validated here and are not advertised as an SPD certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def _matrix(A):
    if not sp.issparse(A) or A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("teacher requires a nonempty square sparse A")
    A = sp.csr_matrix(A, dtype=np.float64, copy=True)
    A.sum_duplicates()
    A.sort_indices()
    if not np.isfinite(A.data).all() or np.any(A.diagonal() <= 0):
        raise ValueError("teacher requires finite A with positive diagonal")
    if sla.norm(A - A.T) > 1e-11 * max(sla.norm(A), np.finfo(float).tiny):
        raise ValueError("teacher requires symmetric A")
    return A


def jacobi_omega(A):
    """A conservative weighted-Jacobi step: 0.8 / ||D^-1 A||_infinity."""
    A = _matrix(A)
    bound = np.max(np.asarray(abs(A).sum(axis=1)).ravel() / A.diagonal())
    return float(0.8 / bound)


def generate_test_vectors(A, count=8, sweeps=8, seed=0, omega=None,
                          bootstrap_cycles=0, bootstrap_max_levels=20,
                          bootstrap_first_cycle_max_growth=10., return_report=False):
    """Seeded homogeneous-error probes, mixing relaxed and random columns.

    All but the first column receive ``sweeps`` weighted-Jacobi steps; the first
    retains a random probe when count > 1. No exact solutions or heldout RHS are
    used. Optional bootstrap cycles refine the relaxed columns with a
    provisional *classical* AMG hierarchy and the same conservative Jacobi
    smoother as the learned solver. This hierarchy never calls the teacher.
    ``bootstrap_max_levels`` is the provisional hierarchy's depth safety cap;
    study callers pass their configured hierarchy limit and first-cycle growth
    threshold explicitly, including for this provisional hierarchy.
    If a numerical guard replaces the root by a direct solve, the homogeneous
    error becomes zero. Preserve the preceding relaxed bank and report the
    incomplete bootstrap instead of inventing replacement vectors.
    Normalization is Euclidean and each column must remain nonzero.
    ``return_report`` returns (bank, diagnostics) for teacher provenance.
    """
    A = _matrix(A)
    if type(count) is not int or count < 1 or type(sweeps) is not int or sweeps < 0:
        raise ValueError("positive integer count and nonnegative sweeps required")
    if type(bootstrap_cycles) is not int or bootstrap_cycles < 0:
        raise ValueError("bootstrap_cycles must be a nonnegative integer")
    if type(bootstrap_max_levels) is not int or bootstrap_max_levels < 1:
        raise ValueError("bootstrap_max_levels must be a positive integer")
    if (isinstance(bootstrap_first_cycle_max_growth, bool)
            or not math.isfinite(bootstrap_first_cycle_max_growth)
            or bootstrap_first_cycle_max_growth < 1):
        raise ValueError("bootstrap_first_cycle_max_growth must be finite and at least 1")
    step = jacobi_omega(A) if omega is None else float(omega)
    bound = np.max(np.asarray(abs(A).sum(axis=1)).ravel() / A.diagonal())
    if not math.isfinite(step) or not 0 < step < 2 / bound:
        raise ValueError("omega must satisfy the conservative Jacobi stability bound")
    Z = np.random.default_rng(seed).normal(size=(A.shape[0], count))
    start = 1 if count > 1 else 0
    for _ in range(sweeps):
        Z[:, start:] -= step * (A @ Z[:, start:]) / A.diagonal()[:, None]
    diagnostics = dict(bootstrap_completed_cycles=0, bootstrap_incomplete=False,
                       bootstrap_stop_reason=None, bootstrap_guard_events=[])
    if bootstrap_cycles:
        if A.shape[0] < 2:
            raise ValueError("bootstrap requires at least two unknowns")
        from .solver import build_hierarchy, _runtime_cycle
        hierarchy = build_hierarchy(A, method="classical", max_levels=bootstrap_max_levels,
                                    max_coarse=min(9, A.shape[0] - 1),
                                    first_cycle_max_growth=bootstrap_first_cycle_max_growth)
        for _ in range(bootstrap_cycles):
            if hierarchy.levels[0].p is None:
                diagnostics['bootstrap_stop_reason'] = 'root_direct_at_setup'
                break
            probes = _runtime_cycle(hierarchy, np.zeros_like(Z[:, start:]), Z[:, start:])
            if hierarchy.levels[0].p is None:
                diagnostics['bootstrap_stop_reason'] = 'root_direct_during_cycle'
                break
            probe_norm = np.linalg.norm(probes, axis=0)
            if not np.isfinite(probes).all() or np.any(probe_norm <= np.finfo(float).tiny):
                raise FloatingPointError("degenerate bootstrap test vectors")
            Z[:, start:] = probes / probe_norm
            diagnostics['bootstrap_completed_cycles'] += 1
        diagnostics['bootstrap_incomplete'] = diagnostics['bootstrap_completed_cycles'] < bootstrap_cycles
        diagnostics['bootstrap_guard_events'] = list(hierarchy.guard_events)
    norm = np.linalg.norm(Z, axis=0)
    if not np.isfinite(Z).all() or np.any(norm <= np.finfo(float).tiny):
        raise FloatingPointError("degenerate relaxed test vectors")
    bank = Z / norm
    return (bank, diagnostics) if return_report else bank


def _digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode() + a.tobytes()).hexdigest()


@dataclass(frozen=True)
class Normalizers:
    bamg: float
    energy: float
    reference: str = "classical_amg"


@dataclass
class TeacherResult:
    p: sp.csr_matrix
    vectors: np.ndarray
    coarse_vectors: np.ndarray
    weights: np.ndarray
    report: dict


def _tangent(base, coarse):
    """Sparse orthonormal Helmert bases for all non-injection row sums."""
    fixed = set(map(int, coarse))
    rr, cc, vv, column = [], [], [], 0
    for row in range(base.shape[0]):
        if row in fixed:
            continue
        start, end = base.indptr[row:row + 2]
        for j in range(1, end - start):
            rr.extend(range(start, start + j + 1))
            cc.extend([column] * (j + 1))
            values = np.ones(j + 1) / np.sqrt(j * (j + 1))
            values[-1] *= -j
            vv.extend(values)
            column += 1
    return sp.csr_matrix((vv, (rr, cc)), shape=(base.nnz, column))


class FixedObjective:
    """BAMG + energy quadratic with a convex soft cancellation penalty."""
    def __init__(self, A, structure, vectors, mu=1e-6, l1_cap=None,
                 cancellation_weight=0.1, weighting="uniform", energy_weight_cap=100.0):
        self.A = _matrix(A)
        self.base = sp.csr_matrix(structure.reference, dtype=np.float64, copy=True)
        self.base.sum_duplicates()
        self.base.sort_indices()
        self.coarse = np.asarray(structure.coarse, dtype=np.int64).copy()
        n, nc = self.base.shape
        if n != self.A.shape[0] or len(self.coarse) != nc or nc == 0 or nc >= n:
            raise ValueError("invalid reference/coarse dimensions")
        if len(np.unique(self.coarse)) != nc or np.any(self.coarse < 0) or np.any(self.coarse >= n):
            raise ValueError("invalid coarse node indices")
        if not np.isfinite(self.base.data).all() or np.any(np.diff(self.base.indptr) == 0):
            raise ValueError("finite reference and nonempty support rows required")
        if sla.norm(self.base[self.coarse, :] - sp.eye(nc, format="csr")) > 1e-12:
            raise ValueError("reference must preserve coarse injection")
        self.targets = np.asarray(self.base.sum(axis=1)).ravel()
        if hasattr(structure, "row_targets") and not np.allclose(
                self.targets, structure.row_targets, rtol=1e-12, atol=1e-12):
            raise ValueError("structure row targets disagree with reference")
        self.mu = float(mu)
        if not math.isfinite(self.mu) or self.mu < 0:
            raise ValueError("mu must be finite and nonnegative")
        if l1_cap is not None:
            raise ValueError("l1_cap has been removed; use None and cancellation_weight")
        self.cancellation_weight = float(cancellation_weight)
        if not math.isfinite(self.cancellation_weight) or self.cancellation_weight < 0:
            raise ValueError("cancellation_weight must be finite and nonnegative")
        reference_l1 = np.asarray(abs(self.base).sum(axis=1)).ravel()
        self.reference_maximum_row_l1 = float(np.max(reference_l1))
        self.fine = np.ones(n, dtype=bool)
        self.fine[self.coarse] = False
        self.fine_count = int(self.fine.sum())
        self.cancellation_scale = np.zeros(n)
        self.cancellation_scale[self.fine] = (
            1 / (self.fine_count * (1 + np.abs(self.targets[self.fine])) ** 2))
        # Rows of equal degree share a vectorized proximal calculation.
        self.prox_groups = []
        degrees = np.diff(self.base.indptr)
        for degree in np.unique(degrees[self.fine]):
            selected = np.flatnonzero(self.fine & (degrees == degree))
            positions = self.base.indptr[selected, None] + np.arange(degree)
            self.prox_groups.append((selected, positions))
        self.z = np.array(vectors, dtype=np.float64, copy=True)
        if self.z.ndim != 2 or self.z.shape[0] != n or self.z.shape[1] == 0:
            raise ValueError("vectors must have shape (N, positive count)")
        norms = np.linalg.norm(self.z, axis=0)
        if not np.isfinite(self.z).all() or np.any(norms <= np.finfo(float).tiny):
            raise ValueError("vectors must be finite and nonzero")
        self.z /= norms
        self.zc = self.z[self.coarse].copy()
        if weighting == "uniform":
            self.w = np.full(self.z.shape[1], 1 / self.z.shape[1])
        elif weighting == "energy":
            energy = np.einsum("ij,ij->j", self.z, self.A @ self.z)
            if np.any(energy <= 0) or not np.isfinite(energy).all():
                raise ValueError("positive vector energies required")
            if not math.isfinite(energy_weight_cap) or energy_weight_cap < 1:
                raise ValueError("energy_weight_cap must be >= 1")
            self.w = 1 / energy
            self.w = np.minimum(self.w, self.w.min() * energy_weight_cap)
            self.w /= self.w.sum()
        else:
            raise ValueError("weighting must be uniform or energy")
        self.rows = np.repeat(np.arange(n), np.diff(self.base.indptr))
        self.cols = self.base.indices.copy()
        self.fixed_positions = np.flatnonzero(np.isin(self.rows, self.coarse))
        self.tangent = _tangent(self.base, self.coarse)
        self.rhs_fit = np.zeros(self.base.nnz)
        hr, hc, hv = [], [], []
        for row in range(n):
            start, end = self.base.indptr[row:row + 2]
            pos = np.arange(start, end)
            features = self.zc[self.cols[pos]]
            block = (features * self.w) @ features.T + self.mu * np.eye(len(pos))
            self.rhs_fit[pos] = features @ (self.w * self.z[row])
            hr.extend(np.repeat(pos, len(pos)))
            hc.extend(np.tile(pos, len(pos)))
            hv.extend(block.ravel())
        self.fit_matrix = sp.csr_matrix((hv, (hr, hc)), shape=(self.base.nnz,) * 2)
        terms = self.raw_terms(self.base.data)
        if any(not math.isfinite(terms[k]) or terms[k] <= 0 for k in ("bamg", "energy")):
            raise ValueError("reference normalizers must be positive and finite")
        self.scales = Normalizers(terms["bamg"], terms["energy"])
        for array in (self.z, self.zc, self.w, self.targets, self.coarse):
            array.setflags(write=False)

    def matrix(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.base.data.shape:
            raise ValueError("values shape does not match support")
        return sp.csr_matrix((values, self.base.indices.copy(), self.base.indptr.copy()),
                             shape=self.base.shape)

    def gather(self, matrix):
        return np.asarray(matrix.tocsr()[self.rows, self.cols]).ravel()

    def raw_terms(self, values):
        p = self.matrix(values)
        error = self.z - p @ self.zc
        fit = float(np.sum(error ** 2 * self.w))
        ridge = float(self.mu * np.dot(values, values))
        energy = float(p.multiply(self.A @ p).sum())
        return dict(fit=fit, ridge=ridge, bamg=fit + ridge, energy=energy,
                    cancellation=self.cancellation(values))

    def coefficients(self, strength):
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and nonnegative")
        return 1 / self.scales.bamg, strength / self.scales.energy

    def evaluate(self, values, strength=10.0):
        b, e = self.coefficients(strength)
        terms = self.raw_terms(values)
        return (b * terms["bamg"] + e * terms["energy"]
                + self.cancellation_weight * terms["cancellation"])

    def half_hessian(self, values, strength=10.0):
        b, e = self.coefficients(strength)
        return b * (self.fit_matrix @ values) + e * self.gather(self.A @ self.matrix(values))

    def quadratic_gradient(self, values, strength=10.0):
        b, _ = self.coefficients(strength)
        return 2 * (self.half_hessian(values, strength) - b * self.rhs_fit)

    def cancellation(self, values):
        row_l1 = np.bincount(self.rows, weights=np.abs(values), minlength=len(self.targets))
        excess = np.maximum(row_l1 - np.abs(self.targets), 0.)
        return float(np.dot(self.cancellation_scale, excess ** 2))

    def gradient(self, values, strength=10.0):
        """A subgradient, equal to the gradient away from sign changes."""
        row_l1 = np.bincount(self.rows, weights=np.abs(values), minlength=len(self.targets))
        excess = np.maximum(row_l1 - np.abs(self.targets), 0.)
        penalty = 2 * self.cancellation_weight * self.cancellation_scale * excess
        return self.quadratic_gradient(values, strength) + penalty[self.rows] * np.sign(values)

    def prox_cancellation(self, values, rho):
        """Exact prox of eta * max(||row||1 - |s|, 0)^2 / (NF*(1+|s|)^2).

        The positive part extends the requested penalty convexly off the affine
        row-sum constraint. It is identical on every feasible interpolation.
        Sorting gives the active soft-threshold set without iterative row solves.
        """
        result = np.asarray(values, dtype=float).copy()
        for selected, positions in self.prox_groups:
            v = values[positions]
            ordered = np.sort(np.abs(v), axis=1)[:, ::-1]
            beta = 2 * self.cancellation_weight * self.cancellation_scale[selected] / rho
            thresholds = (beta[:, None] *
                          (np.cumsum(ordered, axis=1) - np.abs(self.targets[selected, None])) /
                          (1 + beta[:, None] * np.arange(1, v.shape[1] + 1)))
            active = (ordered > thresholds).sum(axis=1)
            threshold = thresholds[np.arange(len(selected)), np.maximum(active - 1, 0)]
            threshold = np.maximum(threshold, 0.)
            result[positions] = np.sign(v) * np.maximum(np.abs(v) - threshold[:, None], 0.)
        return result

    def project(self, values):
        result = np.asarray(values, dtype=float).copy()
        for row in range(self.base.shape[0]):
            start, end = self.base.indptr[row:row + 2]
            v = result[start:end]
            target = self.targets[row]
            v += (target - v.sum()) / len(v)
        result[self.fixed_positions] = self.base.data[self.fixed_positions]
        return result

    def validate(self, values):
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite teacher interpolation")
        row_sum = np.bincount(self.rows, weights=values, minlength=self.base.shape[0])
        row_l1 = np.bincount(self.rows, weights=np.abs(values), minlength=self.base.shape[0])
        equality = float(np.max(np.abs(row_sum - self.targets)))
        injection = float(np.max(np.abs(values[self.fixed_positions] - self.base.data[self.fixed_positions])))
        if equality > 1e-9 * max(1., float(np.max(np.abs(self.targets)))) or injection > 1e-12:
            raise ValueError("teacher violates row sums or injection")
        return dict(equality_defect=equality, injection_defect=injection,
                    maximum_row_l1=float(row_l1.max()), structural_nnz=self.base.nnz)


def minimize(objective, strength=10.0, maxiter=200, rtol=1e-8):
    """Sparse quadratic CG followed by ADMM for the exact cancellation penalty.

    The P update stays in the row-sum/injection affine space. The auxiliary
    update is an exact proximal map. Returned P is always feasible, and a finite
    budget is reported honestly through primal/dual residuals and convergence.
    """
    if type(maxiter) is not int or maxiter < 1 or not math.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("invalid teacher optimizer controls")
    o = objective
    b, e = o.coefficients(strength)
    reference = o.base.data
    base = reference.copy()
    o.validate(base)
    before = o.evaluate(base, strength)
    values = base.copy()
    iterations = admm_iterations = 0
    info = 0
    solver = "tangent_cg"
    converged = True
    primal = dual = 0.
    if o.tangent.shape[1]:
        tangent = o.tangent
        rhs = np.asarray(tangent.T @ (b * o.rhs_fit - o.half_hessian(base, strength))).ravel()
        diagonal = b * o.fit_matrix.diagonal() + e * o.A.diagonal()[o.rows]
        prediag = np.asarray(tangent.power(2).T @ diagonal).ravel()
        if np.any(prediag <= 0):
            raise ValueError("teacher Hessian must have positive tangent diagonal")

        def solve_quadratic(right, rho=0., initial=None):
            def action(q):
                return np.asarray(tangent.T @ o.half_hessian(tangent @ q, strength)).ravel() + rho * q / 2
            operator = sla.LinearOperator((tangent.shape[1],) * 2, matvec=action, dtype=float)
            preconditioner = sla.LinearOperator(operator.shape, matvec=lambda q: q / (prediag + rho / 2))
            def count(_):
                nonlocal iterations
                iterations += 1
            q, status = sla.cg(operator, right, x0=initial, M=preconditioner,
                               maxiter=maxiter, rtol=rtol * .1, atol=0., callback=count)
            if status < 0 or not np.isfinite(q).all():
                raise FloatingPointError("teacher CG failed")
            return q, status

        q, info = solve_quadratic(rhs)
        candidate = base + tangent @ q
        if o.evaluate(candidate, strength) < before:
            values = candidate.copy()
        converged = info == 0
        row_minimum = np.minimum.reduceat(candidate, o.base.indptr[:-1])
        row_maximum = np.maximum.reduceat(candidate, o.base.indptr[:-1])
        mixed_signs = (row_minimum < 0) & (row_maximum > 0)
        if o.cancellation_weight and np.any(mixed_signs & o.fine):
            solver = "tangent_cg_then_cancellation_admm"
            rho = max(float(np.median(2 * prediag)), np.finfo(float).tiny)
            auxiliary = candidate.copy()
            multiplier = np.zeros_like(candidate)
            best = o.evaluate(values, strength)
            converged = False
            for admm_iterations in range(1, maxiter + 1):
                right = rhs + rho / 2 * np.asarray(tangent.T @ (auxiliary - multiplier - base)).ravel()
                q, inner_info = solve_quadratic(right, rho, q)
                info = max(info, inner_info)
                candidate = base + tangent @ q
                previous = auxiliary
                auxiliary = o.prox_cancellation(candidate + multiplier, rho)
                multiplier += candidate - auxiliary
                primal = float(np.linalg.norm(candidate - auxiliary))
                dual = float(rho * np.linalg.norm(auxiliary - previous))
                value = o.evaluate(candidate, strength)
                if math.isfinite(value) and value <= best:
                    values, best = candidate.copy(), value
                primal_limit = rtol * max(1., np.linalg.norm(candidate), np.linalg.norm(auxiliary))
                dual_limit = rtol * max(1., rho * np.linalg.norm(multiplier))
                if inner_info == 0 and primal <= primal_limit and dual <= dual_limit:
                    values = candidate
                    converged = True
                    break
                # Residual balancing changes only the numerical ADMM parameter.
                if primal > 10 * dual:
                    rho *= 2
                    multiplier /= 2
                elif dual > 10 * primal:
                    rho /= 2
                    multiplier *= 2
    constraints = o.validate(values)
    after = o.evaluate(values, strength)
    if not math.isfinite(after) or after > before + 1e-9 * max(1.0, abs(before)):
        raise FloatingPointError("teacher optimization increased initial objective")
    report = dict(solver=solver, strength=float(strength), optimizer_converged=bool(converged),
                  cg_info=int(info), cg_iterations=iterations, admm_iterations=admm_iterations,
                  admm_primal_residual=primal, admm_dual_residual=dual,
                  cancellation_weight=o.cancellation_weight,
                  cancellation_before=o.cancellation(base), cancellation_after=o.cancellation(values),
                  maxiter_per_solver=maxiter, rtol=rtol, objective_before=before,
                  objective_after=after, terms=o.raw_terms(values), s_B=o.scales.bamg,
                  s_E=o.scales.energy, normalizers_fixed=True, reference_policy="classical_amg",
                  reference_digest=_digest(reference), initial_digest=_digest(base),
                  projected_initialization=False,
                  reference_maximum_row_l1=o.reference_maximum_row_l1,
                  reference_objective=before, vectors_digest=_digest(o.z),
                  vector_weights_digest=_digest(o.w), **constraints)
    return o.matrix(values), report


def fit_teacher(A, structure, strength=10.0, *, vectors=None, count=8, sweeps=8,
                seed=0, omega=None, mu=1e-6, l1_cap=None, cancellation_weight=0.1,
                maxiter=200, rtol=1e-8,
                weighting="uniform", energy_weight_cap=100.0, bootstrap_cycles=0,
                bootstrap_max_levels=20, bootstrap_first_cycle_max_growth=10.):
    supplied_bank = vectors is not None
    bootstrap_report = dict(bootstrap_completed_cycles=None, bootstrap_incomplete=None,
                            bootstrap_stop_reason=None, bootstrap_guard_events=[])
    if vectors is None:
        vectors, bootstrap_report = generate_test_vectors(A, count=count, sweeps=sweeps, seed=seed,
                                        omega=omega, bootstrap_cycles=bootstrap_cycles,
                                        bootstrap_max_levels=bootstrap_max_levels,
                                        bootstrap_first_cycle_max_growth=bootstrap_first_cycle_max_growth,
                                        return_report=True)
    objective = FixedObjective(A, structure, vectors, mu=mu, l1_cap=l1_cap,
                               cancellation_weight=cancellation_weight, weighting=weighting,
                               energy_weight_cap=energy_weight_cap)
    p, report = minimize(objective, strength=strength, maxiter=maxiter, rtol=rtol)
    report.update(bootstrap_report)
    report.update(bank_method=("provided_frozen" if supplied_bank else
                               "classical_amg_bootstrap" if bootstrap_cycles else "random_jacobi"),
                  bootstrap_cycles=None if supplied_bank else bootstrap_cycles,
                  bootstrap_max_levels=None if supplied_bank or not bootstrap_cycles else bootstrap_max_levels,
                  bootstrap_first_cycle_max_growth=(None if supplied_bank or not bootstrap_cycles
                                                     else bootstrap_first_cycle_max_growth),
                  vector_count=int(objective.z.shape[1]))
    return TeacherResult(p, objective.z, objective.zc, objective.w, report)
