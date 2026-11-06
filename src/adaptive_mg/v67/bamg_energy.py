"""Fixed-support BAMG + energy objectives, independent of CNN training.

The reference is always geometric bilinear interpolation. Normalizers are
computed once from that reference, actual A, and a frozen vector bank; they are
never recomputed at an optimizer iterate. Full sparse AP couples different rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
import hashlib
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from ..energymin import row_sum_basis
from ..transfer import build_transfer_pattern, coarse_fine_indices
from .bamg_teacher import _checked_matrix, _normalise, _vector_weights, local_fit
from .bamg_transfer import fixed_bilinear_mask, scipy_from_weights


def array_digest(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str((a.shape, str(a.dtype))).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Normalizers:
    bamg: float
    energy: float
    reference: str = 'bilinear'

    def __post_init__(self):
        if self.reference != 'bilinear' or any(
                not math.isfinite(x) or x <= 0 for x in (self.bamg, self.energy)):
            raise ValueError('Fixed bilinear normalizers must be finite and positive')


def _simplex(values, mass):
    """Euclidean row projection onto nonnegative vectors with prescribed sum."""
    if mass == 0:
        return np.zeros_like(values)
    ordered = np.sort(values, axis=1)[:, ::-1]
    means = (np.cumsum(ordered, axis=1) - mass) / np.arange(1, values.shape[1] + 1)
    rho = (ordered > means).sum(axis=1) - 1
    threshold = means[np.arange(len(values)), rho]
    return np.maximum(values - threshold[:, None], 0.)


def project_rows(values, groups, targets, cap, fixed_positions, fixed_values):
    """Exact Euclidean projection onto row sums, injection and signed L1 balls.

    On an active L1 face the positive and negative masses are (cap +/- sum)/2.
    Their two simplex projections have disjoint support. Zero-valued entries
    remain in the declared sparse pattern.
    """
    result = np.array(values, dtype=np.float64, copy=True)
    for positions, rows in groups:
        target = targets[rows]
        v = result[positions]
        v += ((target - v.sum(axis=1)) / v.shape[1])[:, None]
        active = np.abs(v).sum(axis=1) > cap
        for t in np.unique(target[active]):
            use = active & (target == t)
            v[use] = (_simplex(v[use], (cap + t) / 2)
                      - _simplex(-v[use], (cap - t) / 2))
        result[positions] = v
    result[fixed_positions] = fixed_values
    return result


class FixedObjective:
    """Convex quadratic on an explicitly fixed geometric support."""
    def __init__(self, A, pattern, vectors, teacher_config):
        self.A = _checked_matrix(A, pattern.fine_shape)
        self.pattern = pattern
        self.config = teacher_config
        if teacher_config.support_mode != 'fixed_bilinear':
            raise ValueError('Controlled comparison requires fixed_bilinear support')
        self.z = _normalise(vectors).copy()
        self.z.setflags(write=False)
        self.w = _vector_weights(self.A, self.z, teacher_config)
        self.w.setflags(write=False)
        self.zc = self.z[coarse_fine_indices(pattern)]
        self.mask = fixed_bilinear_mask(pattern).cpu().numpy()
        geometric = build_transfer_pattern(pattern.fine_shape, pattern.coarse_shape)
        if not np.array_equal(geometric.columns, pattern.columns):
            raise ValueError('Bilinear/direct candidate indexing mismatch')
        self.base = scipy_from_weights(pattern, geometric.bilinear_weights.copy(), self.mask)
        self.rows = np.repeat(np.arange(self.base.shape[0]), np.diff(self.base.indptr))
        self.cols = self.base.indices.copy()
        self.targets = np.asarray(self.base.sum(axis=1)).ravel()
        if teacher_config.l1_cap < np.max(np.abs(self.targets)):
            raise ValueError('Infeasible row L1 cap')
        fixed = coarse_fine_indices(pattern)
        self.fixed_positions = np.flatnonzero(np.isin(self.rows, fixed))
        self.tangent = row_sum_basis(self.base, fixed)
        counts = np.diff(self.base.indptr)
        self.groups = []
        hr, hc, hv = [], [], []
        self.rhs_fit = np.zeros(self.base.nnz)
        for degree in np.unique(counts):
            rows = np.flatnonzero(counts == degree)
            positions = self.base.indptr[rows, None] + np.arange(degree)[None, :]
            self.groups.append((positions, rows))
            features = self.zc[self.cols[positions]]
            blocks = np.einsum('ikm,m,ilm->ikl', features, self.w, features)
            blocks += teacher_config.mu * np.eye(degree)[None, :, :]
            self.rhs_fit[positions] = np.einsum('ikm,m,im->ik', features, self.w, self.z[rows])
            hr.append(np.broadcast_to(positions[:, :, None], blocks.shape).ravel())
            hc.append(np.broadcast_to(positions[:, None, :], blocks.shape).ravel())
            hv.append(blocks.ravel())
        self.fit_matrix = sp.coo_matrix((np.concatenate(hv),
            (np.concatenate(hr), np.concatenate(hc))), shape=(self.base.nnz,) * 2).tocsr()
        terms = self.raw_terms(self.base.data)
        # Fail explicitly on a zero reference scale; no undocumented floor.
        self.scales = Normalizers(terms['bamg'], terms['energy'])
        self.reference_digest = array_digest(self.base.data)

    def matrix(self, values):
        return sp.csr_matrix((np.asarray(values), self.base.indices.copy(),
                             self.base.indptr.copy()), shape=self.base.shape)

    def gather(self, matrix):
        return np.asarray(matrix.tocsr()[self.rows, self.cols]).ravel()

    def raw_terms(self, values):
        p = self.matrix(values)
        error = self.z - p @ self.zc
        fit = float(np.sum(error ** 2 * self.w))
        ridge = float(self.config.mu * np.dot(values, values))
        energy = float(p.multiply(self.A @ p).sum())
        return dict(fit=fit, ridge=ridge, bamg=fit + ridge, energy=energy)

    def coefficients(self, strength, energy_only=False):
        if not math.isfinite(strength) or strength < 0:
            raise ValueError('lambda must be finite and nonnegative')
        return (0., 1. / self.scales.energy) if energy_only else (
            1. / self.scales.bamg, strength / self.scales.energy)

    def evaluate(self, values, strength, *, energy_only=False):
        b, e = self.coefficients(strength, energy_only)
        terms = self.raw_terms(values)
        return b * terms['bamg'] + e * terms['energy']

    def half_hessian(self, values, strength, *, energy_only=False):
        b, e = self.coefficients(strength, energy_only)
        result = b * (self.fit_matrix @ values)
        if e:
            result = result + e * self.gather(self.A @ self.matrix(values))
        return np.asarray(result).ravel()

    def gradient(self, values, strength, *, energy_only=False):
        b, _ = self.coefficients(strength, energy_only)
        return 2 * (self.half_hessian(values, strength, energy_only=energy_only) - b * self.rhs_fit)

    def project(self, values):
        return project_rows(values, self.groups, self.targets, self.config.l1_cap,
                            self.fixed_positions, self.base.data[self.fixed_positions])

    def validate(self, values):
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite interpolation')
        row_sum = np.bincount(self.rows, weights=values, minlength=len(self.targets))
        row_l1 = np.bincount(self.rows, weights=np.abs(values), minlength=len(self.targets))
        defect = float(np.max(np.abs(row_sum - self.targets)))
        injection = float(np.max(np.abs(values[self.fixed_positions] - self.base.data[self.fixed_positions])))
        if defect > 1e-9 or injection > 1e-12 or row_l1.max() > self.config.l1_cap + 1e-9:
            raise ValueError('Interpolation violates fixed row/injection/L1 constraints')
        return dict(equality_defect=defect, injection_defect=injection,
                    maximum_row_l1=float(row_l1.max()), structural_nnz=self.base.nnz)


def minimize(objective, strength, *, energy_only=False, maxiter=500, rtol=1e-8):
    """Projected CG, with exact projected-gradient fallback for active L1 caps.

    A finite iteration-budget candidate is reported as unconverged, never as an
    exact minimizer. No dense fine-grid or fine-by-coarse matrix is constructed.
    """
    if type(maxiter) is not int or maxiter < 1 or not math.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError('Invalid optimizer budget/tolerance')
    o = objective
    b, e = o.coefficients(strength, energy_only)
    base = o.base.data.copy()
    before = o.evaluate(base, strength, energy_only=energy_only)
    iterations, fallback_iterations = 0, 0
    cg_info = 0
    solver = 'projected_cg'
    if not energy_only and strength == 0:
        weights, mask, _ = local_fit(o.A, o.pattern, o.z, config=o.config, fixed_mask=o.mask)
        values = o.gather(scipy_from_weights(o.pattern, weights, mask))
        solver = 'existing_bamg_local_fit'
        converged = True
    elif not o.tangent.shape[1]:
        values = base
        converged = True
    else:
        z = o.tangent
        half = lambda v: o.half_hessian(v, strength, energy_only=energy_only)
        rhs = np.asarray(z.T @ (b * o.rhs_fit - half(base))).ravel()
        action = lambda q: np.asarray(z.T @ half(z @ q)).ravel()
        operator = sla.LinearOperator((z.shape[1],) * 2, matvec=action, dtype=np.float64)
        diagonal = b * o.fit_matrix.diagonal() + e * o.A.diagonal()[o.rows]
        prediag = np.asarray(z.power(2).T @ diagonal).ravel()
        if np.any(prediag <= 0):
            raise ValueError('Nonpositive projected Hessian diagonal')
        preconditioner = sla.LinearOperator(operator.shape, matvec=lambda q: q / prediag,
                                           dtype=np.float64)
        def count(_):
            nonlocal iterations
            iterations += 1
        key = 'rtol' if 'rtol' in signature(sla.cg).parameters else 'tol'
        q, cg_info = sla.cg(operator, rhs, M=preconditioner, maxiter=maxiter,
                            atol=0., callback=count, **{key: rtol})
        if cg_info < 0 or not np.isfinite(q).all():
            raise FloatingPointError('Projected CG failed')
        values = base + z @ q
        cap_exceeded = np.max(np.bincount(o.rows, weights=np.abs(values))) > o.config.l1_cap + 1e-10
        converged = cg_info == 0
        if cap_exceeded:
            solver = 'projected_cg_then_l1_projected_gradient'
            values = o.project(values)
            if o.evaluate(values, strength, energy_only=energy_only) > before:
                values = base.copy()
            # A safe Lipschitz upper bound for the restricted quadratic Hessian.
            lipschitz = 2 * (b * np.max(np.asarray(abs(o.fit_matrix).sum(axis=1)))
                            + e * np.max(np.asarray(abs(o.A).sum(axis=1))))
            gradient_scale = max(np.linalg.norm(o.gradient(base, strength, energy_only=energy_only)),
                                 np.finfo(float).tiny)
            y = values.copy()
            momentum = 1.
            value = o.evaluate(values, strength, energy_only=energy_only)
            converged = False
            for fallback_iterations in range(1, maxiter + 1):
                proposal = o.project(y - o.gradient(y, strength, energy_only=energy_only) / lipschitz)
                next_value = o.evaluate(proposal, strength, energy_only=energy_only)
                if next_value > value:
                    y, momentum = values.copy(), 1.
                    proposal = o.project(y - o.gradient(y, strength, energy_only=energy_only) / lipschitz)
                    next_value = o.evaluate(proposal, strength, energy_only=energy_only)
                mapping = lipschitz * (proposal - o.project(proposal -
                    o.gradient(proposal, strength, energy_only=energy_only) / lipschitz))
                next_momentum = (1 + math.sqrt(1 + 4 * momentum ** 2)) / 2
                y = proposal + ((momentum - 1) / next_momentum) * (proposal - values)
                values, value, momentum = proposal, next_value, next_momentum
                if np.linalg.norm(mapping) <= rtol * gradient_scale:
                    converged = True
                    break
    constraints = o.validate(values)
    after = o.evaluate(values, strength, energy_only=energy_only)
    if not math.isfinite(after) or after > before + 1e-9 * max(1., before):
        raise FloatingPointError('Constrained optimization increased reference objective')
    report = dict(solver=solver, lambda_value=float(strength), energy_only=energy_only,
        optimizer_converged=bool(converged), cg_info=int(cg_info), cg_iterations=iterations,
        projected_gradient_iterations=fallback_iterations, maxiter_per_solver=maxiter, rtol=rtol,
        objective_before=before, objective_after=after, terms=o.raw_terms(values),
        s_B=o.scales.bamg, s_E=o.scales.energy, reference_policy=o.scales.reference,
        reference_digest=o.reference_digest, normalizers_fixed=True,
        vectors_digest=array_digest(o.z), vector_weights_digest=array_digest(o.w),
        support_digest=array_digest(o.mask), **constraints)
    return o.matrix(values), report
