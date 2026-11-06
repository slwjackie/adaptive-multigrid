from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp

from adaptive_mg.v67.bamg_energy import FixedObjective, minimize, project_rows
from adaptive_mg.v67.bamg_teacher import TeacherConfig, local_fit, relaxed_vectors
from adaptive_mg.v67.bamg_transfer import direct_pattern, scipy_from_weights


def poisson(n):
    t = sp.diags((-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)), (-1, 0, 1), format='csr')
    return sp.kronsum(t, t, format='csr')


def problem(n=7, cap=8.):
    a = poisson(n)
    vectors, _ = relaxed_vectors(a, (n, n), 8, (1, 2), seed=9)
    tc = TeacherConfig(support_mode='fixed_bilinear', l1_cap=cap, mu=1e-5)
    return FixedObjective(a, direct_pattern(n), vectors, tc)


def test_normalizers_are_frozen_reference_values_and_energy_scale_invariant():
    o = problem()
    initial = o.scales
    assert o.evaluate(o.base.data, .7) == pytest.approx(1.7)
    values = o.base.data + o.tangent @ np.ones(o.tangent.shape[1]) * .02
    o.evaluate(values, .7)
    o.gradient(values, .7)
    assert o.scales is initial
    with pytest.raises(FrozenInstanceError):
        o.scales.energy = 4
    scaled = FixedObjective(19 * o.A, o.pattern, o.z, o.config)
    assert scaled.scales.energy == pytest.approx(19 * o.scales.energy)
    assert scaled.scales.bamg == pytest.approx(o.scales.bamg)
    assert scaled.evaluate(values, .7) == pytest.approx(o.evaluate(values, .7))
    p, _ = minimize(o, .7)
    ps, _ = minimize(scaled, .7)
    np.testing.assert_allclose(p.toarray(), ps.toarray(), atol=2e-7)


def test_lambda_zero_reproduces_existing_absolute_ridge_bamg():
    o = problem()
    weights, mask, _ = local_fit(o.A, o.pattern, o.z, config=o.config, fixed_mask=o.mask)
    expected = scipy_from_weights(o.pattern, weights, mask)
    actual, report = minimize(o, 0.)
    np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=1e-13)
    np.testing.assert_array_equal(actual.indptr, o.base.indptr)
    np.testing.assert_array_equal(actual.indices, o.base.indices)
    assert report['solver'] == 'existing_bamg_local_fit'


def test_global_gradient_and_projected_cg_match_dense_independent_quadratic():
    o = problem()
    rng = np.random.default_rng(7)
    q = rng.normal(size=o.tangent.shape[1]) * .01
    values = o.base.data + o.tangent @ q
    direction = o.tangent @ rng.normal(size=len(q))
    delta = 1e-6
    fd = (o.evaluate(values + delta*direction, .4) - o.evaluate(values - delta*direction, .4)) / (2*delta)
    assert fd == pytest.approx(o.gradient(values, .4) @ direction, rel=2e-7)
    # Dense independent Hessian only on the tiny test: full cross-row A couplings.
    rows, cols = o.rows, o.cols
    ae = o.A.toarray()[rows[:, None], rows[None, :]] * (cols[:, None] == cols[None, :])
    h = o.fit_matrix.toarray() / o.scales.bamg + .4 * ae / o.scales.energy
    z = o.tangent.toarray()
    rhs = z.T @ (o.rhs_fit / o.scales.bamg - h @ o.base.data)
    expected = o.base.data + z @ la.solve(z.T @ h @ z, rhs, assume_a='pos')
    p, report = minimize(o, .4, rtol=1e-11)
    np.testing.assert_allclose(o.gather(p), expected, atol=1e-9)
    assert report['optimizer_converged']
    assert report['objective_after'] <= report['objective_before']


@pytest.mark.parametrize('target,cap', [(1., 1.), (.5, 1.), (.25, 2.), (1., 8.)])
def test_row_projection_matches_independent_constrained_optimizer(target, cap):
    rng = np.random.default_rng(4)
    for degree in (2, 4):
        groups = [(np.arange(degree)[None, :], np.array([0]))]
        for _ in range(6):
            v = rng.normal(size=degree) * 5
            actual = project_rows(v, groups, np.array([target]), cap,
                                  np.array([], int), np.array([]))
            # L1 ball is the intersection of all signed linear halfspaces.
            import itertools
            signs = np.asarray(list(itertools.product((-1., 1.), repeat=degree)))
            # All-equal sign rows duplicate the equality when cap == target.
            signs = signs[np.ptp(signs, axis=1) > 0]
            result = opt.minimize(lambda x: .5*np.sum((x-v)**2), np.full(degree, target/degree),
                jac=lambda x: x-v, method='SLSQP', constraints=[
                    {'type': 'eq', 'fun': lambda x: x.sum()-target, 'jac': lambda x: np.ones(degree)},
                    {'type': 'ineq', 'fun': lambda x: cap-signs@x, 'jac': lambda x: -signs}],
                options={'ftol': 1e-11, 'maxiter': 200})
            assert result.success
            np.testing.assert_allclose(actual, result.x, atol=2e-6)


def test_active_l1_hybrid_preserves_injection_and_structural_zeros():
    o = problem(cap=1.)
    # Correlated vectors make the unconstrained fit prefer large signed weights.
    rng = np.random.default_rng(8)
    vectors = rng.normal(size=(49, 8))
    o = FixedObjective(o.A, o.pattern, vectors, replace(o.config, mu=1e-10))
    p, report = minimize(o, .0001, maxiter=1500, rtol=1e-7)
    o.validate(o.gather(p))
    assert report['solver'] == 'projected_cg_then_l1_projected_gradient'
    assert report['objective_after'] <= report['objective_before']
    assert p.nnz == o.base.nnz
    assert report['maximum_row_l1'] <= 1 + 1e-9


def test_energy_only_is_not_ridge_or_vector_weighting():
    o = problem()
    changed = FixedObjective(o.A, o.pattern, np.random.default_rng(12).normal(size=o.z.shape),
                             replace(o.config, mu=.7))
    p, report = minimize(o, 0., energy_only=True)
    other, _ = minimize(changed, 0., energy_only=True)
    np.testing.assert_allclose(p.toarray(), other.toarray(), atol=1e-7)
    assert report['terms']['energy'] <= o.scales.energy
    assert report['energy_only']
