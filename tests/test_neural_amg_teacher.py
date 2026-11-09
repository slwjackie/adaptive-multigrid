"""Numerical checks for algebraic fixed-support BAMG + EM teachers."""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from neural_amg.teacher import FixedObjective, fit_teacher, generate_test_vectors, jacobi_omega


def problem():
    A = sp.diags([-np.ones(6), np.full(7, 2.4), -np.ones(6)], [-1, 0, 1], format="csr")
    coarse = np.array([1, 3, 5])
    P = sp.coo_matrix(([.5, 1., .8, .2, 1., .3, .7, 1., .5],
                       ([0, 1, 2, 2, 3, 4, 4, 5, 6], [0, 0, 0, 1, 1, 1, 2, 2, 2])),
                      shape=(7, 3)).tocsr()
    return A, SimpleNamespace(coarse=coarse, reference=P,
                              row_targets=np.asarray(P.sum(axis=1)).ravel())


def test_teacher_preserves_algebraic_constraints_and_decreases_objective():
    A, structure = problem()
    result = fit_teacher(A, structure, strength=10, seed=9, rtol=1e-11)
    assert result.report["optimizer_converged"]
    assert result.report["objective_after"] < result.report["objective_before"]
    assert result.report["reference_policy"] == "classical_amg"
    np.testing.assert_allclose(result.p[structure.coarse].toarray(), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.asarray(result.p.sum(axis=1)).ravel(), structure.row_targets)
    np.testing.assert_array_equal(result.p.indices, structure.reference.indices)
    np.testing.assert_array_equal(result.p.indptr, structure.reference.indptr)
    assert np.linalg.eigvalsh((result.p.T @ A @ result.p).toarray()).min() > 0
    assert not result.vectors.flags.writeable
    assert not result.weights.flags.writeable


def test_teacher_gradient_matches_finite_difference_full_energy():
    A, structure = problem()
    # Full A includes a coupling beyond the tridiagonal graph used by the fixture.
    A = A + sp.coo_matrix(([.15, .15], ([0, 6], [6, 0])), shape=A.shape).tocsr()
    o = FixedObjective(A, structure, generate_test_vectors(A, seed=22))
    v = o.base.data + np.random.default_rng(2).normal(size=o.base.nnz) * .1
    direction = np.random.default_rng(4).normal(size=o.base.nnz)
    eps = 1e-6
    fd = (o.evaluate(v + eps * direction, 3) - o.evaluate(v - eps * direction, 3)) / (2 * eps)
    np.testing.assert_allclose(o.gradient(v, 3) @ direction, fd, rtol=2e-7, atol=1e-8)
    analytic_energy = np.trace(o.matrix(v).toarray().T @ A.toarray() @ o.matrix(v).toarray())
    np.testing.assert_allclose(o.raw_terms(v)["energy"], analytic_energy)


def test_reference_normalizers_and_vector_bank_are_frozen():
    A, structure = problem()
    vectors = generate_test_vectors(A, seed=3)
    o = FixedObjective(A, structure, vectors)
    scales = (o.scales.bamg, o.scales.energy)
    vectors[:] = 10
    structure.reference.data[:] = 20
    for value in (.1, .5, 2):
        o.evaluate(o.base.data * value)
    assert scales == (o.scales.bamg, o.scales.energy)
    np.testing.assert_allclose(np.linalg.norm(o.z, axis=0), 1)
    np.testing.assert_allclose(o.evaluate(o.base.data, 10), 11)


def test_probe_generation_reproducible_and_jacobi_matches_formula():
    A, _ = problem()
    bank = generate_test_vectors(A, count=4, sweeps=5, seed=6)
    np.testing.assert_array_equal(bank, generate_test_vectors(A, count=4, sweeps=5, seed=6))
    rng = np.random.default_rng(6)
    expected = rng.normal(size=(7, 4))
    for _ in range(5):
        expected[:, 1:] -= jacobi_omega(A) * (A @ expected[:, 1:]) / A.diagonal()[:, None]
    expected /= np.linalg.norm(expected, axis=0)
    np.testing.assert_allclose(bank, expected)
    assert not np.allclose(bank, generate_test_vectors(A, count=4, seed=7))
    with pytest.raises(ValueError, match="stability"):
        generate_test_vectors(A, omega=10)


def test_signed_l1_constraints_with_active_cap():
    A, structure = problem()
    vectors = np.array([.1, .1, 4, .2, -4, .3, .1])[:, None]
    o = FixedObjective(A, structure, vectors, l1_cap=1.0)
    projected = o.project(np.linspace(-30, 30, o.base.nnz))
    o.validate(projected)
    result = fit_teacher(A, structure, strength=0, vectors=vectors, l1_cap=1.0, maxiter=1000)
    assert result.report["solver"] == "tangent_cg_then_projected_gradient"
    assert result.report["maximum_row_l1"] <= 1 + 1e-10
    assert result.report["objective_after"] <= result.report["objective_before"]


def test_signed_teacher_is_not_softmax_interpolation():
    A, structure = problem()
    vectors = np.array([.1, .1, 1, .2, -.5, .3, .1])[:, None]
    result = fit_teacher(A, structure, strength=0, vectors=vectors, l1_cap=8)
    assert np.any(result.p.data < 0)
    assert result.report["maximum_row_l1"] <= 8 + 1e-9


def test_finite_budget_reports_without_claiming_exact_optimum():
    A, structure = problem()
    result = fit_teacher(A, structure, maxiter=1, rtol=1e-14)
    assert not result.report["optimizer_converged"]
    assert result.report["cg_info"] > 0
    assert result.report["objective_after"] <= result.report["objective_before"]


def test_energy_weighted_teacher_and_validation():
    A, structure = problem()
    bank = generate_test_vectors(A)
    result = fit_teacher(A, structure, vectors=bank, weighting="energy", energy_weight_cap=2)
    assert result.weights.max() / result.weights.min() <= 2 + 1e-10
    with pytest.raises(ValueError, match="l1_cap"):
        fit_teacher(A, structure, l1_cap=.1)
    with pytest.raises(ValueError, match="symmetric"):
        fit_teacher(A + sp.coo_matrix(([1], ([0], [2])), shape=A.shape), structure)
    with pytest.raises(ValueError, match="strength"):
        fit_teacher(A, structure, strength=-1)
    with pytest.raises(ValueError, match="nonzero"):
        fit_teacher(A, structure, vectors=np.zeros((7, 2)))
    bad = SimpleNamespace(coarse=structure.coarse, reference=structure.reference * 2)
    with pytest.raises(ValueError, match="injection"):
        fit_teacher(A, bad)


def test_teacher_does_not_depend_on_geometric_node_ordering():
    A, structure = problem()
    vectors = generate_test_vectors(A, seed=8)
    result = fit_teacher(A, structure, vectors=vectors, rtol=1e-12)
    order = np.array([4, 0, 5, 2, 1, 6, 3])
    inverse = np.argsort(order)
    reordered = SimpleNamespace(coarse=inverse[structure.coarse],
                                reference=structure.reference[order],
                                row_targets=structure.row_targets[order])
    other = fit_teacher(A[order][:, order], reordered, vectors=vectors[order], rtol=1e-12)
    np.testing.assert_allclose(other.p.toarray(), result.p[order].toarray(), rtol=1e-10, atol=1e-11)


def test_teacher_on_actual_classical_amg_support():
    pyamg = pytest.importorskip("pyamg")
    from neural_amg.structure import build_structure
    A = pyamg.gallery.poisson((9, 9), format="csr")
    structure = build_structure(A)
    result = fit_teacher(A, structure, count=4)
    assert result.report["optimizer_converged"]
    assert result.report["objective_after"] < result.report["objective_before"]
    np.testing.assert_array_equal(result.p.indices, structure.reference.indices)


def test_bootstrap_vectors_use_provisional_classical_vcycles():
    pyamg = pytest.importorskip("pyamg")
    from neural_amg.solver import build_hierarchy, _runtime_cycle
    from neural_amg.structure import build_structure
    A = pyamg.gallery.poisson((5, 5), format="csr")
    before = generate_test_vectors(A, seed=8, count=3)
    after = generate_test_vectors(A, seed=8, count=3, bootstrap_cycles=1)
    h = build_hierarchy(A, method="classical", max_coarse=9)
    expected = _runtime_cycle(h, np.zeros_like(before[:, 1:]), before[:, 1:])
    expected /= np.linalg.norm(expected, axis=0)
    np.testing.assert_allclose(after[:, 1:], expected, atol=1e-14)
    np.testing.assert_allclose(after[:, 0], before[:, 0])
    assert not np.allclose(before[:, 1:], after[:, 1:])
    result = fit_teacher(A, build_structure(A), bootstrap_cycles=2)
    assert result.report["bank_method"] == "classical_amg_bootstrap"
    assert result.report["bootstrap_cycles"] == 2
    assert result.report["optimizer_converged"]
    provided = fit_teacher(A, build_structure(A), vectors=after, bootstrap_cycles=2)
    assert provided.report["bank_method"] == "provided_frozen"
    assert provided.report["bootstrap_cycles"] is None
