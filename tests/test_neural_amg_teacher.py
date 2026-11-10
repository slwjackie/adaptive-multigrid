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


def test_deep_p1_bootstrap_reproduces_twelfth_research_case():
    from pathlib import Path
    from neural_amg import study
    from neural_amg.data import materialize
    config = study.load_config(Path(__file__).resolve().parents[1] / 'configs/neural_amg_research.json')
    specs = study._specifications(config)['train']
    order = np.random.default_rng(np.random.SeedSequence([config['seed'], 0, 0])).permutation(len(specs))
    spec = specs[int(order[11])]
    assert spec['name'] == 'train_checkerboard_n127_4'
    A = materialize(spec).a
    with pytest.raises(ValueError, match='max_levels reached'):
        generate_test_vectors(A, count=3, bootstrap_cycles=2, bootstrap_max_levels=10)
    bank = generate_test_vectors(A, count=3, bootstrap_cycles=2, bootstrap_max_levels=20)
    assert np.isfinite(bank).all()
    np.testing.assert_allclose(np.linalg.norm(bank, axis=0), 1)


def test_larger_bootstrap_cap_preserves_completed_labels():
    import pyamg
    from neural_amg.structure import build_structure
    A = pyamg.gallery.poisson((7, 7), format='csr')
    structure = build_structure(A)
    old = fit_teacher(A, structure, bootstrap_cycles=2, bootstrap_max_levels=10)
    fixed = fit_teacher(A, structure, bootstrap_cycles=2, bootstrap_max_levels=20)
    np.testing.assert_array_equal(old.vectors, fixed.vectors)
    np.testing.assert_array_equal(old.p.data, fixed.p.data)


def test_infeasible_reference_uses_feasible_start_and_original_normalizers():
    from neural_amg.structure import Structure
    A = sp.diags([-np.ones(2), np.full(3, 2.4), -np.ones(2)], [-1, 0, 1], format='csr')
    reference = sp.csr_matrix([[1., 0.], [4.6, -3.6], [0., 1.]])
    s = Structure(np.array([0, 2]), reference)
    bank = generate_test_vectors(A, seed=7)
    o = FixedObjective(A, s, bank, l1_cap=8.)
    free = FixedObjective(A, s, bank, l1_cap=None)
    assert o.scales == free.scales
    initial = o.project(reference.data)
    np.testing.assert_allclose(initial, [1., 4.5, -3.5, 1.])
    result = fit_teacher(A, s, vectors=bank, l1_cap=8.)
    assert result.report['projected_initialization']
    assert result.report['reference_maximum_row_l1'] == pytest.approx(8.2)
    assert result.report['reference_objective'] == pytest.approx(11.)
    assert result.report['objective_before'] == pytest.approx(o.evaluate(initial))
    assert result.report['objective_after'] <= result.report['objective_before'] + 1e-10
    assert result.report['maximum_row_l1'] <= 8. + 1e-10
    np.testing.assert_array_equal(s.reference.data, reference.data)
    np.testing.assert_array_equal(o.base.data, reference.data)
    np.testing.assert_allclose(np.asarray(result.p.sum(1)).ravel(), s.row_targets)
    np.testing.assert_allclose(result.p[s.coarse].toarray(), np.eye(2))
    assert result.report['initial_digest'] != result.report['reference_digest']


def test_infeasible_row_sum_still_rejected():
    from neural_amg.structure import Structure
    A = sp.eye(3, format='csr')
    s = Structure(np.array([0, 2]), sp.csr_matrix([[1., 0.], [9.25, -.25], [0., 1.]]))
    with pytest.raises(ValueError, match='row sums make l1_cap infeasible'):
        fit_teacher(A, s, l1_cap=8.)


def test_constrained_teacher_may_exceed_infeasible_reference_objective():
    from neural_amg.structure import Structure
    A = sp.eye(3, format='csr')
    s = Structure(np.array([0, 2]), sp.csr_matrix([[1., 0.], [4.6, -3.6], [0., 1.]]))
    # P0 fits this probe exactly, but cap8 forces P[1,0] <= 4.5. The constrained
    # optimum cannot match the infeasible reference's tiny fitting objective.
    bank = np.array([1., 4.6, 0.])[:, None]
    result = fit_teacher(A, s, vectors=bank, strength=0., mu=1e-6, l1_cap=8.)
    report = result.report
    assert report['projected_initialization']
    assert report['objective_after'] > report['reference_objective']
    assert report['objective_after'] <= report['objective_before'] + 1e-9
    assert report['maximum_row_l1'] <= 8. + 1e-9


def test_case62_teacher_labels_and_backward(tmp_path):
    from pathlib import Path
    import torch
    from neural_amg import study
    config = study.load_config(Path(__file__).resolve().parents[1] / 'configs/neural_amg_research.json')
    specs = study._specifications(config)['train']
    order = np.random.default_rng(np.random.SeedSequence([config['seed'], 0, 0])).permutation(len(specs))
    spec = specs[int(order[61])]
    assert spec['name'] == 'train_local_patch_n63_7'
    records = study._prepare_split(tmp_path, 'train', [spec], set())
    labels = study._labels(config, records[0], tmp_path)
    report = labels[0]['report']
    assert report['reference_maximum_row_l1'] == pytest.approx(8.0862399379, rel=1e-7)
    assert report['projected_initialization']
    assert all(label['report']['maximum_row_l1'] <= 8. + 1e-9 for label in labels)
    model = study._model(config)
    loss = study.supervised_loss(model, labels, config['supervised']['action_weight'])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
