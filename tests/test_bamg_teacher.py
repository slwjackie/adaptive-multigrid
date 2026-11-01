"""Numerical contracts for the fixed-coarsening BAMG label generator."""
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from adaptive_mg.config import MGConfig
from adaptive_mg.transfer import coarse_fine_indices
from adaptive_mg.v67 import bamg_teacher as teacher
from adaptive_mg.v67.bamg_teacher import (
    TeacherConfig, build_teacher, generalized_enrichment, local_fit,
    relaxed_vectors, teacher_cycle,
)
from adaptive_mg.v67.bamg_transfer import (
    direct_pattern, geometry_constraints, scipy_from_weights,
)


def poisson(n):
    t = sp.diags((-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)),
                 (-1, 0, 1), format="csr")
    return sp.kronsum(t, t, format="csr")


def small_config(**values):
    config = TeacherConfig(vectors=4, sweeps=(1, 2), setup_passes=2,
        selection_vectors=2, audit_vectors=2, eigen_count=2, eigen_refine=1,
        adaptive_vectors=1, max_vector_retries=0)
    return replace(config, **values)


def test_absolute_ridge_matches_equality_kkt_and_not_bilinear():
    rng = np.random.default_rng(3)
    B, y = rng.normal(size=(12, 4)), rng.normal(size=12)
    w, target, mu = np.full(12, 1 / 12), 0.5, 0.07
    p, info = teacher._row_fit(B, y, w, target, small_config(mu=mu))
    normal = B.T @ (w[:, None] * B) + mu * np.eye(4)
    kkt = np.block([[normal, np.ones((4, 1))], [np.ones((1, 4)), np.zeros((1, 1))]])
    expected = la.solve(kkt, np.r_[B.T @ (w * y), target])[:4]
    np.testing.assert_allclose(p, expected, atol=1e-12)
    expected_objective = np.sum(w * (B @ p - y) ** 2) + mu * (p @ p)
    assert info["objective"] == pytest.approx(expected_objective)
    assert info["effective_mu"] == mu
    assert info["free_rank"] == 3


@pytest.mark.parametrize("count", [16, 128])
@pytest.mark.parametrize("mu", [0.0, 1e-6])
def test_more_identical_test_vectors_do_not_restore_free_rank(count, mu):
    B = np.ones((count, 4))
    p, info = teacher._row_fit(B, np.zeros(count), np.full(count, 1 / count),
                               1.0, small_config(mu=mu))
    assert info["free_rank"] == 0
    assert info["rank_deficient"]
    assert info["free_condition"] is None
    np.testing.assert_allclose(p, np.full(4, 0.25), atol=1e-12)
    assert info["solution_status"] == ("regularized_rank_deficient" if mu else
                                         "minimum_norm_rank_deficient")


def test_signed_l1_constrained_fit_is_feasible_and_optimal():
    p, info = teacher._row_fit(np.array([[1.0, 1.001]]), np.zeros(1), np.ones(1),
                               1.0, small_config(mu=0, l1_cap=2.0))
    np.testing.assert_allclose(p, [1.5, -0.5], atol=2e-7)
    assert info["l1_active"]
    assert np.abs(p).sum() <= 2 + 1e-9


def test_local_fit_constraints_heldout_and_fixed_mask_refit():
    A, pattern = poisson(7), direct_pattern(7)
    rng = np.random.default_rng(31)
    z, heldout = rng.normal(size=(49, 8)), rng.normal(size=(49, 5))
    config = small_config(k=4)
    weights, mask, report = local_fit(A, pattern, z, config=config, heldout_vectors=heldout)
    targets, injection, valid = geometry_constraints(pattern)
    assert not np.any(mask & ~valid)
    expected_degree = np.minimum(4, valid.sum(axis=1))
    expected_degree[injection >= 0] = 1
    np.testing.assert_array_equal(mask.sum(axis=1), expected_degree)
    np.testing.assert_allclose(weights.sum(axis=1), targets, atol=1e-12)
    rows = np.flatnonzero(injection >= 0)
    np.testing.assert_array_equal(weights[rows, injection[rows]], 1)
    P = scipy_from_weights(pattern, weights, mask)
    normalised = heldout / np.linalg.norm(heldout, axis=0)
    error = normalised - P @ normalised[coarse_fine_indices(pattern)]
    assert report["heldout_fit_error"] == pytest.approx(np.mean(np.sum(error ** 2, axis=0)))
    new_weights, fixed_mask, new_report = local_fit(A, pattern, z * rng.uniform(.5, 2., (49, 1)),
        config=config, fixed_mask=mask, heldout_vectors=heldout)
    np.testing.assert_array_equal(fixed_mask, mask)
    assert new_report["search"] == "fixed_mask_refit"
    assert np.linalg.norm(new_weights - weights) > 1e-6
    assert P.nnz == mask.sum()
    # Invalid fixed masks fail rather than changing support under a refit.
    broken = mask.copy()
    broken[rows[0], injection[rows[0]]] = False
    with pytest.raises(ValueError, match="cardinalit"):
        local_fit(A, pattern, z, config=config, fixed_mask=broken)


def test_energy_weights_are_normalized_and_capped():
    A = sp.diags([1., 10., 100.], format="csr")
    config = small_config(weighting="energy", energy_weight_cap=7)
    w = teacher._vector_weights(A, np.eye(3), config)
    assert w.sum() == pytest.approx(1)
    assert w.max() / w.min() <= 7 + 1e-12


def test_vector_sweep_mixture_and_independent_seed_reproducibility():
    A = poisson(7)
    z, report = relaxed_vectors(A, 7, 5, (1, 2, 4), seed=4)
    same, _ = relaxed_vectors(A, 7, 5, (1, 2, 4), seed=4)
    different, _ = relaxed_vectors(A, 7, 5, (1, 2, 4), seed=5)
    np.testing.assert_array_equal(z, same)
    assert not np.array_equal(z, different)
    np.testing.assert_allclose(np.linalg.norm(z, axis=0), 1)
    assert report["sweeps_per_vector"] == [1, 2, 4, 1, 2]


def test_two_pass_bootstrap_rebuilds_actual_operators_and_mass():
    result = build_teacher(poisson(7), 7, config=small_config())
    report = result.diagnostics
    assert report["method"] == "fixed_coarsening_bootstrap_amg"
    assert report["setup_passes_attempted"] == 2
    first, second = report["passes"]
    assert first["initial_construction"] and not second["initial_construction"]
    assert first["valid"] and second["valid"]
    assert first["enrichment"]["enabled"]
    assert second["level_fit_diagnostics"][0]["eigen_count"] == 2
    assert second["level_fit_diagnostics"][0]["actual_vector_count"] > 4
    assert first["A_digests"][1] != second["A_digests"][1]
    best = min(report["passes"], key=lambda p: p["selection"]["geometric_mean_ratio"])
    assert report["selected_pass"] == best["pass"]
    assert report["selection_seed"] != report["audit_seed"] != result.config.seed
    assert report["full_paper_reproduction"] is False
    for fine, coarse in zip(result.levels, result.levels[1:]):
        np.testing.assert_allclose(coarse.A.toarray(), (fine.P.T @ fine.A @ fine.P).toarray(), atol=1e-12)
        np.testing.assert_allclose(coarse.H.toarray(), (fine.P.T @ fine.H @ fine.P).toarray(), atol=1e-12)
    assert result.levels[-1].P is None
    assert not np.allclose(result.levels[-1].H.toarray(), np.eye(9))
    # Every diagnostic can be persisted as strict JSON, including singular rows.
    json.dumps(report, allow_nan=False)
    restored = pickle.loads(pickle.dumps(result))
    assert restored.levels[0].cache is None and restored.levels[-1].lu is None
    rhs = np.random.default_rng(1).normal(size=49)
    np.testing.assert_allclose(teacher_cycle(restored, np.zeros(49), rhs),
                               teacher_cycle(result, np.zeros(49), rhs), atol=1e-14)


def test_generalized_modes_use_nonidentity_mass_and_reduce_eigen_residual():
    result = build_teacher(poisson(7), 7, config=small_config(setup_passes=1))
    modes, report = generalized_enrichment(result.levels, config=result.config,
                                           mg_config=result.mg_config)
    last, vectors = result.levels[-1], modes[len(result.levels) - 1]
    lambdas = np.asarray(report["levels"]["1"]["eigenvalues"])
    np.testing.assert_allclose(last.A @ vectors, (last.H @ vectors) * lambdas, atol=1e-12)
    for index, v in modes.items():
        np.testing.assert_allclose(v.T @ result.levels[index].H @ v, np.eye(v.shape[1]), atol=1e-11)
    fine_report = report["levels"]["0"]
    assert np.linalg.norm(fine_report["residual_after"]) <= np.linalg.norm(fine_report["residual_before"])
    assert np.linalg.norm(modes[0]) > 0.1


def test_three_level_bootstrap_does_not_densify_fine_grid(monkeypatch):
    toarray = sp.csr_matrix.toarray
    def bounded_dense(matrix, *args, **kwargs):
        assert matrix.shape[0] <= 9, "fine or intermediate operator was densified"
        return toarray(matrix, *args, **kwargs)
    monkeypatch.setattr(sp.csr_matrix, "toarray", bounded_dense)
    config = small_config(selection_vectors=1, audit_vectors=1, adaptive_vectors=0)
    result = build_teacher(poisson(15), 15, config=config)
    assert len(result.levels) == 3
    assert result.diagnostics["failed_passes"] == 0
    assert len(result.diagnostics["passes"][0]["enrichment"]["levels"]) == 3
    for fine, coarse in zip(result.levels, result.levels[1:]):
        delta_A = coarse.A - fine.P.T @ fine.A @ fine.P
        delta_H = coarse.H - fine.P.T @ fine.H @ fine.P
        assert np.linalg.norm(delta_A.data) < 1e-11
        assert np.linalg.norm(delta_H.data) < 1e-11


def test_one_pass_and_adaptive_variant_are_honestly_named():
    one = build_teacher(poisson(7), 7, config=small_config(setup_passes=1))
    assert one.diagnostics["method"] == "one_pass_relaxed_vector_ls"
    assert "enrichment" not in one.diagnostics["passes"][0]
    adaptive = build_teacher(poisson(7), 7, config=small_config(eigen_count=0))
    assert adaptive.diagnostics["method"] == "adaptive_ls_variant"
    assert not adaptive.diagnostics["passes"][0]["enrichment"]["enabled"]


def test_audit_failure_does_not_select_or_discard_teacher(monkeypatch):
    probe = teacher._probe_contraction
    config = small_config(setup_passes=1)
    def failing_audit(levels, count, seed, cycles, mg):
        if seed == config.seed + 2_000_003:
            raise FloatingPointError("synthetic audit failure")
        return probe(levels, count, seed, cycles, mg)
    monkeypatch.setattr(teacher, "_probe_contraction", failing_audit)
    result = build_teacher(poisson(7), 7, config=config)
    assert result.diagnostics["passes"][0]["valid"]
    assert result.diagnostics["audit_failures"] == 1
    assert result.diagnostics["selected_pass"] == 0


def test_failed_later_pass_is_reported_and_prior_hierarchy_retained(monkeypatch):
    build = teacher._build_pass
    def fail_second(A, shape, config, mg, pass_index, enrichment, adaptive):
        if pass_index:
            raise RuntimeError("synthetic second-pass failure")
        return build(A, shape, config, mg, pass_index, enrichment, adaptive)
    monkeypatch.setattr(teacher, "_build_pass", fail_second)
    result = build_teacher(poisson(7), 7, config=small_config())
    assert result.diagnostics["failed_passes"] == 1
    assert result.diagnostics["selected_pass"] == 0
    assert "synthetic" in result.diagnostics["passes"][1]["failure"]
    assert result.levels[-1].lu is not None


def test_configuration_roundtrip_validation_and_strategy_scope():
    config = small_config()
    assert TeacherConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    for field, value in [("mu", float("nan")), ("vectors", 0), ("k", 17),
                         ("l1_cap", 0.5), ("sweeps", (1, -1))]:
        with pytest.raises(ValueError):
            replace(config, **{field: value})
    with pytest.raises(ValueError, match="line_alt"):
        build_teacher(poisson(7), 7, config=config,
                       mg_config=MGConfig(mode="classical", strategy_name="jacobi_bilinear_full"))
    with pytest.raises(ValueError, match="finite"):
        local_fit(poisson(7), direct_pattern(7), np.full((49, 4), np.nan))
