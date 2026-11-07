from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg.config import MGConfig
from adaptive_mg.v67 import bamg_energy_labels as labels
from adaptive_mg.v67.bamg_energy import array_digest
from adaptive_mg.v67.bamg_energy_study import candidate_hierarchy
from adaptive_mg.v67.bamg_teacher import TeacherConfig, build_teacher
from adaptive_mg.v67.bamg_training import _aligned_labels, train_supervised
from adaptive_mg.v67.bamg_transfer import (
    WeightCNN, fixed_bilinear_mask, geometry_constraints, scipy_from_weights,
)


@pytest.fixture(scope="module")
def bank():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    n = 15
    line = sp.diags((-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)),
                    (-1, 0, 1), format="csr")
    a = sp.kronsum(line, line, format="csr")
    config = TeacherConfig(
        support_mode="fixed_bilinear", vectors=6, sweeps=(1, 2),
        setup_passes=2, selection_vectors=2, audit_vectors=2,
        eigen_count=0, adaptive_vectors=1, max_vector_retries=0,
    )
    mg = MGConfig(mode="classical", strategy_name="line_alt_bilinear_full__v22",
                  stencil_backend="csr")
    original = build_teacher(a, (n, n), config=config, mg_config=mg)
    yield a, config, mg, original
    torch.set_num_threads(old_threads)


def from_bank(bank, monkeypatch, *, strength=10., **kwargs):
    a, config, mg, original = bank
    calls = []

    def source(operator, shape, **settings):
        calls.append((operator, shape, settings))
        return original

    monkeypatch.setattr(labels, "build_teacher", source)
    result = labels.build_energy_teacher(
        a, (15, 15), config=config, mg_config=mg, strength=strength,
        optimizer=kwargs.pop("optimizer", dict(maxiter=200, rtol=1e-10)), **kwargs,
    )
    assert len(calls) == 1
    return result


def test_actual_galerkin_mass_and_teacher_comparison_match(bank, monkeypatch):
    a, config, mg, original = bank
    result = from_bank(bank, monkeypatch)
    vectors = [level.vectors.copy() for level in original.levels[:-1]]
    reference_ps, reference_reports = candidate_hierarchy(
        a, (15, 15), "hybrid_lambda_10", 10., vectors, None, config, mg,
        dict(maxiter=200, rtol=1e-10),
    )
    assert len(result.levels) == len(original.levels) == 3
    np.testing.assert_array_equal(result.levels[0].H.toarray(), np.eye(a.shape[0]))
    for index, level in enumerate(result.levels[:-1]):
        coarse = result.levels[index + 1]
        np.testing.assert_allclose(level.P.toarray(), reference_ps[index].toarray(), atol=1e-13)
        np.testing.assert_allclose(coarse.A.toarray(), (level.P.T @ level.A @ level.P).toarray(), atol=1e-13)
        np.testing.assert_allclose(coarse.H.toarray(), (level.P.T @ level.H @ level.P).toarray(), atol=1e-13)
        assert level.diagnostics["s_B"] == pytest.approx(reference_reports[index]["s_B"])
        assert level.diagnostics["s_E"] == pytest.approx(reference_reports[index]["s_E"])
        assert np.linalg.eigvalsh(coarse.A.toarray()).min() > 0
        assert np.linalg.eigvalsh(coarse.H.toarray()).min() > 0
    assert not np.allclose(result.levels[1].A.toarray(), original.levels[1].A.toarray())
    last = result.levels[-1]
    assert last.shape == (3, 3)
    assert last.P is last.pattern is last.mask is last.weights is last.vectors is None
    assert result.diagnostics["source_hashes"]["v67/bamg_energy_labels.py"]


def test_frozen_selected_vectors_and_lambda_zero_replay(bank, monkeypatch):
    _, _, _, original = bank
    zero = from_bank(bank, monkeypatch, strength=0.)
    hybrid = from_bank(bank, monkeypatch)
    for source, replay, changed in zip(original.levels[:-1], zero.levels[:-1], hybrid.levels[:-1]):
        np.testing.assert_array_equal(replay.vectors, source.vectors)
        np.testing.assert_array_equal(changed.vectors, source.vectors)
        assert not np.shares_memory(changed.vectors, source.vectors)
        assert changed.diagnostics["bank_vectors_digest"] == array_digest(source.vectors)
        np.testing.assert_allclose(replay.P.toarray(), source.P.toarray(), atol=2e-12)
        assert replay.diagnostics["solver"] == "existing_bamg_local_fit"


def test_fixed_support_signed_constraints_and_aligned_training_labels(bank, monkeypatch):
    result = from_bank(bank, monkeypatch, strength=0.)
    for level in result.levels[:-1]:
        pattern, mask, weights = _aligned_labels(level)
        targets, injection, _ = geometry_constraints(pattern)
        np.testing.assert_array_equal(mask, fixed_bilinear_mask(pattern).numpy())
        np.testing.assert_array_equal(weights, level.weights)
        np.testing.assert_allclose(weights.sum(axis=1), targets, atol=1e-12)
        assert np.max(np.abs(weights).sum(axis=1)) <= result.config.l1_cap + 1e-12
        assert np.all(weights[~mask] == 0.)
        rows = np.flatnonzero(injection >= 0)
        np.testing.assert_array_equal(weights[rows, injection[rows]], np.ones(len(rows)))
        np.testing.assert_allclose(scipy_from_weights(pattern, weights, mask).toarray(),
                                   level.P.toarray(), atol=0.)
    # This bank yields signed fits; neither conversion nor training clips them.
    assert any(np.any(level.weights < 0.) for level in result.levels[:-1])


def test_one_supervised_weight_and_action_update(bank, monkeypatch, tmp_path):
    result = from_bank(bank, monkeypatch)
    torch.manual_seed(17)
    model = WeightCNN(hidden=2, depth=1).double()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    history = train_supervised(
        None, model, [result],
        settings=dict(updates=1, alpha_mask=0., alpha_weight=1., alpha_action=.2),
        checkpoint_path=tmp_path / "supervised.pt",
    )
    assert len(history) == 1 and history[0]["update"] == 1
    assert np.isfinite(history[0]["loss"])
    assert all(np.isfinite(level["action_loss"]) for level in history[0]["levels"])
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_energy_only_and_feasible_budget_limits_are_reported(bank, monkeypatch):
    result = from_bank(bank, monkeypatch, energy_only=True)
    assert result.diagnostics["method"] == "constrained_energy"
    assert result.diagnostics["objective"] == "E/s_E"
    assert all(level.diagnostics["energy_only"] for level in result.levels[:-1])
    limited = from_bank(bank, monkeypatch, optimizer=dict(maxiter=1, rtol=1e-14))
    assert limited.diagnostics["unconverged_levels"]
    assert not limited.diagnostics["optimizer_converged"]


def test_bad_label_and_optimizer_failure_propagate(bank, monkeypatch):
    real_minimize = labels.minimize

    def invalid(objective, strength, **kwargs):
        p, report = real_minimize(objective, strength, **kwargs)
        p.data[objective.fixed_positions[0]] = 2.
        return p, report

    monkeypatch.setattr(labels, "minimize", invalid)
    with pytest.raises(ValueError, match="constraints"):
        from_bank(bank, monkeypatch)

    def failed(*args, **kwargs):
        raise FloatingPointError("optimizer failed")

    monkeypatch.setattr(labels, "minimize", failed)
    with pytest.raises(FloatingPointError, match="optimizer failed"):
        from_bank(bank, monkeypatch)


def test_wrong_support_rejected_before_building_teacher(bank, monkeypatch):
    a, config, mg, _ = bank

    def forbidden(*args, **kwargs):
        pytest.fail("invalid configuration must fail before building the source bank")

    monkeypatch.setattr(labels, "build_teacher", forbidden)
    with pytest.raises(ValueError, match="fixed_bilinear"):
        labels.build_energy_teacher(a, (15, 15), config=replace(config, support_mode="learned"),
                                    mg_config=mg, strength=10., optimizer={})
