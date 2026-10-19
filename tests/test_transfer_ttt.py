"""Per-A adaptation changes only its private head and charges real setup."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from adaptive_mg import DiffusionCase, MGConfig, assemble_stiffness
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.fixed_classical import FixedClassicalBaseline, FIXED_STRATEGY_NAME
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.solver import PreparedAdaptiveMG
from adaptive_mg.v67.transfer_ttt import (PreparedTTTMG, _guard_accepts,
                                        validate_ttt_settings)
from adaptive_mg.v67.unet_transfer import CompactTransferStudent


def case():
    torch.set_num_threads(1)
    cfg = AdaptiveConfig(
        mg=MGConfig(strategy_name=FIXED_STRATEGY_NAME, em_fixed_iterations=True,
                    stencil_backend='csr', coarsest_n=3),
        mode='research', branch='H_P', use_smoother=False, use_transfer=True,
        transfer_levels='all', spatial=False, gate_mode='open')
    model = Components.create(hidden=4, seed=13)
    model.transfer = CompactTransferStudent(hidden=4)
    a = assemble_stiffness(DiffusionCase(n=7, epsilon=.2, angle_deg=25.))
    return a, model, cfg, FixedClassicalBaseline.from_config(cfg)


def state(model):
    return [{key: value.clone() for key, value in module.state_dict().items()}
            for module in model.modules()]


def test_ttt_applies_real_head_gradients_without_mutating_source_or_body():
    a, model, cfg, baseline = case()
    saved, metadata = state(model), deepcopy(model.metadata)
    prepared = PreparedTTTMG(a, 7, model, cfg, baseline)
    report = prepared.ttt_report
    assert report['proposal_adopted'] and report['updates_applied'] == 2
    assert report['updates_retained'] == 2
    assert all(row['gradient_norm'] > 0. for row in report['trace'])
    assert _guard_accepts(report['raw_ratios_before'], report['raw_ratios_after'])
    assert model.metadata == metadata
    for before, module in zip(saved, model.modules()):
        assert all(torch.equal(before[key], value) for key, value in module.state_dict().items())
        assert all(parameter.grad is None for parameter in module.parameters())
    for name, value in model.transfer.state_dict().items():
        adapted = prepared.components.transfer.state_dict()[name]
        if name.startswith('head.'):
            assert not torch.equal(value, adapted)
        else:
            assert torch.equal(value, adapted)
    assert prepared.components is not model
    assert prepared.ttt_seconds > 0.
    assert prepared.adaptation_seconds == prepared.ttt_seconds
    assert prepared.learned is not None
    assert prepared.learned_builds_total == 1


def test_ttt_reuses_em_parent_and_guard_never_calls_safeguarded_solver(monkeypatch):
    a, model, cfg, baseline = case()
    from adaptive_mg import energymin
    from adaptive_mg.v67 import unroll
    real_energy = energymin.energy_weights
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return real_energy(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('TTT must not regenerate EM or call runtime fallback during its guard')

    monkeypatch.setattr(energymin, 'energy_weights', counted)
    monkeypatch.setattr(unroll, 'baseline_weights', forbidden)
    monkeypatch.setattr(PreparedAdaptiveMG, 'solve', forbidden)
    monkeypatch.setattr(PreparedAdaptiveMG, 'solve_many', forbidden)
    prepared = PreparedTTTMG(a, 7, model, cfg, baseline)
    assert prepared.ttt_report['proposal_adopted']
    assert len(calls) == 1
    assert calls[0]['maxiter'] == 5 and calls[0]['fixed_iterations']


def test_guard_rejection_restores_original_pruned_model(monkeypatch):
    a, model, cfg, baseline = case()
    from adaptive_mg.v67 import transfer_ttt
    monkeypatch.setattr(transfer_ttt, '_guard_accepts', lambda before, after: False)
    prepared = PreparedTTTMG(a, 7, model, cfg, baseline)
    assert not prepared.ttt_report['proposal_adopted']
    assert prepared.ttt_report['updates_applied'] == 2
    assert prepared.ttt_report['updates_retained'] == 0
    assert prepared.ttt_report['decision'] == 'guard_rejected'
    for name, value in model.transfer.state_dict().items():
        assert torch.equal(value, prepared.components.transfer.state_dict()[name])
    normal = PreparedAdaptiveMG(a, 7, deepcopy(model), cfg)
    b = np.linspace(.1, 1., a.shape[0])
    np.testing.assert_allclose(prepared.solve(b).x, normal.solve(b).x, rtol=0, atol=1e-14)


def test_ttt_same_a_multiple_rhs_reuses_model_and_charges_once():
    a, model, cfg, baseline = case()
    prepared = PreparedTTTMG(a, 7, model, cfg, baseline)
    adapted = state(prepared.components)
    rhs = np.stack((np.ones(a.shape[0]), np.linspace(.1, 1., a.shape[0])))
    results = prepared.solve_many(rhs)
    assert len(results) == 2 and all(result.converged for result in results)
    assert results[0].setup_stats['ttt_seconds'] == prepared.ttt_seconds
    assert 'ttt_seconds' not in results[1].setup_stats
    assert results[0].setup_seconds >= prepared.ttt_seconds
    assert results[1].setup_seconds == 0.
    assert prepared.solve(rhs[0]).setup_seconds == 0.
    assert prepared.adaptation_runs == 1
    assert prepared.learned_builds_total == 1
    for before, module in zip(adapted, prepared.components.modules()):
        assert all(torch.equal(before[key], value) for key, value in module.state_dict().items())


def test_ttt_new_operator_resets_from_immutable_start_and_has_no_rhs_dependency():
    a, model, cfg, baseline = case()
    saved_start = deepcopy(model)
    prepared = PreparedTTTMG(a, 7, model, cfg, baseline)
    prepared.solve(np.ones(a.shape[0]))
    with torch.no_grad():
        model.transfer.head.bias.add_(3.)  # Caller cannot alter the private origin.
    other = assemble_stiffness(DiffusionCase(n=7, epsilon=.15, angle_deg=40.))
    prepared.replace_operator(other)
    fresh = PreparedTTTMG(other, 7, saved_start, cfg, baseline)
    assert prepared.adaptation_runs == 2
    assert prepared.ttt_report['raw_ratios_before'] == fresh.ttt_report['raw_ratios_before']
    assert prepared.ttt_report['raw_ratios_after'] == fresh.ttt_report['raw_ratios_after']
    for name, value in fresh.components.transfer.state_dict().items():
        assert torch.equal(value, prepared.components.transfer.state_dict()[name])
    # Evaluation solutions are never inputs to either constructor or adaptation.
    a2 = other.copy()
    fresh2 = PreparedTTTMG(a2, 7, saved_start, cfg, baseline)
    fresh2.solve(np.linspace(2., 3., other.shape[0]))
    assert fresh.ttt_report['raw_ratios_after'] == fresh2.ttt_report['raw_ratios_after']


def test_ttt_mutated_operator_adapts_again_and_zero_budget_does_not_train():
    a, model, cfg, baseline = case()
    prepared = PreparedTTTMG(a, 7, model, cfg, baseline, dict(max_updates=0))
    assert prepared.ttt_report['updates_applied'] == 0
    assert prepared.ttt_report['decision'] == 'disabled_zero_update_budget'
    a.data *= 1.1
    prepared.solve(np.ones(a.shape[0]))
    assert prepared.adaptation_runs == 2
    assert prepared.a_digest == prepared.ttt_report['operator_digest']


def test_guard_rejects_one_worse_probe_and_nonfinite_values():
    assert _guard_accepts([.1, .2], [.09, .19])
    assert not _guard_accepts([.1, .2], [.01, .21])
    assert not _guard_accepts([.1], [float('nan')])
    assert not _guard_accepts([.1], [.1])


@pytest.mark.parametrize('settings', [dict(max_updates=-1), dict(max_updates=17),
    dict(max_updates=True), dict(train_probes=0), dict(guard_probes=17),
    dict(raw_cycles=5), dict(learning_rate=float('nan')), dict(learning_rate=.1),
    dict(proximal_weight=-1e-310), dict(smooth_probe_steps=5), dict(seed=-1),
    dict(unknown=1)])
def test_ttt_budget_is_strict_and_bounded(settings):
    with pytest.raises(ValueError):
        validate_ttt_settings(settings)


def test_ttt_requires_fixed_parent_and_transfer_only_config():
    a, model, cfg, baseline = case()
    with pytest.raises(ValueError, match='fixed line_alt'):
        PreparedTTTMG(a, 7, model, cfg, None)
    with pytest.raises(ValueError, match='forced research H_P'):
        PreparedTTTMG(a, 7, model, replace(cfg, branch='C'), baseline)
