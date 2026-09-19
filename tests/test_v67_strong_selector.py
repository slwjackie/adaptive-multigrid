"""Deployment contracts: A-only selection, shared baseline, cache and safety."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from adaptive_mg import DiffusionCase, MGConfig, assemble_stiffness
from adaptive_mg.v67 import AdaptiveConfig, Components, PreparedAdaptiveMG
from adaptive_mg.v67.banks import Stats
from adaptive_mg.v67.strong import (
    StrongRules, PreparedStrongMG, classical_bank, load_strong_rules,
    select_strong_strategy,
)
import adaptive_mg.v67.strong as strong_module


def problem(branch='C', *, n=7, epsilon=1., **mg_changes):
    a = assemble_stiffness(DiffusionCase(n=n, epsilon=epsilon, angle_deg=0., contrast=1.))
    b = np.random.default_rng(83).normal(size=n*n)
    mg = MGConfig(mode='classical', max_cycles=5, tolerance=1e-14,
                  absolute_tolerance=0., near_tolerance_factor=0.,
                  nn_levels=2, stencil_backend='csr')
    cfg = AdaptiveConfig(mg=replace(mg, **mg_changes), mode='research', branch=branch,
                         spatial=False, gate_mode='open')
    return a, b, cfg


def test_frozen_defaults_roundtrip_and_complete_bank(tmp_path):
    rules = StrongRules()
    path = tmp_path/'rules.json'
    path.write_text(json.dumps(rules.to_dict()))
    assert load_strong_rules(path) == rules == load_strong_rules()
    assert load_strong_rules(path).digest() == rules.digest()
    assert len(classical_bank()) == 16 and len(classical_bank('all')) == 48
    with pytest.raises(ValueError):
        classical_bank('audit')
    with pytest.raises(FrozenInstanceError):
        rules.anisotropy_min = 8.


@pytest.mark.parametrize('change', [
    {'version':'arbitrary'}, {'anisotropy_min':1.}, {'anisotropy_min':float('nan')},
    {'isotropic_ratio_max':float('inf')}, {'contrast_min':1.},
    {'alignment_angle_deg':45.}, {'diagonal_angle_tolerance_deg':0.},
    {'off_stencil_fraction_max':-1.}, {'valid_tensor_fraction_min':1.1},
    {'log_diagonal_std_min':False}, {'strategy_by_rule':{'near_isotropic':'jacobi_bilinear_full'}},
    {'provenance':{}},
])
def test_rule_schema_rejects_invalid_changes(change):
    with pytest.raises((ValueError, TypeError)):
        StrongRules.from_dict({**StrongRules().to_dict(), **change})


def test_unknown_rule_and_strategy_rejected():
    with pytest.raises(ValueError):
        StrongRules().replace_strategies({'test_case_name':'jacobi_bilinear_full'})
    with pytest.raises(ValueError):
        StrongRules().replace_strategies({'near_isotropic':'unknown'})
    with pytest.raises(TypeError):
        StrongRules.from_dict({**StrongRules().to_dict(), 'timing':1.})


@pytest.mark.parametrize('epsilon,angle,leaf,strategy', [
    (1., 0., 'near_isotropic', 'jacobi_bilinear_full'),
    (.01, 0., 'aligned_x', 'line_x_operator_semi_y'),
    (.01, 90., 'aligned_y', 'line_y_operator_semi_x'),
    (.01, 45., 'diagonal_positive', 'line_diag45_operator_full'),
    (.01, -45., 'rotated', 'line_alt_operator_full'),
    (.01, 25., 'rotated', 'line_alt_operator_full'),
    (.5, 0., 'moderate', 'chebyshev_operator_full'),
])
def test_declared_rule_routing(epsilon, angle, leaf, strategy):
    a = assemble_stiffness(DiffusionCase(n=7, epsilon=epsilon, angle_deg=angle))
    selected = select_strong_strategy(a, 7)
    assert selected.rule_id == leaf and selected.strategy_name == strategy
    assert selected.rule_evidence['first_matching_rule'] == leaf
    assert selected.rule_evidence['ordered_checks'][leaf]
    assert selected.selector_seconds >= 0.


def test_remapping_changes_strategy_without_changing_feature_leaf():
    a, _, _ = problem()
    original = select_strong_strategy(a, 7)
    rules = StrongRules().replace_strategies({'near_isotropic':'chebyshev_bilinear_full'},
                                            provenance='synthetic schema contract; no measurement fitting')
    changed = select_strong_strategy(a, 7, rules)
    assert original.rule_id == changed.rule_id
    assert original.features == changed.features
    assert original.strategy_name != changed.strategy_name
    assert original.rules_digest != changed.rules_digest


def test_c_uses_identical_selected_hierarchy_and_preserves_requested_config():
    a, b, cfg = problem()
    prepared = PreparedStrongMG(a, 7, config=cfg)
    selected_cfg = replace(cfg, mg=replace(cfg.mg, strategy_name=prepared.selection.strategy_name))
    assert prepared.config == selected_cfg
    assert cfg.mg.strategy_name == 'line_alt_bilinear_full'
    direct = PreparedAdaptiveMG(a, 7, config=selected_cfg).solve(b)
    actual = prepared.solve(b)
    np.testing.assert_array_equal(actual.x, direct.x)
    assert actual.residual_history == direct.residual_history
    assert actual.stats['nn_forward_calls'] == actual.stats['learned_operator_applications'] == 0
    assert actual.setup_seconds >= actual.stats['selector_seconds'] > 0
    assert actual.stats['selector_calls'] == 1
    assert 'selector_seconds' not in actual.setup_stats
    assert not actual.certificate_used


def test_selector_reused_across_rhs_and_batch_but_rebuilt_after_a_change(monkeypatch):
    a, b, cfg = problem()
    original = strong_module.select_strong_strategy
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(strong_module, 'select_strong_strategy', counted)
    prepared = PreparedStrongMG(a, 7, config=cfg)
    first = prepared.solve(b)
    base = prepared.classical
    batch = prepared.solve_many(np.stack((b, -b, 2*b)))
    assert len(calls) == 1 and all(r.stats['selector_calls'] == 0 for r in batch)
    assert all(r.stats['selector_seconds'] == 0 and r.setup_seconds == 0 for r in batch)
    np.testing.assert_array_equal(first.x, batch[0].x)
    assert prepared.classical is base
    old_key = first.cache_key
    a.data *= 1.1
    after = prepared.solve(b)
    assert len(calls) == 2 and prepared.classical is not base
    assert after.cache_key != old_key and after.stats['selector_calls'] == 1


def test_rules_change_drops_all_classical_and_neural_banks():
    a, b, cfg = problem('H_SP')
    c = Components.create(hidden=4)
    prepared = PreparedStrongMG(a, 7, c, cfg)
    prepared.solve(b)
    old_classical, old_neural = prepared.classical, prepared.learned
    prepared.rules = prepared.rules.replace_strategies({'near_isotropic':'chebyshev_bilinear_full'})
    result = prepared.solve(b)
    assert prepared.selection_count == 2
    assert prepared.classical is not old_classical and prepared.learned is not old_neural
    assert prepared.strategy.name == 'chebyshev_bilinear_full'
    assert result.stats['selector_calls'] == 1


def test_replace_operator_reselects_and_shape_changes_safely():
    a, b, cfg = problem()
    prepared = PreparedStrongMG(a, 7, config=cfg)
    prepared.solve(b)
    a15, b15, _ = problem(n=15, epsilon=.01)
    prepared.replace_operator(a15, 15)
    assert prepared.selection.rule_id == 'aligned_x'
    result = prepared.solve(b15)
    assert len(result.x) == 225 and prepared.selection_count == 2
    assert result.stats['selector_calls'] == 1


@pytest.mark.parametrize('branch', ['H_S','H_P','H_SP'])
def test_neural_uses_same_selected_classical_base_with_real_applications(branch):
    a, b, cfg = problem(branch)
    components = Components.create(hidden=4)
    prepared = PreparedStrongMG(a, 7, components, cfg)
    direct = PreparedAdaptiveMG(a, 7, components, prepared.config).solve(b)
    result = prepared.solve(b)
    np.testing.assert_array_equal(result.x, direct.x)
    assert result.residual_history == direct.residual_history
    assert result.stats['branch_'+branch+'_attempts'] > 0
    assert result.stats['learned_operator_applications'] > 0
    assert result.abstention['selected_classical_strategy'] == prepared.classical.strategy.name
    assert result.branch_policy_status == 'forced_component_with_safety_uncalibrated_strong_baseline'
    assert not result.certificate_used and not result.abstention['strong_policy_calibrated']
    if branch == 'H_S':
        assert result.stats['transfer_nn_calls'] == 0
        assert prepared.learned.p is prepared.classical.p
    if branch == 'H_P':
        assert result.stats['smoother_nn_calls'] == 0


def test_neural_failure_recovers_with_selected_classical_not_fixed_default():
    a, b, cfg = problem('H_S', smoother_gain_multiplier=1e80, max_cycles=4)
    components = Components.create(hidden=4)
    prepared = PreparedStrongMG(a, 7, components, cfg)
    result = prepared.solve(b)
    assert result.stats['rejected_neural_cycles'] == 1
    assert result.stats['classical_recovery_cycles'] == 1
    baseline = PreparedAdaptiveMG(a, 7, config=replace(prepared.config, branch='C',
                                   mg=replace(prepared.config.mg, max_cycles=3))).solve(b)
    np.testing.assert_array_equal(result.x, baseline.x)
    assert prepared.selection.strategy_name == 'jacobi_bilinear_full'


@pytest.mark.parametrize('mode', ['research','production'])
def test_old_auto_policy_and_certificate_never_authorize_strong_neural(mode):
    a, b, cfg = problem()
    cfg = replace(cfg, branch='auto', mode=mode)
    components = Components.create(hidden=4)
    components.metadata['controller_trained'] = True
    components.metadata['certificate']['validated'] = True
    metadata = json.dumps(components.metadata, sort_keys=True)
    prepared = PreparedStrongMG(a, 7, components, cfg)
    result = prepared.solve(b)
    assert result.stats['nn_forward_calls'] == 0
    assert result.stats['learned_operator_applications'] == 0
    assert not result.certificate_used and not prepared.certificate_valid()
    assert result.abstention['reason'] == ('invalid_production_certificate' if mode == 'production'
                                         else 'uncalibrated_strong_baseline_policy')
    assert json.dumps(components.metadata, sort_keys=True) == metadata


def test_branch_changes_keep_selection_but_respect_effective_generation():
    a, b, cfg = problem('H_S')
    prepared = PreparedStrongMG(a, 7, Components.create(hidden=4), cfg)
    prepared.solve(b)
    base = prepared.classical
    prepared.config = replace(prepared.config, branch='H_P')
    result = prepared.solve(b)
    assert prepared.selection_count == 1 and prepared.classical is base
    assert result.stats['transfer_nn_calls'] > 0 and result.stats['smoother_nn_calls'] == 0
    assert result.stats['selector_calls'] == 0
