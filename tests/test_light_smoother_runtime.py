"""Direction-wide relaxation, frozen-bank reuse and charged online adaptation."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.data import make_examples
from adaptive_mg.hierarchy import classical_cycle
from adaptive_mg.smoothers import classical_smoothing_correction
from adaptive_mg.v67.banks import Stats
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.em_line_models import make_em_line_components
from adaptive_mg.v67.light_smoother_runtime import (PreparedLightMG,
    directional_smoothing, validate_online)


def inputs(n=15):
    torch.set_num_threads(1)
    example = make_examples([n], per_size=4, seed=32)[-1]
    model = make_em_line_components(hidden=4, seed=18)
    with torch.no_grad():
        model.transfer.head.weight.normal_(0, .02)
        model.transfer.head.bias.normal_(0, .02)
    cfg = AdaptiveConfig(mg=MGConfig(strategy_name='line_alt_bilinear_full',
        pre_steps=2, post_steps=2, coarsest_n=3, max_cycles=60,
        stencil_backend='csr'), mode='research', branch='H_P',
        smoother_levels='all', transfer_levels='all', spatial=False,
        gate_mode='open', use_learned_controller=False)
    return example, model, cfg


class Predictor:
    bounds = (.8, 1.2)
    kind = 'mlp'
    def __init__(self, gains=(1., 1.)):
        self.gains = np.array(gains)
        self.calls = 0
    def predict(self, features):
        self.calls += 1
        return self.gains.copy()


@pytest.mark.parametrize('reverse', [False, True])
def test_unit_gains_match_original_line_alt_exactly(reverse):
    example, model, cfg = inputs()
    prepared = PreparedLightMG(example.a, example.n, model, cfg)
    for level in prepared.levels:
        r = np.random.default_rng(level.index+8).normal(size=level.a.shape[0])
        actual = directional_smoothing(level, r, [1., 1.], reverse=reverse)
        expected = classical_smoothing_correction(level.a, level.shape, r,
            'line_alt', cache=level.cache, reverse=reverse)
        np.testing.assert_array_equal(actual, expected)
    zeros = np.zeros_like(example.b)
    np.testing.assert_array_equal(prepared.raw_cycle(zeros, example.b),
        classical_cycle(prepared.learned, zeros, example.b, cfg.mg, Stats()))


@pytest.mark.parametrize('reverse', [False, True])
def test_scaling_is_direction_wide_and_updates_actual_residual(reverse):
    example, model, cfg = inputs()
    prepared = PreparedLightMG(example.a, example.n, model, cfg)
    level = prepared.levels[1]
    r = np.random.default_rng(55).normal(size=level.a.shape[0])
    omega = np.array([.83, 1.17])
    order = [('y', 1), ('x', 0)] if reverse else [('x', 0), ('y', 1)]
    expected, current = np.zeros_like(r), r.copy()
    for direction, index in order:
        delta = omega[index]*classical_smoothing_correction(level.a, level.shape,
            current, 'line_'+direction, cache=level.cache, reverse=reverse)
        expected += delta
        current = r-level.a@expected
    np.testing.assert_allclose(directional_smoothing(level, r, omega, reverse=reverse),
                               expected, rtol=2e-14, atol=2e-14)


def test_fixed_torch_graph_matches_runtime_and_gradient_without_refactoring():
    example, model, cfg = inputs()
    prepared = PreparedLightMG(example.a, example.n, model, cfg)
    coefficients = np.array([[.91, 1.08], [1.13, .86]])
    prepared.set_coefficients(coefficients)
    b = np.random.default_rng(49).normal(size=example.a.shape[0])
    x = np.random.default_rng(50).normal(size=b.size)
    gains = torch.tensor(coefficients, dtype=torch.float64, requires_grad=True)
    factors = prepared._factor_count()
    pids = [id(level.p) for level in prepared.levels]
    y = prepared.differentiable_cycle(torch.tensor(x), torch.tensor(b), gains)
    np.testing.assert_allclose(y.detach().numpy(), prepared.raw_cycle(x, b), rtol=3e-12, atol=3e-12)
    loss = y.square().sum()
    loss.backward()
    step = 1e-6
    for level in range(2):
        for axis in range(2):
            high, low = coefficients.copy(), coefficients.copy()
            high[level, axis] += step
            low[level, axis] -= step
            prepared.set_coefficients(high)
            fhigh = np.sum(prepared.raw_cycle(x, b)**2)
            prepared.set_coefficients(low)
            flow = np.sum(prepared.raw_cycle(x, b)**2)
            np.testing.assert_allclose(gains.grad[level, axis], (fhigh-flow)/(2*step), rtol=3e-5, atol=1e-6)
    assert prepared._factor_count() == factors
    assert [id(level.p) for level in prepared.levels] == pids
    assert all(parameter.grad is None for parameter in model.transfer.parameters())


def test_cache_and_predictor_are_reused_across_cycles_and_rhs(monkeypatch):
    example, model, cfg = inputs()
    predictor = Predictor((.95, 1.05))
    prepared = PreparedLightMG(example.a, example.n, model, cfg, predictor)
    calls, factors = predictor.calls, prepared._factor_count()
    signatures = [(id(level.a), id(level.p), id(level.cache)) for level in prepared.levels]
    def forbidden(*args, **kwargs):
        raise AssertionError('Unexpected feature extraction or NN forward in solve')
    monkeypatch.setattr(predictor, 'predict', forbidden)
    monkeypatch.setattr(model.transfer, 'forward', forbidden)
    import adaptive_mg.v67.light_smoother_models as features
    monkeypatch.setattr(features, 'extract_features', forbidden)
    results = prepared.solve_many(np.stack([example.b, .5*example.b]))
    assert calls == len(prepared.levels)
    assert prepared._factor_count() == factors
    assert signatures == [(id(level.a), id(level.p), id(level.cache)) for level in prepared.levels]
    assert results[0].setup_seconds > 0 and results[1].setup_seconds == 0
    assert results[0].setup_stats['mlp_inference_calls'] == calls
    assert all(result.stats['mlp_inference_calls'] == 0 for result in results)


@pytest.mark.parametrize('method', ['regression_online', 'mlp_ttt'])
def test_online_same_state_trials_are_charged_and_checkpoint_unchanged(method):
    example, model, cfg = inputs(7)
    predictor = Predictor((.8, .8))
    signature = model.signature()
    online = dict(enabled=True, method=method, trigger_window=1, trigger_rho=1e-8,
        near_tolerance_factor=1., min_remaining_cycles=.1, max_calls=1,
        time_budget_seconds=10., gradient_steps=2)
    prepared = PreparedLightMG(example.a, example.n, model, cfg, predictor, online)
    initial = prepared.get_coefficients()
    factor_count, calls = prepared._factor_count(), predictor.calls
    result = prepared.solve(example.b)
    stats = result.stats
    assert stats['online_adaptation_calls'] == 1
    assert stats['online_adaptation_accepted']+stats['online_adaptation_rejected'] == 1
    assert 0 < stats['online_adaptation_seconds'] <= result.solve_seconds
    assert stats['online_probe_cycles'] >= 1
    assert result.executed_cycles == stats['solver_executed_cycles']
    assert stats['all_executed_cycles'] == stats['solver_executed_cycles']+stats['online_probe_cycles']+stats['online_gradient_vcycles']
    if method == 'mlp_ttt':
        assert stats['online_gradient_steps'] == 2
        assert stats['online_gradient_vcycles'] == 2
        assert stats['online_graph_builds'] == 1
    event = stats['online_events'][0]
    assert not event['fallback_used'] and event['source'] == 'same_current_x_and_b'
    assert model.signature() == signature and predictor.calls == calls
    assert prepared._factor_count() == factor_count
    second = prepared.solve(.5*example.b)
    assert second.stats['online_adaptation_calls'] == 0
    prepared.reset_online()
    np.testing.assert_array_equal(prepared.get_coefficients(), initial)
    assert prepared._factor_count() == factor_count


def test_rejected_budgeted_adaptation_restores_coefficients_and_counts_time():
    example, model, cfg = inputs(7)
    prepared = PreparedLightMG(example.a, 7, model, cfg, Predictor(),
        dict(enabled=True, method='mlp_ttt', time_budget_seconds=1e-15))
    before = prepared.get_coefficients()
    prepared._adapt(np.zeros_like(example.b), example.b)
    event = prepared.online_events[-1]
    assert not event['accepted'] and event['seconds'] > 0
    assert event['counters']['online_budget_exhausted'] == 1
    np.testing.assert_array_equal(before, prepared.get_coefficients())


def test_raw_cycle_and_search_solve_never_use_fallback(monkeypatch):
    example, model, cfg = inputs(7)
    prepared = PreparedLightMG(example.a, 7, model, cfg)
    import adaptive_mg.v67.light_smoother_runtime as runtime
    def forbidden(*args, **kwargs):
        raise AssertionError('No classical recovery allowed in raw evaluation')
    monkeypatch.setattr(runtime, 'classical_cycle', forbidden)
    result = prepared.raw_solve(example.b)
    assert result.stats['raw_no_fallback'] and result.stats['fallback_cycles'] == 0
    assert result.stats['online_adaptation_calls'] == 0


def test_fallback_rolls_back_to_same_state(monkeypatch):
    example, model, cfg = inputs(7)
    prepared = PreparedLightMG(example.a, 7, model, cfg)
    monkeypatch.setattr(prepared, 'raw_cycle', lambda x, b, stats=None: np.full_like(x, np.nan))
    result = prepared.solve(example.b)
    assert result.converged and result.stats['rollback_count'] == 1
    assert result.stats['attempted_neural_cycles'] == 1
    assert result.cycle_path[0] == 'classical_recovery'
    assert result.stats['solver_executed_cycles'] == len(result.cycle_path)+1
    expected = classical_cycle(prepared.classical, np.zeros_like(example.b), example.b, cfg.mg, Stats())
    np.testing.assert_allclose(result.residual_history[1], np.linalg.norm(example.b-example.a@expected))


def test_near_tolerance_does_not_trigger_and_online_validation():
    example, model, cfg = inputs(7)
    prepared = PreparedLightMG(example.a, 7, model, cfg, online=dict(enabled=True))
    assert not prepared._trigger([1., .9, .8], .1)
    with pytest.raises(ValueError):
        validate_online(dict(max_calls=0))
    with pytest.raises(ValueError):
        prepared.set_coefficients([[1.3, 1.]])
    with pytest.raises(ValueError):
        prepared.set_coefficients([[float('nan'), 1.]])


def test_frozen_operator_and_p_mutation_are_rejected():
    example, model, cfg = inputs(7)
    prepared = PreparedLightMG(example.a, 7, model, cfg)
    with torch.no_grad():
        model.transfer.head.bias.add_(.01)
    with pytest.raises(ValueError, match='Frozen operator/P changed'):
        prepared.solve(example.b)


def test_unit_gain_baseline_can_skip_summary_features(monkeypatch):
    example, model, cfg = inputs(7)
    import adaptive_mg.v67.light_smoother_models as features
    def forbidden(*args, **kwargs):
        raise AssertionError('Unit-gain reference does not need twelve summary features')
    monkeypatch.setattr(features, 'extract_features', forbidden)
    prepared = PreparedLightMG(example.a, 7, model, cfg, compute_features=False)
    assert prepared.features is None
    result = prepared.solve(example.b)
    assert result.setup_stats['feature_extractions'] == 0
    assert result.setup_stats['feature_seconds'] == 0
    assert result.setup_stats['coefficient_generations'] == 0


@pytest.mark.parametrize('failure', ['P', 'predictor'])
def test_deployment_setup_failure_retains_classical_but_raw_search_refuses(failure, monkeypatch):
    example, model, cfg = inputs(7)
    predictor = Predictor()
    def failed(*args, **kwargs):
        raise ValueError('injected setup failure')
    monkeypatch.setattr(model.transfer if failure == 'P' else predictor,
                        'forward' if failure == 'P' else 'predict', failed)
    with pytest.raises(ValueError, match='injected setup failure'):
        PreparedLightMG(example.a, 7, model, cfg, predictor)
    prepared = PreparedLightMG(example.a, 7, model, cfg, predictor, allow_setup_fallback=True)
    assert prepared.learned is None and prepared.setup_failure is not None
    result = prepared.solve(example.b)
    assert result.converged and result.p_bank == 'classical'
    assert result.actually_executed_branches == ['C']
    assert result.setup_stats['setup_failures'] == 1
    assert result.stats['attempted_neural_cycles'] == 0
    assert result.setup_seconds > 0
    with pytest.raises(RuntimeError, match='raw candidate'):
        prepared.raw_cycle(np.zeros_like(example.b), example.b)
    with pytest.raises(RuntimeError, match='Raw candidate'):
        prepared.raw_solve(example.b)


def test_batch_freshness_is_checked_once(monkeypatch):
    example, model, cfg = inputs(7)
    prepared = PreparedLightMG(example.a, 7, model, cfg)
    actual, calls = prepared._check_frozen, []
    def counted():
        calls.append(1)
        return actual()
    monkeypatch.setattr(prepared, '_check_frozen', counted)
    results = prepared.solve_many(np.stack([example.b, .5*example.b]))
    assert len(calls) == 1
    assert sum(result.stats['freshness_checks'] for result in results) == 1
