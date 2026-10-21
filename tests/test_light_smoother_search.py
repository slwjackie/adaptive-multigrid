"""Offline finite oracle: failures, held-out RHS confirmation, and resume."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.data import make_examples
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.em_line_models import make_em_line_components
from adaptive_mg.v67 import light_smoother_search as search


class FakePrepared:
    def __init__(self, a, n, components, cfg, **kwargs):
        self.a = a
        self.coefficients = np.ones((2, 2))
        self.features = np.arange(24).reshape(2, 12)

    def get_coefficients(self):
        return self.coefficients.copy()

    def set_coefficients(self, value):
        self.coefficients = np.array(value, copy=True)


def data():
    example = make_examples([7], per_size=1, seed=52)[0]
    example.research_split = 'train'
    components = SimpleNamespace(expert_signature=lambda name: 'fixed-p-signature')
    cfg = AdaptiveConfig(mg=MGConfig(strategy_name='line_alt_bilinear_full'))
    return example, components, cfg


def settings():
    return dict(grid=[.8, 1., 1.2], passes=1, repeats=2, rhs_count=2,
                confirmation_rhs_count=2, minimum_improvement=.03, seed=9)


def recorded_time(prepared, b, cfg):
    delta = np.abs(prepared.get_coefficients()-np.array([.8, 1.])).sum()
    elapsed = .1+float(delta)
    return dict(success=True, cycles=3, residual_history=[1., .01, 1e-6, 1e-10],
                stopping_threshold=1e-8, final_true_residual=1e-10,
                wall_seconds=elapsed, stop_reason='converged', error=None,
                fallback_used=False, online_adaptation_used=False)


def test_grid_protocol_and_disjoint_train_rhs():
    example, _, _ = data()
    spec = search.validate_search_settings(settings())
    assert spec['bounds'] == [.8, 1.2]
    a = search.search_rhs(example, count=2, seed=9, partition='optimization')
    b = search.search_rhs(example, count=2, seed=9, partition='confirmation')
    np.testing.assert_array_equal(a, search.search_rhs(example, count=2, seed=9, partition='optimization'))
    assert all(not np.array_equal(x, y) for x in a for y in b)
    for bad in (dict(grid=[.8, .8]), dict(bounds=[1.1, 1.2]),
                dict(repeats=True), dict(passes=0), dict(minimum_improvement=1)):
        with pytest.raises(ValueError):
            search.validate_search_settings(bad)


def test_full_coefficient_search_and_confirmed_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(search, 'PreparedLightMG', FakePrepared)
    monkeypatch.setattr(search, 'measure_raw_solve', recorded_time)
    result = search.search_case(*data(), settings(), tmp_path/'case')
    assert result['confirmed_improvement'] and result['timing_improvement_confirmed']
    assert result['selected_candidate_success'] and result['label_eligible']
    assert result['selection_outcome'] == 'confirmed_time_improvement'
    np.testing.assert_allclose(result['selected_coefficients'], [[.8, 1.], [.8, 1.]])
    assert len(result['optimization_rounds']) == 2
    assert len(result['trials']) == (9*2 + 2)*2*2
    assert all(not r['fallback_used'] for r in result['trials'])
    assert all(np.shape(r['coefficients']) == (2, 2) for r in result['trials'])
    assert np.shape(result['features']) == (2, 12)
    # At the second coordinate, the first coordinate is preserved throughout.
    for row in result['trials']:
        if row['phase'] == 'coordinate/0/1':
            np.testing.assert_array_equal(row['coefficients'][0], [.8, 1.])
    before = (tmp_path/'case/result.json').read_bytes()
    replay = search.search_case(*data(), settings(), tmp_path/'case', resume=True)
    assert replay == result and (tmp_path/'case/result.json').read_bytes() == before
    with pytest.raises(FileExistsError):
        search.search_case(*data(), settings(), tmp_path/'case')


def test_fast_failures_cannot_win_or_disappear():
    good = dict(eligible=True, median_rhs_batch_seconds=1.)
    bad = dict(eligible=False, median_rhs_batch_seconds=.001)
    assert search.choose_candidate([good, bad], 0, .03) == (0, 'no_confirmed_time_improvement')
    assert search.choose_candidate([bad, good], 0, .03) == (1, 'success_recovery')
    records = [dict(repeat=0, rhs_index=0, success=True, wall_seconds=.1,
                    final_true_residual=0., stopping_threshold=1e-8, cycles=4),
               dict(repeat=0, rhs_index=1, success=False, wall_seconds=.001,
                    final_true_residual=1., stopping_threshold=1e-8, cycles=1)]
    result = search.candidate_summary(records, repeats=1, rhs_count=2)
    assert result['success_count'] == 1 and result['measurement_count'] == 2
    assert not result['eligible'] and result['median_rhs_batch_seconds'] is None


def test_confirmation_rejects_overfit_coefficients(monkeypatch, tmp_path):
    example, components, cfg = data()
    confirm = search.search_rhs(example, count=2, seed=9, partition='confirmation')
    def measured(prepared, b, config):
        value = recorded_time(prepared, b, config)
        if any(np.array_equal(b, rhs) for rhs in confirm) and np.any(prepared.coefficients != 1):
            value.update(success=False, final_true_residual=1., stop_reason='max_cycles')
        return value
    monkeypatch.setattr(search, 'PreparedLightMG', FakePrepared)
    monkeypatch.setattr(search, 'measure_raw_solve', measured)
    result = search.search_case(example, components, cfg, settings(), tmp_path/'case')
    assert not result['confirmed_improvement']
    np.testing.assert_array_equal(result['selected_coefficients'], np.ones((2, 2)))
    assert any(not row['success'] for row in result['trials'])


def test_all_failed_candidates_remain_recorded_but_cannot_be_training_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(search, 'PreparedLightMG', FakePrepared)

    def failed(prepared, b, config):
        value = recorded_time(prepared, b, config)
        value.update(success=False, final_true_residual=1., stop_reason='max_cycles')
        return value

    monkeypatch.setattr(search, 'measure_raw_solve', failed)
    result = search.search_case(*data(), settings(), tmp_path/'failed')
    assert result['status'] == 'complete'  # The search completed, the solve did not.
    assert not result['selected_candidate_success'] and not result['label_eligible']
    assert result['selection_outcome'] == 'no_successful_candidate'
    assert not result['confirmed_improvement']
    np.testing.assert_array_equal(result['selected_coefficients'], np.ones((2, 2)))
    assert result['trials'] and all(not row['success'] for row in result['trials'])
    assert search.search_case(*data(), settings(), tmp_path/'failed', resume=True) == result


def test_confirmation_success_cannot_hide_optimization_failure(monkeypatch, tmp_path):
    example, components, cfg = data()
    optimize = search.search_rhs(example, count=2, seed=9, partition='optimization')
    monkeypatch.setattr(search, 'PreparedLightMG', FakePrepared)

    def measured(prepared, b, config):
        value = recorded_time(prepared, b, config)
        if any(np.array_equal(b, rhs) for rhs in optimize):
            value.update(success=False, final_true_residual=1., stop_reason='max_cycles')
        return value

    monkeypatch.setattr(search, 'measure_raw_solve', measured)
    result = search.search_case(example, components, cfg, settings(), tmp_path/'failed')
    baseline = result['confirmation']['summaries'][0]
    assert baseline['confirmation_eligible'] and not baseline['optimization_eligible']
    assert not result['label_eligible'] and not result['selected_candidate_success']


def test_measurement_resume_keeps_completed_timings(monkeypatch, tmp_path):
    monkeypatch.setattr(search, 'PreparedLightMG', FakePrepared)
    calls = []
    def interrupt(prepared, b, cfg):
        calls.append(1)
        if len(calls) == 8:
            raise KeyboardInterrupt()
        return recorded_time(prepared, b, cfg)
    monkeypatch.setattr(search, 'measure_raw_solve', interrupt)
    with pytest.raises(KeyboardInterrupt):
        search.search_case(*data(), settings(), tmp_path/'resumed')
    state = json.loads((tmp_path/'resumed/progress.json').read_text())
    assert state['completed_measurements'] == 7
    old = deepcopy(state['trials'])
    monkeypatch.setattr(search, 'measure_raw_solve', recorded_time)
    resumed = search.search_case(*data(), settings(), tmp_path/'resumed', resume=True)
    clean = search.search_case(*data(), settings(), tmp_path/'clean')
    assert resumed['trials'][:7] == old
    assert resumed['trials'] == clean['trials']
    assert resumed['optimization_rounds'] == clean['optimization_rounds']
    assert resumed['selected_coefficients'] == clean['selected_coefficients']
    with pytest.raises(ValueError, match='changed'):
        search.search_case(*data(), {**settings(), 'seed': 10}, tmp_path/'resumed', resume=True)
    result_path = tmp_path/'resumed/result.json'
    altered = json.loads(result_path.read_text())
    altered['selected_coefficients'][0][0] = 1.1
    result_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='result changed'):
        search.search_case(*data(), settings(), tmp_path/'resumed', resume=True)


def test_no_validation_final_operators_allowed(tmp_path):
    example, components, cfg = data()
    for split in ('validation', 'final', 'ood_grid', None):
        example.research_split = split
        with pytest.raises(ValueError, match='training-split'):
            search.search_case(example, components, cfg, settings(), tmp_path/str(split))


def test_raw_tolerance_measurement_never_calls_safe_solver():
    _, _, cfg = data()
    cfg = replace(cfg, mg=replace(cfg.mg, max_cycles=3))
    class RawOnly:
        a = np.eye(2)
        def raw_cycle(self, x, b):
            return x+.9999*(b-x)
        def solve(self, *args):
            raise AssertionError('Oracle must not invoke fallback-assisted solve')
    result = search.measure_raw_solve(RawOnly(), np.ones(2), cfg)
    assert result['success'] and 2 <= result['cycles'] <= 3
    assert not result['fallback_used']
    assert len(result['residual_history']) == result['cycles']+1


def test_failed_raw_cycle_is_counted_as_executed_work():
    _, _, cfg = data()

    class FailingRaw:
        a = np.eye(2)

        def raw_cycle(self, x, b):
            raise FloatingPointError('attempted cycle failed')

    result = search.measure_raw_solve(FailingRaw(), np.ones(2), cfg)
    assert not result['success'] and result['cycles'] == 1
    assert result['completed_cycles'] == 0
    assert len(result['residual_history']) == 1
    assert result['wall_seconds'] > 0 and not result['fallback_used']


def test_real_n7_search_smoke_reuses_raw_hierarchy(tmp_path):
    torch.set_num_threads(1)
    example, _, _ = data()
    components = make_em_line_components(hidden=4, seed=2)
    cfg = AdaptiveConfig(mg=MGConfig(strategy_name='line_alt_bilinear_full',
        pre_steps=2, post_steps=2, stencil_backend='csr', max_cycles=100),
        mode='research', branch='H_P', use_smoother=False, transfer_levels='all',
        spatial=False, gate_mode='open', use_learned_controller=False)
    result = search.search_case(example, components, cfg,
        dict(grid=[1.], repeats=1, rhs_count=2, confirmation_rhs_count=1), tmp_path/'real')
    assert len(result['trials']) == 3
    assert all(row['success'] and not row['fallback_used'] for row in result['trials'])
    assert np.shape(result['features']) == (1, 12)
    assert result['selected_coefficients'] == [[1., 1.]]
    assert result['setup_seconds'] > 0 and result['offline_seconds'] > result['setup_seconds']
