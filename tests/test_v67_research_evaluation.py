from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from adaptive_mg import MGConfig
from adaptive_mg.v67 import AdaptiveConfig, Components
from adaptive_mg.v67.strong import StrongRules
from adaptive_mg.v67.research_data import make_research_plan, _example, _specification
import adaptive_mg.v67.research_evaluation as evaluation
from adaptive_mg.v67.research_runtime import _OFFLINE_TEACHER


def example():
    plan = make_research_plan(smoke=True, seed=680139)
    e = _example(_specification('validation', plan['splits']['validation'], 7, 'near_isotropic', 0, 0))
    e.research_split = 'validation'
    e.research_family = 'near_isotropic'
    return e


def config():
    return AdaptiveConfig(mg=MGConfig(mode='classical', tolerance=1e-7, absolute_tolerance=1e-12,
        max_cycles=60, coarsest_n=3, nn_levels=1, pre_steps=2, post_steps=2,
        near_tolerance_factor=0., stencil_backend='csr'), mode='research', branch='H_S',
        spatial=False, gate_mode='open', inference_device='cpu', inference_dtype='float32')


def arms():
    return {'strong_C': dict(model=None, branch='C'),
            'student': dict(model=Components.create(hidden=4, seed=661), branch='H_S')}


def test_rhs_are_actual_distinct_manufactured_vectors_with_shared_prefix():
    e = example()
    b4, x4 = evaluation.manufactured_rhs(e, 4)
    b16, x16 = evaluation.manufactured_rhs(e, 16)
    np.testing.assert_array_equal(b4, b16[:4])
    np.testing.assert_array_equal(x4, x16[:4])
    np.testing.assert_array_equal(b4[0], e.b)
    assert len({x.tobytes() for x in x16}) == 16
    for b, x in zip(b16, x16):
        np.testing.assert_array_equal(b, e.a @ x)


def test_real_cold_warm_and_batches_have_correct_cache_and_cost_scope(tmp_path):
    result = evaluation.evaluate_research([example()], arms(), config(), StrongRules(), tmp_path,
        repeats=2, warmups=0, rhs_counts=(1, 4), regimes=('cold', 'warm', 'multiple'))
    row = result['rows'][0]
    student = row['runs']['student']
    assert student['cold']['1'][0]['success']
    assert student['cold']['1'][0]['classical_setup_seconds'] > 0
    warm = student['warm']['1'][0]
    assert warm['success'] and warm['warm_prime_seconds'] > 0
    assert warm['classical_setup_seconds'] == 0
    assert warm['counters']['nn_forward_calls'] == 0
    assert warm['warm_prime_counters']['nn_forward_calls'] == 1
    for count in ('1', '4'):
        measured = student['multiple'][count][0]
        assert measured['rhs_count'] == int(count)
        assert len(measured['rhs_results']) == int(count)
        assert measured['successful']
        assert measured['counters']['nn_forward_calls'] == 1
        assert measured['wall_seconds'] > measured['constructor_seconds']
        assert measured['seconds_per_rhs'] == measured['wall_seconds'] / int(count)
        assert measured['operator_complexity'] >= 1
        assert measured['inference_dtype'] == 'float32' and measured['numerical_dtype'] == 'float64'
    summary = result['summary']['cold']['1']['student']
    assert summary['successes'] == summary['total'] == 1
    assert summary['geometric_speedup'] > 0
    assert len(summary['ci95']) == 2
    assert result['manifest']['arms']['student']['parameter_count'] > 0
    assert (tmp_path / 'comparison.csv').is_file()
    assert (tmp_path / 'raw_results.json').is_file()


def test_failed_rhs_cannot_become_fast_batch_success(monkeypatch):
    original = evaluation._construct
    def corrupt(*args, **kwargs):
        prepared = original(*args, **kwargs)
        solve_many = prepared.solve_many
        def run(*args, **kwargs):
            results = solve_many(*args, **kwargs)
            results[-1].x += 1.
            return results
        prepared.solve_many = run
        return prepared
    monkeypatch.setattr(evaluation, '_construct', corrupt)
    arm = arms()['strong_C']
    run = evaluation.measured_research(example(), arm, evaluation._arm_config(config(), arm), StrongRules(), regime='multiple', rhs_count=4)
    assert not run['success']
    assert run['wall_seconds'] > 0
    assert run['rhs_results'][-1]['converged']
    assert not run['rhs_results'][-1]['verified_success']


def test_teacher_guard_and_explicit_offline_cost(monkeypatch, tmp_path):
    specs = arms()
    specs['teacher'] = specs.pop('student')
    specs['teacher']['model'].smoother.training_only = True
    with pytest.raises(ValueError, match='training_only'):
        evaluation.evaluate_research([example()], specs, config(), StrongRules(), tmp_path / 'forbidden', repeats=1, warmups=0, regimes=('cold',))
    specs['teacher']['training_only'] = True
    original = evaluation._construct
    flags = []
    def observe(e, arm, *args):
        if arm.get('training_only'):
            flags.append(_OFFLINE_TEACHER.get())
        return original(e, arm, *args)
    monkeypatch.setattr(evaluation, '_construct', observe)
    result = evaluation.evaluate_research([example()], specs, config(), StrongRules(), tmp_path / 'offline', repeats=1, warmups=0, regimes=('cold',))
    assert flags == [True]
    run = result['rows'][0]['runs']['teacher']['cold']['1'][0]
    assert run['training_only']
    assert run['teacher_evaluation_seconds'] == run['wall_seconds']
    assert run['teacher_deployment_seconds'] == 0
    assert not _OFFLINE_TEACHER.get()


def test_resume_uses_records_without_new_solves_and_rejects_drift(monkeypatch, tmp_path):
    specs = arms()
    cfg, rules, e = config(), StrongRules(), example()
    result = evaluation.evaluate_research([e], specs, cfg, rules, tmp_path, repeats=1, warmups=1, regimes=('cold',))
    original_comparison = (tmp_path / 'comparison.json').read_bytes()
    monkeypatch.setattr(evaluation, 'measured_research', lambda *args, **kwargs: pytest.fail('Completed resume ran numerical work'))
    resumed = evaluation.evaluate_research([e], specs, cfg, rules, tmp_path, repeats=1, warmups=1, regimes=('cold',), resume=True)
    assert result['rows'] == resumed['rows']
    assert (tmp_path / 'comparison.json').read_bytes() == original_comparison
    with pytest.raises(ValueError, match='mismatch'):
        evaluation.evaluate_research([e], specs, replace(cfg, mg=replace(cfg.mg, tolerance=1e-6)), rules, tmp_path, repeats=1, warmups=1, regimes=('cold',), resume=True)
    record_path = next((tmp_path / 'repeat_records').glob('*.json'))
    saved = json.loads(record_path.read_text())
    saved['run']['wall_seconds'] *= .1
    record_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluation.evaluate_research([e], specs, cfg, rules, tmp_path, repeats=1, warmups=1, regimes=('cold',), resume=True)


def test_failures_excluded_from_pair_speedup_and_reported(tmp_path):
    specs = arms()
    result = evaluation.evaluate_research([example()], specs, config(), StrongRules(), tmp_path, repeats=2, warmups=0, regimes=('cold',))
    rows = copy.deepcopy(result['rows'])
    rows[0]['runs']['student']['cold']['1'][0].update(success=False, successful=False, wall_seconds=1e-12)
    report = evaluation.aggregate_research(rows, result['manifest']['arms'], repeats=2, regimes=('cold',), rhs_counts=(1,))
    student = report['summary']['cold']['1']['student']
    assert student['successes'] == 0
    assert student['geometric_speedup'] is None
    assert student['new_failure_case_ids'] == [example().name]
    assert not report['final_certificate']


def test_final_example_without_active_claim_is_rejected(tmp_path):
    e = example()
    e.research_split = 'final'
    with pytest.raises(ValueError, match='active final-evaluation'):
        evaluation.evaluate_research([e], arms(), config(), StrongRules(), tmp_path, repeats=1, regimes=('cold',))


def test_baseline_override_and_duplicate_operator_rejected(tmp_path):
    specs = arms()
    specs['strong_C']['selector'] = False
    with pytest.raises(ValueError, match='unchanged selected pure classical'):
        evaluation.evaluate_research([example()], specs, config(), StrongRules(), tmp_path, repeats=1)
    with pytest.raises(ValueError, match='unique'):
        evaluation.evaluate_research([example(), example()], arms(), config(), StrongRules(), tmp_path, repeats=1)
    specs = arms()
    specs['student']['config'] = replace(config(), mg=replace(config().mg, max_cycles=100))
    with pytest.raises(ValueError, match='share numerical protocol'):
        evaluation.evaluate_research([example()], specs, config(), StrongRules(), tmp_path, repeats=1)
