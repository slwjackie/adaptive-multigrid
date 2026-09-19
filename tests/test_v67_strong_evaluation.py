"""Failure-aware comparison, independent accuracy evidence, and repeat integrity."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from adaptive_mg.provenance import json_safe
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.limited import vector_digest
from adaptive_mg.v67.strong import StrongRules, classical_bank
from adaptive_mg.v67 import strong_evaluation as module


def result(seconds=2., success=True, **updates):
    cfg = AdaptiveConfig()
    run = dict(converged=success, wall_seconds=seconds, cycles=3, executed_cycles=3,
        stopping_threshold=cfg.mg.tolerance, final_true_residual=cfg.mg.tolerance / 2 if success else 1.,
        recomputed_true_residual=cfg.mg.tolerance / 2 if success else 1.,
        residual_history=[1., cfg.mg.tolerance / 2 if success else 1.],
        relative_residual_history=[1., cfg.mg.tolerance / 2 if success else 1.],
        stats={}, setup_stats={}, relative_solution_error=1e-9, relative_energy_error=1e-9)
    return dict(run, **updates)


def arms(oracle=False):
    cfg = AdaptiveConfig()
    out = {name: dict(config=cfg, branch='H_S' if 'H_S' in name else 'C',
                      checkpoint_id=name, selector=name.startswith('strong'), oracle=False)
           for name in ('fixed_C', 'strong_C', 'strong_H_S')}
    if oracle:
        for s in ('jacobi_bilinear_full', 'line_alt_bilinear_full'):
            out['oracle::' + s] = dict(config=replace(cfg, mg=replace(cfg.mg, strategy_name=s)),
                branch='C', checkpoint_id='pure_classical', selector=False, oracle=True)
    return out


def row(name='a', oracle=False):
    runs = {name: [result(seconds=seconds)] for name, seconds in (
        ('fixed_C', 4.), ('strong_C', 2.), ('strong_H_S', 1.))}
    if oracle:
        runs['oracle::jacobi_bilinear_full'] = [result(.001, success=False)]
        runs['oracle::line_alt_bilinear_full'] = [result(1.5)]
    return dict(example=dict(name=name, case={'n': 7}, normalized_operator_digest=name), runs=runs)


@pytest.mark.parametrize('change', [
    {'converged': False}, {'final_true_residual': 1.}, {'stopping_threshold': 2.},
    {'executed_cycles': 100000}, {'wall_seconds': 0.}, {'wall_seconds': float('nan')},
    {'recomputed_true_residual': 1.}, {'recomputed_true_residual': None},
    {'recomputed_true_residual': -1.}, {'recomputed_true_residual': float('nan')},
])
def test_failure_or_false_accuracy_never_becomes_speedup(change):
    item = row()
    item['runs']['strong_H_S'] = [result(1.), result(.001, **change)]
    comparison = module.aggregate_strong_comparison([item], arms())
    candidate = next(r for r in comparison['table'] if r['arm'] == 'strong_H_S')
    assert not candidate['success']
    assert candidate['successful_time_to_tolerance_seconds'] is None
    assert candidate['speedup_vs_fixed_C'] is None
    assert candidate['speedup_vs_strong_C'] is None
    assert comparison['summary']['strong_H_S']['new_failure_case_ids']['strong_C'] == ['a']


def test_missing_independent_residual_check_is_not_success():
    item = row()
    item['runs']['strong_H_S'][0].pop('recomputed_true_residual')
    comparison = module.aggregate_strong_comparison([item], arms())
    assert comparison['summary']['strong_H_S']['successes'] == 0
    assert comparison['summary']['strong_H_S']['comparisons']['strong_C']['cases'] == 0


@pytest.mark.parametrize('change', [{'spatial': True}, {'gate_mode': 'closed'}, {'inference_dtype': 'float64'}, {'native_threads': 2}])
def test_different_policy_or_runtime_contract_rejected(tmp_path, monkeypatch, change):
    examples = dataset([7], 7, 13)[:1]
    specs = module.build_strong_arms(tmp_path, AdaptiveConfig(), branches=())
    cfg = specs['strong_C']['config']
    specs['strong_C']['config'] = (replace(cfg, mg=replace(cfg.mg, **change))
                                    if 'native_threads' in change else replace(cfg, **change))
    monkeypatch.setattr(module, 'measured_strong', lambda *args: pytest.fail('must reject before any solve'))
    with pytest.raises(ValueError, match='All arms must share'):
        module.run_strong_benchmark(examples, {}, specs, tmp_path / 'out', tmp_path / 'research')


def test_oracle_never_selects_fast_failed_strategy():
    comparison = module.aggregate_strong_comparison([row(oracle=True)], arms(oracle=True))
    oracle = comparison['offline_oracle'][0]
    assert oracle['selected_strategy'] == 'line_alt_bilinear_full'
    assert oracle['search_total_seconds'] == pytest.approx(1.501)
    candidate = next(r for r in comparison['table'] if r['arm'] == 'strong_H_S')
    assert candidate['speedup_vs_fixed_C'] == 4.
    assert candidate['speedup_vs_strong_C'] == 2.
    assert candidate['speedup_vs_oracle'] == 1.5
    assert not comparison['final_certificate_modified']


def test_global_common_cohort_differs_from_pairwise_without_cherry_picking():
    a, b = row('a'), row('b')
    b['runs']['fixed_C'] = [result(.01, success=False)]
    comparison = module.aggregate_strong_comparison([a, b], arms())
    assert comparison['common_cohort_case_ids'] == ['a']
    pair = comparison['summary']['strong_H_S']['comparisons']['strong_C']
    assert pair['cases'] == 2
    assert pair['common_cohort']['cases'] == 1
    assert not comparison['summary']['strong_H_S']['diagnostic_superiority_eligible']


def test_inaccurate_solution_blocks_time_and_accuracy_superiority():
    rows = []
    for i in range(20):
        item = row(str(i))
        for name, runs in item['runs'].items():
            item['runs'][name] = [deepcopy(runs[0]) for _ in range(5)]
        for run in item['runs']['strong_H_S']:
            run.update(stats={'neural_apply_calls': 10, 'accepted_neural_cycles': 2}, relative_solution_error=1e-4)
        rows.append(item)
    comparison = module.aggregate_strong_comparison(rows, arms())
    candidate = comparison['summary']['strong_H_S']
    assert candidate['comparisons']['strong_C']['ci95'][0] == 2.
    assert len(candidate['accuracy_worse_vs_strong_case_ids']) == 20
    assert not candidate['diagnostic_superiority_eligible']
    for item in rows:
        for run in item['runs']['strong_H_S']:
            run['relative_solution_error'] = 1e-9
    comparison = module.aggregate_strong_comparison(rows, arms())
    assert comparison['summary']['strong_H_S']['diagnostic_superiority_eligible']
    assert not comparison['summary']['strong_H_S']['final_time_improvement_certified']


def test_small_or_no_actual_neural_evidence_never_superior():
    comparison = module.aggregate_strong_comparison([row()], arms(), min_cases=1)
    checks = comparison['summary']['strong_H_S']['diagnostic_superiority_checks']
    assert not checks['actual_accepted_neural_work']
    assert not checks['enough_repeats']
    assert not comparison['summary']['strong_H_S']['diagnostic_superiority_eligible']


def test_protected_original_and_limited_outputs(tmp_path):
    research = tmp_path / 'my_v6_7_research'
    for target in (research, research / 'child', tmp_path, tmp_path / 'my_v6_7_limited_fix' / 'new'):
        with pytest.raises(ValueError):
            module.strong_output(target, research)
    assert module.strong_output(tmp_path / 'my_v6_7_strong_baseline', research).is_dir()


def test_fresh_excludes_scale_normalized_original_operators(tmp_path):
    original = dataset([7], 7, 123)
    manifest = {'splits': {'train': [original[0].manifest()], 'tune': [], 'audit': []}}
    (tmp_path / 'split_manifest.json').write_text(json.dumps(manifest))
    examples, evidence = module.strong_cases(tmp_path, {'seed': 1}, 'fresh', sizes=[7], seed=123)
    assert len(examples) == 6
    assert evidence['excluded_normalized_operators'] == 1
    assert original[0].group_digest not in {e.group_digest for e in examples}
    assert not evidence['independent_certification_evidence']


def test_seven_family_default_and_exact_train_reconstruction(tmp_path):
    settings = {'seed': 1, 'train_sizes': [7], 'train_per_size': 7}
    original = dataset([7], 7, 1)
    manifest = {'splits': {'train': [e.manifest() for e in original], 'tune': [], 'audit': []}}
    (tmp_path / 'split_manifest.json').write_text(json.dumps(manifest))
    examples, evidence = module.strong_cases(tmp_path, settings, sizes=[7])
    assert len(examples) == 7
    assert len(evidence['family_coverage']) == 7
    train, _ = module.strong_cases(tmp_path, settings, 'train')
    assert [e.digest for e in train] == [e.digest for e in original]
    manifest['splits']['train'][0]['operator_digest'] = 'changed'
    (tmp_path / 'split_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='stored manifest'):
        module.strong_cases(tmp_path, settings, 'train')


def fake_measure(example, arm, rules):
    selection = dict(strategy_name=arm['config'].mg.strategy_name, rule_id='test',
                     rules_digest=rules.digest(), selector_seconds=.01) if arm['selector'] else None
    return dict(result(), measurement_config=arm['config'].to_dict(), operator_digest=example.digest,
        rhs_digest=vector_digest(example.b), x0_digest=vector_digest(np.zeros_like(example.b)),
        checkpoint_id=arm['checkpoint_id'], requested_branch=arm['branch'],
        strategy_name=arm['config'].mg.strategy_name, selection=selection)


def test_per_repeat_resume_reuses_completed_runs_and_rejects_rule_changes(tmp_path, monkeypatch):
    examples = dataset([7], 7, 13)[:1]
    arm_specs = module.build_strong_arms(tmp_path, AdaptiveConfig(), branches=())
    out = tmp_path / 'output'
    calls = []
    def interrupted(example, arm, rules):
        calls.append(arm['selector'])
        if len(calls) == 3:
            raise KeyboardInterrupt
        return fake_measure(example, arm, rules)
    monkeypatch.setattr(module, 'measured_strong', interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.run_strong_benchmark(examples, {}, arm_specs, out, tmp_path / 'research', repeats=2, warmups=0)
    assert len(list((out / 'repeats' / '0000').glob('0*_0*.json'))) == 2
    resumed_calls = []
    def resumed(example, arm, rules):
        resumed_calls.append(arm['selector'])
        return fake_measure(example, arm, rules)
    monkeypatch.setattr(module, 'measured_strong', resumed)
    comparison = module.run_strong_benchmark(examples, {}, arm_specs, out, tmp_path / 'research', repeats=2, warmups=0, resume=True)
    assert len(resumed_calls) == 2
    assert comparison['summary']['strong_C']['successes'] == 1
    with pytest.raises(ValueError, match='signature mismatch'):
        module.run_strong_benchmark(examples, {}, arm_specs, out, tmp_path / 'research',
            StrongRules(anisotropy_min=5), repeats=2, warmups=0, resume=True)
    file = sorted((out / 'repeats' / '0000').glob('0*_0*.json'))[0]
    value = json.loads(file.read_text())
    value['run']['wall_seconds'] = .00001
    file.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='hash mismatch'):
        module.run_strong_benchmark(examples, {}, arm_specs, out, tmp_path / 'research', repeats=2, warmups=0, resume=True)


def test_complete_oracle_bank_is_really_measured_and_costs_recorded(tmp_path, monkeypatch):
    examples = dataset([7], 7, 13)[:1]
    specs = module.build_strong_arms(tmp_path, AdaptiveConfig(), branches=())
    calls = []
    def measured(example, arm, rules):
        calls.append(arm['config'].mg.strategy_name)
        return fake_measure(example, arm, rules)
    monkeypatch.setattr(module, 'measured_strong', measured)
    comparison = module.run_strong_benchmark(examples, {}, specs, tmp_path / 'out', tmp_path / 'research',
                                            repeats=2, warmups=1, oracle_bank='controlled')
    assert len(calls) == (2 + len(classical_bank('controlled'))) * 3
    oracle = comparison['offline_oracle'][0]
    assert oracle['tested_strategies'] == 16
    assert oracle['search_total_seconds'] == 16 * 2 * 2.
    assert oracle['search_warmup_seconds'] == 16 * 2.
    assert oracle['search_total_including_warmups_seconds'] == 16 * 3 * 2.


def test_real_measurement_records_recomputed_accuracy_and_selected_hierarchy():
    torch.set_num_threads(1)
    example = dataset([7], 7, 77)[0]
    cfg = AdaptiveConfig()
    arm = module.build_strong_arms('.', cfg, branches=())['strong_C']
    run = module.measured_strong(example, arm, StrongRules())
    assert run['selector_in_wall_time']
    assert run['selection']['selector_seconds'] > 0
    assert run['wall_seconds'] >= run['selection']['selector_seconds']
    assert run['hierarchy_dimensions'][0]['operator_shape'] == [49, 49]
    assert run['relative_solution_error'] >= 0
    assert run['relative_energy_error'] >= 0
    assert run['recomputed_true_residual'] == pytest.approx(run['final_true_residual'])
    assert run['stats']['selector_calls'] == 1
    assert run['stats']['nn_forward_calls'] == 0
