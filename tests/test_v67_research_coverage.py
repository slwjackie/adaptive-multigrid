"""Coverage wiring uses fake final cohorts; it never opens a real held-out set."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from adaptive_mg import MGConfig
from adaptive_mg.v67 import AdaptiveConfig
from adaptive_mg.v67.research_data import make_research_plan, _example, _specification
from adaptive_mg.v67.research_policy import ResearchPolicy, _scope
from adaptive_mg.v67.strong import StrongRules
from adaptive_mg.provenance import hardware_environment
import adaptive_mg.v67.research_evaluation as evaluation
import adaptive_mg.v67.research_pipeline as pipeline
import adaptive_mg.v67.research_certification as certification


def _fixture():
    cfg = AdaptiveConfig(mg=MGConfig(mode='classical', max_cycles=30, coarsest_n=3,
        nn_levels=1, tolerance=1e-7, stencil_backend='csr'), mode='research', branch='auto',
        spatial=False, gate_mode='open')
    rules = StrongRules()
    plan = make_research_plan(smoke=True, seed=698133)
    e = _example(_specification('validation', plan['splits']['validation'], 7, 'near_isotropic', 0, 0))
    e.research_split = 'validation'
    return e, cfg, rules


def test_rhs_blind_changes_hint_and_cache_key_but_runs_actual_batch(monkeypatch):
    e, cfg, rules = _fixture()
    policy = ResearchPolicy({}, {}, rules.digest(), _scope(cfg), hardware_environment(), {})
    hints = []
    def choose(*args, **kwargs):
        hints.append(kwargs['expected_rhs'])
        return 'C', {'expected_rhs': kwargs['expected_rhs'], 'fixture': True}
    monkeypatch.setattr(policy, 'choose', choose)
    rhs, _ = evaluation.manufactured_rhs(e, 4)
    actual = evaluation._construct(e, dict(branch='auto', policy=policy), cfg, rules, 4)
    results_actual = actual.solve_many(rhs)
    assert hints[-1] == 4
    blind = evaluation._construct(e, dict(branch='auto', policy=policy, expected_rhs_mode='blind'), cfg, rules, 4)
    results_blind = blind.solve_many(rhs)
    assert hints[-1] == 1
    assert len(results_actual) == len(results_blind) == 4
    for left, right in zip(results_actual, results_blind):
        np.testing.assert_array_equal(left.x, right.x)
    assert actual.cache_key != blind.cache_key
    assert evaluation._model_metadata(dict(branch='auto', policy=policy, expected_rhs_mode='blind'))['expected_rhs_mode'] == 'blind'


def test_offline_reference_reuses_actual_classical_bank_and_keeps_search_cost(monkeypatch, tmp_path):
    e, cfg, rules = _fixture()
    called = {}
    def bank_run(examples, provenance, arms, out, research, supplied_rules, **kwargs):
        called.update(examples=examples, provenance=provenance, arms=arms, kwargs=kwargs)
        assert supplied_rules.digest() == rules.digest()
        return dict(table=[dict(case_id=e.name, arm=name, n=7, success=True,
            successful_time_to_tolerance_seconds=seconds, strategy_names=['jacobi_bilinear_full'])
            for name, seconds in [('fixed_C', 1.), ('strong_C', .5)]],
            offline_oracle=[dict(case_id=e.name, selected_arm='oracle::jacobi_bilinear_full',
                selected_strategy='jacobi_bilinear_full', successful_median_seconds=.25,
                search_total_including_warmups_seconds=8.75)])
    monkeypatch.setattr(evaluation, 'run_strong_benchmark', bank_run)
    result = evaluation.evaluate_classical_reference([e], cfg, rules, tmp_path / 'bank', research=tmp_path / 'old', repeats=5)
    assert set(called['arms']) == {'fixed_C', 'strong_C'}
    assert all(arm['model'] is None for arm in called['arms'].values())
    assert called['kwargs']['oracle_bank'] == 'controlled'
    assert not called['provenance']['policy_selection_used']
    row = next(r for r in result['table'] if r['arm'] == 'offline_best_classical')
    assert row['median_wall_seconds'] == .25
    assert row['offline_search_seconds'] == 8.75
    assert row['speedup_vs_strong'] == 2.
    assert result['reference_only'] and not result['certification_used']
    e.research_split = 'final'
    with pytest.raises(ValueError, match='active final-evaluation'):
        evaluation.evaluate_classical_reference([e], cfg, rules, tmp_path / 'forbidden', research=tmp_path / 'old')


def _checkpoint_tree(out, research):
    for name in ('ordinary_direct', 'compact_direct', 'd4_direct', 'o2_direct', 'p_cnn_direct', 'gnn_direct', 'edge_direct'):
        path = out / 'checkpoints' / name / 'candidate.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'fixture only')
    for stage in ('smoother', 'transfer', 'joint'):
        path = research / stage / 'candidate.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'historical fixture only')


def test_architecture_inventory_has_all_controls_and_only_created_kd(tmp_path):
    out, research = tmp_path / 'run', tmp_path / 'old'
    _checkpoint_tree(out, research)
    specs = pipeline._architecture_checkpoint_specs(out, research)
    assert {'ordinary_CNN_H_S', 'D4_CNN_H_S', 'O2_steerable_H_S', 'small_GNN_H_P', 'edge_MLP_H_P', 'existing_CNN_P_H_P'} <= set(specs)
    assert specs['old_fixed_H_S']['selector'] is False
    assert specs['old_strong_H_S']['selector'] is True
    assert not any('distilled' in key for key in specs)
    path = out / 'checkpoints/s_distilled/candidate.pt'
    path.parent.mkdir()
    path.write_bytes(b'created student fixture')
    assert 'distilled_smoother_H_S' in pipeline._architecture_checkpoint_specs(out, research)


def test_final_wiring_freezes_every_architecture_and_separates_reference(monkeypatch, tmp_path):
    project, out = tmp_path / 'project', tmp_path / 'project/artifacts/run'
    original = project / 'artifacts/original'
    _checkpoint_tree(out, original)
    (out / 'selected_students.json').write_text(json.dumps({'branches': {b: str(out/'selected_students'/(b+'.pt')) for b in ('H_S', 'H_P', 'H_SP')}}))
    (out / 'policy/deployable_policy').mkdir(parents=True)
    (out / 'policy/deployable_policy/policy.json').write_text('{}')
    (out / 'frozen_rules.json').write_text('{}')
    settings = dict(base_research='artifacts/original', final_repeats=5, final_min_cases=20, warmups=1, smoke=False)
    _, cfg, rules = _fixture()
    policy = SimpleNamespace(digest=lambda: 'frozen-policy')
    monkeypatch.setattr(pipeline.ResearchPolicy, 'load', lambda path: policy)
    frozen = {}
    def freeze(root, checkpoints, rule_path, controllers, sources, config):
        frozen.update(checkpoints=checkpoints, config=config)
        path = root / 'research_freeze.json'
        path.write_text(json.dumps({'config': config}))
        return path
    monkeypatch.setattr(pipeline, 'freeze_research', freeze)
    pipeline._frozen_final(project, out, settings, cfg, rules, 'freeze', False)
    assert frozen['config']['architecture_evaluation']['regimes'] == ['cold']
    assert frozen['config']['classical_reference']['bank'] == 'controlled'
    assert 'architecture:O2_steerable_H_S' in frozen['checkpoints']
    assert 'architecture:old_strong_H_P' in frozen['checkpoints']
    @contextmanager
    def claim(*args, **kwargs):
        yield SimpleNamespace(fingerprint='fixture-claim')
    monkeypatch.setattr(pipeline, 'claim_final_evaluation', claim)
    # Fake iterable markers: no held-out PDE is generated/inspected in this test.
    cohorts = {'final': ['sealed-final-marker'], 'ood_rotation': ['sealed-rotation-marker']}
    monkeypatch.setattr(pipeline, 'materialize_final_data', lambda *args: (cohorts, {}))
    monkeypatch.setattr(pipeline.Components, 'load', lambda path: 'frozen-model:' + str(path))
    calls, bank_calls, certificates, sealed = [], [], [], []
    def evaluate(examples, arms, config, supplied_rules, target, **kwargs):
        calls.append(dict(examples=examples, arms=arms, target=target, kwargs=kwargs))
        return {'cohort': examples, 'target': str(target)}
    monkeypatch.setattr(pipeline, 'evaluate_research', evaluate)
    monkeypatch.setattr(pipeline, 'evaluate_classical_reference', lambda examples, *args, **kwargs: bank_calls.append((examples, kwargs)))
    monkeypatch.setattr(certification, 'certify_research_final', lambda root, claim, result, *args, **kwargs: certificates.append(result) or {'validated': False})
    monkeypatch.setattr(pipeline, 'complete_final_evaluation', lambda *args: sealed.append(args))
    pipeline._frozen_final(project, out, settings, cfg, rules, 'final', False)
    assert len(calls) == 4
    for entry in calls:
        if entry['target'].parent.name == 'final_architecture_comparison':
            assert entry['kwargs']['regimes'] == ['cold']
            assert {'ordinary_CNN_H_S', 'D4_CNN_H_S', 'O2_steerable_H_S'} <= set(entry['arms'])
        else:
            assert entry['kwargs']['regimes'] == ['cold', 'warm', 'multiple']
            assert entry['arms']['adaptive_rhs_blind']['expected_rhs_mode'] == 'blind'
    assert len(bank_calls) == 1 and bank_calls[0][0] == cohorts['final']
    assert certificates[0]['target'].endswith('/final_evaluation/final')
    assert len(sealed) == 1
    assert any(str(path).endswith('/final_classical_reference/offline_reference.json') for path in sealed[0][2])


def test_validation_baseline_reference_is_separate_from_architecture_selection(monkeypatch, tmp_path):
    _, cfg, rules = _fixture()
    examples = [SimpleNamespace(n=15, name='validation15'), SimpleNamespace(n=31, name='validation31')]
    settings = {'smoke': True, 'validation_repeats': 3, 'warmups': 1}
    run = pipeline.DevelopmentRun(tmp_path, settings, tmp_path / 'out', cfg, rules,
        tmp_path / 'old', [], examples, {}, False)
    monkeypatch.setattr(pipeline.Components, 'load', lambda path: object())
    monkeypatch.setattr(run, 'evaluate', lambda *args, **kwargs: None)
    calls = []
    def reference(examples, *args, **kwargs):
        calls.append((examples, kwargs))
        return {'reference_only': True}
    monkeypatch.setattr(pipeline, 'evaluate_classical_reference', reference)
    run.phase_baseline()
    assert [e.n for e in calls[0][0]] == [15]
    assert calls[0][1]['bank'] == 'controlled'
    assert run.results['offline_classical_reference']['reference_only']
    assert pipeline.PHASES.index('baseline') < pipeline.PHASES.index('direct')
