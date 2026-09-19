import copy
import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_mg.provenance import operator_digest
from adaptive_mg.v67.strong import StrongRules, select_strong_strategy
import adaptive_mg.v67.research_data as data


def compact_plan(*, smoke=False):
    plan = data.make_research_plan(smoke=smoke, seed=97031)
    for spec in plan['splits'].values():
        spec.update(sizes=[7], per_family=1, count=len(spec['families']))
    return plan


def development(tmp_path, *, smoke=False):
    history = tmp_path / 'history'
    history.mkdir(exist_ok=True)
    output = tmp_path / 'research'
    rules = StrongRules()
    train, validation, manifest = data.prepare_development_data(output, rules,
        plan=compact_plan(smoke=smoke), historical_roots=[history])
    return output, rules, train, validation, manifest


def frozen(tmp_path):
    output, rules, train, validation, manifest = development(tmp_path)
    paths = {}
    for key in ('checkpoint', 'controller', 'source'):
        path = tmp_path / (key + '.bin')
        path.write_bytes(key.encode())
        paths[key] = path
    rules_path = tmp_path / 'rules.json'
    rules_path.write_text(json.dumps(rules.to_dict()))
    path = data.freeze_research(output, {'student': paths['checkpoint']}, rules_path,
                               {'temporal': paths['controller']}, [paths['source']],
                               {'tolerance': 1e-8, 'repeats': 5, 'branches': ['C', 'H_S']})
    return output, rules, path, paths


def test_plan_is_metadata_only_and_counts_full_and_smoke(monkeypatch):
    monkeypatch.setattr(data, '_assemble_case', lambda _: pytest.fail('Plan assembled an operator'))
    plan = data.make_research_plan()
    assert plan['splits']['train']['count'] == 280
    assert plan['splits']['validation']['count'] == 56
    assert sum(plan['splits'][k]['count'] for k in data.HOLDOUT) == 126
    assert plan['splits']['ood_grid']['sizes'] == [255]
    assert not set(plan['train_angles']) & set(plan['rotation_ood_angles'])
    smoke = data.make_research_plan(smoke=True)
    assert smoke['splits']['train']['count'] == 7
    assert smoke['splits']['validation']['count'] == 14


def test_development_never_assembles_or_selects_final(monkeypatch, tmp_path):
    original = data._generate_split
    calls = []
    def tracked(name, *args):
        calls.append(name)
        assert name in ('train', 'validation')
        return original(name, *args)
    monkeypatch.setattr(data, '_generate_split', tracked)
    output, rules, train, validation, manifest = development(tmp_path, smoke=True)
    assert calls == ['train', 'validation']
    assert set(e.case.family for e in train) == set(data.FAMILIES)
    assert not manifest['final_operators_materialized']
    assert not (output / 'final_data_manifest.json').exists()


def test_strong_selection_stored_exactly_and_selector_not_modified(tmp_path):
    output, rules, train, validation, manifest = development(tmp_path)
    initial = rules.digest()
    for e, record in zip(train, manifest['splits']['train']):
        selected = select_strong_strategy(e.a, e.n, rules)
        assert record['selection']['strategy_name'] == selected.strategy_name
        assert record['selection']['features'] == selected.features
        assert e.strong_selection['rules_digest'] == rules.digest()
    assert rules.digest() == initial
    restored, restored_validation, loaded = data.load_development_data(output, rules)
    assert [e.digest for e in restored] == [e.digest for e in train]
    assert [e.digest for e in restored_validation] == [e.digest for e in validation]
    for left, right in zip(train, restored):
        np.testing.assert_array_equal(left.b, right.b)
        assert left.strong_selection == right.strong_selection


def test_history_original_and_new_local_runs_are_excluded(tmp_path):
    plan = compact_plan()
    initial = data._example(data._specification('train', plan['splits']['train'], 7, 'near_isotropic', 0, 0))
    initial2 = data._example(data._specification('validation', plan['splits']['validation'], 7, 'channel', 0, 0))
    historical = tmp_path / 'historical'
    (historical / 'original').mkdir(parents=True)
    (historical / 'user_new_run').mkdir()
    (historical / 'original' / 'split_manifest.json').write_text(json.dumps({'splits': {'audit': [initial.manifest()]}}))
    (historical / 'user_new_run' / 'raw_results.json').write_text(json.dumps([{'example': initial2.manifest()}]))
    train, validation, manifest = data.prepare_development_data(tmp_path / 'output', StrongRules(), plan=plan, historical_roots=[historical])
    assert initial.group_digest not in {e.group_digest for e in train + validation}
    assert initial2.group_digest not in {e.group_digest for e in train + validation}
    assert manifest['rejected_normalized_duplicates'] == {'train': 1, 'validation': 1}
    assert manifest['historical_index']['count'] == 2
    assert operator_digest(initial.a * 100., scale_invariant=True) == initial.group_digest


def test_historical_scan_skips_arrays_repeat_records_and_own_output(tmp_path):
    excluded = tmp_path / 'own'
    excluded.mkdir()
    (excluded / 'run_manifest.json').write_text('invalid on purpose')
    repeats = tmp_path / 'repeat_records'
    repeats.mkdir()
    (repeats / 'fake_manifest.json').write_text('invalid on purpose')
    (tmp_path / 'checkpoint.pt').write_text('not JSON')
    index = data.historical_operator_index([tmp_path], exclude=[excluded])
    assert index['count'] == 0
    (tmp_path / 'run_manifest.json').write_text('invalid on purpose')
    with pytest.raises(ValueError, match='historical provenance'):
        data.historical_operator_index([tmp_path], exclude=[excluded])


def test_stored_duplicate_and_changed_selector_rejected(tmp_path):
    output, rules, train, validation, manifest = development(tmp_path)
    with pytest.raises(ValueError, match='selector changed'):
        data.load_development_data(output, rules.replace_strategies({'near_isotropic': 'chebyshev_bilinear_full'}))
    altered = copy.deepcopy(manifest)
    altered['splits']['validation'][0] = altered['splits']['train'][0]
    (output / 'development_manifest.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='leakage'):
        data.load_development_data(output, rules)


def test_plan_changed_or_overwrite_rejected(tmp_path):
    output, rules, *_ = development(tmp_path)
    with pytest.raises(FileExistsError):
        data.prepare_development_data(output, rules)
    plan = json.loads((output / 'dataset_plan.json').read_text())
    plan['splits']['final']['seed'] += 1
    (output / 'dataset_plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Plan'):
        data.load_development_data(output, rules)


def test_smoke_cannot_freeze_or_materialize_final(tmp_path):
    output, rules, *_ = development(tmp_path, smoke=True)
    with pytest.raises(ValueError, match='Smoke'):
        data.freeze_research(output, {}, tmp_path / 'rules', {}, [], {})
    with pytest.raises(ValueError, match='active'):
        data.materialize_final_data(output, None, rules)


@pytest.mark.parametrize('key', ['checkpoint', 'controller', 'source'])
def test_freeze_detects_changed_runtime_input(tmp_path, key):
    output, rules, freeze, paths = frozen(tmp_path)
    paths[key].write_bytes(b'changed after freeze')
    with pytest.raises(ValueError, match='Frozen input changed'):
        data.claim_final_evaluation(output, freeze, {'repeats': 5})
    assert not (output / 'final_claim.json').exists()


def test_final_requires_freeze_and_only_identical_interrupted_resume(tmp_path):
    output, rules, freeze, paths = frozen(tmp_path)
    invocation = {'repeats': 5, 'branches': ['C', 'H_S'], 'final_command': True}
    first = data.claim_final_evaluation(output, freeze, invocation)
    with pytest.raises(RuntimeError, match='process lease'):
        data.claim_final_evaluation(output, freeze, invocation, resume=True)
    first.close()
    with pytest.raises(ValueError, match='identical interrupted'):
        data.claim_final_evaluation(output, freeze, invocation)
    with pytest.raises(ValueError, match='identical interrupted'):
        data.claim_final_evaluation(output, freeze, {**invocation, 'repeats': 6}, resume=True)
    with data.claim_final_evaluation(output, freeze, invocation, resume=True) as claim:
        assert claim.fingerprint == first.fingerprint


def test_final_disjoint_reconstructible_and_completed_single_use(tmp_path):
    output, rules, freeze, paths = frozen(tmp_path)
    invocation = {'repeats': 5}
    with data.claim_final_evaluation(output, freeze, invocation) as claim:
        splits, manifest = data.materialize_final_data(output, claim, rules)
        development = json.loads((output / 'development_manifest.json').read_text())
        seen = set(data._digests(development['splits']))
        for name, examples in splits.items():
            assert name in data.HOLDOUT
            for e in examples:
                assert e.group_digest not in seen
                seen.add(e.group_digest)
        restored, again = data.materialize_final_data(output, claim, rules)
        assert manifest == again
        assert {k: [e.digest for e in v] for k, v in restored.items()} == {k: [e.digest for e in v] for k, v in splits.items()}
        result = output / 'complete_results.json'
        result.write_text('{"test": true}')
        ledger = data.complete_final_evaluation(output, claim, [result])
        assert ledger['status'] == 'completed'
    with pytest.raises(ValueError, match='single-use'):
        data.claim_final_evaluation(output, freeze, invocation, resume=True)


def test_final_detects_new_user_run_after_development_commit(tmp_path):
    output, rules, freeze, paths = frozen(tmp_path)
    plan = json.loads((output / 'dataset_plan.json').read_text())
    candidate = data._example(data._specification('final', plan['splits']['final'], 7, 'near_isotropic', 0, 0))
    (tmp_path / 'history' / 'late_manifest.json').write_text(json.dumps({'cases': [candidate.manifest()]}))
    with data.claim_final_evaluation(output, freeze, {'repeats': 5}) as claim:
        splits, manifest = data.materialize_final_data(output, claim, rules)
        assert candidate.group_digest not in {e.group_digest for e in splits['final']}
        assert manifest['rejected_normalized_duplicates']['final'] == 1


@pytest.mark.parametrize('family', data.FAMILIES[3:])
def test_ood_topology_changes_operator_preserves_sparse_spd(family):
    plan = compact_plan()
    record = data._specification('ood_topology', plan['splits']['ood_topology'], 7, family, 0, 0)
    changed = data._example(record)
    standard = copy.deepcopy(record)
    standard['case']['topology'] = 'standard'
    baseline = data._example(standard)
    assert changed.group_digest != baseline.group_digest
    assert changed.a.nnz <= 9 * changed.n * changed.n
    assert np.isfinite(changed.a.data).all()
    # n=7 exact eigentest only; generator always uses sparse assembly.
    assert np.linalg.eigvalsh(changed.a.toarray()).min() > 0


def test_ood_parameter_protocol_is_meaningfully_disjoint():
    plan = compact_plan()
    for split, family in [('ood_rotation', 'rotated_anisotropy'), ('ood_anisotropy', 'aligned_anisotropy'), ('ood_contrast', 'channel')]:
        record = data._specification(split, plan['splits'][split], 7, family, 0, 0)
        case = record['case']
        if split == 'ood_rotation':
            assert case['angle_deg'] in data.OOD_ANGLES and case['angle_deg'] not in data.TRAIN_ANGLES
        elif split == 'ood_anisotropy':
            assert case['epsilon'] < 1e-3
        else:
            assert case['contrast'] > 1e3


def test_split_seed_and_count_validation():
    plan = compact_plan()
    plan['splits']['final']['seed'] = plan['splits']['train']['seed']
    with pytest.raises(ValueError, match='independent seed'):
        data._validate_plan(plan)
    plan = compact_plan()
    plan['splits']['train']['count'] += 1
    with pytest.raises(ValueError, match='count'):
        data._validate_plan(plan)
