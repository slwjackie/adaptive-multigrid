"""Valid-label supervision without hiding search or evaluation failures."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil

import numpy as np
import pytest

from adaptive_mg.provenance import operator_digest
from adaptive_mg.v67 import em_line_study as em
from adaptive_mg.v67 import joint_transfer_study as joint
from adaptive_mg.v67 import light_smoother_study as study
from adaptive_mg.v67 import light_smoother_search as oracle
from adaptive_mg.v67 import light_smoother_fit as fitting
from adaptive_mg.v67 import light_smoother_evaluation as evaluation
from adaptive_mg.v67.limited import digest_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('light_subset', ROOT/'scripts/run_v6_7_light_smoother_subset.py')
subset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subset)


def hashes(folder):
    return {str(p.relative_to(folder)): digest_file(p) for p in folder.rglob('*') if p.is_file()}


@pytest.fixture(scope='module')
def parent(tmp_path_factory):
    root = tmp_path_factory.mktemp('subset_parent')
    original_plan = joint.make_research_plan
    def tiny_plan(**kwargs):
        plan = original_plan(**kwargs)
        for name in study.HOLDOUT:
            family = plan['splits'][name]['families'][0]
            plan['splits'][name].update(sizes=[7], families=[family], per_family=1, count=1)
        return plan
    # Test-only tiny final plans and one injected search failure. These records
    # are workflow fixtures, never performance evidence or research datasets.
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(joint, 'make_research_plan', tiny_plan)
        spec = subset.read(ROOT/'configs/v6_7_em_line_smoke.json')
        spec['smoke'] = False
        for stage in spec['stages'].values():
            stage['updates'] = 1
        subset.write_json(root/'em.json', spec)
        em.prepare(root/'em.json', root/'em')
        em.train(root/'em')
        config = subset.read(ROOT/'configs/v6_7_light_smoother_smoke.json')
        config['smoke'] = False
        config['search'].update(grid=[1.], repeats=1, rhs_count=1, confirmation_rhs_count=1)
        config['fit']['updates'] = 4
        subset.write_json(root/'light.json', config)
        out = root/'parent'
        study.prepare(root/'light.json', out, em_run_dir=root/'em')
        original = oracle.measure_raw_solve
        first = []
        def with_failure(prepared, b, cfg):
            if not first:
                first.append(operator_digest(prepared.a))
            row = original(prepared, b, cfg)
            if operator_digest(prepared.a) == first[0]:
                row.update(success=False, final_true_residual=1., stop_reason='max_cycles')
            return row
        monkey.setattr(oracle, 'measure_raw_solve', with_failure)
        summary = study.search(out)
        assert summary['total_cases'] == 7 and summary['eligible_cases'] == 6
    return out


def test_prepare_snapshots_all_failures_and_unchanged_splits(parent, tmp_path):
    before = hashes(parent)
    out = tmp_path/'subset'
    report = subset.main(['prepare', '--source-run', str(parent), '--run-dir', str(out)])
    assert report['total_training_operators'] == 7
    assert report['included_operators'] == 6 and report['excluded_operators'] == 1
    assert report['validation_operators'] == 7
    assert sum(v['total'] for v in report['by_family'].values()) == 7
    assert report['excluded'][0]['reason'] == 'no_successful_search_label'
    assert report['excluded'][0]['stop_reasons']['max_cycles'] > 0
    assert before == hashes(parent)
    for name, sha in subset.signed_read(out/'subset/manifest.json')['source_files'].items():
        assert sha == digest_file(parent/name) == digest_file(out/name)
    assert subset.main(['prepare', '--run-dir', str(out), '--resume']) == report
    with pytest.raises(ValueError, match='No successful search label'):
        study.fit(parent)  # The original strict path remains strict.


def test_shared_included_rows_scaler_and_exact_fit_resume(parent, tmp_path, monkeypatch):
    out, clean = tmp_path/'interrupted', tmp_path/'clean'
    subset.prepare(out, source_run=parent)
    subset.prepare(clean, source_run=parent)
    original = fitting._save_state
    fired = []
    def interrupt_after_committed_update(path, payload):
        original(path, payload)
        if payload.get('updates') == 2 and not fired:
            fired.append(1)
            raise KeyboardInterrupt()
    with monkeypatch.context() as patch:
        patch.setattr(fitting, '_save_state', interrupt_after_committed_update)
        with pytest.raises(KeyboardInterrupt):
            subset.fit(out)
    subset.fit(out, resume=True)
    subset.fit(clean)
    _, settings, cfg, selection = subset.load(out)
    x, y, provenance = subset.training_inputs(out, settings, cfg, selection)
    fitted = subset.read(out/'fit/fit_manifest.json')
    assert fitted['rows'] == len(x) == selection['included_level_rows'] == 6
    assert fitted['provenance']['operators'] == provenance['operators']
    assert not set(provenance['operators']) & set(provenance['excluded_operators'])
    assert len(provenance['excluded_operators']) == 1
    assert fitted['scaler'] == subset.Standardizer.fit(x).to_dict()
    _, _, _, baseline = study.load(out)
    training, _, _ = study._data(out, settings, baseline)
    all_features = np.asarray([subset.read(out/r['path'])['features'][0]
                              for r in subset.read(out/'search/summary.json')['rows']])
    assert not np.array_equal(np.mean(all_features, axis=0), np.mean(x, axis=0))
    for kind in ('mlp', 'regression'):
        left = subset.read(out/f'fit/{kind}.json')
        right = subset.read(clean/f'fit/{kind}.json')
        # Parent/run provenance differs, but included data and actual fitted
        # numerical weights are identical after interruption and resume.
        left.pop('metadata')
        right.pop('metadata')
        assert left == right
    saved = {k: digest_file(out/f'fit/{k}.json') for k in ('mlp', 'regression')}
    subset.fit(out, resume=True)
    assert saved == {k: digest_file(out/f'fit/{k}.json') for k in saved}
    models = subset.arms(out, settings, cfg, baseline, selection, include_online=True)
    signatures = {a['model'].expert_signature('transfer') for a in models.values() if a['model'] is not None}
    assert signatures == {subset.read(out/'import_manifest.json')['transfer_signature']}
    assert set(models) == {'fixed_C', 'em5_C', *study.BASE_ARMS, *study.ONLINE_ARMS, 'frozen_P_line'}


def test_full_validation_and_all_final_splits_with_failure_retained(parent, tmp_path, monkeypatch):
    out = tmp_path/'evaluated'
    subset.prepare(out, source_run=parent)
    subset.fit(out)
    args = ['benchmark', '--run-dir', str(out), '--include-online', '--repeats', '1',
            '--warmups', '0', '--rhs-counts', '1', '--regimes', 'cold', 'warm', 'multiple']
    original = evaluation.measured_light
    failed = []
    def with_failure(example, arm, *args, **kwargs):
        row = original(example, arm, *args, **kwargs)
        if not failed:
            failed.append(example.name)
        if (example.name == failed[0] and getattr(arm.get('predictor'), 'kind', None) == 'mlp'
                and arm.get('online') is None):
            row.update(success=False, successful=False, error='test-only validation failure')
        return row
    with monkeypatch.context() as patch:
        patch.setattr(evaluation, 'measured_light', with_failure)
        result = subset.main(args)
    assert len(result['rows']) == 7
    assert subset.read(out/'benchmarks/validation/progress.json')['completed_measurements'] == 168
    decision = subset.read(out/'benchmarks/validation/selection.json')
    assert decision['total_cases'] == 7
    assert decision['candidates']['learned_P_MLP']['failures'] == 1
    assert not decision['candidates']['learned_P_MLP']['eligible']
    assert failed[0] in decision['candidates']['learned_P_MLP']['failure_case_ids']
    before = digest_file(out/'benchmarks/validation/raw_results.json')
    replay = subset.main(args+['--resume'])
    assert result['summary'] == replay['summary']
    assert before == digest_file(out/'benchmarks/validation/raw_results.json')
    assert (out/'benchmarks/validation/training_policy.md').exists()
    with pytest.raises(ValueError, match='Validation'):
        subset.fit(out, resume=True)
    with pytest.raises(ValueError, match='Validation'):
        subset.prepare(tmp_path/'invalid_child', source_run=out)
    subset.main(['freeze', '--run-dir', str(out), '--repeats', '1', '--warmups', '0',
                 '--rhs-counts', '1', '--regimes', 'multiple'])
    selected = deepcopy(subset.read(out/'selected/deployment.json'))
    original_eval = study.evaluate_light
    fired = []
    def interrupt_once(*args, **kwargs):
        value = original_eval(*args, **kwargs)
        if not fired:
            fired.append(1)
            raise KeyboardInterrupt()
        return value
    with monkeypatch.context() as patch:
        patch.setattr(study, 'evaluate_light', interrupt_once)
        with pytest.raises(KeyboardInterrupt):
            subset.final(out)
    subset.final(out, resume=True)
    assert subset.read(out/'data/final_claim.json')['status'] == 'completed'
    assert selected == subset.read(out/'selected/deployment.json')
    for name in study.HOLDOUT:
        final_result = subset.read(out/f'final/{name}/raw_results.json')
        assert len(final_result) == 1
        assert subset.read(out/f'final/{name}/training_policy.json')['excluded_operators'] == 1
    with pytest.raises(ValueError, match='already completed'):
        subset.final(out, resume=True)


@pytest.mark.parametrize('target', ['excluded_record', 'selection', 'source_manifest'])
def test_excluded_evidence_and_policy_tampering_rejected(parent, tmp_path, target):
    out = tmp_path/'subset'
    report = subset.prepare(out, source_run=parent)
    if target == 'excluded_record':
        path = out/report['excluded'][0]['result_path']
        row = subset.read(path)
        row['trials'][0]['wall_seconds'] += 1.
    elif target == 'selection':
        path = out/'subset/training_selection.json'
        row = subset.read(path)
        row['excluded'] = []
    else:
        path = out/'light_manifest.json'
        row = subset.read(path)
        row['source_digest'] = 'changed'
    subset.write_json(path, row)
    with pytest.raises(ValueError, match='changed'):
        subset.fit(out)


def test_incomplete_search_is_not_treated_as_excluded_failure(parent, tmp_path):
    source = tmp_path/'incomplete'
    shutil.copytree(parent, source)
    summary = subset.read(source/'search/summary.json')
    summary['status'] = 'running'
    subset.write_json(source/'search/summary.json', summary)
    with pytest.raises(ValueError, match='Complete'):
        subset.prepare(tmp_path/'new', source_run=source)


def test_prepare_resume_after_interrupted_snapshot(parent, tmp_path, monkeypatch):
    out = tmp_path/'resumed_snapshot'
    original = shutil.copy2
    copied = []
    def interrupted(src, dst, *args, **kwargs):
        value = original(src, dst, *args, **kwargs)
        copied.append(1)
        if len(copied) == 3:
            raise KeyboardInterrupt()
        return value
    with monkeypatch.context() as patch:
        patch.setattr(shutil, 'copy2', interrupted)
        with pytest.raises(KeyboardInterrupt):
            subset.prepare(out, source_run=parent)
    assert (out/'subset/manifest.json').exists()
    assert not (out/'subset/prepared.json').exists()
    result = subset.prepare(out, resume=True)
    assert result['included_operators'] == 6
    subset.fit(out)


def test_failed_raw_measurements_cannot_be_approved_by_flags(parent, tmp_path):
    source = tmp_path/'false_approval'
    shutil.copytree(parent, source)
    summary = subset.read(source/'search/summary.json')
    row = next(r for r in summary['rows'] if not r['label_eligible'])
    path = source/row['path']
    record = subset.read(path)
    record['label_eligible'] = record['selected_candidate_success'] = True
    record['confirmation']['summaries'][record['confirmation']['selected_index']]['eligible'] = True
    subset.write_json(path, record)
    ledger = subset.read(path.with_name('progress.json'))
    ledger['result_digest'] = subset._hash(record)
    subset.signed_write(path.with_name('progress.json'), ledger)
    row.update(label_eligible=True, sha256=digest_file(path))
    summary['failed_cases'] = []
    subset.write_json(source/'search/summary.json', summary)
    with pytest.raises(ValueError, match='Failed candidate'):
        subset.prepare(tmp_path/'out', source_run=source)


def test_all_failed_labels_are_rejected(parent, tmp_path, monkeypatch):
    source = tmp_path/'all_failed'
    shutil.copytree(parent, source)
    shutil.rmtree(source/'search')
    original = oracle.measure_raw_solve
    def fail_all(*args, **kwargs):
        row = original(*args, **kwargs)
        row.update(success=False, final_true_residual=1., stop_reason='max_cycles')
        return row
    monkeypatch.setattr(oracle, 'measure_raw_solve', fail_all)
    study.search(source)
    with pytest.raises(ValueError, match='No valid training labels'):
        subset.prepare(tmp_path/'out', source_run=source)


def test_completed_retry_can_supply_labels_without_repeating_search(parent, tmp_path, monkeypatch):
    from test_light_smoother_retry import retry, config
    retry_config = dict(config(), candidate_count=9, finalist_count=8, confirmation_count=3)
    subset.write_json(tmp_path/'retry.json', retry_config)
    retried = tmp_path/'retried'
    # Retain the one failed case through retry, then apply the same exclusion
    # policy. The original successful six are never searched again.
    original = oracle.measure_raw_solve
    def fail_candidates(*args, **kwargs):
        row = original(*args, **kwargs)
        row.update(success=False, final_true_residual=1., stop_reason='max_cycles')
        return row
    with monkeypatch.context() as patch:
        patch.setattr(oracle, 'measure_raw_solve', fail_candidates)
        retry.retry(retried, source_run=parent, config=tmp_path/'retry.json')
    before = hashes(retried)
    def never_search(*args, **kwargs):
        raise AssertionError('Import/fit must not repeat search')
    monkeypatch.setattr(oracle, 'measure_raw_solve', never_search)
    out = tmp_path/'subset'
    selection = subset.prepare(out, source_run=retried)
    assert selection['included_operators'] == 6 and selection['excluded_operators'] == 1
    assert selection['excluded'][0]['result_path'].startswith('retry/cases/')
    assert (out/'retry/source_summary.json').exists()
    subset.fit(out)
    assert hashes(retried) == before
