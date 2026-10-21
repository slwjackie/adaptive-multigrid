"""Joint retry, failure preservation, unchanged parent and usable derived run."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from adaptive_mg.v67 import em_line_study as em
from adaptive_mg.v67 import light_smoother_study as study
from adaptive_mg.v67.limited import digest_file
from test_light_smoother_search import FakePrepared, data, settings, recorded_time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('light_retry', ROOT/'scripts/run_v6_7_light_smoother_retry.py')
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


def config():
    return dict(version=retry.VERSION, grid=[.8, 1., 1.2], candidate_count=81,
                screen_cycles=2, finalist_count=12, confirmation_count=3, seed=101)


def failed_measurement(prepared, b, cfg):
    row = recorded_time(prepared, b, cfg)
    row.update(success=False, final_true_residual=.1, residual_history=[1., .5, .1], stop_reason='max_cycles')
    return row


def parent_case(monkeypatch, path):
    monkeypatch.setattr(retry.oracle, 'PreparedLightMG', FakePrepared)
    monkeypatch.setattr(retry, 'PreparedLightMG', FakePrepared)
    monkeypatch.setattr(retry.oracle, 'measure_raw_solve', failed_measurement)
    return retry.oracle.search_case(*data(), settings(), path)


def test_joint_grid_sampling_bounds_and_determinism():
    protocol = {**config(), 'candidate_count': 128}
    vectors = retry.joint_candidates(3, protocol, 'operator')
    assert len(vectors) == len({x.tobytes() for x in vectors}) == 128
    np.testing.assert_array_equal(vectors[0], np.ones((3, 2)))
    assert all(x.shape == (3, 2) and np.isin(x, [.8, 1., 1.2]).all() for x in vectors)
    assert any(np.all(np.any(x != 1., axis=1)) for x in vectors)
    np.testing.assert_array_equal(vectors, retry.joint_candidates(3, protocol, 'operator'))
    assert len(retry.joint_candidates(2, protocol, 'operator')) == 81
    assert len(retry.joint_candidates(1, protocol, 'operator')) == 9
    with pytest.raises(ValueError):
        retry.validate_settings({**protocol, 'candidate_count': 2}, [.8, 1.2])


def synergy(prepared, b, cfg):
    # Changing either level alone fails. Both simultaneously can succeed.
    if np.all(np.any(prepared.coefficients != 1., axis=1)):
        return recorded_time(prepared, b, cfg)
    return failed_measurement(prepared, b, cfg)


def test_joint_search_recovers_coordinate_failure_and_counts_costs(monkeypatch, tmp_path):
    parent = parent_case(monkeypatch, tmp_path/'parent')
    monkeypatch.setattr(retry.oracle, 'measure_raw_solve', synergy)
    result = retry.retry_case(*data(), settings(), config(), parent, tmp_path/'retry', 'lineage')
    assert result['label_eligible'] and result['selection_outcome'] == 'success_recovery'
    assert not result['timing_improvement_confirmed']  # Failing baseline has no successful timing.
    assert np.all(np.any(np.array(result['selected_coefficients']) != 1., axis=1))
    assert result['rhs_digests'] == parent['rhs_digests']
    assert result['offline_seconds'] >= result['setup_seconds'] > 0
    assert result['setup_count'] == 1
    assert len(result['candidates']) == 81
    assert any(t['surrogate_only'] and not t['success'] for t in result['trials'])
    assert all(not t['fallback_used'] for t in result['trials'])
    assert result['confirmation']['summaries'][0]['historical_failure']


def test_screening_cannot_supply_labels_and_failed_cases_not_dropped(monkeypatch, tmp_path):
    parent = parent_case(monkeypatch, tmp_path/'parent')
    example, model, cfg = data()

    def screen_only(prepared, b, config):
        return recorded_time(prepared, b, config) if config.mg.max_cycles == 2 else failed_measurement(prepared, b, config)

    monkeypatch.setattr(retry.oracle, 'measure_raw_solve', screen_only)
    result = retry.retry_case(example, model, cfg, settings(), config(), parent, tmp_path/'retry', 'lineage')
    assert not result['label_eligible'] and not result['selected_candidate_success']
    assert result['selection_outcome'] == 'no_successful_candidate'
    assert result['trials'] and any(t['surrogate_only'] and t['success'] for t in result['trials'])


def test_confirmation_failure_rejects_joint_candidate(monkeypatch, tmp_path):
    parent = parent_case(monkeypatch, tmp_path/'parent')
    example, model, cfg = data()
    confirm = retry.oracle.search_rhs(example, count=2, seed=9, partition='confirmation')
    def measured(prepared, b, cfg):
        return failed_measurement(prepared, b, cfg) if any(np.array_equal(b, r) for r in confirm) else synergy(prepared, b, cfg)
    monkeypatch.setattr(retry.oracle, 'measure_raw_solve', measured)
    result = retry.retry_case(example, model, cfg, settings(), config(), parent, tmp_path/'retry', 'lineage')
    assert not result['label_eligible']
    assert any(s['eligible'] for s in result['optimization_summaries'].values())


def test_retry_resume_does_not_repeat_completed_measurements(monkeypatch, tmp_path):
    parent = parent_case(monkeypatch, tmp_path/'parent')
    calls = []
    def interrupted(prepared, b, cfg):
        calls.append(1)
        if len(calls) == 8:
            raise KeyboardInterrupt()
        return synergy(prepared, b, cfg)
    monkeypatch.setattr(retry.oracle, 'measure_raw_solve', interrupted)
    with pytest.raises(KeyboardInterrupt):
        retry.retry_case(*data(), settings(), config(), parent, tmp_path/'retry', 'lineage')
    before = retry.read(tmp_path/'retry/progress.json')['trials']
    assert len(before) == 7
    monkeypatch.setattr(retry.oracle, 'measure_raw_solve', synergy)
    result = retry.retry_case(*data(), settings(), config(), parent, tmp_path/'retry', 'lineage', resume=True)
    clean = retry.retry_case(*data(), settings(), config(), parent, tmp_path/'clean', 'lineage')
    assert result['trials'][:7] == before
    assert result['trials'] == clean['trials']
    assert result['selected_coefficients'] == clean['selected_coefficients']
    assert result['setup_count'] == 2
    with pytest.raises(ValueError, match='changed'):
        retry.retry_case(*data(), settings(), {**config(), 'seed': 102}, parent, tmp_path/'retry', 'lineage', resume=True)
    for split in ('validation', 'final', 'ood_grid'):
        example, model, cfg = data()
        example.research_split = split
        with pytest.raises(ValueError, match='training-split'):
            retry.retry_case(example, model, cfg, settings(), config(), parent, tmp_path/split, 'lineage')


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp('retry_pipeline')
    em_config = retry.read(ROOT/'configs/v6_7_em_line_smoke.json')
    for stage in em_config['stages'].values():
        stage['updates'] = 1
    retry.write_json(root/'em_config.json', em_config)
    em.prepare(root/'em_config.json', root/'em')
    em.train(root/'em')
    light = retry.read(ROOT/'configs/v6_7_light_smoother_smoke.json')
    light['search'].update(grid=[1.], repeats=1, rhs_count=1, confirmation_rhs_count=1)
    light['fit']['updates'] = 2
    retry.write_json(root/'light_config.json', light)
    parent = root/'parent'
    study.prepare(root/'light_config.json', parent, em_run_dir=root/'em')
    original = retry.oracle.measure_raw_solve
    calls = []
    def fail_first_operator(prepared, b, cfg):
        if not calls:
            calls.append(operator_digest(prepared.a))
        row = original(prepared, b, cfg)
        if operator_digest(prepared.a) == calls[0]:
            # Synthetic failure ONLY to exercise the new workflow; never a
            # research artifact or evidence of a performance improvement.
            row.update(success=False, final_true_residual=1., stop_reason='max_cycles')
        return row
    from adaptive_mg.provenance import operator_digest
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(retry.oracle, 'measure_raw_solve', fail_first_operator)
        summary = study.search(parent)
    assert len(summary['failed_cases']) == 1
    retry_config = {**config(), 'candidate_count': 9, 'finalist_count': 8, 'confirmation_count': 3}
    retry.write_json(root/'retry_config.json', retry_config)
    return root


def test_real_derived_run_only_retries_failure_preserves_sources_and_fits(pipeline, monkeypatch):
    root, parent, out = pipeline, pipeline/'parent', pipeline/'retry'
    hashes = {str(p.relative_to(parent)): digest_file(p) for p in parent.rglob('*') if p.is_file()}
    original_summary = retry.read(parent/'search/summary.json')
    cases = []
    real = retry.retry_case
    def counted(example, *args, **kwargs):
        cases.append(example.name)
        return real(example, *args, **kwargs)
    monkeypatch.setattr(retry, 'retry_case', counted)
    result = retry.main(['--source-run', str(parent), '--run-dir', str(out), '--config', str(root/'retry_config.json')])
    assert cases == original_summary['failed_cases']
    assert not result['failed_cases'] and result['preserved_cases'] == 6
    assert result['original_offline_seconds'] == original_summary['offline_seconds']
    assert result['retry_offline_seconds'] > 0
    assert result['offline_seconds'] == pytest.approx(result['original_offline_seconds']+result['retry_offline_seconds'])
    assert hashes == {str(p.relative_to(parent)): digest_file(p) for p in parent.rglob('*') if p.is_file()}
    for name in ('light_manifest.json', 'configuration.json', 'data/development_manifest.json', 'data/dataset_plan.json'):
        assert digest_file(parent/name) == digest_file(out/name)
    for old, new in zip(original_summary['rows'], result['rows']):
        if old['label_eligible']:
            assert new == old
    assert retry.main(['--run-dir', str(out), '--resume']) == result
    assert len(cases) == 1  # Completed retry does not run any solver again.
    study.fit(out)
    study.fit(out, resume=True)
    arms = study._arms(out, *[study.load(out)[i] for i in (1, 2, 3)])
    assert 'frozen_P_line' in arms and len(arms) == 6
    # Fit and later freeze pin the merged labels, including retry source hash.
    artifacts = study._artifacts(out)
    for row in result['rows']:
        assert row['path'] in artifacts
    changed = retry.read(out/'retry/manifest.json')
    changed['settings']['seed'] += 1
    retry.write_json(out/'retry/manifest.json', changed)
    with pytest.raises(ValueError, match='changed'):
        retry.main(['--run-dir', str(out), '--resume'])


def test_parent_validation_blocks_retry(pipeline, tmp_path):
    parent = tmp_path/'evaluated_parent'
    import shutil
    shutil.copytree(pipeline/'parent', parent)
    retry.write_json(parent/'benchmarks/validation/run_manifest.json', {})
    with pytest.raises(ValueError, match='Validation'):
        retry.main(['--source-run', str(parent), '--run-dir', str(tmp_path/'retry'),
                    '--config', str(pipeline/'retry_config.json')])
