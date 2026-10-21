"""Warm-only diagnostics preserve the fitted run and original validation."""
import importlib.util
from pathlib import Path

import pytest

from test_light_smoother_subset import parent, subset, hashes
from adaptive_mg.v67 import light_smoother_evaluation as evaluation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('light_warm', ROOT/'scripts/run_v6_7_light_smoother_warm.py')
warm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warm)


@pytest.fixture(scope='module')
def fitted(parent, tmp_path_factory):
    out = tmp_path_factory.mktemp('warm_parent')/'subset'
    subset.prepare(out, source_run=parent)
    subset.fit(out)
    # Existing interrupted benchmark must be left byte-for-byte intact.
    subset.write_json(out/'benchmarks/validation/progress.json',
                      dict(status='running', completed_measurements=288, total_measurements=20160))
    return out


def test_all_eight_arms_full_validation_rhs16_and_resume(fitted, monkeypatch):
    before = hashes(fitted)
    original = evaluation.measured_light
    failed = []
    def with_failure(example, arm, *args, **kwargs):
        row = original(example, arm, *args, **kwargs)
        if not failed:
            failed.append(example.name)
        if (example.name == failed[0] and getattr(arm.get('predictor'), 'kind', None) == 'mlp'
                and arm.get('online') is None):
            row.update(success=False, successful=False, error='test-only failure')
        return row
    args = ['--run-dir', str(fitted), '--warmups', '1']
    with monkeypatch.context() as patch:
        patch.setattr(evaluation, 'measured_light', with_failure)
        result = warm.main(args)
    target = fitted/'benchmarks/validation_warm_rhs1_16'
    assert len(result['rows']) == 7
    assert set(result['manifest']['arms']) == warm.EXPECTED_ARMS
    assert result['manifest']['regimes'] == ['warm_multiple']
    assert subset.read(target/'progress.json') == dict(
        status='complete', completed_measurements=112, total_measurements=112)
    for row in result['rows']:
        for arm, regimes in row['runs'].items():
            assert set(regimes) == {'warm_multiple'}
            assert set(regimes['warm_multiple']) == {'1', '16'}
            for count, runs in regimes['warm_multiple'].items():
                assert len(runs) == 1
                record = runs[0]
                assert record['rhs_count'] == int(count)
                assert record['setup_scope'] == 'excluded_prepared_operator'
                assert record['independent_warm_prime']
                assert not record['warm_prime_online_adaptation']
                assert record['online_adaptation_seconds'] <= record['solve_seconds']
    for count in ('1', '16'):
        record = next(row for row in result['rows'] if row['example']['name'] == failed[0])
        assert not record['runs']['learned_P_MLP']['warm_multiple'][count][0]['success']
    assert subset.read(target/'diagnostic_status.json')['model_selection'] is False
    assert not (target/'selection.json').exists()
    assert not (fitted/'selected').exists()
    assert not (fitted/'benchmark_evidence.json').exists()
    assert (target/'comparison.csv').exists()
    assert (target/'training_policy.json').exists()
    assert all(subset.digest_file(fitted/path) == sha for path, sha in before.items())
    replay = warm.main(args+['--resume'])
    assert replay['summary'] == result['summary']
    with pytest.raises(ValueError, match='mismatch'):
        warm.main(args+['--resume', '--repeats', '2'])
    with pytest.raises(FileExistsError):
        warm.main(args)


@pytest.mark.parametrize('boundary', ['before_evaluator', 'after_first_record'])
def test_interrupted_diagnostic_resume(fitted, monkeypatch, boundary):
    tag = 'validation_warm_'+boundary
    args = ['--run-dir', str(fitted), '--tag', tag, '--rhs-counts', '1', '--warmups', '0']
    with monkeypatch.context() as patch:
        if boundary == 'before_evaluator':
            def interrupt(*args, **kwargs):
                raise KeyboardInterrupt()
            patch.setattr(warm.study, 'evaluate_light', interrupt)
        else:
            original = evaluation._write_json
            fired = []
            def write_then_interrupt(path, value, **kwargs):
                original(path, value, **kwargs)
                if path.parent.name == 'repeat_records' and not fired:
                    fired.append(True)
                    raise KeyboardInterrupt()
            patch.setattr(evaluation, '_write_json', write_then_interrupt)
        with pytest.raises(KeyboardInterrupt):
            warm.main(args)
    target = fitted/'benchmarks'/tag
    saved = hashes(target/'repeat_records') if (target/'repeat_records').exists() else {}
    result = warm.main(args+['--resume'])
    assert len(result['rows']) == 7
    assert subset.read(target/'progress.json')['completed_measurements'] == 56
    for path, sha in saved.items():
        assert subset.digest_file(target/'repeat_records'/path) == sha


def test_original_directory_and_missing_arm_rejected(fitted, monkeypatch):
    with pytest.raises(ValueError, match='separate tag'):
        warm.main(['--run-dir', str(fitted), '--tag', 'validation'])
    original = warm.subset.arms
    def without_cnn(*args, **kwargs):
        arms = original(*args, **kwargs)
        arms.pop('frozen_P_line')
        return arms
    monkeypatch.setattr(warm.subset, 'arms', without_cnn)
    with pytest.raises(ValueError, match='eight arms'):
        warm.main(['--run-dir', str(fitted), '--tag', 'validation_warm_missing'])
    assert not (fitted/'benchmarks/validation_warm_missing').exists()
