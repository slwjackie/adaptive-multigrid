"""BAMG selection accounting, explicit fallback work, and benchmark resume."""
from types import SimpleNamespace
from time import perf_counter

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg.v67 import bamg_study as study


def settings(*, allowed_failure_rate=1.):
    return dict(seed=5, benchmark=dict(failure_penalty_seconds=100.,
        allowed_failure_rate=allowed_failure_rate, repetitions=1, rhs_counts=[1],
        timeout_seconds=1000., warmup_cycles=0))


def measured(method, case_id, seconds, *, converged=True, deployed=None):
    row = dict(method=method, case_id=case_id,
               raw=[dict(converged=converged, warm_seconds=seconds)])
    if deployed is not None:
        row['deployed'] = [dict(converged=True, warm_seconds=deployed)]
    return row


def test_summary_uses_equal_case_medians_and_keeps_missing_and_failed_cases():
    rows = [measured('fixed_C', 'a', seconds) for seconds in [1., 3., 100.]]
    rows += [measured('fixed_C', 'b', 9.),
             measured('bootstrap_two_cnn_supervised', 'a', 2.),
             measured('bootstrap_two_cnn_finetuned_uniform', 'a', 150., converged=False),
             dict(method='bootstrap_two_cnn_finetuned_uniform', case_id='b',
                  status='setup_or_evaluation_failure')]
    report = study.summarize(rows, settings())
    methods = report['methods']
    assert methods['fixed_C']['attempts'] == 4
    assert methods['fixed_C']['mean_case_median_warm_seconds'] == 6.
    # Missing case b is retained, rather than selecting on only a's fast solve.
    supervised = methods['bootstrap_two_cnn_supervised']
    assert supervised['cases'] == 2 and supervised['attempts'] == 2
    assert supervised['failures'] == 1 and supervised['failure_rate'] == .5
    assert supervised['mean_case_median_warm_seconds'] == 51.
    failed = methods['bootstrap_two_cnn_finetuned_uniform']
    assert failed['failures'] == 2 and failed['failure_rate'] == 1.
    # A timed failure exceeding the declared penalty retains its actual cost.
    assert failed['mean_case_median_warm_seconds'] == 125.
    assert methods['em5_C']['failures'] == 2
    assert methods['em5_C']['mean_case_median_warm_seconds'] == 100.
    assert report['fastest_eligible_method'] == 'fixed_C'


def test_fallback_success_cannot_rescue_raw_failure_for_model_selection():
    rows = [measured('fixed_C', 'a', 10.),
            measured('bootstrap_two_cnn_supervised', 'a', 20.),
            measured('bootstrap_two_cnn_finetuned_uniform', 'a', .01,
                     converged=False, deployed=.02),
            measured('bootstrap_two_cnn_finetuned_terminal', 'a', 30.)]
    report = study.summarize(rows, settings(allowed_failure_rate=0.))
    uniform = report['methods']['bootstrap_two_cnn_finetuned_uniform']
    assert uniform['failure_rate'] == 1. and not uniform['eligible']
    assert uniform['mean_case_median_warm_seconds'] == 100.
    assert report['selected_learned_model'] == 'bootstrap_two_cnn_supervised'
    assert report['fastest_eligible_method'] == 'fixed_C'


class Clock:
    def __init__(self):
        self.seconds = 0.

    def __call__(self):
        return self.seconds


class FakeBank:
    def __init__(self, clock, duration, action, *, patience=2):
        self.clock = clock
        self.duration = duration
        self.action = action
        self.calls = 0
        self.root = SimpleNamespace(a=sp.eye(1, format='csr'))
        self.mg_config = SimpleNamespace(max_cycles=5, safety_growth=1.1,
            safety_rtol_slack=0., stagnation_rho=.95,
            stagnation_patience=patience, divergence_factor=1e8)

    def cycle(self, x, b):
        self.clock.seconds += self.duration
        self.calls += 1
        return self.action(x, b)


def test_deployed_growth_rejection_accounts_for_attempt_and_fallback(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(study, 'perf_counter', clock)
    learned = FakeBank(clock, 2., lambda x, b: -b)
    fallback = FakeBank(clock, 3., lambda x, b: b.copy())
    result = study.deployed_solve(learned, fallback, np.ones(1), timeout_seconds=100.)
    assert result['converged'] and result['fallback_used']
    assert result['rejected_learned_cycles'] == 1 and result['fallback_cycles'] == 1
    assert learned.calls == fallback.calls == 1
    assert result['cycles'] == 1 and result['residuals'] == [1., 0.]
    assert result['cycle_seconds'] == [5.] and result['warm_seconds'] == 5.


def test_deployed_stagnation_switches_to_classical_tail_without_rejection(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(study, 'perf_counter', clock)
    learned = FakeBank(clock, 2., lambda x, b: .01 * b, patience=1)
    fallback = FakeBank(clock, 3., lambda x, b: b.copy())
    result = study.deployed_solve(learned, fallback, np.ones(1), timeout_seconds=100.)
    assert result['converged'] and result['fallback_used']
    assert result['rejected_learned_cycles'] == 0 and result['fallback_cycles'] == 1
    assert learned.calls == fallback.calls == 1
    assert result['cycles'] == 2
    assert result['cycle_seconds'] == [2., 3.] and result['warm_seconds'] == 5.


@pytest.mark.parametrize('failure_after_stagnation', [False, True])
def test_deployed_fallback_exception_becomes_timed_failure(monkeypatch, failure_after_stagnation):
    clock = Clock()
    monkeypatch.setattr(study, 'perf_counter', clock)
    action = (lambda x, b: .01 * b) if failure_after_stagnation else (lambda x, b: -b)
    learned = FakeBank(clock, 2., action, patience=1)

    def fail(x, b):
        raise np.linalg.LinAlgError('singular classical factor')

    fallback = FakeBank(clock, 3., fail)
    result = study.deployed_solve(learned, fallback, np.ones(1), timeout_seconds=100.)
    assert not result['converged'] and result['fallback_used']
    assert result['fallback_cycles'] == 1
    assert result['rejected_learned_cycles'] == (0 if failure_after_stagnation else 1)
    assert learned.calls == fallback.calls == 1
    assert result['warm_seconds'] == 5.
    assert sum(result['cycle_seconds']) == 5.


def test_benchmark_resume_reuses_completed_rows_and_rejects_changed_checkpoint(tmp_path, monkeypatch):
    study._write(tmp_path / 'study_manifest.json', {'version': 'test-source'})
    names = ('supervised_models.pt', 'reference.pt', 'deployment_uniform.pt', 'deployment_terminal.pt')
    for name in names:
        (tmp_path / name).write_bytes(b'checkpoint identity only')
    monkeypatch.setattr(study, 'METHODS', ('fixed_C', 'bootstrap_two_cnn_supervised'))
    monkeypatch.setattr(study, '_packages', lambda out: {})
    calls = []

    class BenchmarkBank:
        setup_report = dict(setup_seconds=4.)
        hierarchy_report = dict(levels=1)

        def solve(self, b, *, timeout_seconds):
            return dict(converged=True, warm_seconds=2., residuals=[1., 0.])

    def factory(method, example, config, cfg, packages):
        calls.append((method, example['case_id']))
        return BenchmarkBank()

    monkeypatch.setattr(study, '_factory', factory)
    monkeypatch.setattr(study, '_baseline_bank', lambda *args, **kwargs: BenchmarkBank())
    monkeypatch.setattr(study, 'deployed_solve',
                        lambda *args, **kwargs: dict(converged=True, warm_seconds=3.))
    example = dict(A=sp.eye(1, format='csr'), shape=(1, 1), b=np.ones(1),
                   case_id='one', family='mock', name='one')
    path = tmp_path / 'benchmark.json'
    first = study._benchmark_cases(tmp_path, settings(), None, [example], result_path=path)
    assert len(calls) == 2 and first['completed']
    for row in first['rows']:
        assert row['reuse']['1']['raw_total_seconds'] == 6.
        assert row['reuse']['1']['deployed_total_seconds'] == 11.
    second = study._benchmark_cases(tmp_path, settings(), None, [example], result_path=path, resume=True)
    assert len(calls) == 2 and second == first
    (tmp_path / 'deployment_uniform.pt').write_bytes(b'changed learned model')
    with pytest.raises(ValueError, match='identity'):
        study._benchmark_cases(tmp_path, settings(), None, [example], result_path=path, resume=True)
    assert len(calls) == 2


@pytest.mark.parametrize('n', [7, 15])
def test_complete_study_and_resume_preserve_frozen_masks(tmp_path, n):
    """Exercise the actual CLI, numerical teacher, both objectives, and seven methods."""
    config = study._read(study.PROJECT / 'configs/v6_7_bamg_smoke.json')
    config['data']['families'] = ['near_isotropic']
    for split in ('train', 'validation'):
        config['data'][split] = dict(sizes=[n], per_family=1)
    config['teacher'].update(vectors=6, selection_vectors=4, audit_vectors=4, eigen_count=2)
    for name in ('mask', 'weight'):
        config['models'][name].update(hidden=2, depth=1)
    config['supervised']['updates'] = 2
    config['finetune'].update(updates=2, cycles=3, final_cycles=6, curriculum_step=1)
    config['benchmark'].update(rhs_counts=[1], repetitions=1)
    config['mg']['max_cycles'] = 10
    config_path = tmp_path / 'config.json'
    output = tmp_path / 'run'
    study._write(config_path, config)
    command = ['run', '--config', str(config_path), '--output', str(output)]
    started = perf_counter()
    study.main(command)
    first_seconds = perf_counter() - started

    expected_files = ('teacher/index.json', 'pretrain.pt', 'pretrain_history.json',
        'supervised_models.pt', 'reference.pt', 'frozen_masks.json',
        'finetune_uniform.pt', 'finetune_terminal.pt',
        'finetune_uniform_history.json', 'finetune_terminal_history.json',
        'deployment_uniform.pt', 'deployment_terminal.pt',
        'diagnostics.json', 'benchmark.json')
    assert all((output / name).is_file() for name in expected_files)
    teacher_index = study._read(output / 'teacher/index.json')
    assert len(teacher_index['records']) == 1
    assert (output / teacher_index['records'][0]['path']).is_file()
    assert len(study._read(output / 'pretrain_history.json')) == 2
    histories = {}
    for weighting in ('uniform', 'terminal'):
        histories[weighting] = study._read(output / f'finetune_{weighting}_history.json')
        assert [row['cycles'] for row in histories[weighting]] == [3, 6]
        assert all(row['fixed_support'] for row in histories[weighting])

    reference_hash = study._file_hash(output / 'reference.pt')
    reference = study._load_pt(output / 'reference.pt')
    cache = reference['reference']['cache']
    assert len(cache) == 2  # Separate TRAIN and validation operators.
    expected_depth = 1 if n == 7 else 2
    for entry in cache.values():
        assert len(entry['masks']) == len(entry['columns']) == expected_depth
        for mask, columns in zip(entry['masks'], entry['columns']):
            assert mask.dtype == torch.bool and not mask.requires_grad
            assert mask.shape == columns.shape and mask.shape[1] == 16
            assert torch.all(mask.sum(1) >= 1)
            assert not (mask & (columns < 0)).any()
        assert study.mask_digest(entry['masks']) == entry['digest']

    benchmark = study._read(output / 'benchmark.json')
    assert benchmark['completed']
    assert len(benchmark['rows']) == len(study.METHODS) == 7
    assert {row['method'] for row in benchmark['rows']} == set(study.METHODS)
    assert all(row['status'] == 'measured' for row in benchmark['rows'])
    neural_rows = [row for row in benchmark['rows']
                   if row['method'].startswith('bootstrap_two_cnn_')]
    assert len(neural_rows) == 3
    assert len({row['mask_digest'] for row in neural_rows}) == 1
    assert neural_rows[0]['mask_digest'] in reference['masks'].values()

    resumed = perf_counter()
    study.main(command + ['--resume'])
    resume_seconds = perf_counter() - resumed
    assert study._file_hash(output / 'reference.pt') == reference_hash
    assert study._read(output / 'benchmark.json') == benchmark
    for weighting in ('uniform', 'terminal'):
        assert study._read(output / f'finetune_{weighting}_history.json') == histories[weighting]
    print(f'BAMG end-to-end n={n}: first={first_seconds:.3f}s resume={resume_seconds:.3f}s')
