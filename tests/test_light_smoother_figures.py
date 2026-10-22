"""Diagnostic plots must not hide failures, invent timestamps, or alter runs."""
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pytest

pytest.importorskip('matplotlib')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_v6_7_light_smoother_figures as figures
from test_light_smoother_subset import parent, subset, hashes
from test_light_smoother_runtime import inputs
from test_light_smoother_adaptation_audit import inputs as adaptation_inputs
from adaptive_mg.v67 import light_smoother_evaluation as evaluation
from adaptive_mg.v67 import light_smoother_runtime as runtime
from adaptive_mg.v67 import solver
from adaptive_mg.v67.fixed_classical import BilinearClassicalBaseline, BILINEAR_STRATEGY_NAME
from adaptive_mg.v67.limited import forced_config


@pytest.fixture(scope='module')
def fitted(parent, tmp_path_factory):
    output = tmp_path_factory.mktemp('figure_parent')/'subset'
    subset.prepare(output, source_run=parent)
    subset.fit(output)
    return output


@pytest.fixture(scope='module')
def benchmark(fitted, tmp_path_factory):
    source, settings, cfg, baseline, arms, metadata, examples, artifacts = figures.load_run(fitted)
    output = tmp_path_factory.mktemp('figure_benchmark')/'bench'
    original = evaluation.measured_light
    def failed(example, arm, *args, **kwargs):
        run = original(example, arm, *args, **kwargs)
        if (example.name == examples[0].name and getattr(arm.get('predictor'), 'kind', None) == 'mlp'
                and arm.get('online') is None):
            run.update(success=False, successful=False, error='injected failure retained in report')
        return run
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(evaluation, 'measured_light', failed)
        evaluation.evaluate_light(examples, arms, cfg, baseline, output,
                                  regimes=['warm_multiple'], rhs_counts=[1, 4], repeats=1, warmups=0)
    return output


@pytest.mark.parametrize('kind', ['classical', 'light', 'cnn'])
@pytest.mark.parametrize('regime', ['cold', 'warm'])
def test_clock_observation_preserves_numerics(kind, regime):
    example, model, cfg = inputs(15)
    cfg = replace(cfg, record_trace=True, mg=replace(cfg.mg, strategy_name=BILINEAR_STRATEGY_NAME))
    baseline = BilinearClassicalBaseline.from_config(cfg)
    branch = {'classical': 'C', 'light': 'H_P', 'cnn': 'H_SP'}[kind]
    cfg = forced_config(cfg, branch)
    arm = dict(model=None if kind == 'classical' else model, branch=branch,
               selector=False, light=kind == 'light')
    prime_rhs = .5*example.b
    if kind == 'light':
        prepared = runtime.PreparedLightMG(example.a, example.n, model, cfg)
    else:
        prepared = solver.PreparedAdaptiveMG(example.a, example.n, arm['model'], cfg)
    if regime == 'warm':
        if hasattr(prepared, 'prepare_warm'):
            prepared.prepare_warm()
        prepared.solve(prime_rhs)
        if kind == 'light':
            prepared.reset_online()
    expected = prepared.solve(example.b)
    signature = model.signature()
    clocks = runtime.perf_counter, solver.perf_counter
    record, solution = figures.replay_measure(example, arm, cfg, baseline,
        example.b, example.exact, prime_rhs, regime)
    assert (runtime.perf_counter, solver.perf_counter) == clocks
    assert model.signature() == signature
    np.testing.assert_array_equal(solution, expected.x)
    np.testing.assert_array_equal(record['residual_history'], expected.residual_history)
    assert record['cycle_path'] == expected.cycle_path
    times = record['residual_wall_seconds']
    assert len(times) == len(expected.residual_history)
    assert times[0] == 0 and np.all(np.diff(times) > 0)
    assert 0 < times[-1] <= record['completion_wall_seconds']
    assert record['observer_seconds'] > 0
    assert record['setup_scope'] == ('included' if regime == 'cold' else 'excluded')


@pytest.mark.parametrize('method', ['mlp_ttt', 'regression_online'])
def test_warm_online_and_rejected_work_are_inside_wall_timestamps(method, monkeypatch):
    example, model, cfg, predictor, online = adaptation_inputs()
    online = dict(online, method=method)
    cfg = forced_config(cfg, 'H_P')
    baseline = BilinearClassicalBaseline.from_config(cfg)
    arm = dict(model=model, predictor=predictor, online=online, light=True, branch='H_P', selector=False)
    initial_predictor = predictor.to_dict()
    signature = model.signature()
    original = runtime.PreparedLightMG._adapt
    events = []
    def observed_adapt(prepared, x, b):
        events.append(prepared.setup_pending)
        return original(prepared, x, b)
    monkeypatch.setattr(runtime.PreparedLightMG, '_adapt', observed_adapt)
    # Force candidate verification to fail after consuming real time. The
    # deployed solve still uses its original working raw cycle normally.
    original_raw = runtime.PreparedLightMG.raw_cycle
    def failed_probe(prepared, x, b, stats=None):
        if sys._getframe(1).f_code.co_name == 'score':
            time.sleep(.012)
            raise RuntimeError('test-only delayed rejected candidate')
        return original_raw(prepared, x, b, stats)
    monkeypatch.setattr(runtime.PreparedLightMG, 'raw_cycle', failed_probe)
    record, _ = figures.replay_measure(example, arm, cfg, baseline,
        example.b, example.exact, .5*example.b, 'warm')
    assert events == [False]  # Never during the independent warm prime.
    assert record['stats']['online_adaptation_calls'] == 1
    assert record['stats']['online_adaptation_rejected'] == 1
    assert record['online_adaptation_seconds'] >= .012
    assert record['completion_wall_seconds'] >= record['online_adaptation_seconds']
    assert max(np.diff(record['residual_wall_seconds'])) >= .012
    assert model.signature() == signature and predictor.to_dict() == initial_predictor


def test_observer_restores_module_clocks_on_interruption():
    clocks = runtime.perf_counter, solver.perf_counter
    with pytest.raises(KeyboardInterrupt):
        with figures.WallObserver(time.perf_counter()):
            raise KeyboardInterrupt()
    assert (runtime.perf_counter, solver.perf_counter) == clocks


def test_failure_inclusive_stats_match_declared_cohort(benchmark):
    manifest, rows = figures.load_benchmark(benchmark)
    cases = figures.case_statistics(rows, manifest, ['warm_multiple'], [1, 4])
    families, consistency = figures.family_statistics(cases)
    assert len(cases) == 7*8*2
    failures = [r for r in cases if not r['success']]
    assert len(failures) == 2 and {r['arm'] for r in failures} == {'learned_P_MLP'}
    assert all(r['speedup_vs_fixed'] is None for r in failures)
    assert sum(r['failures'] for r in families) == 2
    # The same failed A is excluded from conditional spread for EVERY arm.
    assert {r['common_success_cases'] for r in consistency} == {6}
    assert all(r['family_time_cv'] is None for r in consistency)
    # Four RHS are not counted as four independent PDE operators.
    assert {r['total_cases'] for r in consistency} == {7}


@pytest.mark.parametrize('corruption', ['raw', 'ledger', 'status', 'partial'])
def test_report_rejects_corrupt_or_incomplete_validation(benchmark, tmp_path, corruption):
    target = tmp_path/'copied'
    shutil.copytree(benchmark, target)
    if corruption == 'raw':
        path = target/'raw_results.json'
        data = figures.read(path)
        data[0]['runs']['fixed_C']['warm_multiple']['1'][0]['wall_seconds'] *= 2
    elif corruption == 'ledger':
        path = next((target/'repeat_records').glob('*.json'))
        data = figures.read(path); data['run_digest'] = 'bad'
    else:
        path = target/'progress.json'
        data = figures.read(path)
        if corruption == 'status':
            data['status'] = 'running'
        else:
            data['completed_measurements'] -= 1
    subset.write_json(path, data)
    with pytest.raises(ValueError):
        figures.load_benchmark(target)


def test_report_is_read_only_and_does_no_solves(benchmark, tmp_path, monkeypatch):
    before = hashes(benchmark)
    def forbidden(*args, **kwargs):
        raise AssertionError('No solver/checkpoint reload is allowed in report')
    monkeypatch.setattr(figures, 'load_run', forbidden)
    monkeypatch.setattr(runtime.PreparedLightMG, 'solve', forbidden)
    monkeypatch.setattr(solver.PreparedAdaptiveMG, 'solve', forbidden)
    # Rendering is verified separately on actual output, not repeated in tests.
    monkeypatch.setattr(figures, 'family_plots', lambda *a: None)
    monkeypatch.setattr(figures, 'curve_figure', lambda *a, **kw: None)
    args = ['report', '--benchmark-dir', str(benchmark), '--output', str(tmp_path/'report')]
    result = figures.main(args)
    assert len(result) == 112 and hashes(benchmark) == before
    assert figures.main(args+['--resume']) == result


def test_replay_interrupt_resume_failure_and_source_immutability(fitted, tmp_path, monkeypatch):
    output = tmp_path/'replay'
    before = hashes(fitted)
    original = figures.signed_record
    fired = []
    def interrupt_after_commit(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path.parent.name == 'records' and not fired:
            fired.append(True)
            raise KeyboardInterrupt()
        return result
    args = ['replay', '--run-dir', str(fitted), '--n', '7', '--output', str(output), '--regimes', 'cold', 'warm']
    with monkeypatch.context() as patch:
        patch.setattr(figures, 'signed_record', interrupt_after_commit)
        with pytest.raises(KeyboardInterrupt):
            figures.main(args)
    committed = hashes(output/'records')
    actual = figures.replay_measure
    first = []
    def one_failure(*args, **kwargs):
        result, solution = actual(*args, **kwargs)
        if not first:
            first.append(True)
            result.update(workload_success=False, error='test-only retained failure')
        return result, solution
    with monkeypatch.context() as patch:
        patch.setattr(figures, 'plot_replay', lambda *a: None)
        patch.setattr(figures, 'replay_measure', one_failure)
        result = figures.main(args+['--resume'])
    assert len(result) == 112 and sum(not r['success'] for r in result) >= 1
    assert hashes(fitted) == before
    assert all(figures.digest_file(output/'records'/path) == sha for path, sha in committed.items())
    saved = hashes(output/'records')
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError('A completed replay must not solve again')
        patch.setattr(figures, 'replay_measure', forbidden)
        patch.setattr(figures, 'plot_replay', lambda *a: None)
        assert figures.main(args+['--resume']) == result
    assert hashes(output/'records') == saved
    with pytest.raises(ValueError, match='mismatch'):
        figures.main(args+['--resume', '--repeats', '2'])
    path = next((output/'records').glob('*.npz'))
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='arrays changed'):
        figures.main(args+['--resume'])


def test_output_does_not_overwrite_source(tmp_path):
    run = tmp_path/'run'; run.mkdir()
    with pytest.raises(ValueError, match='separate'):
        figures.output_directory(run/'figures', run)
    with pytest.raises(ValueError, match='separate'):
        figures.output_directory(tmp_path, run)
