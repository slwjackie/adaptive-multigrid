"""End-to-end fixed EM parent, recovery, provenance and timing contracts."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from adaptive_mg.v67.banks import Stats
from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.fixed_classical import FixedClassicalBaseline
from adaptive_mg.v67.research_runtime import offline_teacher_generation
from adaptive_mg.v67.research_training import sample_config
from adaptive_mg.v67.solver import PreparedAdaptiveMG
from adaptive_mg.v67.unet_transfer_study import initial_model, validate_settings
from adaptive_mg.v67.unroll import make_graph


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _settings():
    settings = json.loads((PROJECT / 'configs/v6_7_unet_transfer_smoke.json').read_text())
    settings['training_sizes'] = settings['validation_sizes'] = [7]
    settings['unet_transfer'].update(student_hidden=4, teacher_hidden=4, teacher_depth=1)
    for name in ('training', 'teacher_training'):
        settings[name].update(updates=2, kd_updates=1, prefix_cycles=1,
                              tail_cycles=0, rhs_variants=1)
    return settings


def _example(baseline, n=7):
    example = dataset([n], 7, 981671)[2]
    example.classical_selection = baseline.selection_record()
    example.research_split = 'train'
    return example


@pytest.mark.parametrize('stage', ['direct', 'teacher'])
@pytest.mark.parametrize('perturbed', [False, True])
def test_fixed_em_parent_matches_runtime_and_training_at_every_level(stage, perturbed):
    settings = _settings()
    cfg = validate_settings(settings)
    baseline = FixedClassicalBaseline.from_config(cfg)
    example = _example(baseline, n=15)
    model = initial_model(settings, stage)
    if perturbed:
        with torch.no_grad():
            model.transfer.head.weight.normal_(0., .002)
    scfg = sample_config(example, cfg, baseline, 'H_P')
    prepared = PreparedAdaptiveMG(example.a, example.n, model, scfg)
    with offline_teacher_generation():
        runtime = prepared.ensure_branch('H_P', Stats())
    graph = make_graph(example.a, (example.n, example.n), model, scfg)
    parent = prepared.classical
    graph_parent = make_graph(example.a, (example.n, example.n), model, scfg, learned=False)
    seen = 0
    while runtime.coarse is not None:
        assert parent.strategy.smoother == 'line_alt'
        assert parent.strategy.transfer == 'energymin'
        assert parent.strategy.coarsening == 'full'
        np.testing.assert_array_equal(runtime.base_weights, parent.base_weights)
        np.testing.assert_allclose(graph.base_weights, parent.base_weights, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(graph_parent.p.numpy().toarray(), parent.p.toarray(), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(runtime.p.toarray(), graph.p.numpy().toarray(), rtol=3e-6, atol=2e-8)
        np.testing.assert_allclose(runtime.coarse.a.toarray(), graph.coarse.a.numpy().toarray(), rtol=5e-6, atol=2e-8)
        np.testing.assert_allclose(runtime.coarse.a.toarray(),
                                  (runtime.p.T @ runtime.a @ runtime.p).toarray(),
                                  rtol=1e-12, atol=1e-12)
        if not perturbed:
            np.testing.assert_array_equal(runtime.p.toarray(), parent.p.toarray())
        seen += 1
        runtime, graph = runtime.coarse, graph.coarse
        parent, graph_parent = parent.coarse, graph_parent.coarse
    assert seen == 2


@pytest.mark.parametrize('failure', ['growth', 'nonfinite'])
def test_fallback_restores_last_accepted_iterate_and_fixed_parent_per_rhs(monkeypatch, failure):
    from adaptive_mg.v67 import solver

    settings = _settings()
    cfg = validate_settings(settings)
    cfg = replace(cfg, mg=replace(cfg.mg, tolerance=1e-14, absolute_tolerance=1e-15,
                                 max_cycles=60))
    baseline = FixedClassicalBaseline.from_config(cfg)
    example = _example(baseline)
    model = initial_model(settings, 'direct')
    prepared = PreparedAdaptiveMG(example.a, example.n, model,
                                   sample_config(example, cfg, baseline, 'H_P'))
    parent = prepared.classical
    real_hybrid, real_classical = solver.hybrid_cycle, solver.classical_cycle
    attempted, recoveries = [], []

    def failing_cycle(root, x, b, *args, **kwargs):
        attempted.append(x.copy())
        if len(attempted) == 1:
            return real_hybrid(root, x, b, *args, **kwargs)
        assert len(attempted) == 2, 'A rejected RHS must stay locked to classical'
        return np.full_like(x, np.nan) if failure == 'nonfinite' else x + 1e9

    def checked_classical(root, x, b, mg, stats):
        assert root is parent, 'Recovery must reuse the identical fixed EM hierarchy'
        assert mg == cfg.mg
        recoveries.append(x.copy())
        return real_classical(root, x, b, mg, stats)

    monkeypatch.setattr(solver, 'hybrid_cycle', failing_cycle)
    monkeypatch.setattr(solver, 'classical_cycle', checked_classical)
    x0 = .1 * np.sin(np.arange(example.a.shape[0]))
    for scale in (1., 1.3):
        attempted.clear()
        recoveries.clear()
        result = prepared.solve(scale * example.b, x0)
        assert len(attempted) == 2, 'Each new RHS must be allowed to attempt NN again'
        assert recoveries
        np.testing.assert_array_equal(recoveries[0], attempted[1])
        assert not np.array_equal(recoveries[0], x0)
        assert result.cycle_path[:2] == ['hybrid', 'classical_recovery']
        assert all(path == 'classical' for path in result.cycle_path[2:])
        assert result.stats['rollback_count'] == 1
        assert result.stats['rejected_neural_cycles'] == 1
        assert result.stats['classical_recovery_cycles'] == 1
        assert result.converged
        assert np.linalg.norm(scale * example.b - example.a @ result.x) <= result.stopping_threshold


def test_raw_teacher_gate_never_solves_with_fallback_or_substitutes_invalid_p(monkeypatch):
    from adaptive_mg.v67.unet_transfer_study import raw_contractions

    settings = _settings()
    cfg = validate_settings(settings)
    baseline = FixedClassicalBaseline.from_config(cfg)
    example = _example(baseline)
    models = {name: initial_model(settings, stage)
              for name, stage in (('direct_CNN', 'direct'), ('teacher', 'teacher'))}
    with torch.no_grad():
        models['teacher'].transfer.head.weight.fill_(float('nan'))

    def forbidden(*args, **kwargs):
        raise AssertionError('The raw contraction gate must not call safeguarded solves')

    for name in ('solve', 'solve_many', '_solve'):
        monkeypatch.setattr(PreparedAdaptiveMG, name, forbidden)
    row = raw_contractions([example], models, cfg, baseline, cycles=3, rhs_count=2)[example.name]
    assert row['fixed_C']['feasible'] and row['direct_CNN']['feasible']
    assert row['fixed_C']['contraction'] < 1.
    assert not row['teacher']['feasible']
    assert row['teacher']['contraction'] is None
    assert 'error' in row['teacher']
    assert all(not arm['fallback_allowed'] for arm in row.values())
    assert all(len(history) == 3 for history in row['fixed_C']['history'])
    np.testing.assert_allclose(row['fixed_C']['history'], row['direct_CNN']['history'], rtol=1e-9, atol=1e-12)


def test_cold_wall_clock_contains_em_and_nn_generation_and_solve(monkeypatch):
    from adaptive_mg import energymin
    from adaptive_mg.v67 import research_evaluation as evaluation

    settings = _settings()
    cfg = validate_settings(settings)
    baseline = FixedClassicalBaseline.from_config(cfg)
    example = _example(baseline)
    model = initial_model(settings, 'direct')
    clock = [0.]
    events = []
    energy_weights = energymin.energy_weights
    forward = model.transfer.forward
    solve = PreparedAdaptiveMG.solve

    def em_cost(*args, **kwargs):
        events.append('energy_minimization')
        clock[0] += 20.
        return energy_weights(*args, **kwargs)

    def nn_cost(*args, **kwargs):
        events.append('cnn_generation')
        clock[0] += 50.
        return forward(*args, **kwargs)

    def solve_cost(self, *args, **kwargs):
        events.append('numerical_solve')
        clock[0] += 10.
        return solve(self, *args, **kwargs)

    monkeypatch.setattr(evaluation, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(energymin, 'energy_weights', em_cost)
    monkeypatch.setattr(model.transfer, 'forward', nn_cost)
    monkeypatch.setattr(PreparedAdaptiveMG, 'solve', solve_cost)
    record = evaluation.measured_research(example,
        dict(branch='H_P', model=model, selector=False), cfg, baseline, regime='cold')
    assert record['successful'] and record['actual_neural_used']
    assert sorted(events) == ['cnn_generation', 'energy_minimization', 'numerical_solve']
    assert record['wall_seconds'] == 80.
    assert record['constructor_seconds'] == 20.
    assert record['setup_scope'] == 'included_once'
    assert not record['selector_in_wall_time']
    assert record['selection'] == baseline.selection_record()


@pytest.mark.parametrize('change', ['legacy_version', 'calibration_bank', 'missing_em_iterations',
                                     'contradictory_em_iterations', 'early_stop_em'])
def test_fixed_study_rejects_legacy_or_ambiguous_baseline_settings(change):
    settings = _settings()
    if change == 'legacy_version':
        settings['version'] = 'residual-unet-transfer-study-v1'
    elif change == 'calibration_bank':
        settings['bank'] = 'em_schedule'
    elif change == 'missing_em_iterations':
        del settings['solver']['mg']['em_iterations']
    elif change == 'contradictory_em_iterations':
        settings['solver']['mg']['em_iterations'] = 4
    else:
        settings['solver']['mg']['em_fixed_iterations'] = False
    with pytest.raises(ValueError):
        validate_settings(settings)


def test_fixed_study_rejects_checkpoint_bound_only_to_legacy_rules(tmp_path):
    from adaptive_mg.v67.limited import digest_file
    from adaptive_mg.v67.unet_transfer_study import expert

    settings = _settings()
    baseline = FixedClassicalBaseline.from_config(validate_settings(settings))
    model = initial_model(settings, 'direct')
    model.metadata.update(training_rules_digest=baseline.digest(), training_branch='H_P')
    directory = tmp_path / 'checkpoints/direct'
    directory.mkdir(parents=True)
    checkpoint = directory / 'candidate.pt'
    model.save(checkpoint)
    (directory / 'status.json').write_text(json.dumps(dict(all_updates_completed=True,
        skipped=0, feasible_task_updates=2, checkpoint_sha256=digest_file(checkpoint))))
    with pytest.raises(ValueError, match='fixed classical baseline'):
        expert(tmp_path, 'direct', baseline)


def test_prepare_train_and_raw_gate_do_not_import_strong_baseline(tmp_path):
    """A fresh interpreter forbids even a transitive legacy selector import."""
    config = tmp_path / 'tiny.json'
    config.write_text(json.dumps(_settings()))
    script = textwrap.dedent('''
        import importlib.abc
        import sys
        blocked = {'adaptive_mg.v67.strong', 'adaptive_mg.v67.strong_calibration',
                   'adaptive_mg.v67.strong_evaluation'}
        class NoStrong(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
                    raise AssertionError('Fixed study imported legacy selector: ' + fullname)
                return None
        sys.meta_path.insert(0, NoStrong())
        from adaptive_mg.v67 import unet_transfer_study as study
        from adaptive_mg.v67.research_data import load_development_data
        study.prepare(sys.argv[1], sys.argv[2])
        status = study.train(sys.argv[2], stage='direct')
        assert status['all_updates_completed'] and status['feasible_task_updates'] == 2
        out, settings, cfg, baseline = study.load(sys.argv[2])
        train, validation, manifest = load_development_data(out / 'data', baseline)
        assert all(e.classical_selection == baseline.selection_record() for e in train + validation)
        assert all(not hasattr(e, 'strong_selection') for e in train + validation)
        model = study.expert(out, 'direct', baseline)
        assert model.metadata['training_baseline_digest'] == baseline.digest()
        rows = study.raw_contractions(validation[:1],
            {'direct_CNN': model, 'teacher': study.initial_model(settings, 'teacher')},
            cfg, baseline, cycles=2, rhs_count=1)
        assert all(arm['feasible'] for row in rows.values() for arm in row.values())
        assert not blocked.intersection(sys.modules)
        assert not (out / 'strong_rules.json').exists()
        print('fixed-only prepare/train/raw verified')
    ''')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(PROJECT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, '-c', script, str(config), str(tmp_path / 'run')],
                            cwd=PROJECT, env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'fixed-only prepare/train/raw verified' in result.stdout
