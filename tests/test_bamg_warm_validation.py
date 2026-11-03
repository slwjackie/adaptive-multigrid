"""Warm model selection is independent of gradient updates and latest resume."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from adaptive_mg.config import MGConfig
from adaptive_mg.pde import DiffusionCase, assemble_stiffness
from adaptive_mg.v67 import bamg_study as study
from adaptive_mg.v67 import bamg_validation as validation
from adaptive_mg.v67.bamg_solver import ReferenceModels
from adaptive_mg.v67.bamg_training import FinetuneSettings, train_finetune, load_training_checkpoint
from adaptive_mg.v67.bamg_transfer import WeightCNN


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def case(name, epsilon, *, n=7, family='near_isotropic', split='validation'):
    a = assemble_stiffness(DiffusionCase(n=n, epsilon=epsilon, angle_deg=30.,
                                       contrast=1., pattern='uniform'))
    return dict(A=a, shape=(n, n), b=np.ones(n*n), case_id=name, name=name,
                family=family, split=split)


def fixture():
    torch.manual_seed(91)
    weight = WeightCNN(hidden=2, depth=1).double()
    cfg = MGConfig(mode='classical', strategy_name='line_alt_bilinear_full',
                   pre_steps=1, post_steps=1, stencil_backend='csr')
    reference = ReferenceModels(None, None, support_mode='fixed_bilinear', mg_config=cfg)
    return weight, cfg, reference, [case('train', .2, split='train')], [case('val', .5)]


def selector(tmp_path, *, every=2, repetitions=3, examples=None):
    weight, cfg, ref, train, val = fixture()
    result = validation.WarmCheckpointSelector(val if examples is None else examples, train, ref, cfg,
        settings=dict(every=every, repetitions=repetitions, warmup_cycles=0,
                      timeout_seconds=100., failure_penalty_seconds=100.),
        best_path=tmp_path / 'best.pt')
    return result


def metrics(seconds, failures=0):
    return dict(failures=failures, attempts=3, failure_rate=failures/3,
        eligible=failures == 0, mean_case_median_warm_seconds=seconds, cases=[])


def test_subset_covers_every_grid_and_family_and_rejects_leakage():
    train = [case('train', .2, split='train')]
    examples = [case(f'{family}-{n}-{i}', .5+i*.1, n=n, family=family)
                for family in ('near_isotropic', 'channel') for n in (7, 15) for i in (1, 0)]
    selected = validation.development_subset(examples, train, 1)
    assert len(selected) == 4
    assert {(e['family'], e['shape'][0]) for e in selected} == {
        (family, n) for family in ('near_isotropic', 'channel') for n in (7, 15)}
    assert [e['case_id'] for e in selected] == [e['case_id'] for e in
        validation.development_subset(list(reversed(examples)), train, 1)]
    with pytest.raises(ValueError, match='overlap'):
        validation.development_subset([dict(train[0], case_id='renamed', split='validation')], train, 1)
    with pytest.raises(ValueError, match='never TRAIN/final/OOD'):
        validation.development_subset([dict(examples[0], split='final')], train, 1)


def test_warm_measurement_counts_setup_failures_and_uses_repeat_medians(tmp_path, monkeypatch):
    selected = selector(tmp_path, examples=[case('a', .4), case('b', .6, family='variable_coefficient')])
    weight, cfg, ref, _, _ = fixture()
    calls = []
    class Bank:
        mask_digest = selected.contract['cases'][0]['mask_digest']
        setup_report = dict(setup_seconds=9999.)
        def solve(self, b, *, x0, timeout_seconds):
            assert not torch.is_grad_enabled() and not weight.training
            assert np.all(x0 == 0) and timeout_seconds == 100.
            calls.append(b.copy())
            return dict(converged=True, reason='converged', cycles=35,
                        warm_seconds=[1., 50., 2.][len(calls)-1])
    def build(a, *args):
        if a is selected.examples[1]['A']:
            raise MemoryError('hierarchy cap')
        return Bank()
    monkeypatch.setattr(validation, 'build_runtime', build)
    report = selected.maybe_evaluate(0, 4, weight)
    assert report['failures'] == 3 and report['attempts'] == 6
    assert report['mean_case_median_warm_seconds'] == 51.  # mean(median(1,50,2), 100)
    assert len(calls) == 3 and all(np.array_equal(b, calls[0]) for b in calls)
    assert report['cases'][0]['samples'][0]['cycles'] == 35  # No 6/12-cycle cutoff.
    assert weight.training and all(p.grad is None for p in weight.parameters())


def test_failure_priority_strict_ties_final_evaluation_and_latest_are_separate(tmp_path, monkeypatch):
    selected = selector(tmp_path, every=2)
    weight, _, _, _, _ = fixture()
    scores = iter([metrics(1., 1), metrics(20.), metrics(30.), metrics(20.)])
    monkeypatch.setattr(selected, '_measure', lambda model: next(scores))
    selected.maybe_evaluate(0, 5, weight)
    assert selected.maybe_evaluate(1, 5, weight) is None
    selected.maybe_evaluate(2, 5, weight)  # Fewer failures wins even though slower.
    assert selected.best['update'] == 2
    original = deepcopy(selected.best['weight_state'])
    with torch.no_grad():
        next(weight.parameters()).add_(1.)
    selected.maybe_evaluate(4, 5, weight)
    selected.maybe_evaluate(5, 5, weight)  # Always evaluate the final update; ties keep earlier.
    assert selected.best['update'] == 2
    assert [r['update'] for r in selected.evaluations] == [0, 2, 4, 5]
    for key in original:
        torch.testing.assert_close(original[key], selected.best['weight_state'][key], rtol=0, atol=0)


def test_exact_resume_keeps_best_and_validation_does_not_change_training_rng(tmp_path, monkeypatch):
    weight, cfg, reference, train, _ = fixture()
    options = FinetuneSettings(updates=4, cycles=1, final_cycles=2, curriculum_step=2,
        probes_per_kind=1, smoothing_sweeps=1, hard_cycles=1, checkpoint_every=3)
    def measure(self, model):
        assert not torch.is_grad_enabled() and not model.training
        np.random.random(3); torch.randn(3)  # Must not perturb TRAIN probes or updates.
        return metrics([30., 10., 20.][len(self.evaluations)])
    monkeypatch.setattr(validation.WarmCheckpointSelector, '_measure', measure)
    full = deepcopy(weight); full_selector = selector(tmp_path / 'full')
    full_history = train_finetune(full, reference, train, settings=options,
        checkpoint_path=tmp_path / 'full.pt', warm_selector=full_selector)
    no_validation = deepcopy(weight)
    train_finetune(no_validation, reference, train, settings=options)
    for key in full.state_dict():
        torch.testing.assert_close(full.state_dict()[key], no_validation.state_dict()[key], rtol=0, atol=0)
    latest = tmp_path / 'latest.pt'
    partial = deepcopy(weight); partial_selector = selector(tmp_path / 'resumed')
    train_finetune(partial, reference, train, settings=options, checkpoint_path=latest,
                   stop_after=3, warm_selector=partial_selector)
    saved = load_training_checkpoint(latest)
    assert saved['update'] == 3 and saved['warm_selection_state']['best']['update'] == 2
    assert [r['update'] for r in saved['warm_selection_state']['evaluations']] == [0, 2]
    partial_selector.best_path.write_bytes(b'interrupted best export')
    restored_selector = selector(tmp_path / 'resumed')
    restored = deepcopy(weight)
    history = train_finetune(restored, reference, train, settings=options, checkpoint_path=latest,
                            resume=True, warm_selector=restored_selector)
    assert history == full_history
    for key in full.state_dict():
        torch.testing.assert_close(full.state_dict()[key], restored.state_dict()[key], rtol=0, atol=0)
    best = torch.load(restored_selector.best_path, weights_only=True)
    assert best['update'] == 2
    assert load_training_checkpoint(latest)['update'] == 4
    assert any(not torch.equal(best['weight_state'][key], restored.state_dict()[key])
               for key in best['weight_state'])
    # Completed resume must neither repeat timing nor lose earlier best weights.
    complete_selector = selector(tmp_path / 'resumed')
    train_finetune(restored, reference, train, settings=options, checkpoint_path=latest,
                   resume=True, warm_selector=complete_selector)
    assert [e['update'] for e in complete_selector.evaluations] == [0, 2, 4]
    with pytest.raises(ValueError, match='provenance'):
        train_finetune(restored, reference, train, settings=options, checkpoint_path=latest,
                       resume=True, warm_selector=selector(tmp_path / 'resumed', every=1))


def test_six_to_twelve_cli_exports_best_weights_and_resumes(tmp_path):
    config = study._read(study.PROJECT / 'configs/v6_7_bamg_compact_smoke_q1.json')
    config['data']['families'] = ['near_isotropic']
    for split in ('train', 'validation'):
        config['data'][split] = dict(sizes=[7], per_family=1)
    config['models']['weight'].update(hidden=2, depth=1)
    config['teacher'].update(vectors=4, selection_vectors=2, audit_vectors=2, eigen_count=0)
    config['supervised']['updates'] = 1
    config['finetune'].update(updates=2, cycles=6, final_cycles=12, curriculum_step=1)
    config['benchmark'].update(rhs_counts=[1], repetitions=1)
    config['mg']['max_cycles'] = 25
    path = tmp_path / 'config.json'; study._write(path, config)
    output = tmp_path / 'run'
    command = ['run', '--config', str(path), '--output', str(output)]
    study.main(command)
    for choice in ('uniform', 'terminal'):
        latest = load_training_checkpoint(output / f'finetune_{choice}.pt')
        assert latest['status'] == 'complete' and latest['update'] == 2
        assert [r['cycles'] for r in latest['history']] == [6, 12]
        selection = study._read(output / f'finetune_{choice}_warm_validation.json')
        assert [r['update'] for r in selection['evaluations']] == [0, 1, 2]
        assert selection['contract']['setup_included'] is False
        best = torch.load(output / f'finetune_{choice}_best_warm.pt', weights_only=True)
        deployment = study._load_pt(output / f'deployment_{choice}.pt')
        assert deployment['selection']['mode'] == 'best_warm'
        assert deployment['selection']['update'] == best['update']
        for key in best['weight_state']:
            torch.testing.assert_close(deployment['weight_state'][key], best['weight_state'][key],
                                       rtol=0, atol=0)
    before = {name: study._file_hash(output / name) for name in
              ('benchmark.json', 'reference.pt', 'deployment_uniform.pt', 'deployment_terminal.pt')}
    study.main(command + ['--resume'])
    assert before == {name: study._file_hash(output / name) for name in before}


@pytest.mark.parametrize('values', [dict(every=0), dict(repetitions=0), dict(warmup_cycles=-1),
                                   dict(timeout_seconds=float('nan')), dict(allowed_failure_rate=2)])
def test_invalid_validation_budget_rejected(values):
    with pytest.raises(ValueError):
        validation.validation_settings(values)


@pytest.mark.parametrize('n', [31, 63])
def test_twelve_cycle_gradient_and_warm_evaluation_on_multilevel_q1(tmp_path, n):
    weight, cfg, reference, _, _ = fixture()
    def q1(name, epsilon, split):
        example = case(name, epsilon, n=n, split=split)
        example['A'] = assemble_stiffness(DiffusionCase(n=n, epsilon=epsilon, angle_deg=30.,
            contrast=1., pattern='uniform', discretization='q1'))
        return example
    train = [q1('train', .2, 'train')]; val = [q1('val', .4, 'validation')]
    selected = validation.WarmCheckpointSelector(val, train, reference, cfg,
        settings=dict(every=1, repetitions=1, timeout_seconds=10.), best_path=tmp_path / 'best.pt')
    mask_before = validation.mask_digest(reference.masks_for(train[0]['A'], (n, n)))
    before = deepcopy(weight.state_dict())
    history = train_finetune(weight, reference, train,
        settings=FinetuneSettings(updates=1, cycles=6, final_cycles=12, curriculum_step=0,
                                  probe_kinds=('random',)),
        checkpoint_path=tmp_path / 'latest.pt', warm_selector=selected)
    assert history[0]['cycles'] == 12 and np.isfinite(history[0]['gradient_norm'])
    assert history[0]['gradient_norm'] > 0
    assert any(not torch.equal(before[k], v) for k, v in weight.state_dict().items())
    assert mask_before == validation.mask_digest(reference.masks_for(train[0]['A'], (n, n)))
    assert [r['update'] for r in selected.evaluations] == [0, 1]
    assert all(r['attempts'] == 1 for r in selected.evaluations)
