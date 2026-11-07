"""Paired CNN training uses weight-only starts and exact latest-state resume."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from adaptive_mg.pde import DiffusionCase, assemble_stiffness
from adaptive_mg.provenance import operator_digest
from adaptive_mg.v67 import bamg_energy_cnn_training as paired
from adaptive_mg.v67.bamg_teacher import TeacherConfig, build_teacher
from adaptive_mg.v67.bamg_training import _digest, load_training_checkpoint
from adaptive_mg.v67.bamg_transfer import WeightCNN
from adaptive_mg.v67.bamg_validation import WarmCheckpointSelector


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def case(name, epsilon, split):
    a = assemble_stiffness(DiffusionCase(n=7, epsilon=epsilon, angle_deg=30.,
        contrast=1., pattern='uniform', discretization='q1'))
    return dict(A=a, shape=(7, 7), b=np.ones(49), case_id=name, family='near_isotropic',
                normalized_operator_digest=operator_digest(a, scale_invariant=True), split=split)


@pytest.fixture
def inputs(tmp_path):
    train = [case('train', .2, 'train')]
    validation = [case('validation', .5, 'validation')]
    labels = [build_teacher(train[0]['A'], (7, 7), config=TeacherConfig(
        support_mode='fixed_bilinear', vectors=6, sweeps=(1, 2), setup_passes=2,
        selection_vectors=2, audit_vectors=2, eigen_count=0, adaptive_vectors=1,
        max_vector_retries=0))]
    settings = dict(support_mode='fixed_bilinear', models=dict(weight=dict(hidden=2, depth=1)),
        k=4, l1_cap=8., mg=dict(mode='classical', strategy_name='line_alt_bilinear_full',
            pre_steps=1, post_steps=1, stencil_backend='csr', max_cycles=50),
        supervised=dict(updates=2, alpha_mask=0., alpha_action=.1,
                        learning_rate=1e-3, seed=9876, checkpoint_every=1),
        finetune=dict(updates=2, cycles=6, final_cycles=12, curriculum_step=1,
                      probe_kinds=['random'], seed=9876, checkpoint_every=1,
                      cycle_loss_weighting='uniform'),
        warm_validation=dict(every=1, per_family_per_size=1, repetitions=1,
                             warmup_cycles=0, timeout_seconds=30., failure_penalty_seconds=30.))
    torch.manual_seed(91)
    source = WeightCNN(hidden=2, depth=1).double()
    path = tmp_path / 'source.pt'
    torch.save(dict(weight_spec=source.spec, weight_state=source.state_dict(),
                    optimizer=dict(unwanted_momentum=torch.tensor([999.])),
                    update=999999, mask_state=dict(unused=torch.tensor([1.]))), path)
    return dict(seed=7, settings=settings, train=train, validation=validation,
                teacher_results=labels, init_weight=path,
                provenance=dict(init_weight_sha256=paired._file_hash(path),
                                labels='shared-train-labels', selected_lambda=10.))


def measurement(self, model):
    # Timing noise is irrelevant to the exact-update resume assertion.
    assert not model.training and not torch.is_grad_enabled()
    np.random.normal(size=3); torch.randn(3)
    seconds = (3., 1., 2.)[len(self.evaluations)]
    return dict(failures=0, attempts=1, failure_rate=0., eligible=True,
                mean_case_median_warm_seconds=seconds, cases=[])


def input_snapshot(inputs):
    matrices = []
    arrays = []
    for split in ('train', 'validation'):
        for example in inputs[split]:
            matrices.append(operator_digest(example['A']))
            arrays.append(example['b'])
    for result in inputs['teacher_results']:
        for level in result.levels:
            matrices.extend(operator_digest(getattr(level, key)) for key in ('A', 'H', 'P')
                            if getattr(level, key) is not None)
            arrays.extend([level.mask, level.weights, level.vectors])
    return _digest(dict(matrices=matrices, arrays=arrays, settings=inputs['settings'],
                        provenance=inputs['provenance']))


def test_zero_budget_imports_only_weights_and_starts_fresh_optimizer(tmp_path, inputs, monkeypatch):
    original = inputs['init_weight'].read_bytes()
    original_input = input_snapshot(inputs)
    calls = []
    real = paired.train_supervised
    def check(mask, weight, *args, **kwargs):
        assert mask is None and all(p.requires_grad for p in weight.parameters())
        calls.append(True)
        return real(mask, weight, *args, **kwargs)
    monkeypatch.setattr(paired, 'train_supervised', check)
    result = paired.train_arm(tmp_path / 'warm', arm='warm', stop_after=0, **inputs)
    checkpoint = load_training_checkpoint(result['paths']['supervised_checkpoint'])
    source = torch.load(inputs['init_weight'], weights_only=True)
    assert result['status'] == 'paused' and result['completed_updates'] == dict(supervised=0, finetune=0)
    assert checkpoint['optimizer']['state'] == {} and checkpoint['update'] == 0
    assert checkpoint['settings']['seed'] == 7 and inputs['settings']['supervised']['seed'] == 9876
    assert _digest(checkpoint['models']['weight']) == _digest(source['weight_state'])
    assert set(checkpoint['models']) == {'weight'}
    assert not Path(result['paths']['deployment']).exists()
    info = json.loads(Path(result['paths']['initialization']).read_text())
    assert info['transfer'] == 'weights_only' and info['optimizer_reused'] is False
    assert info['training_updates_reused'] is False and info['all_layers_trainable'] is True

    def no_transfer(*args, **kwargs):
        raise AssertionError('scratch must not import source weights')
    monkeypatch.setattr(paired, '_initialize_weight', no_transfer)
    scratch = paired.train_arm(tmp_path / 'scratch', arm='scratch', stop_after=0, **inputs)
    scratch_state = load_training_checkpoint(scratch['paths']['supervised_checkpoint'])
    torch.manual_seed(inputs['seed'])
    expected = WeightCNN(**inputs['settings']['models']['weight']).double()
    assert _digest(scratch_state['models']['weight']) == _digest(expected.state_dict())
    assert _digest(scratch_state['models']['weight']) != _digest(source['weight_state'])
    assert calls == [True, True]
    assert inputs['init_weight'].read_bytes() == original
    assert input_snapshot(inputs) == original_input


@pytest.mark.parametrize('arm', ['scratch', 'warm'])
def test_paired_resume_is_exact_across_stage_boundary_and_repairs_best(tmp_path, inputs, monkeypatch, arm):
    monkeypatch.setattr(WarmCheckpointSelector, '_measure', measurement)
    before = input_snapshot(inputs)
    source_hash = paired._file_hash(inputs['init_weight'])
    full = paired.train_arm(tmp_path / 'full', arm=arm, **inputs)
    paused = paired.train_arm(tmp_path / 'resumed', arm=arm, stop_after=1, **inputs)
    assert paused['completed_updates'] == dict(supervised=1, finetune=0)
    paused = paired.train_arm(tmp_path / 'resumed', arm=arm, stop_after=2, resume=True, **inputs)
    assert paused['status'] == 'paused' and paused['completed_updates'] == dict(supervised=2, finetune=1)
    assert not Path(paused['paths']['completion']).exists()
    Path(paused['paths']['best_warm']).write_bytes(b'interrupted derived export')
    resumed = paired.train_arm(tmp_path / 'resumed', arm=arm, resume=True, **inputs)
    assert resumed['status'] == 'complete' and resumed['selection']['best_update'] == 1
    assert resumed['selection']['best_at_budget_boundary'] is False
    assert resumed['selection']['convergence_certified'] is False
    for key in ('supervised_checkpoint', 'finetune_checkpoint'):
        full_state = load_training_checkpoint(full['paths'][key])
        resumed_state = load_training_checkpoint(resumed['paths'][key])
        assert full_state['history'] == resumed_state['history']
        for field in ('models', 'optimizer', 'rng'):
            assert _digest(full_state[field]) == _digest(resumed_state[field])
    fine = load_training_checkpoint(resumed['paths']['finetune_checkpoint'])
    assert [r['cycles'] for r in fine['history']] == [6, 12]
    deployment = torch.load(resumed['paths']['deployment'], weights_only=True)
    best = torch.load(resumed['paths']['best_warm'], weights_only=True)
    assert _digest(deployment['weight_state']) == _digest(best['weight_state'])
    assert _digest(deployment['weight_state']) != _digest(fine['models']['weight'])
    assert deployment['reference']['mask_state'] is deployment['reference']['weight_state'] is None
    assert input_snapshot(inputs) == before and paired._file_hash(inputs['init_weight']) == source_hash

    hashes = {key: paired._file_hash(path) for key, path in resumed['paths'].items()}
    def forbidden(*args, **kwargs):
        raise AssertionError('completed resume must not run training or timing again')
    monkeypatch.setattr(paired, 'train_supervised', forbidden)
    monkeypatch.setattr(paired, 'train_finetune', forbidden)
    monkeypatch.setattr(WarmCheckpointSelector, '_measure', forbidden)
    again = paired.train_arm(tmp_path / 'resumed', arm=arm, resume=True, **inputs)
    assert again['status'] == 'complete'
    assert hashes == {key: paired._file_hash(path) for key, path in again['paths'].items()}


def test_completed_resume_rejects_changed_labels_data_settings_or_checkpoint(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(WarmCheckpointSelector, '_measure', measurement)
    output = tmp_path / 'arm'
    paired.train_arm(output, arm='warm', **inputs)
    modified_settings = deepcopy(inputs['settings'])
    modified_settings['finetune']['learning_rate'] = .02
    changed_validation = deepcopy(inputs['validation'])
    changed_validation[0]['b'][0] += 1.
    changed_labels = deepcopy(inputs['teacher_results'])
    level = changed_labels[0].levels[0]
    row, col = np.argwhere(level.mask)[0]
    level.weights[row, col] += .01
    changed_provenance = dict(inputs['provenance'], selected_lambda=50.)
    for key, value in [('settings', modified_settings), ('validation', changed_validation),
                       ('teacher_results', changed_labels), ('provenance', changed_provenance)]:
        with pytest.raises(ValueError, match='provenance/configuration mismatch'):
            paired.train_arm(output, arm='warm', resume=True, **dict(inputs, **{key: value}))
    with inputs['init_weight'].open('ab') as stream:
        stream.write(b'changed-source')
    with pytest.raises(ValueError, match='SHA256'):
        paired.train_arm(output, arm='warm', resume=True, **inputs)


def test_crash_after_training_recovers_without_new_updates_and_reports_boundary(tmp_path, inputs, monkeypatch):
    def decreasing(self, model):
        return dict(failures=0, attempts=1, failure_rate=0., eligible=True,
                    mean_case_median_warm_seconds=3. - len(self.evaluations), cases=[])
    monkeypatch.setattr(WarmCheckpointSelector, '_measure', decreasing)
    real_save = paired._atomic_save
    def interrupted(value, path):
        if Path(path).name == 'deployment.pt':
            raise OSError('interrupted deployment export')
        return real_save(value, path)
    monkeypatch.setattr(paired, '_atomic_save', interrupted)
    output = tmp_path / 'arm'
    with pytest.raises(OSError, match='interrupted deployment'):
        paired.train_arm(output, arm='scratch', **inputs)
    assert not (output / 'completion.json').exists()
    fine_hash = paired._file_hash(output / 'finetune.pt')
    (output / 'best_warm.pt').unlink()
    monkeypatch.setattr(paired, '_atomic_save', real_save)
    def forbidden(*args, **kwargs):
        raise AssertionError('completed stages must not retrain')
    monkeypatch.setattr(paired, 'train_supervised', forbidden)
    monkeypatch.setattr(paired, 'train_finetune', forbidden)
    result = paired.train_arm(output, arm='scratch', resume=True, **inputs)
    assert result['status'] == 'complete' and result['selection']['best_at_budget_boundary'] is True
    assert paired._file_hash(output / 'finetune.pt') == fine_hash
    assert (output / 'best_warm.pt').is_file()


def test_actual_warm_selection_exports_geometry_only_deployment(tmp_path, inputs):
    result = paired.train_arm(tmp_path / 'real', arm='warm', **inputs)
    assert result['status'] == 'complete'
    report = json.loads(Path(result['paths']['warm_validation']).read_text())
    assert [e['update'] for e in report['evaluations']] == [0, 1, 2]
    assert report['contract']['split'] == 'validation'
    assert report['contract']['setup_included'] is False
    assert report['contract']['fallback'] is False
    assert result['selection']['metrics']['attempts'] == 1


def epoch_inputs(inputs):
    values = deepcopy(inputs)
    values['train'] = [case('train-a', .2, 'train'), case('train-b', .35, 'train')]
    a = assemble_stiffness(DiffusionCase(n=15, epsilon=.7, angle_deg=30.,
        contrast=1., pattern='uniform', discretization='q1'))
    values['train'].append(dict(A=a, shape=(15, 15), b=np.ones(225), case_id='train-c',
        family='near_isotropic', split='train',
        normalized_operator_digest=operator_digest(a, scale_invariant=True)))
    tc = TeacherConfig(support_mode='fixed_bilinear', vectors=6, sweeps=(1, 2),
        setup_passes=2, selection_vectors=2, audit_vectors=2, eigen_count=0,
        adaptive_vectors=1, max_vector_retries=0)
    values['teacher_results'] = [build_teacher(e['A'], e['shape'], config=tc)
                                for e in values['train']]
    values['settings']['sampling'] = 'epoch_shuffle'
    values['settings']['supervised']['updates'] = 6
    values['settings']['finetune'].update(updates=6, cycles=1, final_cycles=2,
                                         curriculum_step=3)
    return values


def constant_measurement(self, model):
    # Deliberate global RNG consumption must not affect TRAIN/probe streams.
    np.random.normal(size=11); torch.randn(11)
    return dict(failures=0, attempts=1, failure_rate=0., eligible=True,
                mean_case_median_warm_seconds=1., cases=[])


@pytest.mark.parametrize('arm', ['scratch', 'warm'])
def test_epoch_shuffle_resume_exact_inside_and_between_epochs(tmp_path, inputs, monkeypatch, arm):
    values = epoch_inputs(inputs)
    monkeypatch.setattr(WarmCheckpointSelector, '_measure', constant_measurement)
    full = paired.train_arm(tmp_path / 'full', arm=arm, **values)
    partial = paired.train_arm(tmp_path / 'resume', arm=arm, stop_after=2, **values)
    assert partial['training_coverage']['supervised']['distinct_cases_seen'] == 2
    partial = paired.train_arm(tmp_path / 'resume', arm=arm, stop_after=5, resume=True, **values)
    assert partial['completed_updates'] == dict(supervised=6, finetune=1)
    partial = paired.train_arm(tmp_path / 'resume', arm=arm, stop_after=2, resume=True, **values)
    assert partial['training_coverage']['finetune']['completed_epochs'] == 1
    result = paired.train_arm(tmp_path / 'resume', arm=arm, resume=True, **values)
    for phase, key in [('supervised', 'supervised_checkpoint'), ('finetune', 'finetune_checkpoint')]:
        a = load_training_checkpoint(full['paths'][key])
        b = load_training_checkpoint(result['paths'][key])
        assert a['history'] == b['history']
        for field in ('models', 'optimizer', 'rng'):
            assert _digest(a[field]) == _digest(b[field])
        assert a['source']['sampler']['version'] == paired.epoch_training.EpochShuffle.VERSION
        for offset in (0, 3):
            epoch = a['history'][offset:offset + 3]
            assert sorted(r['case'] for r in epoch) == [0, 1, 2]
            assert [r['epoch_position'] for r in epoch] == [1, 2, 3]
            assert {r['epoch'] for r in epoch} == {offset // 3 + 1}
            assert len({r['permutation_sha256'] for r in epoch}) == 1
        assert result['training_coverage'][phase]['minimum_visits'] == 2
        assert result['training_coverage'][phase]['maximum_visits'] == 2
        assert result['training_coverage'][phase]['all_cases_seen'] is True
    # A selected early model must not inherit the latest checkpoint's coverage claim.
    assert result['selection']['selected_finetune_coverage']['updates'] == 0
    assert result['selection']['selected_finetune_coverage']['all_cases_seen'] is False
    hashes = {k: paired._file_hash(v) for k, v in result['paths'].items()}
    again = paired.train_arm(tmp_path / 'resume', arm=arm, resume=True, **values)
    assert again == result
    assert hashes == {k: paired._file_hash(v) for k, v in result['paths'].items()}


def test_epoch_shuffle_scratch_and_warm_use_identical_case_orders(tmp_path, inputs, monkeypatch):
    values = epoch_inputs(inputs)
    monkeypatch.setattr(WarmCheckpointSelector, '_measure', constant_measurement)
    scratch = paired.train_arm(tmp_path / 'scratch', arm='scratch', **values)
    warm = paired.train_arm(tmp_path / 'warm', arm='warm', **values)
    for key in ('supervised_checkpoint', 'finetune_checkpoint'):
        a = load_training_checkpoint(scratch['paths'][key])
        b = load_training_checkpoint(warm['paths'][key])
        assert [(r['case'], r['epoch'], r['permutation_sha256']) for r in a['history']] == [
            (r['case'], r['epoch'], r['permutation_sha256']) for r in b['history']]
        assert _digest(a['models']) != _digest(b['models'])


@pytest.mark.parametrize('stage', ['supervised', 'finetune'])
def test_epoch_shuffle_rejects_budget_that_cannot_visit_all_train(tmp_path, inputs, stage):
    values = epoch_inputs(inputs)
    values['settings'][stage]['updates'] = 2
    if stage == 'finetune':
        values['settings'][stage]['curriculum_step'] = 1
    with pytest.raises(ValueError, match='at least one full TRAIN epoch'):
        paired.train_arm(tmp_path / 'short', arm='scratch', **values)
    assert not (tmp_path / 'short').exists()


def test_epoch_policy_change_requires_new_output(tmp_path, inputs):
    values = epoch_inputs(inputs)
    paired.train_arm(tmp_path / 'arm', arm='scratch', stop_after=1, **values)
    values['settings']['sampling'] = 'with_replacement'
    with pytest.raises(ValueError, match='provenance/configuration mismatch'):
        paired.train_arm(tmp_path / 'arm', arm='scratch', resume=True, **values)
