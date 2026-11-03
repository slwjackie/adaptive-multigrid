"""Compact numerical teacher, geometry-only deployment and checkpoint transfer."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from adaptive_mg.pde import DiffusionCase, assemble_stiffness
from adaptive_mg.smoothers import LineSmootherCache
from adaptive_mg.transfer import build_transfer_pattern
from adaptive_mg.v67 import bamg_study as study
from adaptive_mg.v67 import bamg_solver as solver
from adaptive_mg.v67.bamg_teacher import TeacherConfig, build_teacher, local_fit
from adaptive_mg.v67.bamg_transfer import MaskCNN, WeightCNN, direct_pattern, fixed_bilinear_mask
from adaptive_mg.v67.bamg_training import train_supervised, train_finetune


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def operator(n):
    return assemble_stiffness(DiffusionCase(n=n, epsilon=.2, angle_deg=30.,
                                           contrast=1., pattern='uniform', discretization='q1'))


def teacher_settings(**kwargs):
    return TeacherConfig(support_mode='fixed_bilinear', vectors=6, sweeps=(1, 2),
                         setup_passes=2, selection_vectors=2, audit_vectors=2,
                         eigen_count=0, adaptive_vectors=1, max_vector_retries=0, **kwargs)


@pytest.mark.parametrize('n', [31, 63])
def test_compact_bootstrap_preserves_q1_stencil_and_batched_lines(n):
    result = build_teacher(operator(n), (n, n), config=teacher_settings())
    assert len(result.diagnostics['passes']) == 2
    assert all(p['valid'] for p in result.diagnostics['passes'])
    for level in result.levels:
        rows, cols = level.A.nonzero()
        ny = level.shape[1]
        assert np.all(np.abs(rows // ny - cols // ny) <= 1)
        assert np.all(np.abs(rows % ny - cols % ny) <= 1)
        if level.P is None:
            continue
        expected = build_transfer_pattern(level.shape, level.pattern.coarse_shape).bilinear_weights != 0
        np.testing.assert_array_equal(level.mask, expected)
        assert level.P.nnz == int(expected.sum())
        assert not np.any(level.weights[~expected])
        cache = LineSmootherCache(level.A, level.shape)
        assert cache.zebra_batches('x') is not None
        assert cache.zebra_batches('y') is not None
        np.testing.assert_allclose(level.weights.sum(1),
            build_transfer_pattern(level.shape, level.pattern.coarse_shape).bilinear_weights.sum(1), atol=1e-12)


def test_compact_mask_uses_geometry_not_top_k_and_rejects_mismatch():
    pattern = direct_pattern(7)
    z = np.random.default_rng(3).normal(size=(49, 8))
    w4, m4, _ = local_fit(operator(7), pattern, z, config=teacher_settings(k=4))
    w8, m8, _ = local_fit(operator(7), pattern, z, config=teacher_settings(k=8))
    np.testing.assert_array_equal(m4, m8)
    np.testing.assert_array_equal(w4, w8)
    assert set(m4.sum(1)) == {1, 2, 4}
    bad = m4.copy(); bad[0, :] = False
    with pytest.raises(ValueError, match='bilinear geometry'):
        local_fit(operator(7), pattern, z, config=teacher_settings(), fixed_mask=bad)


def test_geometry_reference_never_runs_a_cnn_or_galerkin(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Geometry-only reference performed numerical setup')
    monkeypatch.setattr(solver, '_weights', forbidden)
    monkeypatch.setattr(solver, 'direct_features', forbidden)
    ref = solver.ReferenceModels(None, None, support_mode='fixed_bilinear')
    masks = ref.masks_for(operator(31), (31, 31))
    assert len(masks) == 3
    assert ref.last_report['weight_cnn_seconds'] == 0
    ref.masks_for(2 * operator(31), (31, 31))
    assert ref.last_report['cache_hit'] and len(ref._cache) == 1
    restored = solver.ReferenceModels.from_state_dict(ref.state_dict(include_cache=True))
    assert restored.mask_model is restored.weight_model is None
    assert solver.mask_digest(masks) == solver.mask_digest(restored.masks_for(operator(31), (31, 31)))
    assert restored.last_report['cache_hit']


def test_weight_only_training_and_resume_preserve_geometry(tmp_path):
    torch.manual_seed(8)
    model = WeightCNN(hidden=2, depth=1).double()
    initial = deepcopy(model.state_dict())
    other = deepcopy(model)
    result = build_teacher(operator(7), (7, 7), config=teacher_settings())
    settings = dict(updates=2, alpha_mask=0., learning_rate=1e-3)
    train_supervised(None, model, [result], settings=settings)
    path = tmp_path / 'pretrain.pt'
    train_supervised(None, other, [result], settings=settings, checkpoint_path=path, stop_after=1)
    train_supervised(None, other, [result], settings=settings, checkpoint_path=path, resume=True)
    assert set(torch.load(path, weights_only=True)['models']) == {'weight'}
    assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)
    ref = solver.ReferenceModels(None, None, support_mode='fixed_bilinear')
    masks = ref.masks_for(operator(7), (7, 7))
    before = solver.mask_digest(masks)
    examples = [dict(A=operator(7), shape=(7, 7), split='train')]
    fine = dict(updates=2, cycles=1, final_cycles=1, probe_kinds=['random'])
    fine_path = tmp_path / 'fine.pt'
    train_finetune(other, ref, examples, settings=fine, checkpoint_path=fine_path, stop_after=1)
    train_finetune(other, ref, examples, settings=fine, checkpoint_path=fine_path, resume=True)
    assert before == solver.mask_digest(ref.masks_for(operator(7), (7, 7)))


@pytest.mark.parametrize('kind', ['supervised', 'deployment', 'training'])
def test_old_checkpoint_transfers_weights_without_optimizer(tmp_path, kind):
    old = WeightCNN(hidden=2, depth=1).double()
    target = WeightCNN(hidden=2, depth=1).double()
    package = dict(weight_spec=old.spec, weight_state=old.state_dict())
    if kind == 'supervised':
        package = study._model_package(MaskCNN(hidden=2, depth=1).double(), old)
    if kind == 'training':
        package = dict(specs={'weight': old.spec}, models={'weight': old.state_dict()},
                       optimizer={'do_not_import': True}, update=1000)
    path = tmp_path / 'old.pt'; torch.save(package, path)
    digest = study._file_hash(path)
    report = study._initialize_weight(target, path)
    assert report['sha256'] == digest == study._file_hash(path)
    assert not report['optimizer_reused'] and not report['training_updates_reused']
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='architecture mismatch'):
        study._initialize_weight(WeightCNN(hidden=3, depth=1).double(), path)


def test_compact_cli_runs_and_resumes_from_old_weight_initialization(tmp_path):
    config = study._read(study.PROJECT / 'configs/v6_7_bamg_compact_smoke_q1.json')
    config['data']['families'] = ['near_isotropic']
    for split in ('train', 'validation'):
        config['data'][split] = dict(sizes=[7], per_family=1)
    config['models']['weight'].update(hidden=2, depth=1)
    config['teacher'].update(vectors=4, selection_vectors=2, audit_vectors=2, eigen_count=0)
    config['supervised']['updates'] = 1
    config['finetune'].update(updates=1, cycles=1, final_cycles=1, curriculum_step=0)
    config['benchmark'].update(rhs_counts=[1], repetitions=1)
    config['mg']['max_cycles'] = 10
    old = WeightCNN(**config['models']['weight']).double()
    source = tmp_path / 'old.pt'; torch.save(study._model_package(None, old), source)
    path = tmp_path / 'config.json'; study._write(path, config)
    output = tmp_path / 'new-run'
    command = ['run', '--config', str(path), '--output', str(output)]
    study.main(command + ['--init-weight', str(source)])
    package = study._load_pt(output / 'supervised_models.pt')
    assert package['mask_spec'] is package['mask_state'] is None
    reference = study._load_pt(output / 'reference.pt')['reference']
    assert reference['mask_state'] is reference['weight_state'] is None
    report = study._read(output / 'benchmark.json')
    assert report['completed'] and len(report['rows']) == 7
    assert set(report['selection']['methods']) == set(study.methods_for(config))
    assert any('weight_cnn' in row['method'] for row in report['rows'])
    assert all(row['status'] == 'measured' for row in report['rows'])
    before = torch.load(output / 'pretrain.pt', weights_only=True)
    study.main(command + ['--resume'])
    after = torch.load(output / 'pretrain.pt', weights_only=True)
    assert before['update'] == after['update'] == 1
    assert before['history'] == after['history']
    for key, value in before['models']['weight'].items():
        torch.testing.assert_close(value, after['models']['weight'][key], rtol=0, atol=0)
    assert study._read(output / 'weight_initialization.json')['sha256'] == study._file_hash(source)


def test_compact_settings_reject_teacher_student_mismatch():
    config = study._read(study.PROJECT / 'configs/v6_7_bamg_compact_smoke_q1.json')
    config['teacher']['support_mode'] = 'learned'
    with pytest.raises(ValueError, match='same support_mode'):
        study.validate_settings(config)
