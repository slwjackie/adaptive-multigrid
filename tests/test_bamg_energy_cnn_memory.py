"""Bounded teacher loading preserves cached bytes and exact training updates."""
from copy import deepcopy
import gc
import hashlib
import sys
import weakref

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg.config import MGConfig
from adaptive_mg.v67 import bamg_energy_cnn_study as paired
from adaptive_mg.v67 import bamg_energy_cnn_training as arm_training
from adaptive_mg.v67 import bamg_energy_cnn_epoch_training as epoch_training
from adaptive_mg.v67 import bamg_energy_labels as energy_labels
from adaptive_mg.v67 import bamg_training
from adaptive_mg.v67.bamg_teacher import TeacherConfig, TeacherResult
from adaptive_mg.v67.bamg_transfer import WeightCNN

sys.path.insert(0, str(paired.PROJECT / 'scripts'))
import bamg_energy_cnn_memory as memory


@pytest.fixture
def label_data(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    n = 7
    line = sp.diags((-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)), (-1, 0, 1), format='csr')
    a = sp.kronsum(line, line, format='csr')
    cfg = MGConfig(mode='classical', strategy_name='line_alt_bilinear_full__v22', stencil_backend='csr')
    tc = TeacherConfig(support_mode='fixed_bilinear', vectors=4, sweeps=(1, 2), setup_passes=2,
        selection_vectors=2, audit_vectors=2, eigen_count=0, adaptive_vectors=1, max_vector_retries=0)
    examples = [dict(A=a, shape=(n,n), case_id=f'case{i}', name=f'case{i}') for i in range(2)]
    args = (tmp_path / 'labels', examples, cfg, tc, dict(maxiter=200, rtol=1e-8), 10., dict(study='bound'))
    yield args
    torch.set_num_threads(old_threads)


def test_labels_reuse_cache_bytes_and_recover_unpublished_index(label_data, monkeypatch):
    disk, _ = memory.labels(*label_data)
    assert isinstance(disk, memory.DiskTeacherResults)
    before = {p.name: memory.file_sha256(p) for p in disk.directory.glob('*.pt')}
    index_path = disk.directory / 'manifest.json'
    index = paired._read(index_path)
    del index['records']['case1']
    paired._write(index_path, index)
    monkeypatch.setattr(paired, 'build_energy_teacher', lambda *a, **k: pytest.fail('must reuse saved labels'))
    recovered, digest = memory.labels(*label_data)
    assert len(recovered) == 2 and digest == memory.file_sha256(index_path)
    assert {p.name: memory.file_sha256(p) for p in disk.directory.glob('*.pt')} == before
    assert recovered[0].diagnostics['lambda_value'] == 10.
    assert list(recovered[::-1].keys) == ['case1', 'case0']


@pytest.mark.parametrize('kind', ['bytes', 'identity', 'case_id'])
def test_disk_cache_rejects_mutation(label_data, kind):
    disk, _ = memory.labels(*label_data)
    path = disk.directory / 'case0.pt'
    if kind == 'bytes':
        with path.open('ab') as stream:
            stream.write(b'corruption')
    else:
        package = paired._load_pt(path)
        package[kind] = {'study':'changed'} if kind == 'identity' else 'other'
        paired._save(path, package)
        hashes = [memory.file_sha256(path), disk.hashes[1]]
        disk = memory.DiskTeacherResults(disk.directory, disk.keys, hashes, disk.identity)
    with pytest.raises(ValueError, match='changed|identity differs'):
        disk[0]


def test_generation_and_group_iteration_do_not_retain_all_labels(label_data, monkeypatch):
    args = list(label_data)
    args[1] = [dict(args[1][0], case_id=f'case{i}', name=f'case{i}') for i in range(12)]
    active = []
    max_live = 0

    def fake_build(*a, **k):
        nonlocal max_live
        gc.collect()
        assert sum(r() is not None for r in active) == 0
        value = TeacherResult([], {}, 0., args[3], args[2])
        active.append(weakref.ref(value))
        max_live = max(max_live, sum(r() is not None for r in active))
        return value

    monkeypatch.setattr(paired, 'build_energy_teacher', fake_build)
    disk, _ = memory.labels(*args)
    assert len(disk) == 12 and max_live == 1
    assert all(r() is None for r in active)


def test_lazy_groups_only_hold_current_case(label_data, monkeypatch):
    disk, _ = memory.labels(*label_data)
    original_load = paired._load_pt
    seen = []

    def tracking_load(path):
        package = original_load(path)
        seen.append(weakref.ref(package['result'].levels[0]))
        assert sum(r() is not None for r in seen) <= 2
        return package

    monkeypatch.setattr(paired, '_load_pt', tracking_load)
    groups = memory.teacher_cases(disk)
    assert isinstance(groups, memory.LazyTeacherGroups) and not seen
    for _ in range(3):
        for levels in groups:
            assert len(levels) == 1
    del levels
    gc.collect()
    assert all(r() is None for r in seen)


def test_stream_hash_matches_original_without_read_bytes(tmp_path, monkeypatch):
    path = tmp_path / 'bytes'
    contents = b'bounded hash\0' * 200000
    path.write_bytes(contents)
    expected = hashlib.sha256(contents).hexdigest()
    original_sources = energy_labels._source_hashes()
    monkeypatch.setattr(type(path), 'read_bytes', lambda *a: pytest.fail('unbounded read'))
    assert memory.file_sha256(path) == expected
    assert memory.source_hashes() == original_sources


def test_scoped_loading_restores_every_alias_on_error():
    originals = [bamg_training._teacher_cases, arm_training._teacher_cases,
                 epoch_training._teacher_cases, paired._labels, energy_labels._source_hashes]
    with pytest.raises(RuntimeError, match='test'):
        with memory.memory_mode():
            assert bamg_training._teacher_cases is arm_training._teacher_cases is epoch_training._teacher_cases
            assert paired._labels is memory.labels
            raise RuntimeError('test')
    assert originals == [bamg_training._teacher_cases, arm_training._teacher_cases,
                         epoch_training._teacher_cases, paired._labels, energy_labels._source_hashes]


def test_lazy_supervision_and_resume_are_bitwise_identical(label_data, tmp_path):
    disk, _ = memory.labels(*label_data)
    eager = list(disk)
    torch.manual_seed(909)
    base = WeightCNN(hidden=2, depth=1).double()
    settings = dict(updates=4, learning_rate=1e-3, seed=72, alpha_mask=0.,
                    alpha_weight=1., alpha_action=0.1, checkpoint_every=1)
    eager_model, lazy_model = deepcopy(base), deepcopy(base)
    expected_history = epoch_training.train_supervised(None, eager_model, eager, settings=settings,
        checkpoint_path=tmp_path / 'eager.pt', manifest=dict(test='same'))
    with memory.memory_mode():
        assert arm_training._label_identity(disk) == arm_training._label_identity(eager)
        partial = epoch_training.train_supervised(None, lazy_model, disk, settings=settings,
            checkpoint_path=tmp_path / 'lazy.pt', manifest=dict(test='same'), stop_after=1)
        assert len(partial) == 1
        history = epoch_training.train_supervised(None, lazy_model, disk, settings=settings,
            checkpoint_path=tmp_path / 'lazy.pt', manifest=dict(test='same'), resume=True)
    assert history == expected_history
    assert all(torch.equal(eager_model.state_dict()[k], v) for k,v in lazy_model.state_dict().items())
    left = bamg_training.load_training_checkpoint(tmp_path / 'eager.pt')
    right = bamg_training.load_training_checkpoint(tmp_path / 'lazy.pt')
    assert bamg_training._digest(left) == bamg_training._digest(right)
