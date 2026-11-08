"""Bound teacher-label residency without changing numerical source snapshots.

The labels on disk, numerical kernels, objective, sampling and optimizer remain
unchanged. Only loading/orchestration is replaced during a scoped invocation.
"""
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path

from adaptive_mg.v67 import bamg_energy_cnn_study as paired
from adaptive_mg.v67 import bamg_energy_cnn_training as arm_training
from adaptive_mg.v67 import bamg_energy_cnn_epoch_training as epoch_training
from adaptive_mg.v67 import bamg_energy_labels as energy_labels
from adaptive_mg.v67 import bamg_study
from adaptive_mg.v67 import bamg_training


_EAGER_TEACHER_CASES = bamg_training._teacher_cases


def file_sha256(path):
    """Hash exactly the file bytes using a bounded read buffer."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    root = Path(energy_labels.__file__).resolve().parents[1]
    return {str(path.relative_to(root)): file_sha256(path)
            for path in sorted(root.rglob('*.py'))}


class DiskTeacherResults(Sequence):
    """Immutable index, loading and checking only the requested teacher case."""
    def __init__(self, directory, keys, hashes, identity):
        self.directory = Path(directory)
        self.keys, self.hashes = tuple(keys), tuple(hashes)
        self.identity = deepcopy(identity)
        if len(self.keys) != len(self.hashes):
            raise ValueError('Teacher disk index length mismatch')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(self.directory, self.keys[index], self.hashes[index], self.identity)
        key, expected = self.keys[index], self.hashes[index]
        path = self.directory / (key + '.pt')
        if file_sha256(path) != expected:
            raise ValueError('Shared teacher cache changed: ' + key)
        package = paired._load_pt(path)
        if package['identity'] != self.identity or package['case_id'] != key:
            raise ValueError('Teacher checkpoint identity differs')
        return package['result']


class LazyTeacherGroups(Sequence):
    """Expose active levels per case without materializing every hierarchy."""
    def __init__(self, results):
        self.results = results

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(self.results[index])
        return _EAGER_TEACHER_CASES(self.results[index])[0]


def teacher_cases(results):
    if isinstance(results, DiskTeacherResults):
        if not results:
            raise ValueError('at least one TRAIN teacher result is required')
        return LazyTeacherGroups(results)
    return _EAGER_TEACHER_CASES(results)


def labels(out, examples, cfg, tc, optimizer, strength, identity):
    """Build or validate one unchanged package at a time, then return its index."""
    directory = Path(out) / 'teacher/train'
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / 'manifest.json'
    index = paired._read(index_path) if index_path.exists() else dict(identity=identity, records={})
    if index['identity'] != identity:
        raise ValueError('Shared TRAIN teacher identity changed')
    keys, hashes = [], []
    for number, example in enumerate(examples, 1):
        key = example['case_id']
        path = directory / (key + '.pt')
        if key in index['records']:
            if file_sha256(path) != index['records'][key]:
                raise ValueError('Shared teacher cache changed: ' + key)
            package = paired._load_pt(path)
        elif path.exists():
            # Same recovery rule as the original: validate an atomic package
            # saved just before its index record was durably published.
            package = paired._load_pt(path)
        else:
            print(f'TRAIN teacher {number}/{len(examples)} lambda={strength}: {example["name"]}', flush=True)
            package = dict(identity=identity, case_id=key,
                result=paired.build_energy_teacher(example['A'], example['shape'], config=tc,
                    mg_config=cfg, strength=strength, optimizer=optimizer))
            paired._save(path, package)
        if package['identity'] != identity or package['case_id'] != key:
            raise ValueError('Teacher checkpoint identity differs')
        digest = file_sha256(path)
        index['records'][key] = digest
        paired._write(index_path, index)
        keys.append(key)
        hashes.append(digest)
        # Do not retain either full hierarchy or per-row bootstrap diagnostics.
        del package
    return DiskTeacherResults(directory, keys, hashes, identity), file_sha256(index_path)


@contextmanager
def memory_mode():
    """Install bounded loading for one single-process study invocation."""
    patches = [
        (paired, '_labels', labels),
        (bamg_training, '_teacher_cases', teacher_cases),
        (arm_training, '_teacher_cases', teacher_cases),
        (epoch_training, '_teacher_cases', teacher_cases),
        (energy_labels, '_source_hashes', source_hashes),
        (paired, '_file_hash', file_sha256),
        (bamg_study, '_file_hash', file_sha256),
        (arm_training, '_file_hash', file_sha256),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in patches]
    for module, name, replacement in patches:
        setattr(module, name, replacement)
    try:
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)
