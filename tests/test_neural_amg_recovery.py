import copy
from pathlib import Path
import runpy

import pytest
import torch

from neural_amg import study, teacher

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = runpy.run_path(str(ROOT / 'scripts/recover_neural_amg_bootstrap.py'))['recover']
PINS = study.read_json(ROOT / 'scripts/bootstrap_depth_recovery_sources.json')
L1_RECOVERY = runpy.run_path(str(ROOT / 'scripts/recover_neural_amg_bootstrap.py'))['recover_teacher_l1']
L1_PINS = study.read_json(ROOT / 'scripts/teacher_l1_recovery_sources.json')


@pytest.fixture
def legacy_run(tmp_path, monkeypatch):
    config = study.load_config(ROOT / 'configs/neural_amg_smoke.json')
    config['model']['hidden'] = 4
    config['teacher'].update(count=3, sweeps=2, maxiter=20)
    source = tmp_path / 'legacy'
    # On this shallow case the larger cap changes no arithmetic; emulate the
    # original source identity to exercise strict migration and real resume.
    with monkeypatch.context() as patch:
        patch.setattr(study, '_source_identity', lambda: copy.deepcopy(PINS['before']))
        study.train(config, source, max_updates=1)
    return config, source


def test_recovery_preserves_run_and_exact_training_trajectory(legacy_run, tmp_path):
    config, source = legacy_run
    hashes = {p.relative_to(source): study.file_hash(p) for p in source.rglob('*') if p.is_file()}
    output = tmp_path / 'fixed'
    report = RECOVERY(source, output)
    assert report['preserved_updates'] == 1
    for p, expected in hashes.items():
        assert study.file_hash(source / p) == expected
        if str(p) not in ('identity.json', 'training.pt'):
            assert study.file_hash(output / p) == expected
    old = torch.load(source / 'training.pt', weights_only=False)
    new = torch.load(output / 'training.pt', weights_only=False)
    assert old['updates'] == new['updates']
    for key in old['model']:
        torch.testing.assert_close(old['model'][key], new['model'][key], rtol=0, atol=0)
    resumed = study.train(config, output, resume=True)
    full = study.train(config, tmp_path / 'full')
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
    frozen = study.freeze(config, output)
    assert 'bootstrap_depth_recovery.json' in frozen['files']
    assert study.evaluate(config, output)['complete']
    study.atomic_json(output / 'bootstrap_depth_recovery.json', {})
    with pytest.raises(ValueError, match='Frozen file changed'):
        study.verify_freeze(config, output)
    with pytest.raises(FileExistsError):
        RECOVERY(source, output)


@pytest.mark.parametrize('damage', ['source', 'runtime', 'config', 'mesh', 'labels', 'checkpoint', 'frozen'])
def test_recovery_rejects_changed_or_frozen_run(legacy_run, tmp_path, damage):
    config, source = legacy_run
    if damage in ('source', 'runtime'):
        ident = study.read_json(source / 'identity.json')
        ident[damage]['changed'] = 'not-original'
        study.atomic_json(source / 'identity.json', ident)
    elif damage == 'config':
        config['seed'] += 1
        study.atomic_json(source / 'config.json', config)
    elif damage == 'mesh':
        with next((source / 'data').rglob('*.mesh.npz')).open('ab') as f:
            f.write(b'changed')
    elif damage == 'labels':
        with next((source / 'labels').glob('*.pt')).open('ab') as f:
            f.write(b'changed')
    elif damage == 'checkpoint':
        state = torch.load(source / 'training.pt', weights_only=False)
        state['identity']['config'] = 'changed'
        torch.save(state, source / 'training.pt')
    else:
        study.atomic_json(source / 'freeze.json', {})
    with pytest.raises(ValueError):
        RECOVERY(source, tmp_path / 'fixed')
    assert not (tmp_path / 'fixed').exists()


def test_bootstrap_depth_propagates_to_labels_and_runtime(tmp_path, monkeypatch):
    config = study.load_config(ROOT / 'configs/neural_amg_smoke.json')
    config['hierarchy']['max_levels'] = 17
    config['teacher']['maxiter'] = 2
    data = study.prepare(config, tmp_path / 'run')
    real = teacher.fit_teacher
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs['bootstrap_max_levels'])
        return real(*args, **kwargs)
    monkeypatch.setattr(study, 'fit_teacher', capture)
    monkeypatch.setattr(teacher, 'fit_teacher', capture)
    study._labels(config, data['train'][0], tmp_path / 'run')
    assert seen and set(seen) == {17}
    seen.clear()
    case = study._load_case(tmp_path / 'run', data['train'][0])
    study._hierarchy(config, case.a, method='teacher')
    assert seen and set(seen) == {17}


def test_recovery_source_pins_match_fixed_tree():
    assert study._source_identity() == PINS['after']
    assert study._source_identity() == L1_PINS['after']


@pytest.fixture
def l1_legacy_run(tmp_path, monkeypatch):
    config = study.load_config(ROOT / 'configs/neural_amg_smoke.json')
    config['model']['hidden'] = 4
    config['teacher'].update(count=3, sweeps=2, maxiter=20)
    source = tmp_path / 'legacy-l1'
    with monkeypatch.context() as patch:
        patch.setattr(study, '_source_identity', lambda: copy.deepcopy(L1_PINS['before']))
        study.train(config, source, max_updates=1)
    return config, source


def _prior_bootstrap_audit(source):
    previous = study.read_json(source / 'identity.json')
    original = copy.deepcopy(previous)
    original['source'] = PINS['before']
    report = dict(repair='bootstrap_depth_limit', original_commit=PINS['commit'],
                  previous_identity=original, new_identity=previous,
                  preserved_updates=1, old_bootstrap_max_levels=10,
                  new_bootstrap_max_levels=10)
    study.atomic_json(source / 'bootstrap_depth_recovery.json', report)


@pytest.mark.parametrize('previous_recovery', [False, True])
def test_l1_recovery_preserves_optimizer_trajectory_and_provenance(l1_legacy_run, tmp_path, previous_recovery):
    config, source = l1_legacy_run
    if previous_recovery:
        _prior_bootstrap_audit(source)
    hashes = {p.relative_to(source): study.file_hash(p) for p in source.rglob('*') if p.is_file()}
    output = tmp_path / 'fixed-l1'
    report = L1_RECOVERY(source, output)
    assert report['preserved_updates'] == 1
    assert report['l1_cap'] == config['teacher']['l1_cap']
    assert report['initialization_policy'] == 'project_infeasible_reference_only'
    for p, expected in hashes.items():
        assert study.file_hash(source / p) == expected
        if str(p) not in ('identity.json', 'training.pt'):
            assert study.file_hash(output / p) == expected
    if previous_recovery:
        assert report['preserved_recovery_provenance'] == {
            'bootstrap_depth_recovery.json': hashes[Path('bootstrap_depth_recovery.json')]}
    resumed = study.train(config, output, resume=True)
    full = study.train(config, tmp_path / 'full-l1')
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
    frozen = study.freeze(config, output)
    assert 'teacher_l1_recovery.json' in frozen['files']
    if previous_recovery:
        assert 'bootstrap_depth_recovery.json' in frozen['files']
    assert study.evaluate(config, output)['complete']
    study.atomic_json(output / 'teacher_l1_recovery.json', {})
    with pytest.raises(ValueError, match='Frozen file changed'):
        study.verify_freeze(config, output)


@pytest.mark.parametrize('damage', ['source', 'runtime', 'config', 'mesh', 'labels', 'checkpoint', 'frozen', 'prior_audit'])
def test_l1_recovery_rejects_tampering(l1_legacy_run, tmp_path, damage):
    config, source = l1_legacy_run
    if damage in ('source', 'runtime'):
        ident = study.read_json(source / 'identity.json')
        ident[damage]['changed'] = 'not-original'
        study.atomic_json(source / 'identity.json', ident)
    elif damage == 'config':
        config['seed'] += 1
        study.atomic_json(source / 'config.json', config)
    elif damage in ('mesh', 'labels'):
        path = (next((source / 'data').rglob('*.mesh.npz')) if damage == 'mesh'
                else next((source / 'labels').glob('*.pt')))
        with path.open('ab') as f:
            f.write(b'changed')
    elif damage == 'checkpoint':
        state = torch.load(source / 'training.pt', weights_only=False)
        state['identity']['config'] = 'changed'
        torch.save(state, source / 'training.pt')
    elif damage == 'prior_audit':
        _prior_bootstrap_audit(source)
        report = study.read_json(source / 'bootstrap_depth_recovery.json')
        report['new_identity']['config'] = 'other-run'
        study.atomic_json(source / 'bootstrap_depth_recovery.json', report)
    else:
        study.atomic_json(source / 'freeze.json', {})
    with pytest.raises(ValueError):
        L1_RECOVERY(source, tmp_path / 'fixed-l1')
    assert not (tmp_path / 'fixed-l1').exists()
