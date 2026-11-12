import copy
from pathlib import Path
import random
import runpy

import numpy as np
import pytest
import torch

from neural_amg import study

ROOT = Path(__file__).resolve().parents[1]
START = runpy.run_path(str(ROOT / 'scripts/start_neural_amg_cancellation.py'))['start']
PINS = study.read_json(ROOT / 'scripts/cancellation_start_sources.json')


@pytest.fixture
def source_run(tmp_path):
    config = study.load_config(ROOT / 'configs/neural_amg_smoke.json')
    config['model']['hidden'] = 4
    source = tmp_path / 'v2'
    data = study.prepare(config, source)
    legacy = copy.deepcopy(config)
    legacy['version'] = 'neural-amg-p1-study-v2'
    legacy['model'].pop('l1_cap', None)
    legacy['teacher'].pop('cancellation_weight', None)
    legacy['teacher']['l1_cap'] = 8.
    for phase in ('supervised', 'finetune'):
        legacy[phase].pop('cancellation_weight', None)
    previous = study.identity(legacy)
    previous['source'] = PINS['source']
    study.atomic_json(source / 'config.json', legacy)
    study.atomic_json(source / 'identity.json', previous)
    torch.manual_seed(legacy['seed'])
    model = study._model(config)
    state = dict(identity=previous, development_digest=study.digest(data),
                 model=study._state(model), phase=1, epoch=0, offset=0,
                 updates=legacy['supervised']['epochs'] * len(data['train']),
                 complete=False, history=[], best_score=[0, 1.], best_update=1,
                 best_model=study._state(model), optimizer={'old': 'discard'},
                 torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                 python_rng=random.getstate())
    study.atomic_json(source / 'validation.json', [])
    study.atomic_torch(source / 'training.pt', state)
    (source / 'labels').mkdir()
    (source / 'labels' / 'old.pt').write_bytes(b'unused old objective labels')
    config_path = tmp_path / 'new-config.json'
    study.atomic_json(config_path, config)
    return config, config_path, source


def test_cancellation_start_reuses_weights_resets_selection_and_resumes(source_run, tmp_path):
    config, config_path, source = source_run
    before = {p.relative_to(source): study.file_hash(p) for p in source.rglob('*') if p.is_file()}
    old = torch.load(source / 'training.pt', weights_only=False)
    output = tmp_path / 'cancellation'
    report = START(source, output, config_path)
    new = torch.load(output / 'training.pt', weights_only=False)
    assert report['source_updates'] == old['updates']
    assert report['source_next_case'].startswith('train_')
    assert new['phase'] == 1
    assert new['epoch'] == new['offset'] == new['updates'] == 0
    assert new['optimizer'] is None
    assert new['best_model'] is None and new['best_score'] is None
    assert new['history'] == []
    assert not (output / 'labels').exists()
    for name in old['model']:
        torch.testing.assert_close(old['model'][name], new['model'][name], rtol=0, atol=0)
    for name, expected in before.items():
        assert study.file_hash(source / name) == expected
    finished = study.train(config, output, resume=True)
    assert finished['complete']
    assert finished['updates'] == config['finetune']['epochs']
    assert finished['history'][0]['phase'] == 'finetune'
    assert finished['history'][0]['update'] == 0
    frozen = study.freeze(config, output)
    assert 'cancellation_start.json' in frozen['files']
    assert set(report['archived_files']) <= set(frozen['files'])
    with pytest.raises(FileExistsError):
        START(source, output, config_path)


@pytest.mark.parametrize('damage', ['identity', 'runtime', 'config', 'mesh', 'checkpoint', 'plan', 'frozen'])
def test_cancellation_start_rejects_invalid_sources(source_run, tmp_path, damage):
    _, config_path, source = source_run
    if damage in ('identity', 'runtime'):
        ident = study.read_json(source / 'identity.json')
        ident['source' if damage == 'identity' else 'runtime']['tampered'] = 'bad'
        study.atomic_json(source / 'identity.json', ident)
    elif damage == 'config':
        config = study.read_json(source / 'config.json')
        config['seed'] += 1
        study.atomic_json(source / 'config.json', config)
    elif damage == 'mesh':
        with next((source / 'data').rglob('*.mesh.npz')).open('ab') as stream:
            stream.write(b'changed')
    elif damage == 'checkpoint':
        state = torch.load(source / 'training.pt', weights_only=False)
        state['development_digest'] = 'changed'
        study.atomic_torch(source / 'training.pt', state)
    elif damage == 'plan':
        study.atomic_json(source / 'data_plan.json', {})
    else:
        study.atomic_json(source / 'freeze.json', {})
    with pytest.raises(ValueError):
        START(source, tmp_path / 'new', config_path)
    assert not (tmp_path / 'new').exists()


def test_cancellation_start_audit_tampering_blocks_resume(source_run, tmp_path):
    config, config_path, source = source_run
    output = tmp_path / 'new'
    START(source, output, config_path)
    study.atomic_json(output / 'initialization' / 'validation.json', [{'changed': True}])
    with pytest.raises(ValueError, match='(?i)(initialization|archive|audit)'):
        study.train(config, output, resume=True)
