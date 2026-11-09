import copy
import json
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from neural_amg import study


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    c = study.load_config(root/'configs/neural_amg_smoke.json')
    c['model']['hidden'] = 4
    c['teacher'].update(count=3, sweeps=2, maxiter=20)
    c['solve']['max_cycles'] = 80
    return c


def test_training_resume_freeze_final_and_tampering(tmp_path, config):
    run = tmp_path/'run'
    partial = study.train(config, run, max_updates=1)
    assert not partial['complete']
    with pytest.raises(ValueError, match='Complete training'):
        study.freeze(config, run)
    assert not (run/'data_final.json').exists()
    done = study.train(config, run, resume=True)
    assert done['complete'] and done['updates'] == 2
    frozen = study.freeze(config, run)
    assert frozen['selection'].endswith('setup_plus_solve')
    evaluated = study.evaluate(config, run)
    assert evaluated['complete']
    assert {r['split'] for r in evaluated['results']} == {'final','ood_grid'}
    assert {r['method'] for r in evaluated['results']} == {'classical','teacher','learned'}
    assert study.evaluate(config, run, resume=True) == evaluated
    with pytest.raises(ValueError, match='Frozen runs'):
        study.train(config, run, resume=True)
    with (run/'frozen_model.pt').open('ab') as stream:
        stream.write(b'x')
    with pytest.raises(ValueError, match='Frozen file changed'):
        study.evaluate(config, run, resume=True)


def test_resume_rejects_config_and_data_changes(tmp_path, config):
    run = tmp_path/'run'
    data = study.prepare(config, run)
    modified = copy.deepcopy(config); modified['model']['feature_mode']='symmetric9'
    with pytest.raises(ValueError, match='resume rejected'):
        study.prepare(modified, run, resume=True)
    ap, _ = study._case_paths(run, data['train'][0]['spec'])
    with ap.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Cached case content changed'):
        study.prepare(config, run, resume=True)


def test_external_npz_arbitrary_node_count_and_overlap(tmp_path, config):
    # Eleven unknowns cannot be reshaped to the previous square-grid CNN input.
    for name, scale in [('tr',1),('va',2),('fi',3)]:
        a = sp.diags([-np.ones(10), np.linspace(3,4+scale,11), -np.ones(10)], [-1,0,1], format='csr')
        sp.save_npz(tmp_path/f'{name}.npz', a)
    manifest = tmp_path/'operators.json'
    manifest.write_text(json.dumps({'splits': {'train':[{'A':'tr.npz'}],
        'validation':[{'A':'va.npz'}], 'final':[{'A':'fi.npz'}]}}))
    config['data'] = dict(kind='npz_manifest', manifest=str(manifest), seed=15)
    data = study.prepare(config, tmp_path/'run')
    assert study._load_case(tmp_path/'run', data['train'][0]).a.shape == (11,11)
    assert not (tmp_path/'run'/'data_final.json').exists()
    manifest.write_text(json.dumps({'splits': {'train':[{'A':'tr.npz'}],
        'validation':[{'A':'tr.npz'}], 'final':[{'A':'fi.npz'}]}}))
    with pytest.raises(ValueError, match='Duplicate normalized'):
        study.prepare(config, tmp_path/'overlap')


def test_resumed_optimizer_trajectory_equals_uninterrupted(tmp_path, config, monkeypatch):
    # Timing-based selection is nondeterministic; test training parameters and
    # optimizer trajectory independently of elapsed-time variation.
    full = study.train(config, tmp_path/'full')
    study.train(config, tmp_path/'resumed', max_updates=1)
    resumed = study.train(config, tmp_path/'resumed', resume=True)
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)


def test_source_identity_is_nonempty_and_complete():
    hashes = study._source_identity()
    assert 'neural_amg/model.py' in hashes
    assert 'neural_amg/solver.py' in hashes
    assert 'adaptive_mg/v67/autograd_sparse.py' in hashes
    assert all(len(value) == 64 for value in hashes.values())


def test_checkpoint_identity_rejected(tmp_path, config):
    run = tmp_path/'run'
    study.train(config, run, max_updates=1)
    state = torch.load(run/'training.pt', map_location='cpu', weights_only=False)
    state['identity']['config'] = 'other-run'
    torch.save(state, run/'training.pt')
    with pytest.raises(ValueError, match='checkpoint identity'):
        study.train(config, run, resume=True)


def test_invalid_benchmark_methods_rejected(config):
    for methods in ([], ['learned','learned'], ['unknown'], ['classical']):
        config['benchmark']['methods'] = methods
        with pytest.raises(ValueError, match='Benchmark methods'):
            study.validate_config(config)


def test_failed_nonfinite_benchmark_is_json_safe(tmp_path, config, monkeypatch):
    data = study.prepare(config, tmp_path/'run')
    monkeypatch.setattr(study, '_hierarchy', lambda *args, **kwargs: object())
    monkeypatch.setattr(study, '_solve', lambda *args: dict(
        x=np.array([np.nan]), converged=False, cycles=2,
        residuals=[1.0, float('inf'), float('nan')]))
    rows = study.benchmark(config, data['validation'], tmp_path/'run',
                           study.GraphAttentionTransfer(**config['model']))
    assert not rows[0]['converged']
    assert rows[0]['repetitions'][0]['residuals'] == [1.0, None, None]
    json.dumps(rows, allow_nan=False)
