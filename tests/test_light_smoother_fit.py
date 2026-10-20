"""Shared-label fitting, exact interrupted Adam resume and provenance guards."""
import json

import numpy as np
import pytest
import torch

from adaptive_mg.v67 import light_smoother_fit as fit
from adaptive_mg.v67.limited import digest_file


def data():
    rng = np.random.default_rng(512)
    x = rng.normal(size=(31, 12))
    x[:, -1] = .5
    y = 1. + .09*np.column_stack((np.tanh(x[:, 0]+x[:, 3]), np.tanh(x[:, 1]-x[:, 4])))
    return x, y


def settings():
    return dict(updates=9, batch_size=7, learning_rate=.01, gradient_clip=1.,
                ridge_lambda=.05, seed=81, bounds=[.8, 1.2])


def provenance():
    return dict(split='train', p_signature='fixed-imported-P',
                search_records=['row-A', 'row-B'], source_digest='immutable-source-snapshot')


def test_fit_uses_same_training_rows_and_standardizer_and_exports_verified_pair(tmp_path):
    x, y = data()
    result = fit.fit_models(x, y, tmp_path/'run', settings(), provenance())
    assert set(result) == {'mlp', 'regression'}
    a, b = result['mlp'], result['regression']
    assert a.parameter_count == 242 and b.parameter_count == 26
    assert a.scaler.to_dict() == b.scaler.to_dict()
    np.testing.assert_array_equal(a.scaler.mean, x.mean(0))
    assert a.scaler.count == len(x) and a.scaler.scale[-1] == 1.
    assert a.metadata == b.metadata and a.metadata['shared_label_data']
    assert a.metadata['split'] == 'train'
    for model in result.values():
        assert model.predict(x).shape == y.shape
        assert np.max(np.abs(model.predict(x)-1.)) <= .2 + 1e-15
    status = json.loads((tmp_path/'run/status.json').read_text())
    assert status['status'] == 'complete' and status['all_updates_completed']
    assert status['updates'] == 9 and status['offline_fit_seconds'] > 0
    assert status['coefficient_mse']['regression'] < .001
    restored = fit.load_fitted(tmp_path/'run')
    assert restored['mlp'].to_dict() == a.to_dict()


def test_interrupted_resume_preserves_exact_adam_trajectory(tmp_path, monkeypatch):
    x, y = data()
    straight = fit.fit_models(x, y, tmp_path/'straight', settings(), provenance())
    original = torch.optim.Adam.step
    calls = [0]

    def interrupted_step(optimizer, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 4:
            original(optimizer, *args, **kwargs)
            raise KeyboardInterrupt('interruption after mutation but before fourth commit')
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, 'step', interrupted_step)
    with pytest.raises(KeyboardInterrupt):
        fit.fit_models(x, y, tmp_path/'resumed', settings(), provenance())
    state = torch.load(tmp_path/'resumed/resume.pt', weights_only=True)
    assert state['updates'] == 3
    monkeypatch.setattr(torch.optim.Adam, 'step', original)
    resumed = fit.fit_models(x, y, tmp_path/'resumed', settings(), provenance(), resume=True)
    for name in ('mlp', 'regression'):
        assert resumed[name].to_dict() == straight[name].to_dict()
    assert json.loads((tmp_path/'straight/history.json').read_text()) == json.loads((tmp_path/'resumed/history.json').read_text())
    before = {name: digest_file(tmp_path/'resumed'/name)
              for name in ('mlp.json', 'regression.json', 'status.json', 'history.json', 'resume.pt')}
    fit.fit_models(x, y, tmp_path/'resumed', settings(), provenance(), resume=True)
    assert before == {name: digest_file(tmp_path/'resumed'/name) for name in before}


@pytest.mark.parametrize('split', ['validation', 'final', 'ood_grid', None])
def test_nontraining_rows_cannot_fit(tmp_path, split):
    x, y = data()
    source = provenance()
    source['split'] = split
    with pytest.raises(ValueError, match='train-labelled'):
        fit.fit_models(x, y, tmp_path/'run', settings(), source)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('change', ['features', 'labels', 'provenance', 'settings'])
def test_resume_rejects_changed_data_and_protocol(tmp_path, change):
    x, y = data()
    source, spec = provenance(), settings()
    fit.fit_models(x, y, tmp_path/'run', spec, source)
    if change == 'features':
        x[0, 0] += .01
    elif change == 'labels':
        y[0, 0] += .01
    elif change == 'provenance':
        source['p_signature'] = 'changed-P'
    else:
        spec['ridge_lambda'] *= 2
    with pytest.raises(ValueError, match='identical inputs/protocol'):
        fit.fit_models(x, y, tmp_path/'run', spec, source, resume=True)


def test_completed_checkpoint_tampering_is_rejected(tmp_path):
    x, y = data()
    fit.fit_models(x, y, tmp_path/'run', settings(), provenance())
    path = tmp_path/'run/regression.json'
    document = json.loads(path.read_text())
    document['c'][0] += .1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='checkpoint changed'):
        fit.load_fitted(tmp_path/'run')


def test_failed_nonfinite_data_or_out_of_bounds_label_does_not_create_run(tmp_path):
    x, y = data()
    x[0, 0] = np.inf
    with pytest.raises(ValueError, match='finite'):
        fit.fit_models(x, y, tmp_path/'badx', settings(), provenance())
    x, y = data()
    y[0, 0] = 2
    with pytest.raises(ValueError, match='bounds'):
        fit.fit_models(x, y, tmp_path/'bady', settings(), provenance())
    assert not (tmp_path/'badx').exists() and not (tmp_path/'bady').exists()
