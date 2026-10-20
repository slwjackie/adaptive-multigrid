"""Actual-operator summaries and matched MLP / non-NN ridge predictors."""
import copy

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg.v67.light_smoother_models import (
    FEATURE_NAMES, FEATURE_DEFINITIONS, Standardizer, TinyMLP, MLPPredictor,
    RidgePredictor, extract_features, fit_ridge, load_predictor,
    save_predictor, predictor_from_dict,
)


def feature_matrix():
    return sp.csr_matrix([[4., -2., -1., .5], [-2., 4., 0., -1.],
                          [-1., 0., 4., -2.], [.5, -1., -2., 4.]])


def training_rows():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 12))
    x[:, 11] = .5
    y = np.stack((1.+.08*x[:, 0], 1.-.04*x[:, 4]), axis=1)
    return x, y


def test_feature_math_direction_convention_and_scale_invariance():
    features = extract_features(feature_matrix(), (2, 2), 1, 5)
    expected = [.25, .5, .0625, -1/3, 58/105, 8/15, 0., 1/13,
                1/13, np.log2(3), np.log2(3), .25]
    np.testing.assert_allclose(features, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(extract_features(7.*feature_matrix(), (2, 2), 1, 5), features)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == len(FEATURE_DEFINITIONS) == 12


def test_actual_wide_line_coupling_is_retained_without_dense_conversion(monkeypatch):
    a = sp.eye(9, format='lil')*10
    a[0, 6] = a[6, 0] = -2.
    a = a.tocsr()
    def forbidden(*args, **kwargs):
        raise AssertionError('A must never be densified')
    monkeypatch.setattr(sp.csr_matrix, 'toarray', forbidden)
    values = extract_features(a, 3, 0, 2)
    assert values[0] == pytest.approx(4/90)
    assert values[1] == values[2] == 0.
    assert values[3] == 1.


@pytest.mark.parametrize('change', ['diagonal', 'nonfinite', 'shape', 'terminal'])
def test_invalid_operator_features_rejected(change):
    a, shape, index, total = feature_matrix(), (2, 2), 0, 2
    if change == 'diagonal':
        a[0, 0] = 0
    elif change == 'nonfinite':
        a[0, 0] = np.nan
    elif change == 'shape':
        shape = (3, 3)
    else:
        index = 1
    with pytest.raises(ValueError):
        extract_features(a, shape, index, total)


def test_standardizer_is_training_only_and_immutable():
    x, _ = training_rows()
    scaler = Standardizer.fit(x, split='train')
    saved = copy.deepcopy(scaler.to_dict())
    standardized = scaler.transform(x)
    np.testing.assert_allclose(standardized.mean(axis=0), 0., atol=1e-15)
    np.testing.assert_allclose(standardized[:, :11].std(axis=0), 1., atol=1e-15)
    assert scaler.scale[-1] == 1.
    scaler.transform(x+100.)  # A held-out value never changes fitting statistics.
    assert scaler.to_dict() == saved
    with pytest.raises(ValueError, match='training'):
        Standardizer.fit(x, split='validation')
    with pytest.raises(ValueError):
        scaler.mean[0] = 5
    assert Standardizer.from_dict(saved).to_dict() == saved


def test_mlp_has_242_parameters_baseline_initialization_and_real_gradient():
    torch.manual_seed(31)
    model = TinyMLP(bounds=(.8, 1.2))
    assert sum(p.numel() for p in model.parameters()) == 12*16+16+16*2+2 == 242
    x, y = training_rows()
    scaler = Standardizer.fit(x)
    z, target = torch.tensor(scaler.transform(x)), torch.tensor(y)
    torch.testing.assert_close(model(z), torch.ones((40, 2), dtype=torch.float64), rtol=0, atol=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    before = ((model(z)-target)**2).mean().item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = ((model(z)-target)**2).mean()
        loss.backward()
        optimizer.step()
    assert ((model(z)-target)**2).mean().item() < before
    result = MLPPredictor(model, scaler).predict(x)
    assert np.all((.8 <= result) & (result <= 1.2))


def test_ridge_fits_stated_regularized_objective_without_inverse_or_torch(monkeypatch):
    x, y = training_rows()
    def forbidden(*args, **kwargs):
        raise AssertionError('Regression must not run an inverse or Torch model')
    monkeypatch.setattr(np.linalg, 'inv', forbidden)
    model = fit_ridge(x, y, regularization=.3)
    z = model.scaler.transform(x)
    residual = z@model.W.T+model.c-(y-1.)
    np.testing.assert_allclose(residual.sum(axis=0), 0., atol=1e-14)
    np.testing.assert_allclose(residual.T@z+.3*model.W, 0., atol=1e-13)
    monkeypatch.setattr(torch, 'as_tensor', forbidden)
    monkeypatch.setattr(torch.nn.Module, '__call__', forbidden)
    values = model.predict(x)
    np.testing.assert_allclose(values, np.clip(1.+z@model.W.T+model.c, .5, 1.5))
    assert model.parameter_count == model.W.size+model.c.size == 26
    assert model.kind == 'regression'
    with pytest.raises(ValueError, match='training'):
        fit_ridge(x, y, split='final')


@pytest.mark.parametrize('kind', ['mlp', 'regression'])
def test_predictor_roundtrip_and_bounds(tmp_path, kind):
    x, y = training_rows()
    if kind == 'mlp':
        model = TinyMLP(bounds=(.8, 1.2))
        with torch.no_grad():
            model.output.bias.copy_(torch.tensor([100., -100.]))
        predictor = MLPPredictor(model, Standardizer.fit(x), {'train_digest': 'training-only'})
    else:
        predictor = RidgePredictor(np.zeros((2, 12)), [100., -100.],
                                   Standardizer.fit(x), bounds=(.8, 1.2),
                                   metadata={'train_digest': 'training-only'})
    expected = np.tile([1.2, .8], (len(x), 1))
    np.testing.assert_array_equal(predictor.predict(x), expected)
    save_predictor(predictor, tmp_path/'model.json')
    restored = load_predictor(tmp_path/'model.json')
    assert restored.to_dict() == predictor.to_dict()
    np.testing.assert_array_equal(restored.predict(x), expected)
    assert restored.predict(x[0]).shape == (2,)
    tampered = copy.deepcopy(predictor.to_dict())
    tampered['feature_spec']['names'][0] = 'wrong'
    with pytest.raises(ValueError, match='definition'):
        predictor_from_dict(tampered)


@pytest.mark.parametrize('bounds', [(0., 1.2), (.8, .9), (1.1, 1.4), (1., 1.), (.8, np.nan)])
def test_bounds_require_valid_baseline_inclusion(bounds):
    with pytest.raises(ValueError, match='bounds'):
        TinyMLP(bounds=bounds)
