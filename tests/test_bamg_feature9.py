"""Numerical contracts for the opt-in input ablation (legacy files untouched)."""
from pathlib import Path
import sys

import numpy as np
import pytest
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from adaptive_mg.v67 import bamg_transfer as legacy
from adaptive_mg.v67 import bamg_solver as solver
from adaptive_mg.v67 import bamg_energy_cnn_training as arm
from adaptive_mg.v67 import bamg_energy_cnn_study as paired
from adaptive_mg.v67.autograd_sparse import SparseTensor
from bamg_feature9 import (OFFSETS_8, Feature9WeightCNN, direct_features9,
                          transfer_legacy_weights)
from bamg_feature9_adapter import feature9_mode, model_from_spec, initialize_weight


def operator():
    a = np.diag(np.arange(9, dtype=float) + 12)
    a[0, 0], a[1, 1] = 4, 9
    a[0, 1] = a[1, 0] = -3
    a[1, 2] = a[2, 1] = 2
    a[0, 3] = a[3, 0] = -.5
    assert np.linalg.eigvalsh(a).min() > 0
    return sp.csr_matrix(a)


@pytest.mark.parametrize('mode', ['row9', 'symmetric9'])
def test_exact_formula_signed_symmetry_padding_and_scaling(mode):
    a = operator()
    before = a.copy()
    x = direct_features9(a, 3, feature_mode=mode).numpy()[0]
    diag = a.diagonal()
    np.testing.assert_allclose(x[0].ravel(), np.log(diag / np.median(diag) + 1e-12))
    for channel, (oi, oj) in enumerate(OFFSETS_8, 1):
        for i in range(9):
            ii, jj = i // 3 + oi, i % 3 + oj
            expected = 0.
            if 0 <= ii < 3 and 0 <= jj < 3:
                j = ii * 3 + jj
                denominator = diag[i] if mode == 'row9' else np.sqrt(diag[i] * diag[j])
                expected = a[i, j] / denominator
            assert x[channel, i // 3, i % 3] == pytest.approx(expected)
    right, left = OFFSETS_8.index((0, 1)) + 1, OFFSETS_8.index((0, -1)) + 1
    assert x[right, 0, 0] == pytest.approx(-.75 if mode == 'row9' else -.5)
    assert x[left, 0, 1] == pytest.approx(-1/3 if mode == 'row9' else -.5)
    assert x[right, 0, 1] > 0  # signed positive Q1-style coupling is retained
    np.testing.assert_allclose(direct_features9(a * 11, 3, feature_mode=mode), x[None])
    assert (a != before).nnz == 0


def test_row9_is_exactly_legacy_without_center_including_diagonal_floor():
    for scale in (1., 1e-16):
        a = operator() * scale
        old = legacy.direct_features(a, 3)
        new = direct_features9(a, 3, feature_mode='row9')
        torch.testing.assert_close(new, old[:, [0, *range(2, 10)]], rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['row9', 'symmetric9'])
def test_sparse_tensor_matches_scipy_and_gradients_include_both_diagonals(mode):
    sparse = SparseTensor.from_scipy(operator(), requires_grad=True)
    actual = direct_features9(sparse, 3, feature_mode=mode)
    torch.testing.assert_close(actual, direct_features9(operator(), 3, feature_mode=mode))
    assert torch.autograd.gradcheck(
        lambda values: direct_features9(SparseTensor(sparse.pattern, values), 3, feature_mode=mode),
        (sparse.values,), eps=1e-6, atol=1e-6, rtol=1e-4)
    channel = OFFSETS_8.index((0, 1)) + 1
    grad, = torch.autograd.grad(actual[0, channel, 0, 0], sparse.values)
    d0 = grad[(sparse.rows == 0) & (sparse.cols == 0)].item()
    d1 = grad[(sparse.rows == 1) & (sparse.cols == 1)].item()
    assert d0 == pytest.approx(3 / 16 if mode == 'row9' else .5 / (2 * 4))
    assert d1 == pytest.approx(0 if mode == 'row9' else .5 / (2 * 9))


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf')])
def test_invalid_diagonal_fails_closed(bad):
    a = operator()
    a[0, 0] = bad
    with pytest.raises(ValueError):
        direct_features9(a, 3, feature_mode='symmetric9')


@pytest.mark.parametrize('mode', ['row9', 'symmetric9'])
@pytest.mark.parametrize('mask_mode', ['early', 'late', 'none'])
def test_model_warm_transfer_preserves_other_tensors_and_checkpoint_spec(mode, mask_mode):
    old = legacy.WeightCNN(hidden=2, depth=2, mask_mode=mask_mode).double()
    model = Feature9WeightCNN(hidden=2, depth=2, mask_mode=mask_mode, feature_mode=mode).double()
    transfer_legacy_weights(model, old)
    assert model.backbone[0].in_channels == (28 if mask_mode == 'early' else 12)
    pattern = legacy.direct_pattern(7)
    mask = legacy.fixed_bilinear_mask(pattern)
    features = direct_features9(sp.eye(49, format='csr'), 7, feature_mode=mode)
    output = model(features, mask)
    assert output.shape == (1, 16, 7, 7)
    restored = model_from_spec(model.spec).double()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(output, restored(features, mask), rtol=0, atol=0)
    for key, value in old.state_dict().items():
        target = model.state_dict()[key]
        if key == 'backbone.0.weight':
            value = value[:, [0, *range(2, value.shape[1])]]
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    projected = legacy.project_weights(output, mask, pattern)
    rowsum, injection, _ = legacy.geometry_constraints(pattern)
    np.testing.assert_allclose(projected.detach().numpy().sum(1), rowsum, atol=1e-14)
    assert bool((projected[~mask] == 0).all())
    assert np.all(mask.sum(1).numpy() <= 4)


def test_adapter_restores_legacy_and_keeps_original_source_hashes(tmp_path):
    original = (legacy.model_from_spec, solver._weights, arm.WeightCNN, paired._sources)
    hashes = paired._sources()
    old = legacy.WeightCNN().double()
    path = tmp_path / 'old.pt'
    torch.save(dict(weight_spec=old.spec, weight_state=old.state_dict()), path)
    with pytest.raises(RuntimeError, match='test exit'):
        with feature9_mode():
            assert isinstance(legacy.model_from_spec(old.spec), original[2])
            hashes9 = paired._sources()
            assert all(hashes9[name] == sha for name, sha in hashes.items())
            assert 'scripts/bamg_feature9.py' in hashes9
            new = Feature9WeightCNN(feature_mode='symmetric9').double()
            info = initialize_weight(new, path)
            assert info['function_preserving'] is False
            assert info['optimizer_reused'] is False
            assert info['target_weight_spec'] == new.spec
            with pytest.raises(RuntimeError, match='nested'):
                with feature9_mode():
                    pass
            raise RuntimeError('test exit')
    assert original == (legacy.model_from_spec, solver._weights, arm.WeightCNN, paired._sources)
    assert hashes == paired._sources()
