import numpy as np
import pytest
import torch

from adaptive_mg.models import NeuralComponents
from adaptive_mg.pde import DiffusionCase, assemble_problem
from adaptive_mg.transfer import (
    N_MATRIX_FEATURES,
    baseline_prolongation,
    baseline_weights,
    build_transfer_pattern,
    dense_prolongation_torch,
    galerkin_coarse_operator,
    matrix_feature_array,
    matrix_features_torch,
    validate_multigrid_size,
    weights_from_deltas_torch,
)


def test_transfer_pattern_is_cached_and_features_have_no_extra_channel() -> None:
    assert build_transfer_pattern(7) is build_transfer_pattern(7)
    a, _, _ = assemble_problem(DiffusionCase(n=7))
    features = matrix_feature_array(a, 7)
    assert N_MATRIX_FEATURES == 10
    assert features.shape == (10, 7, 7)
    assert np.all(np.isfinite(features))


def test_zero_initialized_decoder_equals_each_selected_baseline_and_is_tied() -> None:
    case = DiffusionCase(n=7, epsilon=1.0e-2, angle_deg=45.0, pattern="uniform")
    a, _, _ = assemble_problem(case)
    components = NeuralComponents.create(hidden=16).eval()
    pattern = build_transfer_pattern(case.n)
    with torch.no_grad():
        deltas = components.transfer(matrix_features_torch(a, case.n))
        for baseline in ("bilinear", "operator"):
            base = baseline_weights(a, case.n, baseline)
            weights = weights_from_deltas_torch(pattern, deltas, base)
            p_dense = dense_prolongation_torch(pattern, weights).numpy()
            p_classical = baseline_prolongation(a, case.n, baseline).toarray()
            assert np.allclose(p_dense, p_classical, atol=1.0e-7)
            ac = p_dense.T @ a.toarray() @ p_dense
            assert np.allclose(ac, ac.T, atol=1.0e-10)
            assert np.linalg.eigvalsh(ac).min() > 0.0


def test_operator_dependent_prolongation_is_finite_and_full_rank() -> None:
    case = DiffusionCase(
        n=7, epsilon=1.0e-3, angle_deg=37.0, contrast=1.0e3, pattern="local_patch"
    )
    a, _, _ = assemble_problem(case)
    p = baseline_prolongation(a, case.n, "operator")
    assert np.all(np.isfinite(p.data))
    assert np.linalg.matrix_rank(p.toarray()) == p.shape[1]
    ac = galerkin_coarse_operator(a, p).toarray()
    assert np.linalg.eigvalsh(ac).min() > 0.0



def test_vectorized_stencil_features_match_sparse_entries() -> None:
    n = 7
    a, _, _ = assemble_problem(DiffusionCase(n=n, epsilon=1.0e-2, angle_deg=31.0))
    features = matrix_feature_array(a, n)
    diagonal = np.maximum(np.abs(a.diagonal()), 1.0e-14)
    offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for channel, (di, dj) in enumerate(offsets, start=1):
        for i in range(n):
            for j in range(n):
                ci, cj = i + di, j + dj
                expected = 0.0
                if 0 <= ci < n and 0 <= cj < n:
                    expected = a[i * n + j, ci * n + cj] / diagonal[i * n + j]
                assert np.isclose(features[channel, i, j], expected, atol=1.0e-6)

def test_invalid_multigrid_size_fails_early_with_clear_message() -> None:
    with pytest.raises(ValueError, match=r"n=2\*\*L-1"):
        validate_multigrid_size(9, 3)


def test_learned_transfer_preserves_coarse_injection_and_full_rank_under_extreme_deltas() -> None:
    rng = np.random.default_rng(123)
    for coarsening in ("full", "semi_x", "semi_y", "alternating"):
        case = DiffusionCase(n=15, epsilon=1.0e-3, angle_deg=41.0)
        a, _, _ = assemble_problem(case)
        pattern = build_transfer_pattern(case.n, coarsening=coarsening, level_index=0)
        base = baseline_weights(a, (15, 15), "bilinear", coarse=pattern.coarse_shape)
        deltas = torch.as_tensor(
            rng.normal(scale=20.0, size=(1, 16, 15, 15)), dtype=torch.float64
        )
        weights = weights_from_deltas_torch(pattern, deltas, base).detach().numpy()
        p = dense_prolongation_torch(
            pattern, torch.as_tensor(weights, dtype=torch.float64)
        ).numpy()
        from adaptive_mg.transfer import _coarse_fine_indices

        coarse_rows = _coarse_fine_indices(pattern)
        assert np.allclose(weights[coarse_rows], base[coarse_rows], atol=0.0)
        assert np.linalg.matrix_rank(p) == p.shape[1]
