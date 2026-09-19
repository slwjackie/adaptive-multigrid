import numpy as np
import pytest
import torch

from adaptive_mg.detector_features import detector_features_numpy, detector_features_torch
from adaptive_mg.models import NeuralComponents
from adaptive_mg.pde import DiffusionCase, assemble_problem
from adaptive_mg.smoothers import LineSmootherCache, classical_smoothing_correction
from adaptive_mg.solver import MGConfig, solve
from adaptive_mg.strategy import STRATEGIES, strategy_index
from adaptive_mg.transfer import (
    MAX_TRANSFER_CANDIDATES,
    baseline_weights,
    build_transfer_pattern,
    galerkin_coarse_operator,
    matrix_feature_array,
    weights_from_deltas_torch,
)


def test_expanded_transfer_support_has_16_candidates_and_can_activate_new_edges() -> None:
    case = DiffusionCase(n=15)
    a, _, _ = assemble_problem(case)
    pattern = build_transfer_pattern(case.n)
    assert pattern.n_candidates == MAX_TRANSFER_CANDIDATES == 16
    base = baseline_weights(a, case.n, "bilinear")
    # Pick a fine-only point; geometric coarse points are deliberately locked
    # to exact injection in v3.
    i = case.n // 2 - 1
    j = case.n // 2
    row = i * case.n + j
    zero_slots = np.flatnonzero((pattern.columns[row] >= 0) & (np.abs(base[row]) < 1.0e-15))
    assert zero_slots.size > 0
    deltas = torch.zeros((1, 16, case.n, case.n))
    slot = int(zero_slots[0])
    deltas[0, slot, i, j] = 5.0
    learned = weights_from_deltas_torch(pattern, deltas, base).detach().numpy()
    assert abs(learned[row, slot]) > 1.0e-6
    assert np.allclose(learned.sum(axis=1), base.sum(axis=1), atol=1.0e-6)


@pytest.mark.parametrize(
    ("coarsening", "expected"),
    [("semi_x", (7, 15)), ("semi_y", (15, 7)), ("alternating", (7, 15))],
)
def test_semicoarsening_builds_rectangular_tied_galerkin_hierarchy(coarsening, expected) -> None:
    case = DiffusionCase(n=15, epsilon=1.0e-3, angle_deg=0.0)
    a, _, _ = assemble_problem(case)
    pattern = build_transfer_pattern(case.n, coarsening=coarsening, level_index=0)
    assert pattern.coarse_shape == expected
    base = baseline_weights(a, (15, 15), "operator", coarse=expected)
    from adaptive_mg.transfer import scipy_prolongation_from_weights

    p = scipy_prolongation_from_weights(pattern, base)
    ac = galerkin_coarse_operator(a, p).toarray()
    assert np.allclose(ac, ac.T, atol=1.0e-10)
    assert np.linalg.eigvalsh(ac).min() > 0.0


def test_numpy_and_torch_detector_features_match_in_reverse_smoothing() -> None:
    rng = np.random.default_rng(9)
    case = DiffusionCase(n=7, epsilon=1.0e-3, angle_deg=33.0, pattern="local_patch")
    a, _, _ = assemble_problem(case)
    residual = rng.normal(size=case.n * case.n)
    features = matrix_feature_array(a, case.n)
    cache = LineSmootherCache(a, (case.n, case.n))
    correction = classical_smoothing_correction(
        a, (case.n, case.n), residual, "line_alt", cache=cache, reverse=True
    )
    post = residual - a @ correction
    numpy_features = detector_features_numpy(
        features, a.diagonal().reshape(case.n, case.n), residual, post
    )
    torch_features = detector_features_torch(
        torch.from_numpy(features).unsqueeze(0),
        torch.from_numpy(a.diagonal()),
        torch.from_numpy(residual),
        torch.from_numpy(np.asarray(post)),
        (case.n, case.n),
    ).numpy()[0]
    assert np.max(np.abs(numpy_features - torch_features)) < 1.0e-5


def test_patch_selective_neural_execution_skips_easy_tiles_and_runs_hard_tiles() -> None:
    case = DiffusionCase(n=15, epsilon=1.0e-3, angle_deg=45.0, pattern="local_patch")
    a, b, _ = assemble_problem(case)
    components = NeuralComponents.create(hidden=8)
    low = solve(
        a,
        b,
        case.n,
        components,
        MGConfig(
            mode="adaptive",
            max_cycles=1,
            tolerance=0.0,
            safety_fallback=False,
            smoother_safeguard=False,
            selective_nn=True,
            patch_size=4,
        ),
    )
    assert low.active_patch_fraction == 0.0
    assert low.smoother_nn_calls == 0
    assert low.transfer_nn_calls == 0
    assert low.detector_nn_calls > 0

    with torch.no_grad():
        components.detector.head.bias.fill_(10.0)
    high = solve(
        a,
        b,
        case.n,
        components,
        MGConfig(
            mode="adaptive",
            max_cycles=1,
            tolerance=0.0,
            safety_fallback=False,
            smoother_safeguard=False,
            selective_nn=True,
            patch_size=4,
        ),
    )
    assert high.active_patch_fraction > 0.99
    assert high.smoother_nn_calls > 0
    assert high.transfer_nn_calls > 0


def test_adaptive_refresh_reuses_hierarchy_between_cycles() -> None:
    case = DiffusionCase(n=15, epsilon=1.0e-3, angle_deg=45.0, pattern="local_patch")
    a, b, _ = assemble_problem(case)
    components = NeuralComponents.create(hidden=8)
    with torch.no_grad():
        components.detector.head.bias.fill_(10.0)
    every = solve(
        a, b, case.n, components,
        MGConfig(
            mode="adaptive", max_cycles=3, tolerance=0.0,
            safety_fallback=False, adaptive_refresh=1, patch_size=8,
        ),
    )
    cached = solve(
        a, b, case.n, components,
        MGConfig(
            mode="adaptive", max_cycles=3, tolerance=0.0,
            safety_fallback=False, adaptive_refresh=3, patch_size=8,
        ),
    )
    assert cached.hierarchy_rebuilds < every.hierarchy_rebuilds
    assert cached.hierarchy_rebuilds > 0


def test_global_selector_controls_classical_strategy_before_local_adaptation() -> None:
    case = DiffusionCase(n=7, epsilon=1.0e-3, angle_deg=0.0)
    a, b, _ = assemble_problem(case)
    components = NeuralComponents.create(hidden=8)
    target = "line_x_operator_semi_y"
    with torch.no_grad():
        components.selector.head.weight.zero_()
        components.selector.head.bias.fill_(-10.0)
        components.selector.head.bias[strategy_index(target)] = 10.0
    result = solve(
        a,
        b,
        case.n,
        components,
        MGConfig(
            mode="adaptive",
            global_selection="selector",
            strategy_name=None,
            max_cycles=1,
            tolerance=0.0,
            safety_fallback=False,
        ),
    )
    assert result.selected_strategy == target
    assert result.selector_nn_calls == 1
    assert target in {strategy.name for strategy in STRATEGIES}


def test_raw_and_safeguarded_runs_expose_alpha_and_fallback_statistics() -> None:
    case = DiffusionCase(n=7, epsilon=1.0e-3, angle_deg=45.0, pattern="local_patch")
    a, b, _ = assemble_problem(case)
    components = NeuralComponents.create(hidden=8)
    with torch.no_grad():
        torch.nn.init.normal_(components.smoother.head.weight, std=0.2)
        torch.nn.init.normal_(components.smoother.head.bias, std=0.2)
        components.detector.head.bias.fill_(10.0)
    raw = solve(
        a, b, case.n, components,
        MGConfig(
            mode="adaptive", max_cycles=2, tolerance=0.0,
            smoother_safeguard=False, safety_fallback=False,
        ),
    )
    safe = solve(
        a, b, case.n, components,
        MGConfig(
            mode="adaptive", max_cycles=2, tolerance=0.0,
            smoother_safeguard=True, safety_fallback=True,
        ),
    )
    assert raw.safeguard_alphas == []
    assert safe.safeguard_alphas
    assert all(0.0 <= alpha <= 1.0 for alpha in safe.safeguard_alphas)
    assert 0.0 <= safe.safeguard_active_fraction <= 1.0
    assert safe.relative_residual <= raw.relative_residual * (1.0 + 1.0e-10)
