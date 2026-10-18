"""Fixed EM setup budgets must agree in runtime and differentiable training."""
from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp

from adaptive_mg import DiffusionCase, MGConfig, assemble_stiffness
from adaptive_mg.energymin import energy_weights
from adaptive_mg.hierarchy import WorkStats, build_fixed_hierarchy
from adaptive_mg.strategy import get_strategy
from adaptive_mg.transfer import (baseline_weights, build_transfer_pattern,
                                 coarse_fine_indices)
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.unroll import make_graph


def test_fixed_budget_ignores_relative_tolerance_without_changing_legacy_default():
    a = assemble_stiffness(DiffusionCase(15, epsilon=.03, angle_deg=35.,
                                       pattern='checkerboard', contrast=100.))
    pattern = build_transfer_pattern(15)
    loose, loose_report = energy_weights(a, pattern, maxiter=5, rtol=.9,
                                        fixed_iterations=True)
    strict, strict_report = energy_weights(a, pattern, maxiter=5, rtol=1e-12,
                                          fixed_iterations=True)
    np.testing.assert_array_equal(loose, strict)
    for report in (loose_report, strict_report):
        assert report['iterations'] == report['iteration_budget'] == 5
        assert report['fixed_iterations']
        assert report['stop_reason'] == 'fixed_budget_exhausted'
        assert report['energy_after'] <= report['energy_before']
    legacy, legacy_report = energy_weights(a, pattern, maxiter=5, rtol=.9)
    explicit, explicit_report = energy_weights(a, pattern, maxiter=5, rtol=.9,
                                               fixed_iterations=False)
    np.testing.assert_array_equal(legacy, explicit)
    assert legacy_report['iterations'] == explicit_report['iterations'] < 5
    assert not legacy_report['fixed_iterations']
    assert not np.array_equal(legacy, strict)
    np.testing.assert_allclose(strict.sum(1), pattern.bilinear_weights.sum(1), atol=1e-14)
    fixed = coarse_fine_indices(pattern)
    np.testing.assert_array_equal(strict[fixed], pattern.bilinear_weights[fixed])
    assert np.count_nonzero(strict[pattern.bilinear_weights == 0]) == 0


def test_stationary_interpolation_does_not_invent_five_iterations():
    # Identity energy is already stationary for equal row-sum-constrained
    # weights on this 3x3 -> 1x1 support, which has no free tangent variables.
    pattern = build_transfer_pattern(3, (1, 1))
    weights, report = energy_weights(sp.eye(9, format='csr'), pattern,
                                     maxiter=5, fixed_iterations=True)
    np.testing.assert_array_equal(weights, pattern.bilinear_weights)
    assert report['iterations'] == 0
    assert report['iteration_budget'] == 5
    assert report['stop_reason'] == 'stationary_initial_interpolation'


def test_stationary_cg_does_not_divide_by_zero_or_pad_iteration_count():
    # Z.T @ I @ Z is identity: the projected minimizer is reached in one step.
    pattern = build_transfer_pattern(7)
    weights, report = energy_weights(sp.eye(49, format='csr'), pattern,
                                     maxiter=5, fixed_iterations=True)
    assert 0 < report['iterations'] < 5
    assert report['stop_reason'] == 'stationary_at_machine_precision'
    assert report['optimizer_converged']
    assert np.isfinite(weights).all()


def test_fixed_em_configuration_reaches_runtime_and_unrolled_parent_at_all_levels():
    n = 15
    a = assemble_stiffness(DiffusionCase(n, epsilon=.04, angle_deg=28.))
    mg = MGConfig(strategy_name='line_alt_energymin_full__em5__v11',
                  em_fixed_iterations=True, em_rtol=.9, stencil_backend='csr')
    assert mg.em_iterations == 5
    cfg = AdaptiveConfig(mg=mg, mode='research', branch='H_P',
                         use_smoother=False, transfer_levels='all')
    assert AdaptiveConfig.from_dict(cfg.to_dict()) == cfg
    runtime = build_fixed_hierarchy(a, (n, n), get_strategy(mg.strategy_name),
                                    mg, WorkStats())
    graph = make_graph(a, (n, n), Components.create(hidden=4), cfg, learned=False)
    direct = baseline_weights(a, n, 'energymin', em_iterations=5, em_rtol=.9,
                              em_fixed_iterations=True)
    np.testing.assert_array_equal(runtime.base_weights, direct)
    levels = 0
    while runtime.coarse is not None:
        assert runtime.transfer_setup_report['fixed_iterations']
        assert runtime.transfer_setup_report['iteration_budget'] == 5
        np.testing.assert_allclose(runtime.base_weights, graph.base_weights,
                                   rtol=2e-12, atol=2e-14)
        runtime, graph = runtime.coarse, graph.coarse
        levels += 1
    assert levels == 2
    with pytest.raises(ValueError, match='boolean'):
        replace(mg, em_fixed_iterations=1)
    with pytest.raises(ValueError, match='boolean'):
        energy_weights(a, build_transfer_pattern(n), fixed_iterations='yes')
