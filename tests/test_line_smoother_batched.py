import numpy as np
import pytest
import scipy.sparse as sp

from adaptive_mg.pde import DiffusionCase, assemble_stiffness
from adaptive_mg.smoothers import (LineSmootherCache, _directional, classical_smoothing_correction)
from adaptive_mg.transfer import (baseline_weights, build_transfer_pattern, galerkin_coarse_operator,
                                  scipy_prolongation_from_weights)
from adaptive_mg.grid import next_shape


def _sequential(a, shape, residual, smoother, reverse):
    cache = LineSmootherCache(a, shape)
    directions = {'line_x': ('x',), 'line_y': ('y',), 'line_diag45': ('diag45',),
                  'line_alt': ('y', 'x') if reverse else ('x', 'y')}[smoother]
    correction, current = np.zeros_like(residual), residual.copy()
    for direction in directions:
        correction, current = _directional(correction, current, cache.get(direction),
                                           reverse=reverse, work=None, safety=False)
    return correction


def _operators():
    n = 15
    fine = assemble_stiffness(DiffusionCase(n=n, epsilon=1e-3, angle_deg=45., contrast=1e3, pattern='channel'))
    coarse_shape = next_shape((n, n), 'full', 3, level_index=0)
    pattern = build_transfer_pattern((n, n), coarse_shape)
    p = scipy_prolongation_from_weights(pattern, baseline_weights(fine, (n, n), 'operator', coarse=coarse_shape))
    return [((n, n), fine), (coarse_shape, galerkin_coarse_operator(fine, p).tocsr())]


@pytest.mark.parametrize('smoother', ['line_x', 'line_y', 'line_alt', 'line_diag45'])
@pytest.mark.parametrize('reverse', [False, True])
def test_batched_zebra_equals_sequential_sweep(smoother, reverse):
    rng = np.random.default_rng(3)
    for shape, a in _operators():
        residual = rng.standard_normal(a.shape[0])
        expected = _sequential(a, shape, residual, smoother, reverse)
        cache = LineSmootherCache(a, shape)
        actual = classical_smoothing_correction(a, shape, residual, smoother, cache=cache, reverse=reverse)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13 * np.abs(expected).max())


def test_batching_falls_back_when_same_colour_lines_couple():
    # Distance-2 couplings between x-lines break the zebra independence.
    n = 6
    a = sp.diags([4.] * (n * n)).tolil()
    for i in range(n):
        for j in range(n):
            k = i * n + j
            if j + 2 < n:
                a[k, k + 2] = a[k + 2, k] = -1.
    a = a.tocsr()
    cache = LineSmootherCache(a, (n, n))
    assert cache.zebra_batches('x') is None
    assert cache.zebra_batches('y') is not None
