"""Cached learned line factors: numerical parity, gradients and reuse."""
from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg import DiffusionCase, MGConfig, assemble_stiffness
from adaptive_mg.smoothers import classical_smoothing_correction
from adaptive_mg.v67.autograd_sparse import SparseTensor
from adaptive_mg.v67.banks import Stats, hybrid_cycle
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.em_line_models import make_em_line_components, AffineTransferCNN
from adaptive_mg.v67.line_smoother import (LineSmootherCNN, LineSmootherBank,
    factors_from_raw, apply_line_smoother_torch, line_groups)
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.solver import PreparedAdaptiveMG
from adaptive_mg.v67.spatial import SpatialState
from adaptive_mg.v67.unroll import make_graph, cycle


def operator(shape, wider=False):
    nx, ny = shape
    def t(n):
        return sp.diags((-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)), (-1, 0, 1), format='csr')
    a = (sp.kron(t(nx), sp.eye(ny)) + sp.kron(sp.eye(nx), t(ny))).tocsr()
    if wider:
        # SPD contribution between points two columns apart. It creates both
        # wider y-line blocks and couplings between same-colour x lines.
        i = np.array([x*ny+y for x in range(nx) for y in range(ny-2)])
        j = i+2
        edge = sp.coo_matrix((np.tile([.15, .15, -.15, -.15], len(i)),
            (np.column_stack((i, j, i, j)).ravel(),
             np.column_stack((i, j, j, i)).ravel())), shape=a.shape).tocsr()
        a = a+edge
    return a


def numpy_bank(a, shape, factors, gain=1.):
    return LineSmootherBank(a, shape,
        {k: tuple(x.detach().numpy() for x in v) for k, v in factors.items()}, gain=gain)


@pytest.mark.parametrize('shape', [(5, 7), (7, 5), (1, 5), (5, 1)])
@pytest.mark.parametrize('reverse', [False, True])
def test_zero_head_matches_tridiagonal_classical_line_sweeps(shape, reverse):
    a = operator(shape)
    factors = factors_from_raw(a, shape, torch.zeros(1, 4, *shape))
    rhs = np.random.default_rng(41).normal(size=a.shape[0])
    actual = numpy_bank(a, shape, factors).apply(rhs, Stats(), reverse)
    expected = classical_smoothing_correction(a, shape, rhs, 'line_alt', reverse=reverse)
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)


@pytest.mark.parametrize('wider', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_learned_line_runtime_torch_parity_and_gradients(wider, reverse):
    torch.set_num_threads(1)
    shape = (5, 7); a = operator(shape, wider)
    graph = SparseTensor.from_scipy(a)
    graph.values.requires_grad_(True)
    raw = torch.tensor(np.random.default_rng(8).normal(scale=.15, size=(1, 4, *shape)),
                       dtype=torch.float64, requires_grad=True)
    factors = factors_from_raw(graph, shape, raw)
    rhs = torch.tensor(np.random.default_rng(9).normal(size=(a.shape[0], 2)))
    actual = apply_line_smoother_torch(graph, shape, factors, rhs, reverse)
    bank = numpy_bank(a, shape, factors)
    expected = np.column_stack([bank.apply(r.numpy(), Stats(), reverse) for r in rhs.T])
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=2e-14, atol=2e-15)
    actual.square().mean().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    assert torch.isfinite(graph.values.grad).all() and graph.values.grad.abs().sum() > 0
    if wider:
        assert len(line_groups(a, shape, 'x')) == shape[1]
        # This is intentionally a band approximation, not the exact wider solve.
        zero = numpy_bank(a, shape, factors_from_raw(a, shape, torch.zeros_like(raw)))
        classical = classical_smoothing_correction(a, shape, rhs[:, 0].numpy(), 'line_alt')
        assert np.linalg.norm(zero.apply(rhs[:, 0].numpy(), Stats())-classical) > 1e-6


def inputs(n=15):
    torch.set_num_threads(1)
    cfg = AdaptiveConfig(mg=MGConfig(strategy_name='line_alt_bilinear_full',
        stencil_backend='csr', coarsest_n=3, pre_steps=2, post_steps=2), mode='research', branch='H_SP',
        smoother_levels='all', transfer_levels='all', replace_pre=2, replace_post=2,
        spatial=False, gate_mode='open', use_learned_controller=False)
    model = make_em_line_components(hidden=4, seed=25)
    with torch.no_grad():
        model.smoother.head.weight.normal_(0, .002)
        model.transfer.head.weight.normal_(0, .002)
    a = assemble_stiffness(DiffusionCase(n=n, epsilon=.3, angle_deg=25.))
    return a, model, cfg


def test_actual_hierarchy_cycle_parity_and_no_repeated_generation(monkeypatch):
    a, model, cfg = inputs()
    prepared = PreparedAdaptiveMG(a, 15, model, cfg)
    stats = Stats(); root = prepared.ensure_branch('H_SP', stats)
    level = root; levels = 0
    while level.coarse is not None:
        assert isinstance(level.neural_stencil, LineSmootherBank)
        if level.index > 0:
            assert level.cache.factorization_count == 0
        level = level.coarse; levels += 1
    assert levels == 2 and stats.smoother_nn_calls == 2
    assert stats.operator_generation_seconds > 0
    later = Stats()
    assert prepared.ensure_branch('H_SP', later) is root
    assert later.smoother_nn_calls == 0 and later.smoother_bank_cache_hits == 1
    import adaptive_mg.v67.banks as banks
    def forbidden(*args, **kwargs):
        raise AssertionError('No exact classical line solve in raw learned cycle')
    monkeypatch.setattr(banks, 'classical_step', forbidden)
    rhs = np.random.default_rng(42).normal(size=a.shape[0])
    runtime = hybrid_cycle(root, np.zeros_like(rhs), rhs, cfg, Stats(), SpatialState(model, cfg), 0, True)
    graph = make_graph(a, (15, 15), model, cfg)
    expected = cycle(graph, torch.zeros_like(torch.tensor(rhs)), torch.tensor(rhs), model, cfg, 0)
    np.testing.assert_allclose(runtime, expected.detach().numpy(), rtol=3e-8, atol=3e-9)
    expected.square().sum().backward()
    assert model.smoother.head.weight.grad.abs().sum() > 0
    assert model.transfer.head.weight.grad.abs().sum() > 0
    assert graph.coarse.a.values.requires_grad


def test_line_checkpoint_and_frozen_inference_copy(tmp_path):
    a, model, cfg = inputs(7)
    path = tmp_path / 'line.pt'; model.save(path)
    loaded = Components.load(path)
    assert isinstance(loaded.smoother, LineSmootherCNN)
    assert isinstance(loaded.transfer, AffineTransferCNN)
    assert loaded.generation_signature() == model.generation_signature()
    frozen = loaded.frozen_inference_copy()
    assert all(not p.requires_grad for p in frozen.smoother.parameters())
    # The immutable production copy and ordinary checkpoint build equal factors.
    ordinary = PreparedAdaptiveMG(a, 7, loaded, cfg).ensure_branch('H_SP', Stats())
    immutable = PreparedAdaptiveMG(a, 7, frozen, cfg).ensure_branch('H_SP', Stats())
    for key in ('x', 'y'):
        for first, second in zip(ordinary.neural_stencil.factors[key], immutable.neural_stencil.factors[key]):
            np.testing.assert_array_equal(first, second)


def test_three_nonterminal_levels_share_generated_factors_across_rhs():
    a, model, cfg = inputs(31)
    prepared = PreparedAdaptiveMG(a, 31, model, cfg)
    stats = Stats(); root = prepared.ensure_branch('H_SP', stats)
    assert stats.smoother_nn_calls == 3
    banks = []
    item = root
    while item.coarse is not None:
        banks.append(item.neural_stencil)
        item = item.coarse
    results = prepared.solve_many(np.random.default_rng(17).normal(size=(3, a.shape[0])))
    assert all(result.converged for result in results)
    assert all(result.stats['smoother_nn_calls'] == 0 for result in results)
    item = root
    for bank in banks:
        # There is one reusable bank per level, used for both sweep orders.
        assert item.neural_stencil is bank
        item = item.coarse


def test_level_conditioning_and_invalid_generation_guards():
    model = LineSmootherCNN(hidden=4)
    with torch.no_grad(): model.head.weight.normal_(0, .1)
    features = torch.randn(1, 10, 5, 7)
    assert not torch.equal(model(features, 0, 3), model(features, 1, 3))
    with pytest.raises(ValueError, match='nonterminal'):
        model(features, 2, 3)
    with pytest.raises(ValueError, match='positive-diagonal'):
        model.coefficients(-operator((5, 7)), (5, 7), features)
    a, components, cfg = inputs(7)
    prepared = PreparedAdaptiveMG(a, 7, components, replace(cfg, spatial=True))
    with pytest.raises(ValueError, match='whole-level'):
        prepared.ensure_branch('H_SP', Stats())
