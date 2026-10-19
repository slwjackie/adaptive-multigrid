"""Sparse approximate-inverse structure, differentiation and actual cycle parity."""
from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg import DiffusionCase, MGConfig, assemble_stiffness
from adaptive_mg.v67.autograd_sparse import SparseTensor
from adaptive_mg.v67.banks import Stats, hybrid_cycle
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.fixed_classical import FIXED_STRATEGY_NAME
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.solver import PreparedAdaptiveMG
from adaptive_mg.v67.sparse_smoother import SparsePolynomialSmoother
from adaptive_mg.v67.spatial import SpatialState
from adaptive_mg.v67.unet_transfer import CompactTransferStudent
from adaptive_mg.v67.unroll import make_graph, cycle


def inputs(n=15):
    torch.set_num_threads(1)
    cfg = AdaptiveConfig(mg=MGConfig(strategy_name=FIXED_STRATEGY_NAME, em_fixed_iterations=True,
        stencil_backend='csr', coarsest_n=3), mode='research', branch='H_SP', use_smoother=True,
        use_transfer=True, smoother_levels='all', transfer_levels='all', replace_pre=2, replace_post=2,
        spatial=False, gate_mode='open', use_learned_controller=False)
    model = Components.create(hidden=4, seed=25)
    model.transfer = CompactTransferStudent(hidden=4)
    model.smoother = SparsePolynomialSmoother()
    with torch.no_grad():
        model.smoother.coefficients.copy_(torch.tensor([.6, .01, .002, -.03, .02]))
        model.transfer.head.weight.normal_(0, .002)
    a = assemble_stiffness(DiffusionCase(n=n, epsilon=.3, angle_deg=25.))
    return a, model, cfg


def test_sparse_support_symmetry_scaling_and_exact_jacobi_initialization():
    a = sp.csr_matrix([[4., -1., 0., .2], [-1., 3., -.5, 0.],
                      [0., -.5, 2., -.3], [.2, 0., -.3, 3.]])
    model = SparsePolynomialSmoother().double()
    np.testing.assert_allclose(model.sparse_matrix(a).toarray(),
        np.diag(float(model.coefficients[0].detach()) / a.diagonal()))
    with torch.no_grad(): model.coefficients[3:] = torch.tensor([-.1, .02])
    b = model.sparse_matrix(a)
    np.testing.assert_allclose(b.toarray(), b.T.toarray(), atol=1e-15)
    assert not np.any((b.toarray() != 0) & (a.toarray() == 0))
    assert b[0, 3] != 0  # actual support, not a clipped geometric 3x3 neighborhood
    np.testing.assert_allclose(model.sparse_matrix(1e-12*a).toarray()*1e-12, b.toarray())
    with pytest.raises(ValueError, match='positive diagonal'):
        model.sparse_matrix(-a)


def test_sparse_values_gradient_and_runtime_matrix_agree():
    a, components, _ = inputs(7)
    model = components.smoother.double()
    graph = SparseTensor.from_scipy(a)
    values = model.sparse_values(graph)
    generated = SparseTensor.from_coo(graph.rows, graph.cols, values, graph.shape)
    np.testing.assert_allclose(generated.numpy().toarray(), model.sparse_matrix(a).toarray(), atol=1e-15)
    rhs = torch.linspace(-1., 1., a.shape[0], dtype=torch.float64)
    loss = generated.apply(rhs).square().sum(); loss.backward()
    analytic = model.coefficients.grad.clone()
    numerical = []
    for i in range(5):
        original = float(model.coefficients[i].detach()); samples = []
        for delta in (-1e-6, 1e-6):
            with torch.no_grad(): model.coefficients[i] = original + delta
            samples.append(np.linalg.norm(model.sparse_matrix(a) @ rhs.numpy())**2)
        with torch.no_grad(): model.coefficients[i] = original
        numerical.append((samples[1]-samples[0])/2e-6)
    np.testing.assert_allclose(analytic.numpy(), numerical, rtol=2e-7, atol=1e-8)


def test_all_level_sparse_cycle_matches_unroll_without_any_line_smoothing(monkeypatch):
    a, model, cfg = inputs()
    prepared = PreparedAdaptiveMG(a, 15, model, cfg)
    stats = Stats(); root = prepared.ensure_branch('H_SP', stats)
    level = root; levels = 0
    while level.coarse is not None:
        assert level.neural_stencil is not None
        assert level.neural_stencil.csr.nnz <= level.a.nnz
        if level.index > 0:
            assert level.cache.factorization_count == 0
        level = level.coarse; levels += 1
    assert levels == 2 and stats.smoother_nn_calls == levels
    import adaptive_mg.v67.banks as banks
    def forbidden(*args, **kwargs): raise AssertionError('line smoothing executed in raw sparse cycle')
    monkeypatch.setattr(banks, 'classical_step', forbidden)
    b = np.random.default_rng(4).normal(size=a.shape[0]); x = np.zeros_like(b)
    result = hybrid_cycle(root, x, b, cfg, Stats(), SpatialState(model, cfg), 0, True)
    graph = make_graph(a, (15, 15), model, cfg)
    expected = cycle(graph, torch.zeros_like(torch.tensor(b)), torch.tensor(b), model, cfg, 0)
    np.testing.assert_allclose(result, expected.detach().numpy(), rtol=2e-8, atol=2e-9)
    expected.square().sum().backward()
    assert model.smoother.coefficients.grad.abs().sum() > 0
    assert model.transfer.head.weight.grad.abs().sum() > 0
    many = torch.stack((torch.tensor(b), torch.tensor(b)*2), dim=1)
    batched = cycle(graph, torch.zeros_like(many), many, model, cfg, 0)
    np.testing.assert_allclose(batched.detach().numpy(), np.column_stack((result, result*2)), rtol=2e-8, atol=2e-9)


def test_serialization_and_generated_bank_cache(tmp_path):
    a, model, cfg = inputs(7)
    checkpoint = tmp_path / 'model.pt'; model.save(checkpoint)
    restored = Components.load(checkpoint)
    assert model.generation_signature() == restored.generation_signature()
    assert isinstance(restored.smoother, SparsePolynomialSmoother)
    prepared = PreparedAdaptiveMG(a, 7, restored, cfg)
    first = Stats(); root = prepared.ensure_branch('H_SP', first)
    second = Stats(); assert prepared.ensure_branch('H_SP', second) is root
    assert first.operator_generation_seconds > 0 and first.smoother_nn_calls == 1
    assert second.smoother_nn_calls == 0 and second.smoother_bank_cache_hits == 1
    assert AdaptiveConfig.from_dict(cfg.to_dict()) == cfg


def test_bad_sparse_setup_falls_back_to_unchanged_fixed_classical(monkeypatch):
    a, model, cfg = inputs(7)
    b = np.random.default_rng(22).normal(size=a.shape[0])
    from adaptive_mg.v67.limited import forced_config
    classical = PreparedAdaptiveMG(a, 7, None, forced_config(cfg, 'C')).solve(b)
    def reject(*args, **kwargs): raise ValueError('bad sparse proposal')
    monkeypatch.setattr(SparsePolynomialSmoother, 'sparse_matrix', reject)
    result = PreparedAdaptiveMG(a, 7, model, cfg).solve(b)
    assert result.stats['setup_failures'] > 0
    np.testing.assert_allclose(result.x, classical.x, rtol=1e-13, atol=1e-13)
