"""Phase-correct cached CNN stencils and the differentiable full V-cycle."""
import math

import numpy as np
import pytest
import torch

from adaptive_mg import DiffusionCase, MGConfig, assemble_stiffness
from adaptive_mg.v67.banks import Stats, hybrid_cycle, replacement_step
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.paired_smoother import PairedStencilSmoother, PairedStencilBank
from adaptive_mg.v67.solver import PreparedAdaptiveMG
from adaptive_mg.v67.spatial import SpatialState
from adaptive_mg.v67.unet_transfer import CompactTransferStudent
from adaptive_mg.v67.unroll import make_graph, cycle, neural_delta


def inputs(n=15):
    torch.set_num_threads(1)
    cfg = AdaptiveConfig(mg=MGConfig(strategy_name='line_alt_bilinear_full',
        stencil_backend='csr', coarsest_n=3, pre_steps=2, post_steps=2), mode='research', branch='H_SP',
        use_smoother=True, use_transfer=True, smoother_levels='all', transfer_levels='all',
        replace_pre=2, replace_post=2, spatial=False, gate_mode='open',
        use_learned_controller=False)
    model = Components.create(hidden=4, seed=21)
    model.transfer = CompactTransferStudent(hidden=4)
    model.smoother = PairedStencilSmoother(hidden=4)
    a = assemble_stiffness(DiffusionCase(n=n, epsilon=.4, angle_deg=25.))
    return a, model, cfg


def test_paired_initialization_and_independent_phases():
    a, model, cfg = inputs(7)
    smoother = model.smoother
    directions, gains = smoother.paired_directions_and_gains(torch.randn(2, 10, 7, 7))
    assert directions.shape == (2, 2, 9, 7, 7) and gains.shape == (2, 2)
    assert torch.all(directions[:, :, 0] == 1) and not torch.count_nonzero(directions[:, :, 1:])
    torch.testing.assert_close(gains, torch.full_like(gains, 2 / 3))
    with torch.no_grad():
        smoother.gain_heads[0].bias.fill_(math.log(math.expm1(.3)))
        smoother.gain_heads[1].bias.fill_(math.log(math.expm1(.7)))
    prepared = PreparedAdaptiveMG(a, 7, model, cfg)
    root = prepared.ensure_branch('H_SP', Stats())
    assert isinstance(root.neural_stencil, PairedStencilBank)
    residual = np.random.default_rng(6).normal(size=a.shape[0])
    graph = make_graph(a, (7, 7), model, cfg)
    for reverse, gain in ((False, .3), (True, .7)):
        stats = Stats()
        got = replacement_step(root, residual, np.ones_like(residual, bool), cfg, stats, reverse)
        expected = gain * residual / a.diagonal()
        np.testing.assert_allclose(got, expected, rtol=2e-7)
        np.testing.assert_allclose(neural_delta(graph, torch.tensor(residual), reverse).detach(),
                                   expected, rtol=2e-7)
        assert stats.neural_apply_calls == 1  # no application of the other head
        assert stats.multistage_applications == 0


def test_multilevel_paired_cycle_matches_unroll_and_both_heads_receive_gradients(monkeypatch):
    a, model, cfg = inputs()
    with torch.no_grad():
        model.transfer.head.weight.normal_(0, .002)
        for index, head in enumerate(model.smoother.stencil_heads):
            head.weight.normal_(0, .003 * (index + 1))
        model.smoother.gain_heads[1].bias.add_(.2)
    prepared = PreparedAdaptiveMG(a, 15, model, cfg)
    stats = Stats(); root = prepared.ensure_branch('H_SP', stats)
    level = root; levels = 0
    while level.coarse is not None:
        assert isinstance(level.neural_stencil, PairedStencilBank)
        assert level.neural_stencil.pre.csr.nnz <= 9 * level.a.shape[0]
        assert level.neural_stencil.post.csr.nnz <= 9 * level.a.shape[0]
        if level.index:
            assert level.cache.factorization_count == 0
        levels += 1; level = level.coarse
    assert levels == 2 and stats.smoother_nn_calls == levels
    assert stats.smoother_stage_generations == 2 * levels

    def forbidden(*args, **kwargs):
        raise AssertionError('a raw cached paired cycle ran NN generation or line smoothing')
    import adaptive_mg.v67.banks as banks
    monkeypatch.setattr(banks, 'classical_step', forbidden)
    b = np.random.default_rng(3).normal(size=a.shape[0])
    result = hybrid_cycle(root, np.zeros_like(b), b, cfg, Stats(), SpatialState(model, cfg), 0, True)
    graph = make_graph(a, (15, 15), model, cfg)
    rhs = torch.tensor(b)
    expected = cycle(graph, torch.zeros_like(rhs), rhs, model, cfg)
    np.testing.assert_allclose(result, expected.detach(), rtol=3e-6, atol=3e-8)
    expected.square().sum().backward()
    for head in (*model.smoother.stencil_heads, *model.smoother.gain_heads):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
    assert model.transfer.head.weight.grad.abs().sum() > 0
    many = torch.stack((rhs, rhs * 2), dim=1)
    batched = cycle(graph, torch.zeros_like(many), many, model, cfg)
    np.testing.assert_allclose(batched.detach(), np.column_stack((result, result * 2)), rtol=3e-6, atol=3e-8)
    # After generating the bank, repeated cycles cannot call the CNN.
    monkeypatch.setattr(model.smoother, 'paired_directions_and_gains', forbidden)
    for _ in range(2):
        hybrid_cycle(root, np.zeros_like(b), b, cfg, Stats(), SpatialState(model, cfg), 0, True)


def test_paired_checkpoint_and_bank_reuse(tmp_path):
    a, model, cfg = inputs(7)
    path = tmp_path / 'paired.pt'; model.save(path)
    restored = Components.load(path)
    assert isinstance(restored.smoother, PairedStencilSmoother)
    assert restored.generation_signature() == model.generation_signature()
    prepared = PreparedAdaptiveMG(a, 7, restored, cfg)
    first = Stats(); root = prepared.ensure_branch('H_SP', first)
    second = Stats(); assert prepared.ensure_branch('H_SP', second) is root
    assert first.smoother_nn_calls == 1 and first.smoother_stage_generations == 2
    assert second.smoother_nn_calls == 0 and second.smoother_bank_cache_hits == 1


@pytest.mark.parametrize('kwargs', [dict(hidden=0), dict(initial_gain=0),
                                  dict(initial_gain=float('nan')), dict(version=2)])
def test_invalid_paired_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        PairedStencilSmoother(**kwargs)


def test_zero_direction_heads_have_finite_gradients():
    model = PairedStencilSmoother(hidden=2)
    with torch.no_grad():
        for head in model.stencil_heads:
            head.weight.zero_(); head.bias.zero_()
    directions, gains = model.paired_directions_and_gains(torch.randn(1, 10, 3, 3))
    (directions.sum() + gains.sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
