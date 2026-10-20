"""Student-conditioned EM targets and deployable whole-cycle line gradients."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.data import make_examples
from adaptive_mg.provenance import operator_digest
from adaptive_mg.v67.banks import Stats, hybrid_cycle
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.em_line_models import make_em_line_components
from adaptive_mg.v67.em_line_training import em_imitation_objective, line_vcycle_objective
from adaptive_mg.v67.joint_transfer_training import _levels
from adaptive_mg.v67.research_runtime import offline_teacher_generation
from adaptive_mg.v67.solver import PreparedAdaptiveMG
from adaptive_mg.v67.spatial import SpatialState
from adaptive_mg.v67.unroll import make_graph, cycle


def inputs(n=15):
    torch.set_num_threads(1)
    # A heterogeneous case retains difficult modes after several line cycles.
    example = make_examples([n], per_size=4, seed=32)[-1]
    cfg = AdaptiveConfig(mg=MGConfig(strategy_name='line_alt_bilinear_full',
        pre_steps=2, post_steps=2, stencil_backend='csr', max_cycles=50),
        mode='research', branch='H_SP', replace_pre=2, replace_post=2,
        smoother_levels='all', transfer_levels='all', spatial=False,
        gate_mode='open', use_learned_controller=False)
    return example, make_em_line_components(hidden=4, seed=18), cfg


def perturb(model):
    with torch.no_grad():
        model.transfer.head.weight.normal_(0, .02)
        model.transfer.head.bias.normal_(0, .02)
        model.smoother.head.weight.normal_(0, .02)
        model.smoother.head.bias.normal_(0, .02)


def test_em_targets_use_actual_student_coarse_operators_and_no_smoother(monkeypatch):
    import adaptive_mg.v67.em_line_training as training
    example, model, cfg = inputs()
    perturb(model)
    seen = []
    actual_energy = training.energy_weights
    def capture(a, pattern, **kwargs):
        seen.append((a.copy(), kwargs))
        return actual_energy(a, pattern, **kwargs)
    monkeypatch.setattr(training, 'energy_weights', capture)
    def forbidden(*args, **kwargs):
        raise AssertionError('P pretraining generated a line smoother')
    monkeypatch.setattr(model.smoother, 'forward', forbidden)
    loss, report = em_imitation_objective(example, model, cfg)
    pcfg = replace(cfg, branch='H_P', use_smoother=False, replace_pre=0, replace_post=0)
    with offline_teacher_generation():
        student = list(_levels(make_graph(example.a, (15, 15), model, pcfg)))
        parent = list(_levels(make_graph(example.a, (15, 15), model, pcfg, learned=False)))
    assert len(seen) == len(student) == 2
    assert not np.allclose(seen[1][0].toarray(), parent[1].raw_scipy.toarray())
    for (a, kwargs), level, row in zip(seen, student, report['levels']):
        np.testing.assert_allclose(a.toarray(), level.raw_scipy.toarray(), atol=1e-12)
        assert row['operator_digest'] == operator_digest(a)
        assert kwargs == dict(maxiter=5, rtol=1e-8, fixed_iterations=True)
        assert row['teacher_target_detached']
    assert not report['fallback_used'] and not report['online_em']
    loss.backward()
    assert model.transfer.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.smoother.parameters())


@pytest.mark.parametrize('perturbed', [False, True])
def test_vector_and_batched_unroll_match_raw_runtime_cycles(perturbed):
    example, model, cfg = inputs()
    if perturbed:
        perturb(model)
    rhs = np.random.default_rng(40).normal(size=(example.a.shape[0], 2))
    with offline_teacher_generation():
        graph = make_graph(example.a, (15, 15), model, cfg)
    prepared = PreparedAdaptiveMG(example.a, example.n, model, cfg)
    stats = Stats()
    runtime = prepared.ensure_branch('H_SP', stats)
    bt = torch.tensor(rhs, dtype=torch.float64)
    batch, vectors = torch.zeros_like(bt), [torch.zeros_like(bt[:, j]) for j in range(2)]
    numerical = [np.zeros(rhs.shape[0]) for _ in range(2)]
    states = [SpatialState(model, cfg) for _ in range(2)]
    for k in range(3):
        batch = cycle(graph, batch, bt, model, cfg, k)
        for j in range(2):
            vectors[j] = cycle(graph, vectors[j], bt[:, j], model, cfg, k)
            numerical[j] = hybrid_cycle(runtime, numerical[j], rhs[:, j], cfg,
                                         stats, states[j], k, True)
            torch.testing.assert_close(batch[:, j], vectors[j], rtol=2e-11, atol=2e-12)
            np.testing.assert_allclose(batch[:, j].detach().numpy(), numerical[j],
                                       rtol=3e-7, atol=2e-9)


def test_joint_gradients_reach_coarse_a_and_cached_factors_without_regeneration():
    example, model, cfg = inputs()
    perturb(model)
    counts = {'p': 0, 's': 0}
    def hook(key):
        def count(*args):
            counts[key] += 1
        return count
    hooks = [model.transfer.register_forward_hook(hook('p')),
             model.smoother.register_forward_hook(hook('s'))]
    with offline_teacher_generation():
        root = make_graph(example.a, (15, 15), model, cfg)
        levels = list(_levels(root))
        for level in levels:
            level.interpolation_weights.retain_grad()
            level.line_factors['x'][0].retain_grad()
            level.line_factors['y'][1].retain_grad()
        levels[1].a.values.retain_grad()
        rhs = torch.randn(example.a.shape[0], 3, dtype=torch.float64)
        x = torch.zeros_like(rhs)
        for k in range(3):
            x = cycle(root, x, rhs, model, cfg, k)
        loss = (rhs - root.a.apply(x)).square().mean()
        loss.backward()
    for h in hooks:
        h.remove()
    assert counts == {'p': 2, 's': 2}
    for level in levels:
        for value in (level.interpolation_weights,
                      level.line_factors['x'][0], level.line_factors['y'][1]):
            assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert levels[1].a.values.grad.abs().sum() > 0
    for module in (model.transfer, model.smoother):
        assert torch.isfinite(module.head.weight.grad).all()
        assert module.head.weight.grad.abs().sum() > 0


def test_transfer_derivative_includes_recursive_line_and_coarse_solve():
    example, model, cfg = inputs()
    perturb(model)
    rhs = torch.randn(example.a.shape[0], 2, dtype=torch.float64)
    def value():
        root = make_graph(example.a, (15, 15), model, cfg)
        x = cycle(root, torch.zeros_like(rhs), rhs, model, cfg)
        return (rhs-root.a.apply(x)).square().mean()
    with offline_teacher_generation():
        loss = value()
        loss.backward()
        parameter = model.transfer.head.bias
        index = int(parameter.grad.abs().argmax())
        analytic = float(parameter.grad[index])
        original = float(parameter.detach()[index])
        results = []
        for delta in (-1e-3, 1e-3):
            with torch.no_grad():
                parameter[index] = original + delta
            results.append(float(value().detach()))
        with torch.no_grad():
            parameter[index] = original
    assert abs(analytic) > 1e-8
    assert analytic == pytest.approx((results[1]-results[0])/.002, rel=.025, abs=2e-6)


def test_frozen_p_stage_updates_only_s_and_joint_unfreezes_both():
    example, model, cfg = inputs()
    perturb(model)
    before = deepcopy(model.transfer.state_dict())
    for p in model.transfer.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.smoother.parameters(), lr=1e-3)
    loss, report = line_vcycle_objective(example, model, cfg, cycles=3,
                                        hard_smoothing=1, hard_cycles=1)
    assert report['probe_labels'] == ['random', 'classical_smoothed', 'learned_hard']
    assert report['line_factor_generations'] == 2
    assert not report['fallback_used'] and report['pre_post_coefficients_shared']
    loss.backward()
    assert all(p.grad is None for p in model.transfer.parameters())
    assert model.smoother.head.weight.grad.abs().sum() > 0
    optimizer.step()
    for key, value in before.items():
        torch.testing.assert_close(value, model.transfer.state_dict()[key], rtol=0, atol=0)
    for p in model.transfer.parameters():
        p.requires_grad_(True)
    model.smoother.zero_grad()
    loss, _ = line_vcycle_objective(example, model, cfg, cycles=3,
                                    hard_smoothing=1, hard_cycles=1)
    loss.backward()
    assert model.transfer.head.weight.grad.abs().sum() > 0
    assert model.smoother.head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('objective', [em_imitation_objective, line_vcycle_objective])
def test_training_rejects_held_out_operators(objective):
    example, model, cfg = inputs(7)
    example.research_split = 'validation'
    with pytest.raises(ValueError, match='TRAIN'):
        objective(example, model, cfg)


def test_line_objective_reports_raw_cycle_norms_with_fixed_supplied_probes():
    example, model, cfg = inputs(7)
    rhs = torch.tensor(example.b[:, None], dtype=torch.float64)
    _, report = line_vcycle_objective(example, model, cfg, cycles=3, rhs=rhs)
    with offline_teacher_generation():
        root = make_graph(example.a, (7, 7), model, cfg)
        rhs = rhs/torch.linalg.vector_norm(rhs, dim=0)
        x = torch.zeros_like(rhs)
        for k in range(3):
            x = cycle(root, x, rhs, model, cfg, k)
            residual = torch.linalg.vector_norm(rhs-root.a.apply(x), dim=0)
            np.testing.assert_allclose(report['history'][k], residual.detach().numpy())
    assert report['cycles'] == 3 and report['probe_labels'] == ['supplied']
