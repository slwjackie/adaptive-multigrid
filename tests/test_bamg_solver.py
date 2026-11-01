"""Frozen-support numerical and autograd contracts for direct BAMG P."""
from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg.config import MGConfig
from adaptive_mg.data import make_examples
from adaptive_mg.hierarchy import WorkStats, build_fixed_hierarchy, classical_cycle
from adaptive_mg.strategy import get_strategy
from adaptive_mg.v67.autograd_sparse import SparseTensor
from adaptive_mg.v67.bamg_transfer import MaskCNN, WeightCNN
from adaptive_mg.v67.bamg_solver import (
    HierarchyBudgetError, ReferenceModels, StructuralCaps, build_graph, build_runtime,
    graph_cycle, graph_levels, mask_digest, progress_rates, residual_loss,
    runtime_from_prolongations,
)


def fixture(n=7):
    torch.set_num_threads(1); torch.manual_seed(73)
    e = make_examples([n], per_size=4, seed=32)[-1]
    mask = MaskCNN(hidden=3, depth=1, activation='silu').double()
    weight = WeightCNN(hidden=3, depth=1, activation='silu').double()
    cfg = MGConfig(mode='classical', strategy_name='line_alt_bilinear_full',
                   pre_steps=2, post_steps=2, max_cycles=30, stencil_backend='csr')
    reference = ReferenceModels(mask, weight, mg_config=cfg)
    return e, mask, weight, cfg, reference


def test_reference_owns_storage_is_immutable_and_restores_with_weights_only():
    e, phi, theta, cfg, ref = fixture(15)
    fixed = ref.masks_for(e.a, (15, 15)); original = mask_digest(fixed)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(phi.parameters(), ref.mask_model.parameters()))
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(theta.parameters(), ref.weight_model.parameters()))
    with torch.no_grad():
        for p in phi.parameters():
            p.add_(.5)
        for p in theta.parameters():
            p.mul_(1.3)
    assert mask_digest(ref.masks_for(e.a, (15, 15))) == original
    fixed[0].fill_(False)  # defensive return clone
    assert mask_digest(ref.masks_for(e.a, (15, 15))) == original
    buffer = BytesIO(); torch.save(ref.state_dict(include_cache=True), buffer); buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored = ReferenceModels.from_state_dict(state)
    assert mask_digest(restored.masks_for(e.a, (15, 15))) == original
    assert restored.last_report['cache_hit']
    assert all(not p.requires_grad and p.grad is None for p in restored.mask_model.parameters())
    with torch.no_grad():
        next(restored.weight_model.parameters()).add_(1.)
    with pytest.raises(RuntimeError, match='changed'):
        restored.masks_for(e.a, (15, 15))


def test_reference_cache_rejects_mapping_corruption_and_keys_operator_dtype():
    e, phi, theta, cfg, ref = fixture()
    first = ref.masks_for(e.a, (7, 7)); firstkey = ref.last_report['cache_key']
    ref.masks_for(e.a.astype(np.float32), (7, 7))
    assert ref.last_report['cache_key'] != firstkey
    perturbed = e.a.copy(); perturbed.data[0] += .01
    ref.masks_for(perturbed, (7, 7))
    assert ref.last_report['cache_key'] != firstkey
    state = ref.state_dict(include_cache=True)
    state['cache'][firstkey]['columns'][0][0, 0] += 1
    with pytest.raises(RuntimeError, match='candidate mapping'):
        ReferenceModels.from_state_dict(state)
    ref.clear_cache(); ref.masks_for(e.a, (7, 7))
    assert not ref.last_report['cache_hit']
    assert mask_digest(first) == mask_digest(ref.masks_for(e.a, (7, 7)))


@pytest.mark.parametrize('batch', [False, True])
def test_differentiable_and_warm_runtime_parity_and_no_setup_in_cycles(monkeypatch, batch):
    e, phi, theta, cfg, ref = fixture(15)
    masks = ref.masks_for(e.a, (15, 15))
    graph = build_graph(e.a, (15, 15), theta, masks, cfg)
    runtime = build_runtime(e.a, (15, 15), theta, ref, cfg)
    b = np.random.default_rng(4).normal(size=(225, 2) if batch else (225,))
    x = torch.zeros_like(torch.tensor(b)); xr = np.zeros_like(b)
    for forbidden_model in (phi, theta, ref.mask_model, ref.weight_model):
        monkeypatch.setattr(forbidden_model, 'forward', lambda *a, **k: pytest.fail('CNN called in warm cycle'))
    import scipy.sparse.linalg as sla
    original_splu = sla.splu
    # Differentiable factors are created once on first rollout, so warm those first.
    x = graph_cycle(graph, x, torch.tensor(b)); xr = runtime.cycle(xr, b)
    np.testing.assert_allclose(x.detach().numpy(), xr, rtol=1e-11, atol=3e-12)
    monkeypatch.setattr(sla, 'splu', lambda *a, **k: pytest.fail('Factorization in warm cycle'))
    x = graph_cycle(graph, x, torch.tensor(b)); xr = runtime.cycle(xr, b)
    np.testing.assert_allclose(x.detach().numpy(), xr, rtol=1e-11, atol=3e-12)
    assert runtime.setup_report['factorizations']['coarse_factorizations'] == 1
    assert runtime.setup_report['factorizations']['line_factorizations'] > 0


def test_multilevel_gradient_matches_finite_difference_and_rebuilds_actual_A():
    e, phi, theta, cfg, ref = fixture(15)
    masks = ref.masks_for(e.a, (15, 15)); frozen_digest = mask_digest(masks)
    captured = []
    def hook(module, args):
        if args[0].requires_grad:
            args[0].retain_grad(); captured.append(args[0])
    handle = theta.register_forward_pre_hook(hook)
    graph = build_graph(e.a, (15, 15), theta, masks, cfg)
    old_a = graph.coarse.a.values.detach().clone()
    rhs = np.random.default_rng(5).normal(size=(225, 2))
    with torch.no_grad():
        graph_cycle(graph, torch.tensor(rhs), torch.zeros_like(torch.tensor(rhs)))
    loss, report, _ = residual_loss(graph, rhs, cycles=2)
    loss.backward(); handle.remove()
    assert len(captured) == 1 and captured[0].grad.abs().sum() > 0  # actual A1 features
    assert report['weights'] == [.5, .5]
    assert all(p.grad is None for p in phi.parameters())
    assert all(p.grad is None for p in ref.mask_model.parameters())
    bias = theta.head.bias
    idx = int(bias.grad.abs().argmax()); actual = float(bias.grad[idx])
    assert abs(actual) > 1e-8
    old = float(bias.detach()[idx]); h = 2e-5; values = []
    for direction in (1., -1.):
        with torch.no_grad():
            bias[idx] = old + direction * h
        candidate = build_graph(e.a, (15, 15), theta, masks, cfg)
        value, _, _ = residual_loss(candidate, rhs, cycles=2)
        values.append(float(value.detach()))
    with torch.no_grad():
        bias[idx] = old
    numerical = (values[0] - values[1]) / (2 * h)
    assert actual == pytest.approx(numerical, rel=2e-4, abs=2e-7)
    opt = torch.optim.SGD(theta.parameters(), lr=.2); opt.step()
    updated = build_graph(e.a, (15, 15), theta, masks, cfg)
    assert not torch.allclose(updated.coarse.a.values, old_a)
    assert updated.coarse.a.cache is not graph.coarse.a.cache
    assert mask_digest(ref.masks_for(e.a, (15, 15))) == frozen_digest
    # A freshly built graph must use new line and coarse numerical factors.
    residual_loss(updated, rhs, cycles=1)
    assert list(graph_levels(updated))[-1].a.cache['lu'] is not list(graph_levels(graph))[-1].a.cache['lu']


def test_loss_weighting_zero_probe_and_nonfinite_guard():
    e, phi, theta, cfg, ref = fixture()
    g = build_graph(e.a, (7, 7), theta, ref.masks_for(e.a, (7, 7)), cfg)
    b = np.column_stack([e.b, np.zeros(49)])
    loss, report, _ = residual_loss(g, b, cycles=3, weighting='terminal')
    assert report['weights'] == [0., 0., 1.]
    assert report['valid_probes'] == 1 and report['skipped_zero'] == 1
    assert loss == torch.log(report['history'][-1].clamp_min(1e-12))
    zero, report, x = residual_loss(g, np.zeros(49), cycles=3)
    assert zero.item() == 0 and report['skipped']
    zero.backward()
    assert np.count_nonzero(x.detach().numpy()) == 0
    with pytest.raises(ValueError, match='finite RHS'):
        residual_loss(g, np.full(49, np.nan))
    with pytest.raises(ValueError, match='Custom weights'):
        residual_loss(g, e.b, custom_weights=[1., -1., 1.], weighting='custom')
    with pytest.raises(ValueError, match='rate_objective'):
        residual_loss(g, e.b, cycles=2, time_weights=[.001, .001])


def test_symbolic_Galerkin_zeros_keep_value_gradient_and_sparse_budget():
    # Matrix product cancellation must keep a derivative path at zero entries.
    values = torch.tensor([1., 1., 1., -1.], dtype=torch.float64, requires_grad=True)
    p = SparseTensor.from_coo([0, 0, 1, 1], [0, 1, 0, 1], values, (2, 2))
    a = SparseTensor.from_scipy(sp.eye(2, format='csr'))
    ac = p.transpose().product(a.product(p))
    assert ac.pattern.nnz == 4 and int((ac.values == 0).sum()) == 2
    ac.values[1].backward()
    assert values.grad.abs().sum() > 0
    e, phi, theta, cfg, ref = fixture()
    with pytest.raises(HierarchyBudgetError, match='p_stored_nnz'):
        build_graph(e.a, (7, 7), theta, ref.masks_for(e.a, (7, 7)), cfg,
                    caps=StructuralCaps(max_p_nnz=1))


def test_baseline_runtime_and_zero_rhs_tolerance_report():
    e, phi, theta, cfg, ref = fixture(15)
    root = build_fixed_hierarchy(e.a, (15, 15), get_strategy(cfg.strategy_name), cfg, WorkStats())
    ps = []; level = root
    while level.coarse is not None:
        ps.append(level.p); level = level.coarse
    bank = runtime_from_prolongations(e.a, (15, 15), ps, cfg)
    actual = bank.cycle(np.zeros(225), e.b)
    expected = classical_cycle(root, np.zeros(225), e.b, cfg, WorkStats())
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    zero = bank.solve(np.zeros(225))
    assert zero['converged'] and zero['cycles'] == 0 and zero['rho'] == []
    assert zero['trajectory_log_rate'] is None
    failed = bank.solve(e.b, max_cycles=0)
    assert not failed['converged'] and failed['reason'] == 'max_cycles'
    assert failed['tolerance'] == max(1e-12, 1e-8 * np.linalg.norm(e.b))
    nonfinite = bank.solve(np.full(225, np.nan))
    assert not nonfinite['converged'] and nonfinite['reason'] == 'nonfinite_initial_residual'


def test_rate_counterexample_and_negative_progress():
    first = progress_rates([1., .8], [.001])
    second = progress_rates([1., .25], [.004])
    assert first['relative_rate'][0] > second['relative_rate'][0]
    assert np.ceil(np.log(1e-8) / np.log(.8)) * .001 == pytest.approx(.083)
    assert np.ceil(np.log(1e-8) / np.log(.25)) * .004 == pytest.approx(.056)
    rates = progress_rates([1., 2., 0.], [.01, .01])
    assert rates['log_rate'][0] < 0 and rates['relative_rate'][0] < 0
    assert rates['log_rate'][1] is None


def test_no_grad_hard_probe_warmup_does_not_detach_line_value_cache():
    e, phi, theta, cfg, ref = fixture(15)
    masks = ref.masks_for(e.a, (15, 15))
    rhs = np.random.default_rng(20).normal(size=(225, 2))
    gradients = []
    for warm_probes in (False, True):
        theta.zero_grad(set_to_none=True)
        graph = build_graph(e.a, (15, 15), theta, masks, cfg)
        if warm_probes:
            with torch.no_grad():
                graph_cycle(graph, torch.tensor(rhs), torch.zeros_like(torch.tensor(rhs)))
        loss, _, _ = residual_loss(graph, rhs, cycles=2)
        loss.backward()
        gradients.append(torch.cat([p.grad.flatten() for p in theta.parameters()]))
    torch.testing.assert_close(gradients[0], gradients[1], atol=2e-12, rtol=2e-11)


def test_fixed_bilinear_edges_ablation_still_predicts_direct_trainable_values(monkeypatch):
    from adaptive_mg.transfer import build_transfer_pattern
    e, phi, theta, cfg, learned = fixture(15)
    fixed = ReferenceModels(phi, theta, mg_config=cfg, support_mode='fixed_bilinear')
    monkeypatch.setattr(fixed.mask_model, 'forward', lambda *a, **k: pytest.fail('Fixed support called Mask CNN'))
    masks = fixed.masks_for(e.a, (15, 15))
    assert fixed.last_report['mask_cnn_seconds'] == 0.
    assert fixed.last_report['support_mode'] == 'fixed_bilinear'
    for mask, shape, coarse in zip(masks, [(15, 15), (7, 7)], [(7, 7), (3, 3)]):
        baseline = build_transfer_pattern(shape, coarse)
        np.testing.assert_array_equal(mask.numpy(), baseline.bilinear_weights != 0.)
    assert fixed.cache_key(e.a, (15, 15)) != learned.cache_key(e.a, (15, 15))
    initial = build_graph(e.a, (15, 15), theta, masks, cfg)
    with torch.no_grad():
        theta.head.bias[5] += .1
    changed = build_graph(e.a, (15, 15), theta, masks, cfg)
    assert not torch.allclose(initial.p.values, changed.p.values)
    assert initial.mask_digest == changed.mask_digest
    assert mask_digest(fixed.masks_for(e.a, (15, 15))) == changed.mask_digest
    restored = ReferenceModels.from_state_dict(fixed.state_dict(include_cache=True))
    assert restored.support_mode == 'fixed_bilinear'
    assert mask_digest(restored.masks_for(e.a, (15, 15))) == changed.mask_digest


def test_actual_coarse_off_stencil_feature_omission_is_reported():
    e, phi, theta, cfg, ref = fixture(15)
    graph = build_graph(e.a, (15, 15), theta, ref.masks_for(e.a, (15, 15)), cfg)
    reports = graph.hierarchy_report['feature_coverage']
    assert reports[0]['outside_9point_stored_nnz'] == 0
    assert reports[1]['outside_9point_numerical_nnz'] > 0
    ac = graph.coarse.a.numpy(); rows, cols = ac.nonzero()
    outside = (np.abs(rows // 7 - cols // 7) > 1) | (np.abs(rows % 7 - cols % 7) > 1)
    values = np.asarray(ac[rows[outside], cols[outside]]).reshape(-1)
    expected = np.linalg.norm(values) / np.linalg.norm(ac.data)
    assert reports[1]['outside_9point_numerical_nnz'] == int(outside.sum())
    assert reports[1]['outside_9point_relative_frobenius'] == pytest.approx(expected)
