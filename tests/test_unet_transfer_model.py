"""Residual U-net teacher, shared P constraints and deployable student contracts."""
import numpy as np
import pytest
import torch

from adaptive_mg import DiffusionCase, MGConfig, assemble_stiffness
from adaptive_mg.transfer import build_transfer_pattern, coarse_fine_indices
from adaptive_mg.v67.banks import Stats
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.models import Components, TransferNet
from adaptive_mg.v67.research_runtime import (offline_teacher_generation,
    require_generation_allowed, research_spec, restore_expert)
from adaptive_mg.v67.research_transfer import project_transfer_weights
from adaptive_mg.v67.strong import PreparedStrongMG, StrongRules
from adaptive_mg.v67.unet_transfer import (CompactTransferStudent,
    ResidualFeatureBlock, ResidualUNetTransferTeacher, make_unet_transfer)
from adaptive_mg.v67.unroll import cycle, make_graph


@pytest.fixture(autouse=True)
def single_torch_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('shape', [(1, 1), (3, 3), (7, 7), (15, 15), (31, 31),
                                  (7, 15), (15, 7), (1, 7)])
def test_teacher_restores_exact_spatial_shape_and_zero_delta(shape):
    net = ResidualUNetTransferTeacher(hidden=4, depth=2)
    prediction = net(torch.randn(2, 10, *shape))
    assert prediction.shape == (2, 16, *shape)
    assert torch.count_nonzero(prediction) == 0
    assert net.training_only


def test_residual_block_keeps_identity_when_update_is_zero():
    block = ResidualFeatureBlock(4, 4)
    with torch.no_grad():
        for p in block.update.parameters():
            p.zero_()
    x = torch.randn(1, 4, 7, 9)
    torch.testing.assert_close(block(x), x, rtol=0, atol=0)


@pytest.mark.parametrize('parameterization', ['softmax', 'affine'])
def test_identical_teacher_student_projection_and_boundary_constraints(parameterization):
    teacher = ResidualUNetTransferTeacher(hidden=4, parameterization=parameterization).double()
    student = CompactTransferStudent(hidden=4, parameterization=parameterization).double()
    pattern = build_transfer_pattern(7)
    base = pattern.bilinear_weights
    features = torch.randn(1, 10, 7, 7, dtype=torch.double)
    baseline = torch.tensor(base)
    for net in (teacher, student):
        initial = project_transfer_weights(net, pattern, net(features), base)
        torch.testing.assert_close(initial, baseline, rtol=0, atol=0)
    raw = torch.randn(1, 16, 7, 7, dtype=torch.double, requires_grad=True)
    weights = project_transfer_weights(teacher, pattern, raw, base)
    torch.testing.assert_close(weights, project_transfer_weights(student, pattern, raw, base),
                               rtol=0, atol=0)
    torch.testing.assert_close(weights.sum(1), baseline.sum(1), rtol=0, atol=1e-14)
    fixed = torch.tensor(coarse_fine_indices(pattern))
    torch.testing.assert_close(weights[fixed], baseline[fixed], rtol=0, atol=0)
    assert torch.count_nonzero(weights[baseline == 0]) == 0
    weights.square().sum().backward()
    gradient = raw.grad[0].permute(1, 2, 0).reshape_as(baseline)
    assert torch.count_nonzero(gradient[baseline == 0]) == 0
    assert torch.count_nonzero(gradient[fixed]) == 0
    assert gradient.abs().sum() > 0


@pytest.mark.parametrize('architecture', ['teacher', 'student'])
def test_checkpoint_preserves_architecture_constraints_and_signature(tmp_path, architecture):
    model = Components.create(hidden=4)
    model.transfer = make_unet_transfer(architecture, hidden=4, parameterization='affine')
    with torch.no_grad():
        model.transfer.head.weight.normal_(0., .01)
    signature = model.generation_signature()
    spec = research_spec(model.transfer)
    model.save(tmp_path / 'expert.pt')
    loaded = Components.load(tmp_path / 'expert.pt')
    assert research_spec(loaded.transfer) == spec
    assert loaded.generation_signature() == signature
    x = torch.randn(1, 10, 7, 15)
    torch.testing.assert_close(loaded.transfer(x), model.transfer(x), rtol=0, atol=0)
    assert loaded.transfer.training_only == (architecture == 'teacher')
    assert loaded.transfer.reference == 'frozen_parent'
    assert loaded.transfer.support == 'support_preserving'
    loaded.transfer.complexity_caps['max_ac_ratio'] = 1.3
    assert loaded.generation_signature() != signature
    bad = dict(spec['config'], training_only=architecture != 'teacher')
    with pytest.raises(ValueError, match='metadata'):
        restore_expert(dict(family='unet_transfer', config=bad))


def test_existing_transfer_checkpoints_still_load(tmp_path):
    model = Components.create(hidden=4)
    model.save(tmp_path / 'legacy.pt')
    loaded = Components.load(tmp_path / 'legacy.pt')
    assert type(loaded.transfer) is TransferNet
    assert loaded.generation_signature() == model.generation_signature()


def _setup(plan='jacobi_bilinear_full', *, teacher=True):
    cfg = AdaptiveConfig(mg=MGConfig(mode='classical', strategy_name=plan,
        nn_levels=1, stencil_backend='csr', max_cycles=10), mode='research',
        branch='H_P', use_smoother=False, use_transfer=True, transfer_levels='all',
        spatial=False, gate_mode='open')
    rules = StrongRules()
    rules = rules.replace_strategies({key: plan for key in rules.rule_ids})
    model = Components.create(hidden=4)
    model.transfer = make_unet_transfer('teacher' if teacher else 'student', hidden=4)
    return cfg, rules, model


def test_teacher_generation_guard_including_cached_bank():
    a = assemble_stiffness(DiffusionCase(7, epsilon=1., angle_deg=0.))
    cfg, rules, model = _setup()
    prepared = PreparedStrongMG(a, 7, model, cfg, rules)
    with pytest.raises(RuntimeError, match='training-only'):
        prepared.ensure_branch('H_P', Stats())
    with offline_teacher_generation():
        assert prepared.ensure_branch('H_P', Stats()) is prepared.learned
        with pytest.raises(RuntimeError, match='training-only'):
            require_generation_allowed(model.transfer, production=True)
    with pytest.raises(RuntimeError, match='training-only'):
        prepared.solve(np.ones(49))
    require_generation_allowed(CompactTransferStudent(hidden=4), production=True)


@pytest.mark.parametrize('plan', ['jacobi_bilinear_full', 'line_x_operator_semi_y'])
@pytest.mark.parametrize('teacher', [True, False])
@pytest.mark.parametrize('perturbed', [True, False])
def test_runtime_and_unrolled_hierarchy_match(plan, teacher, perturbed):
    torch.manual_seed(137)
    a = assemble_stiffness(DiffusionCase(15, epsilon=.05, angle_deg=27.))
    cfg, rules, model = _setup(plan, teacher=teacher)
    if perturbed:
        with torch.no_grad():
            model.transfer.head.weight.normal_(0., .01)
    prepared = PreparedStrongMG(a, 15, model, cfg, rules)
    with offline_teacher_generation():
        actual = prepared.ensure_branch('H_P', Stats())
    graph = make_graph(a, (15, 15), model, cfg)
    parent = prepared.classical
    while actual.coarse is not None:
        np.testing.assert_allclose(actual.p.toarray(), graph.p.numpy().toarray(), rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(actual.coarse.a.toarray(), graph.coarse.a.numpy().toarray(),
                                   rtol=4e-6, atol=2e-8)
        np.testing.assert_array_equal(actual.base_weights, parent.base_weights)
        if not perturbed:
            np.testing.assert_array_equal(actual.p.toarray(), parent.p.toarray())
        actual, graph, parent = actual.coarse, graph.coarse, parent.coarse


def test_vcycle_loss_backpropagates_into_encoder_and_bottleneck():
    torch.manual_seed(29)
    a = assemble_stiffness(DiffusionCase(7, epsilon=.05, angle_deg=27.))
    cfg, _, model = _setup()
    # Initial zero head preserves P; encoder receives gradients after it moves.
    with torch.no_grad():
        model.transfer.head.weight.normal_(0., .01)
    graph = make_graph(a, (7, 7), model, cfg)
    b = torch.randn(49, dtype=torch.double)
    x = cycle(graph, torch.zeros_like(b), b, model, cfg)
    residual = b - graph.a.apply(x)
    loss = residual.square().sum() / b.square().sum()
    loss.backward()
    for module in (model.transfer.stem, model.transfer.downsamples[0], model.transfer.bottleneck):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
