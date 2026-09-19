import math

import pytest
import torch

from adaptive_mg.models import NeuralSmootherNet
from adaptive_mg.transfer import OFFSETS_9
from adaptive_mg.v67.research_smoothers import (
    O2Fields, O2PointwiseBlock, analytic_harmonics, d4_feature_transform,
    d4_stencil_transform, make_research_smoother, research_smoother_from_spec,
    stencil_to_o2_fields, transform_o2_fields,
)


KINDS = ('ordinary_control', 'student_cnn', 'student_residual', 'd4', 'o2',
         'teacher_cnn', 'teacher_o2')


def _excite_heads(model):
    # Avoid passing equivariance checks only because the initial stencil is I.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'head' in name:
                parameter.normal_(0., .2)


def _rotation(angle, reflect=False):
    c, s = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    if reflect:
        rotation = rotation @ torch.diag(torch.tensor([1., -1.], dtype=torch.float64))
    return rotation


@pytest.mark.parametrize('kind', KINDS)
def test_shared_shapes_normalization_gain_and_differentiability(kind):
    torch.manual_seed(217)
    model = make_research_smoother(kind, hidden=4, depth=2).double()
    _excite_heads(model)
    x = torch.randn(2, 10, 7, 5, dtype=torch.float64, requires_grad=True)
    direction, gain = model.direction_and_gain(x)
    assert direction.shape == (2, 1, 9, 7, 5)
    assert gain.shape == (2, 1)
    assert torch.isfinite(direction).all() and torch.isfinite(gain).all()
    assert (direction.square().sum(2) <= 1. + 1e-12).all()
    assert torch.allclose(direction.square().sum(2), torch.ones(2, 1, 7, 5, dtype=torch.float64), atol=3e-6)
    assert (gain >= model.gain_min).all() and (gain <= model.gain_max).all()
    residual = torch.randn(2, 1, 7, 5, dtype=torch.float64)
    expected = NeuralSmootherNet.apply_coefficients(direction, residual) * gain[:, :, None, None]
    assert torch.allclose(model(x, residual), expected)
    (expected.square().mean() + gain.mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('kind', KINDS)
def test_spec_roundtrip_and_teacher_tag(kind):
    model = make_research_smoother(kind, hidden=4, depth=2)
    clone = research_smoother_from_spec(model.architecture_spec())
    clone.load_state_dict(model.state_dict(), strict=True)
    assert clone.architecture_spec() == model.architecture_spec()
    assert clone.training_only == kind.startswith('teacher_')
    x = torch.randn(1, 10, 5, 5)
    for a, b in zip(model.direction_and_gain(x), clone.direction_and_gain(x)):
        assert torch.equal(a, b)
    assert clone.cost_metadata()['parameters'] == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize('shape', [(7, 7), (7, 5)])
@pytest.mark.parametrize('kind', ['d4', 'o2'])
def test_full_network_lattice_d4_rotation_and_reflection(kind, shape):
    torch.manual_seed(229)
    model = make_research_smoother(kind, hidden=4, depth=2).double()
    _excite_heads(model)
    x = torch.randn(2, 10, *shape, dtype=torch.float64)
    directions, gain = model.direction_and_gain_single(x)
    # All eight actions; scalar feature alone must not be channel-permuted.
    for reflection in (False, True):
        for k in range(4):
            transformed = d4_feature_transform(x, k, reflection)
            td, tg = model.direction_and_gain_single(transformed)
            expected = d4_stencil_transform(directions, k, reflection)
            assert torch.allclose(td, expected, rtol=1e-9, atol=2e-10)
            assert torch.allclose(tg, gain, rtol=1e-11, atol=1e-12)
            assert torch.equal(d4_feature_transform(transformed, k, reflection, inverse=True), x)


@pytest.mark.parametrize('shape,expected_calls', [((7, 7), 1), ((7, 5), 2)])
def test_d4_batches_encoder_calls_and_discloses_eightfold_compute(shape, expected_calls):
    model = make_research_smoother('d4', hidden=4)
    calls = []
    hook = model.encoder.body.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[0]))
    model.direction_and_gain(torch.randn(2, 10, *shape))
    hook.remove()
    assert len(calls) == expected_calls
    assert sum(calls) == 8 * 2
    assert model.cost_metadata()['equivalent_encoder_evaluations'] == 8


@pytest.mark.parametrize('angle', [.173, .731, 1.382])
@pytest.mark.parametrize('reflection', [False, True])
def test_analytic_kernel_moment_lifting_and_fiber_nonlinearity_continuous_o2(angle, reflection):
    torch.manual_seed(239)
    matrix = _rotation(angle, reflection)
    points = torch.randn(17, 2, dtype=torch.float64)
    moved = points @ matrix.T
    for order in (1, 2):
        c, s = math.cos(order * angle), math.sin(order * angle)
        rep = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
        if reflection:
            rep = rep @ torch.diag(torch.tensor([1., -1.], dtype=torch.float64))
        assert torch.allclose(analytic_harmonics(moved, order),
                              analytic_harmonics(points, order) @ rep.T, atol=2e-15)
    features = torch.randn(2, 10, 5, 5, dtype=torch.float64)
    offsets = torch.tensor(OFFSETS_9, dtype=torch.float64)
    lifted = stencil_to_o2_fields(features)
    moved_lifted = stencil_to_o2_fields(features, offsets @ matrix.T)
    for actual, expected in zip(moved_lifted, transform_o2_fields(lifted, angle, reflection)):
        assert torch.allclose(actual, expected, atol=1e-14)
    fields = O2Fields(torch.randn(2, 4, 5, 5, dtype=torch.float64),
                      torch.randn(2, 2, 2, 5, 5, dtype=torch.float64),
                      torch.randn(2, 2, 2, 5, 5, dtype=torch.float64))
    block = O2PointwiseBlock(4, 2).double()
    actual = block(transform_o2_fields(fields, angle, reflection))
    expected = transform_o2_fields(block(fields), angle, reflection)
    for a, b in zip(actual, expected):
        assert torch.allclose(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('reflection', [False, True])
def test_continuous_tensor_stencil_head_changes_frame_not_square_lattice(reflection):
    torch.manual_seed(241)
    model = make_research_smoother('o2', hidden=4, depth=2).double()
    _excite_heads(model)
    fields = O2Fields(torch.randn(2, 4, 3, 3, dtype=torch.float64),
                      torch.randn(2, 2, 2, 3, 3, dtype=torch.float64),
                      torch.randn(2, 2, 2, 3, 3, dtype=torch.float64))
    angle = .317
    offsets = torch.tensor(OFFSETS_9, dtype=torch.float64)
    values = model.stencil_logits_from_fields(fields)
    rotated = model.stencil_logits_from_fields(transform_o2_fields(fields, angle, reflection),
                                                offsets @ _rotation(angle, reflection).T)
    assert torch.allclose(values, rotated, atol=1e-13)
    # The SAME fixed nine offsets are not closed under an arbitrary rotation.
    fixed_offsets = model.stencil_logits_from_fields(transform_o2_fields(fields, angle, reflection))
    assert not torch.allclose(values, fixed_offsets, atol=1e-6)
    assert model.cost_metadata()['arbitrary_angle_lattice_equivariance'] == 'approximate_requires_measurement'


def test_ordinary_control_retains_current_body_and_is_not_accidentally_d4():
    torch.manual_seed(251)
    model = make_research_smoother('ordinary_control', hidden=4).double()
    legacy = NeuralSmootherNet(hidden=4, basis_count=1, split_direction_gain=True).double()
    legacy.body.load_state_dict(model.body.state_dict(), strict=True)
    legacy.head.load_state_dict(model.head.state_dict(), strict=True)
    x = torch.randn(1, 10, 7, 7, dtype=torch.float64)
    assert torch.equal(model.direction_and_gain(x)[0], legacy.direction_and_gain(x)[0])
    _excite_heads(model)
    actual = model.direction_and_gain_single(d4_feature_transform(x, 1))[0]
    expected = d4_stencil_transform(model.direction_and_gain_single(x)[0], 1)
    assert not torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize('kwargs', [{'kind': 'no_such_kind'}, {'hidden': 0}, {'gain_min': 0},
                                  {'gain_min': 2., 'gain_max': 1.}, {'direction_epsilon': 0},
                                  {'depth': 0}])
def test_invalid_architecture_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        make_research_smoother(**kwargs)


def test_invalid_spec_version_rejected():
    with pytest.raises(ValueError, match='version'):
        research_smoother_from_spec({'version': 2, 'kind': 'o2'})


def test_o2_moment_lifting_retains_all_ten_input_channels():
    # Treat each input coordinate as a basis vector. The lifted 10x10 map must
    # have full rank, so equal input information is more than a shape claim.
    basis = torch.eye(10, dtype=torch.float64).reshape(10, 10, 1, 1)
    fields = stencil_to_o2_fields(basis)
    lifted = torch.cat([field.reshape(10, -1) for field in fields], 1)
    assert lifted.shape == (10, 10)
    assert torch.linalg.matrix_rank(lifted) == 10
