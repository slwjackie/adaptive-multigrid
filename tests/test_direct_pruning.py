"""The compact network must equal channel masking, including residual skips."""
from copy import deepcopy
import json

import pytest
import torch

from adaptive_mg.transfer import build_transfer_pattern, coarse_fine_indices
from adaptive_mg.v67.direct_pruning import prune_direct
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.research_runtime import research_spec
from adaptive_mg.v67.research_transfer import project_transfer_weights
from adaptive_mg.v67.unet_transfer import CompactTransferStudent, ResidualUNetTransferTeacher


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _model(*, dtype=torch.float64):
    model = Components.create(hidden=4, seed=97)
    model.transfer = CompactTransferStudent(hidden=8).to(dtype=dtype)
    with torch.no_grad():
        for parameter in model.transfer.parameters():
            parameter.normal_(0., .08)
    model.eval()
    model.metadata.update(training_baseline_digest='fixed-em5-test-binding',
                          certificate={'validated': True})
    return model


def _masked_reference(net, features, kept):
    mask = torch.zeros(1, net.hidden, 1, 1, dtype=features.dtype)
    mask[:, kept] = 1.
    body = net.body
    h = body.net[1](body.net[0](features)) * mask
    h = body.net[3](body.net[2](h)) * mask
    branch = body.residual[1](body.residual[0](h)) * mask
    h = h + body.residual[2](branch) * mask
    h = h + body.refine(body.refine_activation(h)) * mask
    return net.head(h)


def test_full_width_is_exact_copy_without_mutating_source_or_rng():
    source = _model()
    features = torch.randn(2, 10, 7, 9, dtype=torch.float64)
    old_state = deepcopy(source.transfer.state_dict())
    old_metadata = deepcopy(source.metadata)
    old_rng = torch.random.get_rng_state()
    pruned, report = prune_direct(source, 8)
    torch.testing.assert_close(torch.random.get_rng_state(), old_rng, rtol=0, atol=0)
    torch.testing.assert_close(pruned.transfer(features), source.transfer(features), rtol=0, atol=0)
    assert report['original_parameter_count'] == report['pruned_parameter_count'] == 3208
    assert report['source_transfer_signature'] == report['pruned_transfer_signature']
    assert source.metadata == old_metadata
    for key, value in old_state.items():
        torch.testing.assert_close(source.transfer.state_dict()[key], value, rtol=0, atol=0)
    assert pruned is not source and pruned.transfer is not source.transfer


@pytest.mark.parametrize('hidden,expected_parameters', [(6, 1978), (4, 1036)])
def test_physical_pruning_matches_residual_aware_masking(hidden, expected_parameters):
    source = _model()
    features = torch.randn(2, 10, 7, 9, dtype=torch.float64)
    pruned, report = prune_direct(source, hidden)
    torch.testing.assert_close(pruned.transfer(features),
        _masked_reference(source.transfer, features, report['kept_channels']), rtol=1e-12, atol=1e-12)
    assert type(pruned.transfer) is CompactTransferStudent
    assert report['pruned_parameter_count'] == expected_parameters
    assert pruned.transfer.head.in_channels == hidden
    assert all('mask' not in name and 'orig' not in name for name in pruned.transfer.state_dict())
    assert all(not module._forward_hooks for module in pruned.transfer.modules())
    assert pruned.metadata['training_baseline_digest'] == source.metadata['training_baseline_digest']
    assert source.metadata['certificate']['validated']
    assert not pruned.metadata['certificate']['validated']
    assert not pruned.metadata['final_time_improvement_certified']
    for name in ('smoother', 'controller', 'detector'):
        assert pruned.expert_signature(name) == source.expert_signature(name)
    json.dumps(report, allow_nan=False)


def test_ranking_ties_are_stable_and_source_is_independent():
    source = _model()
    with torch.no_grad():
        for p in source.transfer.parameters():
            p.zero_()
        source.transfer.head.weight[0, 6, 0, 0] = 7.
        source.transfer.head.weight[0, 2, 0, 0] = 5.
    first, a = prune_direct(source, 4)
    second, b = prune_direct(source, 4)
    assert a == b
    assert a['kept_channels'] == [0, 1, 2, 6]
    with torch.no_grad():
        first.transfer.head.bias.add_(1.)
    assert torch.count_nonzero(source.transfer.head.bias) == 0
    assert torch.count_nonzero(second.transfer.head.bias) == 0


@pytest.mark.parametrize('parameterization', ['softmax', 'affine'])
def test_pruned_checkpoint_roundtrip_and_transfer_constraints(tmp_path, parameterization):
    source = _model(dtype=torch.float32)
    source.transfer.parameterization = parameterization
    pruned, report = prune_direct(source, 6)
    path = tmp_path / 'pruned.pt'
    pruned.save(path)
    restored = Components.load(path)
    assert research_spec(restored.transfer) == research_spec(pruned.transfer)
    assert restored.metadata['direct_pruning'] == report
    assert restored.generation_signature() == pruned.generation_signature()
    features = torch.randn(1, 10, 7, 7)
    torch.testing.assert_close(restored.transfer(features), pruned.transfer(features), rtol=0, atol=0)
    restored.transfer.double()
    pattern = build_transfer_pattern(7)
    base = torch.tensor(pattern.bilinear_weights)
    weights = project_transfer_weights(restored.transfer, pattern,
        restored.transfer(features.double()), pattern.bilinear_weights)
    torch.testing.assert_close(weights.sum(1), base.sum(1), rtol=0, atol=1e-14)
    fixed = torch.tensor(coarse_fine_indices(pattern))
    torch.testing.assert_close(weights[fixed], base[fixed], rtol=0, atol=0)
    assert torch.count_nonzero(weights[base == 0]) == 0
    assert restored.transfer.reference == 'frozen_parent'
    assert restored.transfer.complexity_caps == source.transfer.complexity_caps


@pytest.mark.parametrize('width', [0, 9, True, 4.5])
def test_invalid_width_rejected(width):
    with pytest.raises(ValueError, match='hidden'):
        prune_direct(_model(), width)


def test_teacher_and_nonfinite_source_rejected():
    model = _model()
    model.transfer = ResidualUNetTransferTeacher(hidden=4)
    with pytest.raises(ValueError, match='not a teacher'):
        prune_direct(model, 2)
    model = _model()
    with torch.no_grad():
        model.transfer.head.bias[0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        prune_direct(model, 4)
