"""Same-depth CNN ablation, checkpoint identity, and actual EM/V-cycle learning."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from adaptive_mg import MGConfig
from adaptive_mg.data import make_examples
from adaptive_mg.models import PlainConvBody, ResidualConvBody, make_spatial_body
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.em_line_models import make_em_line_components
from adaptive_mg.v67.em_line_training import em_imitation_objective, line_vcycle_objective
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.unet_transfer import CompactTransferStudent

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('hidden', [4, 8])
def test_plain_matches_residual_depth_width_parameters_and_receptive_field(hidden):
    plain = make_spatial_body('plain5', 10, hidden)
    residual = make_spatial_body('residual5', 10, hidden)
    assert isinstance(plain, PlainConvBody)
    assert isinstance(residual, ResidualConvBody)
    def geometry(body):
        return [(m.in_channels, m.out_channels, m.kernel_size, m.padding, m.stride)
                for m in body.modules() if isinstance(m, nn.Conv2d)]
    assert geometry(plain) == geometry(residual)
    assert len(geometry(plain)) == 5
    assert sum(isinstance(m, nn.GELU) for m in plain.modules()) == 4
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in residual.parameters())
    assert plain.spatial_convolutions == residual.spatial_convolutions == 5
    assert plain.receptive_radius == residual.receptive_radius == 5
    for shape in ((1, 1), (7, 15)):
        assert plain(torch.randn(2, 10, *shape)).shape == (2, hidden, *shape)


def test_plain_has_no_skip_around_last_convolution_and_no_zeroed_body_layers():
    torch.manual_seed(71)
    body = PlainConvBody(10, 4)
    convolutions = [m for m in body.modules() if isinstance(m, nn.Conv2d)]
    assert all(torch.count_nonzero(m.weight) > 0 for m in convolutions)
    x = torch.randn(2, 10, 7, 7)
    body(x).square().mean().backward()
    assert all(m.weight.grad is not None and torch.isfinite(m.weight.grad).all()
               and m.weight.grad.abs().sum() > 0 for m in convolutions)
    with torch.no_grad():
        convolutions[-1].weight.zero_()
        convolutions[-1].bias.zero_()
    assert torch.count_nonzero(body(x)) == 0


@pytest.mark.parametrize('transfer_kind,smoother_kind', [
    ('plain5', 'residual5'), ('residual5', 'plain5'), ('plain5', 'plain5')])
def test_em_components_roundtrip_body_selection_and_initial_behavior(tmp_path, transfer_kind, smoother_kind):
    default = make_em_line_components(hidden=4, seed=18)
    model = make_em_line_components(hidden=4, seed=18,
        transfer_body_kind=transfer_kind, smoother_body_kind=smoother_kind)
    features = torch.randn(1, 10, 7, 15)
    torch.testing.assert_close(model.transfer(features), default.transfer(features), rtol=0, atol=0)
    assert model.transfer.body_kind == transfer_kind
    assert model.smoother.body_kind == smoother_kind
    assert model.generation_signature() != default.generation_signature()
    with torch.no_grad():
        model.transfer.head.weight.normal_(0, .01)
        model.smoother.head.weight.normal_(0, .01)
    model.save(tmp_path/'plain.pt')
    restored = Components.load(tmp_path/'plain.pt')
    assert restored.generation_signature() == model.generation_signature()
    assert restored.transfer.body_kind == transfer_kind
    assert restored.smoother.body_kind == smoother_kind
    torch.testing.assert_close(restored.transfer(features), model.transfer(features), rtol=0, atol=0)


def test_unet_student_plain_checkpoint_roundtrip_and_teacher_unchanged(tmp_path):
    from adaptive_mg.v67 import unet_transfer_study as study
    settings = json.loads((ROOT/'configs/v6_7_unet_transfer_smoke.json').read_text())
    teacher = study.initial_model(settings, 'teacher')
    settings['unet_transfer']['student_body_kind'] = 'plain5'
    study.validate_settings(settings)
    model = study.initial_model(settings, 'direct')
    distilled = study.initial_model(settings, 'distilled')
    assert isinstance(model.transfer, CompactTransferStudent)
    assert isinstance(model.transfer.body, PlainConvBody)
    assert model.transfer.state_dict().keys() == distilled.transfer.state_dict().keys()
    for key, value in model.transfer.state_dict().items():
        torch.testing.assert_close(value, distilled.transfer.state_dict()[key], rtol=0, atol=0)
    assert teacher.expert_signature('transfer') == study.initial_model(settings, 'teacher').expert_signature('transfer')
    model.save(tmp_path/'student.pt')
    restored = Components.load(tmp_path/'student.pt')
    assert restored.transfer.body_kind == 'plain5'
    assert restored.generation_signature() == model.generation_signature()


def test_plain_transfer_learns_all_level_em_targets_then_joint_cycle_gradients():
    example = make_examples([15], per_size=4, seed=32)[-1]
    cfg = AdaptiveConfig(mg=MGConfig(strategy_name='line_alt_bilinear_full',
        pre_steps=2, post_steps=2, stencil_backend='csr', max_cycles=50),
        mode='research', branch='H_SP', replace_pre=2, replace_post=2,
        smoother_levels='all', transfer_levels='all', spatial=False,
        gate_mode='open', use_learned_controller=False)
    model = make_em_line_components(hidden=4, seed=18,
        transfer_body_kind='plain5', smoother_body_kind='plain5')
    optimizer = torch.optim.Adam(model.transfer.parameters(), lr=1e-3)
    # The deliberately zero output head needs its first update before gradients
    # can reach the body, in either architecture.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, report = em_imitation_objective(example, model, cfg)
        assert report['teacher_targets_detached'] and len(report['levels']) == 2
        loss.backward()
        optimizer.step()
    for layer in model.transfer.body.modules():
        if isinstance(layer, nn.Conv2d):
            assert torch.isfinite(layer.weight.grad).all()
            assert layer.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.smoother.parameters())
    with torch.no_grad():
        model.smoother.head.weight.normal_(0, .01)
    for module in model.modules():
        module.zero_grad(set_to_none=True)
    loss, report = line_vcycle_objective(example, model, cfg, cycles=2,
        hard_smoothing=1, hard_cycles=1)
    assert torch.isfinite(loss) and len(report['history']) == 2
    loss.backward()
    for module in (model.transfer, model.smoother):
        assert all(layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
                   and layer.weight.grad.abs().sum() > 0
                   for layer in module.body.modules() if isinstance(layer, nn.Conv2d))


@pytest.mark.parametrize('field', ['transfer_body_kind', 'smoother_body_kind'])
def test_em_study_rejects_unmatched_depth_ablation(field):
    from adaptive_mg.v67.em_line_study import validate_settings
    settings = json.loads((ROOT/'configs/v6_7_em_line_smoke.json').read_text())
    settings[field] = 'compact3'
    with pytest.raises(ValueError):
        validate_settings(settings)


@pytest.mark.parametrize('size', ['smoke', 'research'])
def test_plain_config_only_changes_transfer_body(size):
    from adaptive_mg.v67.em_line_study import validate_settings
    base = json.loads((ROOT/f'configs/v6_7_em_line_{size}.json').read_text())
    plain = json.loads((ROOT/f'configs/v6_7_em_line_plain5_{size}.json').read_text())
    validate_settings(plain)
    expected = deepcopy(base)
    expected['transfer_body_kind'] = 'plain5'
    assert plain == expected


def test_plain_em_study_prepare_train_and_resume_preserve_architecture(tmp_path):
    from adaptive_mg.v67 import em_line_study as study
    from adaptive_mg.v67.limited import digest_file
    run = tmp_path/'plain_run'
    study.main(['prepare', '--config', str(ROOT/'configs/v6_7_em_line_plain5_smoke.json'),
                '--run-dir', str(run)])
    initial = Components.load(run/'initial.pt')
    assert initial.transfer.body_kind == 'plain5'
    assert initial.smoother.body_kind == 'residual5'
    study.main(['train', '--run-dir', str(run)])
    digests = {}
    for stage in study.STAGES:
        path = run/'checkpoints'/stage/'candidate.pt'
        digests[stage] = digest_file(path)
        model = Components.load(path)
        assert model.transfer.body_kind == 'plain5'
        assert model.smoother.body_kind == 'residual5'
        assert model.metadata['em_line_stage'] == stage
        status = json.loads((path.parent/'status.json').read_text())
        assert status['all_updates_completed'] and status['skipped'] == 0
    study.main(['train', '--run-dir', str(run), '--resume'])
    assert digests == {stage: digest_file(run/'checkpoints'/stage/'candidate.pt')
                       for stage in study.STAGES}
