import json
import math
from dataclasses import replace

import pytest
import torch

from adaptive_mg.config import MGConfig
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.research_generation import (
    benchmark_generation, rotate_feature_representation, rotation_resampling_diagnostic,
)
from adaptive_mg.v67.research_smoothers import make_research_smoother, d4_feature_transform
from adaptive_mg.v67.strong import load_strong_rules


def _config():
    return AdaptiveConfig(mg=MGConfig(mode='classical', pre_steps=2, post_steps=2,
        nn_levels=1, stencil_backend='csr', max_cycles=5), inference_dtype='float32')


def _components(kind='o2'):
    result = Components.create(hidden=4, seed=401)
    result.smoother = make_research_smoother(kind, hidden=4, depth=1)
    return result.eval()


def test_real_bank_generation_precision_cache_and_unavailable_devices(tmp_path, monkeypatch):
    from adaptive_mg.v67 import research_generation as generation
    monkeypatch.setattr(generation, 'available_generation_devices', lambda: {
        'cpu': {'available': True, 'hardware': 'CPU', 'dtype': 'float32'},
        'mps': {'available': False, 'hardware': None, 'reason': 'test unavailable'},
        'cuda': {'available': False, 'hardware': None, 'reason': 'test unavailable'},
    })
    components = _components()
    before = components.signature()
    result = benchmark_generation({'student': components}, dataset([7], 1, seed=410),
        _config(), load_strong_rules(), tmp_path, repeats=2, warmups=1)
    assert components.signature() == before
    assert result['status'] == 'complete' and not result['performance_certified']
    assert result['manifest']['a100_measurement'] == 'not measured'
    assert len(result['summary']) == 3
    for row in result['summary']:
        assert row['status'] == 'complete' and row['completed_repeats'] == 2
        assert row['cold_preparation_seconds_median'] >= row['bank_setup_seconds_median'] > 0
        assert row['classical_level_operators_identical_to_cpu']
        assert row['actual_learned_level_operators_identical_to_cpu']
    for row in result['runs']:
        if row['status'] == 'not_measured':
            assert row['device'] in {'mps', 'cuda'}
            assert 'generation_seconds' not in row
            continue
        assert row['warm_heavy_generation_calls'] == 0
        assert row['generation_seconds'] > 0 and row['cpu_device_copy_seconds'] > 0
        assert row['smoother_calls'] == int(row['branch'] in {'H_S', 'H_SP'})
        assert row['transfer_calls'] == int(row['branch'] in {'H_P', 'H_SP'})
        assert all(level['a_dtype'] == 'float64' for level in row['hierarchy'])
        assert not row['teacher'] and row['scope'] == 'student_generation'
    assert json.loads((tmp_path / 'generation.json').read_text())['status'] == 'complete'
    assert (tmp_path / 'generation.csv').is_file()


def test_teacher_generation_is_explicit_offline_and_never_student_runtime(tmp_path):
    components = _components('teacher_o2')
    result = benchmark_generation({'offline_teacher': components}, dataset([7], 1, seed=411),
        _config(), load_strong_rules(), tmp_path, repeats=1, warmups=0,
        branches=('H_S',), devices=('cpu',))
    assert result['summary'][0]['status'] == 'complete'
    assert result['summary'][0]['teacher']
    row = result['runs'][0]
    assert row['scope'] == 'offline_teacher_generation' and row['teacher']
    # The helper context closes; direct production and unsanctioned generation
    # remain guarded elsewhere by the unchanged runtime context contract.
    from adaptive_mg.v67.research_runtime import require_generation_allowed
    with pytest.raises(RuntimeError, match='training-only'):
        require_generation_allowed(components.smoother)
    with pytest.raises(RuntimeError, match='training-only'):
        require_generation_allowed(components.smoother, production=True)


def test_failed_generation_is_not_aggregated_as_fast(tmp_path):
    components = _components('student_cnn')
    with torch.no_grad():
        components.smoother.head.bias.fill_(float('nan'))
    result = benchmark_generation({'invalid': components}, dataset([7], 1, seed=412),
        _config(), load_strong_rules(), tmp_path, repeats=2, warmups=0,
        branches=('H_S',), devices=('cpu',))
    summary = result['summary'][0]
    assert summary['status'] == 'failed' and summary['failed_repeats'] == 2
    assert summary['cold_preparation_seconds_median'] is None
    assert all(row['failed_elapsed_seconds'] > 0 for row in result['runs'])
    assert all('nonfinite' in row['error'] for row in result['runs'])


@pytest.mark.parametrize('split', ['test', 'final', 'ood', 'audit'])
def test_generation_helper_rejects_final_data_scope(tmp_path, split):
    with pytest.raises(ValueError, match='final/OOD'):
        benchmark_generation({'student': _components()}, dataset([7], 1, seed=413),
            _config(), load_strong_rules(), tmp_path, repeats=1, split=split)
    assert not (tmp_path / 'generation.json').exists()


def test_generation_preserves_existing_output(tmp_path):
    marker = tmp_path / 'generation.json'
    marker.write_text('existing result')
    with pytest.raises(FileExistsError):
        benchmark_generation({'student': _components()}, dataset([7], 1, seed=414),
            _config(), load_strong_rules(), tmp_path, repeats=1)
    assert marker.read_text() == 'existing result'


@pytest.mark.parametrize('reflection', [False, True])
def test_continuous_feature_action_agrees_with_discrete_d4_on_quarter_turns(reflection):
    x = torch.randn(2, 10, 9, 9, dtype=torch.float64)
    for k in range(4):
        actual, _ = rotate_feature_representation(x, math.pi * k / 2, reflection)
        expected = d4_feature_transform(x, k, reflection)
        assert torch.allclose(actual, expected, atol=1e-13)


@pytest.mark.parametrize('kind', ['ordinary_control', 'd4', 'o2'])
def test_actual_arbitrary_angle_diagnostic_records_nontrivial_sampling_error(kind, tmp_path):
    torch.manual_seed(420)
    model = make_research_smoother(kind, hidden=4, depth=1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'head' in name:
                parameter.normal_(0., .2)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    features = torch.nn.functional.avg_pool2d(torch.randn(1, 10, 17, 17), 3, stride=1, padding=1)
    report = rotation_resampling_diagnostic(model, features, angles=(0., 23.),
                                            output=tmp_path / 'rotation.json')
    assert not report['exact_arbitrary_spatial_equivariance_claim']
    assert report['numerical_rotation_ood_solve'] == 'not measured by this helper'
    no_rotation = [r for r in report['samples'] if r['angle_deg'] == 0. and not r['reflected']][0]
    assert no_rotation['relative_direction_l2'] < 1e-5
    nontrivial = [r for r in report['samples'] if r['angle_deg'] == 23.]
    assert all(r['interior_pixels'] > 0 and math.isfinite(r['relative_direction_l2']) for r in nontrivial)
    assert any(r['relative_direction_l2'] > 1e-4 for r in nontrivial)
    assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items())
