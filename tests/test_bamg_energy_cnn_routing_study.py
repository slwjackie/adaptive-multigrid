from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from adaptive_mg.v67 import bamg_energy_cnn_routing_study as study
from adaptive_mg.v67 import bamg_energy_cnn_routing as routing
from adaptive_mg.v67.bamg_energy_study import file_digest, read_json, write_json
from test_bamg_energy_cnn_study import inputs


def routing_inputs(tmp_path):
    source, teacher, config = inputs(tmp_path)
    settings = read_json(config)
    settings.update(version=study.VERSION, sampling='epoch_shuffle', lambdas=[10, 30, 50],
        numerical_baselines=['fixed_C', 'em5_C', 'bootstrap_C'],
        routing=dict(stage='finetuned', unseen_grid_policy='lower_anchor_clamp'))
    settings['benchmark']['cache_hierarchies'] = False
    write_json(config, settings)
    return source, teacher, config


def test_three_lambda_training_shares_cases_banks_and_resumes_without_retiming(tmp_path, monkeypatch):
    source, teacher, config = routing_inputs(tmp_path)
    original = {str(p): file_digest(p) for p in source.rglob('*') if p.is_file()}
    out = tmp_path / 'routing'
    first = study.run(config, out, source_study=source, teacher_comparison=teacher, stop_after=1)
    assert not first['completed'] and not (out / 'routing_selection.json').exists()
    labels = {str(p): file_digest(p) for p in (out / 'labels').rglob('*') if p.is_file()}
    result = study.run(config, out, source_study=source, teacher_comparison=teacher, resume=True)
    assert result['completed'] and result['lambdas'] == [10., 30., 50.]
    assert len(result['runs']) == 6
    assert result['train_cases'] == result['validation_cases'] == 1
    assert set(study.BASELINES) <= set(result['benchmark']['methods'])
    assert len(result['benchmark']['methods']) == 16  # 12 CNN exports, old CNN, three controls.
    assert len(result['routed_benchmark']['methods']) == 5
    assert not (out / 'final').exists()
    identities = [read_json(out / 'runs' / name / 'identity.json') for name in result['runs']]
    first_identity = identities[0]
    for identity in identities:
        assert identity['train'] == first_identity['train']
        assert identity['validation'] == first_identity['validation']
        assert identity['supervised']['seed'] == first_identity['supervised']['seed']
        assert identity['provenance']['teacher_bank_manifest_sha256'] == file_digest(out / 'teacher_bank_manifest.json')
    for strength in (10, 30, 50):
        scratch = read_json(out / 'runs' / routing.run_name(strength, 'scratch', 6101) / 'identity.json')
        warm = read_json(out / 'runs' / routing.run_name(strength, 'warm', 6101) / 'identity.json')
        assert scratch['teacher'] == warm['teacher']
        assert scratch['provenance'] == warm['provenance']
        assert scratch['provenance']['lambda_value'] == strength
    assert all(file_digest(path) == expected for path, expected in labels.items())
    before = {str(p): file_digest(p) for folder in ('validation', 'validation_routed')
              for p in (out / folder / 'benchmark_cases').glob('*.json')}
    monkeypatch.setattr(study.validation_benchmark, '_build_bank',
                        lambda *a, **k: pytest.fail('completed observations must not be retimed'))
    repeated = study.run(config, out, source_study=source, teacher_comparison=teacher, resume=True)
    assert repeated == result
    assert all(file_digest(path) == expected for path, expected in before.items())
    assert original == {str(p): file_digest(p) for p in source.rglob('*') if p.is_file()}


def test_routing_cli_pause_resume_and_report_across_processes(tmp_path):
    source, teacher, config = routing_inputs(tmp_path)
    settings = read_json(config)
    settings['holdout']['splits'] = {
        name: dict(families=['local_patch' if name in ('ood_contrast', 'ood_topology')
                            else 'rotated_anisotropy'], per_family=1)
        for name in ('final', 'ood_grid', 'ood_rotation', 'ood_anisotropy', 'ood_contrast', 'ood_topology')}
    write_json(config, settings)
    out = tmp_path / 'cli'
    cli = [sys.executable, str(study.PROJECT / 'scripts/run_v6_7_bamg_energy_cnn_routing_study.py')]
    command = cli + ['run', '--config', str(config), '--source-study', str(source),
                     '--teacher-comparison', str(teacher), '--output', str(out)]
    for args in (['--stop-after', '1'], ['--resume']):
        result = subprocess.run(command + args, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(cli + ['report', '--output', str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'complete=True' in result.stdout and 'global fixed lambda=' in result.stdout
    for args in (['freeze'], ['evaluate-final', '--max-cases', '1'],
                 ['evaluate-final', '--resume'], ['report-final', '--group', 'ood_grid']):
        result = subprocess.run(cli + args + ['--output', str(out)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    assert read_json(out / 'final/report.json')['completed']
    assert 'Frozen holdout ood_grid' in result.stdout


@pytest.mark.parametrize('key,value', [('version', 'old-version'), ('lambdas', [10., 30.]),
    ('routing', dict(stage='supervised')), ('numerical_baselines', ['fixed_C', 'em5_C'])])
def test_development_reference_rejects_changed_routing_protocol(tmp_path, key, value):
    settings = read_json(study.PROJECT / 'configs/v6_7_bamg_energy_cnn_routing_q1.json')
    previous = deepcopy(settings)
    previous[key] = value
    write_json(tmp_path / 'configuration.json', previous)
    with pytest.raises(ValueError, match='protocol differs'):
        study._development_reference(tmp_path, settings, {}, {})


def test_research_configs_have_equal_complete_epochs_and_common_validation():
    pilot = read_json(study.PROJECT / 'configs/v6_7_bamg_energy_cnn_routing_q1.json')
    extended = read_json(study.PROJECT / 'configs/v6_7_bamg_energy_cnn_routing_1120_q1.json')
    assert pilot['data']['validation'] == extended['data']['validation']
    assert pilot['data']['seed'] == extended['data']['seed']
    for settings, count in ((pilot, 560), (extended, 1120)):
        assert routing.validate_routing_settings(settings)['lambdas'] == [10., 30., 50.]
        assert settings['sampling'] == 'epoch_shuffle'
        assert settings['benchmark']['cache_hierarchies'] is False
        assert settings['supervised']['updates'] == 15 * count
        assert settings['finetune']['updates'] == 8 * count
        assert settings['warm_validation']['every'] == count
        assert len(settings['model_seeds']) == 3


def test_same_cohort_expansion_reuses_validation_and_train_prefix(tmp_path):
    source, teacher, config = routing_inputs(tmp_path)
    settings = read_json(config)
    settings['data']['sampling'] = 'stratified_nested_v1'
    settings['data']['train']['per_family'] = 2
    settings['data']['validation']['per_family'] = 2
    settings['supervised']['updates'] = settings['finetune']['updates'] = 2
    settings['finetune']['curriculum_step'] = 1
    settings['warm_validation']['every'] = 2
    write_json(config, settings)
    pilot = tmp_path / 'pilot'
    study.run(config, pilot, source_study=source, teacher_comparison=teacher, stop_after=0)
    larger = deepcopy(settings)
    larger['data']['train']['per_family'] = 4
    larger['supervised']['updates'] = larger['finetune']['updates'] = 4
    larger['finetune']['curriculum_step'] = 2
    larger['warm_validation']['every'] = 4
    extended_config = tmp_path / 'expanded.json'
    write_json(extended_config, larger)
    extended = tmp_path / 'extended'
    study.run(extended_config, extended, source_study=source, teacher_comparison=teacher,
              development_study=pilot, stop_after=0)
    first = read_json(pilot / 'data/development_manifest.json')['splits']
    second = read_json(extended / 'data/development_manifest.json')['splits']
    assert first['validation'] == second['validation']
    assert first['train'] == second['train'][:2]
    assert file_digest(pilot / 'inputs/exclusion_manifest.json') == file_digest(extended / 'inputs/exclusion_manifest.json')
