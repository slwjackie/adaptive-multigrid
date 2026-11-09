"""End-to-end opt-in studies, including legacy baseline, resume and final/OOD."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from adaptive_mg.v67 import bamg_energy_cnn_routing_study as study
from adaptive_mg.v67.bamg_energy_study import file_digest, read_json, write_json
from adaptive_mg.v67.bamg_energy_cnn_source_compat import source_reader
from bamg_energy_cnn_memory import memory_mode
from bamg_feature9_adapter import feature9_mode
from test_bamg_energy_cnn_routing_study import routing_inputs

CLI = study.PROJECT / 'scripts/run_v6_7_bamg_energy_cnn_feature9.py'


@pytest.mark.parametrize('mode', ['row9', 'symmetric9'])
def test_cli_train_resume_baselines_freeze_final_and_ood(tmp_path, mode):
    source, teacher, config = routing_inputs(tmp_path)
    original = {str(p): file_digest(p) for root in (source, teacher)
                for p in root.rglob('*') if p.is_file()}
    settings = read_json(config)
    settings['models']['weight']['feature_mode'] = mode
    settings['holdout']['splits'] = {
        name: dict(families=['local_patch' if name in ('ood_contrast', 'ood_topology')
                            else 'rotated_anisotropy'], per_family=1)
        for name in ('final', 'ood_grid', 'ood_rotation', 'ood_anisotropy', 'ood_contrast', 'ood_topology')}
    write_json(config, settings)
    out = tmp_path / mode
    cli = [sys.executable, str(CLI)]
    command = cli + ['run', '--config', str(config), '--source-study', str(source),
                     '--teacher-comparison', str(teacher), '--output', str(out)]
    for args in (['--stop-after', '1'], ['--resume']):
        result = subprocess.run(command + args, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    report = read_json(out / 'comparison.json')
    assert report['completed'] and len(report['runs']) == 6
    assert set(study.BASELINES) <= set(report['benchmark']['methods'])
    # In particular, the legacy old_cnn has not failed an architecture check.
    cases = list((out / 'validation/benchmark_cases').glob('*.json'))
    assert cases
    for path in cases:
        setups = read_json(path)['setups']
        assert setups['old_cnn']['success'], setups['old_cnn']
        assert all(item['success'] for item in setups.values()), setups
    for run in report['runs']:
        package = torch.load(out / 'runs' / run / 'deployment.pt', weights_only=True)
        assert package['weight_spec']['feature_mode'] == mode
        assert package['weight_state']['backbone.0.weight'].shape[1] == 28
        if 'warm' in run:
            assert read_json(out / 'runs' / run / 'initialization.json')['function_preserving'] is False
    observed = {str(p): file_digest(p) for p in cases}
    repeated = subprocess.run(command + ['--resume'], capture_output=True, text=True, timeout=180)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert observed == {str(p): file_digest(p) for p in cases}
    for args in (['report'], ['freeze'], ['evaluate-final', '--max-cases', '1'],
                 ['evaluate-final', '--resume'], ['report-final', '--group', 'ood_grid']):
        result = subprocess.run(cli + args + ['--output', str(out)],
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    assert read_json(out / 'final/report.json')['completed']
    assert 'Frozen holdout ood_grid' in result.stdout
    assert original == {str(p): file_digest(p) for root in (source, teacher)
                        for p in root.rglob('*') if p.is_file()}
    settings['models']['weight']['feature_mode'] = 'row9' if mode == 'symmetric9' else 'symmetric9'
    write_json(config, settings)
    result = subprocess.run(command + ['--resume'], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'Different feature mode' in result.stderr
    # Execution-source identity is also bound into the frozen study.
    manifest = read_json(out / 'study_manifest.json')
    assert 'bamg_feature9_adapter.py' in str(manifest)


def test_features_can_share_legacy_development_cohort_without_changing_budget(tmp_path):
    source, teacher, config = routing_inputs(tmp_path)
    settings = read_json(config)
    settings['data']['sampling'] = 'stratified_nested_v1'
    write_json(config, settings)
    baseline = tmp_path / 'baseline'
    study.run(config, baseline, source_study=source, teacher_comparison=teacher, stop_after=0)
    original = {str(p): file_digest(p) for p in baseline.rglob('*') if p.is_file()}
    for mode in ('row9', 'symmetric9'):
        settings['models']['weight']['feature_mode'] = mode
        write_json(config, settings)
        out = tmp_path / mode
        with source_reader(), memory_mode(), feature9_mode():
            study.run(config, out, source_study=source, teacher_comparison=teacher,
                      development_study=baseline, stop_after=0)
        assert read_json(out / 'data/development_manifest.json')['splits'] == read_json(
            baseline / 'data/development_manifest.json')['splits']
    assert original == {str(p): file_digest(p) for p in baseline.rglob('*') if p.is_file()}
    bad = deepcopy(settings)
    bad['finetune']['learning_rate'] *= 2
    write_json(config, bad)
    with source_reader(), memory_mode(), feature9_mode(), pytest.raises(ValueError, match='protocol differs'):
        study.run(config, tmp_path / 'bad', source_study=source, teacher_comparison=teacher,
                  development_study=baseline, stop_after=0)


def test_configs_only_change_feature_mode_and_legacy_outputs_are_rejected(tmp_path):
    for mode in ('row9', 'symmetric9'):
        for suffix in ('q1', '1120_q1', 'smoke_q1'):
            old = read_json(study.PROJECT / f'configs/v6_7_bamg_energy_cnn_routing_{suffix}.json')
            new = read_json(study.PROJECT / f'configs/v6_7_bamg_energy_cnn_feature9_{mode}_{suffix}.json')
            assert new['models']['weight'].pop('feature_mode') == mode
            assert old == new
    write_json(tmp_path / 'configuration.json', old)
    result = subprocess.run([sys.executable, str(CLI), 'report', '--output', str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'feature_mode must' in result.stderr
