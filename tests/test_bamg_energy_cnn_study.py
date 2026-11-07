from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from adaptive_mg.config import MGConfig
from adaptive_mg.v67 import bamg_energy_cnn_study as study
from adaptive_mg.v67 import bamg_energy_study as numerical
from adaptive_mg.v67.bamg_study import _file_hash, _model_package, _read, _save, _write
from adaptive_mg.v67.bamg_transfer import WeightCNN
from adaptive_mg.v67.fixed_classical import BilinearClassicalBaseline
from adaptive_mg.v67.research_data import make_research_plan, prepare_development_data


def inputs(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    source = _read(study.PROJECT / 'configs/v6_7_bamg_compact_smoke_q1.json')
    rules = BilinearClassicalBaseline.from_config(SimpleNamespace(mg=MGConfig(**source['mg'])))
    plan = make_research_plan(smoke=True, seed=21273, discretization='q1')
    for name in ('train', 'validation'):
        plan['splits'][name].update(sizes=[7], families=['near_isotropic'], per_family=1, count=1)
    prepare_development_data(root / 'data', rules, plan=plan, historical_roots=[root])
    _write(root / 'fixed_classical.json', rules.to_dict())
    _write(root / 'configuration.json', source)
    torch.manual_seed(71)
    package = _model_package(None, WeightCNN(**source['models']['weight']).double())
    files = {}
    for name in ('supervised_models.pt', 'deployment_uniform.pt', 'deployment_terminal.pt'):
        _save(root / name, package)
        files['checkpoint:' + name] = dict(path='/old/mac/' + name, sha256=_file_hash(root / name))
    for key, name in [('plan', 'data/dataset_plan.json'), ('development', 'data/development_manifest.json'),
                      ('baseline', 'fixed_classical.json')]:
        files[key] = dict(path='/old/mac/' + name, sha256=_file_hash(root / name))
    frozen_config = dict(settings=source,
        selection=dict(selected_learned_model='bootstrap_weight_cnn_finetuned_uniform'))
    _write(root / 'data/research_freeze.json', dict(files=files, config=frozen_config,
        config_digest=numerical.digest(frozen_config)))
    _write(root / 'data/final_data_manifest.json', dict(claim_fingerprint='old-final',
        splits=dict(final=[dict(normalized_operator_digest='a' * 64)]),
        historical_index=dict(normalized_operator_digests=['b' * 64])))
    _write(root / 'data/final_claim.json', dict(fingerprint='old-final',
        freeze_sha256=_file_hash(root / 'data/research_freeze.json'),
        final_data_sha256=_file_hash(root / 'data/final_data_manifest.json')))
    settings = _read(study.PROJECT / 'configs/v6_7_bamg_energy_teacher_q1.json')
    settings.update(smoke=True, lambdas=[0., 10.])
    settings['data'].update(sizes=[7], per_family=1, families=['near_isotropic'])
    settings['teacher'].update(vectors=4, sweeps=[1, 2], selection_vectors=2, audit_vectors=2,
                               eigen_count=2, eigen_refine=1, adaptive_vectors=1,
                               augment_vectors=0, max_vector_retries=0)
    settings['benchmark'].update(repetitions=1, rhs_count=1)
    config = tmp_path / 'teacher.json'
    _write(config, settings)
    teacher = tmp_path / 'teacher'
    numerical.run(config, teacher)
    config = tmp_path / 'cnn.json'
    settings = _read(study.PROJECT / 'configs/v6_7_bamg_energy_cnn_compare_smoke_q1.json')
    settings['data']['families'] = ['near_isotropic']
    _write(config, settings)
    return root, teacher, config


def test_full_shared_label_pipeline_pause_resume_and_disjoint_data(tmp_path):
    source, teacher, config = inputs(tmp_path)
    original = {str(p): _file_hash(p) for p in source.rglob('*') if p.is_file()}
    out = tmp_path / 'comparison'
    result = study.run(config, out, source_study=source, teacher_comparison=teacher, stop_after=1)
    assert not result['completed']
    labels = {str(p): _file_hash(p) for p in (out / 'teacher').rglob('*') if p.is_file()}
    result = study.run(config, out, source_study=source, teacher_comparison=teacher, resume=True)
    assert result['completed'] and result['train_cases'] == result['validation_cases'] == 1
    assert len(result['benchmark']['methods']) == 7
    assert result['runs']['scratch_seed6101']['status'] == 'complete'
    assert result['runs']['warm_seed6101']['status'] == 'complete'
    for name, value in result['seed_summary']['groups'].items():
        assert value['attempts'] == 1 and value['std_warm_seconds'] is None
    identities = [_read(out / 'runs' / name / 'identity.json') for name in result['runs']]
    assert identities[0]['teacher'] == identities[1]['teacher']
    assert identities[0]['train'] == identities[1]['train']
    assert identities[0]['validation'] == identities[1]['validation']
    assert identities[0]['supervised']['seed'] == identities[1]['supervised']['seed']
    manifest = _read(out / 'data/development_manifest.json')
    forbidden = set(manifest['historical_index']['normalized_operator_digests'])
    assert len(forbidden) >= 4  # old TRAIN/validation/final plus lambda selection validation
    assert 'a' * 64 in forbidden
    assert 'b' * 64 in forbidden  # inherited Mac history without the original directories
    assert all(r['normalized_operator_digest'] not in forbidden
               for split in manifest['splits'].values() for r in split)
    assert not (out / 'data/final_data_manifest.json').exists()
    before = {str(p): _file_hash(p) for p in (out / 'validation/benchmark_cases').glob('*.json')}
    study.run(config, out, source_study=source, teacher_comparison=teacher, resume=True)
    assert before == {str(p): _file_hash(p) for p in (out / 'validation/benchmark_cases').glob('*.json')}
    assert labels == {str(p): _file_hash(p) for p in (out / 'teacher').rglob('*') if p.is_file()}
    assert original == {str(p): _file_hash(p) for p in source.rglob('*') if p.is_file()}
    changed = _read(config)
    changed['model_seeds'].append(6102)
    _write(config, changed)
    with pytest.raises(ValueError, match='changed'):
        study.run(config, out, source_study=source, teacher_comparison=teacher, resume=True)


def test_source_migration_and_teacher_selection_fail_closed(tmp_path):
    source, teacher, config = inputs(tmp_path)
    info, _ = study.source_inputs(source)
    assert info['checkpoint'] == 'deployment_uniform.pt'
    assert info['init_weight_sha256'] == _file_hash(source / 'deployment_uniform.pt')
    info, _, _, _ = study.teacher_inputs(teacher)
    assert info['selected_lambda'] in (0., 10.)
    summary = _read(teacher / 'teacher_comparison.json')
    summary['selected_lambda'] = 50.
    _write(teacher / 'teacher_comparison.json', summary)
    with pytest.raises(ValueError, match='disagrees'):
        study.teacher_inputs(teacher)
    with (source / 'deployment_uniform.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='differs'):
        study.source_inputs(source)


def test_resume_and_report_in_fresh_python_processes(tmp_path):
    source, teacher, config = inputs(tmp_path)
    output = tmp_path / 'cli'
    command = [sys.executable, str(study.PROJECT / 'scripts/run_v6_7_bamg_energy_cnn_study.py'),
               'run', '--config', str(config), '--source-study', str(source),
               '--teacher-comparison', str(teacher), '--output', str(output)]
    for extra in (['--stop-after', '1'], ['--resume']):
        result = subprocess.run(command + extra, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(command[:2] + ['report', '--output', str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'complete=True' in result.stdout


def test_expanded_epoch_cohort_shared_validation_and_resume(tmp_path):
    source, teacher, config = inputs(tmp_path)
    settings = _read(config)
    settings['sampling'] = 'epoch_shuffle'
    settings['data']['sampling'] = 'stratified_nested_v1'
    settings['data']['train']['per_family'] = 2
    settings['data']['validation']['per_family'] = 2
    settings['supervised']['updates'] = 6
    settings['finetune'].update(updates=4, curriculum_step=2)
    settings['warm_validation']['every'] = 2
    _write(config, settings)
    smaller, larger = tmp_path / 'small', tmp_path / 'large'
    study.run(config, smaller, source_study=source, teacher_comparison=teacher, stop_after=1)
    expanded = deepcopy(settings)
    expanded['data']['train']['per_family'] = 4
    expanded['supervised']['updates'] = 12
    expanded['finetune'].update(updates=8, curriculum_step=4)
    expanded['warm_validation']['every'] = 4
    large_config = tmp_path / 'large.json'
    _write(large_config, expanded)
    kwargs = dict(source_study=source, teacher_comparison=teacher, development_study=smaller)
    result = study.run(large_config, larger, stop_after=3, **kwargs)
    assert not result['completed']
    a = _read(smaller / 'data/development_manifest.json')['splits']
    b = _read(larger / 'data/development_manifest.json')['splits']
    assert a['validation'] == b['validation']
    assert a['train'] == b['train'][:2]
    assert len(b['train']) == 4
    assert _file_hash(smaller / 'inputs/exclusion_manifest.json') == _file_hash(larger / 'inputs/exclusion_manifest.json')
    result = study.run(large_config, larger, resume=True, **kwargs)
    assert result['completed']
    for name in ('scratch_seed6101', 'warm_seed6101'):
        history = _read(larger / 'runs' / name / 'pretrain_history.json')
        assert len(history) == 12
        for start in range(0, 12, 4):
            assert sorted(r['case'] for r in history[start:start + 4]) == list(range(4))
    metrics = _read(larger / 'runtime_metrics.json')
    assert metrics['phases']['prepare_data']['process_peak_rss_mib'] > 0
    assert metrics['phases']['prepare_data']['invocations'] == 2
    _, source_settings = study.source_inputs(source)
    ti, ts, cfg, tc = study.teacher_inputs(teacher)
    resolved = study.resolve_settings(expanded, source_settings, ts, cfg, tc)
    si, _ = study.source_inputs(source)
    bad = deepcopy(resolved)
    bad['supervised']['learning_rate'] *= 2
    with pytest.raises(ValueError, match='protocol differs'):
        study._development_reference(smaller, bad, si, ti)
    bad = deepcopy(resolved)
    bad['supervised']['updates'] += 4
    with pytest.raises(ValueError, match='equal training epochs'):
        study._development_reference(smaller, bad, si, ti)
    (smaller / 'final').mkdir()
    _write(smaller / 'final/claim.json', {})
    with pytest.raises(ValueError, match='before opening final'):
        study._development_reference(smaller, resolved, si, ti)


def test_research_configs_have_matched_epochs_and_full_grid_validation():
    for suffix, count in [('q1', 560), ('1120_q1', 1120)]:
        settings = _read(study.PROJECT / f'configs/v6_7_bamg_energy_cnn_compare_{suffix}.json')
        assert settings['data']['sampling'] == 'stratified_nested_v1'
        assert settings['sampling'] == 'epoch_shuffle'
        assert settings['supervised']['updates'] == 15 * count
        assert settings['finetune']['updates'] == 8 * count
        assert settings['finetune']['curriculum_step'] == 4 * count
        assert settings['data']['validation'] == dict(sizes=[15, 31, 63, 127], per_family=10)
        assert settings['warm_validation']['per_family_per_size'] == 2
