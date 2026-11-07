import json
from pathlib import Path
import subprocess
import sys

import pytest

from adaptive_mg.v67 import bamg_energy_cnn_final as final
from adaptive_mg.v67 import bamg_energy_cnn_study as study
from adaptive_mg.v67.bamg_study import _file_hash, _read, _write
from test_bamg_energy_cnn_study import inputs


def ready(tmp_path):
    source, teacher, config = inputs(tmp_path)
    settings = _read(config)
    settings['holdout']['splits'] = {
        name: dict(families=['local_patch' if name in ('ood_contrast', 'ood_topology')
                            else 'rotated_anisotropy'], per_family=1)
        for name in ('final', 'ood_grid', 'ood_rotation', 'ood_anisotropy', 'ood_contrast', 'ood_topology')}
    _write(config, settings)
    out = tmp_path / 'comparison'
    study.run(config, out, source_study=source, teacher_comparison=teacher)
    return out, source, teacher, config


def test_frozen_full_workflow_resume_groups_and_no_holdout_selection(tmp_path, monkeypatch):
    out, source, teacher, config = ready(tmp_path)
    before = {str(p): _file_hash(p) for p in source.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='freeze'):
        final.evaluate_final(out)
    assert not (out / 'final').exists()
    frozen = final.freeze(out)
    assert not frozen['final_operators_materialized'] and not (out / 'final').exists()
    assert 'a' * 64 in frozen['forbidden_operator_digests']
    assert 'b' * 64 in frozen['forbidden_operator_digests']
    assert frozen['selection']['selected_lambda'] == _read(out / 'comparison.json')['selected_lambda']
    with pytest.raises(ValueError, match='frozen'):
        study.run(config, out, source_study=source, teacher_comparison=teacher, resume=True)
    assert final.freeze(out) == frozen
    first = final.evaluate_final(out, max_cases=1)
    assert not first['completed'] and first['benchmark']['cases_completed'] == 1
    first_shard = next((out / 'final/benchmark/benchmark_cases').glob('*.json'))
    first_hash = _file_hash(first_shard)
    result = final.evaluate_final(out, resume=True)
    assert result['completed'] and result['benchmark']['cases_expected'] == 6
    assert _file_hash(first_shard) == first_hash
    assert result['benchmark']['selected_method'] is None
    assert result['holdout_selection_permitted'] is False
    assert result['selection'] == frozen['selection']
    assert result['evaluation_scope'] == 'smoke_only'
    state = _read(out / 'final/data/data_manifest.json')
    assert state['completed'] and len(state['cases']) == 6
    assert len({v['normalized_operator_digest'] for v in state['cases'].values()}) == 6
    assert set(frozen['forbidden_operator_digests']).isdisjoint(v['normalized_operator_digest'] for v in state['cases'].values())
    monkeypatch.setattr(final.heldout_benchmark, 'benchmark', lambda *a, **k: pytest.fail('completed evaluation retimed'))
    assert final.evaluate_final(out, resume=True) == result
    with pytest.raises(FileExistsError, match='claimed'):
        final.evaluate_final(out)
    assert before == {str(p): _file_hash(p) for p in source.rglob('*') if p.is_file()}
    final.report_final(out, group='final')
    saved = first_shard.read_bytes()
    first_shard.write_bytes(saved + b' ')
    with pytest.raises(ValueError, match='changed'):
        final.report_final(out)


def test_freeze_rejects_stale_validation_and_final_rejects_changed_checkpoint(tmp_path):
    out, source, teacher, config = ready(tmp_path)
    path = out / 'comparison.json'
    original = _read(path)
    _write(path, {**original, 'selected_lambda': 999})
    with pytest.raises(ValueError, match='inputs'):
        final.freeze(out)
    _write(path, original)
    manifest_path = out / 'validation/benchmark_manifest.json'
    manifest = _read(manifest_path)
    changed = {**manifest, 'identity': {**manifest['identity'], 'study': 'f' * 64}}
    _write(manifest_path, changed)
    with pytest.raises(ValueError):
        final.freeze(out)
    _write(manifest_path, manifest)
    final.freeze(out)
    checkpoint = out / 'runs/scratch_seed6101/deployment.pt'
    with checkpoint.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        final.evaluate_final(out)


def test_final_resume_rejects_missing_data_even_if_benchmark_case_complete(tmp_path):
    out, *_ = ready(tmp_path)
    final.freeze(out)
    final.evaluate_final(out, max_cases=1)
    path = out / 'final/data/data_manifest.json'
    state = _read(path)
    key = next(iter(state['cases']))
    (out / 'final/data/cases' / (key + '.json')).unlink()
    del state['cases'][key]
    _write(path, state)
    with pytest.raises(ValueError, match='saved data case'):
        final.evaluate_final(out, resume=True)


def test_final_cli_fresh_process_pause_resume_and_report(tmp_path):
    out, *_ = ready(tmp_path)
    command = [sys.executable, str(study.PROJECT / 'scripts/run_v6_7_bamg_energy_cnn_study.py')]
    for args in (['freeze'], ['evaluate-final', '--max-cases', '1'],
                 ['evaluate-final', '--resume'], ['report-final', '--group', 'ood_grid']):
        result = subprocess.run(command + args + ['--output', str(out)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    assert 'Frozen holdout ood_grid' in result.stdout


def test_research_holdout_rejects_grid_seen_by_warm_model_and_untrained_final_family():
    requested = _read(study.PROJECT / 'configs/v6_7_bamg_energy_cnn_compare_q1.json')
    source = _read(study.PROJECT / 'configs/v6_7_bamg_compact_research_q1.json')
    teacher = _read(study.PROJECT / 'configs/v6_7_bamg_energy_teacher_q1.json')
    cfg, tc = study.numerical.validate(teacher)
    requested['data']['train']['sizes'] = [15, 31]
    requested['data']['validation']['sizes'] = [15, 31]
    with pytest.raises(ValueError, match='warm-source'):
        study.resolve_settings(requested, source, teacher, cfg, tc)
    requested['holdout']['splits'] = dict(ood_grid=dict(sizes=[255]))
    assert study.resolve_settings(requested, source, teacher, cfg, tc)['holdout'] == requested['holdout']
    requested['data']['families'] = ['near_isotropic']
    with pytest.raises(ValueError, match='families'):
        study.resolve_settings(requested, source, teacher, cfg, tc)
