"""Migrate authentic source freezes without changing completed teacher evidence."""
import subprocess
import sys

import pytest

from adaptive_mg.v67 import bamg_energy_cnn_study as paired
from adaptive_mg.v67 import bamg_energy_cnn_source_compat as compat
from adaptive_mg.v67 import bamg_energy_cnn_routing_study as routing
from adaptive_mg.v67 import bamg_energy_study as numerical
from adaptive_mg.v67 import research_data
from adaptive_mg.v67.bamg_study import _file_hash, _read, _write
from test_bamg_energy_cnn_study import inputs


COMPAT_SOURCE = 'src/adaptive_mg/v67/bamg_energy_cnn_source_compat.py'


def snapshot(root):
    return {str(path.relative_to(root)): _file_hash(path)
            for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def canonical_inputs(tmp_path, monkeypatch):
    # Reproduce a comparison completed before the additive migration module
    # existed. Keep every old source hash rather than weakening verification.
    original_manifest = numerical.manifest

    def prior_manifest(settings):
        record = original_manifest(settings)
        record['sources'].pop(COMPAT_SOURCE, None)
        return record

    with monkeypatch.context() as patch:
        patch.setattr(numerical, 'manifest', prior_manifest)
        source, teacher, config = inputs(tmp_path)
    freeze_path = source / 'data/research_freeze.json'
    frozen = _read(freeze_path)
    frozen['version'] = research_data.VERSION
    frozen['config_digest'] = research_data._hash(frozen['config'])
    _write(freeze_path, frozen)
    claim_path = source / 'data/final_claim.json'
    claim = _read(claim_path)
    claim['freeze_sha256'] = _file_hash(freeze_path)
    _write(claim_path, claim)
    return source, teacher, config


def test_producer_digest_loads_without_changing_source_or_teacher(canonical_inputs):
    source, teacher, _ = canonical_inputs
    source_before, teacher_before = snapshot(source), snapshot(teacher)
    frozen = _read(source / 'data/research_freeze.json')
    assert frozen['config_digest'] != numerical.digest(frozen['config'])
    with pytest.raises(ValueError, match='Source freeze configuration hash changed'):
        paired.source_inputs(source)
    info, settings = compat.source_inputs(source)
    assert info['checkpoint'] == 'deployment_uniform.pt'
    assert info['init_weight_sha256'] == _file_hash(source / 'deployment_uniform.pt')
    assert settings == frozen['config']['settings']
    manifest = _read(teacher / 'comparison_manifest.json')
    assert COMPAT_SOURCE not in manifest['sources']
    old_reader_path = 'src/adaptive_mg/v67/bamg_energy_cnn_study.py'
    assert manifest['sources'][old_reader_path] == _file_hash(paired.PROJECT / old_reader_path)
    teacher_info, _, _, _ = paired.teacher_inputs(teacher)
    assert teacher_info['selected_lambda'] in (0., 10.)
    assert snapshot(source) == source_before
    assert snapshot(teacher) == teacher_before


def test_legacy_spaced_digest_remains_readable(tmp_path):
    source, _, _ = inputs(tmp_path)
    before = snapshot(source)
    old_info, old_settings = paired.source_inputs(source)
    new_info, new_settings = compat.source_inputs(source)
    assert (new_info, new_settings) == (old_info, old_settings)
    assert snapshot(source) == before


@pytest.mark.parametrize('target,match', [
    ('config', 'Source freeze configuration hash changed'),
    ('checkpoint', 'Source checkpoint differs'),
    ('metadata', 'Source metadata differs'),
    ('claim', 'Old final claim does not match'),
    ('final_metadata', 'Old final metadata hash changed'),
])
def test_source_tampering_still_rejected(canonical_inputs, target, match):
    source, _, _ = canonical_inputs
    if target == 'config':
        path = source / 'data/research_freeze.json'
        frozen = _read(path)
        frozen['config']['settings']['seed'] += 1
        _write(path, frozen)
    elif target == 'checkpoint':
        with (source / 'deployment_uniform.pt').open('ab') as stream:
            stream.write(b'changed checkpoint')
    elif target == 'metadata':
        with (source / 'data/dataset_plan.json').open('a') as stream:
            stream.write('\n')
    elif target == 'claim':
        path = source / 'data/final_claim.json'
        claim = _read(path)
        claim['freeze_sha256'] = '0' * 64
        _write(path, claim)
    else:
        path = source / 'data/final_data_manifest.json'
        final = _read(path)
        final['changed'] = True
        _write(path, final)
    before = snapshot(source)
    with pytest.raises(ValueError, match=match):
        compat.source_inputs(source)
    assert snapshot(source) == before


def test_reader_context_restores_on_normal_exit_and_exception():
    original_reader, original_digest = paired.source_inputs, numerical.digest
    with compat.source_reader():
        assert paired.source_inputs is compat.source_inputs
        assert numerical.digest is original_digest
    assert paired.source_inputs is original_reader
    with pytest.raises(RuntimeError, match='expected test exception'):
        with compat.source_reader():
            assert paired.source_inputs is compat.source_inputs
            raise RuntimeError('expected test exception')
    assert paired.source_inputs is original_reader
    assert numerical.digest is original_digest


def test_routing_cli_accepts_canonical_source_and_prior_teacher(canonical_inputs, tmp_path):
    source, teacher, config = canonical_inputs
    settings = _read(config)
    settings.update(version=routing.VERSION, sampling='epoch_shuffle', lambdas=[10, 30, 50],
        numerical_baselines=['fixed_C', 'em5_C', 'bootstrap_C'],
        routing=dict(stage='finetuned', unseen_grid_policy='lower_anchor_clamp'))
    settings['benchmark']['cache_hierarchies'] = False
    _write(config, settings)
    source_before, teacher_before = snapshot(source), snapshot(teacher)
    output = tmp_path / 'routing_cli'
    command = [sys.executable,
        str(paired.PROJECT / 'scripts/run_v6_7_bamg_energy_cnn_routing_study.py'),
        'run', '--config', str(config), '--source-study', str(source),
        '--teacher-comparison', str(teacher), '--output', str(output), '--stop-after', '0']
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not _read(output / 'progress.json')['completed']
    assert (output / 'study_manifest.json').is_file()
    assert snapshot(source) == source_before
    assert snapshot(teacher) == teacher_before
