"""Recover a partially saved legacy study without changing its scientific identity."""
import errno
import subprocess
import sys

import pytest

from adaptive_mg.v67 import bamg_energy_cnn_study as paired
from adaptive_mg.v67 import bamg_energy_cnn_routing_study as routing
from adaptive_mg.v67.bamg_study import _file_hash, _read, _write
from test_bamg_energy_cnn_routing_study import routing_inputs


def test_low_memory_cli_resumes_old_partial_cache_and_freezes_adapter(tmp_path, monkeypatch):
    source, teacher, config = routing_inputs(tmp_path)
    settings = _read(config)
    settings['data']['train']['per_family'] = 2
    settings['supervised']['updates'] = settings['finetune']['updates'] = 2
    settings['finetune']['curriculum_step'] = 1
    settings['warm_validation']['every'] = 1
    settings['holdout']['splits'] = {
        name: dict(families=['local_patch' if name in ('ood_contrast', 'ood_topology')
                            else 'rotated_anisotropy'], per_family=1)
        for name in ('final', 'ood_grid', 'ood_rotation', 'ood_anisotropy', 'ood_contrast', 'ood_topology')}
    _write(config, settings)
    output = tmp_path / 'study'
    original_builder = paired.build_energy_teacher
    calls = 0

    def fail_after_one_saved_case(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.ENOMEM, 'simulated exhausted memory')
        return original_builder(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(paired, 'build_energy_teacher', fail_after_one_saved_case)
        with pytest.raises(OSError, match='simulated exhausted memory'):
            routing.run(config, output, source_study=source, teacher_comparison=teacher)

    old_manifest = _file_hash(output / 'study_manifest.json')
    prepared = _file_hash(output / 'prepared_manifest.json')
    old_labels = {str(path): _file_hash(path) for path in output.glob('labels/*/teacher/train/*.pt')}
    assert len(old_labels) == 1
    cli = [sys.executable, str(routing.PROJECT / 'scripts/run_v6_7_bamg_energy_cnn_routing_low_memory.py')]
    arguments = ['run', '--config', str(config), '--source-study', str(source),
                 '--teacher-comparison', str(teacher), '--output', str(output), '--resume']
    paused = subprocess.run(cli + arguments + ['--stop-after', '1'], capture_output=True, text=True, timeout=180)
    assert paused.returncode == 0, paused.stdout + paused.stderr
    assert not _read(output / 'progress.json')['completed']
    assert _file_hash(output / 'study_manifest.json') == old_manifest
    assert _file_hash(output / 'prepared_manifest.json') == prepared
    for path, expected in old_labels.items():
        assert _file_hash(path) == expected
    completed = subprocess.run(cli + arguments, capture_output=True, text=True, timeout=240)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _read(output / 'comparison.json')['completed']
    assert _file_hash(output / 'study_manifest.json') == old_manifest
    frozen = subprocess.run(cli + ['freeze', '--output', str(output)], capture_output=True, text=True, timeout=120)
    assert frozen.returncode == 0, frozen.stdout + frozen.stderr
    record = _read(output / 'research_freeze.json')
    assert 'memory_execution.json' in record['files']
    adapter_files = [name for name in record['files'] if name.endswith('.py')]
    assert len(adapter_files) == 3
    # Original final verifier also checks the adapter snapshots through files.
    from adaptive_mg.v67 import bamg_energy_cnn_routing_final as final
    final._verify_freeze(output)
    changed = output / adapter_files[0]
    changed.write_text(changed.read_text() + '\n# changed adapter\n')
    with pytest.raises(ValueError, match='changed|mismatch'):
        final._verify_freeze(output)
