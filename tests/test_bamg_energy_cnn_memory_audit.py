"""Audit memory-only execution without rebinding saved scientific identities."""
import importlib.util
from pathlib import Path

import pytest

from adaptive_mg.v67 import bamg_energy_cnn_study as paired
from adaptive_mg.v67 import bamg_energy_cnn_routing_final as final
from adaptive_mg.v67.bamg_energy_study import read_json, write_json


_spec = importlib.util.spec_from_file_location('memory_audit_under_test',
    paired.PROJECT / 'scripts/bamg_energy_cnn_memory_audit.py')
audit_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_module)


@pytest.fixture
def audit_inputs(tmp_path):
    sources = []
    for name in ('runner.py', 'memory.py', 'audit.py'):
        path = tmp_path / name
        path.write_text('# ' + name + '\n')
        sources.append(path)
    out = tmp_path / 'study'
    out.mkdir()
    write_json(out / 'study_manifest.json', dict(original=True, identity='unchanged'))
    return out, sources


def activate(out, sources, monkeypatch):
    monkeypatch.setattr(paired, '_prepare', lambda *args, **kwargs: 'prepared')
    with audit_module.execution_audit(out, 'run', sources) as audit:
        assert paired._prepare(out) == 'prepared'
        return audit


def test_existing_study_identity_and_sources_remain_unchanged(audit_inputs, monkeypatch):
    out, sources = audit_inputs
    manifest_before = (out / 'study_manifest.json').read_bytes()
    sources_before = {path: path.read_bytes() for path in sources}
    audit = activate(out, sources, monkeypatch)
    before = (out / audit_module.RECORD).read_bytes()
    assert audit.ensure() == audit.verify()
    assert (out / audit_module.RECORD).read_bytes() == before
    assert (out / 'study_manifest.json').read_bytes() == manifest_before
    assert all(path.read_bytes() == content for path, content in sources_before.items())
    assert len(audit.frozen_files()) == 4


def test_new_study_keeps_directory_empty_until_original_prepare(tmp_path, monkeypatch):
    source = tmp_path / 'runner.py'
    source.write_text('pass\n')
    out = tmp_path / 'new'
    monkeypatch.setattr(paired, '_prepare', lambda *args, **kwargs: None)
    with audit_module.execution_audit(out, 'run', [source]):
        assert not out.exists()
        out.mkdir()
        write_json(out / 'study_manifest.json', dict(original=True))
        paired._prepare(out)
        assert (out / audit_module.RECORD).is_file()


@pytest.mark.parametrize('changed', ['adapter', 'record', 'snapshot', 'manifest'])
def test_audit_rejects_changed_execution_evidence(audit_inputs, monkeypatch, changed):
    out, sources = audit_inputs
    audit = activate(out, sources, monkeypatch)
    path = {'adapter': sources[0], 'record': out / audit_module.RECORD,
            'snapshot': out / audit_module.SNAPSHOTS / sources[0].name,
            'manifest': out / 'study_manifest.json'}[changed]
    if changed == 'record':
        value = read_json(path)
        value['original_study_identity_preserved'] = False
        write_json(path, value)
    else:
        with path.open('a') as stream:
            stream.write('\n')
    with pytest.raises(ValueError, match='changed'):
        with audit_module.execution_audit(out, 'report', sources):
            pass


def test_freeze_atomically_binds_execution_and_verifies_coverage(audit_inputs, monkeypatch):
    out, sources = audit_inputs
    audit = activate(out, sources, monkeypatch)
    original_manifest = (out / 'study_manifest.json').read_bytes()
    monkeypatch.setattr(final, '_verify_freeze',
        lambda output: ({}, read_json(Path(output) / 'research_freeze.json'), {}, {}))
    frozen = dict(files={'study_manifest.json': audit_module._file_hash(out / 'study_manifest.json')})
    with audit_module.execution_audit(out, 'freeze', sources):
        final.write_json(out / 'research_freeze.json', frozen)
        assert final._verify_freeze(out)[1] == frozen
    assert read_json(out / 'research_freeze.json') == frozen
    assert all(frozen['files'][name] == expected for name, expected in audit.frozen_files().items())
    assert (out / 'study_manifest.json').read_bytes() == original_manifest
    frozen['files'].pop(audit_module.RECORD)
    write_json(out / 'research_freeze.json', frozen)
    with audit_module.execution_audit(out, 'report-final', sources):
        with pytest.raises(ValueError, match='does not bind'):
            final._verify_freeze(out)


def test_cannot_add_to_or_train_frozen_study_and_reports_are_read_only(audit_inputs):
    out, sources = audit_inputs
    write_json(out / 'research_freeze.json', dict(files={}))
    before = {str(path): path.read_bytes() for path in out.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='frozen'):
        with audit_module.execution_audit(out, 'run', sources):
            pass
    with pytest.raises(ValueError, match='missing'):
        with audit_module.execution_audit(out, 'freeze', sources):
            pass
    with audit_module.execution_audit(out, 'report', sources):
        pass
    assert before == {str(path): path.read_bytes() for path in out.rglob('*') if path.is_file()}


def test_context_restores_patches_on_error_and_cannot_rewrite_claimed_freeze(audit_inputs, monkeypatch):
    out, sources = audit_inputs
    activate(out, sources, monkeypatch)
    original = paired._prepare, final.write_json, final._verify_freeze
    write_json(out / 'final/claim.json', dict(claimed=True))
    with pytest.raises(ValueError, match='after final'):
        with audit_module.execution_audit(out, 'evaluate-final', sources):
            final.write_json(out / 'research_freeze.json', dict(files={}))
    assert (paired._prepare, final.write_json, final._verify_freeze) == original
    assert not (out / 'research_freeze.json').exists()
