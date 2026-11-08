"""Bind the outside-src memory adapter without rewriting an existing study.

The original study identity, cached labels and checkpoints remain unchanged.
An immutable execution record pins the adapter sources. The original routing
freeze includes that record and source snapshots before its atomic publication.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


VERSION = 'bamg-energy-cnn-memory-execution-v1'
RECORD = 'memory_execution.json'
SNAPSHOTS = 'inputs/memory_execution'


def _file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            result.update(block)
    return result.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class ExecutionAudit:
    def __init__(self, output, adapter_paths):
        self.output = Path(output).resolve()
        paths = [Path(path).resolve() for path in adapter_paths]
        if not paths or len({path.name for path in paths}) != len(paths):
            raise ValueError('Memory execution needs distinct adapter source filenames')
        self.paths = {path.name: path for path in paths}
        if any(not path.is_file() for path in paths):
            raise ValueError('Memory execution adapter source is missing')

    def _expected(self):
        manifest = self.output / 'study_manifest.json'
        if not manifest.is_file():
            raise ValueError('Memory execution requires a validated study manifest')
        value = dict(version=VERSION, study_manifest_sha256=_file_hash(manifest),
            adapters={name: dict(sha256=_file_hash(path),
                       snapshot=SNAPSHOTS + '/' + name)
                      for name, path in sorted(self.paths.items())},
            scope='Storage and materialization only; objective, labels, case order and RNG unchanged',
            original_study_identity_preserved=True)
        return {**value, 'record_digest': _digest(value)}

    def exists(self):
        return (self.output / RECORD).is_file()

    def verify(self):
        path = self.output / RECORD
        if not path.is_file():
            raise ValueError('Memory execution record is missing; use the original runner for this study')
        record = json.loads(path.read_text())
        if record != self._expected():
            raise ValueError('Memory execution adapter, record or study manifest changed')
        for entry in record['adapters'].values():
            snapshot = self.output / entry['snapshot']
            if not snapshot.is_file() or _file_hash(snapshot) != entry['sha256']:
                raise ValueError('Memory execution source snapshot changed or disappeared')
        return record

    def ensure(self):
        """Called only after the original runner acquires its study lock."""
        if self.exists():
            return self.verify()
        if ((self.output / 'research_freeze.json').exists()
                or (self.output / 'final/claim.json').exists()):
            raise ValueError('Cannot add memory execution to a frozen or final-claimed study')
        expected = self._expected()
        directory = self.output / SNAPSHOTS
        directory.mkdir(parents=True, exist_ok=True)
        for name, source in sorted(self.paths.items()):
            entry = expected['adapters'][name]
            target = self.output / entry['snapshot']
            if target.exists():
                if _file_hash(target) != entry['sha256']:
                    raise ValueError('Unpublished memory execution source snapshot changed')
                continue
            fd, temporary = tempfile.mkstemp(prefix='.' + name + '-', dir=directory)
            try:
                with os.fdopen(fd, 'wb') as destination, source.open('rb') as stream:
                    shutil.copyfileobj(stream, destination, length=65536)
                    destination.flush()
                    os.fsync(destination.fileno())
                if _file_hash(temporary) != entry['sha256']:
                    raise ValueError('Memory execution source changed while copying')
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        # All source snapshots are durable before publishing their record.
        from adaptive_mg.v67.bamg_energy_study import write_json
        write_json(self.output / RECORD, expected)
        return self.verify()

    def frozen_files(self):
        record = self.verify()
        return {RECORD: _file_hash(self.output / RECORD),
                **{entry['snapshot']: entry['sha256']
                   for entry in record['adapters'].values()}}

    def verify_freeze_binding(self, frozen):
        required = self.frozen_files()
        if any(frozen.get('files', {}).get(name) != expected
               for name, expected in required.items()):
            raise ValueError('Final freeze does not bind the memory execution record and sources')


@contextmanager
def execution_audit(output, command, adapter_paths):
    """Audit one CLI invocation; enter after source_reader() and memory_mode()."""
    from adaptive_mg.v67 import bamg_energy_cnn_study as paired
    from adaptive_mg.v67 import bamg_energy_cnn_routing_final as final

    if command not in ('run', 'report', 'freeze', 'evaluate-final', 'report-final'):
        raise ValueError('Unknown memory execution command')
    audit = ExecutionAudit(output, adapter_paths)
    if audit.exists():
        audit.verify()
    if command == 'run' and ((audit.output / 'research_freeze.json').exists()
                            or (audit.output / 'final/claim.json').exists()):
        raise ValueError('Study is frozen; memory-mode training is closed')
    if command in ('freeze', 'evaluate-final') and not audit.exists():
        raise ValueError('Memory execution record is missing; use the original runner for this study')

    original_prepare = paired._prepare
    original_write = final.write_json
    original_verify = final._verify_freeze

    def prepare(out, *args, **kwargs):
        if Path(out).resolve() != audit.output:
            raise ValueError('Memory execution output changed')
        audit.ensure()
        return original_prepare(out, *args, **kwargs)

    def write(path, value):
        if Path(path).resolve() == audit.output / 'research_freeze.json':
            if (audit.output / 'final/claim.json').exists():
                raise ValueError('Cannot alter a freeze after final evaluation is claimed')
            required = audit.frozen_files()
            files = value['files']
            if any(name in files and files[name] != expected
                   for name, expected in required.items()):
                raise ValueError('Conflicting memory execution freeze entry')
            files.update(required)
        return original_write(path, value)

    def verify(out):
        result = original_verify(out)
        if Path(out).resolve() == audit.output and audit.exists():
            audit.verify_freeze_binding(result[1])
        return result

    paired._prepare, final.write_json, final._verify_freeze = prepare, write, verify
    try:
        yield audit
    finally:
        paired._prepare = original_prepare
        final.write_json = original_write
        final._verify_freeze = original_verify
