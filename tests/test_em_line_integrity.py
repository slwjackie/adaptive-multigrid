"""Commit/freeze guards for the staged study without expensive solver work."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

import adaptive_mg.v67.em_line_study as study


PROTOCOL = dict(repeats=1, warmups=0, rhs_counts=[1], regimes=['multiple'])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup_freeze(tmp_path, monkeypatch):
    out = tmp_path/'run'
    out.mkdir()
    cfg = SimpleNamespace(to_dict=lambda: {'same_tolerance': True})
    monkeypatch.setattr(study, 'load', lambda _: (out, {'target_rhs_count': 1}, cfg, None))
    arms = {'fixed_C': {'model': None}, 'em5_C': {'model': None}}
    for name in study.STAGES:
        arms[name] = {'model': SimpleNamespace(signature=lambda n=name: f'{n}-signature')}
        write(out/'checkpoints'/name/'candidate.pt', {'name': name})
        write(out/'checkpoints'/name/'status.json', {'status': 'complete'})
        write(out/'checkpoints'/name/'training_manifest.json', {'stage': name})
    monkeypatch.setattr(study, '_arms', lambda *args: arms)
    monkeypatch.setattr(study, '_sources', lambda: [])
    target = out/'benchmarks/validation'
    selection = dict(recommended_deployment_arm='joint_line')
    files = {}
    for key in ('raw_results.json', 'cycle_diagnostics.json', 'raw_contractions.json', 'selection.json'):
        path = target/key
        write(path, selection if key == 'selection.json' else {})
        files[key] = dict(path=str(path), sha256=study.digest_file(path))
    write(out/'benchmark_evidence.json', dict(files=files,
          selection=selection, checkpoints={name: f'{name}-signature' for name in study.STAGES}))
    exported = out/'selected/candidate.pt'
    exported.parent.mkdir()
    exported.write_bytes((out/'checkpoints/joint_line/candidate.pt').read_bytes())
    write(out/'selected/deployment.json', dict(arm='joint_line', checkpoint=str(exported),
          sha256=study.digest_file(exported), branch='H_SP', selection=str(target/'selection.json')))
    for path in ('configuration.json', 'em_line_manifest.json', 'initial.pt', 'initial.json'):
        write(out/path, {})
    held = {'value': False, 'calls': 0}

    @contextmanager
    def lease(_):
        assert not held['value']
        held['value'] = True
        held['calls'] += 1
        try:
            yield
        finally:
            held['value'] = False

    monkeypatch.setattr(study, '_lease', lease)
    captured = {}

    def freeze(output, checkpoints, baseline, extra, sources, evaluation):
        assert held['value'], 'Evidence and file hashes must be frozen under the work lease'
        captured.update(checkpoints=checkpoints, files=extra, evaluation=evaluation)
        return output/'research_freeze.json'

    monkeypatch.setattr(study, 'freeze_research', freeze)
    return out, held, captured


def test_freeze_pins_exported_checkpoint_and_sequential_lineage_under_lock(tmp_path, monkeypatch):
    out, held, captured = setup_freeze(tmp_path, monkeypatch)
    assert study.freeze(out, splits=['final'], **PROTOCOL) == out/'data/research_freeze.json'
    assert held == {'value': False, 'calls': 1}
    assert captured['evaluation']['selected_arm'] == 'joint_line'
    assert captured['checkpoints']['selected_deployment'] == out/'selected/candidate.pt'
    assert set(study.STAGES).issubset(captured['checkpoints'])
    assert captured['files']['initial'] == out/'initial.pt'
    for name in study.STAGES:
        assert captured['files'][f'{name}:training_manifest.json'].exists()
        assert captured['files'][f'{name}:status.json'].exists()


@pytest.mark.parametrize('mutation', ['export_bytes', 'export_arm', 'embedded_selection'])
def test_freeze_rejects_export_or_selection_disagreement(tmp_path, monkeypatch, mutation):
    out, held, captured = setup_freeze(tmp_path, monkeypatch)
    if mutation == 'export_bytes':
        (out/'selected/candidate.pt').write_bytes(b'changed selected checkpoint')
    elif mutation == 'export_arm':
        path = out/'selected/deployment.json'
        deployment = study._read(path)
        deployment['arm'] = 'em_P'
        write(path, deployment)
    else:
        path = out/'benchmark_evidence.json'
        evidence = study._read(path)
        evidence['selection']['recommended_deployment_arm'] = 'em_P'
        write(path, evidence)
    with pytest.raises(ValueError, match='differs'):
        study.freeze(out, splits=['final'], **PROTOCOL)
    assert not captured
    assert not held['value']


@pytest.mark.parametrize('command', ['train', 'benchmark', 'freeze'])
def test_development_guard_is_rechecked_after_acquiring_work_lease(tmp_path, monkeypatch, command):
    out = tmp_path/'run'
    out.mkdir()
    monkeypatch.setattr(study, 'load', lambda _: (out, {'target_rhs_count': 1}, None, None))

    @contextmanager
    def lease(_):
        # Another process freezes after the first pre-lock check.  No training,
        # benchmark writes, or second freeze may proceed after acquiring it.
        write(out/'data/research_freeze.json', {})
        yield

    monkeypatch.setattr(study, '_lease', lease)
    with pytest.raises(ValueError, match='frozen'):
        if command == 'train':
            study.train(out)
        else:
            getattr(study, command)(out, **PROTOCOL)
