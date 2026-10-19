"""Offline search interruption/failure contracts without expensive PDE training."""
from types import SimpleNamespace

import pytest
import torch

import adaptive_mg.v67.joint_transfer_study as study


@pytest.fixture
def search_run(tmp_path, monkeypatch):
    example = SimpleNamespace(name='train_case', group_digest='a' * 64,
                              digest='b' * 64, n=7, research_split='train')
    baseline = SimpleNamespace(digest=lambda: 'baseline')
    source = SimpleNamespace(signature=lambda: 'joint_source')
    settings = dict(seed=19, target_rhs_count=1, raw_cycles=2,
        search=dict(starts=2, updates=1, cycles=1, repeats=1, learning_rate=.01, perturbation=.01),
        training=dict(probes_per_kind=1, hard_smoothing=1, hard_cycles=1, gradient_clip=1.))
    binding = dict(operator_digest=example.digest, shape=[7, 7])

    def make(*args, **kwargs):
        return SimpleNamespace(transfer=torch.nn.Linear(1, 1), smoother=torch.nn.Linear(1, 1),
                               metadata=dict(oracle_binding=binding))

    def export(oracle):
        return dict(metadata=oracle.metadata, transfer=oracle.transfer.state_dict(),
                    smoother=oracle.smoother.state_dict())

    def restore(payload):
        result = make()
        result.transfer.load_state_dict(payload['transfer'])
        result.smoother.load_state_dict(payload['smoother'])
        result.metadata = payload['metadata']
        return result

    def objective(example, oracle, cfg, **kwargs):
        return sum(p.square().sum() for module in (oracle.transfer, oracle.smoother)
                   for p in module.parameters()), {}

    monkeypatch.setattr(study, 'load', lambda out: (tmp_path, settings, None, baseline))
    monkeypatch.setattr(study, '_development', lambda *args: ([example], [], {}))
    monkeypatch.setattr(study, 'load_development_data', lambda *args: ([example], [], {}))
    monkeypatch.setattr(study, '_checkpoint', lambda *args: source)
    monkeypatch.setattr(study, 'make_oracle', make)
    monkeypatch.setattr(study, 'export_oracle', export)
    monkeypatch.setattr(study, 'import_oracle', restore)
    monkeypatch.setattr(study, 'joint_objective', objective)
    monkeypatch.setattr(study, 'measure_raw_candidate', lambda *args, **kwargs: dict(
        successful=True, seconds=1., fallback_used=False))
    return tmp_path, example, source, make


def test_failed_refinement_keeps_measured_initial_candidate(search_run, monkeypatch):
    out, example, _, _ = search_run
    def failed(*args, **kwargs):
        raise FloatingPointError('numerically infeasible update')
    monkeypatch.setattr(study, 'joint_objective', failed)
    study.search(out)
    report = study._read(out / 'search/report.json')
    assert report['coverage'] == 1 and report['total'] == 1
    row = report['rows'][0]
    assert row['selected']['label'] == 'initial'
    for start in row['starts']:
        assert start['status'] == 'failed' and start['updates'] == 0
        assert start['selected']['timing']['successful']
        assert start['selected']['timing']['fallback_used'] is False
    assert set(study._oracle_records(out)) == {example.group_digest}


def test_initialization_failure_records_start_and_continues(search_run, monkeypatch):
    out, _, _, make = search_run
    def initialize(*args, **kwargs):
        if kwargs['initialization'] == 'network':
            raise ValueError('invalid network hierarchy')
        return make()
    monkeypatch.setattr(study, 'make_oracle', initialize)
    study.search(out)
    row = study._read(out / 'search/report.json')['rows'][0]
    assert row['starts'][0]['status'] == 'failed'
    assert row['starts'][0]['candidates'] == []
    assert row['starts'][1]['status'] == 'complete'
    assert row['selected'] is not None


def test_completed_search_resume_preserves_evidence_and_checks_hashes(search_run, monkeypatch):
    out, _, _, _ = search_run
    study.search(out)
    evidence = (out / 'search/report.json').read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail('Completed search reran generation, optimization or timing')
    monkeypatch.setattr(study, 'make_oracle', forbidden)
    monkeypatch.setattr(study, 'joint_objective', forbidden)
    monkeypatch.setattr(study, 'measure_raw_candidate', forbidden)
    study.search(out, resume=True)
    assert (out / 'search/report.json').read_bytes() == evidence
    selected = study._read(out / 'search/report.json')['rows'][0]['selected']
    (out / selected['path']).write_bytes(b'changed operator')
    with pytest.raises(ValueError, match='hash changed'):
        study.search(out, resume=True)


def test_distillation_requires_complete_search_coverage(search_run):
    out, _, _, _ = search_run
    study.search(out)
    path = out / 'search/report.json'
    evidence = study._read(path)
    evidence['rows'] = []
    study.write_json(path, evidence)
    with pytest.raises(ValueError, match='every unique TRAIN operator'):
        study._oracle_records(out)


def test_distillation_rejects_changed_source_checkpoint(search_run):
    out, _, source, _ = search_run
    study.search(out)
    source.signature = lambda: 'changed_joint_source'
    with pytest.raises(ValueError, match='Search source'):
        study._oracle_records(out)
