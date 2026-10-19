"""Fresh data, frozen-P control, joint training, raw diagnostics and resume."""
import json
from pathlib import Path

import pytest

from adaptive_mg.v67 import sparse_smoother_study as study
from adaptive_mg.v67 import unet_transfer_study as source_study
from adaptive_mg.v67.limited import digest_file
from adaptive_mg.v67.research_data import _digests

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def prepared_study(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('sparse_smoother_study')
    settings = json.loads((ROOT / 'configs/v6_7_unet_transfer_smoke.json').read_text())
    settings.update(training_sizes=[7], validation_sizes=[7], training_per_family=1, validation_per_family=1)
    settings['training'].update(updates=2, kd_updates=1)
    settings['unet_transfer']['student_hidden'] = 4
    config = tmp / 'source.json'; config.write_text(json.dumps(settings))
    source = tmp / 'source'
    source_study.prepare(config, source); source_study.train(source, 'direct')
    before = {str(p): digest_file(p) for p in source.rglob('*') if p.is_file()}
    out = tmp / 'sparse'
    study.prepare(ROOT / 'configs/v6_7_sparse_smoother_smoke.json', source, out)
    study.train(out)
    assert all(digest_file(p) == value for p, value in before.items())
    old = json.loads((source / 'data/development_manifest.json').read_text())
    new = json.loads((out / 'data/development_manifest.json').read_text())
    assert not _digests(old) & _digests(new)
    return out


def test_frozen_p_joint_updates_and_training_resume(prepared_study):
    out, settings, cfg, baseline = study.load(prepared_study)
    arms = study._arms(out, settings, baseline)
    original, sparse, joint = (arms[n]['model'] for n in study.ARMS[1:])
    assert original.expert_signature('transfer') == sparse.expert_signature('transfer')
    assert sparse.expert_signature('transfer') != joint.expert_signature('transfer')
    assert sparse.expert_signature('smoother') != joint.expert_signature('smoother')
    digest = digest_file(out / 'checkpoints/joint/candidate.pt')
    study.train(out, resume=True)
    assert digest_file(out / 'checkpoints/joint/candidate.pt') == digest


def test_benchmark_resume_timing_cycles_and_no_fallback_raw_check(prepared_study, monkeypatch):
    out = prepared_study
    kwargs = dict(repeats=1, warmups=0, rhs_counts=[1, 4], regimes=['cold', 'warm', 'multiple'])
    result = study.benchmark(out, **kwargs)
    path = out / 'benchmarks/validation/raw_results.json'; digest = digest_file(path)
    study.benchmark(out, resume=True, **kwargs)
    assert digest_file(path) == digest
    summary = study.cycle_diagnostics(result)
    assert summary['cold/rhs1/direct_CNN']['common_cases'] > 0
    _, settings, cfg, baseline = study.load(out)
    for case in result['rows']:
        for name in study.ARMS:
            cold = case['runs'][name]['cold']['1'][0]
            warm = case['runs'][name]['warm']['1'][0]
            assert cold['setup_scope'] == 'included_once' and warm['setup_scope'] == 'excluded_prepared_operator'
            assert cold['measurement_config']['mg']['strategy_name'] == cfg.mg.strategy_name
    _, validation, _ = study._development(out, settings, baseline)
    arms = study._arms(out, settings, baseline)
    def forbidden(*args, **kwargs): raise AssertionError('Raw diagnostic called safeguarded solve')
    monkeypatch.setattr(study.PreparedAdaptiveMG, 'solve', forbidden)
    monkeypatch.setattr(study.PreparedAdaptiveMG, 'solve_many', forbidden)
    raw = study.raw_contractions(validation[:1], arms, cfg, cycles=2, probes=1)
    assert raw['fallback_allowed'] is False
    assert all(row['valid'] for row in raw['rows'][0]['arms'].values())
    with pytest.raises(ValueError, match='Smoke'):
        study.freeze(out, **kwargs)


def test_bad_protocol_and_import_changes_are_rejected(prepared_study):
    out, settings, _, _ = study.load(prepared_study)
    settings['smoother_training']['trainable_components'] = ['transfer']
    with pytest.raises(ValueError, match='trainable components'):
        study.validate_settings(settings)
    with pytest.raises(ValueError, match='repeat/warmup'):
        study.benchmark(out, repeats=0)
