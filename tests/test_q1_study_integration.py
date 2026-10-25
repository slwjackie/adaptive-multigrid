"""Q1 operators enter the real data, training and multigrid study paths."""
import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_mg.v67 import em_line_study, unet_transfer_study
from adaptive_mg.v67.research_data import load_development_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('study,name', [
    (em_line_study, 'em_line'), (unet_transfer_study, 'unet_transfer')])
def test_q1_prepare_roundtrip_and_selector_validation(tmp_path, study, name):
    settings = json.loads((ROOT / f'configs/v6_7_{name}_q1_smoke.json').read_text())
    # Keep all seven families and use a small, genuine nonterminal grid.
    settings.update(training_sizes=[7], validation_sizes=[7])
    config = tmp_path / 'settings.json'
    config.write_text(json.dumps(settings))
    out = tmp_path / 'run'
    study.prepare(config, out)
    _, stored, _, baseline = study.load(out)
    train, validation, manifest = load_development_data(out / 'data', baseline)
    assert stored['discretization'] == 'q1'
    assert len(train) == len(validation) == 7
    for example in train + validation:
        assert example.case.discretization == 'q1'
        np.testing.assert_allclose(example.a @ example.exact, example.b, atol=0.)
    assert all(r['case']['discretization'] == 'q1'
               for records in manifest['splits'].values() for r in records)
    assert not (out / 'data/final_data_manifest.json').exists()
    # A resume cannot silently reinterpret the data as P1.
    settings['discretization'] = 'p1'
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        study.prepare(config, out, resume=True)
    settings['discretization'] = 'q2'
    with pytest.raises(ValueError, match='discretization'):
        study.validate_settings(settings)


def test_q1_em_line_smoke_trains_all_stages(tmp_path):
    """Exercise EM labels, learned P, and smoother losses on actual Q1 A."""
    config = ROOT / 'configs/v6_7_em_line_q1_smoke.json'
    out = tmp_path / 'run'
    em_line_study.prepare(config, out)
    em_line_study.train(out)
    _, _, _, baseline = em_line_study.load(out)
    for stage in em_line_study.STAGES:
        status = json.loads((out / 'checkpoints' / stage / 'status.json').read_text())
        assert status['status'] == 'complete'
        assert status['all_updates_completed'] and status['skipped'] == 0
        model = em_line_study._checkpoint(out, stage, baseline)
        assert model.metadata['transfer_trained']
    assert not (out / 'data/final_claim.json').exists()
