"""Actual wall-time accounting and provenance for online transfer adaptation."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.fixed_classical import FixedClassicalBaseline
from adaptive_mg.v67.research_evaluation import evaluate_research, measured_research, _arm_config
from adaptive_mg.v67.unet_transfer_study import initial_model, validate_settings


@pytest.fixture
def setup():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    project = Path(__file__).resolve().parents[1]
    settings = json.loads((project / 'configs/v6_7_unet_transfer_smoke.json').read_text())
    cfg = validate_settings(settings)
    cfg = replace(cfg, mg=replace(cfg.mg, max_cycles=60))
    rules = FixedClassicalBaseline.from_config(cfg)
    e = dataset([7], 7, 162843)[0]
    e.research_split = 'validation'
    e.classical_selection = rules.selection_record()
    model = initial_model(settings, 'direct')
    arms = dict(fixed_C=dict(model=None, branch='C', selector=False),
                pruned_CNN=dict(model=model, branch='H_P', selector=False),
                pruned_CNN_TTT=dict(model=model, branch='H_P', selector=False,
                    ttt=dict(max_updates=1, train_probes=1, guard_probes=1, raw_cycles=1)))
    yield e, cfg, rules, arms
    torch.set_num_threads(previous)


def test_ttt_inclusive_cost_and_explicit_warm_exclusion(setup, tmp_path):
    e, cfg, rules, arms = setup
    original_signature = arms['pruned_CNN']['model'].signature()
    result = evaluate_research([e], arms, cfg, rules, tmp_path,
        reference_arm='fixed_C', repeats=1, warmups=0,
        regimes=('cold', 'warm', 'multiple'), rhs_counts=(2,))
    runs = result['rows'][0]['runs']['pruned_CNN_TTT']
    for regime, count in (('cold', '1'), ('multiple', '2')):
        record = runs[regime][count][0]
        assert record['successful']
        assert record['ttt_seconds'] > 0
        assert record['constructor_seconds'] >= record['ttt_seconds']
        assert record['wall_seconds'] >= record['constructor_seconds']
        assert record['total_setup_seconds'] == pytest.approx(
            record['classical_setup_seconds'] + record['neural_setup_seconds'] + record['ttt_seconds'])
        assert record['warm_ttt_seconds'] == 0
        assert record['ttt_report'] is not None
    warm = runs['warm']['1'][0]
    assert warm['successful']
    assert warm['ttt_seconds'] == 0
    assert warm['total_setup_seconds'] == 0
    assert warm['warm_ttt_seconds'] > 0
    assert warm['warm_prime_seconds'] >= warm['warm_ttt_seconds']
    assert warm['ttt_preparation_seconds'] == warm['warm_ttt_seconds']
    assert arms['pruned_CNN']['model'].signature() == original_signature
    assert result['manifest']['arms']['pruned_CNN_TTT']['ttt_settings']['max_updates'] == 1
    assert 'ttt_scope' in result['manifest']
    assert all('ttt_seconds' in row for row in result['table'])


def test_ttt_settings_are_bound_to_resume(setup, tmp_path):
    e, cfg, rules, arms = setup
    evaluate_research([e], arms, cfg, rules, tmp_path, reference_arm='fixed_C',
        repeats=1, warmups=0, regimes=('cold',))
    arms['pruned_CNN_TTT']['ttt']['learning_rate'] = 2e-4
    with pytest.raises(ValueError, match='mismatch'):
        evaluate_research([e], arms, cfg, rules, tmp_path, reference_arm='fixed_C',
            repeats=1, warmups=0, regimes=('cold',), resume=True)


def test_ttt_cannot_be_hidden_in_classical_reference(setup, tmp_path):
    e, cfg, rules, arms = setup
    arms['fixed_C']['ttt'] = {}
    with pytest.raises(ValueError, match='reference'):
        evaluate_research([e], arms, cfg, rules, tmp_path, reference_arm='fixed_C',
            repeats=1, warmups=0, regimes=('cold',))
