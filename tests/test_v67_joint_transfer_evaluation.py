from dataclasses import replace

import numpy as np
import pytest

from adaptive_mg import MGConfig
from adaptive_mg.v67 import AdaptiveConfig, Components
from adaptive_mg.v67.fixed_classical import (
    BilinearClassicalBaseline, FixedClassicalBaseline, BILINEAR_STRATEGY_NAME,
    FIXED_STRATEGY_NAME,
)
from adaptive_mg.v67.research_data import make_research_plan, _example, _specification
import adaptive_mg.v67.joint_transfer_evaluation as evaluation


def example():
    plan = make_research_plan(smoke=True, seed=680139)
    return _example(_specification('validation', plan['splits']['validation'],
                                  7, 'near_isotropic', 0, 0))


def config():
    return AdaptiveConfig(mg=MGConfig(mode='classical', strategy_name=BILINEAR_STRATEGY_NAME,
        tolerance=1e-7, max_cycles=60, coarsest_n=3, nn_levels=1,
        pre_steps=2, post_steps=2, near_tolerance_factor=0., stencil_backend='csr'),
        mode='research', branch='H_P', spatial=False, gate_mode='open')


def test_raw_cycles_use_per_arm_parent_and_never_solver_fallback(monkeypatch):
    cfg = config()
    emcfg = replace(cfg, mg=replace(cfg.mg, strategy_name=FIXED_STRATEGY_NAME,
        em_iterations=5, em_fixed_iterations=True, em_rtol=1e-8))
    arms = {
        'fixed_C': dict(model=None, branch='C', selector=False,
            rules=BilinearClassicalBaseline.from_config(cfg)),
        'em5_C': dict(model=None, branch='C', selector=False, config=emcfg,
            rules=FixedClassicalBaseline.from_config(emcfg)),
        'joint': dict(model=Components.create(hidden=4, seed=17), branch='H_SP', selector=False),
    }
    original = evaluation.PreparedAdaptiveMG.ensure_branch

    def fail_learned(self, branch, stats):
        if branch != 'C':
            raise ValueError('infeasible learned hierarchy')
        return original(self, branch, stats)

    def forbidden_solve(*args, **kwargs):
        raise AssertionError('Raw diagnostics must never call safeguarded solve')

    monkeypatch.setattr(evaluation.PreparedAdaptiveMG, 'ensure_branch', fail_learned)
    monkeypatch.setattr(evaluation.PreparedAdaptiveMG, 'solve', forbidden_solve)
    result = evaluation.raw_contractions([example()], arms, cfg, cycles=3, probes=2)
    assert not result['fallback_allowed']
    row = result['rows'][0]['arms']
    for name, strategy in (('fixed_C', BILINEAR_STRATEGY_NAME), ('em5_C', FIXED_STRATEGY_NAME)):
        assert row[name]['valid']
        assert row[name]['strategy'] == strategy
        assert np.shape(row[name]['histories']) == (2, 3)
        assert row[name]['contraction'] > 0
        assert row[name]['cycle_stats']['rollback_count'] == 0
    assert not row['joint']['valid']
    assert row['joint']['contraction'] is None
    assert 'infeasible learned hierarchy' in row['joint']['error']
    assert not row['joint']['fallback_allowed']


def test_raw_cycle_failure_retains_partial_history(monkeypatch):
    cfg, e = config(), example()
    calls = 0
    original = evaluation.hybrid_cycle

    def divergent(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return np.full_like(args[1], np.nan)
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, 'hybrid_cycle', divergent)
    arms = {'learned_P': dict(model=Components.create(hidden=4, seed=71),
                             branch='H_P', selector=False)}
    result = evaluation.raw_contractions([e], arms, cfg, cycles=3, probes=2)
    row = result['rows'][0]['arms']['learned_P']
    assert not row['valid']
    assert len(row['histories']) == 1 and len(row['histories'][0]) == 1
    assert 'Nonfinite' in row['error']


def run(seconds, *, successful=True, cycles=10, accepted=0, fallback=0, rollback=0):
    return dict(wall_seconds=seconds, successful=successful, setup_scope='included_once',
        total_setup_seconds=seconds * .4, solve_seconds=seconds * .6,
        executed_cycles=cycles, accepted_neural_cycles=accepted, fallback_count=fallback,
        rollback_count=rollback, setup_failures=0, operator_generation_seconds=seconds * .1)


def synthetic_result():
    timings = {
        'fixed_C': [100., 1.],
        'em5_C': [80., .5],
        'learned_P': [50., 2.],
        'learned_S': [70., 3.],
        'joint': [90., .1],
        'distilled_joint': [.01, .01],
    }
    rows = []
    for index, n in enumerate((7, 15)):
        runs = {}
        for name, values in timings.items():
            success = name != 'distilled_joint' or index == 0
            measurements = [run(values[index], successful=success,
                cycles=10 if name == 'fixed_C' else 8, accepted=4 if name not in ('fixed_C', 'em5_C') else 0,
                fallback=2 if name == 'joint' else 0, rollback=1 if name == 'joint' else 0)
                for _ in range(2)]
            runs[name] = {'multiple': {'1': measurements},
                          'warm': {'1': [run(.01 if name == 'joint' else 1.)] * 2}}
        rows.append(dict(example=dict(name=f'case{index}', case=dict(n=n)),
            research_split='validation', runs=runs))
    arms = {name: dict(model_signature=f'{name}-signature', reference_only=name == 'em5_C')
            for name in timings}
    return dict(rows=rows, manifest=dict(repeats=2, arms=arms),
        summary={'multiple': {'1': {name: {} for name in arms}},
                 'warm': {'1': {name: {} for name in arms}}})


def test_selection_uses_expected_measured_time_not_contraction_or_warm_speed():
    result = synthetic_result()
    selected = evaluation.select_measured_candidate(result)
    assert selected['selected_arm'] == 'learned_P'
    assert selected['recommended_deployment_arm'] == 'learned_P'
    assert selected['selected_model_signature'] == 'learned_P-signature'
    assert selected['candidates']['learned_P']['mean_case_median_wall_seconds'] == 26.
    assert selected['candidates']['joint']['mean_case_median_wall_seconds'] == 45.05
    # An extremely fast failed run is not allowed to win a success-only cohort.
    failed = selected['candidates']['distilled_joint']
    assert not failed['eligible'] and failed['failures'] == 1
    assert failed['failure_case_ids'] == ['case1']
    assert failed['mean_case_median_wall_seconds'] is None
    assert selected['warm_secondary']


def test_selection_keeps_classical_deployment_when_all_learned_are_slower():
    result = synthetic_result()
    for row in result['rows']:
        row['runs']['fixed_C']['multiple']['1'] = [run(.001)] * 2
    selected = evaluation.select_measured_candidate(result)
    assert selected['selected_arm'] == 'learned_P'
    assert selected['recommended_deployment_arm'] == 'fixed_C'
    assert not selected['faster_than_fixed_on_target']


def test_selection_rejects_heldout_or_setup_excluded_primary_evidence():
    result = synthetic_result()
    with pytest.raises(ValueError, match='setup-inclusive'):
        evaluation.select_measured_candidate(result, regime='warm')
    result['rows'][0]['research_split'] = 'final'
    with pytest.raises(ValueError, match='Final/OOD'):
        evaluation.select_measured_candidate(result)
    result['rows'][0]['research_split'] = 'validation'
    result['rows'][0]['runs']['joint']['multiple']['1'][0]['setup_scope'] = 'excluded_prepared_operator'
    with pytest.raises(ValueError, match='excludes setup'):
        evaluation.select_measured_candidate(result)


def test_missing_repetition_disqualifies_candidate_but_retains_evidence():
    result = synthetic_result()
    result['rows'][0]['runs']['learned_P']['multiple']['1'].pop()
    selected = evaluation.select_measured_candidate(result)
    assert selected['selected_arm'] == 'learned_S'
    failed = selected['candidates']['learned_P']
    assert failed['failure_case_ids'] == ['case0']
    assert failed['rows'][0]['observed_repeats'] == 1


def test_cycle_diagnostics_separate_accepted_executed_and_recovery_by_grid():
    result = synthetic_result()
    report = evaluation.cycle_diagnostics(result)
    joint = report['multiple/rhs1/joint']
    assert joint['fewer_cycles'] == 2 and joint['candidate_cycle_sum'] == 16
    assert joint['paired_totals']['candidate']['accepted_neural_cycles'] == 8
    assert joint['paired_totals']['candidate']['fallback_count'] == 4
    assert joint['paired_totals']['candidate']['rollback_count'] == 2
    assert joint['by_n']['7']['total_cases'] == 1
    assert joint['by_n']['15']['candidate_cycle_sum'] == 8
    distilled = report['multiple/rhs1/distilled_joint']
    assert distilled['successful_cases'] == 1 and distilled['failed_cases'] == 1
    assert distilled['common_successful_cases'] == 1
    assert distilled['excluded_failure_case_ids'] == ['case1']
    assert distilled['by_n']['15']['failed_cases'] == 1
    assert distilled['by_n']['15']['paired_totals']['candidate']['wall_seconds'] is None
    assert not distilled['rows'][1]['candidate_success']


def test_raw_diagnostic_rejects_zero_budget_or_online_policy():
    cfg, e = config(), example()
    with pytest.raises(ValueError, match='positive integer'):
        evaluation.raw_contractions([e], {'fixed_C': dict(branch='C')}, cfg, cycles=0, probes=1)
    with pytest.raises(ValueError, match='fixed explicit branch'):
        evaluation.raw_contractions([e], {'joint': dict(branch='H_SP', ttt={})}, cfg, cycles=1, probes=1)
