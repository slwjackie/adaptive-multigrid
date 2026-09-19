from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from adaptive_mg import MGConfig
from adaptive_mg.provenance import hardware_environment, json_safe
from adaptive_mg.v67 import AdaptiveConfig
from adaptive_mg.v67.limited import initialize_timing_runtime
from adaptive_mg.v67.strong import StrongRules
from adaptive_mg.v67.research_data import (
    make_research_plan, prepare_development_data, freeze_research,
    claim_final_evaluation, materialize_final_data,
)
from adaptive_mg.v67.research_evaluation import evaluate_research
from adaptive_mg.v67.research_policy import ResearchPolicy, _scope
from adaptive_mg.v67.research_certification import _assess_final, certify_research_final


def cfg():
    return AdaptiveConfig(mg=MGConfig(mode='classical', max_cycles=60, tolerance=1e-8,
        absolute_tolerance=1e-12, coarsest_n=3, nn_levels=1, stencil_backend='csr'),
        mode='research', branch='auto', spatial=False, gate_mode='open')


def synthetic_evidence(count=21, repeats=5):
    """Synthetic proof records for acceptance logic, never benchmark artifacts."""
    config = cfg()
    cases, rows = [], []
    for index in range(count):
        expected = dict(name=f'test_fixture_{index}', case={'n': 7}, seed=index, scale=1.,
                        operator_digest=f'{index:064x}', normalized_operator_digest=f'{index+100:064x}',
                        rhs_digest=f'{index+200:064x}', exact_digest=f'{index+300:064x}')
        cases.append(expected)
        runs = {}
        for arm, branch, time in [('strong_C', 'C', 2.), ('adaptive', 'auto', 1.)]:
            inner = dict(converged=True, stopping_threshold=1e-8, final_true_residual=1e-9,
                residual_history=[1., 1e-9], relative_residual_history=[1., 1e-9], executed_cycles=4,
                recomputed_true_residual=1e-9, relative_solution_error=1e-9, relative_energy_error=1e-9,
                rhs_digest=expected['rhs_digest'], exact_digest=expected['exact_digest'],
                x0_digest=hashlib.sha256(np.zeros(49, np.float64).tobytes()).hexdigest(),
                measurement_config=config.to_dict(),
                stats={'branch_H_S_cycles': 1, 'neural_apply_calls': 2} if arm == 'adaptive' else {})
            run = dict(rhs_results=[inner], measurement_config=config.to_dict(),
                numerical_dtype='float64', inference_dtype='float32', regime='cold', rhs_count=1,
                selector_in_wall_time=True, wall_seconds=time, error=None,
                operator_digest=expected['operator_digest'], normalized_operator_digest=expected['normalized_operator_digest'],
                rhs_digests=[expected['rhs_digest']], exact_digests=[expected['exact_digest']],
                training_only=False, requested_branch=branch)
            runs[arm] = {'cold': {'1': [deepcopy(run) for _ in range(repeats)]}}
        rows.append(dict(example={k: v for k, v in expected.items() if k not in {'rhs_digest', 'exact_digest'}},
                         research_split='final', runs=runs))
    return rows, cases, config


def assess(rows, cases, config, repeats=5):
    return _assess_final(rows, cases, config, adaptive_arm='adaptive', repeats=repeats)


def test_synthetic_complete_final_can_pass_strict_criteria():
    rows, cases, config = synthetic_evidence()
    result = assess(rows, cases, config)
    assert result['validated']
    assert all(result['checks'].values())
    assert result['acceptance']['geometric_speedup'] == 2.
    assert result['acceptance']['minimum_speedup'] == 1 / (1 - config.win_margin)
    assert result['acceptance']['neural_used_cases'] == 21


@pytest.mark.parametrize('mutation', ['failure', 'residual', 'budget', 'rhs', 'x0', 'teacher', 'precision', 'no_neural', 'slow', 'error'])
def test_invalid_or_nonpassing_evidence_never_certifies(mutation):
    rows, cases, config = synthetic_evidence()
    run = rows[0]['runs']['adaptive']['cold']['1'][0]
    inner = run['rhs_results'][0]
    if mutation == 'failure':
        inner['converged'] = False
        run['wall_seconds'] = 1e-30
    elif mutation == 'residual':
        inner['recomputed_true_residual'] = 1e-3
    elif mutation == 'budget':
        inner['executed_cycles'] = config.mg.max_cycles + 1
    elif mutation == 'rhs':
        inner['rhs_digest'] = '0' * 64
    elif mutation == 'x0':
        inner['x0_digest'] = '0' * 64
    elif mutation == 'teacher':
        run['training_only'] = True
    elif mutation == 'precision':
        run['numerical_dtype'] = 'float32'
    elif mutation == 'error':
        inner['relative_energy_error'] = .1
    elif mutation == 'no_neural':
        for row in rows:
            for repeat in row['runs']['adaptive']['cold']['1']:
                repeat['rhs_results'][0]['stats'] = {}
    elif mutation == 'slow':
        for row in rows:
            for repeat in row['runs']['adaptive']['cold']['1']:
                repeat['wall_seconds'] = 2.1
    assert not assess(rows, cases, config)['validated']


def test_success_swap_is_not_aggregate_success_preservation():
    rows, cases, config = synthetic_evidence()
    rows[0]['runs']['strong_C']['cold']['1'][0]['rhs_results'][0]['converged'] = False
    rows[1]['runs']['adaptive']['cold']['1'][0]['rhs_results'][0]['converged'] = False
    result = assess(rows, cases, config)
    assert result['acceptance']['strong_successes'] == result['acceptance']['neural_successes']
    assert result['acceptance']['new_failures'] == 1
    assert not result['validated']


def test_small_repeated_or_cherry_picked_cohort_rejected():
    rows, cases, config = synthetic_evidence(count=19)
    assert not assess(rows, cases, config)['validated']
    rows, cases, config = synthetic_evidence(repeats=4)
    assert not assess(rows, cases, config, repeats=4)['validated']
    rows, cases, config = synthetic_evidence()
    assert not assess(rows[:-1], cases, config)['validated']
    rows[0]['research_split'] = 'ood_rotation'
    assert not assess(rows, cases, config)['validated']


def test_real_gated_small_final_is_pinned_but_never_certified(tmp_path):
    initialize_timing_runtime()
    plan = make_research_plan(seed=140998)
    for spec in plan['splits'].values():
        spec.update(sizes=[7], per_family=1, count=len(spec['families']))
    history = tmp_path / 'history'
    history.mkdir()
    root = tmp_path / 'run'
    rules, config = StrongRules(), cfg()
    prepare_development_data(root, rules, plan=plan, historical_roots=[history])
    policy = ResearchPolicy({}, {}, rules.digest(), _scope(config), hardware_environment(), {}, {'test_fixture': True})
    checkpoint = root / 'fixture.pt'
    checkpoint.write_bytes(b'not a deployable model: final-gate test fixture only')
    rule_path = root / 'rules.json'
    rule_path.write_text(json.dumps(rules.to_dict()))
    frozen_config = dict(solver=config.to_dict(), policy_digest=policy.digest(), rules_digest=rules.digest(),
                         final_evaluation=dict(repeats=5, regimes=['cold'], rhs_counts=[1]))
    freeze = freeze_research(root, {'fixture': checkpoint}, rule_path, {}, [__file__], frozen_config)
    with claim_final_evaluation(root, freeze, {'test_only': True}) as claim:
        splits, _ = materialize_final_data(root, claim, rules)
        arms = dict(strong_C=dict(model=None, branch='C'), adaptive=dict(model=None, branch='auto', policy=policy))
        result = evaluate_research(splits['final'], arms, config, rules, root / 'final_measurements',
                                  repeats=5, warmups=0, regimes=('cold',), rhs_counts=(1,))
        certificate = certify_research_final(root, claim, result, policy, config, rules)
        assert not certificate['validated']
        assert certificate['acceptance']['final_cases'] == 7
        assert certificate['acceptance']['neural_used_cases'] == 0
        assessment = json.loads((root / 'final_assessment.json').read_text())
        assert assessment['acceptance'] == certificate['acceptance']
        assert set(assessment['raw_evidence']) == {'manifest', 'raw_results', 'comparison'}
        assert certificate == certify_research_final(root, claim, result, policy, config, rules)
        with pytest.raises(ValueError, match='precommitted'):
            certify_research_final(root, claim, result, policy, config, rules, require_solution_error_not_worse=False)
        result['rows'][0]['runs']['adaptive']['cold']['1'][0]['wall_seconds'] = 1e-30
        with pytest.raises(ValueError, match='durable raw'):
            certify_research_final(root, claim, result, policy, config, rules)
    with pytest.raises(ValueError, match='active final-evaluation'):
        certify_research_final(root, claim, result, policy, config, rules)
