"""The staged line study selects on full-cohort measured setup plus solve."""
from copy import deepcopy

import pytest

from adaptive_mg.v67.joint_transfer_evaluation import (
    LEARNED_ARMS, select_measured_candidate,
)
from adaptive_mg.v67.research_data import HOLDOUT


STAGES = ('em_P', 'frozen_P_line', 'joint_line')


def measured_result():
    # em_P has the fastest complete workload.  frozen_P_line has the fastest
    # solve alone, and joint_line the fastest warm diagnostic: neither wins.
    times = {
        'fixed_C': ((10., 5.), (4., 3.)),
        'em_P': ((3., 2.), (2., 2.)),
        'frozen_P_line': ((8., .5), (3., .5)),
        'joint_line': ((4., 2.), (3., 2.)),
    }
    rows = []
    for index, n in enumerate((15, 31)):
        runs = {}
        for name, values in times.items():
            setup, solve = values[index]
            run = dict(successful=True, wall_seconds=setup + solve,
                       total_setup_seconds=setup, solve_seconds=solve,
                       setup_scope='included_once')
            warm = dict(run, wall_seconds=.001 if name == 'joint_line' else solve,
                        total_setup_seconds=0., setup_scope='excluded_prepared_operator')
            runs[name] = {'multiple': {'4': [deepcopy(run) for _ in range(3)]},
                          'warm': {'4': [deepcopy(warm) for _ in range(3)]}}
        rows.append(dict(example=dict(name=f'validation_{index}', case=dict(n=n)),
                         research_split='validation', runs=runs))
    arms = {name: dict(model_signature=f'{name}-checkpoint') for name in times}
    return dict(manifest=dict(repeats=3, arms=arms), rows=rows,
                summary={regime: {'4': {name: {} for name in times}}
                         for regime in ('multiple', 'warm')})


def choose(result, **kwargs):
    return select_measured_candidate(result, rhs_count=4,
                                     candidate_arms=STAGES, **kwargs)


def test_staged_names_are_opt_in_and_choose_total_time():
    result = measured_result()
    selected = choose(result)
    assert not set(STAGES) & set(LEARNED_ARMS)
    assert selected['selected_arm'] == 'em_P'
    assert selected['recommended_deployment_arm'] == 'em_P'
    assert selected['selected_model_signature'] == 'em_P-checkpoint'
    assert selected['candidates']['em_P']['mean_case_median_wall_seconds'] == 4.5
    assert selected['candidates']['frozen_P_line']['mean_case_median_wall_seconds'] == 6.
    assert selected['warm_secondary']
    assert not selected['final_certificate']


@pytest.mark.parametrize('problem', ['one_failed_repeat', 'missing_repeat'])
def test_fast_candidate_requires_every_repeat(problem):
    result = measured_result()
    runs = result['rows'][1]['runs']['em_P']['multiple']['4']
    if problem == 'one_failed_repeat':
        # One failure among otherwise successful runs must not disappear when
        # timings are aggregated into a median or successful-only subset.
        runs[0].update(successful=False, wall_seconds=.00001)
    else:
        runs.pop()
    selected = choose(result)
    assert selected['selected_arm'] == 'joint_line'
    rejected = selected['candidates']['em_P']
    assert not rejected['eligible']
    assert rejected['failure_case_ids'] == ['validation_1']
    assert rejected['mean_case_median_wall_seconds'] is None
    assert rejected['total'] == 2


def test_no_complete_candidate_retains_classical_deployment():
    result = measured_result()
    for name in STAGES:
        result['rows'][0]['runs'][name]['multiple']['4'][0]['successful'] = False
    selected = choose(result)
    assert selected['selected_arm'] is None
    assert selected['recommended_deployment_arm'] == 'fixed_C'


def test_warm_only_or_excluded_setup_cannot_select_staged_model():
    result = measured_result()
    with pytest.raises(ValueError, match='setup-inclusive'):
        choose(result, regime='warm')
    result['rows'][0]['runs']['joint_line']['multiple']['4'][0]['setup_scope'] = (
        'excluded_prepared_operator')
    with pytest.raises(ValueError, match='excludes setup'):
        choose(result)


@pytest.mark.parametrize('split', HOLDOUT)
def test_final_and_ood_cannot_select_staged_model(split):
    result = measured_result()
    result['rows'][1]['research_split'] = split
    with pytest.raises(ValueError, match='Final/OOD'):
        choose(result)


def test_faster_reference_only_model_does_not_become_deployment_candidate():
    result = measured_result()
    result['manifest']['arms']['em_P']['reference_only'] = True
    selected = choose(result)
    assert not selected['candidates']['em_P']['deployable']
    assert selected['selected_arm'] == 'joint_line'
