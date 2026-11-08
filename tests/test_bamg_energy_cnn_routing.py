from copy import deepcopy
import json
from statistics import fmean

import pytest

from adaptive_mg.v67 import bamg_energy_cnn_routing as routing


def settings():
    return dict(lambdas=[10, 30, 50], model_seeds=[6101, 6102, 6103],
        data=dict(families=['near_isotropic', 'rotated_anisotropy'],
                  validation=dict(sizes=[15, 127], per_family=2)),
        benchmark=dict(repetitions=3, allowed_failure_rate=0.))


def summarize_cases(cases, config):
    names = list(cases[0]['methods'])
    methods = {}
    for name in names:
        rows = [case['methods'][name] for case in cases]
        methods[name] = dict(missing=sum(row['missing'] for row in rows),
            failures=sum(row['failures'] for row in rows),
            attempts=sum(row['attempts'] for row in rows),
            warm_seconds=fmean(row['warm_seconds'] for row in rows))
    return dict(completed=True, cases_expected=len(cases), cases_completed=len(cases),
        repetitions=config['benchmark']['repetitions'], rhs_count=1,
        methods=methods, case_results=cases)


def evidence(config=None):
    config = settings() if config is None else config
    cases = []
    for n in config['data']['validation']['sizes']:
        for family in config['data']['families']:
            for index in range(config['data']['validation']['per_family']):
                methods = {}
                for strength in config['lambdas']:
                    for arm in ('scratch', 'warm'):
                        for seed in config['model_seeds']:
                            if arm == 'warm':
                                seconds = {10: 1.2, 30: 1., 50: 1.1}[strength]
                            elif n == 15:
                                seconds = {10: 1., 30: 1.2, 50: 2.}[strength]
                            else:
                                seconds = {10: 2., 30: 1.9, 50: 1.8}[strength]
                            name = routing.method_name(strength, arm, seed)
                            methods[name] = dict(warm_seconds=seconds, failures=0,
                                                attempts=3, missing=0, cycles=4.)
                cases.append(dict(case_id=f'validation_{n}_{family}_{index}',
                    shape=[n, n], family=family, methods=methods))
    return summarize_cases(cases, config)


def test_names_and_paired_grid_rules_are_deterministic_and_serializable():
    config, summary = settings(), evidence()
    original = deepcopy((config, summary))
    selected = routing.select_routes(summary, config)
    assert (config, summary) == original
    assert json.loads(json.dumps(selected)) == selected
    assert selected == routing.select_routes(summary, config)
    assert routing.lambda_key(10.) == '10'
    assert routing.run_name(30, 'scratch', 6101) == 'lambda_30_scratch_seed6101'
    assert routing.method_name(50, 'warm', 6102, 'supervised') == 'lambda_50_warm_seed6102_supervised'
    assert routing.routed_name('warm', 6103) == 'routed_warm_seed6103_finetuned'
    assert selected['arms']['scratch']['grid_to_lambda'] == {'15': 10., '127': 50.}
    assert selected['arms']['scratch']['global_lambda'] == 10.
    assert selected['arms']['warm']['grid_to_lambda'] == {'15': 30., '127': 30.}
    assert selected['arms']['scratch']['model_seeds'] == config['model_seeds']
    assert selected['all_prespecified_seeds_retained']
    assert selected['heldout_selection_permitted'] is False
    assert selected['selection_estimate_only'] is True
    # Separate per-seed evidence is diagnostic; one shared policy is deployed.
    grid = selected['evidence']['scratch']['grids']['127']
    assert grid['unanimous_seed_winner']
    assert grid['per_seed_selected_lambda'] == dict.fromkeys(['6101', '6102', '6103'], 50.)
    assert grid['candidates']['50']['independent_operators'] == 4
    assert grid['candidates']['50']['attempts'] == 4 * 3 * 3


def test_unseen_grid_uses_fixed_lower_anchor_clamp_without_any_results():
    policy = routing.select_routes(evidence(), settings())['arms']['scratch']
    assert routing.route_lambda(policy, 3) == 10.
    assert routing.route_lambda(policy, 15) == 10.
    assert routing.route_lambda(policy, 63) == 10.
    assert routing.route_lambda(policy, 127) == 50.
    assert routing.route_lambda(policy, 255) == 50.
    assert routing.route_lambda(policy, 1023) == 50.
    for bad in (True, 2, 127., None, '255'):
        with pytest.raises(ValueError, match='grid size'):
            routing.route_lambda(policy, bad)


def test_failure_rate_precedes_timing_and_no_fallback_is_inserted():
    config, summary = settings(), evidence()
    for case in summary['case_results']:
        for strength in (10, 30, 50):
            for seed in config['model_seeds']:
                row = case['methods'][routing.method_name(strength, 'scratch', seed)]
                row['failures'] = {10: 2, 30: 1, 50: 3}[strength]
                row['warm_seconds'] = {10: .001, 30: 1000., 50: .0001}[strength]
    summary = summarize_cases(summary['case_results'], config)
    policy = routing.select_routes(summary, config)['arms']['scratch']
    assert policy['grid_to_lambda'] == {'15': 30., '127': 30.}
    assert policy['global_lambda'] == 30.
    assert not policy['global_eligible']
    assert not any(policy['grid_eligible'].values())
    assert routing.route_lambda(policy, 255) == 30.


def test_seed_disagreement_is_exposed_and_favorable_seeds_are_not_selected():
    config, summary = settings(), evidence()
    for case in summary['case_results']:
        for strength in (10, 30, 50):
            for seed, winner in zip(config['model_seeds'], (10, 30, 50)):
                case['methods'][routing.method_name(strength, 'scratch', seed)]['warm_seconds'] = (
                    .1 if winner == strength else 1.)
    selected = routing.select_routes(summarize_cases(summary['case_results'], config), config)
    result = selected['evidence']['scratch']['global_result']
    assert result['selected_lambda'] == 10.  # Deterministic lower-lambda tie break.
    assert result['per_seed_selected_lambda'] == {'6101': 10., '6102': 30., '6103': 50.}
    assert not result['unanimous_seed_winner']
    assert selected['arms']['scratch']['grid_to_lambda'] == {'15': 10., '127': 10.}
    assert result['candidates']['10']['warm_seconds'] == pytest.approx(.7)
    assert result['candidates']['10']['std_across_seed_means_seconds'] > 0


@pytest.mark.parametrize('change', [
    lambda s: s.update(completed=False),
    lambda s: s.update(cases_completed=7),
    lambda s: s.update(cases_expected=7),
    lambda s: s.update(rhs_count=3),
    lambda s: s.update(repetitions=1),
    lambda s: s.update(scope='final'),
    lambda s: s['case_results'][0].update(split='ood_grid'),
    lambda s: s['case_results'][0].update(research_split='train'),
    lambda s: s['case_results'][0].update(shape=[31, 31]),
    lambda s: s['case_results'][0].update(shape=[15, 31]),
    lambda s: s['case_results'][0].update(case_id=s['case_results'][1]['case_id']),
    lambda s: s['case_results'][0]['methods'].pop('lambda_50_warm_seed6103_finetuned'),
    lambda s: s['case_results'][0]['methods']['lambda_10_scratch_seed6101_finetuned'].update(missing=1),
    lambda s: s['case_results'][0]['methods']['lambda_10_scratch_seed6101_finetuned'].update(failures=4),
    lambda s: s['case_results'][0]['methods']['lambda_10_scratch_seed6101_finetuned'].update(attempts=2),
    lambda s: s['case_results'][0]['methods']['lambda_10_scratch_seed6101_finetuned'].update(warm_seconds=float('nan')),
    lambda s: s['methods']['lambda_10_scratch_seed6101_finetuned'].update(warm_seconds=.1),
    lambda s: s['methods']['lambda_10_scratch_seed6101_finetuned'].update(failures=1),
])
def test_selection_rejects_incomplete_corrupted_or_heldout_evidence(change):
    summary = evidence()
    change(summary)
    with pytest.raises(ValueError):
        routing.select_routes(summary, settings())


def test_missing_family_cannot_be_replaced_with_more_operators_from_another():
    summary = evidence()
    summary['case_results'][0]['family'] = 'rotated_anisotropy'
    with pytest.raises(ValueError, match='grid/family'):
        routing.select_routes(summary, settings())


@pytest.mark.parametrize('change', [
    lambda s: s.update(lambdas=[10, 30]),
    lambda s: s.update(lambdas=[10, 30, 30]),
    lambda s: s.update(lambdas=[10, 30, 100]),
    lambda s: s.update(lambdas=[True, 30, 50]),
    lambda s: s.update(model_seeds=[]),
    lambda s: s.update(model_seeds=[6101, 6101]),
    lambda s: s.update(model_seeds=[True]),
    lambda s: s.update(routing=dict(stage='supervised')),
    lambda s: s.update(routing=dict(unseen_grid_policy='oracle')),
    lambda s: s.update(routing=dict(select_fastest_seed=True)),
    lambda s: s['data']['validation'].update(sizes=[15, 15]),
    lambda s: s['benchmark'].update(allowed_failure_rate=float('nan')),
])
def test_invalid_protocol_is_rejected(change):
    config = settings()
    change(config)
    with pytest.raises(ValueError):
        routing.select_routes(evidence(), config)


@pytest.mark.parametrize('change', [
    lambda p: p.update(version='other'),
    lambda p: p.update(stage='supervised'),
    lambda p: p.update(arm='fastest_seed'),
    lambda p: p.update(unseen_grid_policy='nearest_test_winner'),
    lambda p: p.update(granularity='per_level'),
    lambda p: p.update(candidates=[10, 50]),
    lambda p: p.update(global_lambda=100),
    lambda p: p.update(global_eligible=1),
    lambda p: p['grid_to_lambda'].update({'255': 50}),
    lambda p: p['grid_to_lambda'].update({'15': 100}),
    lambda p: p['grid_eligible'].update({'15': 1}),
])
def test_malformed_inference_policy_is_rejected(change):
    policy = routing.select_routes(evidence(), settings())['arms']['scratch']
    change(policy)
    with pytest.raises(ValueError):
        routing.route_lambda(policy, 255)
