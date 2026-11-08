"""Validation-only selection of one complete CNN hierarchy by finest grid size.

Routing is a pure, deterministic function of a frozen policy and the input grid.
No RHS, solver residual, family label, holdout measurement, or model seed is used
at inference. All prespecified model seeds share one rule within each training
arm; selecting a favorable seed is not part of this protocol.
"""
from __future__ import annotations

from copy import deepcopy
import math
from statistics import fmean, stdev

from .research_data import FAMILIES

VERSION = 'bamg-energy-cnn-grid-routing-v1'
CANDIDATES = (10., 30., 50.)
ARMS = ('scratch', 'warm')
UNSEEN_GRID_POLICY = 'lower_anchor_clamp'
GRANULARITY = 'finest_grid_whole_hierarchy'


def lambda_key(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Lambda must be a finite positive number')
    return format(float(value), '.15g')


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Model seed must be an integer in [0, 2**32)')
    return seed


def _arm(arm):
    if arm not in ARMS:
        raise ValueError('Expected scratch or warm arm')
    return arm


def run_name(strength, arm, seed):
    return f'lambda_{lambda_key(strength)}_{_arm(arm)}_seed{_seed(seed)}'


def method_name(strength, arm, seed, stage='finetuned'):
    if stage not in ('supervised', 'finetuned'):
        raise ValueError('Unknown CNN checkpoint stage')
    return run_name(strength, arm, seed) + '_' + stage


def routed_name(arm, seed):
    return f'routed_{_arm(arm)}_seed{_seed(seed)}_finetuned'


def _sizes(values):
    if (not isinstance(values, list) or not values or
            any(type(n) is not int or n < 3 for n in values) or
            len(set(values)) != len(values)):
        raise ValueError('Distinct integer validation grid sizes >= 3 are required')
    return sorted(values)


def _seeds(values):
    if not isinstance(values, list) or not values:
        raise ValueError('Prespecified model seeds are required')
    for value in values:
        _seed(value)
    if len(set(values)) != len(values):
        raise ValueError('Prespecified model seeds must be distinct')
    return list(values)


def _lambdas(values):
    if not isinstance(values, list) or len(values) != len(CANDIDATES):
        raise ValueError('Routing comparison requires lambdas 10, 30 and 50')
    for value in values:
        lambda_key(value)
    if sorted(values) != list(CANDIDATES):
        raise ValueError('Routing comparison requires lambdas 10, 30 and 50 exactly once')
    return list(CANDIDATES)


def validate_routing_settings(settings):
    """Validate and return canonical routing fields without mutating settings."""
    routing = dict(settings.get('routing', {}))
    if set(routing) - {'unseen_grid_policy', 'stage'}:
        raise ValueError('Unknown routing configuration field')
    routing.setdefault('unseen_grid_policy', UNSEEN_GRID_POLICY)
    routing.setdefault('stage', 'finetuned')
    if routing != dict(unseen_grid_policy=UNSEEN_GRID_POLICY, stage='finetuned'):
        raise ValueError('Routing uses finetuned CNNs and lower_anchor_clamp only')
    return dict(lambdas=_lambdas(settings.get('lambdas')),
                model_seeds=_seeds(settings.get('model_seeds')),
                validation_sizes=_sizes(settings['data']['validation']['sizes']),
                routing=routing)


def validate_policy(policy):
    """Check an immutable inference policy independently of any case/results."""
    if not isinstance(policy, dict) or policy.get('version') != VERSION:
        raise ValueError('Unsupported grid routing policy')
    if policy.get('stage') != 'finetuned':
        raise ValueError('Grid routing selects finetuned CNNs only')
    if policy.get('unseen_grid_policy') != UNSEEN_GRID_POLICY:
        raise ValueError('Unseen grids must use the preregistered lower_anchor_clamp policy')
    if policy.get('granularity', GRANULARITY) != GRANULARITY:
        raise ValueError('Choose one CNN from the finest grid for the complete hierarchy')
    _arm(policy.get('arm'))
    _seeds(policy.get('model_seeds'))
    candidates = _lambdas(policy.get('candidates'))
    sizes = _sizes(policy.get('validation_sizes'))
    expected = {str(n) for n in sizes}
    mapping = policy.get('grid_to_lambda')
    eligible = policy.get('grid_eligible')
    if not isinstance(mapping, dict) or set(mapping) != expected:
        raise ValueError('Routing policy must map every declared validation grid exactly once')
    if (not isinstance(eligible, dict) or set(eligible) != expected or
            any(type(value) is not bool for value in eligible.values())):
        raise ValueError('Routing policy requires eligibility for every validation grid')
    for value in [*mapping.values(), policy.get('global_lambda')]:
        if type(value) not in (int, float) or value not in candidates:
            raise ValueError('Routing policy selects an undeclared lambda')
    if type(policy.get('global_eligible')) is not bool:
        raise ValueError('Routing policy requires global eligibility')
    result = deepcopy(policy)
    result['candidates'] = candidates
    result['validation_sizes'] = sizes
    result['grid_to_lambda'] = {str(n): float(mapping[str(n)]) for n in sizes}
    result['global_lambda'] = float(policy['global_lambda'])
    return result


def route_lambda(policy, n):
    """Choose once from the finest n; n >= maximum anchor reuses that anchor."""
    policy = validate_policy(policy)
    if type(n) is not int or n < 3:
        raise ValueError('Input finest grid size must be an integer >= 3')
    sizes = policy['validation_sizes']
    anchor = max((value for value in sizes if value <= n), default=sizes[0])
    return policy['grid_to_lambda'][str(anchor)]


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError('Invalid validation ' + name)
    return value


def _validate_summary(summary, settings, normalized):
    if summary.get('completed') is not True:
        raise ValueError('Complete full development validation before selecting routes')
    if summary.get('heldout_selection') or summary.get('scope') in ('final', 'ood', 'holdout'):
        raise ValueError('Routing selection accepts development validation only')
    cases = summary.get('case_results')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Routing requires per-operator development validation results')
    data = settings['data']
    families = data.get('families', list(FAMILIES))
    if (not isinstance(families, list) or not families or
            any(not isinstance(family, str) or family not in FAMILIES for family in families) or
            len(set(families)) != len(families)):
        raise ValueError('Distinct declared validation families are required')
    per_family = data['validation']['per_family']
    if type(per_family) is not int or per_family < 1:
        raise ValueError('Positive validation problems per grid/family are required')
    expected = len(normalized['validation_sizes']) * len(families) * per_family
    if (len(cases) != expected or summary.get('cases_expected') != expected or
            summary.get('cases_completed') != expected):
        raise ValueError('Routing requires the complete declared validation cohort')
    repetitions = settings['benchmark']['repetitions']
    if (type(repetitions) is not int or repetitions < 1 or
            summary.get('repetitions') != repetitions or summary.get('rhs_count') != 1):
        raise ValueError('Routing validation repetitions or RHS protocol differs')
    names = [method_name(strength, arm, seed)
             for strength in normalized['lambdas'] for arm in ARMS
             for seed in normalized['model_seeds']]
    counts = {(n, family): 0 for n in normalized['validation_sizes'] for family in families}
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('Invalid validation case record')
        case_id = case.get('case_id')
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError('Validation operator identifiers must be distinct')
        seen.add(case_id)
        for field in ('split', 'research_split'):
            if case.get(field, 'validation') != 'validation':
                raise ValueError('Routing selection cannot use TRAIN/final/OOD observations')
        shape = case.get('shape')
        if (not isinstance(shape, (list, tuple)) or len(shape) != 2 or
                any(type(n) is not int for n in shape) or shape[0] != shape[1] or
                (shape[0], case.get('family')) not in counts):
            raise ValueError('Validation grid/family differs from the declared cohort')
        counts[(shape[0], case['family'])] += 1
        rows = case.get('methods', {})
        if not isinstance(rows, dict) or any(name not in rows for name in names):
            raise ValueError('Validation omits a prespecified lambda/arm/seed')
        for name in names:
            row = rows[name]
            if not isinstance(row, dict):
                raise ValueError('Invalid validation method record')
            missing = _nonnegative_integer(row.get('missing'), 'missing count')
            failures = _nonnegative_integer(row.get('failures'), 'failure count')
            attempts = _nonnegative_integer(row.get('attempts'), 'attempt count')
            elapsed = row.get('warm_seconds')
            if missing or attempts != repetitions or failures > attempts:
                raise ValueError('Missing or incompatible validation measurements prohibit selection')
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError('Validation timing must be finite and positive')
    if any(count != per_family for count in counts.values()):
        raise ValueError('Validation is missing declared grid/family operators')
    # Selection is recomputed from case records; stale aggregate summaries are errors.
    totals = summary.get('methods', {})
    for name in names:
        rows = [case['methods'][name] for case in cases]
        total = totals.get(name)
        if (not isinstance(total, dict) or total.get('missing') != 0 or
                total.get('attempts') != sum(row['attempts'] for row in rows) or
                total.get('failures') != sum(row['failures'] for row in rows)):
            raise ValueError('Validation aggregate disagrees with per-operator evidence')
        elapsed = total.get('warm_seconds')
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or
                not math.isclose(elapsed, fmean(row['warm_seconds'] for row in rows),
                                 rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError('Validation aggregate timing disagrees with per-operator evidence')
    return cases


def _candidate(cases, strength, arm, seeds, allowed):
    per_seed = {}
    for seed in seeds:
        rows = [case['methods'][method_name(strength, arm, seed)] for case in cases]
        failures = sum(row['failures'] for row in rows)
        attempts = sum(row['attempts'] for row in rows)
        per_seed[str(seed)] = dict(failures=failures, attempts=attempts,
            failure_rate=failures / attempts,
            warm_seconds=fmean(row['warm_seconds'] for row in rows),
            eligible=failures / attempts <= allowed)
    values = list(per_seed.values())
    failures, attempts = sum(v['failures'] for v in values), sum(v['attempts'] for v in values)
    times = [v['warm_seconds'] for v in values]
    return dict(lambda_value=float(strength), independent_operators=len(cases),
        failures=failures, attempts=attempts, failure_rate=failures / attempts,
        warm_seconds=fmean(times), eligible=failures / attempts <= allowed,
        std_across_seed_means_seconds=stdev(times) if len(times) > 1 else None,
        per_seed=per_seed)


def _select(cases, arm, lambdas, seeds, allowed):
    candidates = {lambda_key(value): _candidate(cases, value, arm, seeds, allowed)
                  for value in lambdas}
    selected = min(lambdas, key=lambda value: (
        candidates[lambda_key(value)]['failure_rate'],
        candidates[lambda_key(value)]['warm_seconds'], value))
    per_seed = {}
    for seed in seeds:
        winner = min(lambdas, key=lambda value: (
            candidates[lambda_key(value)]['per_seed'][str(seed)]['failure_rate'],
            candidates[lambda_key(value)]['per_seed'][str(seed)]['warm_seconds'], value))
        per_seed[str(seed)] = float(winner)
    return dict(selected_lambda=float(selected), eligible=candidates[lambda_key(selected)]['eligible'],
        candidates=candidates, per_seed_selected_lambda=per_seed,
        unanimous_seed_winner=all(value == selected for value in per_seed.values()),
        selected_by='failure_rate_then_mean_penalized_warm_seconds_then_lower_lambda')


def select_routes(summary, settings):
    """Fit one shared grid rule per arm using only the complete validation split.

    Failed solves retain the benchmark's declared penalty. If every candidate
    violates the eligibility threshold, the least-failing candidate is retained
    with eligible=False; it is never replaced by a classical fallback. Evidence
    is selection data, not an unbiased estimate of the selected rule's gain.
    """
    normalized = validate_routing_settings(settings)
    allowed = settings['benchmark'].get('allowed_failure_rate', 0.)
    if type(allowed) not in (int, float) or not math.isfinite(allowed) or not 0 <= allowed <= 1:
        raise ValueError('Allowed failure rate must be finite and in [0, 1]')
    cases = _validate_summary(summary, settings, normalized)
    lambdas, seeds, sizes = (normalized[key] for key in ('lambdas', 'model_seeds', 'validation_sizes'))
    arms, evidence = {}, {}
    for arm in ARMS:
        global_result = _select(cases, arm, lambdas, seeds, allowed)
        grids = {str(n): _select([case for case in cases if case['shape'][0] == n],
                                arm, lambdas, seeds, allowed) for n in sizes}
        policy = dict(version=VERSION, arm=arm, stage='finetuned', candidates=lambdas,
            model_seeds=seeds, validation_sizes=sizes,
            unseen_grid_policy=UNSEEN_GRID_POLICY, granularity=GRANULARITY,
            grid_to_lambda={n: result['selected_lambda'] for n, result in grids.items()},
            global_lambda=global_result['selected_lambda'],
            grid_eligible={n: result['eligible'] for n, result in grids.items()},
            global_eligible=global_result['eligible'])
        arms[arm] = validate_policy(policy)
        evidence[arm] = dict(global_result=global_result, grids=grids)
    return dict(version=VERSION, scope='development_validation_only', stage='finetuned',
        lambdas=lambdas, model_seeds=seeds, validation_sizes=sizes,
        arms=arms, evidence=evidence, all_prespecified_seeds_retained=True,
        heldout_selection_permitted=False, selection_estimate_only=True,
        uncertainty='Across-seed diagnostics only; no claim that small timing differences are significant.',
        inference='Read finest n once; use one selected CNN for the whole learned hierarchy.')
