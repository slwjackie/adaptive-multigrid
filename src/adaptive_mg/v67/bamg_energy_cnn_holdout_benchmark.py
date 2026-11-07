"""Frozen final/OOD evaluation with bounded-memory, paired scalar measurements.

The caller must freeze model/teacher choices before materializing any cases.
This module never selects a model, seed, lambda, or stopping checkpoint from
held-out results. Confidence intervals resample operators, not RHS or repeats.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .bamg_energy import array_digest
from .bamg_energy_study import digest, file_digest, read_json, write_json
from .bamg_energy_cnn_benchmark import (
    BASELINES, _FAILURES, _build_bank, _failure, _measurement,
    _penalized_seconds, _settings as _validation_settings,
)
from .bamg_teacher import matrix_digest
from ..provenance import operator_digest

VERSION = 'bamg-energy-cnn-frozen-holdout-benchmark-v1'
RHS_KINDS = ('manufactured', 'random', 'localized')
SPLITS = ('final', 'ood_grid', 'ood_rotation', 'ood_anisotropy', 'ood_contrast', 'ood_topology')


def _settings(settings):
    values = dict(settings.get('holdout_benchmark', settings['benchmark']))
    bench = _validation_settings({**settings, 'benchmark': values})
    kinds = bench.get('rhs_kinds', list(RHS_KINDS))
    if list(kinds) != list(RHS_KINDS):
        raise ValueError('Holdout uses manufactured, random and localized RHS in declared order')
    bench['rhs_kinds'] = list(RHS_KINDS)
    bench.setdefault('bootstrap_resamples', 2000)
    bench.setdefault('bootstrap_seed', 2026110791)
    if type(bench['bootstrap_resamples']) is not int or bench['bootstrap_resamples'] < 100:
        raise ValueError('At least 100 operator bootstrap resamples are required')
    if type(bench['bootstrap_seed']) is not int or bench['bootstrap_seed'] < 0:
        raise ValueError('bootstrap_seed must be nonnegative')
    return bench


def _case_path(output, case_id):
    return Path(output) / 'benchmark_cases' / (hashlib.sha256(case_id.encode()).hexdigest() + '.json')


def _manifest(cases, settings, checkpoints, identity):
    bench = _settings(settings)
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Expected distinct planned holdout cases')
    for case in cases:
        if (not isinstance(case.get('case_id'), str) or not case['case_id']
                or case.get('split') not in SPLITS or type(case.get('n')) is not int
                or case['n'] < 3 or not isinstance(case.get('family'), str)):
            raise ValueError('Invalid planned final/OOD case')
    frozen = identity.get('freeze_sha256', '')
    if not isinstance(frozen, str) or len(frozen) != 64 or any(c not in '0123456789abcdef' for c in frozen):
        raise ValueError('Holdout benchmark requires the caller-verified freeze SHA256')
    if any(not isinstance(name, str) or not name or name in BASELINES for name in checkpoints):
        raise ValueError('Invalid or reserved checkpoint method name')
    hashes = {name: file_digest(path) for name, path in sorted(checkpoints.items())}
    seeds = settings.get('model_seeds', [])
    if len(set(seeds)) != len(seeds):
        raise ValueError('Repeated model seeds in heldout comparison')
    model_groups = {}
    for arm in ('scratch', 'warm'):
        for stage in ('supervised', 'finetuned'):
            names = [f'{arm}_seed{seed}_{stage}' for seed in seeds]
            present = [name in checkpoints for name in names]
            if any(present) and not all(present):
                raise ValueError('Heldout cohort omits a prespecified seed')
            if names and all(present):
                model_groups[f'{arm}_{stage}'] = names
    value = dict(version=VERSION, identity=identity, settings_digest=digest(settings),
        cases=cases, methods=[*BASELINES, *sorted(checkpoints)], checkpoint_hashes=hashes,
        model_seeds=seeds, model_groups=model_groups,
        repetitions=bench['repetitions'], rhs_kinds=bench['rhs_kinds'],
        warmup_cycles=bench['warmup_cycles'], timeout_seconds=bench['timeout_seconds'],
        failure_penalty_seconds=bench['failure_penalty_seconds'],
        bootstrap_resamples=bench['bootstrap_resamples'], bootstrap_seed=bench['bootstrap_seed'],
        setup_excluded=True, fallback=False, initial_iterate='zero',
        heldout_selection=False,
        aggregation='mean RHS times per repeat; median repeats per operator; arithmetic mean operators',
        uncertainty='paired stratified operator bootstrap; RHS/repeats/model seeds are not independent samples')
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _keys(manifest):
    return [f'{repeat}:{kind}' for repeat in range(manifest['repetitions']) for kind in manifest['rhs_kinds']]


def _read_case(output, case, manifest, manifest_digest=None):
    manifest_digest = digest(manifest) if manifest_digest is None else manifest_digest
    path = _case_path(output, case['case_id'])
    if not path.exists():
        return dict(case=case, manifest_digest=manifest_digest, actual=None,
                    setups={}, measurements={})
    state = read_json(path)
    if state.get('case') != case or state.get('manifest_digest') != manifest_digest:
        raise ValueError('Saved heldout case identity changed')
    expected = set(_keys(manifest))
    if state['measurements'] and state.get('actual') is None:
        raise ValueError('Heldout measurements have no operator/RHS identity')
    for method, rows in state['measurements'].items():
        if method not in manifest['methods'] or set(rows) - expected:
            raise ValueError('Unexpected heldout method/repetition/RHS')
        for row in rows.values():
            if type(row.get('success')) is not bool or row.get('fallback_used'):
                raise ValueError('Invalid heldout result or forbidden fallback')
            if row['success']:
                for field in ('warm_seconds', 'tolerance', 'final_residual'):
                    v = row.get(field)
                    if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                        raise ValueError('Invalid successful heldout measurement')
                if (row['warm_seconds'] <= 0 or row['final_residual'] > row['tolerance']
                        or type(row.get('cycles')) is not int or row['cycles'] < 0):
                    raise ValueError('Inconsistent successful heldout measurement')
    return state


def _complete(state, manifest):
    keys = set(_keys(manifest))
    return all(set(state['measurements'].get(method, {})) == keys for method in manifest['methods'])


def _actual(example, case):
    for key in ('case_id', 'family', 'split'):
        if example.get(key) != case[key]:
            raise ValueError('Materialized holdout case differs: ' + key)
    shape = tuple(example['shape'])
    if shape != (case['n'], case['n']) or example['A'].shape != (case['n'] ** 2,) * 2:
        raise ValueError('Materialized holdout grid differs')
    rhs = example.get('rhs', {})
    if set(rhs) != set(RHS_KINDS):
        raise ValueError('Materialized holdout case must provide all three RHS kinds')
    fingerprints = {}
    for kind in RHS_KINDS:
        b = np.asarray(rhs[kind], dtype=np.float64)
        if b.shape != (case['n'] ** 2,) or not np.isfinite(b).all() or not np.any(b):
            raise ValueError('Invalid heldout RHS: ' + kind)
        fingerprints[kind] = array_digest(b)
    normalized = operator_digest(example['A'], scale_invariant=True)
    if example.get('normalized_operator_digest') != normalized:
        raise ValueError('Materialized holdout normalized operator digest differs')
    return dict(A_digest=matrix_digest(example['A']),
                normalized_operator_digest=normalized,
                data_record_sha256=example.get('data_record_sha256'),
                rhs_digests=fingerprints)


def _case_summary(state, manifest):
    case = state['case']
    summary = {key: case[key] for key in ('case_id', 'split', 'n', 'family')}
    summary['methods'] = {}
    repetitions, kinds = manifest['repetitions'], manifest['rhs_kinds']
    for method in manifest['methods']:
        indexed = state['measurements'].get(method, {})
        rows = [indexed.get(key) for key in _keys(manifest)]
        missing = sum(row is None for row in rows)
        observed_failures = sum(row is not None and not row['success'] for row in rows)
        failures = missing + observed_failures
        repeat_times = [float(np.mean([_penalized_seconds(indexed.get(f'{r}:{kind}'),
                                      manifest['failure_penalty_seconds']) for kind in kinds]))
                        for r in range(repetitions)]
        cycles = None if failures else float(np.median([
            np.mean([indexed[f'{r}:{kind}']['cycles'] for kind in kinds]) for r in range(repetitions)]))
        per_rhs = {kind: dict(
            warm_seconds=float(np.median([_penalized_seconds(indexed.get(f'{r}:{kind}'),
                                         manifest['failure_penalty_seconds']) for r in range(repetitions)])),
            failures=sum(indexed.get(f'{r}:{kind}') is None or not indexed[f'{r}:{kind}']['success']
                         for r in range(repetitions))) for kind in kinds}
        summary['methods'][method] = dict(warm_seconds=float(np.median(repeat_times)),
            cycles=cycles, attempts=len(rows), observed=len(rows) - missing,
            missing=missing, failures=failures, observed_failures=observed_failures,
            rhs=per_rhs)
    summary['cohorts'] = {}
    for name, members in manifest.get('model_groups', {}).items():
        values = [summary['methods'][member] for member in members]
        totals = {key: sum(v[key] for v in values)
                  for key in ('attempts', 'observed', 'missing', 'failures', 'observed_failures')}
        totals.update(warm_seconds=float(np.mean([v['warm_seconds'] for v in values])),
            cycles=float(np.mean([v['cycles'] for v in values])) if not totals['failures'] else None,
            model_seeds=manifest['model_seeds'], seed_count=len(members), members=members)
        summary['cohorts'][name] = totals
    return summary


def _paired_interval(cases, method, manifest, group, reference='fixed_C'):
    """Resample independent A's within cells, keeping all RHS/repeats together."""
    if len(cases) < 2:
        return None
    strata = {}
    for i, case in enumerate(cases):
        strata.setdefault((case['split'], case['n'], case['family']), []).append(i)
    # A one-A cell contains no within-cell information about sampling variability.
    if any(len(indices) < 2 for indices in strata.values()):
        return None
    fixed = np.array([c['methods'][reference]['warm_seconds'] for c in cases])
    candidate = np.array([c['methods'][method]['warm_seconds'] for c in cases])
    seed = int(digest([manifest['bootstrap_seed'], group])[:16], 16)
    rng = np.random.default_rng(seed)
    ratios = np.empty(manifest['bootstrap_resamples'])
    differences = np.empty_like(ratios)
    # Work in small batches to keep memory independent of the large full suite.
    for start in range(0, len(ratios), 64):
        stop = min(start + 64, len(ratios))
        indices = np.concatenate([rng.choice(group_indices, size=(stop - start, len(group_indices)), replace=True)
                                  for group_indices in strata.values()], axis=1)
        f, c = fixed[indices].mean(1), candidate[indices].mean(1)
        ratios[start:stop] = f / c
        differences[start:stop] = c - f
    interval = list(map(float, np.quantile(ratios, [.025, .975])))
    result = dict(level=.95, resamples=len(ratios), independent_operators=len(cases),
        reference=reference, candidate=method,
        time_ratio_reference_over_candidate=interval,
        time_difference_seconds=list(map(float, np.quantile(differences, [.025, .975]))),
        sampling_unit='operator; paired methods; stratified by split/grid/family',
        includes_training_seed_uncertainty=False, includes_holdout_tuning_correction=False)
    if reference == 'fixed_C':
        result['speedup_vs_fixed'] = interval
    return result


def _cohort_summary(cases, manifest, group, *, intervals):
    names = list(manifest.get('model_groups', {}))
    if not names:
        return dict(cohorts={}, cohort_comparisons={})
    cohort_cases = [{**case, 'methods': {'fixed_C': case['methods']['fixed_C'], **case['cohorts']}}
                    for case in cases]
    cohort_manifest = {**manifest, 'methods': ['fixed_C', *names]}
    totals = _aggregate(cohort_cases, cohort_manifest, group + '/cohorts', intervals=intervals)
    cohorts = {name: dict(totals['methods'][name], seed_count=len(manifest['model_groups'][name]),
                         model_seeds=manifest['model_seeds'], members=manifest['model_groups'][name],
                         aggregation='average prespecified seed times within each A, then average A',
                         ensemble=False, fastest_seed_selection=False)
               for name in names}
    for name, cohort in cohorts.items():
        per_seed = {member: float(np.mean([case['methods'][member]['warm_seconds'] for case in cases]))
                    for member in manifest['model_groups'][name]}
        cohort['seed_mean_warm_seconds'] = per_seed
        cohort['std_across_seed_means_seconds'] = float(np.std(list(per_seed.values()), ddof=1)) if len(per_seed) > 1 else None
    comparisons = {}
    for stage in ('supervised', 'finetuned'):
        scratch, warm = f'scratch_{stage}', f'warm_{stage}'
        if scratch not in cohorts or warm not in cohorts:
            continue
        s, w = cohorts[scratch], cohorts[warm]
        completed = s['missing'] == w['missing'] == 0
        reliable = completed and s['failures'] == w['failures'] == 0
        comparisons[stage] = dict(completed=completed, independent_operators=len(cases),
            seed_count=len(manifest['model_seeds']),
            scratch_over_warm_time_ratio=s['warm_seconds'] / w['warm_seconds'] if reliable else None,
            penalized_scratch_over_warm_time_ratio=s['warm_seconds'] / w['warm_seconds'] if completed else None,
            paired_ci95=_paired_interval(cohort_cases, warm, manifest,
                group + '/scratch-vs-warm/' + stage, reference=scratch) if reliable and intervals else None,
            ratio_interpretation='greater than one means warm start is faster',
            uncertainty='conditional on these prespecified seeds; bootstrap resamples A, never seed/RHS/repeat rows')
    return dict(cohorts=cohorts, cohort_comparisons=comparisons)


def _aggregate(cases, manifest, group, *, intervals):
    methods = {}
    for method in manifest['methods']:
        values = [c['methods'][method] for c in cases]
        totals = {key: sum(v[key] for v in values)
                  for key in ('attempts', 'observed', 'missing', 'failures', 'observed_failures')}
        totals.update(independent_operators=len(cases),
            distinct_failed_operators=sum(v['observed_failures'] > 0 for v in values),
            operators_with_missing=sum(v['missing'] > 0 for v in values),
            warm_seconds=float(np.mean([v['warm_seconds'] for v in values])),
            cycles=float(np.mean([v['cycles'] for v in values])) if not totals['failures'] else None)
        totals['warm_ms'] = 1000 * totals['warm_seconds']
        totals['failure_rate'] = totals['failures'] / totals['attempts']
        methods[method] = totals
    completed = all(v['missing'] == 0 for v in methods.values())
    fixed = methods['fixed_C']
    for method, totals in methods.items():
        count, failed = len(cases), totals['distinct_failed_operators']
        fraction, z = failed / count, 1.959963984540054
        center = (fraction + z*z/(2*count)) / (1 + z*z/count)
        radius = z * math.sqrt(fraction*(1-fraction)/count + z*z/(4*count*count)) / (1 + z*z/count)
        totals['operator_failure_rate'] = fraction if completed else None
        totals['operator_failure_wilson_ci95'] = [max(0., center-radius), min(1., center+radius)] if completed else None
        totals['operator_failure_interval_note'] = 'Descriptive Wilson approximation for the declared stratified case mix, not a guaranteed reliability bound.'
        # Time ratios for failures are penalty-sensitive and are not speedup claims.
        reliable = completed and fixed['failures'] == totals['failures'] == 0
        totals['speedup_vs_fixed'] = fixed['warm_seconds'] / totals['warm_seconds'] if reliable else None
        totals['penalized_time_ratio_vs_fixed'] = fixed['warm_seconds'] / totals['warm_seconds'] if completed else None
        totals['paired_ci95'] = _paired_interval(cases, method, manifest, group) if reliable and intervals else None
    return dict(completed=completed, cases_expected=len(cases),
        cases_completed=sum(all(v['missing'] == 0 for v in c['methods'].values()) for c in cases),
        methods=methods)


def summarize(output, *, intervals=True):
    """Recompute partial or complete reports without loading an operator."""
    output = Path(output)
    manifest = read_json(output / 'benchmark_manifest.json')
    fingerprint = digest(manifest)
    states = [_read_case(output, case, manifest, fingerprint) for case in manifest['cases']]
    normalized = [state['actual']['normalized_operator_digest'] for state in states if state['actual'] is not None]
    if len(normalized) != len(set(normalized)):
        raise ValueError('Heldout benchmark repeats normalized operators')
    cases = [_case_summary(state, manifest) for state in states]
    overall = _aggregate(cases, manifest, 'all', intervals=intervals)
    overall.update(_cohort_summary(cases, manifest, 'all', intervals=intervals))
    groups = {}
    for case in cases:
        for name in (case['split'], f'n{case["n"]}', case['family'],
                     f'{case["split"]}/n{case["n"]}', f'{case["split"]}/{case["family"]}',
                     f'{case["split"]}/n{case["n"]}/{case["family"]}'):
            groups.setdefault(name, []).append(case)
    return dict(version=VERSION, **overall,
        groups={name: dict(**_aggregate(values, manifest, name, intervals=intervals),
                           **_cohort_summary(values, manifest, name, intervals=intervals))
                for name, values in groups.items()},
        case_results=cases, heldout_selection=False, selected_method=None,
        repetitions=manifest['repetitions'], rhs_count=len(manifest['rhs_kinds']),
        rhs_kinds=manifest['rhs_kinds'], aggregation=manifest['aggregation'],
        uncertainty=manifest['uncertainty'], setup_excluded=True, fallback=False,
        interpretation='Predeclared synthetic holdout distributions; no universal-generalization guarantee. '
                       'Global averages mix distinct shifts; inspect split/grid/family groups.')


def benchmark(output, *, cases, load_example, settings, checkpoints, identity,
              resume=False, max_cases=None):
    """Evaluate one operator and one hierarchy at a time; preserve saved solves.

    ``load_example`` is called only for cases with unfinished observations. Root
    code must first validate its frozen metadata cache, including completed A's.
    ``max_cases`` bounds newly finished operators, never changes the planned set.
    """
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError('max_cases must be a positive integer')
    cases = list(cases)
    output = Path(output)
    manifest = _manifest(cases, settings, checkpoints, identity)
    path = output / 'benchmark_manifest.json'
    if path.exists():
        if not resume:
            raise FileExistsError('Holdout benchmark exists; use --resume')
        if read_json(path) != manifest:
            raise ValueError('Frozen holdout settings/cases/checkpoints/identity changed')
    else:
        if (output / 'benchmark_cases').exists():
            raise ValueError('Holdout results exist without their identity manifest')
        write_json(path, manifest)
    bench, processed = _settings(settings), 0
    fingerprint = digest(manifest)
    for number, case in enumerate(manifest['cases'], 1):
        state = _read_case(output, case, manifest, fingerprint)
        if _complete(state, manifest):
            continue
        example = load_example(case)
        actual = _actual(example, case)
        if state['actual'] is not None and state['actual'] != actual:
            raise ValueError('Heldout operator or RHS changed on resume')
        state['actual'] = actual
        write_json(_case_path(output, case['case_id']), state)
        for repeat in range(bench['repetitions']):
            seed = int(digest([settings.get('seed', 0), case['case_id'], repeat])[:16], 16)
            rng = np.random.default_rng(seed)
            for method in rng.permutation(manifest['methods']).tolist():
                rows = state['measurements'].setdefault(method, {})
                pending = [kind for kind in bench['rhs_kinds'] if f'{repeat}:{kind}' not in rows]
                if not pending:
                    continue
                if method in checkpoints and file_digest(checkpoints[method]) != manifest['checkpoint_hashes'][method]:
                    raise ValueError('Frozen checkpoint changed while evaluating')
                print(f'Holdout {number}/{len(cases)} {case["case_id"]} {method} '
                      f'repeat {repeat + 1}/{bench["repetitions"]}', flush=True)
                setup = state['setups'].setdefault(method, {}).get(str(repeat))
                bank = None
                if setup is None or setup['success']:
                    started = perf_counter()
                    try:
                        with torch.no_grad():
                            bank = _build_bank(example, method, checkpoints.get(method), settings)
                            for kind in bench['rhs_kinds']:
                                rhs = np.asarray(example['rhs'][kind], dtype=np.float64)
                                for _ in range(bench['warmup_cycles']):
                                    if not np.isfinite(bank.cycle(np.zeros_like(rhs), rhs)).all():
                                        raise FloatingPointError('Nonfinite holdout warmup cycle')
                        setup = dict(success=True, setup_seconds=perf_counter() - started,
                                     reason='prepared', fallback_used=False)
                    except _FAILURES as exc:
                        setup = dict(success=False, setup_seconds=perf_counter() - started,
                                     reason=f'{type(exc).__name__}: {exc}'[:1000], fallback_used=False)
                    state['setups'][method][str(repeat)] = setup
                for kind in rng.permutation(pending).tolist():
                    rows[f'{repeat}:{kind}'] = (_measurement(bank, np.asarray(example['rhs'][kind], dtype=np.float64), bench)
                                               if setup['success'] else _failure('setup_failed: ' + setup['reason']))
                    write_json(_case_path(output, case['case_id']), state)
                # Do not retain all CNN/numerical hierarchies of a large n=255 A.
                del bank
        del example
        processed += 1
        if max_cases is not None and processed >= max_cases:
            break
    result = summarize(output)
    write_json(output / 'benchmark_summary.json', result)
    return result


def print_summary(summary, group=None):
    value = summary if group is None else summary['groups'][group]
    print(f'\nFrozen holdout {group or "all"}: {value["cases_completed"]}/{value["cases_expected"]} operators')
    print(f'{"Method":40s} {"Warm ms*":>12s} {"Fail/attempts":>17s} {"Failed A":>10s} {"Speedup":>9s}')
    for method, row in value['methods'].items():
        speed = '--' if row['speedup_vs_fixed'] is None else f'{row["speedup_vs_fixed"]:.3f}'
        print(f'{method:40s} {row["warm_ms"]:12.3f} '
              f'{row["failures"]:6d}/{row["attempts"]:<10d} '
              f'{row["distinct_failed_operators"]:10d} {speed:>9s}')
    for name, cohort in value.get('cohorts', {}).items():
        print(f'All-seed cohort {name}: {cohort["warm_ms"]:.3f} ms; '
              f'{cohort["failures"]}/{cohort["attempts"]} failed solves; '
              f'{cohort["distinct_failed_operators"]}/{cohort["independent_operators"]} failed A; '
              f'seeds={cohort["seed_count"]}')
    for stage, pair in value.get('cohort_comparisons', {}).items():
        ratio = pair['scratch_over_warm_time_ratio']
        if ratio is not None:
            interval = pair['paired_ci95']
            detail = '' if interval is None else f'; CI95={interval["time_ratio_reference_over_candidate"]}'
            print(f'Paired {stage}: scratch/warm time={ratio:.3f}{detail}')
    print('* Missing/failed solves penalized; setup excluded; no fallback; no heldout model selection.')
    print('CI95 uses independent operators within split/grid/family cells, not RHS/repetitions.')
