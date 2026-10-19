"""Unprotected cycle diagnostics and wall-time selection for joint P/B studies.

The fixed-cycle diagnostic never calls the safeguarded solver.  Selection uses
the measured complete setup-plus-solve workload, never a contraction proxy or
warm-only timing.  A failed operator stays in the declared cohort.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..hierarchy import classical_cycle
from ..provenance import json_safe, stable_norm, write_json
from .banks import Stats, hybrid_cycle
from .evaluation_metrics import _speed_summary
from .limited import forced_config
from .research_data import HOLDOUT, _check_claim
from .research_evaluation import manufactured_rhs
from .research_runtime import offline_teacher_generation
from .solver import PreparedAdaptiveMG
from .spatial import SpatialState

ARMS = ('fixed_C', 'em5_C', 'learned_P', 'learned_S', 'joint', 'distilled_joint')
LEARNED_ARMS = ARMS[2:]
_METRICS = ('executed_cycles', 'accepted_neural_cycles', 'fallback_count',
            'rollback_count', 'setup_failures', 'wall_seconds',
            'total_setup_seconds', 'solve_seconds', 'operator_generation_seconds')
_NUMERICAL_FAILURES = (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def raw_contractions(examples, arms, cfg, *, cycles, probes):
    """Run actual generated hierarchies for exactly K cycles, without fallback.

    ``arms`` follows ``evaluate_research``'s model/branch/config/rules schema.
    In particular, the independent EM5 reference uses its own config/rules.
    Numerical infeasibility is an explicit failed row; no classical substitute
    is allowed to stand in for a failed learned setup or cycle.
    """
    _positive_int(cycles, 'cycles')
    _positive_int(probes, 'probes')
    examples = list(examples)
    if not examples or not arms:
        raise ValueError('Raw contraction diagnostics need examples and arms')
    if len({e.name for e in examples}) != len(examples):
        raise ValueError('Raw contraction examples must have unique names')
    for name, arm in arms.items():
        if (arm.get('branch') not in {'C', 'H_P', 'H_S', 'H_SP'}
                or arm.get('selector', False) or arm.get('policy') is not None
                or arm.get('ttt') is not None):
            raise ValueError(f'{name}: raw diagnostic requires a fixed explicit branch')
    rows = []
    for example in examples:
        if getattr(example, 'research_split', None) in HOLDOUT:
            _check_claim(getattr(example, 'research_run_root', None),
                         getattr(example, 'research_final_claim', None))
            if any(arm.get('training_only') for arm in arms.values()):
                raise ValueError('Offline teachers cannot consume final/OOD data')
        rhs, _ = manufactured_rhs(example, probes)
        row = dict(case=example.name, n=example.n,
                   operator_digest=example.digest, arms={})
        for name, arm in arms.items():
            branch = arm['branch']
            acfg = forced_config(arm.get('config', cfg), branch)
            rules = arm.get('rules')
            if rules is not None:
                # A protocol mismatch is a caller error, not a model failure.
                rules.validate_config(acfg)
            model = arm.get('model')
            setup_stats, cycle_stats = Stats(), Stats()
            histories = []
            context = offline_teacher_generation() if arm.get('training_only') else nullcontext()
            try:
                with context:
                    prepared = PreparedAdaptiveMG(example.a, example.n, model, acfg)
                    root = prepared.ensure_branch(branch, setup_stats)
                    for b in rhs:
                        norm0 = stable_norm(b)
                        if not np.isfinite(norm0) or norm0 <= 0:
                            raise FloatingPointError('Invalid raw residual probe')
                        x, history = np.zeros_like(b), []
                        spatial = SpatialState(model, acfg) if branch != 'C' else None
                        histories.append(history)
                        for k in range(cycles):
                            x = (classical_cycle(root, x, b, acfg.mg, cycle_stats)
                                 if branch == 'C' else hybrid_cycle(
                                     root, x, b, acfg, cycle_stats, spatial, k, True))
                            ratio = float(stable_norm(b - example.a @ x) / norm0)
                            if not np.isfinite(ratio):
                                raise FloatingPointError('Nonfinite raw V-cycle residual')
                            history.append(ratio)
                final_ratios = [history[-1] for history in histories]
                contraction = float(np.exp(np.log(np.maximum(final_ratios, 1e-300)).mean()))
                row['arms'][name] = dict(valid=True, contraction=contraction,
                    per_cycle_factor=contraction ** (1. / cycles), histories=histories,
                    maximum_final_ratio=max(final_ratios), error=None)
            except _NUMERICAL_FAILURES as error:
                row['arms'][name] = dict(valid=False, contraction=None,
                    per_cycle_factor=None, histories=histories,
                    maximum_final_ratio=None,
                    error=f'{type(error).__name__}: {error}')
            row['arms'][name].update(fallback_allowed=False, requested_branch=branch,
                strategy=acfg.mg.strategy_name, setup_stats=asdict(setup_stats),
                cycle_stats=asdict(cycle_stats))
        rows.append(row)
    return json_safe(dict(fallback_allowed=False, cycles=cycles, probes=probes,
        scope='raw complete fixed-count V-cycles; setup/cycle failures are explicit; '
              'no solver safeguards, rollback, classical replacement, or tolerance stopping; '
              'contraction is final relative residual, not a wall-time ranking',
        rows=rows))


def _median(runs, metric):
    values = [run.get(metric) for run in runs]
    return (float(np.median(values)) if values and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and np.isfinite(value) for value in values) else None)


def _valid_runs(runs, repeats):
    return (len(runs) == repeats and all(run.get('successful') is True
        and isinstance(run.get('wall_seconds'), (float, int))
        and not isinstance(run.get('wall_seconds'), bool)
        and np.isfinite(run['wall_seconds']) and run['wall_seconds'] > 0
        for run in runs))


def _repeats(result):
    repeats = result.get('manifest', {}).get('repeats')
    _positive_int(repeats, 'result manifest repeats')
    return repeats


def _diagnostic_group(rows):
    paired = [row for row in rows if row['paired_success']]
    base_cycles = [row['reference']['executed_cycles'] for row in paired]
    candidate_cycles = [row['candidate']['executed_cycles'] for row in paired]
    known = [(base, candidate) for base, candidate in zip(base_cycles, candidate_cycles)
             if base is not None and candidate is not None]
    return dict(total_cases=len(rows), successful_cases=sum(row['candidate_success'] for row in rows),
        failed_cases=sum(not row['candidate_success'] for row in rows),
        reference_successful_cases=sum(row['reference_success'] for row in rows),
        reference_failed_cases=sum(not row['reference_success'] for row in rows),
        common_successful_cases=len(paired),
        excluded_failure_case_ids=[row['case'] for row in rows if not row['paired_success']],
        fewer_cycles=sum(candidate < base for base, candidate in known),
        equal_cycles=sum(candidate == base for base, candidate in known),
        more_cycles=sum(candidate > base for base, candidate in known),
        reference_cycle_sum=sum(base for base, _ in known),
        candidate_cycle_sum=sum(candidate for _, candidate in known),
        speedup_vs_fixed=_speed_summary([row['reference']['wall_seconds'] /
            row['candidate']['wall_seconds'] for row in paired]),
        paired_totals={side: {metric: (sum(row[side][metric] for row in paired)
            if paired and all(row[side][metric] is not None for row in paired) else None)
            for metric in _METRICS} for side in ('reference', 'candidate')},
        rows=rows)


def cycle_diagnostics(result):
    """Compare median counts/times against fixed_C, retaining failed rows.

    Counts and timings are per-operator medians over repetitions; workload
    counts already include every RHS.  Summaries pair only fully successful
    operators and explicitly display exclusions, including within each grid.
    """
    repeats = _repeats(result)
    report = {}
    for regime, counts in result['summary'].items():
        for count, arms in counts.items():
            for name in arms:
                if name == 'fixed_C':
                    continue
                rows = []
                for case in result['rows']:
                    base, runs = (case['runs'].get(key, {}).get(regime, {}).get(str(count), [])
                                  for key in ('fixed_C', name))
                    base_ok, candidate_ok = _valid_runs(base, repeats), _valid_runs(runs, repeats)
                    rows.append(dict(case=case['example']['name'], n=case['example']['case']['n'],
                        reference_success=base_ok, candidate_success=candidate_ok,
                        paired_success=base_ok and candidate_ok,
                        reference_successful_repeats=sum(run.get('successful') is True for run in base),
                        candidate_successful_repeats=sum(run.get('successful') is True for run in runs),
                        expected_repeats=repeats,
                        reference={metric: _median(base, metric) for metric in _METRICS},
                        candidate={metric: _median(runs, metric) for metric in _METRICS}))
                entry = _diagnostic_group(rows)
                entry['by_n'] = {str(n): _diagnostic_group([row for row in rows if row['n'] == n])
                                 for n in sorted({row['n'] for row in rows})}
                entry['scope'] = ('per-operator median repetitions; paired successful workload counts '
                                  'and times; rejected neural trials count as executed cycles; '
                                  'accepted neural, fallback, rollback and setup failures are distinct')
                report[f'{regime}/rhs{count}/{name}'] = entry
    return json_safe(report)


def report(result, target):
    """Write common failure-inclusive output plus joint-study cycle diagnostics."""
    from .unet_transfer_study import _report as common_report
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    common_report(result, target)
    diagnostics = cycle_diagnostics(result)
    write_json(target / 'cycle_diagnostics.json', diagnostics)
    for key, entry in diagnostics.items():
        print(key, 'cycles fewer/equal/more=',
              (entry['fewer_cycles'], entry['equal_cycles'], entry['more_cycles']),
              'failures=', entry['failed_cases'], flush=True)
    return diagnostics


def select_measured_candidate(result, *, rhs_count=1, regime='multiple',
                              candidate_arms=LEARNED_ARMS):
    """Select a learned arm by observed setup+solve time on the full cohort.

    Every repeat and RHS must succeed on every operator.  Candidate timing is
    the arithmetic mean of per-operator median wall times, matching equally
    weighted expected time.  Geometric paired speedups remain diagnostics.
    Warm measurements are secondary evidence and never determine selection.
    This helper is for validation only and refuses final/OOD-labelled rows.
    """
    _positive_int(rhs_count, 'rhs_count')
    if regime not in {'cold', 'multiple'} or (regime == 'cold' and rhs_count != 1):
        raise ValueError('Selection requires setup-inclusive cold RHS=1 or multiple workload')
    candidates = tuple(candidate_arms)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('Supply distinct learned candidate arms')
    if set(candidates) - set(LEARNED_ARMS):
        raise ValueError('Only learned joint-study arms may be candidates')
    rows, repeats = result.get('rows', []), _repeats(result)
    if not rows:
        raise ValueError('Selection requires a nonempty validation cohort')
    if any(row.get('research_split') in HOLDOUT for row in rows):
        raise ValueError('Final/OOD results cannot select an arm or checkpoint')
    if len({row['example']['name'] for row in rows}) != len(rows):
        raise ValueError('Selection cohort must have unique operators')
    if str(rhs_count) not in result.get('summary', {}).get(regime, {}):
        raise ValueError('Requested setup-inclusive workload was not measured')
    metadata = result.get('manifest', {}).get('arms', {})
    evidence = {}
    for name in ('fixed_C',) + candidates:
        case_rows, failures = [], []
        for row in rows:
            runs = row['runs'].get(name, {}).get(regime, {}).get(str(rhs_count), [])
            success = _valid_runs(runs, repeats)
            if any(run.get('setup_scope') not in (None, 'included_once') for run in runs):
                raise ValueError('Primary selection workload excludes setup')
            if not success:
                failures.append(row['example']['name'])
            case_rows.append(dict(case=row['example']['name'], n=row['example']['case']['n'],
                successful=success, observed_repeats=len(runs), expected_repeats=repeats,
                successful_repeats=sum(run.get('successful') is True for run in runs),
                median_wall_seconds=_median(runs, 'wall_seconds'),
                median_setup_seconds=_median(runs, 'total_setup_seconds'),
                median_solve_seconds=_median(runs, 'solve_seconds')))
        deployable = not any(metadata.get(name, {}).get(key) for key in
                             ('training_only', 'reference_only', 'robustness_only'))
        eligible = not failures and deployable
        mean_time = float(np.mean([case['median_wall_seconds'] for case in case_rows])) if eligible else None
        evidence[name] = dict(eligible=eligible, successes=len(rows) - len(failures),
            failures=len(failures), total=len(rows), failure_case_ids=failures,
            deployable=deployable, mean_case_median_wall_seconds=mean_time,
            model_signature=metadata.get(name, {}).get('model_signature'),
            rows=case_rows)
    eligible = [name for name in candidates if evidence[name]['eligible']]
    selected = min(eligible, key=lambda name: evidence[name]['mean_case_median_wall_seconds']) if eligible else None
    baseline = evidence['fixed_C']
    deployment = selected
    beats_fixed = None
    if baseline['eligible']:
        beats_fixed = bool(selected is not None and evidence[selected]['mean_case_median_wall_seconds']
                           < baseline['mean_case_median_wall_seconds'])
        if not beats_fixed:
            deployment = 'fixed_C'
    warm = {key: result.get('summary', {}).get(key, {}).get(str(rhs_count))
            for key in ('warm', 'warm_multiple')
            if str(rhs_count) in result.get('summary', {}).get(key, {})}
    return json_safe(dict(selected_arm=selected, recommended_deployment_arm=deployment,
        selected_model_signature=evidence[selected]['model_signature'] if selected else None,
        faster_than_fixed_on_target=beats_fixed, regime=regime, rhs_count=rhs_count,
        total_cases=len(rows), candidates=evidence, warm_secondary=warm,
        objective='arithmetic mean over operators of median repeated measured setup+solve wall seconds',
        failure_policy='all repeats and all RHS must succeed for every operator; failures invalidate '
                       'candidate eligibility and remain listed; no narrowed success-only ranking',
        scope='validation checkpoint/arm selection only; warm is secondary; no performance certificate',
        final_certificate=False))
