"""Failure-aware stage promotion. This module never trains or certifies a model.

``evaluate_stage_acceptance`` also accepts the original validation/raw.json rows,
so retained measurements can be diagnosed without rewriting the original run.
"""
from __future__ import annotations
import json
import math
import numpy as np
import torch


STAGE_ACCEPTANCE_DEFAULTS = {
    'stage_acceptance_policy': 'stage_relative_v2',
    'stage_min_common_success': 5,
    'stage_min_common_fraction': .5,
}


def component_changes(candidate, incumbent, stage=None, cfg=None):
    """Compare parameter VALUES; metadata, buffers and signature text cannot win."""
    names = {'smoother': ('smoother',), 'transfer': ('transfer',),
             'joint': ('smoother', 'transfer'),
             'hierarchical': ('smoother', 'transfer', 'detector')}.get(stage)
    if names is None:
        names = tuple(n for n in ('smoother', 'transfer')
                      if cfg is None or getattr(cfg, 'use_' + n))
    before = incumbent.component_signatures()
    after = candidate.component_signatures()
    changes = {}
    for name in ('smoother', 'transfer', 'detector', 'controller'):
        old = dict(getattr(incumbent, name).named_parameters())
        new = dict(getattr(candidate, name).named_parameters())
        changed = sorted(k for k in old.keys() | new.keys()
                         if k not in old or k not in new
                         or old[k].shape != new[k].shape
                         or not torch.equal(old[k].detach().cpu(), new[k].detach().cpu()))
        changes[name] = {'incumbent': before[name], 'candidate': after[name],
                         'parameters_changed': bool(changed),
                         'changed_parameter_names': changed,
                         'trained_by_stage': name in names}
    return changes


def _config_from_run(run, cfg):
    # Original v6.7 embeds its complete solve scope after the cache signatures.
    if isinstance(run.get('measurement_config'), dict):
        return run['measurement_config'], 'recorded_measurement_config'
    cache_key = run.get('cache_key', '')
    start = cache_key.find('{')
    if start >= 0:
        try:
            value = json.loads(cache_key[start:])
            if isinstance(value.get('mg'), dict):
                return value, 'original_cache_key_config'
        except (ValueError, TypeError):
            pass
    return cfg.to_dict(), 'caller_supplied_shared_config'


def _protocol(run, cfg):
    value, source = _config_from_run(run, cfg)
    # Ignore trace verbosity and deployment mode in old scope serialization.
    value = dict(value)
    value.pop('record_trace', None)
    value.pop('mode', None)
    if value.get('branch') == 'auto':
        value.pop('branch')
    mg = dict(value['mg'])
    mg.pop('verbose', None)
    value['mg'] = mg
    cache_key=run.get('cache_key','')
    digest=run.get('operator_digest')
    if digest is None and ':{' in cache_key:
        digest=cache_key.split(':',1)[0]  # legacy format only; modern key is opaque
    return (json.dumps(value, sort_keys=True), run.get('time_scope', 'cold'), digest), source


def _threshold_matches_config(run, cfg):
    try:
        threshold=float(run['stopping_threshold'])
        history=run['residual_history']
        reference=float(history[0])
        if 'residual_reference_norm' in run:
            reference=float(run['residual_reference_norm'])
        elif cfg.mg.residual_reference != 'initial':
            relative=float(run['relative_residual_history'][0])
            if relative > 0:reference /= relative
            elif reference != 0:return False
        expected=max(cfg.mg.absolute_tolerance,cfg.mg.tolerance*reference)
        return math.isfinite(reference) and math.isclose(threshold,expected,rel_tol=1e-12,abs_tol=0)
    except (KeyError,IndexError,TypeError,ValueError,ZeroDivisionError):
        return False


def _successful(run, cfg):
    """Every repeat must actually meet the recorded true-residual threshold."""
    try:
        threshold = float(run['stopping_threshold'])
        residual = float(run['final_true_residual'])
        attempts = int(run['executed_cycles'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(run.get('converged') and _threshold_matches_config(run,cfg) and math.isfinite(residual)
                and math.isfinite(threshold) and threshold >= 0
                and 0 <= residual <= threshold
                and 0 <= attempts <= cfg.mg.max_cycles)


_COUNTERS = ('neural_setup_calls', 'smoother_nn_calls', 'transfer_nn_calls',
             'neural_apply_calls', 'learned_transfer_apply_calls',
             'attempted_neural_cycles', 'accepted_neural_cycles',
             'rejected_neural_cycles', 'classical_recovery_cycles',
             'rollback_count', 'controller_calls', 'controller_nn_calls',
             'detector_calls', 'nn_setup_seconds', 'neural_apply_seconds',
             'controller_seconds', 'detector_seconds', 'setup_failures')


def _counts(runs):
    # Constructor/setup and solve stats are disjoint in v6.7 SolveResult.
    return {key: (float if key.endswith('_seconds') else int)(sum(float(run.get(section, {}).get(key, 0))
                     for run in runs for section in ('stats', 'setup_stats'))
                )
            for key in _COUNTERS}


def _numerical_use(run, cfg):
    stats = _counts([run])
    smoother = cfg.use_smoother and stats['neural_apply_calls'] > 0
    transfer = cfg.use_transfer and stats['learned_transfer_apply_calls'] > 0
    # Legacy raws lack a dedicated learned-P application counter. Their
    # coarse_path_count counts the actual learned-bank coarse correction;
    # transfer_nn_calls proves it contains generated P, not classical P.
    old = run.get('stats', {})
    if 'learned_transfer_apply_calls' not in old:
        transfer = bool(cfg.use_transfer and stats['transfer_nn_calls'] > 0
                        and old.get('coarse_path_count', 0) > 0)
    return bool(smoother or transfer)


def evaluate_stage_acceptance(rows, candidate, incumbent, cfg, settings=None, *,
                              stage=None, incumbent_arm='previous', candidate_arm='candidate'):
    """Return a reviewable decision from paired raw rows, without file writes.

    ``strict_v1`` reproduces the original six checks, including its full-model
    signature and all-common-success requirement. ``stage_relative_v2`` uses
    stage parameter values, case identities, a sample floor, recorded residuals,
    matching protocols, and numerical neural use on the common-success cases.
    Neither policy changes or grants the final independent certificate.
    """
    settings = dict(STAGE_ACCEPTANCE_DEFAULTS, **(settings or {}))
    policy = settings['stage_acceptance_policy']
    if policy not in {'strict_v1', 'stage_relative_v2'}:
        raise ValueError('stage_acceptance_policy must be strict_v1 or stage_relative_v2')
    count = int(settings['stage_min_common_success'])
    fraction = float(settings['stage_min_common_fraction'])
    margin = float(settings.get('stage_time_margin', .03))
    if count < 1 or not 0 < fraction <= 1 or not 0 <= margin < 1:
        raise ValueError('invalid stage sample floor, fraction or time margin')
    expected = int(settings.get('validation_repeats', 3))
    if expected < 1:
        raise ValueError('validation_repeats must be positive')
    ids = [str(row.get('case_id', row.get('example', {}).get('name', ''))) for row in rows]
    if any(not name for name in ids) or len(set(ids)) != len(ids):
        raise ValueError('stage cases require unique nonempty case identities')
    previous_success, candidate_success = set(), set()
    ratios = []
    details = []
    all_previous, all_candidate = [], []
    protocol_ok = repeat_ok = timing_ok = True
    protocol_sources = set()
    neural_common = legacy_used = 0
    for name, row in zip(ids, rows):
        previous = row['runs'][incumbent_arm]
        new = row['runs'][candidate_arm]
        all_previous.extend(previous)
        all_candidate.extend(new)
        complete = len(previous) == len(new) == expected
        repeat_ok = repeat_ok and complete
        signatures = []
        thresholds = []
        for run in previous + new:
            signature, source = _protocol(run, cfg)
            signatures.append(signature)
            protocol_sources.add(source)
            try:
                thresholds.append(float(run['stopping_threshold']))
            except (KeyError, TypeError, ValueError):
                thresholds.append(float('nan'))
        same = bool(signatures and len(set(signatures)) == 1 and thresholds
                    and all(_threshold_matches_config(run,cfg) for run in previous + new)
                    and all(math.isfinite(x) and x >= 0 for x in thresholds)
                    and all(math.isclose(x, thresholds[0], rel_tol=1e-12, abs_tol=0)
                            for x in thresholds))
        protocol_ok = protocol_ok and same
        predicate = (lambda run: bool(run.get('converged'))) if policy == 'strict_v1' else lambda run: _successful(run, cfg)
        ps = bool(previous) and all(predicate(run) for run in previous)
        ns = bool(new) and all(predicate(run) for run in new)
        if policy != 'strict_v1':
            ps, ns = bool(ps and complete), bool(ns and complete)
        if ps:
            previous_success.add(name)
        if ns:
            candidate_success.add(name)
        used = any(run.get('stats', {}).get('accepted_neural_cycles', 0) > 0 for run in new)
        legacy_used += int(used)
        tc = tn = speedup = None
        if ps and ns:
            times = [float(run.get('wall_seconds', float('nan'))) for run in previous + new]
            valid_time = all(math.isfinite(t) and t > 0 for t in times)
            timing_ok = timing_ok and valid_time
            if valid_time:
                tc = float(np.median([run['wall_seconds'] for run in previous]))
                tn = float(np.median([run['wall_seconds'] for run in new]))
                speedup = tc / tn
                if math.isfinite(speedup) and speedup > 0:
                    ratios.append(speedup)
                else:
                    speedup=None
                    timing_ok=False
            neural_common += int(any(_numerical_use(run, cfg) for run in new))
        details.append({'case_id': name, 'incumbent_success': ps, 'candidate_success': ns,
                        'incumbent_successful_repeats': sum(_successful(run, cfg) for run in previous),
                        'candidate_successful_repeats': sum(_successful(run, cfg) for run in new),
                        'repeat_counts': {'incumbent': len(previous), 'candidate': len(new)},
                        'same_protocol': same, 'incumbent_successful_seconds': tc,
                        'candidate_successful_seconds': tn, 'speedup_vs_incumbent': speedup})
    common = previous_success & candidate_success
    lost = previous_success - candidate_success
    rescued = candidate_success - previous_success
    failed = set(ids) - (previous_success | candidate_success)
    gm = float(np.exp(np.log(ratios).mean())) if ratios else None
    ci = [None, None]
    if ratios:
        logs = np.log(ratios)
        rng = np.random.default_rng(93)
        boots = np.exp(rng.choice(logs, (2000, len(logs)), replace=True).mean(axis=1))
        ci = np.quantile(boots, [.025, .975]).tolist()
    required = 1 / (1 - margin)
    changes = component_changes(candidate, incumbent, stage, cfg)
    weight_changed = any(value['trained_by_stage'] and value['parameters_changed']
                         for value in changes.values())
    minimum = max(count, math.ceil(fraction * len(rows)))
    checks = {'model_weights_changed': weight_changed, 'no_new_failures': not lost,
              'minimum_common_successes': len(common) >= minimum,
              'speedup_threshold': gm is not None and gm > required,
              'actual_neural_use': neural_common > 0, 'same_measurement_protocol': protocol_ok,
              'complete_repeats': repeat_ok, 'valid_successful_timings': timing_ok and bool(ratios)}
    legacy_checks = {'model_signature_changed': candidate.signature() != incumbent.signature(),
                     'actual_neural_use': legacy_used > 0, 'no_new_failures': not lost,
                     'all_common_success': len(common) == len(rows),
                     'speedup_available': gm is not None,
                     'speedup_threshold': gm is not None and gm > required}
    selected_checks = legacy_checks if policy == 'strict_v1' else checks
    accepted = all(selected_checks.values())
    return {'acceptance_policy_version': policy, 'accepted': accepted,
            'stage_promoted': accepted, 'direct_classical_speedup_observed': None,
            'final_time_improvement_certified': False, 'performance_certified': False,
            'checks': selected_checks, 'stage_relative_checks': checks,
            'legacy_strict_checks': legacy_checks,
            'failed_conditions': [name for name, passed in selected_checks.items() if not passed],
            'incumbent_successes': len(previous_success), 'candidate_successes': len(candidate_success),
            'common_successes': len(common), 'total': len(rows), 'new_failures': len(lost),
            'incumbent_success_case_ids': sorted(previous_success),
            'candidate_success_case_ids': sorted(candidate_success),
            'common_success_case_ids': sorted(common), 'new_failure_case_ids': sorted(lost),
            'rescued_case_ids': sorted(rescued), 'both_failed_case_ids': sorted(failed),
            'minimum_common_samples': minimum, 'stage_min_common_success': count,
            'stage_min_common_fraction': fraction,
            'common_fraction': len(common) / len(rows) if rows else 0.,
            'geometric_speedup_vs_incumbent': gm,
            'geometric_mean_speedup_vs_incumbent': gm, 'speedup_ci95': ci,
            'required_speedup': required, 'minimum_speedup': required,
            'required_time_reduction': margin, 'changed_component_signatures': changes,
            'neural_used_cases': legacy_used, 'neural_numerical_use_common_cases': neural_common,
            'neural_counts': {'incumbent': _counts(all_previous), 'candidate': _counts(all_candidate)},
            'case_results': details, 'protocol_sources': sorted(protocol_sources),
            'ci_scope': 'paired case bootstrap, conditional on all-repeat common success; CI is reported, not a new promotion threshold',
            'scope': 'tuning-based stage promotion only; never final independent audit',
            'timing_scope': 'same imported-process constructor+solve wall time; no process startup',
            'failure_policy': 'all repeats must succeed; failed-case runtimes never enter speedup medians'}
