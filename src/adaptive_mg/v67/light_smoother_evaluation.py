"""Independent measured evaluation for cached scalar line-smoother controls.

Every timed workload owns a fresh solver. Warm preparation runs with online
adaptation disabled; the measured solve starts from the original predicted
coefficients and may adapt *inside* its timer. Adaptation is a subset of solve
time, never an additional quantity added to measured wall time.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..provenance import hardware_environment, json_safe
from .evaluation_metrics import _hierarchy, _source_signature, _speed_summary
from .fixed_classical import BilinearClassicalBaseline
from .joint_transfer_evaluation import _median, _valid_runs
from .limited import initialize_timing_runtime, vector_digest
from .research_data import HOLDOUT, _check_claim, _hash, _write_json
from .research_evaluation import (
    _arm_config, _counter_sum, _independent_em_reference, _model_metadata,
    _result_record, aggregate_research, manufactured_rhs, _construct,
)

VERSION = 'light-smoother-measured-v2'
REGIMES = ('cold', 'warm', 'multiple', 'warm_multiple')
LIGHT_ARMS = ('learned_P_line', 'learned_P_MLP', 'learned_P_regression',
              'learned_P_MLP_TTT', 'learned_P_regression_online')
EXTRA_METRICS = ('online_adaptation_seconds', 'online_adaptation_calls',
    'online_adaptation_accepted', 'online_adaptation_rejected',
    'online_candidate_cycles', 'coefficient_prediction_seconds',
    'feature_seconds', 'mlp_inference_calls', 'coefficient_generation_count',
    'coefficient_change_count', 'coefficient_delta_l2', 'accepted_cycles',
    'hierarchy_rebuilds', 'factor_rebuilds', 'all_executed_cycles',
    'online_gradient_vcycles', 'online_gradient_steps',
    'online_probe_completed_cycles', 'online_gradient_completed_vcycles',
    'fallback_event_count', 'classical_safety_cycles')


def _construct_light(example, arm, cfg):
    from .light_smoother_runtime import PreparedLightMG
    return PreparedLightMG(example.a, example.n, arm['model'], cfg,
        predictor=arm.get('predictor'), online=arm.get('online'),
        bounds=tuple(arm.get('bounds', (.8, 1.2))),
        compute_features=arm.get('predictor') is not None,
        allow_setup_fallback=True)


def measured_light(example, arm, cfg, baseline, *, regime='cold', rhs_count=1):
    """One directly timed workload; exact solutions are used only afterwards."""
    if regime not in REGIMES or (regime not in ('multiple', 'warm_multiple') and rhs_count != 1):
        raise ValueError('Cold/warm use RHS=1; multiple uses an actual RHS batch')
    light = bool(arm.get('light', False))
    def construct():
        # One measurement protocol for every arm. In particular a classical
        # or CNN reference must not prime on the timed RHS while scalar arms
        # prime on a distinct one. Legacy evaluators remain unchanged.
        return (_construct_light(example, arm, cfg) if light else
                _construct(example, arm, cfg, baseline, rhs_count))
    rhs, exacts = manufactured_rhs(example, rhs_count)
    warm = regime in ('warm', 'warm_multiple')
    prepared = None
    prime_records, results = [], []
    prime_seconds, construction = 0., 0.
    failure = None
    if warm:
        prime_start = perf_counter()
        try:
            prepared = construct()
            if hasattr(prepared, 'prepare_warm'):
                prepared.prepare_warm()
            configured_online = prepared.online_enabled if light else False
            # The priming solve cannot consume the adaptation budget or tune
            # coefficients for the measured RHS. It uses a distinct RHS too.
            if light:
                prepared.online_enabled = False
            prime_rhs, prime_exact = manufactured_rhs(example, rhs_count + 1)
            prime_b, prime_u = prime_rhs[-1], prime_exact[-1]
            if vector_digest(prime_b) in {vector_digest(b) for b in rhs}:
                raise ValueError('Warm priming requires a distinct RHS')
            prime = prepared.solve(prime_b, np.zeros_like(prime_b))
            prime_seconds = perf_counter() - prime_start
            prime_records = [_result_record(prime, example, prime_b, prime_u, prepared.config)]
            if light:
                prepared.reset_online()
                prepared.online_enabled = configured_online
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            prime_seconds = perf_counter() - prime_start
            failure = f'{type(exc).__name__}: {exc}'
    start = perf_counter()
    try:
        if failure is not None:
            raise RuntimeError('Warm preparation failed: ' + failure)
        if prepared is None:
            try:
                prepared = construct()
            finally:
                # A constructor that fails still consumes setup time.
                construction = perf_counter() - start
        if regime in ('multiple', 'warm_multiple'):
            results = prepared.solve_many(rhs, np.zeros_like(rhs))
        else:
            results = [prepared.solve(rhs[0], np.zeros_like(rhs[0]))]
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        failure = f'{type(exc).__name__}: {exc}'
    elapsed = perf_counter() - start
    actual_cfg = getattr(prepared, 'config', cfg)
    checked = [_result_record(result, example, b, exact, actual_cfg)
               for result, b, exact in zip(results, rhs, exacts)]
    counters = _counter_sum(checked)
    classical = _hierarchy(getattr(prepared, 'classical', None))
    learned = _hierarchy(getattr(prepared, 'learned', None))
    hierarchy = learned or classical
    success = bool(failure is None and len(checked) == rhs_count
        and all(row['verified_success'] for row in checked)
        and (not warm or all(row['verified_success'] for row in prime_records))
        and np.isfinite(elapsed) and elapsed > 0)
    accepted_neural = sum(counters.get('branch_' + branch + '_cycles', 0)
                          for branch in ('H_S', 'H_P', 'H_SP'))
    factors = [row.get('convergence_factor') for row in checked]
    contraction = (float(np.exp(np.mean(np.log(np.maximum(factors, 1e-300)))))
        if factors and all(isinstance(x, (int, float)) and np.isfinite(x) and x >= 0 for x in factors)
        else None)
    extra = {key: counters.get(key, 0) for key in EXTRA_METRICS}
    extra['online_candidate_cycles'] = counters.get('online_probe_cycles', 0)
    extra['coefficient_generation_count'] = counters.get('coefficient_generations', 0)
    extra['coefficient_change_count'] = counters.get('coefficient_changes', 0)
    extra['all_executed_cycles'] = counters.get('all_executed_cycles',
        sum(row.get('executed_cycles', 0) for row in checked))
    # Endpoint norms are not additive over RHS. Report the final cache state.
    last_stats = checked[-1].get('stats', {}) if checked else {}
    extra['coefficient_delta_l2'] = last_stats.get('coefficient_distance_from_ones', 0.)
    extra['accepted_cycles'] = sum(row.get('cycles', 0) for row in checked)
    # Legacy and scalar banks assign their raw fallback_cycles differently.
    # Use shared primitive counters so reference and light arms report the
    # same safety work, including the lock tail, excluding ordinary C solves.
    extra['classical_safety_cycles'] = max(0, counters.get('classical_cycles', 0)
        - counters.get('classical_bypass_cycles', 0))
    extra['fallback_event_count'] = (counters.get('lock_count', 0)
        + counters.get('gate_closed_fallback_cycles', 0))
    # Light banks are eager; reference CNN banks can be built lazily in the
    # first solve. Their instrumented setup is already inside elapsed.
    neural_setup = counters.get('light_setup_seconds', 0.) if light else counters.get('branch_setup_seconds', counters.get('nn_setup_seconds', 0.))
    total_setup = construction if light else construction + neural_setup
    record = dict(regime=regime, rhs_count=rhs_count, wall_seconds=elapsed,
        seconds_per_rhs=elapsed/rhs_count, success=success, successful=success,
        error=failure, convergence_factor=contraction,
        constructor_seconds=construction, total_setup_seconds=total_setup,
        classical_setup_seconds=counters.get('classical_setup_seconds', 0.) if light else construction,
        neural_setup_seconds=neural_setup,
        solve_seconds=max(0., elapsed-total_setup),
        solve_seconds_scope='observed wall minus constructor and any lazy bank setup; adaptation/guards/rejected trials are included',
        online_timing_scope='online_adaptation_seconds is a subset of solve_seconds, not added to wall_seconds',
        warm_prime_seconds=prime_seconds, warm_prime_results=prime_records,
        warm_prime_counters=_counter_sum(prime_records),
        warm_prime_online_adaptation=False, independent_warm_prime=warm,
        warm_starts_from='original predicted coefficients; adaptation budget reset',
        timed_rhs_reuse='fixed P/Ac/line factors; accepted coefficients reused across RHS; zero initial guesses',
        setup_scope='excluded_prepared_operator' if warm else 'included_once',
        rhs_results=checked, counters=counters,
        operator_digest=example.digest, normalized_operator_digest=example.group_digest,
        rhs_digests=[vector_digest(b) for b in rhs],
        exact_digests=[vector_digest(x) for x in exacts],
        requested_branch=arm['branch'], selection=(arm.get('rules') or baseline).selection_record(),
        selector_in_wall_time=False, classical_hierarchy=classical, learned_hierarchy=learned,
        operator_complexity=sum(row['operator_nnz'] for row in hierarchy)/example.a.nnz if hierarchy else None,
        transfer_complexity=sum(row['prolongation_nnz'] for row in hierarchy)/example.a.shape[0] if hierarchy else None,
        executed_cycles=sum(row.get('executed_cycles', 0) for row in checked),
        accepted_neural_cycles=accepted_neural, actual_neural_used=accepted_neural > 0,
        rollback_count=counters.get('rollback_count', 0),
        fallback_count=extra['classical_safety_cycles'],
        fallback_count_scope='all classical safety cycles including recovery and lock tail; excludes ordinary classical baseline',
        fallback_event_count_scope='safety lock entries (setup failure, rejection, near-tolerance) plus gate-closed decisions',
        setup_failures=counters.get('setup_failures', 0),
        operator_generation_seconds=counters.get('operator_generation_seconds', 0.),
        controller_seconds=0., detector_seconds=0., teacher_evaluation_seconds=0.,
        training_only=False, expected_rhs_mode='actual', expected_rhs_hint=rhs_count,
        inference_device='cpu', inference_dtype=cfg.inference_dtype, numerical_dtype='float64',
        ttt_seconds=extra['online_adaptation_seconds'],
        warm_ttt_seconds=extra['online_adaptation_seconds'] if warm else 0.,
        ttt_preparation_seconds=0., ttt_operator_generation_seconds=0.,
        ttt_updates_applied=counters.get('online_gradient_steps', 0),
        ttt_adopted=extra['online_adaptation_accepted'] > 0,
        measurement_config=actual_cfg.to_dict(), **extra)
    record.update(solver_executed_cycles=record['executed_cycles'],
        setup_fallback=bool(getattr(prepared, 'setup_failure', None)),
        setup_fallback_reason=getattr(prepared, 'setup_failure', None),
        coefficient_delta_from_prediction_l2=last_stats.get('coefficient_distance_from_prediction', 0.))
    return json_safe(record)


def _metadata(arm, cfg, baseline):
    value = _model_metadata(arm)
    predictor = arm.get('predictor')
    if arm.get('light'):
        value['parameter_count'] = sum(p.numel() for p in arm['model'].transfer.parameters())
        count = 242 if predictor is not None and predictor.kind == 'mlp' else 26 if predictor is not None else 0
        value.update(smoother_parameter_count=count,
            smoother_nn_parameter_count=count if predictor is not None and predictor.kind == 'mlp' else 0,
            parameter_scope='common P CNN plus scalar smoother predictor coefficients',
            predictor_signature=_hash(predictor.to_dict()) if predictor is not None else None,
            predictor= predictor.to_dict() if predictor is not None else None,
            bounds=list(arm.get('bounds', (.8, 1.2))), online=arm.get('online'),
            p_signature=arm['model'].expert_signature('transfer'))
        value['parameter_count'] += count
    elif arm.get('model') is not None:
        value.update(smoother_parameter_count=sum(p.numel() for p in arm['model'].smoother.parameters()),
            smoother_nn_parameter_count=sum(p.numel() for p in arm['model'].smoother.parameters()),
            p_signature=arm['model'].expert_signature('transfer'))
    else:
        value.update(smoother_parameter_count=0, smoother_nn_parameter_count=0)
    value['p_parameter_count'] = (sum(p.numel() for p in arm['model'].transfer.parameters())
                                  if arm.get('model') is not None else 0)
    return {**value, 'light': bool(arm.get('light')), 'branch': arm['branch'],
        'selector': False, 'config': cfg.to_dict(),
        'baseline_digest': (arm.get('rules') or baseline).digest(),
        'robustness_only': bool(arm.get('robustness_only', False))}


def _validated_arms(arm_specs, cfg, baseline):
    if not isinstance(baseline, BilinearClassicalBaseline):
        raise ValueError('Light smoother experiments require the fixed bilinear parent')
    baseline.validate_config(cfg)
    reference = arm_specs.get('fixed_C', {})
    if (reference.get('branch') != 'C' or reference.get('model') is not None
            or reference.get('selector', True) or reference.get('light')
            or reference.get('online') or reference.get('predictor')):
        raise ValueError('fixed_C must be the unchanged classical bilinear reference')
    if reference.get('rules') is not None and reference['rules'].digest() != baseline.digest():
        raise ValueError('fixed_C cannot use a different parent')
    arms, metadata, signatures = {}, {}, set()
    for name, spec in arm_specs.items():
        if spec.get('selector', True) or spec.get('policy') or spec.get('ttt') or spec.get('training_only'):
            raise ValueError('Light study forbids selectors, policies and training-only/old-TTT arms')
        if spec.get('branch') not in ('C', 'H_P', 'H_SP'):
            raise ValueError('Unsupported light-study branch')
        independent_em = _independent_em_reference(spec, baseline, cfg)
        alternate = spec.get('rules')
        if alternate is not None and alternate.digest() != baseline.digest() and not independent_em:
            raise ValueError('Only explicit classical EM5 may use a different parent')
        if spec.get('light'):
            if name not in LIGHT_ARMS or spec.get('branch') != 'H_P' or spec.get('model') is None:
                raise ValueError('Light scalar smoother arms use the frozen common H_P model')
            expected_kind = ('mlp' if name in ('learned_P_MLP', 'learned_P_MLP_TTT') else
                             'regression' if name in ('learned_P_regression', 'learned_P_regression_online') else None)
            if getattr(spec.get('predictor'), 'kind', None) != expected_kind:
                raise ValueError('Light arm label and predictor kind disagree')
            expected_online = {'learned_P_MLP_TTT': 'mlp_ttt',
                               'learned_P_regression_online': 'regression_online'}.get(name)
            online = spec.get('online') or {}
            if expected_online:
                if online.get('enabled') is not True or online.get('method') != expected_online:
                    raise ValueError('Online arm must explicitly enable its matching adaptation method')
            elif online.get('enabled', False):
                raise ValueError('Basic arms must keep online adaptation disabled')
        elif name == 'frozen_P_line':
            if spec.get('branch') != 'H_SP' or spec.get('model') is None:
                raise ValueError('frozen_P_line requires an actual CNN-line H_SP checkpoint')
        elif spec.get('branch') != 'C':
            raise ValueError('Only frozen_P_line is an optional non-light learned arm')
        if spec.get('model') is not None:
            signatures.add(spec['model'].expert_signature('transfer'))
        arm = {**spec, 'config': _arm_config(cfg, spec)}
        (alternate or baseline).validate_config(arm['config'])
        if not independent_em and arm['config'].mg != cfg.mg:
            raise ValueError('All learned arms must share the full numerical protocol')
        allowed = {'mode', 'branch', 'use_smoother', 'use_transfer', 'spatial',
                   'gate_mode', 'record_trace', 'use_learned_controller'}
        if independent_em:
            allowed.add('mg')
        if ({k: v for k, v in arm['config'].to_dict().items() if k not in allowed}
                != {k: v for k, v in cfg.to_dict().items() if k not in allowed}):
            raise ValueError('All arms must share the replacement, precision and safety protocol')
        arms[name] = arm
        metadata[name] = _metadata(arm, arm['config'], baseline)
    if len(signatures) > 1:
        raise ValueError('All learned comparison arms, including frozen_P_line, must have identical P weights')
    return arms, metadata


def _extra_table(report, rows, regimes, rhs_counts, metadata):
    lookup = {row['example']['name']: row for row in rows}
    for entry in report['table']:
        runs = lookup[entry['case']]['runs'][entry['arm']][entry['regime']][str(entry['rhs_count'])]
        entry.update({key: _median(runs, key) for key in EXTRA_METRICS})
        meta = metadata[entry['arm']]
        entry.update({key: meta.get(key, 0) for key in
            ('p_parameter_count', 'smoother_parameter_count', 'smoother_nn_parameter_count')})


def paired_effects(result):
    """Separate predictor effects from the incremental online adaptation effect."""
    pairs = (('learned_P_MLP', 'learned_P_line'),
             ('learned_P_regression', 'learned_P_line'),
             ('learned_P_MLP_TTT', 'learned_P_MLP'),
             ('learned_P_regression_online', 'learned_P_regression'))
    repeats = result['manifest']['repeats']
    metadata = result['manifest']['arms']
    effects = {}
    for regime, counts in result['summary'].items():
        for count in counts:
            for candidate, reference in pairs:
                if candidate not in metadata or reference not in metadata:
                    continue
                rows, ratios, excluded = [], [], []
                for example in result['rows']:
                    runs, base = (example['runs'][name][regime][str(count)]
                                  for name in (candidate, reference))
                    good, baseline_good = _valid_runs(runs, repeats), _valid_runs(base, repeats)
                    ratio = _median(base, 'wall_seconds')/_median(runs, 'wall_seconds') if good and baseline_good else None
                    if ratio is not None:
                        ratios.append(ratio)
                    else:
                        excluded.append(example['example']['name'])
                    rows.append(dict(case=example['example']['name'], candidate_success=good,
                        reference_success=baseline_good, paired_speedup=ratio,
                        candidate_wall_seconds=_median(runs, 'wall_seconds'),
                        reference_wall_seconds=_median(base, 'wall_seconds'),
                        candidate_online_seconds=_median(runs, 'online_adaptation_seconds')))
                effects[f'{regime}/rhs{count}/{candidate}_vs_{reference}'] = dict(
                    candidate=candidate, reference=reference, rows=rows,
                    paired_speedup=_speed_summary(ratios), total_cases=len(rows),
                    candidate_successes=sum(row['candidate_success'] for row in rows),
                    reference_successes=sum(row['reference_success'] for row in rows),
                    excluded_failure_case_ids=excluded,
                    scope='diagnostic paired successful cohort; failures retained; not the full-cohort selection rule')
    return effects


def evaluate_light(examples, arm_specs, cfg, baseline, out, *, repeats=3, warmups=1,
                   rhs_counts=(1, 4, 16, 64), regimes=('cold', 'warm', 'multiple'),
                   resume=False, order_seed=2107):
    """Paired actual timings with one durable record per measurement."""
    examples, regimes, rhs_counts = list(examples), tuple(regimes), tuple(rhs_counts)
    if (not examples or not arm_specs or type(repeats) is not int or repeats < 1
            or type(warmups) is not int or warmups < 0):
        raise ValueError('Nonempty evaluation and valid repeat/warmup counts required')
    if not regimes or len(set(regimes)) != len(regimes) or set(regimes)-set(REGIMES):
        raise ValueError('Invalid or duplicate regimes')
    if (not rhs_counts or len(set(rhs_counts)) != len(rhs_counts)
            or any(type(k) is not int or k < 1 for k in rhs_counts)):
        raise ValueError('Invalid RHS counts')
    if type(order_seed) is not int or order_seed < 0:
        raise ValueError('order_seed must be a nonnegative integer')
    if len({e.group_digest for e in examples}) != len(examples) or len({e.name for e in examples}) != len(examples):
        raise ValueError('Evaluation names and operators must be unique')
    arms, metadata = _validated_arms(arm_specs, cfg, baseline)
    for example in examples:
        if getattr(example, 'classical_selection', None) != baseline.selection_record():
            raise ValueError('Example must carry the identical fixed bilinear baseline')
        if getattr(example, 'strong_selection', None) is not None:
            raise ValueError('Selector-labelled data are not valid for the fixed baseline')
        if getattr(example, 'research_split', None) in HOLDOUT:
            _check_claim(getattr(example, 'research_run_root', None), getattr(example, 'research_final_claim', None))
    initialize_timing_runtime()
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = [dict(e.manifest(), research_split=getattr(e, 'research_split', None),
        family=getattr(e, 'research_family', getattr(e.case, 'family', e.case.pattern)),
        rhs_digest=vector_digest(e.b), exact_digest=vector_digest(e.exact),
        classical_selection=e.classical_selection) for e in examples]
    manifest = json_safe(dict(version=VERSION, reference_arm='fixed_C', case_inputs=inputs,
        arms=metadata, source_signature=_source_signature(), baseline=baseline.to_dict(),
        baseline_digest=baseline.digest(), config=cfg.to_dict(), repeats=repeats,
        warmups=warmups, rhs_counts=rhs_counts, regimes=regimes, order_seed=order_seed,
        hardware=hardware_environment(refresh=True),
        cold_scope='fresh constructor plus solve, including P, fallback hierarchy, features, coefficient prediction, line factors, adaptation, rejected trials and fallback',
        warm_scope='constructor and independent unadapted prime excluded; coefficients and adaptation budget reset; all adaptation in timed solve included',
        multiple_scope='fresh constructor once; actual solves and adaptation costs for every RHS; accepted coefficients reused',
        timing_scope='online_adaptation_seconds is a subset of solve_seconds; only measured wall_seconds selects a model',
        final_test_eligibility='untouched final gate required; evaluator does not certify performance'))
    path = output/'run_manifest.json'
    previous = None
    if path.exists():
        if not resume:
            raise FileExistsError('Evaluation exists; use identical --resume')
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Resume source/model/environment/data/protocol mismatch')
        if (output/'progress.json').exists() and json.loads((output/'progress.json').read_text()).get('status') == 'complete':
            previous = json.loads((output/'comparison.json').read_text())
    elif resume:
        raise ValueError('Cannot resume an unstarted evaluation')
    else:
        if any((output/p).exists() for p in ('raw_results.json', 'comparison.json', 'progress.json', 'repeat_records')):
            raise FileExistsError('Refusing to overwrite unmanifested evaluation records')
        _write_json(path, manifest, exclusive=True)
    digest = _hash(manifest)
    rows = [dict(example=e.manifest(), research_split=getattr(e, 'research_split', None),
        family=getattr(e, 'research_family', getattr(e.case, 'family', e.case.pattern)),
        runs={name: {regime: {str(k): [] for k in (rhs_counts if regime in ('multiple', 'warm_multiple') else (1,))}
                    for regime in regimes} for name in arms}) for e in examples]
    jobs = [(name, regime, k) for name in arms for regime in regimes
            for k in (rhs_counts if regime in ('multiple', 'warm_multiple') else (1,))]
    total = len(examples)*repeats*len(jobs)
    completed, warmup_wall = 0, 0.
    # Deterministically shuffle both cases and arm/workload jobs. Raw rows stay
    # in the declared dataset order regardless of interruptions.
    order = np.random.default_rng(order_seed).permutation(len(examples))
    for case_index in order:
        case_index = int(case_index)
        example, row = examples[case_index], rows[case_index]
        def key_for(name, regime, count, repeat):
            return dict(manifest_digest=digest, case_index=case_index, arm=name,
                        regime=regime, rhs_count=count, repeat=repeat)
        pending = any(not (output/'repeat_records'/(_hash(key_for(name, regime, count, repeat))+'.json')).exists()
            for repeat in range(repeats) for name, regime, count in jobs)
        if pending:
            for warmup in range(warmups):
                rng = np.random.default_rng(np.random.SeedSequence([order_seed, case_index, 0, warmup]))
                for i in rng.permutation(len(jobs)):
                    name, regime, count = jobs[i]
                    run = measured_light(example, arms[name], arms[name]['config'], baseline, regime=regime, rhs_count=count)
                    warmup_wall += run['wall_seconds']+run['warm_prime_seconds']
        for repeat in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([order_seed, case_index, 1, repeat]))
            for i in rng.permutation(len(jobs)):
                name, regime, count = jobs[i]
                key = key_for(name, regime, count, repeat)
                record_path = output/'repeat_records'/(_hash(key)+'.json')
                if record_path.exists():
                    saved = json.loads(record_path.read_text())
                    if saved.get('inputs') != key or saved.get('run_digest') != _hash(saved.get('run')):
                        raise ValueError('Resume measurement record changed')
                    run = saved['run']
                else:
                    run = measured_light(example, arms[name], arms[name]['config'], baseline, regime=regime, rhs_count=count)
                    _write_json(record_path, dict(inputs=key, run=run, run_digest=_hash(run)))
                row['runs'][name][regime][str(count)].append(run)
                completed += 1
                _write_json(output/'progress.json', dict(status='running', completed_measurements=completed, total_measurements=total))
    report = aggregate_research(rows, metadata, repeats=repeats, regimes=regimes,
                                rhs_counts=rhs_counts, reference_arm='fixed_C')
    _extra_table(report, rows, regimes, rhs_counts, metadata)
    report.update(rows=rows, manifest=manifest, output_dir=str(output), reference_arm='fixed_C',
                  warmup_wall_seconds_this_invocation=warmup_wall)
    report['paired_effects'] = paired_effects(report)
    if previous is not None:
        ignored = {'rows', 'manifest', 'warmup_wall_seconds_this_invocation'}
        if json_safe({k: v for k, v in report.items() if k not in ignored}) != {k: v for k, v in previous.items() if k not in ignored}:
            raise ValueError('Completed comparison differs from verified raw measurements')
        report['warmup_wall_seconds_this_invocation'] = previous['warmup_wall_seconds_this_invocation']
    _write_json(output/'raw_results.json', rows)
    _write_json(output/'paired_effects.json', report['paired_effects'])
    _write_json(output/'comparison.json', {k: v for k, v in report.items() if k not in ('rows', 'manifest')})
    with (output/'comparison.csv').open('w', newline='') as stream:
        table = report['table']
        writer = csv.DictWriter(stream, fieldnames=list(table[0]) if table else [])
        writer.writeheader()
        writer.writerows(table)
    _write_json(output/'progress.json', dict(status='complete', completed_measurements=completed, total_measurements=total))
    return report


def select_light_candidate(result, *, rhs_count=1, candidate_arms=None):
    """Choose a full-cohort validation arm by actual setup+solve mean time."""
    if type(rhs_count) is not int or rhs_count < 1:
        raise ValueError('rhs_count must be positive')
    repeats = result.get('manifest', {}).get('repeats')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Invalid evaluation repeats')
    rows = result.get('rows', [])
    if not rows or len({row['example']['name'] for row in rows}) != len(rows):
        raise ValueError('Nonempty unique validation cohort required')
    if any(row.get('research_split') != 'validation' for row in rows):
        raise ValueError('Only validation may select a model; never train/final/OOD')
    metadata = result.get('manifest', {}).get('arms', {})
    candidates = tuple(name for name in LIGHT_ARMS if name in metadata) if candidate_arms is None else tuple(candidate_arms)
    if not candidates or len(set(candidates)) != len(candidates) or set(candidates)-set((*LIGHT_ARMS, 'frozen_P_line')):
        raise ValueError('Choose explicit light-study learned comparison arms')
    if str(rhs_count) not in result.get('summary', {}).get('multiple', {}):
        raise ValueError('Selection requires measured setup-inclusive multiple RHS workload')
    evidence = {}
    for name in ('fixed_C',)+candidates:
        cases, failures = [], []
        for row in rows:
            runs = row['runs'].get(name, {}).get('multiple', {}).get(str(rhs_count), [])
            if any(run.get('setup_scope') != 'included_once' for run in runs):
                raise ValueError('Selection cannot exclude setup')
            success = _valid_runs(runs, repeats)
            if not success:
                failures.append(row['example']['name'])
            cases.append(dict(case=row['example']['name'], successful=success,
                median_wall_seconds=_median(runs, 'wall_seconds'),
                median_setup_seconds=_median(runs, 'total_setup_seconds'),
                median_solve_seconds=_median(runs, 'solve_seconds')))
        deployable = name in metadata and not any(metadata.get(name, {}).get(key) for key in ('training_only', 'reference_only', 'robustness_only'))
        eligible = not failures and deployable
        evidence[name] = dict(eligible=eligible, deployable=deployable,
            successes=len(rows)-len(failures), failures=len(failures), total=len(rows),
            failure_case_ids=failures, rows=cases,
            model_signature=metadata.get(name, {}).get('model_signature'),
            predictor_signature=metadata.get(name, {}).get('predictor_signature'),
            mean_case_median_wall_seconds=float(np.mean([r['median_wall_seconds'] for r in cases])) if eligible else None)
    eligible = [name for name in candidates if evidence[name]['eligible']]
    selected = min(eligible, key=lambda name: evidence[name]['mean_case_median_wall_seconds']) if eligible else None
    deployment, beats_fixed = selected, None
    if evidence['fixed_C']['eligible']:
        beats_fixed = bool(selected and evidence[selected]['mean_case_median_wall_seconds'] < evidence['fixed_C']['mean_case_median_wall_seconds'])
        if not beats_fixed:
            deployment = 'fixed_C'
    return json_safe(dict(selected_arm=selected, recommended_deployment_arm=deployment,
        faster_than_fixed_on_target=beats_fixed, regime='multiple', rhs_count=rhs_count,
        total_cases=len(rows), candidates=evidence,
        objective='arithmetic mean across all operators of median repeated actual setup+solve wall seconds',
        failure_policy='every repeat and RHS on every operator must succeed; failures invalidate eligibility',
        warm_secondary={key: result.get('summary', {}).get(key, {}) for key in ('warm', 'warm_multiple') if key in result.get('summary', {})},
        final_certificate=False))
