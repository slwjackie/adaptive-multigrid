"""Measured cold, warm and same-A multi-RHS research comparisons.

All times are observed, never inferred from contraction or a setup amortization
formula.  This evaluator does not select architectures or certify final claims;
the caller must use research_data's final gate for final/OOD examples.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import json

import numpy as np

from ..provenance import hardware_environment, json_safe, operator_digest, stable_norm
from .acceptance import _successful
from .banks import resolve_device
from .limited import forced_config, initialize_timing_runtime, vector_digest
from .research_data import HOLDOUT, _check_claim, _hash, _write_json
from .research_runtime import is_teacher, offline_teacher_generation
from .solver import PreparedAdaptiveMG
from .strong import PreparedStrongMG, classical_bank
from .strong_evaluation import (_hierarchy, _source_signature, _speed_summary,
                                build_strong_arms, run_strong_benchmark)

VERSION = 'strong-aware-research-measurement-v1'
REGIMES = ('cold', 'warm', 'multiple')


def manufactured_rhs(example, count):
    """Deterministic distinct manufactured RHS; prefixes match every batch size."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('RHS count must be a positive integer')
    exacts = [np.asarray(example.exact, np.float64)]
    coords = np.arange(1, example.n + 1) / (example.n + 1)
    x, y = np.meshgrid(coords, coords, indexing='ij')
    for index in range(1, count):
        seed = int(hashlib.sha256(f'{example.group_digest}:research_rhs:{index}'.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        exact = np.sin(np.pi * x) * np.sin(np.pi * y)
        for _ in range(5):
            k, l = rng.integers(1, min(example.n, 12) + 1, size=2)
            exact += rng.uniform(-.35, .35) * np.sin(k * np.pi * x) * np.sin(l * np.pi * y)
        exacts.append(exact.ravel())
    exacts = np.stack(exacts)
    rhs = np.stack([example.b] + [np.asarray(example.a @ exact) for exact in exacts[1:]])
    return rhs, exacts


def _active_modules(arm):
    model = arm.get('model')
    if model is None:
        return []
    branch = arm['branch']
    names = {'C': (), 'H_S': ('smoother',), 'H_P': ('transfer',), 'H_SP': ('smoother', 'transfer'),
             'auto': ('smoother', 'transfer', 'detector', 'controller')}[branch]
    return [getattr(model, name) for name in names]


def _model_metadata(arm):
    model, policy = arm.get('model'), arm.get('policy')
    modules = _active_modules(arm)
    if any(is_teacher(module) for module in modules) and not arm.get('training_only', False):
        raise ValueError('Teacher arm must be explicitly labelled training_only')
    if policy is not None and arm.get('training_only', False):
        raise ValueError('A deployment policy cannot be an offline teacher arm')
    parameters = sum(parameter.numel() for module in modules for parameter in module.parameters())
    if policy is not None:
        if hasattr(policy, 'digest'):
            policy_signature = policy.digest()
        elif hasattr(policy, 'signature'):
            policy_signature = policy.signature()
        elif hasattr(policy, 'to_dict'):
            policy_signature = _hash(policy.to_dict())
        else:
            raise ValueError('Policy needs a stable digest()/signature()/to_dict() for reproducible evaluation')
        parameter_ids = set()
        parameters = 0
        for branch, components in getattr(policy, 'models', {}).items():
            for module in _active_modules(dict(model=components, branch=branch)):
                if is_teacher(module):
                    raise ValueError('Teacher cannot belong to a deployment policy')
                for parameter in module.parameters():
                    if id(parameter) not in parameter_ids:
                        parameter_ids.add(id(parameter))
                        parameters += parameter.numel()
    else:
        policy_signature = None
    return dict(model_signature=model.signature() if model is not None else None,
        policy_signature=policy_signature, parameter_count=parameters,
        parameter_scope='enabled heavy experts (plus tiny online models for auto); inactive checkpoint experts excluded',
        training_only=bool(arm.get('training_only', False)),
        teacher_modules=sum(is_teacher(module) for module in modules),
        deployment_teacher_allowed=False,
        expected_rhs_mode=arm.get('expected_rhs_mode', 'actual'),
        reference_only=bool(arm.get('reference_only', False)),
        model_metadata=json_safe(model.metadata) if model is not None else None)


def _arm_config(cfg, arm):
    base = arm.get('config', cfg)
    branch = arm['branch']
    return replace(base, mode='research', branch='auto') if arm.get('policy') is not None else forced_config(base, branch)


def _construct(example, arm, cfg, rules, expected_rhs):
    if arm.get('policy') is not None:
        from .research_policy import PreparedResearchMG
        cls = PreparedResearchMG
        if arm.get('expected_rhs_mode', 'actual') == 'blind':
            class RHSBlindResearchMG(PreparedResearchMG):
                """Same batch kernel/cache, deliberately withheld RHS-count hint."""
                def _refresh_key(self):
                    super()._refresh_key()
                    self.cache_key = _hash((self.cache_key, 'expected_rhs_blind'))

                def solve_many(self, bs, x0=None):
                    # Bypass only PreparedResearchMG's len(bs) hint update. The
                    # inherited actual batch solve/freshness/safety path remains.
                    self.expected_rhs = 1
                    return PreparedStrongMG.solve_many(self, bs, x0)
            cls, expected_rhs = RHSBlindResearchMG, 1
        return cls(example.a, example.n, policy=arm['policy'], config=cfg,
                   rules=rules, expected_rhs=expected_rhs)
    if arm.get('selector', True):
        return PreparedStrongMG(example.a, example.n, arm.get('model'), cfg, rules=rules)
    return PreparedAdaptiveMG(example.a, example.n, arm.get('model'), cfg)


def _counter_sum(results):
    counters = {}
    for result in results:
        for container in ('stats', 'setup_stats'):
            for key, value in result.get(container, {}).items():
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    counters[key] = counters.get(key, 0) + value
    return counters


def _result_record(result, example, rhs, exact, cfg):
    record = result.to_dict()
    error = result.x - exact
    denominator = float(exact @ (example.a @ exact))
    numerator = float(error @ (example.a @ error))
    record.update(recomputed_true_residual=stable_norm(rhs - example.a @ result.x),
        relative_solution_error=stable_norm(error) / max(stable_norm(exact), 1e-300),
        relative_energy_error=float(np.sqrt(max(numerator, 0.) / denominator)) if denominator > 0 and np.isfinite(numerator) else None,
        measurement_config=cfg.to_dict(), rhs_digest=vector_digest(rhs), exact_digest=vector_digest(exact),
        x0_digest=vector_digest(np.zeros_like(rhs)),
        residual_reference_norm=stable_norm(rhs))
    checked = record['recomputed_true_residual']
    record['verified_success'] = bool(_successful(record, cfg) and np.isfinite(checked)
        and 0 <= checked <= record['stopping_threshold'])
    return json_safe(record)


def _root(prepared):
    # Policy wrappers may expose a selected cached solver rather than hierarchy
    # properties directly. Missing complexity is reported as unavailable.
    if hasattr(prepared, 'classical'):
        return prepared
    for name in ('prepared', 'solver', 'active_solver', 'selected_solver'):
        inner = getattr(prepared, name, None)
        if inner is not None and hasattr(inner, 'classical'):
            return inner
    return prepared


def measured_research(example, arm, cfg, rules, *, regime='cold', rhs_count=1):
    """Measure one actual constructor/solve workload; verify after stopping timer."""
    if regime not in REGIMES or (regime != 'multiple' and rhs_count != 1):
        raise ValueError('Cold/warm use exactly one RHS; multiple uses an actual batch')
    rhs, exacts = manufactured_rhs(example, rhs_count)
    zeros = np.zeros_like(rhs)
    prepared, elapsed, construction = None, None, 0.
    prime_seconds, prime_records = 0., []
    results = []
    failure = None
    context = offline_teacher_generation() if arm.get('training_only', False) else nullcontext()
    with context:
        # Warm cache priming has an explicit separate time. It is never silently
        # included in warm solve time or treated as a cold single-RHS result.
        if regime == 'warm':
            prime_started = perf_counter()
            try:
                prepared = _construct(example, arm, cfg, rules, 1)
                prime = prepared.solve(rhs[0], zeros[0])
                prime_seconds = perf_counter() - prime_started
                prime_records = [_result_record(prime, example, rhs[0], exacts[0], getattr(prepared, 'config', cfg))]
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                prime_seconds = perf_counter() - prime_started
                failure = f'{type(exc).__name__}: {exc}'
        started = perf_counter()
        try:
            if failure:
                raise RuntimeError('Warm cache prime failed: ' + failure)
            if prepared is None:
                prepared = _construct(example, arm, cfg, rules, rhs_count)
                construction = perf_counter() - started
            if regime == 'multiple':
                results = prepared.solve_many(rhs, zeros)
            else:
                results = [prepared.solve(rhs[0], zeros[0])]
            elapsed = perf_counter() - started
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            elapsed = perf_counter() - started
            failure = f'{type(exc).__name__}: {exc}'
    actual_cfg = getattr(prepared, 'config', cfg) if prepared is not None else cfg
    checked = [_result_record(result, example, b, exact, actual_cfg) for result, b, exact in zip(results, rhs, exacts)]
    counters = _counter_sum(checked)
    root = _root(prepared) if prepared is not None else None
    classical = _hierarchy(getattr(root, 'classical', None))
    learned = _hierarchy(getattr(root, 'learned', None))
    hierarchy = learned or classical
    complexity = dict(operator_complexity=sum(row['operator_nnz'] for row in hierarchy) / example.a.nnz,
                      transfer_complexity=sum(row['prolongation_nnz'] for row in hierarchy) / example.a.shape[0]) if hierarchy else dict(operator_complexity=None, transfer_complexity=None)
    selection = getattr(root, 'selection', None)
    neural_setup = float(counters.get('branch_setup_seconds', counters.get('nn_setup_seconds', 0.)))
    total_setup = construction + neural_setup
    accepted_neural = sum(counters.get('branch_' + b + '_cycles', 0) for b in ('H_S', 'H_P', 'H_SP'))
    success = bool(failure is None and len(checked) == rhs_count and all(r['verified_success'] for r in checked)
                   and (regime != 'warm' or all(r['verified_success'] for r in prime_records))
                   and np.isfinite(elapsed) and elapsed > 0)
    factors = [r.get('convergence_factor') for r in checked]
    contraction = float(np.exp(np.log(np.maximum(factors, 1e-300)).mean())) if factors and all(isinstance(v, (int, float)) and np.isfinite(v) and v >= 0 for v in factors) else None
    record = dict(regime=regime, rhs_count=rhs_count, wall_seconds=elapsed,
        seconds_per_rhs=elapsed / rhs_count, success=success, successful=success, error=failure,
        convergence_factor=contraction,
        constructor_seconds=construction, classical_setup_seconds=construction,
        classical_setup_scope='constructor including A-only selection, hierarchy, factorization and cache-key work',
        neural_setup_seconds=neural_setup, total_setup_seconds=total_setup,
        solve_seconds=max(0., elapsed - total_setup),
        solve_seconds_scope='measured outer wall minus separately instrumented constructor/branch setup; diagnostic decomposition',
        warm_prime_seconds=prime_seconds, warm_prime_results=prime_records,
        warm_prime_counters=_counter_sum(prime_records), rhs_results=checked, counters=counters,
        operator_digest=example.digest, normalized_operator_digest=example.group_digest,
        rhs_digests=[vector_digest(b) for b in rhs], exact_digests=[vector_digest(x) for x in exacts],
        requested_branch=arm['branch'], selector_in_wall_time=bool(arm.get('selector', True) and regime != 'warm'),
        selection=selection.to_dict() if selection is not None else None,
        classical_hierarchy=classical, learned_hierarchy=learned, **complexity,
        executed_cycles=sum(r.get('executed_cycles', 0) for r in checked),
        accepted_neural_cycles=accepted_neural, actual_neural_used=accepted_neural > 0,
        rollback_count=counters.get('rollback_count', 0),
        setup_failures=counters.get('setup_failures', 0),
        fallback_count=counters.get('fallback_cycles', 0),
        fallback_count_scope='actual classical recovery/gate-closed fallback cycles; setup failures and rollback events reported separately',
        inference_device=str(resolve_device(cfg,cells=example.a.shape[0])) if arm.get('model') is not None or arm.get('policy') is not None else None,
        inference_dtype=cfg.inference_dtype, numerical_dtype='float64',
        expected_rhs_mode=arm.get('expected_rhs_mode', 'actual'),
        expected_rhs_hint=1 if arm.get('expected_rhs_mode') == 'blind' else rhs_count,
        operator_generation_seconds=counters.get('operator_generation_seconds', 0.),
        controller_seconds=counters.get('controller_seconds', 0.), detector_seconds=counters.get('detector_seconds', 0.),
        training_only=bool(arm.get('training_only', False)),
        teacher_evaluation_seconds=elapsed if arm.get('training_only', False) else 0.,
        teacher_deployment_seconds=0., measurement_config=actual_cfg.to_dict())
    return json_safe(record)


def _median(runs, key):
    values = [r.get(key) for r in runs]
    return float(np.median(values)) if values and all(isinstance(v, (int, float)) and np.isfinite(v) for v in values) else None


def _max_error(runs, metric):
    values = [r.get(metric) for run in runs for r in run.get('rhs_results', [])]
    return max(values) if values and all(isinstance(v, (int, float)) and np.isfinite(v) for v in values) else None


def aggregate_research(rows, arm_metadata, *, repeats, regimes, rhs_counts):
    """Paired operator bootstrap; all repeats/all RHS must meet FP64 tolerance."""
    summary, table = {}, []
    for regime in regimes:
        summary[regime] = {}
        counts = rhs_counts if regime == 'multiple' else (1,)
        for count in counts:
            group = {}
            successes = {name: [] for name in arm_metadata}
            measurements = {}
            for index, row in enumerate(rows):
                for name in arm_metadata:
                    runs = row['runs'][name][regime][str(count)]
                    measurements[index, name] = runs
                    successes[name].append(len(runs) == repeats and all(r.get('successful') for r in runs))
            production = [name for name, metadata in arm_metadata.items()
                          if not metadata['training_only'] and not metadata.get('reference_only', False)]
            common = [i for i in range(len(rows)) if all(successes[name][i] for name in production)]
            for name, metadata in arm_metadata.items():
                ratios, common_ratios, new_failures, rescued, successful, neural, worse_errors = [], [], [], [], [], [], []
                for index, row in enumerate(rows):
                    case = row['example']['name']
                    runs, baseline = measurements[index, name], measurements[index, 'strong_C']
                    good, baseline_good = successes[name][index], successes['strong_C'][index]
                    wall, baseline_wall = _median(runs, 'wall_seconds'), _median(baseline, 'wall_seconds')
                    speedup = baseline_wall / wall if good and baseline_good else None
                    if speedup is not None:
                        ratios.append(speedup)
                        if index in common:
                            common_ratios.append(speedup)
                    if good:
                        successful.append(case)
                    if baseline_good and not good:
                        new_failures.append(case)
                    if good and not baseline_good:
                        rescued.append(case)
                    if any(run.get('actual_neural_used') for run in runs):
                        neural.append(case)
                    errors = {metric: _max_error(runs, metric) for metric in ('relative_solution_error', 'relative_energy_error')}
                    worse = False
                    if good and baseline_good:
                        for metric, value in errors.items():
                            reference = _max_error(baseline, metric)
                            if value is None or reference is None or value > 1.05 * reference + 1e-12:
                                worse = True
                    if worse:
                        worse_errors.append(case)
                    entry = dict(case=case, split=row.get('research_split'), family=row.get('family'), n=row['example']['case']['n'],
                        arm=name, regime=regime, rhs_count=count, successful=good, successful_repeats=sum(bool(r.get('successful')) for r in runs),
                        speedup_vs_strong=speedup, parameter_count=metadata['parameter_count'],
                        training_only=metadata['training_only'], actual_neural_used=case in neural,
                        expected_rhs_mode=metadata.get('expected_rhs_mode', 'actual'),
                        accuracy_worse_vs_strong=worse, **errors)
                    for key in ('wall_seconds', 'seconds_per_rhs', 'classical_setup_seconds', 'neural_setup_seconds',
                                'total_setup_seconds', 'solve_seconds', 'warm_prime_seconds', 'operator_generation_seconds',
                                'executed_cycles', 'accepted_neural_cycles', 'rollback_count', 'fallback_count', 'setup_failures',
                                'operator_complexity', 'transfer_complexity', 'controller_seconds', 'detector_seconds', 'teacher_evaluation_seconds'):
                        entry[key] = _median(runs, key)
                    table.append(entry)
                speed = _speed_summary(ratios)
                successful_times = [_median(measurements[i, name], 'wall_seconds') for i in range(len(rows)) if successes[name][i]]
                group[name] = dict(successes=len(successful), total=len(rows), success_case_ids=successful,
                    geometric_speedup=speed['geometric_speedup'], ci95=speed['ci95'],
                    median_wall_seconds=float(np.median(successful_times)) if successful_times else None,
                    speedup_vs_strong=speed, common_cohort_speedup_vs_strong=_speed_summary(common_ratios),
                    common_cohort_case_ids=[rows[i]['example']['name'] for i in common],
                    new_failure_case_ids=new_failures, rescued_case_ids=rescued, actual_neural_case_ids=neural,
                    accuracy_worse_vs_strong_case_ids=worse_errors, training_only=metadata['training_only'],
                    parameter_count=metadata['parameter_count'], final_certificate=False)
            summary[regime][str(count)] = group
    crossover = {}
    for name, metadata in arm_metadata.items():
        qualifying = []
        for count in sorted(rhs_counts):
            entry = summary.get('multiple', {}).get(str(count), {}).get(name)
            if entry and not entry['new_failure_case_ids'] and entry['speedup_vs_strong']['geometric_speedup'] is not None and entry['speedup_vs_strong']['geometric_speedup'] > 1:
                qualifying.append(count)
        crossover[name] = dict(first_measured_faster_rhs_count=min(qualifying) if qualifying else None,
                               tested_counts=list(rhs_counts) if 'multiple' in regimes else [],
                               scope='observed batch workloads only; no extrapolated amortization or certificate')
    return dict(summary=summary, table=table, amortization_crossover=crossover,
        confidence_scope='paired bootstrap over unique operators of per-operator median observed wall-time ratios; only named successful cohort',
        failure_policy='all repeats and all RHS must meet recomputed FP64 residual and attempt budget; failure timings remain raw, never speedups',
        accuracy_scope='manufactured discrete exact solution; post-timer L2/energy error, 5%+1e-12 diagnostic comparison slack',
        final_certificate=False)


def evaluate_research(examples, arm_specs, cfg, rules, out, *, repeats=5, warmups=1,
                      rhs_counts=(1, 4, 16, 64), regimes=REGIMES, resume=False, order_seed=2107):
    """Run paired research workloads and return rows/summary/table/manifest.

    ``arm_specs[name]``: model=Components|None, branch=C/H_S/H_P/H_SP,
    selector=True, training_only=False.  Optional policy=ResearchPolicy uses auto.
    Optional config may alter only branch/policy flags, not numerical protocol.
    Resume requires exact sources, models, rules, hardware, vectors and settings.
    """
    examples, regimes, rhs_counts = list(examples), tuple(regimes), tuple(rhs_counts)
    if not examples or not arm_specs or 'strong_C' not in arm_specs:
        raise ValueError('Nonempty examples/arms and explicit strong_C reference required')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1 or isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError('Invalid repeat/warmup count')
    if not regimes or set(regimes) - set(REGIMES) or len(set(regimes)) != len(regimes):
        raise ValueError('Invalid or duplicate regimes')
    if not rhs_counts or len(set(rhs_counts)) != len(rhs_counts) or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in rhs_counts):
        raise ValueError('Invalid RHS counts')
    if len({e.group_digest for e in examples}) != len(examples) or len({e.name for e in examples}) != len(examples):
        raise ValueError('Evaluation operators and names must be unique')
    for example in examples:
        stored_selection = getattr(example, 'strong_selection', None)
        if stored_selection is not None and stored_selection.get('rules_digest') != rules.digest():
            raise ValueError('Example was labelled by different frozen strong rules')
        if getattr(example, 'research_split', None) in HOLDOUT:
            _check_claim(getattr(example, 'research_run_root', None), getattr(example, 'research_final_claim', None))
            if any(spec.get('training_only') for spec in arm_specs.values()):
                raise ValueError('Teacher upper bounds cannot consume the untouched final/OOD test')
    baseline = arm_specs['strong_C']
    if baseline.get('branch') != 'C' or baseline.get('model') is not None or not baseline.get('selector', True) or baseline.get('policy') is not None:
        raise ValueError('strong_C must be the unchanged selected pure classical solver')
    arms, metadata = {}, {}
    for name, spec in arm_specs.items():
        if spec.get('branch') not in {'C', 'H_S', 'H_P', 'H_SP', 'auto'}:
            raise ValueError('Invalid branch')
        if spec['branch'] == 'auto' and spec.get('policy') is None:
            raise ValueError('Research auto requires an explicit refitted policy')
        if spec.get('expected_rhs_mode', 'actual') not in {'actual', 'blind'}:
            raise ValueError('expected_rhs_mode must be actual or blind')
        if spec.get('expected_rhs_mode') == 'blind' and spec.get('policy') is None:
            raise ValueError('RHS-count blindness is only a policy ablation')
        arm = {**spec, 'config': _arm_config(cfg, spec)}
        if arm['config'].mg != cfg.mg or arm['config'].inference_dtype != cfg.inference_dtype or arm['config'].inference_device != cfg.inference_device:
            raise ValueError('All arms must share numerical protocol, generation dtype and device')
        allowed_flags = {'mode', 'branch', 'use_smoother', 'use_transfer', 'spatial', 'gate_mode', 'record_trace', 'use_learned_controller'}
        if {k: v for k, v in arm['config'].to_dict().items() if k not in allowed_flags} != {k: v for k, v in cfg.to_dict().items() if k not in allowed_flags}:
            raise ValueError('All arms must share replacement budget, safety, and numerical protocol')
        arms[name] = arm
        metadata[name] = {**_model_metadata(arm), 'branch': arm['branch'], 'selector': arm.get('selector', True), 'config': arm['config'].to_dict()}
    initialize_timing_runtime()
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = [dict(e.manifest(), research_split=getattr(e, 'research_split', None), family=getattr(e, 'research_family', getattr(e.case, 'family', e.case.pattern)),
                   rhs_digest=vector_digest(e.b), exact_digest=vector_digest(e.exact), strong_selection=getattr(e, 'strong_selection', None)) for e in examples]
    manifest = json_safe(dict(version=VERSION, case_inputs=inputs, arms=metadata, source_signature=_source_signature(),
        rules=rules.to_dict(), rules_digest=rules.digest(), config=cfg.to_dict(), repeats=repeats,
        warmups=warmups, rhs_counts=rhs_counts, regimes=regimes, order_seed=order_seed,
        hardware=hardware_environment(), final_test_eligibility='caller must enforce untouched final gate; evaluator never certifies',
        cold_scope='imported process; selector+constructor+generation+solve, rollback included; checkpoint disk load/native library initialization excluded',
        warm_scope='one explicit successful cache-prime solve excluded and reported; actual subsequent same-A solve timed',
        multiple_scope='fresh constructor and actual solve_many for independent manufactured RHS; setup included once',
        teacher_scope='offline training-only upper-bound evaluation; never a student/deployment result'))
    manifest_path = output / 'run_manifest.json'
    previous_complete_report = None
    if manifest_path.exists():
        if not resume:
            raise FileExistsError('Evaluation exists; use exact --resume instead of overwrite')
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume source/model/rules/hardware/data/protocol mismatch')
        if (output / 'comparison.json').exists() and (output / 'progress.json').exists() and json.loads((output / 'progress.json').read_text()).get('status') == 'complete':
            previous_complete_report = json.loads((output / 'comparison.json').read_text())
    elif resume:
        raise ValueError('Cannot resume an unstarted evaluation')
    else:
        if any((output / name).exists() for name in ('raw_results.json', 'comparison.json', 'progress.json')):
            raise FileExistsError('Refusing to overwrite pre-existing unmanifested results')
        _write_json(manifest_path, manifest, exclusive=True)
    manifest_digest = _hash(manifest)
    rows = [dict(example=e.manifest(), research_split=getattr(e, 'research_split', None), family=getattr(e, 'research_family', getattr(e.case, 'family', e.case.pattern)),
                 runs={name: {regime: {str(k): [] for k in (rhs_counts if regime == 'multiple' else (1,))} for regime in regimes} for name in arms}) for e in examples]
    total = len(examples) * len(arms) * repeats * sum(len(rhs_counts) if regime == 'multiple' else 1 for regime in regimes)
    completed, warmup_wall = 0, 0.
    for case_index, (example, row) in enumerate(zip(examples, rows)):
        jobs = [(name, regime, k) for name in arms for regime in regimes for k in (rhs_counts if regime == 'multiple' else (1,))]
        # Warming CPU/framework paths uses independent prepared objects and is
        # separately observable. It does not replace the explicit warm regime.
        keys = [dict(manifest_digest=manifest_digest, case_index=case_index, arm=name, regime=regime, rhs_count=count, repeat=repeat)
                for repeat in range(repeats) for name, regime, count in jobs]
        pending = any(not (output / 'repeat_records' / (_hash(key) + '.json')).exists() for key in keys)
        if pending:
            for warmup in range(warmups):
                warm_rng = np.random.default_rng(np.random.SeedSequence([order_seed, case_index, 0, warmup]))
                for name, regime, count in [jobs[i] for i in warm_rng.permutation(len(jobs))]:
                    run = measured_research(example, arms[name], arms[name]['config'], rules, regime=regime, rhs_count=count)
                    warmup_wall += run['wall_seconds'] + run['warm_prime_seconds']
        for repeat in range(repeats):
            repeat_rng = np.random.default_rng(np.random.SeedSequence([order_seed, case_index, 1, repeat]))
            for name, regime, count in [jobs[i] for i in repeat_rng.permutation(len(jobs))]:
                key = dict(manifest_digest=manifest_digest, case_index=case_index, arm=name, regime=regime, rhs_count=count, repeat=repeat)
                path = output / 'repeat_records' / (_hash(key) + '.json')
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved.get('inputs') != key or saved.get('run_digest') != _hash(saved.get('run')):
                        raise ValueError('Resume measured-record input/hash mismatch')
                    run = saved['run']
                else:
                    run = measured_research(example, arms[name], arms[name]['config'], rules, regime=regime, rhs_count=count)
                    _write_json(path, dict(inputs=key, run=run, run_digest=_hash(run)))
                row['runs'][name][regime][str(count)].append(run)
                completed += 1
                _write_json(output / 'progress.json', dict(status='running', completed_measurements=completed, total_measurements=total))
    report = aggregate_research(rows, metadata, repeats=repeats, regimes=regimes, rhs_counts=rhs_counts)
    report.update(rows=rows, manifest=manifest, output_dir=str(output), warmup_wall_seconds_this_invocation=warmup_wall)
    if previous_complete_report is not None:
        ignored = {'rows', 'manifest', 'warmup_wall_seconds_this_invocation'}
        if json_safe({k: v for k, v in report.items() if k not in ignored}) != {k: v for k, v in previous_complete_report.items() if k not in ignored}:
            raise ValueError('Saved completed comparison differs from verified raw measurements')
        # Replaying a completed result must preserve its immutable evidence hash;
        # this is the original measured invocation, not a newly timed benchmark.
        report['warmup_wall_seconds_this_invocation'] = previous_complete_report['warmup_wall_seconds_this_invocation']
    _write_json(output / 'raw_results.json', rows)
    _write_json(output / 'comparison.json', {k: v for k, v in report.items() if k not in {'rows', 'manifest'}})
    table = report['table']
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]) if table else [])
        writer.writeheader()
        writer.writerows(table)
    _write_json(output / 'progress.json', dict(status='complete', completed_measurements=completed, total_measurements=total))
    return report


def evaluate_classical_reference(examples, cfg, rules, out, *, research, bank='controlled',
                                 repeats=5, warmups=1, resume=False, order_seed=1007):
    """Offline best-of-measured-bank reference; never fit a rule/policy/certificate.

    Reuses the existing tested classical benchmark without changing its selector
    or aggregation. Bank trials use the identical A,b,x0 and numerical budget.
    Only cold time-to-tolerance is relevant; total bank search cost is separate.
    """
    examples = list(examples)
    classical_bank(bank)  # Validate a concrete predeclared bank, never a null oracle.
    if not examples or len({e.group_digest for e in examples}) != len(examples):
        raise ValueError('Classical reference needs unique operators')
    for example in examples:
        if getattr(example, 'research_split', None) in HOLDOUT:
            _check_claim(getattr(example, 'research_run_root', None), getattr(example, 'research_final_claim', None))
        stored = getattr(example, 'strong_selection', None)
        if stored is not None and stored['rules_digest'] != rules.digest():
            raise ValueError('Classical reference rules differ from committed C*(A)')
    provenance = dict(split=sorted({getattr(e, 'research_split', 'diagnostic') for e in examples}),
        scope='offline empirical best among measured classical bank; reference only',
        examples=[e.manifest() for e in examples], independent_certification_evidence=False,
        selector_refit=False, policy_selection_used=False, architecture_selection_used=False)
    comparison = run_strong_benchmark(examples, provenance,
        build_strong_arms(research, cfg, branches=()), out, research, rules,
        repeats=repeats, warmups=warmups, resume=resume, oracle_bank=bank,
        order_seed=order_seed)
    by_case = {(row['case_id'], row['arm']): row for row in comparison['table']}
    tables, ratios = [], []
    for oracle in comparison['offline_oracle']:
        case = oracle['case_id']
        strong = by_case[case, 'strong_C']
        for arm in ('fixed_C', 'strong_C', 'offline_best_classical'):
            if arm == 'offline_best_classical':
                success = oracle['selected_arm'] is not None
                seconds = oracle['successful_median_seconds']
                strategy = oracle['selected_strategy']
                search = oracle['search_total_including_warmups_seconds']
            else:
                measured = by_case[case, arm]
                success, seconds = measured['success'], measured['successful_time_to_tolerance_seconds']
                strategy = measured['strategy_names'][0]
                search = 0.
            speed = strong['successful_time_to_tolerance_seconds'] / seconds if strong['success'] and success else None
            if arm == 'offline_best_classical' and speed is not None:
                ratios.append(speed)
            tables.append(dict(case=case, n=strong['n'], arm=arm, strategy=strategy, successful=success,
                median_wall_seconds=seconds, speedup_vs_strong=speed, offline_search_seconds=search,
                reference_only=arm == 'offline_best_classical'))
    result = dict(summary=dict(offline_best_classical=dict(
        successes=sum(r['selected_arm'] is not None for r in comparison['offline_oracle']), total=len(examples),
        speedup_vs_strong=_speed_summary(ratios),
        search_total_seconds=sum(r['search_total_including_warmups_seconds'] for r in comparison['offline_oracle']))),
        table=tables, offline_oracle=comparison['offline_oracle'], bank=bank, reference_only=True,
        policy_selection_used=False, certification_used=False, output_dir=str(Path(out).resolve()),
        scope='finite-bank offline empirical reference, conditional on A/RHS/budget/hardware; not deployable or theoretical upper bound')
    _write_json(Path(out) / 'offline_reference.json', result)
    with (Path(out) / 'offline_reference.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    return result
