"""Paired, no-training diagnostics against an A-only classical strategy selector.

This module never calls certify(), fits a neural network, or mutates a checkpoint.
Oracle results are an empirical best-of-tested-bank reference, with search cost.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import json
import os
import tempfile
import numpy as np

from ..provenance import hardware_environment, json_safe, operator_digest, stable_norm
from ..strategy import get_strategy
from .acceptance import _successful
from .config import AdaptiveConfig
from .data import dataset
from .limited import (digest_file, diagnostic_cases, forced_config, initialize_timing_runtime,
                      make_arm, read_settings, vector_digest)
from .solver import PreparedAdaptiveMG
from .strong import PreparedStrongMG, classical_bank, load_strong_rules, select_strong_strategy

REVISION = 'v67-strong-comparison-1'
BASELINES = ('fixed_C', 'strong_C')


def _hash(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _atomic_json(path, value):
    """A completed repeat is either fully durable or absent after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(json_safe(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def strong_output(output, research):
    output, research = Path(output).resolve(), Path(research).resolve()
    protected = [research, research.parent / 'my_v6_7_limited_fix']
    for tree in protected:
        if output == tree or tree in output.parents or output in tree.parents:
            raise ValueError('Output must be separate from original research and limited-fix results')
    output.mkdir(parents=True, exist_ok=True)
    return output


def strong_cases(research, settings, split='smoke', limit=None, sizes=None, seed=None, per_size=7):
    """New validation excludes normalized operators from every original split.

    Fresh draws remain diagnostic: operator disjointness is not evidence that a
    previously examined model, generator, or selector is an independent audit.
    """
    if split in ('tune', 'audit'):
        if seed is not None or per_size != 7:
            raise ValueError('Stored splits cannot change seed or per-size count')
        return diagnostic_cases(research, settings, split, limit, sizes)
    if split == 'train':
        if sizes or seed is not None or per_size != 7:
            raise ValueError('Stored train split cannot change generator settings')
        examples = dataset(settings['train_sizes'], settings['train_per_size'], int(settings['seed']))
        stored = json.loads((Path(research) / 'split_manifest.json').read_text())['splits']['train']
        if json_safe([e.manifest() for e in examples]) != stored:
            raise ValueError('Reconstructed train operator specifications differ from stored manifest')
        scope = 'offline calibration/diagnosis on original train split; not validation'
        used_seed = int(settings['seed'])
        excluded = 0
    elif split in ('smoke', 'fresh'):
        if per_size < 7:
            raise ValueError('New strong-comparison sets require at least seven family slots per size')
        used_seed = int(seed if seed is not None else int(settings['seed']) + (400004 if split == 'smoke' else 500005))
        original = json.loads((Path(research) / 'split_manifest.json').read_text())['splits']
        forbidden = {r['normalized_operator_digest'] for rows in original.values() for r in rows}
        examples, excluded = [], 0
        for e in dataset(sizes or [15], per_size, used_seed):
            if e.group_digest in forbidden:
                excluded += 1
                continue
            forbidden.add(e.group_digest)
            examples.append(e)
        if not examples:
            raise ValueError('All generated operators overlap original splits; choose a different seed')
        scope = ('new seven-family smoke diagnostic' if split == 'smoke' else
                 'new-seed operator-disjoint validation diagnostic; not an independent certificate')
    else:
        raise ValueError('Unknown data split')
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        examples = examples[:limit]
    records = [dict(e.manifest(), rhs_digest=vector_digest(e.b), exact_digest=vector_digest(e.exact),
                    x0_digest=vector_digest(np.zeros_like(e.b))) for e in examples]
    return examples, dict(split=split, scope=scope, seed=used_seed, excluded_normalized_operators=excluded,
        family_coverage=sorted({str(e.case.pattern) if e.case.pattern != 'uniform' else e.name.split('_' + str(e.n))[0]
                                for e in examples}),
        verification='same A, b, zero x0, tolerance and cycle budget for every arm and repeat',
        cases=records, independent_certification_evidence=False)


def build_strong_arms(research, cfg, branches=('H_S',), checkpoint=None):
    branches = tuple(branches)
    if len(set(branches)) != len(branches) or set(branches) - {'H_S', 'H_P', 'H_SP'}:
        raise ValueError('branches must be a distinct subset of H_S,H_P,H_SP')
    arms = {}
    for name in BASELINES:
        arms[name] = dict(make_arm(None, forced_config(cfg, 'C'), 'C',
            'fixed classical' if name == 'fixed_C' else 'deployable deterministic A-only classical'),
            selector=name == 'strong_C', oracle=False)
    if not branches:
        return arms
    stages = {'H_S': 'smoother', 'H_P': 'transfer', 'H_SP': 'joint'}
    # Always retain H_S on the original baseline for continuity; weights stay read-only.
    for branch in dict.fromkeys(('H_S',) + branches):
        path = Path(checkpoint) if checkpoint else Path(research) / stages[branch] / 'candidate.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Requested saved {branch} checkpoint missing: {path}')
        arm = make_arm(path, forced_config(cfg, branch), branch, 'forced component with rollback; no policy refit')
        if branch == 'H_S':
            arms['fixed_H_S'] = dict(arm, selector=False, oracle=False)
        if branch in branches:
            arms['strong_' + branch] = dict(arm, selector=True, oracle=False)
    return arms


def _hierarchy(level):
    rows = []
    while level is not None:
        rows.append(dict(grid_shape=list(level.shape), operator_shape=list(level.a.shape),
                         operator_nnz=int(level.a.nnz),
                         prolongation_shape=list(level.p.shape) if level.p is not None else None,
                         prolongation_nnz=int(level.p.nnz) if level.p is not None else 0))
        level = level.coarse
    return rows


def measured_strong(example, arm, rules):
    """Imported-process cold constructor+solve, including selector and rollback."""
    cfg = arm['config']
    started = perf_counter()
    prepared = None
    failed_selection = None
    try:
        if arm.get('selector'):
            prepared = PreparedStrongMG(example.a, example.n, arm['model'], cfg, rules=rules)
        else:
            prepared = PreparedAdaptiveMG(example.a, example.n, arm['model'], cfg)
        result = prepared.solve(example.b, np.zeros_like(example.b))
        wall = perf_counter() - started
        actual_cfg = prepared.config
        error = result.x - example.exact
        exact_energy = float(example.exact @ (example.a @ example.exact))
        error_energy = float(error @ (example.a @ error))
        energy_error = float(np.sqrt(max(error_energy, 0.) / exact_energy)) if exact_energy > 0 and np.isfinite(error_energy) else None
        run = result.to_dict()
        run.update(wall_seconds=wall, time_scope='cold', measurement_config=actual_cfg.to_dict(),
            residual_reference_norm=result.residual_history[0] if cfg.mg.residual_reference == 'initial' else stable_norm(example.b),
            relative_solution_error=stable_norm(result.x - example.exact) / max(stable_norm(example.exact), 1e-300),
            recomputed_true_residual=stable_norm(example.b - example.a @ result.x),
            relative_energy_error=energy_error,
            hierarchy_dimensions=_hierarchy(prepared.classical), learned_hierarchy_dimensions=_hierarchy(prepared.learned))
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        # A failing bank member remains in the table and can never win the oracle.
        failed_wall = perf_counter() - started
        # Preserve attempted strategy evidence even if hierarchy construction failed.
        # This second extraction is a post-timing diagnostic, never an online probe.
        if prepared is None and arm.get('selector'):
            try:
                failed_selection = select_strong_strategy(example.a, example.n, rules)
            except (ValueError, RuntimeError, FloatingPointError):
                pass
        failed_cfg = prepared.config if prepared else (replace(cfg, mg=replace(cfg.mg, strategy_name=failed_selection.strategy_name)) if failed_selection else cfg)
        run = dict(converged=False, wall_seconds=failed_wall, time_scope='cold',
                   measurement_config=failed_cfg.to_dict(),
                   stop_reason='measurement_exception', error=f'{type(exc).__name__}: {exc}',
                   stats={}, setup_stats={}, final_true_residual=None, executed_cycles=None)
    strategy_name = run['measurement_config']['mg']['strategy_name']
    strategy = get_strategy(strategy_name)
    run.update(operator_digest=operator_digest(example.a), rhs_digest=vector_digest(example.b),
        x0_digest=vector_digest(np.zeros_like(example.b)), checkpoint_id=arm['checkpoint_id'],
        requested_branch=arm['branch'], strategy_name=strategy_name,
        strategy_components=dict(smoother=strategy.smoother, transfer=strategy.transfer, coarsening=strategy.coarsening),
        selector_in_wall_time=bool(arm.get('selector')),
        selection=prepared.selection.to_dict() if prepared is not None and arm.get('selector') else (failed_selection.to_dict() if failed_selection else None),
        selection_origin='reconstructed_after_failed_constructor' if failed_selection else 'timed_constructor' if arm.get('selector') else 'fixed')
    return json_safe(run)


def _valid_time(value):
    return isinstance(value, (float, int)) and np.isfinite(value) and value > 0


def _run_success(r, cfg):
    checked = r.get('recomputed_true_residual')
    return bool(_successful(r, cfg) and _valid_time(r.get('wall_seconds'))
        and isinstance(checked, (float, int)) and np.isfinite(checked)
        and 0 <= checked <= r['stopping_threshold'])


def _success(runs, cfg):
    return bool(runs) and all(_run_success(r, cfg) for r in runs)


def _median(runs, key):
    values = [r.get(key) for r in runs]
    return float(np.median(values)) if values and all(isinstance(v, (float, int)) and np.isfinite(v) for v in values) else None


def _counters(runs):
    total = {}
    for run in runs:
        for container in ('stats', 'setup_stats'):
            for k, v in run.get(container, {}).items():
                if isinstance(v, (float, int)) and not isinstance(v, bool):
                    total[k] = total.get(k, 0) + v
    return total


def _speed_summary(ratios):
    if not ratios:
        return dict(geometric_speedup=None, ci95=[None, None], cases=0)
    logs = np.log(ratios)
    samples = np.exp(np.random.default_rng(93).choice(logs, (2000, len(logs)), replace=True).mean(axis=1))
    return dict(geometric_speedup=float(np.exp(logs.mean())), ci95=np.quantile(samples, [.025, .975]).tolist(), cases=len(logs))


def aggregate_strong_comparison(rows, arms, *, min_cases=20, minimum_speedup=1.0, accuracy_rtol=.05, accuracy_atol=1e-12):
    """All-repeat success, explicit pair cohorts, and one common deployment cohort."""
    deployment = [name for name, spec in arms.items() if not spec.get('oracle')]
    oracle_arms = [name for name, spec in arms.items() if spec.get('oracle')]
    successful = [{name: _success(row['runs'][name], arm['config']) for name, arm in arms.items()} for row in rows]
    common_indices = [i for i, ok in enumerate(successful) if all(ok[name] for name in deployment)]
    common_ids = [rows[i]['example']['name'] for i in common_indices]
    oracle, table, summary = [], [], {}
    for index, row in enumerate(rows):
        eligible = [name for name in oracle_arms if successful[index][name]]
        best = min(eligible, key=lambda name: (_median(row['runs'][name], 'wall_seconds'), name)) if eligible else None
        oracle.append(dict(case_id=row['example']['name'], selected_arm=best,
            selected_strategy=arms[best]['config'].mg.strategy_name if best else None,
            successful_median_seconds=_median(row['runs'][best], 'wall_seconds') if best else None,
            tested_strategies=len(oracle_arms), all_repeat_success_strategies=len(eligible),
            failed_strategy_ids=[arms[name]['config'].mg.strategy_name for name in oracle_arms if not successful[index][name]],
            search_total_seconds=sum(r.get('wall_seconds', 0.) for name in oracle_arms for r in row['runs'][name]),
            selected_baseline_strategy=row['runs']['strong_C'][0].get('strategy_name')))
    for name, arm in arms.items():
        pair_ratios = {base: [] for base in BASELINES + ('oracle',)}
        shared_ratios = {base: [] for base in BASELINES + ('oracle',)}
        loss = {base: [] for base in BASELINES}
        rescue = {base: [] for base in BASELINES}
        common_pairs = {base: [] for base in BASELINES}
        neural_cases, accepted_neural_cases, worse_accuracy, unavailable_accuracy = [], [], [], []
        error_values = {'relative_solution_error': [], 'relative_energy_error': []}
        for index, row in enumerate(rows):
            runs = row['runs'][name]
            ok = successful[index][name]
            case_id = row['example']['name']
            seconds = _median(runs, 'wall_seconds')
            counters = _counters(runs)
            numerical = counters.get('neural_apply_calls', 0) + counters.get('learned_transfer_apply_calls', 0)
            if numerical:
                neural_cases.append(case_id)
            if numerical and counters.get('accepted_neural_cycles', 0):
                accepted_neural_cases.append(case_id)
            accuracy = {}
            for metric in error_values:
                candidate_errors = [r.get(metric) for r in runs]
                baseline_errors = [r.get(metric) for r in row['runs']['strong_C']]
                valid_candidate_errors = all(isinstance(v, (float, int)) and np.isfinite(v) and v >= 0 for v in candidate_errors)
                valid_errors = valid_candidate_errors and all(isinstance(v, (float, int)) and np.isfinite(v) and v >= 0 for v in baseline_errors)
                if valid_candidate_errors:
                    error_values[metric].extend(candidate_errors)
                accuracy[metric + '_max'] = max(candidate_errors) if valid_candidate_errors else None
                if valid_errors:
                    accuracy[metric + '_not_worse_vs_strong'] = max(candidate_errors) <= max(baseline_errors) * (1 + accuracy_rtol) + accuracy_atol
                else:
                    accuracy[metric + '_not_worse_vs_strong'] = None
            not_worse = [accuracy[key + '_not_worse_vs_strong'] for key in error_values]
            if any(value is False for value in not_worse):
                worse_accuracy.append(case_id)
            if any(value is None for value in not_worse):
                unavailable_accuracy.append(case_id)
            speedups = {}
            for baseline in BASELINES + ('oracle',):
                bok = oracle[index]['selected_arm'] is not None if baseline == 'oracle' else successful[index][baseline]
                base_seconds = oracle[index]['successful_median_seconds'] if baseline == 'oracle' else _median(row['runs'][baseline], 'wall_seconds')
                ratio = base_seconds / seconds if bok and ok else None
                speedups['speedup_vs_' + baseline] = ratio
                if ratio is not None:
                    pair_ratios[baseline].append(ratio)
                    if index in common_indices:
                        shared_ratios[baseline].append(ratio)
                if baseline in BASELINES:
                    if bok and ok:
                        common_pairs[baseline].append(case_id)
                    if bok and not ok:
                        loss[baseline].append(case_id)
                    if ok and not bok:
                        rescue[baseline].append(case_id)
            table.append(dict(case_id=case_id, n=row['example']['case']['n'], arm=name, branch=arm['branch'],
                checkpoint_id=arm['checkpoint_id'], success=ok, repeats=len(runs),
                successful_repeats=sum(_run_success(r, arm['config']) for r in runs),
                total_seconds=seconds, successful_time_to_tolerance_seconds=seconds if ok else None,
                setup_seconds=_median(runs, 'setup_seconds'), solve_seconds=_median(runs, 'solve_seconds'),
                selector_seconds=_median([r.get('selection') or {} for r in runs], 'selector_seconds'),
                final_residual=_median(runs, 'final_true_residual'), relative_residual=_median(runs, 'relative_residual'),
                recomputed_true_residual=_median(runs, 'recomputed_true_residual'),
                relative_solution_error=_median(runs, 'relative_solution_error'),
                relative_energy_error=_median(runs, 'relative_energy_error'), **accuracy,
                tolerance=arm['config'].mg.tolerance, absolute_tolerance=arm['config'].mg.absolute_tolerance,
                residual_reference=arm['config'].mg.residual_reference,
                stopping_threshold=_median(runs, 'stopping_threshold'), max_cycles=arm['config'].mg.max_cycles,
                committed_cycles=_median(runs, 'cycles'), attempted_cycles=_median(runs, 'executed_cycles'),
                strategy_names=sorted({r.get('strategy_name', arm['config'].mg.strategy_name) for r in runs}),
                strategy_components=runs[0].get('strategy_components'), selection=runs[0].get('selection'),
                hierarchy_dimensions=runs[0].get('hierarchy_dimensions'), learned_hierarchy_dimensions=runs[0].get('learned_hierarchy_dimensions'),
                actually_executed_branches=sorted({b for r in runs for b in r.get('actually_executed_branches', [])}),
                actual_p_bank=sorted({r.get('p_bank', 'unknown') for r in runs}),
                nn_numerical_applications=numerical, accepted_neural=counters.get('accepted_neural_cycles', 0),
                rejected_neural=counters.get('rejected_neural_cycles', 0), rollback=counters.get('rollback_count', 0),
                fallback=counters.get('fallback_cycles', 0), counters_sum_all_repeats=counters,
                in_all_deployment_common_cohort=index in common_indices, **speedups))
        by_reference = {base: dict(_speed_summary(pair_ratios[base]),
            common_cohort=_speed_summary(shared_ratios[base])) for base in pair_ratios}
        strong = by_reference['strong_C']['common_cohort']
        checks = dict(minimum_unique_operator_count=len({rows[i]['example'].get('normalized_operator_digest', rows[i]['example']['name']) for i in common_indices}) >= min_cases,
            all_deployment_arms_succeed_on_all_cases=len(common_indices) == len(rows) and bool(rows),
            no_losses_vs_strong=not loss['strong_C'],
            exact_solution_errors_not_worse_vs_strong=not worse_accuracy and not unavailable_accuracy,
            actual_accepted_neural_work=bool(set(accepted_neural_cases) & set(common_ids)),
            enough_repeats=bool(rows) and all(len(row['runs'][name]) >= 5 for row in rows),
            common_cohort_ci_lower_exceeds_threshold=strong['ci95'][0] is not None and strong['ci95'][0] > minimum_speedup)
        summary[name] = dict(successes=sum(ok[name] for ok in successful), total=len(rows),
            comparisons=by_reference, common_success_case_ids=common_pairs,
            new_failure_case_ids=loss, rescued_case_ids=rescue, actual_neural_case_ids=neural_cases,
            accepted_neural_case_ids=accepted_neural_cases,
            accuracy={metric: dict(maximum=max(values) if values else None, median=float(np.median(values)) if values else None) for metric, values in error_values.items()},
            accuracy_worse_vs_strong_case_ids=worse_accuracy, accuracy_unavailable_case_ids=unavailable_accuracy,
            diagnostic_superiority_eligible=all(checks.values()), diagnostic_superiority_checks=checks,
            independent_certification_evidence=False, final_time_improvement_certified=False)
    quality = []
    for index, row in enumerate(rows):
        strong_time = _median(row['runs']['strong_C'], 'wall_seconds')
        best_time = oracle[index]['successful_median_seconds']
        quality.append(dict(oracle[index], selector_success=successful[index]['strong_C'],
            selector_over_best_tested_time=strong_time / best_time if successful[index]['strong_C'] and best_time else None))
    return dict(summary=summary, table=table, offline_oracle=quality,
        oracle_scope='empirical best all-repeat-success median among genuinely tested strategies; offline reference, not deployable or theoretical bound',
        oracle_search_cost_scope='sum of measured bank constructor+solve wall times; warmups excluded',
        common_cohort_case_ids=common_ids, common_cohort_arms=deployment,
        repeat_failure_policy='Every repeat must meet true-residual tolerance within attempted-cycle budget; failed runtimes never yield speedups',
        confidence_scope='paired operator bootstrap of per-operator median wall-time ratios, conditional on the named success cohort; repeated solves are not independent operators',
        superiority_scope='diagnostic eligibility only; fresh seed and disjoint digests do not establish untouched independent audit',
        accuracy_scope=dict(reference='manufactured exact discrete solution; diagnostics computed after timing', relative_error_slack=accuracy_rtol, absolute_error_slack=accuracy_atol,
            comparison='case maximum over repeats; relative L2 and energy errors must both be no worse within explicit slack for diagnostic superiority eligibility; stopping rule unchanged'),
        final_certificate_modified=False, no_training=True)


def _source_signature():
    source_root = Path(__file__).resolve().parents[1]
    files = sorted(source_root.rglob('*.py'))
    return _hash({str(path.relative_to(source_root)): digest_file(path) for path in files})


def _run_inputs(example, arm, repeat, manifest_digest):
    return dict(manifest_digest=manifest_digest, example=example.manifest(),
        rhs_digest=vector_digest(example.b), x0_digest=vector_digest(np.zeros_like(example.b)),
        exact_digest=vector_digest(example.exact), arm=arm, repeat=repeat)


def _validate_saved_run(saved, inputs, arm, example):
    if saved.get('inputs') != json_safe(inputs) or saved.get('run_digest') != _hash(saved.get('run')):
        raise ValueError('Resume repeat input/hash mismatch')
    run = saved['run']
    if (run.get('operator_digest') != example.digest or run.get('rhs_digest') != vector_digest(example.b)
            or run.get('x0_digest') != vector_digest(np.zeros_like(example.b))
            or run.get('requested_branch') != arm['branch'] or run.get('checkpoint_id') != arm['checkpoint_id']):
        raise ValueError('Resume saved run operator/RHS/checkpoint/branch mismatch')
    actual = AdaptiveConfig.from_dict(run['measurement_config'])
    expected = arm['config']
    if arm.get('selector'):
        expected = replace(expected, mg=replace(expected.mg, strategy_name=actual.mg.strategy_name))
    if actual != expected:
        raise ValueError('Resume saved run numerical configuration mismatch')
    return run


def run_strong_benchmark(examples, provenance, arms, out, research, rules=None, *, repeats=5,
                         warmups=1, resume=False, oracle_bank=None, order_seed=1007,
                         min_cases=20, minimum_speedup=1.0):
    if repeats < 1 or warmups < 0 or min_cases < 1 or minimum_speedup < 1:
        raise ValueError('Invalid repeat/warmup/minimum count or speedup threshold')
    if not examples or any(name not in arms for name in BASELINES):
        raise ValueError('Need examples and both fixed_C/strong_C baselines')
    rules = rules or load_strong_rules()
    arms = dict(arms)
    if oracle_bank:
        for strategy in classical_bank(oracle_bank):
            strategy_name = strategy.name if hasattr(strategy, 'name') else str(strategy)
            cfg = arms['fixed_C']['config']
            cfg = replace(cfg, mg=replace(cfg.mg, strategy_name=strategy_name))
            arms['oracle::' + strategy_name] = dict(make_arm(None, cfg, 'C', 'offline classical bank measurement'),
                                                   selector=False, oracle=True)
    out = strong_output(out, research)
    native_ready = initialize_timing_runtime()
    # Freeze every input affecting repeat selection or cost; no source/config mixing on resume.
    serial_arms = {name: {key: value for key, value in arm.items() if key not in ('model', 'config')}
                   | {'config': arm['config'].to_dict()} for name, arm in arms.items()}
    def common_contract(config):
        contract = config.to_dict()
        contract['mg'].pop('strategy_name')
        for name in ('branch', 'mode', 'use_smoother', 'use_transfer'):
            contract.pop(name)
        return contract
    if len({_hash(common_contract(arm['config'])) for arm in arms.values()}) != 1:
        raise ValueError('All arms must share tolerance, budget, sweeps, replacement budget, precision, device and threads')
    case_inputs = [dict(e.manifest(), rhs_digest=vector_digest(e.b), exact_digest=vector_digest(e.exact),
                        x0_digest=vector_digest(np.zeros_like(e.b))) for e in examples]
    spec = dict(revision=REVISION, source_signature=_source_signature(), provenance=provenance,
        case_inputs=case_inputs, repeats=repeats, warmups=warmups, order_seed=order_seed,
        arms=serial_arms, rules=rules.to_dict(), rules_digest=rules.digest(), oracle_bank=oracle_bank,
        environment=hardware_environment(), native_library_available=native_ready,
        min_cases=min_cases, minimum_speedup=minimum_speedup,
        time_scope='cold per-system selector(if applicable)+constructor+solve wall time; includes numerical safeguards, rejected trials, recovery and learned setup',
        excluded_costs='process imports, checkpoint loading, existing native-library loading, diagnostic extraction, warmups; no autotune/compilation',
        checkpoint_scope='frozen original fixed-baseline weights applied over selected classical hierarchy; baseline-transfer ablation, no refit or certificate',
        no_training=True, independent_certification_evidence=False)
    manifest = out / 'run_manifest.json'
    if manifest.exists():
        if not resume:
            raise FileExistsError('Use a fresh output directory or --resume with identical inputs')
        if json.loads(manifest.read_text()) != json_safe(spec):
            raise ValueError('Resume input/rules/source/config/environment signature mismatch')
    else:
        if resume:
            raise FileNotFoundError('No benchmark manifest exists to resume')
        if any(out.iterdir()):
            raise FileExistsError('New benchmark output directory must be empty')
        _atomic_json(manifest, spec)
    manifest_digest = _hash(spec)
    rows = []
    warmup_accounting = []
    for index, example in enumerate(examples):
        case_dir = out / 'repeats' / f'{index:04d}'
        case_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(order_seed + index)
        order = [rng.permutation(list(arms)).tolist() for _ in range(repeats)]
        warmup_file = case_dir / 'warmups.json'
        if warmup_file.exists():
            stored_warmup = json.loads(warmup_file.read_text())
            if stored_warmup.get('manifest_digest') != manifest_digest:
                raise ValueError('Resume warmup manifest mismatch')
        else:
            costs = {name: [] for name in arms}
            for _ in range(warmups):
                for name, arm in arms.items():
                    costs[name].append(measured_strong(example, arm, rules)['wall_seconds'])
            stored_warmup = dict(manifest_digest=manifest_digest, costs=costs)
            _atomic_json(warmup_file, stored_warmup)
        warmup_accounting.append(stored_warmup['costs'])
        runs = {name: [] for name in arms}
        for repeat, execution_order in enumerate(order):
            for name in execution_order:
                arm_index = list(arms).index(name)
                rawfile = case_dir / f'{repeat:03d}_{arm_index:03d}.json'
                inputs = _run_inputs(example, name, repeat, manifest_digest)
                if rawfile.exists():
                    run = _validate_saved_run(json.loads(rawfile.read_text()), inputs, arms[name], example)
                else:
                    run = measured_strong(example, arms[name], rules)
                    run.update(repeat=repeat, arm=name)
                    _atomic_json(rawfile, dict(inputs=inputs, run=run, run_digest=_hash(run)))
                runs[name].append(run)
            _atomic_json(out / 'progress.json', dict(status='running', completed_cases=len(rows),
                case_id=example.name, completed_repeats_current_case=repeat + 1, total_cases=len(examples), repeats=repeats))
            print(f'[NO TRAINING] {index + 1}/{len(examples)} {example.name}: repeat {repeat + 1}/{repeats}', flush=True)
        selections = [(r['selection']['strategy_name'], r['selection']['rule_id'], r['selection']['rules_digest'])
            for name in arms if arms[name].get('selector') for r in runs[name] if r.get('selection')]
        if len(set(selections)) > 1:
            raise ValueError('A-only selector gave different decisions across arms/repeats on identical A')
        rows.append(dict(example=example.manifest(), runs=runs, execution_order=order))
    comparison = aggregate_strong_comparison(rows, arms, min_cases=min_cases, minimum_speedup=minimum_speedup)
    for index, item in enumerate(comparison['offline_oracle']):
        item['search_warmup_seconds'] = sum(sum(times) for name, times in warmup_accounting[index].items() if arms[name].get('oracle'))
        item['search_total_including_warmups_seconds'] = item['search_total_seconds'] + item['search_warmup_seconds']
    quality = comparison['offline_oracle']
    comparison['selector_quality'] = dict(
        scope='candidate strong baseline; measured quality must be assessed before claiming it is a tuned strong competitor',
        selector_failed_while_bank_succeeded_case_ids=[r['case_id'] for r in quality if not r['selector_success'] and r['selected_arm']],
        selected_same_as_empirical_oracle_case_ids=[r['case_id'] for r in quality if r['selected_strategy'] and r['selected_strategy'] == r['selected_baseline_strategy']],
        oracle_reference_depends_on='this A, RHS, x0, budget, hardware, repetitions and finite tested bank',
        per_case=quality)
    comparison.update(provenance=provenance, rules_digest=rules.digest(), run_manifest_digest=manifest_digest,
                      checkpoint_scope=spec['checkpoint_scope'])
    _atomic_json(out / 'raw_results.json', rows)
    _atomic_json(out / 'comparison.json', comparison)
    with (out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison['table'][0]))
        writer.writeheader()
        for row in comparison['table']:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    _atomic_json(out / 'progress.json', dict(status='complete', completed_cases=len(rows), total_cases=len(examples)))
    return comparison


def benchmark_main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='NO TRAINING: fixed/strong classical and frozen neural branches; diagnostics only.')
    parser.add_argument('--research-dir', default='artifacts/my_v6_7_research')
    parser.add_argument('--output-dir', default='artifacts/my_v6_7_strong_baseline/direct_smoke')
    parser.add_argument('--split', choices=('smoke', 'fresh', 'train', 'tune', 'audit'), default='smoke')
    parser.add_argument('--sizes', help='comma-separated grid sizes for new smoke/fresh sets, default 15')
    parser.add_argument('--seed', type=int, help='new generator seed for smoke/fresh only')
    parser.add_argument('--per-size', type=int, default=7, help='new set examples per size; >=7 for seven families')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--branches', default='H_S', help='H_S,H_P,H_SP; default H_S only, with fixed_H_S continuity')
    parser.add_argument('--classical-only', action='store_true', help='measure fixed/selected C and optional oracle bank without checkpoint loading')
    parser.add_argument('--checkpoint', help='explicit saved checkpoint for all requested branches; default stage candidates remain unchanged')
    parser.add_argument('--rules', help='frozen deterministic rule JSON; no fitting during benchmark')
    parser.add_argument('--oracle-bank', choices=('controlled', 'all'), help='measure every classical strategy in this bank; adds substantial offline search cost')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--order-seed', type=int, default=1007)
    parser.add_argument('--min-cases', type=int, default=20)
    parser.add_argument('--minimum-speedup', type=float, default=1.0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    print('NO TRAINING. Frozen checkpoint transfer diagnostic; no independent certificate or policy refit.', flush=True)
    research, settings, cfg = read_settings(args.research_dir)
    sizes = [int(x) for x in args.sizes.split(',')] if args.sizes else None
    examples, provenance = strong_cases(research, settings, args.split, args.limit, sizes, args.seed, args.per_size)
    branches = () if args.classical_only else tuple(x.strip() for x in args.branches.split(','))
    arms = build_strong_arms(research, cfg, branches, args.checkpoint)
    result = run_strong_benchmark(examples, provenance, arms, args.output_dir, research, load_strong_rules(args.rules),
        repeats=args.repeats, warmups=args.warmups, resume=args.resume, oracle_bank=args.oracle_bank,
        order_seed=args.order_seed, min_cases=args.min_cases, minimum_speedup=args.minimum_speedup)
    for name, summary in result['summary'].items():
        if name.startswith('oracle::'):
            continue
        speed = summary['comparisons']['strong_C']
        print(json.dumps(dict(arm=name, successes=summary['successes'], total=summary['total'],
            speedup_vs_strong_pairwise=speed['geometric_speedup'], ci95=speed['ci95'],
            all_deployment_common_cohort=speed['common_cohort'],
            new_failures_vs_strong=summary['new_failure_case_ids']['strong_C'],
            diagnostic_superiority_eligible=summary['diagnostic_superiority_eligible']), ensure_ascii=False))
    print('NO TRAINING. Results:', Path(args.output_dir).resolve())
