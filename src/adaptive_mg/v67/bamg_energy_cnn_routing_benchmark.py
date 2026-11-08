"""Compact, paired development validation for numerical and CNN hierarchies.

Each method sees the same operator and one RHS, with a fresh zero iterate for
every repetition. Only scalar measurements are persisted. Setup, CNN inference,
Galerkin products, factorization, and warmup are outside the warm solve timer.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..config import MGConfig
from .bamg_energy import array_digest
from .bamg_energy_study import candidate_hierarchy, digest, file_digest, read_json, write_json
from .bamg_solver import ReferenceModels, build_runtime, runtime_from_prolongations
from .bamg_teacher import TeacherConfig, build_teacher, matrix_digest
from .bamg_transfer import model_from_spec

VERSION = 'bamg-energy-cnn-routed-validation-v1'
BASELINES = ('fixed_C', 'em5_C')
NUMERICAL_BASELINES = (*BASELINES, 'bootstrap_C')
_FAILURES = (OSError, ValueError, RuntimeError, FloatingPointError, MemoryError,
             KeyError, TypeError, np.linalg.LinAlgError)


def numerical_baselines(settings):
    """Opt in to BAMG without changing existing single-teacher experiments."""
    names = settings.get('numerical_baselines', BASELINES)
    if (not isinstance(names, (list, tuple)) or not names or 'fixed_C' not in names
            or any(not isinstance(name, str) or name not in NUMERICAL_BASELINES for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('Distinct supported numerical baselines including fixed_C are required')
    return tuple(names)


def _descriptor(value):
    """Validate the complete routing artifact before accepting any measurements."""
    from .bamg_energy_cnn_routing import lambda_key, validate_policy
    if (set(value) != {'kind', 'policy', 'checkpoints'}
            or value['kind'] != 'grid_router_v1'):
        raise ValueError('Invalid grid router checkpoint descriptor')
    policy = validate_policy(value['policy'])
    paths = value['checkpoints']
    expected = {lambda_key(v) for v in policy['candidates']}
    if not isinstance(paths, dict) or set(paths) != expected:
        raise ValueError('Grid router must bind every candidate checkpoint')
    if any(not isinstance(path, (str, Path)) for path in paths.values()):
        raise ValueError('Grid router checkpoint paths must be paths')
    return policy, {key: Path(path) for key, path in paths.items()}


def checkpoint_files(checkpoints):
    """All model files, including routing alternatives unused on a given grid."""
    paths = []
    for value in checkpoints.values():
        if isinstance(value, dict):
            _, choices = _descriptor(value)
            paths.extend(choices.values())
        elif isinstance(value, (str, Path)):
            paths.append(Path(value))
        else:
            raise ValueError('Checkpoint must be a path or a grid router descriptor')
    return sorted(set(paths), key=str)


def checkpoint_hashes(checkpoints, *, allow_missing=False):
    """Bind every candidate tensor file and the complete validation-only policy."""
    def fingerprint(path):
        return file_digest(path) if not allow_missing or path.is_file() else None
    result = {}
    for name, value in sorted(checkpoints.items()):
        if isinstance(value, dict):
            policy, paths = _descriptor(value)
            result[name] = dict(kind='grid_router_v1', policy=policy,
                checkpoint_hashes={key: fingerprint(path) for key, path in sorted(paths.items())})
        elif isinstance(value, (str, Path)):
            result[name] = fingerprint(Path(value))
        else:
            raise ValueError('Checkpoint must be a path or a grid router descriptor')
    return result


def resolve_checkpoint(example, checkpoint):
    """Choose one model from the finest grid; MG recursion never routes again."""
    if not isinstance(checkpoint, dict):
        return checkpoint, None
    from .bamg_energy_cnn_routing import lambda_key, route_lambda
    policy, paths = _descriptor(checkpoint)
    shape = tuple(example['shape'])
    if (len(shape) != 2 or shape[0] != shape[1]
            or any(type(n) not in (int, np.int64, np.int32) for n in shape)):
        raise ValueError('Grid routing requires an integer square finest grid')
    strength = route_lambda(policy, int(shape[0]))
    path = paths[lambda_key(strength)]
    audit = dict(kind='grid_router_v1', finest_n=int(shape[0]),
                 selected_lambda=strength, selected_checkpoint=str(path),
                 selected_checkpoint_sha256=file_digest(path) if path.is_file() else None,
                 policy_digest=digest(policy), one_model_for_entire_hierarchy=True)
    return path, audit


def _case_path(output, case_id):
    # A case id is provenance, never a user-supplied filesystem path.
    key = hashlib.sha256(case_id.encode()).hexdigest()
    return Path(output) / 'benchmark_cases' / (key + '.json')


def _settings(settings):
    bench = dict(settings['benchmark'])
    numerical_baselines(settings)
    if type(bench.get('cache_hierarchies', True)) is not bool:
        raise ValueError('benchmark.cache_hierarchies must be boolean')
    for key, minimum in (('repetitions', 1), ('warmup_cycles', 0)):
        if type(bench.get(key)) is not int or bench[key] < minimum:
            raise ValueError(f'benchmark.{key} must be an integer >= {minimum}')
    for key in ('timeout_seconds', 'failure_penalty_seconds'):
        if not math.isfinite(bench.get(key, float('nan'))) or bench[key] <= 0:
            raise ValueError(f'benchmark.{key} must be finite and positive')
    allowed = bench.get('allowed_failure_rate', 0.)
    if not math.isfinite(allowed) or not 0 <= allowed <= 1:
        raise ValueError('benchmark.allowed_failure_rate must be in [0, 1]')
    bench['allowed_failure_rate'] = allowed
    if settings.get('support_mode') != 'fixed_bilinear':
        raise ValueError('Paired energy CNN comparison requires fixed_bilinear support')
    cfg = MGConfig(**settings['mg'])
    if (cfg.tolerance != 1e-8 or cfg.absolute_tolerance != 1e-12 or
            cfg.residual_reference != 'initial'):
        raise ValueError('RuntimeBank uses max(1e-12, 1e-8 * initial residual)')
    return bench


def _manifest(examples, settings, checkpoints, identity):
    bench = _settings(settings)
    if not examples:
        raise ValueError('Full development validation must contain cases')
    if any(method in NUMERICAL_BASELINES for method in checkpoints):
        raise ValueError('Checkpoint names cannot replace numerical baselines')
    if any(not isinstance(name, str) or not name for name in checkpoints):
        raise ValueError('Checkpoint method names must be nonempty strings')
    cases, seen = [], set()
    for example in examples:
        case_id = example['case_id']
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError('Development validation case ids must be unique strings')
        seen.add(case_id)
        for key in ('split', 'research_split'):
            if example.get(key, 'validation') != 'validation':
                raise ValueError('Benchmark accepts development validation, never TRAIN/final/OOD')
        shape = tuple(int(n) for n in example['shape'])
        a, rhs = example['A'], np.asarray(example['b'], dtype=np.float64)
        if (len(shape) != 2 or a.shape != (int(np.prod(shape)),) * 2 or
                rhs.shape != (a.shape[0],) or not np.isfinite(rhs).all()):
            raise ValueError('Each validation case needs one finite vector RHS and matching grid')
        cases.append(dict(case_id=case_id, family=example['family'], shape=list(shape),
                          A_digest=matrix_digest(a), rhs_digest=array_digest(rhs),
                          normalized_operator_digest=example.get('normalized_operator_digest')))
    manifest = dict(version=VERSION, identity=identity, settings_digest=digest(settings),
                    checkpoint_hashes=checkpoint_hashes(checkpoints, allow_missing=True),
                    methods=[*numerical_baselines(settings), *sorted(checkpoints)], cases=cases,
                    repetitions=bench['repetitions'], rhs_count=1,
                    warmup_cycles=bench['warmup_cycles'], timeout_seconds=bench['timeout_seconds'],
                    failure_penalty_seconds=bench['failure_penalty_seconds'],
                    allowed_failure_rate=bench['allowed_failure_rate'],
                    selection_workload='warm_rhs_1', setup_excluded=True, fallback=False,
                    initial_iterate='zero',
                    aggregation='arithmetic mean of per-case medians across repetitions')
    # Normalize tuple/list differences before comparing persisted identities.
    return json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))


def _load_weight(path, settings):
    package = torch.load(path, map_location='cpu', weights_only=True)
    spec = package.get('weight_spec', package.get('specs', {}).get('weight'))
    state = package.get('weight_state', package.get('models', {}).get('weight'))
    expected = dict(kind='weight', **settings['models']['weight'])
    if spec != expected or state is None:
        raise ValueError('Checkpoint Weight CNN architecture does not match the comparison')
    model = model_from_spec(spec).double()
    model.load_state_dict(state, strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Checkpoint contains nonfinite Weight CNN parameters')
    return model.eval().requires_grad_(False)


def _build_bank(example, method, checkpoint, settings):
    cfg = MGConfig(**settings['mg'])
    if method in numerical_baselines(settings):
        tc = TeacherConfig(**{**settings['teacher'], 'k': settings['k'],
                              'l1_cap': settings['l1_cap'], 'support_mode': 'fixed_bilinear'})
        vectors, original_ps, diagnostics = None, None, None
        if method == 'bootstrap_C':
            teacher = build_teacher(example['A'], example['shape'], config=tc, mg_config=cfg)
            vectors = [level.vectors for level in teacher.levels[:-1]]
            original_ps = [level.P for level in teacher.levels[:-1]]
            diagnostics = dict(selected_pass=teacher.diagnostics['selected_pass'],
                               failed_passes=teacher.diagnostics['failed_passes'],
                               role='original BAMG selected pass; setup-only vectors; no timing RHS')
        ps, _ = candidate_hierarchy(example['A'], example['shape'], method, None,
                                    vectors, original_ps, tc, cfg, {})
        bank = runtime_from_prolongations(example['A'], example['shape'], ps, cfg,
                                         caps=settings.get('caps'))
        if diagnostics is not None:
            bank._benchmark_setup_metadata = {'bootstrap_teacher': diagnostics}
        return bank
    checkpoint, _ = resolve_checkpoint(example, checkpoint)
    weight = _load_weight(checkpoint, settings)
    reference = ReferenceModels(None, None, k=settings['k'], l1_cap=settings['l1_cap'],
                                mg_config=cfg, caps=settings.get('caps'),
                                support_mode='fixed_bilinear')
    return build_runtime(example['A'], example['shape'], weight, reference, cfg)


def _failure(reason, elapsed=None):
    return dict(success=False, warm_seconds=elapsed, reason=str(reason)[:1000],
                cycles=None, tolerance=None, final_residual=None, fallback_used=False)


def _measurement(bank, rhs, bench):
    start = perf_counter()
    try:
        # Do not persist x, residual histories, per-cycle timings, or matrices.
        result = bank.solve(rhs, x0=np.zeros_like(rhs), timeout_seconds=bench['timeout_seconds'])
        elapsed = float(result['warm_seconds'])
        if not math.isfinite(elapsed) or elapsed <= 0:
            return _failure('invalid_warm_seconds')
        residual = float(result['residuals'][-1])
        tolerance = float(result['tolerance'])
        cycles = int(result['cycles'])
        success = bool(result['converged'])
        reason = str(result['reason'])[:1000]
        if not math.isfinite(residual) or not math.isfinite(tolerance) or tolerance < 0:
            success, reason = False, 'nonfinite_or_invalid_residual_tolerance'
        elif residual > tolerance or residual < 0:
            success = False
            if bool(result['converged']):
                reason = 'convergence_claim_failed_residual_check'
        if elapsed > bench['timeout_seconds']:
            success, reason = False, 'timeout'
        if result.get('fallback_used', False):
            success, reason = False, 'forbidden_fallback'
        if cycles < 0:
            success, reason = False, 'invalid_cycles'
        return dict(success=success, warm_seconds=elapsed, reason=reason,
                    cycles=cycles if cycles >= 0 else None,
                    tolerance=tolerance if math.isfinite(tolerance) else None,
                    final_residual=residual if math.isfinite(residual) else None,
                    fallback_used=bool(result.get('fallback_used', False)))
    except _FAILURES as exc:
        return _failure(f'{type(exc).__name__}: {exc}', perf_counter() - start)


def _read_case(output, case, manifest):
    path = _case_path(output, case['case_id'])
    if not path.exists():
        return dict(case=case, manifest_digest=digest(manifest), setups={}, measurements={})
    state = read_json(path)
    if state['case'] != case or state['manifest_digest'] != digest(manifest):
        raise ValueError('Benchmark case identity changed')
    for method, rows in state['measurements'].items():
        if method not in manifest['methods']:
            raise ValueError('Unexpected benchmark method in saved case')
        for repeat, row in rows.items():
            if repeat not in {str(r) for r in range(manifest['repetitions'])}:
                raise ValueError('Unexpected benchmark repetition in saved case')
            if type(row['success']) is not bool:
                raise ValueError('Invalid saved success flag')
            if row['success'] and (row['warm_seconds'] is None or
                    not math.isfinite(row['warm_seconds']) or row['warm_seconds'] <= 0):
                raise ValueError('Invalid successful benchmark timing')
    return state


def _penalized_seconds(row, penalty):
    if row is None:
        return penalty
    measured = row.get('warm_seconds')
    if row['success']:
        return measured
    if (isinstance(measured, (int, float)) and math.isfinite(measured)
            and measured >= 0):
        return max(penalty, measured)
    return penalty


def summarize(output):
    """Report partial validation with missing observations penalized, never selected."""
    output = Path(output)
    manifest = read_json(output / 'benchmark_manifest.json')
    repetitions = manifest['repetitions']
    penalty = manifest['failure_penalty_seconds']
    methods = {method: dict(attempts=len(manifest['cases']) * repetitions,
                            observed=0, missing=0, failures=0, case_times=[], case_cycles=[])
               for method in manifest['methods']}
    cases = []
    for case in manifest['cases']:
        state = _read_case(output, case, manifest)
        case_summary = {key: case[key] for key in ('case_id', 'family', 'shape')}
        case_summary['methods'] = {}
        for method, totals in methods.items():
            recorded = state['measurements'].get(method, {})
            rows = [recorded.get(str(r)) for r in range(repetitions)]
            failures = sum(row is None or not row['success'] for row in rows)
            missing = sum(row is None for row in rows)
            seconds = float(np.median([_penalized_seconds(row, penalty) for row in rows]))
            cycles = float(np.median([row['cycles'] for row in rows])) if not failures else None
            totals['observed'] += repetitions - missing
            totals['missing'] += missing
            totals['failures'] += failures
            totals['case_times'].append(seconds)
            if cycles is not None:
                totals['case_cycles'].append(cycles)
            case_summary['methods'][method] = dict(warm_seconds=seconds, cycles=cycles,
                failures=failures, missing=missing, attempts=repetitions)
        cases.append(case_summary)
    completed = all(value['missing'] == 0 for value in methods.values())
    for totals in methods.values():
        totals['warm_seconds'] = float(np.mean(totals.pop('case_times')))
        totals['warm_ms'] = 1000 * totals['warm_seconds']
        cycles = totals.pop('case_cycles')
        totals['cycles'] = float(np.mean(cycles)) if not totals['failures'] else None
        totals['failure_rate'] = totals['failures'] / totals['attempts']
        totals['eligible'] = completed and totals['failure_rate'] <= manifest['allowed_failure_rate']
    fixed = methods['fixed_C']
    for totals in methods.values():
        totals['speedup'] = (fixed['warm_seconds'] / totals['warm_seconds']
                             if fixed['eligible'] and totals['eligible'] else None)
    eligible = [name for name, value in methods.items() if value['eligible']]
    selected = (min(eligible, key=lambda name: (methods[name]['failures'],
                                               methods[name]['warm_seconds'], name))
                if eligible else None)
    return dict(version=VERSION, completed=completed, cases_expected=len(cases),
                cases_completed=sum(all(v['missing'] == 0 for v in c['methods'].values())
                                    for c in cases),
                rhs_count=1, repetitions=repetitions, setup_excluded=True,
                fallback=False, aggregation=manifest['aggregation'],
                failure_penalty_seconds=penalty, selected_method=selected,
                methods=methods, case_results=cases)


def benchmark(output, *, examples, settings, checkpoints, identity, resume=False):
    """Measure full paired validation, resuming individual scalar observations.

    The caller's identity must include its source/configuration provenance.
    This function additionally binds actual operator/RHS and checkpoint hashes.
    An interrupted setup or solve is rerun; already saved observations are not.
    """
    output = Path(output)
    manifest = _manifest(examples, settings, checkpoints, identity)
    manifest_path = output / 'benchmark_manifest.json'
    if manifest_path.exists():
        if not resume:
            raise FileExistsError('Benchmark exists; use --resume')
        if read_json(manifest_path) != manifest:
            raise ValueError('Benchmark sources, settings, cases, RHS, or checkpoint identity changed')
    else:
        if (output / 'benchmark_cases').exists():
            raise ValueError('Benchmark case files exist without their identity manifest')
        write_json(manifest_path, manifest)
    bench = _settings(settings)
    write_json(output / 'benchmark_summary.json', summarize(output))
    for example, case in zip(examples, manifest['cases']):
        state = _read_case(output, case, manifest)
        banks = {}
        rhs = np.asarray(example['b'], dtype=np.float64)
        for repeat in range(bench['repetitions']):
            # Stable per-case/per-repeat order is independent of resume progress.
            seed = int(digest([settings.get('seed', 0), case['case_id'], repeat])[:16], 16)
            order = np.random.default_rng(seed).permutation(manifest['methods']).tolist()
            for method in order:
                rows = state['measurements'].setdefault(method, {})
                if str(repeat) in rows:
                    continue
                if method in checkpoints and checkpoint_hashes(
                        {method: checkpoints[method]}, allow_missing=True)[method] != manifest['checkpoint_hashes'][method]:
                    raise ValueError('Validation checkpoint or routing policy changed while evaluating')
                print(f'CNN validation {case["case_id"]} {method} repeat {repeat + 1}/{bench["repetitions"]}',
                      flush=True)
                setup = state['setups'].get(method)
                if method not in banks and (setup is None or setup['success']):
                    started = perf_counter()
                    routing = None
                    try:
                        selected_checkpoint, routing = resolve_checkpoint(example, checkpoints.get(method))
                        with torch.no_grad():
                            bank = _build_bank(example, method, selected_checkpoint, settings)
                            for _ in range(bench['warmup_cycles']):
                                if not np.isfinite(bank.cycle(np.zeros_like(rhs), rhs)).all():
                                    raise FloatingPointError('Nonfinite warmup cycle')
                        banks[method] = bank
                        setup = dict(success=True, setup_seconds=perf_counter() - started,
                                     reason='prepared', fallback_used=False)
                        setup.update(getattr(bank, '_benchmark_setup_metadata', {}))
                    except _FAILURES as exc:
                        setup = dict(success=False, setup_seconds=perf_counter() - started,
                                     reason=f'{type(exc).__name__}: {exc}'[:1000], fallback_used=False)
                    if routing is not None:
                        setup['routing'] = routing
                    state['setups'][method] = setup
                if setup['success']:
                    rows[str(repeat)] = _measurement(banks[method], rhs, bench)
                else:
                    rows[str(repeat)] = _failure('setup_failed: ' + setup['reason'])
                write_json(_case_path(output, case['case_id']), state)
                if not bench.get('cache_hierarchies', True):
                    banks.pop(method, None)
                    bank = None
        write_json(output / 'benchmark_summary.json', summarize(output))
    return summarize(output)
