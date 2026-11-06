"""Teacher-only, paired development-validation comparison; never trains a CNN.

Frozen BAMG vector banks isolate the interpolation objective. Each candidate
still rebuilds its OWN actual Galerkin hierarchy. Deeper A_l therefore differ;
this is recorded, not disguised as a common-operator comparison at every level.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
import scipy.sparse as sp
import torch
from threadpoolctl import threadpool_limits

from ..config import MGConfig
from ..energymin import energy_weights
from ..grid import next_shape, terminal
from ..transfer import build_transfer_pattern
from .bamg_energy import FixedObjective, array_digest, minimize
from .bamg_solver import runtime_from_prolongations
from .bamg_teacher import TeacherConfig, build_teacher, matrix_digest
from .bamg_transfer import direct_pattern, fixed_bilinear_mask, scipy_from_weights
from .research_data import FAMILIES, _example, _specification

VERSION = 'bamg-energy-teacher-comparison-v1'
PROJECT = Path(__file__).resolve().parents[3]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate(settings):
    if settings.get('version') != VERSION or settings.get('reference_policy') != 'bilinear':
        raise ValueError('Use the declared fixed-bilinear-reference comparison version')
    cfg = MGConfig(**settings['mg'])
    if (cfg.strategy_name != 'line_alt_bilinear_full__v22' or cfg.stencil_backend != 'csr'
            or cfg.pre_steps != 2 or cfg.post_steps != 2 or cfg.coarsest_n != 3
            or cfg.device != 'cpu' or cfg.tolerance != 1e-8 or cfg.absolute_tolerance != 1e-12
            or cfg.residual_reference != 'initial' or cfg.max_cycles < 1):
        raise ValueError('Expected classical line_alt/full, 2+2 sweeps, CSR CPU and existing tolerance')
    tc = TeacherConfig(**settings['teacher'])
    if tc.support_mode != 'fixed_bilinear' or tc.weighting != 'uniform' or tc.setup_passes < 2:
        raise ValueError('Common-bank comparison needs bootstrap, fixed_bilinear and uniform vector weights')
    if tc.l1_cap != 8. or tc.mu <= 0:
        raise ValueError('Use positive ridge and the native EM safety cap 8 for matched constraints')
    strengths = settings['lambdas']
    if (not strengths or len(set(strengths)) != len(strengths) or 0 not in strengths
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in strengths)):
        raise ValueError('Distinct finite nonnegative lambdas including zero are required')
    data = settings['data']
    if data['discretization'] not in ('p1', 'q1') or not data['families'] or set(data['families']) - set(FAMILIES):
        raise ValueError('Invalid development families/discretization')
    if len(set(data['families'])) != len(data['families']) or len(set(data['sizes'])) != len(data['sizes']):
        raise ValueError('Duplicate development families/sizes')
    allowed = (7, 15, 31, 63) if settings.get('smoke') else (31, 63)
    if not data['sizes'] or any(n not in allowed for n in data['sizes']):
        raise ValueError('Research comparison uses n=31,63; smoke may use n=7,15')
    for name, value in [('per_family', data['per_family']), ('threads', settings['threads']),
                        ('repetitions', settings['benchmark']['repetitions']),
                        ('rhs_count', settings['benchmark']['rhs_count']),
                        ('maxiter', settings['optimizer']['maxiter'])]:
        if type(value) is not int or value < 1:
            raise ValueError('Expected positive integer ' + name)
    for seed in (settings['seed'], data['seed']):
        if type(seed) is not int or seed < 0:
            raise ValueError('Seeds must be nonnegative integers')
    bench = settings['benchmark']
    if type(bench['warmup_cycles']) is not int or bench['warmup_cycles'] < 0:
        raise ValueError('Invalid warmup budget')
    for key in ('timeout_seconds', 'failure_penalty_seconds'):
        if not math.isfinite(bench[key]) or bench[key] <= 0:
            raise ValueError('Invalid ' + key)
    if not 0 <= bench['allowed_failure_rate'] <= 1 or not 0 < settings['optimizer']['rtol'] < 1:
        raise ValueError('Invalid selection/optimizer threshold')
    return cfg, tc


def method_table(settings):
    result = {'fixed_C': None, 'em5_C': None, 'em_constrained_C': None, 'bootstrap_C': None}
    result.update({f'hybrid_lambda_{float(v):.12g}': float(v) for v in settings['lambdas']})
    if len(result) != 4 + len(settings['lambdas']):
        raise ValueError('Lambda names collide at serialized precision')
    return result


def specifications(settings):
    data = settings['data']
    records = [_specification('validation', data, n, family, index, 0)
               for n in data['sizes'] for family in data['families']
               for index in range(data['per_family'])]
    return json.loads(json.dumps(records))


def manifest(settings):
    paths = sorted((PROJECT / 'src/adaptive_mg').rglob('*.py'))
    paths += [PROJECT / 'scripts/run_v6_7_bamg_energy_study.py']
    return dict(version=VERSION, settings_digest=digest(settings),
        sources={str(p.relative_to(PROJECT)): file_digest(p) for p in paths},
        environment=dict(python=platform.python_version(), numpy=np.__version__,
            scipy=scipy.__version__, torch=str(torch.__version__), platform=platform.platform(),
            threads=settings['threads']), split='development_validation_only',
        final_or_ood_access=False, cnn_training=False)


def _save_arrays(path, arrays):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def vector_bank(example, directory, tc, cfg):
    """One common bootstrap bank per A; timing RHS never enters this bank."""
    metadata = directory / 'vectors.json'
    path = directory / 'vectors.npz'
    identity = dict(A_digest=matrix_digest(example.a), teacher=asdict(tc), mg=asdict(cfg))
    identity = json.loads(json.dumps(identity))
    if metadata.exists():
        record = read_json(metadata)
        if record['identity'] != identity:
            raise ValueError('Cached vector-bank identity changed')
        if record['status'] != 'ok':
            return None, None, record
        if file_digest(path) != record['sha256']:
            raise ValueError('Corrupt vector bank')
        with np.load(path, allow_pickle=False) as arrays:
            vectors = [arrays[f'z_{i}'].copy() for i in range(record['levels'])]
            ps = [sp.csr_matrix((arrays[f'p_{i}'], arrays[f'col_{i}'], arrays[f'ptr_{i}']),
                  shape=tuple(arrays[f'shape_{i}'])) for i in range(record['levels'])]
        return vectors, ps, record
    start = perf_counter()
    try:
        result = build_teacher(example.a, (example.n, example.n), config=tc, mg_config=cfg)
        vectors = [l.vectors.copy() for l in result.levels[:-1]]
        ps = [l.P.copy() for l in result.levels[:-1]]
        arrays = {}
        for i, (z, p) in enumerate(zip(vectors, ps)):
            arrays.update({f'z_{i}': z, f'p_{i}': p.data, f'col_{i}': p.indices,
                           f'ptr_{i}': p.indptr, f'shape_{i}': np.asarray(p.shape)})
        _save_arrays(path, arrays)
        record = dict(identity=identity, status='ok', levels=len(vectors), sha256=file_digest(path),
            generation_seconds=perf_counter() - start,
            selected_pass=result.diagnostics['selected_pass'],
            failed_passes=result.diagnostics['failed_passes'],
            vector_digests=[array_digest(z) for z in vectors],
            reference_A_digests=[matrix_digest(l.A) for l in result.levels],
            role='setup-only common bank; no benchmark RHS used')
    except (ValueError, RuntimeError, FloatingPointError, MemoryError, np.linalg.LinAlgError) as exc:
        vectors, ps = None, None
        record = dict(identity=identity, status='failed', generation_seconds=perf_counter() - start,
                      failure=f'{type(exc).__name__}: {exc}')
    write_json(metadata, record)
    return vectors, ps, record


def candidate_hierarchy(a, shape, method, strength, vectors, original_ps, tc, cfg, optimizer):
    """Build actual Galerkin matrices; never reuse the reference coarse A."""
    ps, reports = [], []
    current, sh, index = a, tuple(shape), 0
    while not terminal(sh, cfg.coarsest_n):
        cs = next_shape(sh, 'full', cfg.coarsest_n, level_index=index)
        pattern = direct_pattern(sh, cs)
        geometric = build_transfer_pattern(sh, cs)
        mask = fixed_bilinear_mask(pattern).cpu().numpy()
        common = dict(level=index, shape=list(sh), A_digest=matrix_digest(current),
                      support_digest=array_digest(mask), structural_nnz=int(mask.sum()))
        if method in ('fixed_C', 'em5_C'):
            if not np.array_equal(pattern.columns, geometric.columns):
                raise ValueError('Native/reference indexing differs')
            if method == 'fixed_C':
                weights, details = geometric.bilinear_weights.copy(), dict(solver='bilinear')
            else:
                weights, details = energy_weights(current, geometric, maxiter=5,
                                                  rtol=1e-8, fixed_iterations=True)
                native_mask = geometric.bilinear_weights != 0
                # Native EM uses its declared original support, never P != 0.
                details['native_support_matches_common'] = bool(np.array_equal(native_mask, mask))
                details['native_support_digest'] = array_digest(native_mask)
                if not details['native_support_matches_common']:
                    # The controlled, fully constrained EM candidate is separate.
                    mask = native_mask
                    common['support_digest'] = array_digest(mask)
                    common['structural_nnz'] = int(mask.sum())
            p = scipy_from_weights(pattern, weights, mask)
        else:
            if vectors is None:
                raise ValueError('Common BAMG vector bank unavailable; failure retained')
            objective = FixedObjective(current, pattern, vectors[index], tc)
            if method == 'bootstrap_C':
                p = original_ps[index].copy()
                values = objective.gather(p)
                details = dict(solver='original_bootstrap_selected_pass',
                    s_B=objective.scales.bamg, s_E=objective.scales.energy,
                    reference_policy='bilinear', reference_digest=objective.reference_digest,
                    normalizers_fixed=True, terms=objective.raw_terms(values),
                    vectors_digest=array_digest(objective.z), **objective.validate(values))
            else:
                p, details = minimize(objective, strength or 0.,
                    energy_only=method == 'em_constrained_C', **optimizer)
            details['bank_vectors_digest'] = array_digest(vectors[index])
        # Record the full numerical matrices' hashes, not only fine-grid data.
        common.update(details, P_digest=matrix_digest_rectangular(p))
        reports.append(common)
        ps.append(p)
        current = (p.T @ current @ p).tocsr()
        current.sum_duplicates(); current.sort_indices()
        complexity = (a.nnz + sum(r.get('coarse_nnz', 0) for r in reports[:-1]) + current.nnz) / a.nnz
        if complexity > tc.max_operator_complexity:
            raise MemoryError('Candidate exceeded operator-complexity budget')
        common['coarse_nnz'] = current.nnz
        common['coarse_A_digest'] = matrix_digest(current)
        sh, index = cs, index + 1
    if vectors is not None and index != len(vectors):
        raise ValueError('Reference/candidate depth mismatch')
    return ps, reports


def matrix_digest_rectangular(p):
    # matrix_digest hashes rectangular matrices too; no SPD validation is done.
    return matrix_digest(p)


def _measurement(bank, rhs, bench):
    started = perf_counter()
    try:
        result = bank.solve(rhs, timeout_seconds=bench['timeout_seconds'])
        success = bool(result['converged'])
        elapsed = float(result['warm_seconds'])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise FloatingPointError('Invalid measured solve time')
        return dict(success=success, warm_seconds=elapsed, reason=result['reason'],
                    cycles=result['cycles'], tolerance=result['tolerance'],
                    final_residual=(float(result['residuals'][-1])
                                    if np.isfinite(result['residuals'][-1]) else None),
                    fallback_used=False)
    except (ValueError, RuntimeError, FloatingPointError, MemoryError, np.linalg.LinAlgError) as exc:
        return dict(success=False, warm_seconds=perf_counter() - started, cycles=None,
                    reason=f'{type(exc).__name__}: {exc}', fallback_used=False)


def run_case(record, directory, settings, cfg, tc):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'results.json'
    methods = method_table(settings)
    bench = settings['benchmark']
    if path.exists():
        state = read_json(path)
        if state['specification'] != record or state['settings_digest'] != digest(settings):
            raise ValueError('Case identity changed')
        if state['completed']:
            return state
    else:
        state = dict(specification=record, settings_digest=digest(settings), completed=False,
                     setups={}, measurements={})
    example = _example(record)
    state.update(case_id=example.digest, family=record['family'], n=example.n,
                 normalized_operator_digest=example.group_digest)
    vectors, original_ps, bank_info = vector_bank(example, directory, tc, cfg)
    state['vector_bank'] = bank_info
    banks = {}
    for method, strength in methods.items():
        if len(state['measurements'].get(method, [])) == bench['repetitions'] * bench['rhs_count']:
            continue
        start = perf_counter()
        try:
            ps, details = candidate_hierarchy(example.a, (example.n, example.n), method,
                strength, vectors, original_ps, tc, cfg, settings['optimizer'])
            bank = runtime_from_prolongations(example.a, (example.n, example.n), ps, cfg)
            # Numeric factors are prepared before timing; no CNN or setup in solve.
            for _ in range(bench['warmup_cycles']):
                bank.cycle(np.zeros_like(example.b), example.b)
            banks[method] = bank
            setup = dict(status='ok', candidate_setup_seconds=perf_counter() - start,
                         common_bank_seconds=(0. if method in ('fixed_C', 'em5_C')
                                              else bank_info['generation_seconds']),
                         levels=details, hierarchy=bank.hierarchy_report)
        except (ValueError, RuntimeError, FloatingPointError, MemoryError, np.linalg.LinAlgError) as exc:
            banks[method] = None
            setup = dict(status='failed', candidate_setup_seconds=perf_counter() - start,
                         failure=f'{type(exc).__name__}: {exc}')
        if method in state['setups']:
            old = state['setups'][method]
            if old['status'] == 'ok' and setup['status'] == 'ok':
                if [v['P_digest'] for v in old['levels']] != [v['P_digest'] for v in setup['levels']]:
                    raise ValueError('Resumed candidate interpolation changed')
            elif old['status'] != setup['status']:
                raise ValueError('Resumed setup status changed; use a new experiment')
        else:
            state['setups'][method] = setup
        write_json(path, state)
    rng = np.random.default_rng(record['seed'] ^ 0x454D42414D47)
    rhs = [example.b]
    for _ in range(1, bench['rhs_count']):
        value = rng.standard_normal(example.a.shape[0])
        rhs.append(value / np.linalg.norm(value))
    state['benchmark_rhs_digests'] = [array_digest(b) for b in rhs]
    for repeat in range(bench['repetitions']):
        for rhs_index, b in enumerate(rhs):
            for method in rng.permutation(list(methods)):
                done = state['measurements'].setdefault(method, [])
                if any(v['repeat'] == repeat and v['rhs'] == rhs_index for v in done):
                    continue
                bank = banks[method]
                row = (_measurement(bank, b, bench) if bank is not None else
                       dict(success=False, warm_seconds=0., cycles=None,
                            reason='setup_failure', fallback_used=False))
                row.update(repeat=repeat, rhs=rhs_index)
                done.append(row)
                write_json(path, state)
                print(f"Teacher comparison {record['name']} {method} repeat={repeat} rhs={rhs_index} "
                      f"{'converged' if row['success'] else 'FAILED'}", flush=True)
    state['completed'] = True
    write_json(path, state)
    return state


def summarize(records, settings, expected_cases):
    bench = settings['benchmark']
    methods = method_table(settings)
    complete = len(records) == expected_cases and all(r.get('completed') for r in records)
    output = {}
    expected_per_case = bench['repetitions'] * bench['rhs_count']
    for method in methods:
        times, cycles, failures, setups = [], [], 0, []
        optimizer_unconverged = 0
        for case in records:
            observations = case.get('measurements', {}).get(method, [])
            indexed = {(v['repeat'], v['rhs']): v for v in observations}
            if len(indexed) != len(observations) or len(indexed) > expected_per_case:
                raise ValueError('Duplicate/excess benchmark measurements')
            repeat_times, repeat_cycles = [], []
            for repeat in range(bench['repetitions']):
                rt, rc = [], []
                for rhs in range(bench['rhs_count']):
                    row = indexed.get((repeat, rhs), {})
                    success = row.get('success', False)
                    failures += int(not success)
                    seconds = row.get('warm_seconds', 0.)
                    rt.append(seconds if success else max(seconds, bench['failure_penalty_seconds']))
                    rc.append(row.get('cycles') if success else None)
                repeat_times.append(float(np.mean(rt)))
                repeat_cycles.append(float(np.mean(rc)) if all(v is not None for v in rc) else None)
            times.append(float(np.median(repeat_times)))
            if all(v is not None for v in repeat_cycles):
                cycles.append(float(np.median(repeat_cycles)))
            setup = case.get('setups', {}).get(method, {})
            if setup.get('status') == 'ok':
                setups.append(setup['candidate_setup_seconds'] + setup.get('common_bank_seconds', 0.))
                optimizer_unconverged += sum(v.get('optimizer_converged') is False
                                             for v in setup.get('levels', []))
        missing = expected_cases - len(records)
        failures += missing * expected_per_case
        times += [bench['failure_penalty_seconds']] * missing
        attempts = expected_cases * expected_per_case
        rate = failures / attempts
        output[method] = dict(cases=expected_cases, attempts=attempts, failures=failures,
            failure_rate=rate, eligible=bool(complete and rate <= bench['allowed_failure_rate']),
            mean_case_median_warm_seconds=float(np.mean(times)),
            mean_case_median_cycles=float(np.mean(cycles)) if failures == 0 else None,
            mean_setup_seconds=float(np.mean(setups)) if len(setups) == expected_cases else None,
            optimizer_unconverged_levels=optimizer_unconverged)
    for value in output.values():
        value['speedup_vs_fixed'] = (output['fixed_C']['mean_case_median_warm_seconds'] /
            value['mean_case_median_warm_seconds'] if complete and value['failures'] == 0
            and output['fixed_C']['failures'] == 0 else None)
    def best(names):
        eligible = [n for n in names if output[n]['eligible']]
        return min(eligible, key=lambda n: (output[n]['failure_rate'],
            output[n]['mean_case_median_warm_seconds'], n)) if eligible else None
    selected = best([n for n in methods if n.startswith('hybrid_lambda_')])
    return dict(completed=complete, cases_completed=sum(r.get('completed', False) for r in records),
        cases_expected=expected_cases, methods=output, fastest_eligible_method=best(methods),
        selected_lambda=methods[selected] if selected else None, selected_method=selected,
        selection_scope='development validation; no final/OOD claim',
        selection_rule='failure rate first, then penalized warm time; eligible threshold enforced',
        aggregation='mean RHS time per repeat; median repeats per case; arithmetic mean cases',
        common_conditions='controlled methods share frozen vectors, geometric C/support, line_alt 2+2, RHS and tolerance',
        native_em5_support_matches_common=all(v.get('native_support_matches_common', False)
            for r in records for v in r.get('setups', {}).get('em5_C', {}).get('levels', []))
            if complete and all(r.get('setups', {}).get('em5_C', {}).get('status') == 'ok' for r in records)
            else None,
        coarse_operators='each candidate rebuilds actual P.T A P; deeper A_l may differ',
        setup_accounting='includes shared-bank generation for bank-dependent methods; excludes CNN training')


def print_summary(summary):
    print(f"\nTeacher-only development validation: {summary['cases_completed']}/{summary['cases_expected']} cases")
    print(f"{'Method':28s} {'Warm ms*':>12s} {'Cycles':>9s} {'Fail/total':>12s} {'Speedup':>9s}")
    for name, v in summary['methods'].items():
        cycles = '--' if v['mean_case_median_cycles'] is None else f"{v['mean_case_median_cycles']:.2f}"
        speed = '--' if v['speedup_vs_fixed'] is None else f"{v['speedup_vs_fixed']:.3f}"
        print(f"{name:28s} {1000*v['mean_case_median_warm_seconds']:12.3f} {cycles:>9s} "
              f"{v['failures']:>5d}/{v['attempts']:<6d} {speed:>9s}")
    print('Selected lambda:', summary['selected_lambda'], '| Best eligible:', summary['fastest_eligible_method'])
    unconverged = {name: v['optimizer_unconverged_levels'] for name, v in summary['methods'].items()
                   if v['optimizer_unconverged_levels']}
    if unconverged:
        print('Optimizer budget reached (feasible levels retained):', unconverged)
    print('* Failed/missing solves retain the declared time penalty. No fallback. Setup excluded.')


def run(config, output, *, resume=False, max_cases=None):
    settings = read_json(config)
    cfg, tc = validate(settings)
    method_table(settings)
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError('max_cases must be positive')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.comparison.lock').open('a') as lock, threadpool_limits(limits=settings['threads']):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(settings['threads'])
        identity = manifest(settings)
        path = out / 'comparison_manifest.json'
        if path.exists():
            if not resume:
                raise FileExistsError('Comparison exists; use --resume')
            if read_json(path) != identity or read_json(out / 'configuration.json') != settings:
                raise ValueError('Comparison source/configuration/environment changed; use a new directory')
        else:
            if resume or any(p.name != '.comparison.lock' for p in out.iterdir()):
                raise FileExistsError('New comparison requires an empty directory')
            write_json(out / 'configuration.json', settings)
            write_json(path, identity)
        specs = specifications(settings)
        write_json(out / 'development_cases.json', dict(split='validation', cases=specs))
        processed = 0
        for spec in specs:
            directory = out / 'cases' / spec['name']
            result_path = directory / 'results.json'
            if result_path.exists() and read_json(result_path).get('completed'):
                continue
            run_case(spec, directory, settings, cfg, tc)
            processed += 1
            if max_cases is not None and processed >= max_cases:
                break
        records = [read_json(out / 'cases' / s['name'] / 'results.json') for s in specs
                   if (out / 'cases' / s['name'] / 'results.json').exists()]
        if len({r['normalized_operator_digest'] for r in records}) != len(records):
            raise ValueError('Duplicate development operators; use a new seed/directory')
        result = summarize(records, settings, len(specs))
        result['settings_digest'] = digest(settings)
        result['groups'] = {}
        for label, predicate in [(f'n{n}', lambda r, n=n: r['n'] == n) for n in settings['data']['sizes']] + [
                (f, lambda r, f=f: r['family'] == f) for f in settings['data']['families']]:
            group = [r for r in records if predicate(r)]
            count = sum(predicate(dict(n=s['case']['n'], family=s['family'])) for s in specs)
            result['groups'][label] = summarize(group, settings, count)
            # Group tables diagnose heterogeneity, never select different lambdas.
            result['groups'][label]['selected_lambda'] = None
            result['groups'][label]['selected_method'] = None
        write_json(out / 'teacher_comparison.json', result)
        print_summary(result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'report'))
    parser.add_argument('--config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--group', help='Report a grid (n31/n63) or PDE family; does not select a group-specific lambda')
    parser.add_argument('--max-cases', type=int, help='Bound NEW completed cases; unfinished selection stays disabled')
    args = parser.parse_args(argv)
    if args.command == 'report':
        summary = read_json(Path(args.output) / 'teacher_comparison.json')
        if args.group:
            if args.group not in summary['groups']:
                parser.error('Unknown group; choose from ' + ', '.join(summary['groups']))
            summary = summary['groups'][args.group]
        print_summary(summary)
        return
    if args.group:
        parser.error('--group is only available for report')
    if args.config is None:
        parser.error('run requires --config')
    run(args.config, args.output, resume=args.resume, max_cases=args.max_cases)


if __name__ == '__main__':
    main()
