#!/usr/bin/env python3
"""Read-only validation reports and checkpoint replay for the light study.

report: no solves, training or checkpoint loading; uses a completed benchmark.
replay: fresh diagnostic solves, actual wall checkpoints and solution fields.
Neither command changes a study, chooses a model, or opens final/OOD data.
Kept outside src so existing study source/freeze fingerprints remain valid.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np

from adaptive_mg.provenance import hardware_environment, json_safe, stable_norm
from adaptive_mg.v67.research_data import FAMILIES, _hash, _write_json

VERSION = 'light-smoother-figures-v1'
ARMS = ('fixed_C', 'em5_C', 'learned_P_line', 'learned_P_MLP',
        'learned_P_regression', 'learned_P_MLP_TTT',
        'learned_P_regression_online', 'frozen_P_line')
LABELS = dict(zip(ARMS, ('Classical bilinear', 'Classical EM5', 'Learned P + line',
    'Learned P + MLP', 'Learned P + ridge', 'MLP + TTT', 'Ridge + online',
    'Learned P + CNN line')))
FAMILY_LABELS = ('Near isotropic', 'Aligned anisotropy', 'Rotated anisotropy',
                 'Local patch', 'Checkerboard', 'Channel', 'Variable coefficient')
COLORS = dict(zip(ARMS, ('#30343b', '#bd7b0d', '#137d85', '#2564cf', '#c14c81',
                        '#5e4ba3', '#188347', '#c84830')))


def read(path):
    return json.loads(Path(path).read_text())


def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _file_digest(stream)


def _file_digest(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def output_directory(path, source, *, resume=False):
    target, source = Path(path).resolve(), Path(source).resolve()
    if (target == source or target.is_relative_to(source) or source.is_relative_to(target)
            or any(target == ROOT / name or target.is_relative_to(ROOT / name)
                   for name in ('src', 'scripts', 'configs', 'tests', 'docs'))):
        raise ValueError('Use a separate output directory outside the source run and code directories')
    if target.exists() and any(target.iterdir()) and not resume:
        raise FileExistsError('Output exists; use a new output or identical --resume')
    target.mkdir(parents=True, exist_ok=True)
    return target


def signed_record(path, inputs, data, arrays=None):
    entry = dict(inputs=inputs, data=json_safe(data), data_digest=_hash(json_safe(data)))
    if arrays is not None:
        array_path = path.with_suffix('.npz')
        temp = array_path.with_suffix('.npz.tmp')
        path.parent.mkdir(parents=True, exist_ok=True)
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(array_path)
        entry['arrays_sha256'] = digest_file(array_path)
    _write_json(path, entry)
    return entry


def checked_record(path, inputs):
    entry = read(path)
    if entry['inputs'] != inputs or entry['data_digest'] != _hash(entry['data']):
        raise ValueError(f'Diagnostic record changed: {path}')
    if entry.get('arrays_sha256') != digest_file(path.with_suffix('.npz')):
        raise ValueError(f'Solution arrays changed: {path}')
    return entry


def commit_manifest(output, manifest, resume):
    path = output / 'manifest.json'
    if path.exists():
        if not resume or read(path) != json_safe(manifest):
            raise ValueError('Resume source/model/environment/protocol mismatch')
    else:
        if resume and any(output.iterdir()):
            raise ValueError('Cannot resume an unmanifested directory')
        _write_json(path, manifest, exclusive=True)


def load_benchmark(folder):
    """Require the full declared validation cohort and verify every saved repeat."""
    folder = Path(folder).resolve()
    manifest = read(folder / 'run_manifest.json')
    if manifest.get('version') != 'light-smoother-measured-v2':
        raise ValueError('Expected a light-smoother benchmark, not another experiment')
    progress = read(folder / 'progress.json')
    if progress.get('status') != 'complete':
        raise ValueError('Use a completed benchmark; interrupted subsets are not a full-cohort report')
    rows = read(folder / 'raw_results.json')
    inputs = manifest['case_inputs']
    if (len(rows) != len(inputs) or not rows
            or len({r['example']['name'] for r in rows}) != len(rows)):
        raise ValueError('Incomplete or duplicate validation cohort')
    signatures = {v['p_signature'] for v in manifest['arms'].values() if v.get('p_signature')}
    if len(signatures) > 1:
        raise ValueError('Learned arms have different P weights')
    if set(manifest['arms']) - set(ARMS) or 'fixed_C' not in manifest['arms']:
        raise ValueError('Unsupported arms or missing classical reference')
    manifest_digest = _hash(manifest)
    total = len(rows)*len(manifest['arms'])*manifest['repeats']*sum(
        len(manifest['rhs_counts']) if r in ('multiple', 'warm_multiple') else 1 for r in manifest['regimes'])
    if progress.get('completed_measurements') != total or progress.get('total_measurements') != total:
        raise ValueError('Completion counts differ from the declared cohort')
    for index, (row, expected) in enumerate(zip(rows, inputs)):
        if (row.get('research_split') != 'validation' or expected.get('research_split') != 'validation'
                or row['family'] not in FAMILIES
                or any(row['example'].get(k) != expected.get(k) for k in row['example'])
                or set(row['runs']) != set(manifest['arms'])):
            raise ValueError('Validation case/arm identity mismatch; final/OOD are not accepted')
        for arm in manifest['arms']:
            if set(row['runs'][arm]) != set(manifest['regimes']):
                raise ValueError('Incomplete regimes')
            for regime in manifest['regimes']:
                counts = manifest['rhs_counts'] if regime in ('multiple', 'warm_multiple') else [1]
                if set(row['runs'][arm][regime]) != {str(k) for k in counts}:
                    raise ValueError('Incomplete RHS counts')
                for count in counts:
                    runs = row['runs'][arm][regime][str(count)]
                    if len(runs) != manifest['repeats']:
                        raise ValueError('Incomplete repeated measurements')
                    for repeat, run in enumerate(runs):
                        key = dict(manifest_digest=manifest_digest, case_index=index,
                            arm=arm, regime=regime, rhs_count=count, repeat=repeat)
                        saved = read(folder / 'repeat_records' / (_hash(key) + '.json'))
                        if (saved.get('inputs') != key or saved.get('run_digest') != _hash(run)
                                or saved.get('run') != run):
                            raise ValueError('Raw result / measurement ledger mismatch')
    return manifest, rows


def median(values):
    values = [float(x) for x in values if isinstance(x, (int, float)) and np.isfinite(x)]
    return float(np.median(values)) if values else None


def geometric(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x) and x > 0]
    return float(np.exp(np.mean(np.log(values)))) if values else None


def percentile(values, q):
    return float(np.percentile(list(values), q)) if len(values) else None


def case_statistics(rows, manifest, regimes, rhs_counts):
    output = []
    for row in rows:
        for regime in regimes:
            for count in rhs_counts:
                if regime in ('cold', 'warm') and count != 1:
                    continue
                entries = []
                for arm in manifest['arms']:
                    runs = row['runs'][arm][regime][str(count)]
                    rhs = [r for run in runs for r in run['rhs_results']]
                    expected = count * manifest['repeats']
                    success = (len(rhs) == expected and all(r.get('verified_success') is True for r in rhs)
                               and all(r.get('success') is True for r in runs))
                    entries.append(dict(case=row['example']['name'], family=row['family'],
                        n=row['example']['case']['n'], arm=arm, regime=regime, rhs_count=count,
                        success=success, repeats=len(runs), expected_rhs_solves=expected,
                        successful_rhs_solves=sum(r.get('verified_success') is True for r in rhs),
                        missing_rhs_solves=expected-len(rhs),
                        wall_seconds=median(r['wall_seconds'] for r in runs),
                        seconds_per_rhs=median(r['seconds_per_rhs'] for r in runs),
                        accepted_cycles_per_rhs=median(sum(v['cycles'] for v in r['rhs_results']) / count for r in runs),
                        executed_cycles_per_rhs=median(r['executed_cycles'] / count for r in runs),
                        all_executed_cycles_per_rhs=median(r['all_executed_cycles'] / count for r in runs),
                        convergence_factor=median(v.get('convergence_factor') for v in rhs),
                        relative_solution_error=median(v.get('relative_solution_error') for v in rhs),
                        relative_energy_error=median(v.get('relative_energy_error') for v in rhs),
                        max_final_relative_residual=max((v['relative_residual'] for v in rhs
                            if isinstance(v.get('relative_residual'), (int, float))), default=None),
                        fallback_any=any(r['fallback_count'] > 0 or r.get('setup_fallback') for r in runs),
                        rollback_any=any(r['rollback_count'] > 0 for r in runs),
                        neural_used_any=any(r.get('actual_neural_used') for r in runs),
                        fallback_cycles_per_rhs=median(r['fallback_count'] / count for r in runs),
                        online_calls=median(r['online_adaptation_calls'] for r in runs),
                        online_accepted=median(r['online_adaptation_accepted'] for r in runs),
                        online_seconds=median(r['online_adaptation_seconds'] for r in runs),
                        speedup_vs_fixed=None))
                fixed = next(e for e in entries if e['arm'] == 'fixed_C')
                for entry in entries:
                    if entry['success'] and fixed['success']:
                        entry['speedup_vs_fixed'] = fixed['wall_seconds'] / entry['wall_seconds']
                    output.append(entry)
    return output


def family_statistics(cases):
    """Failures always counted; absolute spread uses one common successful cohort."""
    output, robustness = [], []
    groups = sorted({(r['regime'], r['rhs_count'], r['n']) for r in cases})
    for regime, count, n in groups:
        group = [r for r in cases if (r['regime'], r['rhs_count'], r['n']) == (regime, count, n)]
        arms = [a for a in ARMS if any(r['arm'] == a for r in group)]
        common = {case for case in {r['case'] for r in group}
                  if sum(r['case'] == case and r['success'] for r in group) == len(arms)}
        families = []
        for arm in arms:
            arm_rows = []
            for family in FAMILIES:
                rows = [r for r in group if r['arm'] == arm and r['family'] == family]
                if not rows:
                    continue
                selected = [r for r in rows if r['case'] in common]
                ratios = [r['speedup_vs_fixed'] for r in rows if r['speedup_vs_fixed'] is not None]
                entry = dict(regime=regime, rhs_count=count, n=n, family=family, arm=arm,
                    total=len(rows), successes=sum(r['success'] for r in rows),
                    failures=sum(not r['success'] for r in rows),
                    success_rate=sum(r['success'] for r in rows)/len(rows),
                    fallback_cases=sum(r['fallback_any'] for r in rows),
                    fallback_rate=sum(r['fallback_any'] for r in rows)/len(rows),
                    neural_used_cases=sum(r['neural_used_any'] for r in rows),
                    paired_success_count=len(ratios), paired_geomean_speedup=geometric(ratios),
                    common_success_count=len(selected),
                    common_median_seconds_per_rhs=median(r['seconds_per_rhs'] for r in selected),
                    common_median_cycles_per_rhs=median(r['accepted_cycles_per_rhs'] for r in selected),
                    common_p10_seconds_per_rhs=percentile([r['seconds_per_rhs'] for r in selected], 10),
                    common_p90_seconds_per_rhs=percentile([r['seconds_per_rhs'] for r in selected], 90),
                    common_p10_cycles_per_rhs=percentile([r['accepted_cycles_per_rhs'] for r in selected], 10),
                    common_p90_cycles_per_rhs=percentile([r['accepted_cycles_per_rhs'] for r in selected], 90),
                    observed_all_attempts_seconds=sum(r['wall_seconds'] for r in rows),
                    scope='absolute spread: all-arm common success cohort; speedup: pairwise successes; failures retained')
                output.append(entry)
                arm_rows.append(entry)
            families = [r['family'] for r in arm_rows]
            entry = dict(regime=regime, rhs_count=count, n=n, arm=arm,
                total_cases=sum(r['total'] for r in arm_rows),
                failures=sum(r['failures'] for r in arm_rows), common_success_cases=len(common),
                families_with_common_success=sum(r['common_success_count'] > 0 for r in arm_rows),
                family_time_cv=None, family_cycle_cv=None, worst_family_median_seconds_per_rhs=None,
                family_time_max_over_min=None,
                interpretation='CV of seven family medians on the identical all-arm successful cohort; not a proof of robustness')
            if set(families) == set(FAMILIES) and all(r['common_success_count'] for r in arm_rows):
                for field, key in [('common_median_seconds_per_rhs', 'family_time_cv'),
                                   ('common_median_cycles_per_rhs', 'family_cycle_cv')]:
                    values = np.array([r[field] for r in arm_rows])
                    entry[key] = float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else None
                values = [r['common_median_seconds_per_rhs'] for r in arm_rows]
                entry['worst_family_median_seconds_per_rhs'] = max(values)
                entry['family_time_max_over_min'] = max(values)/min(values)
            robustness.append(entry)
    return output, robustness


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    return plt


def save_figure(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def family_plots(cases, families, output, dpi):
    plt = plotting()
    from matplotlib.colors import LogNorm
    for regime, count, n in sorted({(r['regime'], r['rhs_count'], r['n']) for r in families}):
        rows = [r for r in families if (r['regime'], r['rhs_count'], r['n']) == (regime, count, n)]
        arms = [a for a in ARMS if any(r['arm'] == a for r in rows)]
        lookup = {(r['arm'], r['family']): r for r in rows}
        specs = [('success_rate', 'Success rate (all declared operators)', 'viridis', False),
                 ('fallback_rate', 'Classical safety-use rate (includes near-tolerance tail)', 'magma_r', False),
                 ('paired_geomean_speedup', 'Speedup vs fixed_C; >1 is faster (paired successes)', 'RdYlGn', True),
                 ('common_median_cycles_per_rhs', 'Accepted cycles / RHS (same successful cohort for all arms)', 'viridis_r', True),
                 ('common_median_seconds_per_rhs', 'Seconds / RHS (same successful cohort for all arms)', 'viridis_r', True)]
        stem = f'{regime}_rhs{count}_n{n}'
        print(f'Plotting PDE-family statistics: {stem}', flush=True)
        for metric, title, cmap, log in specs:
            values = np.full((len(arms), len(FAMILIES)), np.nan)
            for i, arm in enumerate(arms):
                for j, family in enumerate(FAMILIES):
                    value = lookup.get((arm, family), {}).get(metric)
                    if value is not None:
                        values[i, j] = value
            fig, ax = plt.subplots(figsize=(12.8, 5.5), layout='constrained')
            positive = values[np.isfinite(values) & (values > 0)]
            norm = LogNorm(max(min(positive), 1e-12), max(max(positive), min(positive)*1.001)) if log and len(positive) else None
            art = ax.imshow(values, aspect='auto', cmap=cmap, norm=norm,
                            vmin=None if norm else 0, vmax=None if norm else 1)
            ax.set_xticks(range(len(FAMILIES)), FAMILY_LABELS, rotation=18, ha='right')
            ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms])
            for i, arm in enumerate(arms):
                for j, family in enumerate(FAMILIES):
                    row = lookup.get((arm, family))
                    value = values[i, j]
                    if row is None or not np.isfinite(value):
                        text = 'N/A'
                    elif metric == 'success_rate':
                        text = f'{row["successes"]}/{row["total"]}'
                    elif metric == 'fallback_rate':
                        text = f'{row["fallback_cases"]}/{row["total"]}'
                    else:
                        cohort = row['paired_success_count'] if metric == 'paired_geomean_speedup' else row['common_success_count']
                        text = f'{value:.3g}\n(n={cohort})'
                    ax.text(j, i, text, ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=1.7))
            ax.set_title(f'{title}\n{regime}, RHS batch={count}, interior grid={n} x {n}')
            fig.colorbar(art, ax=ax, shrink=.8)
            save_figure(fig, output/'families'/f'{stem}_{metric}', dpi)
        # Failure-aware baseline-relative completion curves: denominator is ALL A.
        group = [r for r in cases if (r['regime'], r['rhs_count'], r['n']) == (regime, count, n)]
        fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
        for arm in arms:
            selected = [r for r in group if r['arm'] == arm]
            ratios = np.sort([1/r['speedup_vs_fixed'] for r in selected if r['speedup_vs_fixed'] is not None])
            xx = np.r_[0., ratios, max(4., float(ratios[-1]) if len(ratios) else 4.)]
            yy = np.r_[0., np.arange(1, len(ratios)+1)/len(selected), len(ratios)/len(selected)]
            ax.step(xx, yy, where='post', color=COLORS[arm], label=f'{LABELS[arm]} ({len(ratios)}/{len(selected)} paired successes)')
        ax.axvline(1, color='grey', ls='--', lw=1)
        ax.set(xlabel='Time / fixed_C time (smaller is faster)', ylabel='Fraction of ALL operators',
               ylim=(0, 1.02), title=f'Baseline-relative completion profile | {stem}\nFailure of either method never counts as a completed pair')
        ax.grid(alpha=.2); ax.legend(fontsize=7, loc='lower right')
        save_figure(fig, output/'families'/f'{stem}_completion_profile', dpi)
        for arm in arms:
            if arm in ('fixed_C', 'em5_C'):
                continue
            fig, axs = plt.subplots(2, 1, figsize=(11.2, 8.5), layout='constrained')
            for ax, metric, ylabel in zip(axs, ('seconds', 'cycles'),
                    ('Seconds / RHS', 'Accepted cycles / RHS')):
                for name in dict.fromkeys(('fixed_C', 'em5_C', arm)):
                    if name not in arms:
                        continue
                    selected = [lookup.get((name, family), {}) for family in FAMILIES]
                    y = np.array([r.get(f'common_median_{metric}_per_rhs') or np.nan for r in selected])
                    lo = np.array([r.get(f'common_p10_{metric}_per_rhs') or np.nan for r in selected])
                    hi = np.array([r.get(f'common_p90_{metric}_per_rhs') or np.nan for r in selected])
                    ax.plot(range(7), y, marker='o', color=COLORS[name], label=LABELS[name])
                    ax.fill_between(range(7), lo, hi, color=COLORS[name], alpha=.13)
                family_ticks = [f'{label}\n(n={lookup.get(("fixed_C", family), {}).get("common_success_count", 0)})'
                                for family, label in zip(FAMILIES, FAMILY_LABELS)]
                ax.set_xticks(range(7), family_ticks, rotation=16, ha='right')
                ax.set_ylabel(ylabel); ax.set_yscale('log'); ax.grid(alpha=.2); ax.legend(fontsize=8)
            fig.suptitle(f'{LABELS[arm]} across PDE families | {stem}\n'
                         'Same all-arm successful cohort: median and operator P10-P90 (not confidence intervals)\n'
                         'Read with the full-cohort success/fallback heatmaps; no failures were deleted from the report', fontsize=11)
            save_figure(fig, output/'consistency'/f'{stem}_{arm}', dpi)


def curve_figure(items, arm, *, axis, title, path, dpi):
    """Seven fixed, outcome-independent examples; accepted states only."""
    plt = plotting()
    fig, axs = plt.subplots(2, 4, figsize=(14.6, 7.8), layout='constrained')
    handles = {}
    for ax, family, label in zip(axs.ravel(), FAMILIES, FAMILY_LABELS):
        records = items.get(family, {})
        for name in dict.fromkeys(('fixed_C', 'em5_C', arm)):
            record = records.get(name)
            if not record or not record.get('residual_history'):
                if name == arm:
                    ax.text(.03, .05, 'Missing result / failure; see CSV', transform=ax.transAxes, color='darkred', fontsize=8)
                continue
            residual = np.array([np.nan if x is None else x for x in record['residual_history']], float)
            norm = max(record['residual_reference_norm'], 1e-300)
            yy = np.maximum(residual / norm, 1e-18)
            xx = np.arange(len(yy)) if axis == 'cycles' else np.asarray(record['residual_wall_seconds'])
            line, = ax.semilogy(xx, yy, color=COLORS[name], lw=1.6, label=LABELS[name])
            handles[name] = line
            if name == arm and name not in ('fixed_C', 'em5_C'):
                safety = [i+1 for i, p in enumerate(record['cycle_path']) if p != 'hybrid']
                if safety:
                    ax.scatter(xx[safety], yy[safety], marker='s', s=15, facecolors='none', edgecolors=COLORS[name])
            if not record.get('verified_success'):
                finite = np.flatnonzero(np.isfinite(yy))
                if len(finite):
                    ax.scatter(xx[finite[-1]], yy[finite[-1]], color=COLORS[name], marker='x', s=45)
            if axis == 'wall':
                ax.plot([xx[-1], record['completion_wall_seconds']], [yy[-1], yy[-1]], color=COLORS[name], ls=':', lw=1)
                ax.scatter(record['completion_wall_seconds'], yy[-1], color=COLORS[name], s=15)
            if name == 'fixed_C':
                ax.axhline(record['stopping_threshold']/norm, color='grey', ls='--', lw=.9)
        ax.set_title(label); ax.grid(alpha=.2)
        ax.set_xlabel('Accepted solution updates' if axis == 'cycles' else 'Measured wall seconds')
        ax.set_ylabel(r'$\|b-Au_k\|_2/\|b\|_2$')
    axs.ravel()[-1].axis('off')
    axs.ravel()[-1].legend(list(handles.values()), [LABELS[a] for a in handles], loc='upper left', fontsize=9)
    axs.ravel()[-1].text(0, .49, 'Dashed grey: stopping tolerance\nSquare: classical safety step\nCross: failed RHS\n\nAccepted updates exclude rejected\ntrials and adaptation probe cycles.\nTheir costs remain in measured time.\n\nOne predeclared case per family;\nnot a full-cohort performance claim.', va='top', fontsize=9, linespacing=1.5)
    fig.suptitle(title, fontsize=13)
    save_figure(fig, path, dpi)


def matches_index(name, index):
    return re.search(rf'_{index}_r\d+$', name) is not None


def report(args):
    folder = Path(args.benchmark_dir).resolve()
    manifest, rows = load_benchmark(folder)
    regimes = args.regimes or manifest['regimes']
    counts = args.rhs_counts or manifest['rhs_counts']
    if set(regimes)-set(manifest['regimes']) or set(counts)-set(manifest['rhs_counts']):
        raise ValueError('Requested regime/RHS count was not measured')
    if args.rhs_index >= min(counts) or len(counts) != len(set(counts)):
        raise ValueError('Use distinct counts and --rhs-index smaller than the smallest selected batch')
    source = folder.parent.parent if (folder.parent.parent/'light_manifest.json').is_file() else folder
    output = output_directory(args.output, source, resume=args.resume)
    provenance = dict(version=VERSION, mode='existing_validation_report',
        runner_sha256=digest_file(__file__), source_benchmark=str(folder),
        inputs={name: digest_file(folder/name) for name in ('run_manifest.json', 'raw_results.json', 'progress.json')},
        regimes=regimes, rhs_counts=counts, sample_index=args.sample_index,
        curve_rhs_index=args.rhs_index, repeat_for_curves=0, dpi=args.dpi,
        validation_count=len(rows), no_new_solves=True, selection=False,
        note='No absolute per-cycle timestamps or final solution vectors exist in the legacy benchmark. No wall curve or field error is fabricated.')
    commit_manifest(output, provenance, args.resume)
    cases = case_statistics(rows, manifest, regimes, counts)
    families, robustness = family_statistics(cases)
    write_csv(output/'case_metrics.csv', cases)
    write_csv(output/'family_metrics.csv', families)
    write_csv(output/'robustness_summary.csv', robustness)
    family_plots(cases, families, output, args.dpi)
    # One RHS workload (the smallest requested count); all counts remain in statistics.
    count = min(counts)
    for n in sorted({r['example']['case']['n'] for r in rows}):
        selected = [r for r in rows if r['example']['case']['n'] == n and matches_index(r['example']['name'], args.sample_index)]
        for regime in regimes:
            if regime in ('cold', 'warm') and count != 1:
                continue
            items = {r['family']: {arm: (runs[regime][str(count)][0]['rhs_results'][args.rhs_index]
                     if len(runs[regime][str(count)][0]['rhs_results']) > args.rhs_index else {})
                     for arm, runs in r['runs'].items()} for r in selected}
            for arm in manifest['arms']:
                if arm in ('fixed_C', 'em5_C'):
                    continue
                curve_figure(items, arm, axis='cycles', dpi=args.dpi,
                    title=f'{LABELS[arm]} | existing validation, {regime}, n={n}, batch={count}, RHS index={args.rhs_index}, sample={args.sample_index}, repeat=0',
                    path=output/'cycles'/f'{regime}_n{n}_rhs{count}_{arm}')
    _write_json(output/'progress.json', dict(status='complete', cases=len(rows)))
    (output/'README_KR.md').write_text(
        '# 기존 validation 기록 분석\n\n재학습·추가 solve 없이 완전한 validation 기록과 repeat ledger를 확인했습니다.\n'
        'cycles/: 승인된 해 갱신 횟수–상대 잔차. 대표 사례는 이름의 사전 지정 sample index로 선택합니다.\n'
        'families/: 격자 크기·RHS 수·시간 조건별 성공률, fallback 비율, paired speedup, cycle/시간, completion profile.\n'
        'case_metrics.csv: 실패 포함 모든 A. family_metrics.csv: family 집계. robustness_summary.csv: family 중앙값의 CV.\n\n'
        'CV=7개 family 중앙값의 모집단 표준편차/평균. 모든 비교군이 성공한 동일 A 집합만 절대 시간/cycle 산포에 사용합니다. '
        '이 조건부 산포와 전체 성공률·fallback 비율을 함께 해석해야 합니다. 작은 CV만으로 강건성을 입증하지 않습니다. '
        '반복 RHS와 측정 반복을 독립적인 PDE 표본으로 세지 않습니다.\n\n'
        'speedup은 pairwise 성공 사례만의 기하평균이며 표본 수를 표시합니다. '
        'completion profile은 전체 A를 분모로 하며 fixed_C 또는 비교군이 실패한 쌍은 완주로 세지 않습니다. '
        '실패까지 소비한 시간은 기록하되 성공한 time-to-solution으로 해석하지 않습니다.\n\n'
        '기존 cycle_seconds의 합은 함수 외부 비용과 일부 overhead를 포함하지 않아 정확한 wall-time 축이 아닙니다. '
        '따라서 wall-time 곡선과 공간 오차는 replay로 생성합니다. Warm 결과는 setup을 제외한 비교입니다.\n', encoding='utf-8')
    print(f'Report complete: {output}; {len(rows)} validation A, no new solves.', flush=True)
    return cases


class WallObserver(AbstractContextManager):
    """Observe existing timing calls without changing numerical solver sources.

    At the existing end-of-update clock call, the production function has
    already appended its accepted residual. Record the absolute timestamp,
    including all work since the external start (adaptation/rejections too).
    The only extra work is diagnostic observation, included in measured time.
    Module clocks are restored even on an exception. Single-threaded runner.
    """
    def __init__(self, start):
        from adaptive_mg.v67 import light_smoother_runtime, solver
        self.modules = (light_smoother_runtime, solver)
        self.codes = {light_smoother_runtime.PreparedLightMG._solve.__code__,
                      solver.PreparedAdaptiveMG._solve.__code__,
                      solver.PreparedAdaptiveMG._classical_tail.__code__}
        self.start, self.prepared, self.events = start, None, {}
        self.observer_seconds = 0.

    def clock(self):
        now = perf_counter()
        frame = sys._getframe(1)
        if frame.f_code in self.codes:
            local = frame.f_locals
            if local.get('self') is self.prepared and local.get('history'):
                history = local['history']
                index = len(history)-1
                if index not in self.events:
                    self.events[index] = (now-self.start, float(history[-1]))
        self.observer_seconds += perf_counter()-now
        return now

    def __enter__(self):
        self.original = [m.perf_counter for m in self.modules]
        for module in self.modules:
            module.perf_counter = self.clock
        return self

    def __exit__(self, *args):
        for module, clock in zip(self.modules, self.original):
            module.perf_counter = clock

    def timestamps(self, result):
        if set(self.events) != set(range(len(result.residual_history))):
            raise RuntimeError('Wall observer missed an accepted residual; refusing fabricated timestamps')
        observed = [self.events[i][1] for i in range(len(self.events))]
        np.testing.assert_allclose(observed, result.residual_history, rtol=0, atol=0, equal_nan=True)
        # Initial x=0 is known at t=0. All preparation/initial checks are then
        # included before the first accepted solution update.
        return [0.] + [self.events[i][0] for i in range(1, len(self.events))]


def replay_measure(example, arm, cfg, baseline, rhs, exact, prime_rhs, regime):
    from adaptive_mg.v67.light_smoother_evaluation import _construct_light
    from adaptive_mg.v67.research_evaluation import _construct, _result_record
    light = bool(arm.get('light'))
    prepared, prime_success, prime_error = None, True, None
    prime_seconds, constructor_seconds = 0., 0.
    def construct():
        return _construct_light(example, arm, cfg) if light else _construct(example, arm, cfg, baseline, 1)
    if regime == 'warm':
        start = perf_counter()
        try:
            prepared = construct()
            if hasattr(prepared, 'prepare_warm'):
                prepared.prepare_warm()
            enabled = prepared.online_enabled if light else False
            if light:
                prepared.online_enabled = False
            prime = prepared.solve(prime_rhs, np.zeros_like(prime_rhs))
            prime_success = bool(prime.converged and stable_norm(prime_rhs-example.a@prime.x) <= prime.stopping_threshold)
            if light:
                prepared.reset_online()
                prepared.online_enabled = enabled
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            prime_error, prime_success = f'{type(exc).__name__}: {exc}', False
        prime_seconds = perf_counter()-start
    started = perf_counter()
    result, failure = None, None
    with WallObserver(started) as observer:
        try:
            if prime_error:
                raise RuntimeError('Warm preparation failed: '+prime_error)
            if prepared is None:
                try:
                    prepared = construct()
                finally:
                    constructor_seconds = perf_counter()-started
            observer.prepared = prepared
            result = prepared.solve(rhs, np.zeros_like(rhs))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failure = f'{type(exc).__name__}: {exc}'
    elapsed = perf_counter()-started
    record = _result_record(result, example, rhs, exact, prepared.config) if result is not None else {}
    if result is not None:
        record['residual_wall_seconds'] = observer.timestamps(result)
        if any(t > elapsed for t in record['residual_wall_seconds']):
            raise RuntimeError('Residual timestamps exceed measured wall interval')
    record.update(completion_wall_seconds=elapsed, constructor_wall_seconds=constructor_seconds,
        warm_prime_seconds=prime_seconds, warm_prime_success=prime_success,
        warm_prime_error=prime_error, error=failure,
        workload_success=bool(record.get('verified_success') and prime_success and failure is None),
        observer_seconds=observer.observer_seconds, timing_instrumented=True,
        timing_scope='warm: constructor/independent unadapted prime excluded; cold: constructor included; adaptation, rejected trials, fallback and observer included; reference solve/plot/export excluded',
        initial_residual_timestamp_scope='zero initial iterate at t=0; first accepted timestamp includes setup/initial checks',
        setup_scope='excluded' if regime == 'warm' else 'included',
        online_adaptation_seconds=record.get('stats', {}).get('online_adaptation_seconds', 0.))
    return record, (result.x.copy() if result is not None else None)


def load_run(run_dir):
    import run_v6_7_light_smoother_subset as subset
    study = subset.study
    source = Path(run_dir).resolve()
    if not (source/'data/development_manifest.json').is_file():
        raise FileNotFoundError('An already prepared/fitted light run is required')
    print('Verifying saved data, imported P, fit and training provenance (no training)...', flush=True)
    if (source/'subset/manifest.json').exists():
        source, settings, cfg, selection = subset.load(source)
        baseline = study.load_baseline(source/'fixed_classical.json')
        arms = subset.arms(source, settings, cfg, baseline, selection, include_online=True)
        artifacts = subset.artifacts(source)
    else:
        source, settings, cfg, baseline = study.load(source)
        arms = study._arms(source, settings, cfg, baseline, include_online=True)
        artifacts = study._artifacts(source)
    _, examples, _ = study._data(source, settings, baseline)
    from adaptive_mg.v67.light_smoother_evaluation import _validated_arms
    # Trace verbosity does not change numerical controls or trained weights.
    arms = {name: dict(arm, config=replace(arm.get('config', cfg), record_trace=True)) for name, arm in arms.items()}
    arms, metadata = _validated_arms(arms, cfg, baseline)
    return source, settings, cfg, baseline, arms, metadata, examples, artifacts


def replay(args):
    from scipy.sparse.linalg import splu
    from adaptive_mg.v67.research_evaluation import manufactured_rhs
    from adaptive_mg.v67.limited import vector_digest
    source, settings, cfg, baseline, arms, metadata, examples, artifacts = load_run(args.run_dir)
    if args.all_validation:
        chosen = examples
    else:
        chosen = [e for e in examples if e.n == args.n and matches_index(e.name, args.sample_index)]
        if len(chosen) != len(FAMILIES) or {e.case.family for e in chosen} != set(FAMILIES):
            raise ValueError('Need one recorded validation example per family; check --n and --sample-index')
    if any(e.research_split != 'validation' for e in chosen):
        raise ValueError('Only validation replay is allowed')
    output = output_directory(args.output, source, resume=args.resume)
    manifest = dict(version=VERSION, mode='checkpoint_replay', source_run=str(source),
        source_artifacts=artifacts, runner_sha256=digest_file(__file__), arms=metadata,
        missing_arms={name: 'Not available in this fitted run; never substituted' for name in ARMS if name not in arms},
        examples=[e.manifest() for e in chosen], total_validation_cases=len(examples),
        all_validation=args.all_validation, n=args.n, sample_index=args.sample_index,
        rhs_index=args.rhs_index, regimes=args.regimes, repeats=args.repeats, order_seed=args.order_seed,
        dpi=args.dpi, hardware=hardware_environment(refresh=True), diagnostic_only=True, model_selection=False,
        rhs_scope='One independently solved RHS per workload, with zero initial guess and fresh coefficients; not the tail of an already adapted RHS batch',
        reference='Fine-grid sparse LU on the same A and b, computed outside solver timing',
        timing_scope='Actual absolute wall checkpoints; observer overhead included; use uninstrumented validation for performance claims')
    commit_manifest(output, manifest, args.resume)
    signatures = {name: arm['model'].signature() for name, arm in arms.items() if arm.get('model') is not None}
    predictor_signatures = {name: _hash(arm['predictor'].to_dict()) for name, arm in arms.items() if arm.get('predictor') is not None}
    total = len(chosen)*len(arms)*len(args.regimes)*args.repeats
    done, index_rows = 0, []
    print(f'Replay: {len(chosen)} validation A, {len(arms)} arms, {args.regimes}, repeats={args.repeats}: {total} timed solves plus warm priming.', flush=True)
    for name, reason in manifest['missing_arms'].items():
        print(f'Unavailable: {name}: {reason}', flush=True)
    for case_index in np.random.default_rng(args.order_seed).permutation(len(chosen)):
        example = chosen[int(case_index)]
        rhs_all, _ = manufactured_rhs(example, args.rhs_index+2)
        rhs, prime_rhs = rhs_all[args.rhs_index], rhs_all[args.rhs_index+1]
        reference_path = output/'references'/(example.name+'.json')
        ref_inputs = dict(case=example.name, operator_digest=example.digest, rhs_digest=vector_digest(rhs))
        if reference_path.exists():
            checked_record(reference_path, ref_inputs)
            with np.load(reference_path.with_suffix('.npz')) as arrays:
                reference = arrays['reference'].copy()
        else:
            reference = splu(example.a.tocsc()).solve(rhs)
            residual = stable_norm(rhs-example.a@reference)
            threshold = max(cfg.mg.absolute_tolerance, cfg.mg.tolerance*stable_norm(rhs))
            if not np.isfinite(reference).all() or not np.isfinite(residual) or residual > threshold:
                raise ValueError(f'LU reference did not meet the original tolerance: {example.name}')
            signed_record(reference_path, ref_inputs, dict(residual=residual, threshold=threshold),
                          dict(reference=reference, rhs=rhs))
        jobs = [(name, regime, repeat) for repeat in range(args.repeats) for name in arms for regime in args.regimes]
        rng = np.random.default_rng(np.random.SeedSequence([args.order_seed, int(case_index)]))
        for job_index in rng.permutation(len(jobs)):
            name, regime, repeat = jobs[int(job_index)]
            key = dict(manifest_digest=_hash(manifest), case=example.name, arm=name, regime=regime, repeat=repeat)
            path = output/'records'/(_hash(key)+'.json')
            if path.exists():
                saved = checked_record(path, key)
            else:
                arm = arms[name]
                record, solution = replay_measure(example, arm, arm['config'], baseline,
                                                  rhs, reference, prime_rhs, regime)
                saved = signed_record(path, key, record, dict(solution=solution if solution is not None else np.array([])))
            record = saved['data']
            index_rows.append(dict(case=example.name, family=example.case.family, n=example.n,
                arm=name, regime=regime, repeat=repeat, record=str(path.relative_to(output)),
                success=record.get('workload_success', False), rhs_success=record.get('verified_success', False),
                accepted_cycles=record.get('cycles'), executed_cycles=record.get('executed_cycles'),
                all_executed_cycles=record.get('stats', {}).get('all_executed_cycles', record.get('executed_cycles')),
                wall_seconds=record['completion_wall_seconds'], observer_seconds=record['observer_seconds'],
                online_seconds=record['online_adaptation_seconds'],
                rollback_count=record.get('stats', {}).get('rollback_count', 0),
                classical_safety_cycles=max(0, record.get('stats', {}).get('classical_cycles', 0)
                                           - record.get('stats', {}).get('classical_bypass_cycles', 0)),
                online_calls=record.get('stats', {}).get('online_adaptation_calls', 0),
                online_accepted=record.get('stats', {}).get('online_adaptation_accepted', 0),
                relative_l2_error=record.get('relative_solution_error'),
                relative_energy_error=record.get('relative_energy_error'), stop_reason=record.get('stop_reason', record.get('error'))))
            done += 1
            _write_json(output/'progress.json', dict(status='running', completed_measurements=done, total_measurements=total))
            print(f'[{done}/{total}] {example.name} {name} {regime} repeat={repeat}: '
                  f'{"success" if record.get("workload_success") else "FAILED"}, cycles={record.get("cycles")}', flush=True)
    if any(arms[name]['model'].signature() != sig for name, sig in signatures.items()):
        raise RuntimeError('Replay mutated a base checkpoint in memory')
    if any(_hash(arms[name]['predictor'].to_dict()) != sig for name, sig in predictor_signatures.items()):
        raise RuntimeError('Replay mutated a base coefficient predictor in memory')
    if any(digest_file(value['path']) != value['sha256'] for value in artifacts.values()):
        raise RuntimeError('Source artifacts changed during replay')
    index_rows.sort(key=lambda r: (r['case'], r['regime'], r['repeat'], r['arm']))
    write_csv(output/'replay_metrics.csv', index_rows)
    _write_json(output/'index.json', index_rows)
    plot_replay(output, args.dpi)
    _write_json(output/'progress.json', dict(status='complete', completed_measurements=done, total_measurements=total))
    print(f'Replay complete: {output}. Instrumented diagnostic; no model selection.', flush=True)
    return index_rows


def plot_replay(output, dpi):
    plt = plotting()
    from matplotlib.colors import LogNorm
    manifest, rows = read(output/'manifest.json'), read(output/'index.json')
    # Repeat zero is fixed in advance; never select the fastest/best repeat.
    selected = [r for r in rows if r['repeat'] == 0]
    for regime in manifest['regimes']:
        for n in sorted({r['n'] for r in selected}):
            items = {}
            for row in selected:
                if row['regime'] == regime and row['n'] == n and matches_index(row['case'], manifest['sample_index']):
                    items.setdefault(row['family'], {})[row['arm']] = read(output/row['record'])['data']
            if not items:
                continue
            for arm in manifest['arms']:
                if arm in ('fixed_C', 'em5_C'):
                    continue
                for axis in ('cycles', 'wall'):
                    curve_figure(items, arm, axis=axis, dpi=dpi,
                        title=f'{LABELS[arm]} | replay {regime}, n={n}, RHS index={manifest["rhs_index"]}, repeat=0\n'
                              + ('Actual wall checkpoints; diagnostic observer overhead included' if axis == 'wall' else 'Accepted updates; all attempts counted separately in replay_metrics.csv'),
                        path=output/axis/f'{regime}_n{n}_{arm}')
        for case in sorted({r['case'] for r in selected if r['regime'] == regime}):
            subset = [r for r in selected if r['case'] == case and r['regime'] == regime]
            n = subset[0]['n']
            with np.load(output/'references'/(case+'.npz')) as arrays:
                reference = arrays['reference']
            errors = {}
            for row in subset:
                with np.load((output/row['record']).with_suffix('.npz')) as arrays:
                    solution = arrays['solution']
                if solution.shape == reference.shape and np.isfinite(solution).all():
                    errors[row['arm']] = np.abs(solution-reference)
            maximum = max((float(v.max()) for v in errors.values()), default=1.)
            maximum = max(maximum, 1e-16)
            norm = LogNorm(maximum*1e-6, maximum)
            fig, axs = plt.subplots(2, 4, figsize=(14.2, 7.4), layout='constrained')
            art = None
            for ax, arm in zip(axs.ravel(), ARMS):
                row = next((r for r in subset if r['arm'] == arm), None)
                if arm not in errors:
                    ax.text(.5, .5, 'No finite solution / unavailable', ha='center', va='center', wrap=True)
                    ax.set_title(LABELS[arm]); ax.axis('off')
                    continue
                full = np.zeros((n+2, n+2)); full[1:-1, 1:-1] = errors[arm].reshape(n, n)
                coordinates = np.arange(n+2)/(n+1)
                art = ax.pcolormesh(coordinates, coordinates, np.maximum(full.T, norm.vmin),
                                   norm=norm, cmap='magma', shading='nearest', rasterized=True)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
                ax.set_title(f'{LABELS[arm]} | {"OK" if row["success"] else "FAILED"}\n'
                             f'rel. L2={row["relative_l2_error"]:.2e}, safety cycles={row["classical_safety_cycles"]}', fontsize=9)
                ax.set(xlabel='x', ylabel='y')
            if art is not None:
                fig.colorbar(art, ax=axs.ravel().tolist(), shrink=.82, label='Absolute nodal error |u - u_LU| (common log scale; lower values clipped)')
            fig.suptitle(f'{case} | {regime}, RHS index={manifest["rhs_index"]}, repeat=0\nDiscrete linear-system error, not continuum discretization error')
            save_figure(fig, output/'errors'/f'{regime}_{case}', dpi)
    (output/'README_KR.md').write_text(
        '# Checkpoint replay 진단\n\n학습된 checkpoint를 읽고 validation의 A,b를 다시 풀었습니다. 재학습·모델 선택·final/OOD 사용은 없습니다.\n'
        'cycles/: 승인된 해 갱신–상대 잔차. wall/: 실제 외부 시작 시각부터 잔차 관측까지의 경과 시간. '
        'cold는 constructor 포함, warm은 constructor와 독립 RHS prime 제외입니다. '
        'warm prime 동안 적응을 끈 뒤 원래 예측 계수와 적응 예산으로 초기화합니다.\n\n'
        '기존 solver의 clock 호출을 관측해 timestamp를 기록하며 수치 연산은 바꾸지 않습니다. '
        '관측 overhead는 wall 시간 안에 포함됩니다. 시간 우열의 최종 판단에는 원래의 비계측 validation을 사용하세요. '
        'TTT·후보 검사·거절·fallback 시간은 wall 안에 포함되므로 online_seconds를 다시 더하지 않습니다.\n\n'
        'errors/: 같은 A,b의 fine-grid LU 기준 해 대비 최종 절대 오차. 기준 LU 계산과 그림 저장은 비교 시간 밖입니다. '
        'NPZ는 references/와 records/에 보존합니다. 같은 오차 색상 범위를 모든 비교군에 적용하며 작은 값은 하한에서 표시합니다.\n'
        '그림은 사전에 정한 repeat=0입니다. 모든 반복·실패·fallback은 replay_metrics.csv에 보존합니다. '
        'NN 곡선의 네모는 classical safety step입니다. 성공해도 순수 NN만으로 성공했다고 주장하지 마세요.\n\n'
        'RHS index는 기존 manufactured_rhs의 동일 번호입니다. 각 진단 workload는 fresh hierarchy/계수에서 RHS 한 개를 풀므로 '
        'RHS=16 batch의 앞선 15개 RHS에서 적응한 상태를 재현하는 결과는 아닙니다. '
        'RHS batch의 실제 성능/일관성은 기존 validation report를 사용합니다.\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--output', required=True)
    common.add_argument('--resume', action='store_true')
    common.add_argument('--sample-index', type=int, default=0)
    common.add_argument('--rhs-index', type=int, default=0)
    common.add_argument('--dpi', type=int, default=160)
    p = commands.add_parser('report', parents=[common], help='Plot an existing completed validation; no solves')
    p.add_argument('--benchmark-dir', required=True)
    p.add_argument('--regimes', nargs='+', choices=('cold', 'warm', 'multiple', 'warm_multiple'))
    p.add_argument('--rhs-counts', nargs='+', type=int)
    p = commands.add_parser('replay', parents=[common], help='Load fitted checkpoints and solve validation examples for actual wall traces/fields')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--n', type=int, default=63)
    p.add_argument('--all-validation', action='store_true')
    p.add_argument('--regimes', nargs='+', choices=('cold', 'warm'), default=['warm'])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--order-seed', type=int, default=4317)
    args = parser.parse_args(argv)
    if (args.sample_index < 0 or args.rhs_index < 0 or not 50 <= args.dpi <= 600
            or (args.command == 'replay' and (args.repeats < 1 or args.order_seed < 0))):
        parser.error('Require nonnegative indices/seed, positive repeats and 50<=dpi<=600')
    if args.regimes and len(args.regimes) != len(set(args.regimes)):
        parser.error('Regimes must be distinct')
    return globals()[args.command](args)


if __name__ == '__main__':
    main()
