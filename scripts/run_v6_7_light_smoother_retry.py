#!/usr/bin/env python3
"""Retry only failed training oracles in a new, provenance-linked light run.

This supplementary experiment has its own source manifest. The original study
implementation and its source/environment checks are deliberately unchanged.
No parent manifest, label, checkpoint, validation, or final record is edited.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
import shutil
import sys
from time import perf_counter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

import numpy as np

from adaptive_mg.provenance import json_safe, operator_digest
from adaptive_mg.v67 import light_smoother_study as study
from adaptive_mg.v67 import light_smoother_search as oracle
from adaptive_mg.v67.light_smoother_runtime import PreparedLightMG
from adaptive_mg.v67.limited import digest_file
from adaptive_mg.v67.research_data import _hash, _write_json as write_json

VERSION = 'light-smoother-joint-retry-v1'
read = study._read


def validate_settings(settings, bounds):
    required = {'version', 'grid', 'candidate_count', 'screen_cycles',
                'finalist_count', 'confirmation_count', 'seed'}
    if set(settings) != required or settings['version'] != VERSION:
        raise ValueError('Require the complete joint-retry configuration')
    grid = oracle.validate_search_settings(dict(grid=settings['grid'], bounds=bounds))['grid']
    if 1. not in grid:
        raise ValueError('Retry grid must contain the all-ones baseline')
    for key in required - {'version', 'grid'}:
        if type(settings[key]) is not int or settings[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'retry.{key} must be an integer in its valid range')
    if settings['candidate_count'] < len(grid)**2:
        raise ValueError('Candidate budget must cover every uniform x/y pair')
    if settings['confirmation_count'] > settings['finalist_count']:
        raise ValueError('Confirmation count cannot exceed finalist count')
    return {**settings, 'grid': grid}


def joint_candidates(levels, spec, operator_key):
    """Baseline, all uniform pairs, then seeded sampling without replacement.

    Each sampled vector sets both axes on ALL levels. The finite grid is
    exhausted only if its cardinality is within the configured budget.
    """
    grid = spec['grid']
    if levels < 1:
        raise ValueError('Require nonterminal levels')
    total = len(grid)**(2*levels)
    count = min(total, spec['candidate_count'])
    codes = []

    def encode(values):
        code = 0
        for value in values:
            code = code*len(grid)+grid.index(value)
        return code

    codes.append(encode([1.]*(2*levels)))
    for pair in product(grid, repeat=2):
        code = encode(list(pair)*levels)
        if code not in codes:
            codes.append(code)
    rng = np.random.default_rng(int(_hash((spec['seed'], operator_key))[:16], 16))
    # Integer grid indices avoid constructing an exponentially large mesh.
    if total <= count:
        codes.extend(i for i in range(total) if i not in codes)
    else:
        while len(codes) < count:
            code = int(rng.integers(total))
            if code not in codes:
                codes.append(code)
    result = []
    for code in codes:
        values = []
        for _ in range(2*levels):
            code, index = divmod(code, len(grid))
            values.append(grid[index])
        result.append(np.asarray(values[::-1]).reshape(levels, 2))
    return result


def _signed_write(path, value):
    value = {k: v for k, v in value.items() if k != 'digest'}
    value['digest'] = _hash(value)
    write_json(path, value)


def _signed_read(path):
    value = read(path)
    if value.get('digest') != _hash({k: v for k, v in value.items() if k != 'digest'}):
        raise ValueError(f'Retry ledger/manifest changed: {path}')
    return value


def retry_case(example, model, cfg, search_settings, spec, parent, output,
               lineage, *, resume=False):
    if getattr(example, 'research_split', None) != 'train':
        raise ValueError('Retry accepts training-split operators only')
    if parent.get('label_eligible') is not False:
        raise ValueError('Retry accepts failed oracle cases only')
    base = oracle.validate_search_settings(search_settings)
    spec = validate_settings(spec, base['bounds'])
    out = Path(output)
    manifest = json_safe(dict(version=VERSION, settings=spec, original_search=base,
        solver=cfg.to_dict(), example=example.name, split='train',
        operator_digest=operator_digest(example.a), normalized_operator_digest=example.group_digest,
        transfer_signature=model.expert_signature('transfer'), parent_result_digest=_hash(parent),
        lineage=lineage, fallback_allowed=False, runner_sha256=digest_file(Path(__file__))))
    path = out/'progress.json'
    if path.exists():
        if not resume:
            raise FileExistsError('Retry case exists; use --resume')
        state = _signed_read(path)
        if state['manifest'] != manifest:
            raise ValueError('Retry source/operator/P/settings changed')
        if state['status'] == 'complete':
            result = read(out/'result.json')
            if _hash(result) != state['result_digest']:
                raise ValueError('Retry result changed')
            return result
    else:
        state = dict(manifest=manifest, status='running', trials=[],
                     offline_seconds=0., setup_seconds=0., setup_count=0)
    started, prior_seconds = perf_counter(), state['offline_seconds']

    def persist():
        state['offline_seconds'] = prior_seconds+perf_counter()-started
        state['completed_measurements'] = len(state['trials'])
        _signed_write(path, state)

    persist()
    try:
        setup_started = perf_counter()
        try:
            prepared = PreparedLightMG(example.a, example.n, model, cfg, bounds=base['bounds'])
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            state.setdefault('setup_failures', []).append(str(exc))
            raise
        finally:
            state['setup_seconds'] += perf_counter()-setup_started
            state['setup_count'] += 1
        if not np.array_equal(prepared.features, np.asarray(parent['features'])):
            raise ValueError('Actual-level features changed from the original fixed-P hierarchy')
        candidates = joint_candidates(len(prepared.get_coefficients()), spec, example.group_digest)
        rhs = {p: oracle.search_rhs(example, count=base[key], seed=base['seed'], partition=p)
               for p, key in [('optimization', 'rhs_count'), ('confirmation', 'confirmation_rhs_count')]}
        rhs_digests = {p: [oracle._vector_digest(b) for b in values] for p, values in rhs.items()}
        if rhs_digests != parent['rhs_digests']:
            raise ValueError('Retry must use the original optimization/confirmation RHS')
        journal = {r['id']: r for r in state['trials']}
        if len(journal) != len(state['trials']):
            raise ValueError('Duplicate retry measurements')

        def evaluate(phase, indices, partition, repeats, solve_cfg):
            groups = {index: [] for index in indices}
            rng = np.random.default_rng(int(_hash((spec['seed'], example.group_digest, phase))[:16], 16))
            order = rng.permutation(indices)
            for repeat in range(repeats):
                for rhs_index, b in enumerate(rhs[partition]):
                    for index in np.roll(order, repeat+rhs_index):
                        index = int(index)
                        identity = f'{phase}/{index}/{repeat}/{rhs_index}'
                        meta = dict(id=identity, phase=phase, partition=partition, candidate_index=index,
                            repeat=repeat, rhs_index=rhs_index, rhs_digest=rhs_digests[partition][rhs_index],
                            coefficients=candidates[index].tolist(), surrogate_only=phase == 'screen')
                        if identity in journal:
                            row = journal[identity]
                            if any(row.get(k) != v for k, v in meta.items()):
                                raise ValueError('Retry measurement provenance changed')
                        else:
                            prepared.set_coefficients(candidates[index])
                            row = {**meta, **oracle.measure_raw_solve(prepared, b, solve_cfg)}
                            state['trials'].append(row)
                            journal[identity] = row
                            persist()
                            if len(journal) % 32 == 0:
                                print(f'  {example.name} {phase}: {len(journal)} measurements', flush=True)
                        groups[index].append(row)
            return groups

        # Screening is a ranking surrogate, NEVER a successful oracle label.
        screen_cfg = replace(cfg, mg=replace(cfg.mg, max_cycles=min(spec['screen_cycles'], cfg.mg.max_cycles)))
        screened = evaluate('screen', list(range(len(candidates))), 'optimization', 1, screen_cfg)
        def score(index):
            ratios = []
            for row in screened[index]:
                if (row['error'] is not None or row['stop_reason'] not in ('converged', 'max_cycles')
                        or row['final_true_residual'] is None):
                    return float('inf')
                ratios.append(row['final_true_residual']/max(row['residual_history'][0], np.finfo(float).tiny))
            return max(ratios)
        ranked = sorted(range(1, len(candidates)), key=lambda i: (score(i), i))
        finalists = [0]+[i for i in ranked if np.isfinite(score(i))][:spec['finalist_count']]
        optimized = evaluate('optimization', finalists, 'optimization', base['repeats'], cfg)
        historical_failures = {oracle._coefficient_key(r['coefficients']) for r in parent['trials']
                               if not r['success']}
        def summarize(records, partition, index):
            summary = oracle.candidate_summary(records, repeats=base['repeats'], rhs_count=len(rhs[partition]))
            summary['historical_failure'] = oracle._coefficient_key(candidates[index]) in historical_failures
            summary['eligible'] = bool(summary['eligible'] and not summary['historical_failure'])
            return summary
        optimization = {i: summarize(rows, 'optimization', i) for i, rows in optimized.items()}
        successful = sorted((i for i in finalists if i != 0 and optimization[i]['eligible']),
                            key=lambda i: (optimization[i]['median_rhs_batch_seconds'], i))
        confirmed_indices = [0]+successful[:spec['confirmation_count']]
        confirmed = evaluate('confirmation', confirmed_indices, 'confirmation', base['repeats'], cfg)
        summaries = []
        for i in confirmed_indices:
            summary = summarize(confirmed[i], 'confirmation', i)
            summary['confirmation_eligible'] = summary['eligible']
            summary['optimization_eligible'] = optimization[i]['eligible']
            summary['eligible'] = bool(summary['eligible'] and optimization[i]['eligible'])
            summaries.append(summary)
        winner, reason = oracle.choose_candidate(summaries, 0, base['minimum_improvement'])
        selected = confirmed_indices[winner]
        valid = bool(summaries[winner]['eligible'])
        result = json_safe(dict(version=VERSION, status='complete', manifest=manifest,
            features=np.asarray(prepared.features).tolist(), selected_coefficients=candidates[selected].tolist(),
            baseline_coefficients=candidates[0].tolist(), selected_candidate_success=valid, label_eligible=valid,
            selection_outcome=reason if valid else 'no_successful_candidate',
            confirmed_improvement=valid and selected != 0,
            timing_improvement_confirmed=valid and reason == 'measured_time_improvement',
            candidates=[c.tolist() for c in candidates], candidate_count=len(candidates),
            grid_cardinality=len(spec['grid'])**(2*len(candidates[0])),
            screening=dict(surrogate_only=True, cycles=screen_cfg.mg.max_cycles,
                           scores={str(i): score(i) for i in screened}, finalists=finalists),
            optimization_summaries=optimization,
            confirmation=dict(candidates=[candidates[i].tolist() for i in confirmed_indices],
                candidate_indices=confirmed_indices, summaries=summaries, selected_index=winner, reason=reason),
            trials=state['trials'], rhs_digests=rhs_digests,
            offline_seconds=prior_seconds+perf_counter()-started, setup_seconds=state['setup_seconds'],
            setup_count=state['setup_count'], fallback_allowed=False, online_adaptation_allowed=False,
            objective='Repeated full solves to unchanged tolerance/cycle limit; median RHS-batch wall time',
            optimality='Sampled joint grid, screened finalists only; not a global optimum',
            deployment_cost_scope='Offline only; deployment uses fresh setup'))
        write_json(out/'result.json', result)
        state.update(status='complete', result_digest=_hash(result))
        persist()
        return result
    except BaseException:
        persist()
        raise


def _snapshot_mapping(source, training):
    names = ['configuration.json', 'fixed_classical.json', 'light_manifest.json', 'import_manifest.json',
             'data/dataset_plan.json', 'data/development_manifest.json', 'search/summary.json']
    names += [r['path'] for r in read(source/'import_manifest.json')['files'].values()]
    names += [f'search/{e.name}/{name}.json' for e in training for name in ('progress', 'result')]
    mapping = {}
    for name in sorted(set(names)):
        path = source/name
        if path.resolve() != path.absolute() or not path.is_file():
            raise ValueError(f'Require regular snapshot files within source run: {name}')
        if not path.resolve().is_relative_to(source):
            raise ValueError('Snapshot path escapes source run')
        target = 'retry/source_summary.json' if name == 'search/summary.json' else name
        mapping[name] = dict(path=target, sha256=digest_file(path))
    return mapping


def _verify_snapshot(out, manifest):
    for entry in manifest['source_files'].values():
        path = out/entry['path']
        if not path.resolve().is_relative_to(out) or digest_file(path) != entry['sha256']:
            raise ValueError(f'Original run snapshot changed: {path}')


def initialize(source_run, output, config):
    source, settings, cfg, baseline = study.load(source_run)
    study._development_mutable(source)
    if (source/'retry/manifest.json').exists():
        raise ValueError('Use the original coordinate-search run as the parent')
    spec = validate_settings(read(config), settings['bounds'])
    training, _, _ = study._data(source, settings, baseline)
    original = read(source/'search/summary.json')
    if original['status'] != 'complete' or len(original['rows']) != len(training):
        raise ValueError('Complete the original search before retrying failed cases')
    model = study._p_model(source)
    failures = []
    for example, row in zip(training, original['rows']):
        expected = f'search/{example.name}/result.json'
        if (row['case'] != example.name or row['normalized_operator_digest'] != example.group_digest
                or row['path'] != expected or digest_file(source/expected) != row['sha256']):
            raise ValueError('Original search summary/operator/hash mismatch')
        # The existing reader verifies the completed ledger, full solver/P
        # settings and result digest, without executing another solve.
        progress = read(source/f'search/{example.name}/progress.json')
        if progress.get('status') != 'complete':
            raise ValueError('Original case search is incomplete')
        result = oracle.search_case(example, model, cfg, dict(settings['search'], bounds=settings['bounds']),
                                    source/f'search/{example.name}', resume=True)
        selected = result['confirmation']['selected_index']
        eligible = result['confirmation']['summaries'][selected]['eligible']
        if not (row['label_eligible'] == result['label_eligible'] == result['selected_candidate_success'] == eligible):
            raise ValueError('Inconsistent original label eligibility')
        if not eligible:
            failures.append(example.name)
    if failures != original['failed_cases'] or not failures:
        raise ValueError('No failed cases, or original failure list is inconsistent')
    manifest = json_safe(dict(version=VERSION, source_run=str(source), settings=spec,
        runner_sha256=digest_file(Path(__file__)), source_files=_snapshot_mapping(source, training),
        failed_cases=failures, preserved_cases=len(training)-len(failures),
        transfer_signature=model.expert_signature('transfer'),
        policy='Retry failed training cases only; keep all original trials and successful labels',
        validation_final_used=False))
    out = Path(output).resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Retry requires a separate sibling run directory')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new empty retry run directory')
    out.mkdir(parents=True, exist_ok=True)
    _signed_write(out/'retry/manifest.json', manifest)
    return out


def retry(output, *, source_run=None, config=None, resume=False):
    out = Path(output).resolve()
    if not (out/'retry/manifest.json').exists():
        if resume or source_run is None:
            raise ValueError('New retry requires --source-run; resume requires an initialized retry run')
        config = config or PROJECT/'configs/v6_7_light_smoother_retry.json'
        # Prevent a concurrent original search from changing the input snapshot.
        with study._lease(Path(source_run).resolve()):
            initialize(source_run, out, config)
    elif not resume:
        raise FileExistsError('Retry run exists; use --resume')
    with study._lease(out):
        manifest = _signed_read(out/'retry/manifest.json')
        if manifest['version'] != VERSION or manifest['runner_sha256'] != digest_file(Path(__file__)):
            raise ValueError('Retry implementation changed; do not rewrite its source manifest')
        if source_run is not None and str(Path(source_run).resolve()) != manifest['source_run']:
            raise ValueError('Retry parent changed')
        if config is not None and read(config) != manifest['settings']:
            raise ValueError('Retry configuration changed')
        marker = out/'retry/snapshot_complete.json'
        if not marker.exists():
            source = Path(manifest['source_run'])
            with study._lease(source):
                study._development_mutable(source)
                for name, entry in manifest['source_files'].items():
                    path, target = source/name, out/entry['path']
                    if digest_file(path) != entry['sha256']:
                        raise ValueError('Parent changed during snapshot; use a new retry run')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                _verify_snapshot(out, manifest)
                write_json(marker, dict(manifest_digest=manifest['digest']))
        if read(marker) != dict(manifest_digest=manifest['digest']):
            raise ValueError('Retry snapshot commit changed')
        _verify_snapshot(out, manifest)
        _, settings, cfg, baseline = study.load(out)
        study._development_mutable(out)
        training, _, _ = study._data(out, settings, baseline)
        original = read(out/'retry/source_summary.json')
        status_path = out/'retry/status.json'
        if status_path.exists() and _signed_read(status_path)['status'] == 'complete':
            status = _signed_read(status_path)
            if digest_file(out/'search/summary.json') != status['summary_sha256']:
                raise ValueError('Merged retry summary changed')
            for row in read(out/'search/summary.json')['rows']:
                if digest_file(out/row['path']) != row['sha256']:
                    raise ValueError('Merged retry label changed')
            return read(out/'search/summary.json')
        write_json(out/'search/summary.json', dict(status='retry_running', retry_manifest=manifest))
        model = study._p_model(out)
        rows, completed, retry_seconds = [], [], 0.
        print(f'Retrying {len(manifest["failed_cases"])} failed training A; '
              f'reusing {manifest["preserved_cases"]} successful labels. '
              f'Tolerance/cycle limit unchanged; fallback OFF.', flush=True)
        for example, old in zip(training, original['rows']):
            if example.name not in manifest['failed_cases']:
                rows.append(old)
                continue
            folder = out/'retry/cases'/example.name
            result = retry_case(example, model, cfg, dict(settings['search'], bounds=settings['bounds']),
                manifest['settings'], read(out/old['path']), folder, manifest['digest'],
                resume=resume and (folder/'progress.json').exists())
            row = dict(old, path=str((folder/'result.json').relative_to(out)),
                sha256=digest_file(folder/'result.json'), label_eligible=result['label_eligible'],
                confirmed_improvement=result['confirmed_improvement'], selection_outcome=result['selection_outcome'],
                offline_seconds=old['offline_seconds']+result['offline_seconds'],
                original_result=dict(path=old['path'], sha256=old['sha256']))
            rows.append(row)
            completed.append(example.name)
            retry_seconds += result['offline_seconds']
            _signed_write(status_path, dict(status='running', completed_cases=completed))
            print(f'[RETRY {len(completed)}/{len(manifest["failed_cases"])}] {example.name}: '
                  f'{result["selection_outcome"]}; label_eligible={result["label_eligible"]}', flush=True)
        summary = dict(original, rows=rows, retry_manifest=manifest,
            improved_cases=sum(r['confirmed_improvement'] for r in rows),
            eligible_cases=sum(r['label_eligible'] for r in rows),
            failed_cases=[r['case'] for r in rows if not r['label_eligible']],
            offline_seconds=sum(r['offline_seconds'] for r in rows),
            original_offline_seconds=original['offline_seconds'], retry_offline_seconds=retry_seconds,
            recovered_cases=[r['case'] for r in rows if r['case'] in completed and r['label_eligible']],
            preserved_cases=manifest['preserved_cases'])
        write_json(out/'search/summary.json', summary)
        _signed_write(status_path, dict(status='complete', completed_cases=completed,
            summary_sha256=digest_file(out/'search/summary.json')))
        print(f'Recovered {len(summary["recovered_cases"])}/{len(completed)}; '
              f'remaining failures: {summary["failed_cases"]}', flush=True)
        print('All labels eligible: run the original fit command on this NEW run.' if not summary['failed_cases']
              else 'fit remains blocked; failed cases were retained, not dropped or relabelled.', flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', help='Completed original coordinate-search run; required for a new retry')
    parser.add_argument('--run-dir', required=True, help='Separate new run; never the original directory')
    parser.add_argument('--config', help='Joint-retry JSON; default configs/v6_7_light_smoother_retry.json')
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args(argv)
    return retry(a.run_dir, source_run=a.source_run, config=a.config, resume=a.resume)


if __name__ == '__main__':
    main()
