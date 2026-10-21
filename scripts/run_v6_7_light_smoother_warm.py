#!/usr/bin/env python3
"""Warm-only validation of an already fitted valid-label light smoother run.

All validation operators and all eight arms are required. This writes a new
diagnostic directory, never replaces the original benchmark or selects a model.
The numerical evaluator, source fingerprints and training checkpoints are reused.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v6_7_light_smoother_subset as subset

study = subset.study
VERSION = 'light-smoother-warm-diagnostic-v1'
EXPECTED_ARMS = {'fixed_C', 'em5_C', *study.BASE_ARMS, *study.ONLINE_ARMS, 'frozen_P_line'}


def benchmark(run_dir, *, tag='validation_warm_rhs1_16', repeats=1, warmups=1,
              rhs_counts=(1, 16), resume=False):
    if not re.fullmatch(r'validation_warm_[A-Za-z0-9_]+', tag):
        raise ValueError('Use a separate tag starting with validation_warm_')
    if type(repeats) is not int or repeats < 1 or type(warmups) is not int or warmups < 0:
        raise ValueError('Invalid repeat/warmup count')
    if (not rhs_counts or len(set(rhs_counts)) != len(rhs_counts)
            or any(type(k) is not int or k not in (1, 4, 16, 64) for k in rhs_counts)):
        raise ValueError('Use distinct RHS counts from 1, 4, 16, 64')
    out = Path(run_dir).resolve()
    if not out.is_dir():
        raise ValueError('An existing fitted subset run is required')
    target = out/'benchmarks'/tag
    if target.resolve() != target:
        raise ValueError('Diagnostic output must not follow a symlink')
    with study._lease(out):
        out, settings, cfg, selection = subset.load(out)
        study._ensure_development_open(out)
        baseline = study.load_baseline(out/'fixed_classical.json')
        _, examples, _ = study._data(out, settings, baseline)
        if (len(examples) != selection['validation_operators']
                or any(e.research_split != 'validation' for e in examples)):
            raise ValueError('The complete validation cohort is required')
        models = subset.arms(out, settings, cfg, baseline, selection, include_online=True)
        if set(models) != EXPECTED_ARMS:
            raise ValueError('All eight arms are required, including the identical-P frozen_P_line checkpoint')
        protocol = dict(repeats=repeats, warmups=warmups,
                        rhs_counts=list(rhs_counts), regimes=['warm_multiple'])
        manifest = dict(version=VERSION, runner_sha256=subset.digest_file(Path(__file__)),
            source_run=str(out), source_artifacts=subset.artifacts(out), protocol=protocol,
            validation_cases=[e.name for e in examples], arms=sorted(models),
            evaluation_filtering=False, diagnostic_only=True, model_selection=False,
            timing_scope='setup and independent unadapted prime excluded; adaptation, rejected trials and fallback during timed solves included')
        path = target/'diagnostic_manifest.json'
        if resume:
            if not path.exists() or subset.read(path) != manifest:
                raise ValueError('Warm diagnostic resume source/model/protocol mismatch')
        else:
            if target.exists():
                raise FileExistsError('Warm diagnostic directory exists; use identical --resume or a new tag')
            subset.write_json(path, manifest, exclusive=True)
        total = len(examples)*len(models)*len(rhs_counts)*repeats
        print(f'Warm validation: ALL {len(examples)} A, {len(models)} arms, '
              f'RHS={list(rhs_counts)}, repeats={repeats}, warmups={warmups}; '
              f'{total} timed measurements plus warmups.', flush=True)
        print(f'Progress: {target / "progress.json"}', flush=True)
        # A process may have stopped after writing our manifest but before the
        # evaluator created its own manifest. Resume that boundary safely too.
        evaluation_resume = resume and (target/'run_manifest.json').exists()
        result = study.evaluate_light(examples, models, cfg, baseline, target,
                                      resume=evaluation_resume, **protocol)
        subset.report(result, target, selection)
        subset.write_json(target/'diagnostic_status.json', dict(
            version=VERSION, status='complete', diagnostic_only=True,
            model_selection=False, manifest_digest=subset.study._hash(manifest),
            note='Warm validation only; setup-inclusive validation is still required for freeze/final selection.'))
        print(f'Warm diagnostic complete: {target / "comparison.csv"}', flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Existing fitted subset run')
    parser.add_argument('--tag', default='validation_warm_rhs1_16')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--rhs-counts', nargs='+', type=int, default=[1, 16])
    parser.add_argument('--resume', action='store_true')
    return benchmark(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
