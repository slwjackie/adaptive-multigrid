"""Direct CNN with cached sparse smoothing; fixed EM5 classical comparison.

Stage S freezes the imported direct P; stage SP fine-tunes P and S together.
Teachers, pruning and online TTT are outside this independent experiment.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np

from ..hierarchy import classical_cycle
from ..provenance import write_json, json_safe, hardware_environment
from .banks import Stats, hybrid_cycle
from ..pde import validate_discretization
from .config import AdaptiveConfig
from .direct_pruning_study import (_read, _initialize, _source_snapshot, _verify_import,
    _development, _ensure_development_open, _validate_protocol, _verified_checkpoint)
from .fixed_classical import FixedClassicalBaseline, load_baseline
from .limited import digest_file, forced_config
from .models import Components
from .research_data import (freeze_research, claim_final_evaluation, materialize_final_data,
    complete_final_evaluation, HOLDOUT, _hash)
from .research_evaluation import evaluate_research, manufactured_rhs
from .research_training import train_expert
from .solver import PreparedAdaptiveMG
from .sparse_smoother import SparsePolynomialSmoother
from .spatial import SpatialState

VERSION = 'direct-cnn-sparse-smoother-v1'
PROJECT = Path(__file__).resolve().parents[3]
ARMS = ('fixed_C', 'direct_CNN', 'direct_CNN_sparse', 'direct_CNN_sparse_joint')


def validate_settings(settings):
    validate_discretization(settings.get('discretization', 'p1'))
    if settings.get('version') != VERSION:
        raise ValueError('Use a sparse smoother study configuration')
    cfg = AdaptiveConfig.from_dict(settings['solver'])
    FixedClassicalBaseline.from_config(cfg)
    raw = settings['solver']['mg']
    for key in ('strategy_name', 'em_iterations', 'em_fixed_iterations', 'em_rtol', 'pre_steps', 'post_steps'):
        if raw.get(key) != getattr(cfg.mg, key):
            raise ValueError(f'Explicit fixed baseline setting required: {key}')
    if (cfg.branch != 'H_SP' or cfg.mode != 'research' or not cfg.use_smoother or not cfg.use_transfer
            or cfg.smoother_levels != 'all' or cfg.transfer_levels != 'all'
            or cfg.replace_pre != cfg.mg.pre_steps or cfg.replace_post != cfg.mg.post_steps
            or cfg.replacement_group_pre != 1 or cfg.replacement_group_post != 1
            or cfg.spatial or cfg.gate_mode != 'open' or cfg.use_learned_controller
            or cfg.application != 'replace' or not cfg.lazy):
        raise ValueError('Require all-level sparse replacement of every pre/post slot, with learned P')
    for key in ('seed', 'torch_threads', 'training_per_family', 'validation_per_family', 'raw_cycles', 'raw_probes'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('training_sizes', 'validation_sizes'):
        values = settings[key]
        if (not values or len(set(values)) != len(values)
                or any(type(n) is not int or n < 3 or n & (n + 1) for n in values)):
            raise ValueError('Use distinct 2**L-1 grid sizes >=3')
    for key, names in (('smoother_training', ['smoother']), ('joint_training', ['smoother', 'transfer'])):
        train = settings[key]
        if (type(train['updates']) is not int or train['updates'] < 1 or train.get('prefix_cycles', 0) < 1
                or train.get('tail_cycles') != 0 or train.get('trainable_components') != names
                or train.get('lambda_kd', 0) != 0 or train.get('kd_updates', 0) != 0):
            raise ValueError('Declare positive task-only stage budgets and the exact trainable components')
    SparsePolynomialSmoother(**settings['sparse_smoother'])
    return cfg


def _sources():
    return sorted([p for p in (PROJECT / 'src').rglob('*') if p.suffix in ('.py', '.cpp')]
                  + [PROJECT / 'scripts/run_v6_7_sparse_smoother_study.py'])


def _manifest(settings, baseline, provenance):
    return dict(version=VERSION, settings_digest=_hash(settings), baseline_digest=baseline.digest(),
        imported_provenance_digest=_hash(provenance),
        source_digest=_hash({str(p.relative_to(PROJECT)): digest_file(p) for p in _sources()}),
        hardware=hardware_environment(refresh=True))


def load(output):
    out = Path(output).resolve()
    settings = _read(out / 'configuration.json')
    cfg = validate_settings(settings)
    _initialize(settings)
    baseline = load_baseline(out / 'fixed_classical.json')
    baseline.validate_config(cfg)
    if _read(out / 'sparse_smoother_manifest.json') != json_safe(_manifest(settings, baseline, _verify_import(out))):
        raise ValueError('Sources/settings/imports/environment changed; start a new run')
    return out, settings, cfg, baseline


def prepare(config, source_run, output, *, resume=False):
    settings = _read(config)
    cfg = validate_settings(settings)
    _initialize(settings)
    baseline = FixedClassicalBaseline.from_config(cfg)
    out, source = Path(output).resolve(), Path(source_run).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('Use a separate run directory outside the source run')
    if (out / 'sparse_smoother_manifest.json').exists():
        if not resume:
            raise FileExistsError('Run exists; use --resume or a new directory')
        _, stored, _, _ = load(out)
        if settings != stored or _verify_import(out)['source_run'] != str(source):
            raise ValueError('Prepare resume settings/source differ')
        _ensure_development_open(out)
    else:
        if resume:
            raise FileNotFoundError('No sparse smoother study to resume')
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Prepare needs an empty output directory')
        model, relative = _source_snapshot(source, baseline)
        model = deepcopy(model)
        model.smoother = SparsePolynomialSmoother(**settings['sparse_smoother'])
        model.metadata.update(research_version=VERSION, role='student', final_test_seen=False,
            smoother_trained=False, sparse_schedule='all levels, 2 pre + 2 post SpMV', teacher_used=False)
        out.mkdir(parents=True)
        files = {}
        for rel in relative:
            src, dest = source / rel, out / 'imported/source' / rel
            expected = digest_file(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            if digest_file(dest) != expected or digest_file(src) != expected:
                raise ValueError('Source import changed while copying')
            files[rel] = dict(copy=str(dest.relative_to(out)), sha256=expected)
        initial = out / 'imported/sparse_initial.pt'
        model.save(initial, extra={'role': 'untrained_sparse_smoother'})
        files['sparse_initial'] = dict(copy=str(initial.relative_to(out)), sha256=digest_file(initial))
        provenance = dict(version=VERSION, source_run=str(source), files=files,
            source_teacher_untouched=True, data_policy='fresh operator-disjoint development and sealed holdouts',
            offline_training_scope='S: imported P frozen; SP: extra joint P/S fine-tuning; no distillation or TTT')
        write_json(out / 'imported/provenance.json', provenance)
        write_json(out / 'configuration.json', settings)
        write_json(out / 'fixed_classical.json', baseline.to_dict())
        write_json(out / 'sparse_smoother_manifest.json', _manifest(settings, baseline, provenance))
    train_data, validation, _ = _development(out, settings, baseline)
    print(f'Sparse smoother prepared: train={len(train_data)}, validation={len(validation)}; 5 shared coefficients', flush=True)


def _checkpoint(out, name, baseline):
    path = out / 'checkpoints' / name / 'candidate.pt'
    status = _read(path.parent / 'status.json')
    if (status.get('status') != 'complete' or not status.get('all_updates_completed')
            or status.get('skipped') or not status.get('feasible_task_updates')
            or status.get('checkpoint_sha256') != digest_file(path)
            or status.get('baseline_digest') != baseline.digest() or status.get('teacher_used_for_training')):
        raise ValueError('Require a complete, checksum-verified sparse training checkpoint')
    model = Components.load(path)
    if (not isinstance(model.smoother, SparsePolynomialSmoother)
            or model.metadata.get('training_branch') != 'H_SP'
            or model.metadata.get('training_baseline_digest') != baseline.digest()):
        raise ValueError('Wrong sparse smoother checkpoint lineage')
    return model


def train(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if (out / 'benchmark_evidence.json').exists():
        raise ValueError('Validation already recorded; start a new run to change training')
    examples, _, _ = _development(out, settings, baseline)
    initial = Components.load(out / 'imported/sparse_initial.pt')
    for stage, key in (('sparse', 'smoother_training'), ('joint', 'joint_training')):
        target = out / 'checkpoints' / stage
        initial, status = train_expert(initial, examples, cfg, baseline, settings[key], target,
            branch='H_SP', resume=resume and (target / 'training_manifest.json').exists())
        if not status['all_updates_completed'] or status['skipped'] or not status['feasible_task_updates']:
            raise RuntimeError('Sparse training failed; inspect training.json before evaluation')
    _arms(out, settings, baseline)


def _arms(out, settings, baseline):
    direct = _verified_checkpoint(out / 'imported/source/checkpoints/direct/candidate.pt', baseline)
    sparse = _checkpoint(out, 'sparse', baseline)
    joint = _checkpoint(out, 'joint', baseline)
    if direct.expert_signature('transfer') != sparse.expert_signature('transfer'):
        raise ValueError('S-only training changed imported P')
    for name, model, expected in (('sparse', sparse, ['smoother']), ('joint', joint, ['smoother', 'transfer'])):
        record = _read(out / 'checkpoints' / name / 'training_manifest.json')
        key = 'smoother_training' if name == 'sparse' else 'joint_training'
        if (model.metadata.get('trainable_components') != expected
                or record['settings'] != settings[key] or record['config'] != settings['solver']):
            raise ValueError('Sparse checkpoint training protocol changed')
    return {'fixed_C': dict(model=None, branch='C', selector=False),
        'direct_CNN': dict(model=direct, branch='H_P', selector=False),
        'direct_CNN_sparse': dict(model=sparse, branch='H_SP', selector=False),
        'direct_CNN_sparse_joint': dict(model=joint, branch='H_SP', selector=False)}


def raw_contractions(examples, arms, cfg, *, cycles, probes):
    """Independent fixed-cycle diagnostic. Never calls solve or runtime fallback."""
    rows = []
    for example in examples:
        rhs, _ = manufactured_rhs(example, probes)
        row = dict(case=example.name, n=example.n, arms={})
        for name, arm in arms.items():
            acfg = forced_config(cfg, arm['branch'])
            try:
                prepared = PreparedAdaptiveMG(example.a, example.n, arm['model'], acfg)
                root = prepared.ensure_branch(arm['branch'], Stats())
                histories = []
                for b in rhs:
                    x = np.zeros_like(b); norms = []; spatial = SpatialState(arm['model'], acfg) if arm['model'] else None
                    for k in range(cycles):
                        x = (classical_cycle(root, x, b, acfg.mg, Stats()) if arm['branch'] == 'C' else
                             hybrid_cycle(root, x, b, acfg, Stats(), spatial, k, True))
                        ratio = float(np.linalg.norm(b - example.a @ x) / max(np.linalg.norm(b), 1e-300))
                        if not np.isfinite(ratio):
                            raise FloatingPointError('Nonfinite raw contraction')
                        norms.append(ratio)
                    histories.append(norms)
                row['arms'][name] = dict(valid=True, histories=histories)
            except (ValueError, RuntimeError, FloatingPointError) as error:
                row['arms'][name] = dict(valid=False, reason=str(error))
        rows.append(row)
    return dict(fallback_allowed=False, cycles=cycles, probes=probes,
        scope='raw complete V-cycles; failure is not replaced by classical output; not a timing benchmark', rows=rows)


def cycle_diagnostics(result):
    """Pair on common successful cases; report executed and accepted learned cycles."""
    report = {}
    for regime, counts in result['summary'].items():
        for count in counts:
            for name in ARMS[1:]:
                rows = []; failures = []
                for case in result['rows']:
                    base, runs = (case['runs'][key][regime][count] for key in ('fixed_C', name))
                    if not all(r['successful'] for r in base + runs):
                        failures.append(case['example']['name']); continue
                    med = lambda values, key: float(np.median([r[key] for r in values]))
                    rows.append(dict(case=case['example']['name'], n=case['example']['case']['n'],
                        classical_cycles=med(base, 'executed_cycles'), candidate_cycles=med(runs, 'executed_cycles'),
                        learned_cycles=med(runs, 'accepted_neural_cycles'), rollbacks=med(runs, 'rollback_count'),
                        fallback_cycles=med(runs, 'fallback_count'), setup_failures=med(runs, 'setup_failures')))
                item = dict(common_cases=len(rows), excluded_failure_cases=failures,
                    fewer=sum(r['candidate_cycles'] < r['classical_cycles'] for r in rows),
                    equal=sum(r['candidate_cycles'] == r['classical_cycles'] for r in rows),
                    more=sum(r['candidate_cycles'] > r['classical_cycles'] for r in rows),
                    classical_cycle_sum=sum(r['classical_cycles'] for r in rows),
                    candidate_cycle_sum=sum(r['candidate_cycles'] for r in rows), rows=rows)
                report[f'{regime}/rhs{count}/{name}'] = item
    return report


def _report(result, target):
    from .unet_transfer_study import _report as common_report
    common_report(result, target)
    diagnostics = cycle_diagnostics(result)
    write_json(target / 'cycle_diagnostics.json', diagnostics)
    for key, row in diagnostics.items():
        print(key, 'cycles fewer/equal/more=', (row['fewer'], row['equal'], row['more']), flush=True)


def benchmark(output, *, tag='validation', resume=False, **protocol):
    protocol = _validate_protocol(protocol)
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('Use a simple benchmark tag')
    _, validation, _ = _development(out, settings, baseline)
    arms = _arms(out, settings, baseline); target = out / 'benchmarks' / tag
    result = evaluate_research(validation, arms, cfg, baseline, target,
        reference_arm='fixed_C', resume=resume, **protocol)
    _report(result, target)
    write_json(target / 'raw_contractions.json', raw_contractions(validation, arms, cfg,
        cycles=settings['raw_cycles'], probes=settings['raw_probes']))
    files = {key: dict(path=str(target / key), sha256=digest_file(target / key))
        for key in ('raw_results.json', 'cycle_diagnostics.json', 'raw_contractions.json')}
    write_json(out / 'benchmark_evidence.json', dict(files=files,
        checkpoints={n: a['model'].signature() for n, a in arms.items() if a['model']}, performance_certified=False))
    return result


def freeze(output, **protocol):
    protocol = _validate_protocol(protocol, final=True)
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    arms = _arms(out, settings, baseline)
    evidence = _read(out / 'benchmark_evidence.json')
    if evidence['checkpoints'] != {n: a['model'].signature() for n, a in arms.items() if a['model']}:
        raise ValueError('Checkpoint changed after validation')
    paths = {}
    for key, value in evidence['files'].items():
        if digest_file(value['path']) != value['sha256']:
            raise ValueError('Validation evidence changed')
        paths[key] = value['path']
    paths.update(configuration=out / 'configuration.json', manifest=out / 'sparse_smoother_manifest.json',
        provenance=out / 'imported/provenance.json', evidence=out / 'benchmark_evidence.json')
    paths.update({'import:' + key: out / value['copy'] for key, value in _verify_import(out)['files'].items()})
    checkpoints = {'direct_CNN': out / 'imported/source/checkpoints/direct/candidate.pt',
        'direct_CNN_sparse': out / 'checkpoints/sparse/candidate.pt',
        'direct_CNN_sparse_joint': out / 'checkpoints/joint/candidate.pt'}
    evaluation = dict(version=VERSION, solver=cfg.to_dict(), arms=list(arms),
        raw_cycles=settings['raw_cycles'], raw_probes=settings['raw_probes'], **protocol)
    return freeze_research(out / 'data', checkpoints, out / 'fixed_classical.json', paths, _sources(), evaluation)


def final(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    path = out / 'data/research_freeze.json'
    evaluation = _read(path)['config']
    if (evaluation['version'] != VERSION or evaluation['solver'] != cfg.to_dict()
            or evaluation['raw_cycles'] != settings['raw_cycles'] or evaluation['raw_probes'] != settings['raw_probes']):
        raise ValueError('Wrong frozen sparse smoother evaluation')
    arms = _arms(out, settings, baseline)
    if list(arms) != evaluation['arms']:
        raise ValueError('Frozen evaluation arms changed')
    with claim_final_evaluation(out / 'data', path, evaluation, resume=resume) as claim:
        splits, _ = materialize_final_data(out / 'data', claim, baseline)
        outputs = []
        for name in evaluation['splits']:
            target = out / 'final' / name
            result = evaluate_research(splits[name], arms, cfg, baseline, target, reference_arm='fixed_C',
                resume=resume and (target / 'run_manifest.json').exists(),
                **{key: evaluation[key] for key in ('repeats', 'warmups', 'regimes', 'rhs_counts')})
            _report(result, target)
            outputs.extend(target / key for key in ('raw_results.json', 'cycle_diagnostics.json'))
        complete_final_evaluation(out / 'data', claim, outputs)
    print('Fresh final/OOD sparse smoother evaluation sealed.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'train', 'benchmark', 'freeze', 'final'):
        p = sub.add_parser(name); p.add_argument('--run-dir', required=True)
        if name != 'freeze': p.add_argument('--resume', action='store_true')
        if name == 'prepare':
            p.add_argument('--source-run', required=True)
            p.add_argument('--config', default='configs/v6_7_sparse_smoother_research.json')
        if name == 'benchmark': p.add_argument('--tag', default='validation')
        if name in ('benchmark', 'freeze'):
            p.add_argument('--repeats', type=int, default=3); p.add_argument('--warmups', type=int, default=1)
            p.add_argument('--rhs-counts', nargs='+', type=int, default=[1, 4, 16, 64])
            p.add_argument('--regimes', nargs='+', choices=('cold', 'warm', 'multiple', 'warm_multiple'),
                           default=['cold', 'warm', 'multiple'])
        if name == 'freeze': p.add_argument('--splits', nargs='+', choices=HOLDOUT, default=list(HOLDOUT))
    a = parser.parse_args(argv)
    if a.command == 'prepare': return prepare(a.config, a.source_run, a.run_dir, resume=a.resume)
    if a.command == 'train': return train(a.run_dir, resume=a.resume)
    if a.command == 'final': return final(a.run_dir, resume=a.resume)
    protocol = dict(repeats=a.repeats, warmups=a.warmups, rhs_counts=a.rhs_counts, regimes=a.regimes)
    if a.command == 'freeze': return freeze(a.run_dir, splits=a.splits, **protocol)
    return benchmark(a.run_dir, tag=a.tag, resume=a.resume, **protocol)
