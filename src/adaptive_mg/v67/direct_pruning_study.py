"""Imported direct CNN -> structured pruning -> fine-tuning -> optional online TTT.

The source U-net study is read-only. This experiment owns its configuration,
fresh development/holdout data, checkpoints, timing records and final freeze.
Teacher/distilled checkpoints are neither read nor trained. TTT has no access
to evaluation solutions and is charged to setup in cold/multiple workloads.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import shutil

import numpy as np
import torch

from ..provenance import write_json, json_safe, hardware_environment
from .config import AdaptiveConfig
from .direct_pruning import prune_direct
from .evaluation_metrics import _speed_summary
from .fixed_classical import FixedClassicalBaseline, load_baseline
from .limited import digest_file, initialize_timing_runtime
from .models import Components
from .research_data import (freeze_research, claim_final_evaluation,
    materialize_final_data, complete_final_evaluation, make_research_plan,
    prepare_development_data, load_development_data, HOLDOUT, _hash)
from .research_evaluation import evaluate_research, _max_error, REGIMES
from .research_training import train_expert
from .unet_transfer import CompactTransferStudent

VERSION = 'direct-cnn-pruning-ttt-v1'
SOURCE_VERSION = 'residual-unet-fixed-em-study-v2'
PROJECT = Path(__file__).resolve().parents[3]
ARMS = ('fixed_C', 'direct_CNN', 'pruned_CNN', 'pruned_CNN_TTT')


def _read(path):
    return json.loads(Path(path).read_text())


def validate_settings(settings):
    if settings.get('version') != VERSION:
        raise ValueError('Use a direct-CNN pruning/TTT study configuration')
    cfg = AdaptiveConfig.from_dict(settings['solver'])
    FixedClassicalBaseline.from_config(cfg)
    raw = settings['solver']['mg']
    for key in ('strategy_name', 'em_iterations', 'em_fixed_iterations', 'em_rtol', 'pre_steps', 'post_steps'):
        if raw.get(key) != getattr(cfg.mg, key):
            raise ValueError(f'Explicit fixed baseline setting required: {key}')
    if (cfg.mode != 'research' or cfg.branch != 'H_P' or cfg.use_smoother
            or not cfg.use_transfer or cfg.transfer_levels != 'all' or cfg.spatial
            or cfg.use_learned_controller or not cfg.lazy or cfg.application != 'replace'
            or cfg.gate_mode != 'open'):
        raise ValueError('Pruning/TTT needs forced H_P, all P levels and classical smoothing')
    for key in ('seed', 'torch_threads', 'pruned_hidden', 'training_per_family', 'validation_per_family'):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('training_sizes', 'validation_sizes'):
        values = settings[key]
        if (not values or len(set(values)) != len(values)
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 3 or n & (n + 1) for n in values)):
            raise ValueError('Use distinct 2**L-1 grid sizes >=3')
    train = settings['training']
    if int(train['updates']) < 1 or int(train.get('prefix_cycles', 0)) < 1:
        raise ValueError('Require positive fine-tuning and learned V-cycle budgets')
    if float(train.get('lambda_kd', 0.)) != 0. or int(train.get('kd_updates', 0)) != 0:
        raise ValueError('Direct pruning fine-tuning does not use teacher distillation')
    from .transfer_ttt import validate_ttt_settings
    if settings['ttt'] != validate_ttt_settings(settings['ttt']):
        raise ValueError('TTT configuration must declare every canonical setting')
    if settings['ttt']['max_updates'] < 1:
        raise ValueError('The TTT comparison arm requires at least one online update')
    deployment = settings['deployment']
    if float(deployment['minimum_speedup']) <= 1. or int(deployment['min_cases']) < 1:
        raise ValueError('Invalid setup-inclusive deployment thresholds')
    return cfg


def _initialize(settings):
    torch.set_num_threads(settings['torch_threads'])
    initialize_timing_runtime()


def _sources():
    return sorted([p for p in (PROJECT / 'src').rglob('*') if p.suffix in ('.py', '.cpp')]
                  + [PROJECT / 'scripts/run_v6_7_direct_pruning_study.py'])


def _manifest(settings, baseline, provenance):
    return dict(version=VERSION, settings_digest=_hash(settings), baseline_digest=baseline.digest(),
        imported_provenance_digest=_hash(provenance),
        source_digest=_hash({str(p.relative_to(PROJECT)): digest_file(p) for p in _sources()}),
        hardware=hardware_environment(refresh=True))


def _ensure_development_open(out):
    if (out / 'data/research_freeze.json').exists() or (out / 'data/final_claim.json').exists():
        raise ValueError('This run is frozen; start a new development run')


def _verified_checkpoint(path, baseline):
    status = _read(path.parent / 'status.json')
    if (not status.get('all_updates_completed') or status.get('skipped')
            or status.get('feasible_task_updates', 0) < 1 or status.get('status') != 'complete'):
        raise ValueError('Checkpoint needs complete, feasible training without skipped updates')
    if digest_file(path) != status['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after training')
    model = Components.load(path)
    if (model.metadata.get('training_baseline_digest') != baseline.digest()
            or status.get('baseline_digest') != baseline.digest()
            or model.metadata.get('training_branch') != 'H_P'):
        raise ValueError('Checkpoint is not bound to the fixed H_P baseline')
    if not isinstance(model.transfer, CompactTransferStudent) or model.transfer.body_kind != 'residual5':
        raise ValueError('Only the direct residual5 CNN can enter this pruning study')
    if model.metadata.get('role') == 'teacher' or status.get('teacher_used_for_training'):
        raise ValueError('Import the directly trained CNN, not teacher/distilled weights')
    return model


def _source_snapshot(source, baseline):
    """Validate data/checkpoint lineage without applying old source-code guards.

    A new experiment necessarily runs new sources. It may import old weights,
    but never rewrite/unfreeze the source experiment or ignore weight/data hashes.
    """
    settings = _read(source / 'configuration.json')
    manifest = _read(source / 'unet_transfer_manifest.json')
    if settings.get('version') != SOURCE_VERSION or manifest.get('version') != SOURCE_VERSION:
        raise ValueError('Expected a fixed-EM residual-U-net source study')
    if manifest.get('settings_digest') != _hash(settings) or manifest.get('baseline_digest') != baseline.digest():
        raise ValueError('Source settings/baseline manifest mismatch')
    source_baseline = load_baseline(source / 'fixed_classical.json')
    if source_baseline.digest() != baseline.digest():
        raise ValueError('Source and pruning study must use identical fixed classical settings')
    source_cfg = AdaptiveConfig.from_dict(settings['solver'])
    source_baseline.validate_config(source_cfg)
    checkpoint = source / 'checkpoints/direct/candidate.pt'
    model = _verified_checkpoint(checkpoint, baseline)
    training = _read(checkpoint.parent / 'training_manifest.json')
    _, _, development = load_development_data(source / 'data', baseline)
    trained = {r['normalized_operator_digest'] for r in development['splits']['train']}
    if (training.get('baseline_digest') != baseline.digest() or training.get('branch') != 'H_P'
            or training.get('teacher_signature') is not None
            or training.get('config') != source_cfg.to_dict()
            or training.get('settings') != settings['training']
            or set(training['sample_digests']) != trained
            or set(model.metadata.get('training_operator_digests', [])) != trained):
        raise ValueError('Source direct checkpoint/data/training lineage mismatch')
    relative = ['configuration.json', 'unet_transfer_manifest.json', 'fixed_classical.json',
        'data/dataset_plan.json', 'data/development_manifest.json',
        'checkpoints/direct/candidate.pt', 'checkpoints/direct/status.json',
        'checkpoints/direct/training_manifest.json']
    freeze_path = source / 'data/research_freeze.json'
    if freeze_path.exists():
        frozen = _read(freeze_path)
        if frozen.get('config_digest') != _hash(frozen['config']):
            raise ValueError('Source final freeze configuration changed')
        required = {'checkpoint:direct_CNN': checkpoint, 'baseline': source / 'fixed_classical.json',
                    'plan': source / 'data/dataset_plan.json',
                    'development': source / 'data/development_manifest.json'}
        for key, path in required.items():
            if frozen['files'].get(key, {}).get('sha256') != digest_file(path):
                raise ValueError(f'Source frozen import changed: {key}')
        relative.append('data/research_freeze.json')
    return model, relative


def _verify_import(out):
    provenance = _read(out / 'imported/provenance.json')
    for record in provenance['files'].values():
        if digest_file(out / record['copy']) != record['sha256']:
            raise ValueError(f'Imported input changed: {record["copy"]}')
    return provenance


def load(output):
    out = Path(output).resolve()
    settings = _read(out / 'configuration.json')
    cfg = validate_settings(settings)
    _initialize(settings)
    baseline = load_baseline(out / 'fixed_classical.json')
    baseline.validate_config(cfg)
    provenance = _verify_import(out)
    if _read(out / 'pruning_manifest.json') != json_safe(_manifest(settings, baseline, provenance)):
        raise ValueError('Sources/settings/imports/environment changed; start a new run')
    return out, settings, cfg, baseline


def _development(out, settings, baseline):
    data = out / 'data'
    if (data / 'development_manifest.json').exists():
        return load_development_data(data, baseline)
    plan = make_research_plan(smoke=bool(settings['smoke']), seed=int(settings['seed']) + 1700003)
    for name, key, per in (('train', 'training_sizes', 'training_per_family'),
                            ('validation', 'validation_sizes', 'validation_per_family')):
        spec = plan['splits'][name]
        spec.update(sizes=settings[key], per_family=settings[per])
        spec['count'] = len(spec['sizes']) * len(spec['families']) * spec['per_family']
    provenance = _verify_import(out)
    roots = list(dict.fromkeys([provenance['source_run'], str(out.parent)] + settings.get('historical_roots', [])))
    return prepare_development_data(data, baseline, plan=plan, historical_roots=roots)


def prepare(config, source_run, output, *, hidden=None, resume=False):
    settings = _read(config)
    if hidden is not None:
        settings['pruned_hidden'] = hidden
    cfg = validate_settings(settings)
    _initialize(settings)
    baseline = FixedClassicalBaseline.from_config(cfg)
    out, source = Path(output).resolve(), Path(source_run).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('Use a separate run directory, outside the source run')
    if (out / 'pruning_manifest.json').exists():
        if not resume:
            raise FileExistsError('Run exists; use --resume or a new directory')
        _, stored, _, _ = load(out)
        if settings != stored or _verify_import(out)['source_run'] != str(source):
            raise ValueError('Prepare resume settings/source differ')
        _ensure_development_open(out)
    else:
        if resume:
            raise FileNotFoundError('No pruning study to resume')
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Prepare needs an empty output directory')
        model, relative = _source_snapshot(source, baseline)
        pruned, report = prune_direct(model, hidden=settings['pruned_hidden'])
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for rel in relative:
            src, dest = source / rel, out / 'imported/source' / rel
            expected = digest_file(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            if digest_file(dest) != expected or digest_file(src) != expected:
                raise ValueError('Source import changed while copying')
            files[rel] = dict(copy=str(dest.relative_to(out)), sha256=expected)
        pruned.metadata.update(research_version=VERSION, role='student', final_test_seen=False,
            pruning_source_checkpoint_sha256=digest_file(source / 'checkpoints/direct/candidate.pt'))
        initial = out / 'imported/pruned_initial.pt'
        pruned.save(initial, extra={'role': 'structured_pruned_initial', 'requires_fine_tuning': True})
        write_json(out / 'imported/pruning_report.json', report)
        for label, path in (('pruned_initial', initial), ('pruning_report', out / 'imported/pruning_report.json')):
            files[label] = dict(copy=str(path.relative_to(out)), sha256=digest_file(path))
        provenance = dict(version=VERSION, source_run=str(source), files=files,
            source_teacher_untouched=True, source_code_digest=_read(source / 'unet_transfer_manifest.json')['source_digest'],
            import_scope='verified direct checkpoint and development lineage; old source/final results are not resumed',
            data_policy='fresh normalized-operator-disjoint train, validation and sealed holdouts',
            offline_training_scope='original direct is reused; pruning arm receives additional declared fine-tuning')
        write_json(out / 'imported/provenance.json', provenance)
        write_json(out / 'configuration.json', settings)
        write_json(out / 'fixed_classical.json', baseline.to_dict())
        write_json(out / 'pruning_manifest.json', _manifest(settings, baseline, provenance))
    train_data, validation, _ = _development(out, settings, baseline)
    report = _read(out / 'imported/pruning_report.json')
    print(f"Direct pruning prepared: width {report['original_hidden']} -> {report['hidden']}; "
          f"parameters {report['original_parameter_count']} -> {report['pruned_parameter_count']}; "
          f"fresh train={len(train_data)}, validation={len(validation)}", flush=True)
    return report


def train(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if (out / 'benchmark_evidence.json').exists():
        raise ValueError('Validation already recorded; use a new run to change training')
    examples, _, _ = _development(out, settings, baseline)
    initial = Components.load(out / 'imported/pruned_initial.pt')
    initial.metadata['pruning_finetuning'] = dict(requested_updates=settings['training']['updates'],
        initial_pruning_report_scope='immediately_after_channel_slicing_before_fine_tuning',
        teacher_used=False, completion_record='optimizer_updates and checkpoints/pruned/status.json')
    _, status = train_expert(initial, examples, cfg, baseline, settings['training'],
        out / 'checkpoints/pruned', branch='H_P', teacher=None, resume=resume)
    if not status['all_updates_completed'] or status['skipped'] or not status['feasible_task_updates']:
        raise RuntimeError('Pruning fine-tuning did not complete usable learned-P updates')
    return status


def _arms(out, settings, baseline):
    direct = _verified_checkpoint(out / 'imported/source/checkpoints/direct/candidate.pt', baseline)
    pruned = _verified_checkpoint(out / 'checkpoints/pruned/candidate.pt', baseline)
    if pruned.transfer.hidden != settings['pruned_hidden']:
        raise ValueError('Pruned checkpoint architecture differs from configuration')
    return {'fixed_C': dict(model=None, branch='C', selector=False),
            'direct_CNN': dict(model=direct, branch='H_P', selector=False),
            'pruned_CNN': dict(model=pruned, branch='H_P', selector=False),
            'pruned_CNN_TTT': dict(model=pruned, branch='H_P', selector=False, ttt=settings['ttt'])}


def paired_comparisons(result):
    """Directly measured paired TTT/pruning gains, not timing subtractions."""
    comparisons = {}
    for regime, counts in result['summary'].items():
        for count in counts:
            for reference, candidate in (('direct_CNN', 'pruned_CNN'), ('pruned_CNN', 'pruned_CNN_TTT')):
                ratios, cases, failures, accuracy = [], [], [], []
                for row in result['rows']:
                    base, runs = (row['runs'][name][regime][count] for name in (reference, candidate))
                    base_ok, ok = (bool(values) and all(r['successful'] for r in values) for values in (base, runs))
                    case = row['example']['name']
                    if base_ok and not ok:
                        failures.append(case)
                    if base_ok and ok:
                        ratios.append(float(np.median([r['wall_seconds'] for r in base])) /
                                      float(np.median([r['wall_seconds'] for r in runs])))
                        cases.append(case)
                        for metric in ('relative_solution_error', 'relative_energy_error'):
                            value, reference_error = _max_error(runs, metric), _max_error(base, metric)
                            if value is None or reference_error is None or value > 1.05 * reference_error + 1e-12:
                                accuracy.append(case)
                                break
                comparisons[f'{regime}/rhs{count}/{candidate}_vs_{reference}'] = dict(
                    **_speed_summary(ratios), reference=reference, candidate=candidate,
                    common_case_ids=cases, new_failure_case_ids=failures,
                    worse_accuracy_case_ids=accuracy, setup_included=regime in ('cold', 'multiple'),
                    scope='wall time per workload; warm excludes prior setup/TTT and is not end-to-end evidence')
    return comparisons


def deployment_decisions(result, *, minimum_speedup=1.03, min_cases=5):
    decisions = {}
    for regime in ('cold', 'multiple'):
        for count, arms in result['summary'].get(regime, {}).items():
            for name in ARMS[1:]:
                entry = arms[name]
                common, used = [], []
                for row in result['rows']:
                    base, runs = (row['runs'][key][regime][count] for key in ('fixed_C', name))
                    if all(r['successful'] for r in base + runs):
                        common.append(row['example']['name'])
                        used.append(all(r['actual_neural_used'] for r in runs))
                gain, ci = entry['geometric_speedup'], entry['ci95']
                time_pass = bool(len(common) >= min_cases and not entry['new_failure_case_ids']
                    and all(used) and gain is not None and gain >= minimum_speedup)
                confidence = bool(ci and ci[0] is not None and ci[0] > 1.)
                accuracy = not entry['accuracy_worse_vs_fixed_case_ids']
                decisions[f'{regime}/rhs{count}/{name}'] = dict(
                    observed_time_target_met=time_pass, confidence_supports_speedup=confidence,
                    accuracy_check_passed=accuracy, qualifies=bool(time_pass and confidence and accuracy),
                    geometric_speedup_vs_fixed=gain, ci95=ci, common_case_ids=common,
                    new_failure_case_ids=entry['new_failure_case_ids'], setup_included=True,
                    time_scope='EM parent + neural generation + optional TTT/guard/rebuild + safeguarded solve',
                    performance_certified=False)
    return decisions


def _report(result, target, deployment):
    from .unet_transfer_study import _report as report_common
    report_common(result, target)
    paired = paired_comparisons(result)
    write_json(target / 'paired_pruning_ttt.json', paired)
    decisions = deployment_decisions(result, **deployment)
    write_json(target / 'deployment_decisions.json', decisions)
    for key, value in paired.items():
        print(key, 'paired_speedup=', value['geometric_speedup'], flush=True)
    return decisions


def _validate_protocol(protocol, *, final=False):
    """Reject unusable timing invocations before writing an immutable freeze."""
    allowed = {'repeats', 'warmups', 'regimes', 'rhs_counts'} | ({'splits'} if final else set())
    if set(protocol) - allowed:
        raise ValueError('Unknown evaluation protocol field')
    result = dict(repeats=3, warmups=1, regimes=['cold', 'warm', 'multiple'], rhs_counts=[1, 4, 16, 64])
    result.update(protocol)
    for key, lower in (('repeats', 1), ('warmups', 0)):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < lower:
            raise ValueError('Invalid repeat/warmup count')
    regimes, counts = result['regimes'], result['rhs_counts']
    if not regimes or len(set(regimes)) != len(regimes) or set(regimes) - set(REGIMES):
        raise ValueError('Invalid or duplicate regimes')
    if (not counts or len(set(counts)) != len(counts)
            or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in counts)):
        raise ValueError('Invalid RHS counts')
    result.update(regimes=list(regimes), rhs_counts=list(counts))
    if final:
        if not set(regimes) & {'cold', 'multiple'}:
            raise ValueError('Final evaluation needs a setup-inclusive cold or multiple workload')
        splits = result.setdefault('splits', list(HOLDOUT))
        if not splits or len(set(splits)) != len(splits) or set(splits) - set(HOLDOUT):
            raise ValueError('Use distinct known final/OOD splits')
        result['splits'] = list(splits)
    return result


def benchmark(output, *, tag='validation', resume=False, **protocol):
    protocol = _validate_protocol(protocol)
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('Use a simple benchmark tag')
    _, validation, _ = _development(out, settings, baseline)
    arms = _arms(out, settings, baseline)
    target = out / 'benchmarks' / tag
    result = evaluate_research(validation, arms, cfg, baseline, target,
        reference_arm='fixed_C', resume=resume, **protocol)
    decisions = _report(result, target, settings['deployment'])
    write_json(out / 'benchmark_evidence.json', dict(
        report=str(target / 'raw_results.json'), report_sha256=digest_file(target / 'raw_results.json'),
        decisions=str(target / 'deployment_decisions.json'),
        decisions_sha256=digest_file(target / 'deployment_decisions.json'),
        paired=str(target / 'paired_pruning_ttt.json'), paired_sha256=digest_file(target / 'paired_pruning_ttt.json'),
        checkpoints={name: arm['model'].signature() for name, arm in arms.items() if arm['model']},
        ttt=settings['ttt'], performance_certified=False))
    return decisions


def freeze(output, **protocol):
    protocol = _validate_protocol(protocol, final=True)
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    arms = _arms(out, settings, baseline)
    evidence = _read(out / 'benchmark_evidence.json')
    for key in ('report', 'decisions', 'paired'):
        if digest_file(evidence[key]) != evidence[key + '_sha256']:
            raise ValueError('Pruning/TTT validation evidence changed')
    signatures = {name: arm['model'].signature() for name, arm in arms.items() if arm['model']}
    if signatures != evidence['checkpoints'] or settings['ttt'] != evidence['ttt']:
        raise ValueError('Checkpoint or online adaptation protocol changed after validation')
    paths = {key: evidence[key] for key in ('report', 'decisions', 'paired')}
    paths.update(configuration=out / 'configuration.json', manifest=out / 'pruning_manifest.json',
        provenance=out / 'imported/provenance.json', benchmark_evidence=out / 'benchmark_evidence.json')
    paths.update({'import:' + key: out / value['copy'] for key, value in _verify_import(out)['files'].items()})
    checkpoints = {'direct_CNN': out / 'imported/source/checkpoints/direct/candidate.pt',
                   'pruned_CNN': out / 'checkpoints/pruned/candidate.pt'}
    evaluation = dict(version=VERSION, solver=cfg.to_dict(), arms=list(arms), ttt=settings['ttt'],
                      deployment=settings['deployment'], **protocol)
    return freeze_research(out / 'data', checkpoints, out / 'fixed_classical.json', paths, _sources(), evaluation)


def final(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    path = out / 'data/research_freeze.json'
    evaluation = _read(path)['config']
    if (evaluation['version'] != VERSION or evaluation['solver'] != cfg.to_dict()
            or evaluation['ttt'] != settings['ttt']):
        raise ValueError('Wrong frozen pruning/TTT evaluation')
    arms = _arms(out, settings, baseline)
    if list(arms) != evaluation['arms']:
        raise ValueError('Frozen evaluation arms changed')
    with claim_final_evaluation(out / 'data', path, evaluation, resume=resume) as claim:
        splits, _ = materialize_final_data(out / 'data', claim, baseline)
        outputs = []
        for name in evaluation['splits']:
            examples = splits[name]
            target = out / 'final' / name
            result = evaluate_research(examples, arms, cfg, baseline, target, reference_arm='fixed_C',
                resume=resume and (target / 'run_manifest.json').exists(),
                **{key: evaluation[key] for key in ('repeats', 'warmups', 'regimes', 'rhs_counts')})
            _report(result, target, evaluation['deployment'])
            outputs.extend(target / name for name in ('raw_results.json', 'deployment_decisions.json', 'paired_pruning_ttt.json'))
        complete_final_evaluation(out / 'data', claim, outputs)
    print('Fresh final/OOD pruning/TTT evaluation sealed.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'train', 'benchmark', 'freeze', 'final'):
        p = sub.add_parser(name)
        p.add_argument('--run-dir', required=True)
        if name != 'freeze':
            p.add_argument('--resume', action='store_true')
        if name == 'prepare':
            p.add_argument('--source-run', required=True)
            p.add_argument('--config', default='configs/v6_7_direct_pruning_research.json')
            p.add_argument('--hidden', type=int, default=None)
        if name == 'benchmark':
            p.add_argument('--tag', default='validation')
        if name in ('benchmark', 'freeze'):
            p.add_argument('--repeats', type=int, default=3)
            p.add_argument('--warmups', type=int, default=1)
            p.add_argument('--rhs-counts', nargs='+', type=int, default=[1, 4, 16, 64])
            p.add_argument('--regimes', nargs='+', choices=('cold', 'warm', 'multiple', 'warm_multiple'),
                           default=['cold', 'warm', 'multiple'])
        if name == 'freeze':
            p.add_argument('--splits', nargs='+', choices=HOLDOUT, default=list(HOLDOUT))
    a = parser.parse_args(argv)
    if a.command == 'prepare':
        return prepare(a.config, a.source_run, a.run_dir, hidden=a.hidden, resume=a.resume)
    if a.command == 'train':
        return train(a.run_dir, resume=a.resume)
    if a.command == 'final':
        return final(a.run_dir, resume=a.resume)
    protocol = dict(repeats=a.repeats, warmups=a.warmups, rhs_counts=a.rhs_counts, regimes=a.regimes)
    if a.command == 'freeze':
        return freeze(a.run_dir, splits=a.splits, **protocol)
    return benchmark(a.run_dir, tag=a.tag, resume=a.resume, **protocol)
