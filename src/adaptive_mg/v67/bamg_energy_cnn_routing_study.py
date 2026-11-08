"""Three hybrid-label CNNs and validation-frozen finest-grid routing.

Every lambda uses the same development operators and paired seed/epoch policy.
The numerical comparison supplies verified teacher settings, not the CNN winner.
No final/OOD operator or result participates in training or router selection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
from pathlib import Path
import time

import numpy as np

from . import bamg_energy_cnn_study as paired
from . import bamg_energy_cnn_routing as routing
from . import bamg_energy_cnn_routing_benchmark as validation_benchmark
from .bamg_energy_cnn_final import study_lock
from .bamg_energy_study import digest, file_digest, read_json, write_json

VERSION = 'bamg-energy-cnn-grid-routing-study-v1'
PROJECT = paired.PROJECT
_initialize = paired._initialize
BASELINES = ['fixed_C', 'em5_C', 'bootstrap_C']


def resolve_settings(requested, source_settings, teacher_settings, cfg, tc):
    if requested.get('version') != VERSION:
        raise ValueError('Unsupported grid-routing study configuration')
    translated = {**deepcopy(requested), 'version': paired.VERSION}
    settings = paired.resolve_settings(translated, source_settings, teacher_settings, cfg, tc)
    settings['version'] = VERSION
    normalized = routing.validate_routing_settings(settings)
    settings['lambdas'] = normalized['lambdas']
    settings['routing'] = normalized['routing']
    if settings.get('numerical_baselines') != BASELINES:
        raise ValueError('Grid routing requires fixed_C, em5_C and bootstrap_C controls')
    if set(settings['data']['train']['sizes']) != set(settings['data']['validation']['sizes']):
        raise ValueError('Routing validation must cover every TRAIN grid')
    if settings.get('sampling') != 'epoch_shuffle':
        raise ValueError('Grid routing requires complete epoch_shuffle training')
    if settings.get('holdout') is None:
        raise ValueError('Declare final/OOD plan before training the grid router')
    return settings


def _manifest(settings, source, teacher):
    manifest = paired._manifest(settings, source, teacher)
    manifest['version'] = VERSION
    path = PROJECT / 'scripts/run_v6_7_bamg_energy_cnn_routing_study.py'
    manifest['sources'][str(path.relative_to(PROJECT))] = file_digest(path)
    return manifest


def _development_reference(directory, settings, source, teacher):
    previous = read_json(Path(directory) / 'configuration.json')
    for key in ('version', 'lambdas', 'routing', 'numerical_baselines'):
        if previous.get(key) != settings.get(key):
            raise ValueError('Routing development reference protocol differs: ' + key)
    return paired._development_reference(directory, settings, source, teacher)


def _label_root(out, strength):
    return Path(out) / 'labels' / ('lambda_' + routing.lambda_key(strength))


def arm_provenance(out, manifest, strength):
    return dict(study=digest(manifest), prepared=read_json(Path(out) / 'prepared_manifest.json'),
        teacher_manifest_sha256=file_digest(_label_root(out, strength) / 'teacher/train/manifest.json'),
        teacher_bank_manifest_sha256=file_digest(Path(out) / 'teacher_bank_manifest.json'),
        init_weight_sha256=manifest['source']['init_weight_sha256'], lambda_value=float(strength))


def provenance(out, manifest):
    settings = read_json(Path(out) / 'configuration.json')
    return dict(study=digest(manifest), prepared=read_json(Path(out) / 'prepared_manifest.json'),
        teacher_bank_manifest_sha256=file_digest(Path(out) / 'teacher_bank_manifest.json'),
        teacher_manifest_sha256_by_lambda={routing.lambda_key(value):
            file_digest(_label_root(out, value) / 'teacher/train/manifest.json')
            for value in settings['lambdas']},
        init_weight_sha256=manifest['source']['init_weight_sha256'])


def routed_provenance(out, manifest):
    return {**provenance(out, manifest),
            'routing_selection_sha256': file_digest(Path(out) / 'routing_selection.json')}


def checkpoint_map(out, settings, selection=None, *, include_routed=True):
    out = Path(out)
    checkpoints = {'old_cnn': out / 'inputs/initial_weight.pt'}
    for strength in settings['lambdas']:
        for seed in settings['model_seeds']:
            for arm in ('scratch', 'warm'):
                root = out / 'runs' / routing.run_name(strength, arm, seed)
                for stage, filename in (('supervised', 'supervised_models.pt'),
                                        ('finetuned', 'deployment.pt')):
                    checkpoints[routing.method_name(strength, arm, seed, stage)] = root / filename
    if include_routed:
        if selection is None:
            selection = read_json(out / 'routing_selection.json')
        checkpoints.update(routed_checkpoint_map(out, settings, selection))
    return checkpoints


def routed_checkpoint_map(out, settings, selection):
    out = Path(out)
    return {routing.routed_name(arm, seed): dict(kind='grid_router_v1',
        policy=deepcopy(selection['arms'][arm]), checkpoints={routing.lambda_key(value):
            out / 'runs' / routing.run_name(value, arm, seed) / 'deployment.pt'
            for value in settings['lambdas']})
        for arm in ('scratch', 'warm') for seed in settings['model_seeds']}


def _audit_label_banks(out, examples, results, manifest):
    """Assert that changing lambda never changes the selected BAMG vector bank."""
    if len(examples) != len(results):
        raise ValueError('Every TRAIN operator requires one teacher label')
    records = {}
    for example, result in zip(examples, results):
        diagnostics = result.diagnostics
        records[example['case_id']] = {key: diagnostics[key] for key in
            ('vector_digests', 'reference_A_digests', 'reference_P_digests',
             'support_digests', 'selected_pass')}
    audit = dict(study=digest(manifest), records=records,
                 policy='Same selected-pass BAMG vectors/support per TRAIN operator for every lambda')
    path = Path(out) / 'teacher_bank_manifest.json'
    if path.exists():
        if read_json(path) != audit:
            raise ValueError('Lambda candidates do not share the same TRAIN vector bank/support')
    else:
        write_json(path, audit)


def seed_summary(fixed, routed, settings):
    groups = {}
    for strength in settings['lambdas']:
        for arm in ('scratch', 'warm'):
            for stage in ('supervised', 'finetuned'):
                members = [routing.method_name(strength, arm, seed, stage)
                           for seed in settings['model_seeds']]
                groups[f'lambda_{routing.lambda_key(strength)}_{arm}_{stage}'] = (fixed, members)
    for arm in ('scratch', 'warm'):
        groups[f'routed_{arm}_finetuned'] = (routed,
            [routing.routed_name(arm, seed) for seed in settings['model_seeds']])
    result = {}
    for name, (summary, members) in groups.items():
        values = [summary['methods'][member] for member in members]
        times = [value['warm_seconds'] for value in values]
        result[name] = dict(members=members, model_seeds=settings['model_seeds'],
            mean_warm_seconds=float(np.mean(times)),
            std_warm_seconds=float(np.std(times, ddof=1)) if len(times) > 1 else None,
            failures=sum(v['failures'] for v in values), attempts=sum(v['attempts'] for v in values),
            eligible=all(v['eligible'] for v in values))
    return dict(groups=result, fastest_seed_selection=False,
        note='Across-seed variation, not a confidence interval. Routed validation is a separate measurement pass.')


def run(config, output, *, source_study, teacher_comparison, resume=False,
        stop_after=None, development_study=None):
    requested = read_json(config)
    source, source_settings = paired.source_inputs(source_study)
    teacher, teacher_settings, cfg, tc = paired.teacher_inputs(teacher_comparison)
    settings = resolve_settings(requested, source_settings, teacher_settings, cfg, tc)
    if development_study is not None:
        if Path(development_study).resolve() == Path(output).resolve():
            raise ValueError('Development reference and new output must differ')
        settings['development_reference'] = _development_reference(development_study, settings, source, teacher)
    _initialize(settings)
    manifest = _manifest(settings, source, teacher)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with study_lock(out):
        if (out / 'research_freeze.json').exists() or (out / 'final/claim.json').exists():
            raise ValueError('Study is frozen; training and routing selection are closed')
        path = out / 'study_manifest.json'
        if path.exists():
            if not resume:
                raise FileExistsError('Study exists; use --resume')
            if read_json(path) != manifest:
                raise ValueError('Source/configuration/inputs/environment changed; use a new study directory')
        else:
            if resume or any(p.name != '.cnn-comparison.lock' for p in out.iterdir()):
                raise FileExistsError('New study requires an empty output directory')
            write_json(path, manifest)
        write_json(out / 'configuration.json', settings)
        started = time.perf_counter()
        train, validation, init_weight, inventory = paired._prepare(out, settings, source, teacher, cfg)
        paired._runtime_record(out, 'prepare_data', started)
        completions = {}
        # Only one lambda's potentially large label set occupies memory at once.
        for strength in settings['lambdas']:
            key = routing.lambda_key(strength)
            started = time.perf_counter()
            teacher_identity = dict(study=digest(manifest), prepared=inventory, lambda_value=float(strength))
            results, _ = paired._labels(_label_root(out, strength), train, cfg, tc,
                settings['teacher_optimizer'], strength, teacher_identity)
            _audit_label_banks(out, train, results, manifest)
            paired._runtime_record(out, 'prepare_labels_lambda_' + key, started)
            arm_identity = arm_provenance(out, manifest, strength)
            for index, seed in enumerate(settings['model_seeds']):
                arms = ('scratch', 'warm') if index % 2 == 0 else ('warm', 'scratch')
                for arm in arms:
                    name = routing.run_name(strength, arm, seed)
                    print(f'Train {name}: finest-grid routing candidate', flush=True)
                    started = time.perf_counter()
                    result = paired.train_arm(out / 'runs' / name, arm=arm, seed=seed, settings=settings,
                        train=train, validation=validation, teacher_results=results, init_weight=init_weight,
                        provenance=arm_identity, resume=resume, stop_after=stop_after)
                    paired._runtime_record(out, name, started)
                    completions[name] = result
                    if result['status'] != 'complete':
                        progress = dict(version=VERSION, completed=False, runs=completions)
                        write_json(out / 'progress.json', progress)
                        return progress
            del results
            gc.collect()
        write_json(out / 'progress.json', dict(completed=False, runs=completions))
        started = time.perf_counter()
        fixed = validation_benchmark.benchmark(out / 'validation', examples=validation, settings=settings,
            checkpoints=checkpoint_map(out, settings, include_routed=False),
            identity=provenance(out, manifest), resume=resume)
        paired._runtime_record(out, 'validation_fixed', started)
        selection = routing.select_routes(fixed, settings)
        selection_path = out / 'routing_selection.json'
        if selection_path.exists():
            if read_json(selection_path) != selection:
                raise ValueError('Saved routing selection differs from full fixed-CNN validation')
        else:
            write_json(selection_path, selection)
        started = time.perf_counter()
        routed = validation_benchmark.benchmark(out / 'validation_routed', examples=validation,
            settings=settings, checkpoints=routed_checkpoint_map(out, settings, selection),
            identity=routed_provenance(out, manifest), resume=resume)
        paired._runtime_record(out, 'validation_routed', started)
        report = dict(version=VERSION, completed=fixed['completed'] and routed['completed'],
            lambdas=settings['lambdas'], numerical_selected_lambda=teacher['selected_lambda'],
            source_model=source['selected_model'], init_weight_sha256=source['init_weight_sha256'],
            train_cases=len(train), validation_cases=len(validation), model_seeds=settings['model_seeds'],
            runs=completions, benchmark=fixed, routed_benchmark=routed, routing_selection=selection,
            seed_summary=seed_summary(fixed, routed, settings),
            interpretation='Validation selects a policy, not a test winner. Fixed budgets do not certify convergence; final/OOD remains unopened.')
        write_json(out / 'comparison.json', report)
        print_report(report)
        return report


def print_report(result):
    print(f'\nThree-lambda CNN development validation; complete={result["completed"]}')
    print('Candidate lambdas:', result['lambdas'])
    for arm, policy in result['routing_selection']['arms'].items():
        print(f'{arm}: grid -> lambda {policy["grid_to_lambda"]}; global fixed lambda={policy["global_lambda"]}')
    print('Unseen grids: largest validation grid no greater than n; clamp outside observed range.')
    for title, summary in (('Fixed candidates', result['benchmark']),
                           ('Routed deployment (separate validation pass)', result['routed_benchmark'])):
        print('\n' + title)
        print(f'{"Method":48s} {"Warm ms*":>12s} {"Fail/total":>14s} {"Speedup":>9s}')
        for name, value in summary['methods'].items():
            speed = value.get('speedup')
            formatted = '--' if speed is None else f'{speed:.3f}'
            print(f'{name:48s} {1000 * value["warm_seconds"]:12.3f} '
                  f'{value["failures"]:5d}/{value["attempts"]:<8d} {formatted:>9s}')
    print('* Failed/missing solves retain penalties. Setup excluded. No fallback or best-seed selection.')
    print('Use freeze, then evaluate-final: all fixed and routed methods are paired on the heldout cases.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'report', 'freeze', 'evaluate-final', 'report-final'))
    parser.add_argument('--config')
    parser.add_argument('--source-study')
    parser.add_argument('--teacher-comparison', help='Completed numerical comparison; inherits verified teacher settings')
    parser.add_argument('--development-study', help='Smaller routing study sharing validation and nested TRAIN')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Testing: cap new updates in each unfinished arm')
    parser.add_argument('--max-cases', type=int, help='Pause final after this many new completed operators')
    parser.add_argument('--group')
    args = parser.parse_args(argv)
    if args.command in ('freeze', 'evaluate-final', 'report-final'):
        from .bamg_energy_cnn_routing_final import freeze, evaluate_final, report_final
        if args.command == 'freeze':
            freeze(args.output)
        elif args.command == 'evaluate-final':
            result = evaluate_final(args.output, resume=args.resume, max_cases=args.max_cases)
            print(f'Final/OOD completed: {result["completed"]}')
        else:
            report_final(args.output, group=args.group)
    elif args.command == 'report':
        path = Path(args.output) / 'comparison.json'
        if not path.exists():
            parser.error('Comparison is not complete yet; resume run')
        print_report(read_json(path))
    else:
        if not args.config or not args.source_study or not args.teacher_comparison:
            parser.error('run requires --config, --source-study and --teacher-comparison')
        run(args.config, args.output, source_study=args.source_study,
            teacher_comparison=args.teacher_comparison, resume=args.resume, stop_after=args.stop_after,
            development_study=args.development_study)


if __name__ == '__main__':
    main()
