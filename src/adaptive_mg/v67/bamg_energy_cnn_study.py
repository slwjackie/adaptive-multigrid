"""Paired initialization study on shared, fresh hybrid-teacher TRAIN labels.

Development choices are frozen before a separate final/OOD evaluation. Existing
studies and their freezes are inputs, never mutated. Use trusted local checkpoints.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from . import bamg_energy_study as numerical
from .bamg_energy_cnn_benchmark import benchmark
from .bamg_energy_cnn_training import train_arm
from .bamg_energy_labels import build_energy_teacher
from . import bamg_energy_holdout_data as holdout_data
from .bamg_study import (
    PROJECT, VERSION as OLD_VERSION, _environment, _examples, _file_hash,
    _initialize, _initialize_weight, _load_pt, _read, _save, _sources, _write,
    validate_settings,
)
from .bamg_teacher import TeacherConfig
from .bamg_transfer import WeightCNN
from .fixed_classical import BilinearClassicalBaseline
from .research_data import (
    FAMILIES, historical_operator_index, make_research_plan,
    prepare_development_data,
)

VERSION = 'bamg-energy-cnn-initialization-comparison-v2'


def _holdout_plan(settings):
    declaration = settings.get('holdout')
    if declaration is None:
        return None
    if set(declaration) - {'seed', 'splits'}:
        raise ValueError('Unknown holdout declaration field')
    plan = holdout_data.make_holdout_plan(smoke=settings['smoke'], seed=declaration['seed'],
        discretization=settings['discretization'], training_sizes=settings['data']['train']['sizes'])
    for name, override in declaration.get('splits', {}).items():
        if name not in plan['splits'] or set(override) - {'sizes', 'families', 'per_family'}:
            raise ValueError('Invalid holdout split override')
        plan['splits'][name].update(override)
    for spec in plan['splits'].values():
        spec['count'] = len(spec['sizes']) * len(spec['families']) * spec['per_family']
    plan['cases_expected'] = sum(s['count'] for s in plan['splits'].values())
    holdout_data.validate_plan(plan)
    if not settings['smoke']:
        training_families = set(settings['data'].get('families', FAMILIES))
        if set(plan['splits']['final']['families']) - training_families:
            raise ValueError('In-distribution final families must have been represented in TRAIN')
    return plan


def source_inputs(source_study):
    """Verify a migrated package without following its obsolete absolute paths."""
    root = Path(source_study).resolve()
    frozen = _read(root / 'data/research_freeze.json')
    if numerical.digest(frozen['config']) != frozen['config_digest']:
        raise ValueError('Source freeze configuration hash changed')
    selection = frozen['config']['selection']['selected_learned_model']
    choices = {
        'bootstrap_weight_cnn_finetuned_uniform': 'deployment_uniform.pt',
        'bootstrap_weight_cnn_finetuned_terminal': 'deployment_terminal.pt',
        'bootstrap_weight_cnn_supervised': 'supervised_models.pt',
    }
    if selection not in choices:
        raise ValueError('Source selection is not a compatible fixed-support Weight CNN')
    hashes = {}
    for name in choices.values():
        hashes[name] = _file_hash(root / name)
        if hashes[name] != frozen['files']['checkpoint:' + name]['sha256']:
            raise ValueError('Source checkpoint differs from frozen original: ' + name)
    # These metadata files are necessary to exclude old training and test A's.
    for relative, key in [('data/dataset_plan.json', 'plan'),
                          ('data/development_manifest.json', 'development'),
                          ('fixed_classical.json', 'baseline' if 'baseline' in frozen['files'] else 'rules')]:
        actual = _file_hash(root / relative)
        if actual != frozen['files'][key]['sha256']:
            raise ValueError('Source metadata differs from frozen original: ' + relative)
        hashes[relative] = actual
    for relative in ['configuration.json', 'study_manifest.json',
                     'data/research_freeze.json', 'data/final_claim.json',
                     'data/final_data_manifest.json']:
        if (root / relative).exists():
            hashes[relative] = _file_hash(root / relative)
    settings = _read(root / 'configuration.json')
    if settings != frozen['config']['settings']:
        raise ValueError('Source configuration differs from its research freeze')
    if (root / 'data/final_claim.json').exists() and not (root / 'data/final_data_manifest.json').exists():
        raise ValueError('Copy final_data_manifest.json as well to exclude previously exposed final operators')
    if (root / 'data/final_claim.json').exists():
        claim = _read(root / 'data/final_claim.json')
        final = _read(root / 'data/final_data_manifest.json')
        if claim['freeze_sha256'] != _file_hash(root / 'data/research_freeze.json'):
            raise ValueError('Old final claim does not match its research freeze')
        if final['claim_fingerprint'] != claim['fingerprint']:
            raise ValueError('Old final metadata does not match its claim')
        if claim.get('final_data_sha256') is not None and claim['final_data_sha256'] != _file_hash(root / 'data/final_data_manifest.json'):
            raise ValueError('Old final metadata hash changed')
    if settings.get('support_mode') != 'fixed_bilinear':
        raise ValueError('Source must use fixed bilinear support')
    model = WeightCNN(**settings['models']['weight']).double()
    checkpoint = root / choices[selection]
    info = _initialize_weight(model, checkpoint)
    return dict(path=str(root), selected_model=selection, checkpoint=choices[selection],
                hashes=hashes, weight_spec=info['weight_spec'],
                init_weight_sha256=info['sha256']), settings


def teacher_inputs(comparison):
    """Recompute lambda selection from all completed scalar case records."""
    root = Path(comparison).resolve()
    settings = _read(root / 'configuration.json')
    cfg, tc = numerical.validate(settings)
    saved_manifest = _read(root / 'comparison_manifest.json')
    if saved_manifest['settings_digest'] != numerical.digest(settings):
        raise ValueError('Teacher comparison configuration hash changed')
    # Adding this entry point is allowed; changing an old numerical kernel is not.
    for relative, expected in saved_manifest['sources'].items():
        path = PROJECT / relative
        if not path.is_file() or _file_hash(path) != expected:
            raise ValueError('Teacher comparison source differs: ' + relative)
    records, hashes = [], {}
    for spec in numerical.specifications(settings):
        relative = 'cases/' + spec['name'] + '/results.json'
        record = _read(root / relative)
        if (record.get('specification') != spec or not record.get('completed')
                or record.get('settings_digest') != numerical.digest(settings)):
            raise ValueError('Teacher comparison has incomplete or incompatible cases')
        records.append(record)
        hashes[relative] = _file_hash(root / relative)
    if len({r['normalized_operator_digest'] for r in records}) != len(records):
        raise ValueError('Teacher comparison repeats normalized operators')
    result = numerical.summarize(records, settings, len(records))
    saved = _read(root / 'teacher_comparison.json')
    for key in ('completed', 'cases_expected', 'cases_completed', 'selected_method',
                'selected_lambda', 'methods'):
        if saved.get(key) != result[key]:
            raise ValueError('Teacher summary disagrees with raw cases: ' + key)
    if not result['completed'] or result['selected_lambda'] is None:
        raise ValueError('Finish teacher comparison and select an eligible lambda first')
    for name in ('configuration.json', 'comparison_manifest.json', 'teacher_comparison.json'):
        hashes[name] = _file_hash(root / name)
    return dict(path=str(root), hashes=hashes, selected_lambda=result['selected_lambda'],
                selected_method=result['selected_method'],
                normalized_operator_digests=sorted(r['normalized_operator_digest'] for r in records)), settings, cfg, tc


def resolve_settings(requested, source_settings, teacher_settings, cfg, tc):
    settings = deepcopy(requested)
    if settings.get('version') != VERSION:
        raise ValueError('Unsupported paired CNN study configuration')
    seeds = settings.get('model_seeds', [])
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Distinct nonnegative model seeds below 2**32 are required')
    if type(settings['data'].get('seed')) is not int or settings['data']['seed'] < 0:
        raise ValueError('Declare a data seed independently of model seeds')
    if settings['models']['weight'] != source_settings['models']['weight']:
        raise ValueError('Use exactly the source Weight CNN architecture in both arms')
    if settings['discretization'] != teacher_settings['data']['discretization']:
        raise ValueError('Teacher comparison and CNN study discretizations differ')
    if settings['discretization'] != source_settings['discretization']:
        raise ValueError('Source and new CNN discretizations differ')
    if not settings['smoke'] and teacher_settings.get('smoke'):
        raise ValueError('A smoke teacher comparison cannot select a research teacher')
    if settings.get('support_mode') != 'fixed_bilinear' or settings['finetune']['cycle_loss_weighting'] != 'uniform':
        raise ValueError('Paired study uses fixed bilinear support and uniform residual loss')
    settings.update(mg=asdict(cfg), teacher=asdict(tc), k=tc.k, l1_cap=tc.l1_cap,
                    teacher_optimizer=deepcopy(teacher_settings['optimizer']))
    if settings['benchmark']['rhs_counts'] != [1]:
        raise ValueError('Paired comparison declares exactly one RHS per A')
    # Reuse the established solver/training protocol validation.
    validate_settings({**settings, 'version': OLD_VERSION})
    holdout_plan = _holdout_plan(settings)
    if holdout_plan is not None:
        from .bamg_energy_cnn_holdout_benchmark import _settings as holdout_settings
        holdout_settings(settings)
        if not settings['smoke']:
            seen_sizes = set()
            for data in (settings['data'], source_settings['data']):
                for split in ('train', 'validation'):
                    seen_sizes.update(data[split]['sizes'])
            if set(holdout_plan['splits']['ood_grid']['sizes']) & seen_sizes:
                raise ValueError('Grid OOD must be unseen by both new and warm-source development data')
    return settings


def _manifest(settings, source, teacher):
    sources = _sources()
    path = PROJECT / 'scripts/run_v6_7_bamg_energy_cnn_study.py'
    sources[str(path.relative_to(PROJECT))] = _file_hash(path)
    environment = _environment()
    # Loaded-library enumeration order can differ between fresh Python processes.
    environment['threadpools'] = sorted(environment['threadpools'],
                                       key=lambda pool: json.dumps(pool, sort_keys=True))
    return dict(version=VERSION, settings_digest=numerical.digest(settings),
                sources=sources, environment=environment, source=source,
                teacher=teacher, final_or_ood_evaluated=False)


def _copy_verified(source, target, expected):
    if target.exists():
        if _file_hash(target) != expected:
            raise ValueError('Immutable input copy changed: ' + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    shutil.copyfile(source, temporary)
    if _file_hash(temporary) != expected:
        raise ValueError('Input changed during copying: ' + str(source))
    temporary.replace(target)


def _prepare(out, settings, source, teacher, cfg):
    inventory_path = out / 'prepared_manifest.json'
    if inventory_path.exists():
        for relative, expected in _read(inventory_path).items():
            path = out / relative
            if not path.is_file() or _file_hash(path) != expected:
                raise ValueError('Prepared input changed or disappeared: ' + relative)
    checkpoint = out / 'inputs/initial_weight.pt'
    _copy_verified(Path(source['path']) / source['checkpoint'], checkpoint,
                   source['init_weight_sha256'])
    baseline = BilinearClassicalBaseline.from_config(SimpleNamespace(mg=cfg))
    holdout = _holdout_plan(settings)
    if holdout is not None:
        path = out / 'holdout_plan.json'
        if path.exists() and _read(path) != holdout:
            raise ValueError('Preregistered final/OOD plan changed')
        _write(path, holdout)
    history_path = out / 'inputs/exclusion_manifest.json'
    if not history_path.exists():
        roots = list(dict.fromkeys([source['path'], str(out.parent)] + settings.get('historical_roots', [])))
        history = historical_operator_index(roots, exclude=[out])
        inherited = set()
        for name in ('development_manifest.json', 'final_data_manifest.json'):
            path = Path(source['path']) / 'data' / name
            if path.exists():
                inherited.update(_read(path).get('historical_index', {}).get('normalized_operator_digests', []))
        history['normalized_operator_digests'] = sorted(set(history['normalized_operator_digests']) |
                                                       set(teacher['normalized_operator_digests']) | inherited)
        history['count'] = len(history['normalized_operator_digests'])
        # Historical scanner recognizes normalized_operator_digest inside records.
        history['operators'] = [dict(normalized_operator_digest=h)
                                for h in history['normalized_operator_digests']]
        history['teacher_comparison'] = teacher
        _write(history_path, history)
    if not (out / 'data/development_manifest.json').exists():
        plan = make_research_plan(smoke=settings['smoke'], seed=settings['data']['seed'],
                                  discretization=settings['discretization'])
        for split in ('train', 'validation'):
            spec = plan['splits'][split]
            spec.update(settings['data'][split])
            spec['families'] = settings['data'].get('families', list(FAMILIES))
            spec['count'] = len(spec['sizes']) * len(spec['families']) * spec['per_family']
        if holdout is not None:
            for name, holdout_spec in holdout['splits'].items():
                plan['splits'][name].update({k: holdout_spec[k] for k in ('sizes', 'families', 'per_family', 'count', 'seed')})
            plan['holdout_materializer'] = dict(version=holdout['version'], plan='holdout_plan.json',
                note='The dedicated frozen holdout generator is authoritative; legacy final generator is unused.')
        # Publish the pair of metadata files together; interrupted generation is redone.
        with tempfile.TemporaryDirectory(prefix='.prepare-data-', dir=out) as directory:
            prepared = Path(directory) / 'data'
            prepare_development_data(prepared, baseline, plan=plan, historical_roots=[history_path])
            prepared.rename(out / 'data')
        _write(out / 'fixed_classical.json', baseline.to_dict())
    train, validation, _ = _examples(out, baseline)
    for split, examples in [('train', train), ('validation', validation)]:
        for example in examples:
            example['split'] = split
    files = [checkpoint, history_path, out / 'data/dataset_plan.json', out / 'data/development_manifest.json']
    if holdout is not None:
        files.append(out / 'holdout_plan.json')
    inventory = {str(p.relative_to(out)): _file_hash(p) for p in files}
    path = out / 'prepared_manifest.json'
    if path.exists() and _read(path) != inventory:
        raise ValueError('Prepared data/initialization changed; use a new output directory')
    _write(path, inventory)
    return train, validation, checkpoint, inventory


def _labels(out, examples, cfg, tc, optimizer, strength, identity):
    directory = out / 'teacher/train'
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / 'manifest.json'
    index = _read(index_path) if index_path.exists() else dict(identity=identity, records={})
    if index['identity'] != identity:
        raise ValueError('Shared TRAIN teacher identity changed')
    results = []
    for number, example in enumerate(examples, 1):
        key = example['case_id']
        path = directory / (key + '.pt')
        if key in index['records']:
            if _file_hash(path) != index['records'][key]:
                raise ValueError('Shared teacher cache changed: ' + key)
            package = _load_pt(path)
        elif path.exists():
            # Recovery after atomic checkpoint save but before index publication.
            package = _load_pt(path)
        else:
            print(f'TRAIN teacher {number}/{len(examples)} lambda={strength}: {example["name"]}', flush=True)
            result = build_energy_teacher(example['A'], example['shape'], config=tc,
                        mg_config=cfg, strength=strength, optimizer=optimizer)
            package = dict(identity=identity, case_id=key, result=result)
            _save(path, package)
        if package['identity'] != identity or package['case_id'] != key:
            raise ValueError('Teacher checkpoint identity differs')
        index['records'][key] = _file_hash(path)
        _write(index_path, index)
        results.append(package['result'])
    return results, _file_hash(index_path)


def seed_summary(benchmark_summary, seeds):
    """Aggregate prespecified seeds; do not present the fastest seed as an arm."""
    groups, pairs = {}, {}
    for stage in ('supervised', 'finetuned'):
        for arm in ('scratch', 'warm'):
            values = [benchmark_summary['methods'][f'{arm}_seed{seed}_{stage}'] for seed in seeds]
            times = [v['warm_seconds'] for v in values]
            groups[f'{arm}_{stage}'] = dict(
                seeds=seeds, mean_warm_seconds=float(np.mean(times)),
                std_warm_seconds=float(np.std(times, ddof=1)) if len(times) > 1 else None,
                failures=sum(v['failures'] for v in values),
                attempts=sum(v['attempts'] for v in values),
                eligible=all(v['eligible'] for v in values),
                uncertainty='Across training seeds, not a confidence interval.')
        pairs[stage] = [dict(seed=seed,
            scratch_warm_seconds=benchmark_summary['methods'][f'scratch_seed{seed}_{stage}']['warm_seconds'],
            warm_start_seconds=benchmark_summary['methods'][f'warm_seed{seed}_{stage}']['warm_seconds'])
            for seed in seeds]
    return dict(groups=groups, paired_seeds=pairs,
                policy='All prespecified seeds are retained, including failures. No fastest-seed selection.')


def run(config, output, *, source_study, teacher_comparison, resume=False, stop_after=None):
    requested = _read(config)
    source, source_settings = source_inputs(source_study)
    teacher, teacher_settings, cfg, tc = teacher_inputs(teacher_comparison)
    settings = resolve_settings(requested, source_settings, teacher_settings, cfg, tc)
    _initialize(settings)
    identity = _manifest(settings, source, teacher)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.cnn-comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'research_freeze.json').exists() or (out / 'final/claim.json').exists():
            raise ValueError('Study is frozen; training and validation selection are closed')
        path = out / 'study_manifest.json'
        if path.exists():
            if not resume:
                raise FileExistsError('Study exists; use --resume')
            if _read(path) != identity:
                raise ValueError('Source/configuration/inputs/environment changed; use a new study directory')
        else:
            if resume or any(p.name != '.cnn-comparison.lock' for p in out.iterdir()):
                raise FileExistsError('New study requires an empty output directory')
            _write(path, identity)
        _write(out / 'configuration.json', settings)
        train, validation, init_weight, inventory = _prepare(out, settings, source, teacher, cfg)
        teacher_identity = dict(study=numerical.digest(identity), prepared=inventory,
                                lambda_value=teacher['selected_lambda'])
        results, label_hash = _labels(out, train, cfg, tc, settings['teacher_optimizer'],
                                     teacher['selected_lambda'], teacher_identity)
        provenance = dict(study=numerical.digest(identity), prepared=inventory,
                          teacher_manifest_sha256=label_hash,
                          init_weight_sha256=source['init_weight_sha256'])
        completions = {}
        checkpoints = {'old_cnn': init_weight}
        for seed in settings['model_seeds']:
            # Alternate arm order to reduce systematic temperature/time-of-day bias.
            arms = ('scratch', 'warm') if settings['model_seeds'].index(seed) % 2 == 0 else ('warm', 'scratch')
            for arm in arms:
                name = f'{arm}_seed{seed}'
                print(f'Train {name}: shared lambda={teacher["selected_lambda"]}', flush=True)
                result = train_arm(out / 'runs' / name, arm=arm, seed=seed, settings=settings,
                            train=train, validation=validation, teacher_results=results,
                            init_weight=init_weight, provenance=provenance,
                            resume=resume, stop_after=stop_after)
                completions[name] = result
                if result['status'] != 'complete':
                    _write(out / 'progress.json', dict(completed=False, runs=completions))
                    return dict(completed=False, runs=completions)
                checkpoints[name + '_supervised'] = out / 'runs' / name / 'supervised_models.pt'
                checkpoints[name + '_finetuned'] = out / 'runs' / name / 'deployment.pt'
        _write(out / 'progress.json', dict(completed=False, runs=completions))
        summary = benchmark(out / 'validation', examples=validation, settings=settings,
                            checkpoints=checkpoints, identity=provenance, resume=resume)
        report = dict(version=VERSION, completed=summary['completed'],
                      selected_lambda=teacher['selected_lambda'], source_model=source['selected_model'],
                      init_weight_sha256=source['init_weight_sha256'],
                      train_cases=len(train), validation_cases=len(validation),
                      model_seeds=settings['model_seeds'], runs=completions, benchmark=summary,
                      seed_summary=seed_summary(summary, settings['model_seeds']),
                      interpretation='Fixed training budgets; not proof of convergence or final/OOD performance.')
        _write(out / 'comparison.json', report)
        print_report(report)
        return report


def print_report(result):
    print(f'\nPaired CNN development validation; lambda={result["selected_lambda"]}, complete={result["completed"]}')
    print(f'{"Method":40s} {"Warm ms*":>12s} {"Fail/total":>14s} {"Speedup":>9s}')
    for name, value in result['benchmark']['methods'].items():
        speed = value.get('speedup')
        formatted = '--' if speed is None else f'{speed:.3f}'
        print(f'{name:40s} {1000 * value["warm_seconds"]:12.3f} '
              f'{value["failures"]:5d}/{value["attempts"]:<8d} {formatted:>9s}')
    print('* Penalized failures/missing solves; setup excluded; no fallback.')
    for name, value in result['seed_summary']['groups'].items():
        std = value['std_warm_seconds']
        spread = '' if std is None else f' +/- {1000 * std:.3f}'
        print(f'Seed mean {name}: {1000 * value["mean_warm_seconds"]:.3f}{spread} ms, '
              f'failures={value["failures"]}/{value["attempts"]}')
    boundaries = [name for name, run in result['runs'].items()
                  if run.get('selection', {}).get('best_at_budget_boundary')]
    if boundaries:
        print('Best checkpoint at the training limit (consider a longer fresh study):', ', '.join(boundaries))
    print('Fixed budgets only: inspect training histories and best-update positions before judging convergence.')
    print('Validation selection only. Use freeze, then evaluate-final for the preregistered holdout.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'report', 'freeze', 'evaluate-final', 'report-final'))
    parser.add_argument('--config')
    parser.add_argument('--source-study', help='Migrated original study containing its selected deployment and data metadata')
    parser.add_argument('--teacher-comparison', help='Complete numerical comparison directory; selects its eligible lambda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Testing: cap new updates in the next unfinished training phase')
    parser.add_argument('--max-cases', type=int, help='evaluate-final: limit NEW completed operator cases before pausing')
    parser.add_argument('--group', help='report-final: split/grid/family or crossed subgroup key')
    args = parser.parse_args(argv)
    if args.command in ('freeze', 'evaluate-final', 'report-final'):
        from .bamg_energy_cnn_final import freeze, evaluate_final, report_final
        if args.command == 'freeze':
            freeze(args.output)
        elif args.command == 'evaluate-final':
            result = evaluate_final(args.output, resume=args.resume, max_cases=args.max_cases)
            print(f'Final/OOD completed: {result["completed"]}; use report-final to inspect groups.')
        else:
            report_final(args.output, group=args.group)
        return
    if args.command == 'report':
        path = Path(args.output) / 'comparison.json'
        if not path.exists():
            parser.error('Comparison is not complete yet; resume run to finish training and validation')
        print_report(_read(path))
    else:
        if not args.config or not args.source_study or not args.teacher_comparison:
            parser.error('run requires --config, --source-study and --teacher-comparison')
        run(args.config, args.output, source_study=args.source_study,
            teacher_comparison=args.teacher_comparison, resume=args.resume, stop_after=args.stop_after)


if __name__ == '__main__':
    main()
