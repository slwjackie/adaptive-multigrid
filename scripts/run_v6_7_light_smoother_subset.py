#!/usr/bin/env python3
"""Fit valid training oracle labels; retain failures and the full evaluation cohort.

This explicitly different supervision policy uses a new run. Original search,
retry, EM-line and P-only sources/manifests are never modified. Shared numerical
fit and evaluation implementations remain the original ones.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))

import numpy as np

from adaptive_mg.provenance import json_safe, operator_digest
from adaptive_mg.v67 import light_smoother_study as study
from adaptive_mg.v67 import light_smoother_search as oracle
from adaptive_mg.v67.light_smoother_fit import fit_models, load_fitted, _manifest as fit_manifest
from adaptive_mg.v67.light_smoother_models import Standardizer
from adaptive_mg.v67.limited import digest_file
from adaptive_mg.v67.research_data import _hash, _write_json as write_json

VERSION = 'light-smoother-valid-labels-v1'
POLICY = 'exclude_failed_training_labels_keep_all_evaluation'
read = study._read


def sources():
    return study._sources()+[Path(__file__).resolve()]


def signed_write(path, value):
    value = {k: v for k, v in value.items() if k != 'digest'}
    write_json(path, dict(value, digest=_hash(value)))


def signed_read(path):
    value = read(path)
    if value.get('digest') != _hash({k: v for k, v in value.items() if k != 'digest'}):
        raise ValueError(f'Subset manifest/ledger changed: {path}')
    return value


def safe_file(root, relative):
    path = root/relative
    if (Path(relative).is_absolute() or not path.resolve().is_relative_to(root)
            or path.resolve() != path.absolute() or not path.is_file()):
        raise ValueError(f'Expected a regular file inside run: {relative}')
    return path


def verify_search_record(out, row, example, cfg, settings, p_signature):
    """Check complete evidence even for excluded cases; exclusion is not an escape hatch."""
    path = safe_file(out, row['path'])
    allowed = (f'search/{example.name}/result.json', f'retry/cases/{example.name}/result.json')
    if (row['path'] not in allowed or row['case'] != example.name
            or row['normalized_operator_digest'] != example.group_digest or digest_file(path) != row['sha256']):
        raise ValueError('Search case/operator/hash mismatch')
    state = signed_read(path.with_name('progress.json'))
    record = read(path)
    manifest = record['manifest']
    if record['version'] not in (oracle.VERSION, 'light-smoother-joint-retry-v1'):
        raise ValueError('Unsupported search label provenance')
    if (state['status'] != 'complete' or state.get('result_digest') != _hash(record)
            or state['manifest'] != manifest or record['status'] != 'complete'
            or record['fallback_allowed'] or record['online_adaptation_allowed']
            or manifest['example'] != example.name or manifest['split'] != 'train'
            or manifest['operator_digest'] != operator_digest(example.a)
            or manifest['normalized_operator_digest'] != example.group_digest
            or manifest['transfer_signature'] != p_signature
            or manifest['solver'] != json_safe(cfg.to_dict())):
        raise ValueError('Search evidence is incomplete or differs from fixed P/solver/train data')
    spec = oracle.validate_search_settings(dict(settings['search'], bounds=settings['bounds']))
    key = 'original_search' if record['version'] == 'light-smoother-joint-retry-v1' else 'settings'
    if manifest[key] != spec:
        raise ValueError('Search numerical settings changed')
    selection = record['confirmation']
    selected = selection['selected_index']
    if type(selected) is not int or not 0 <= selected < len(selection['summaries']):
        raise ValueError('Invalid selected search candidate')
    flags = [row.get('label_eligible'), record.get('label_eligible'),
             record.get('selected_candidate_success'), selection['summaries'][selected].get('eligible')]
    if any(type(v) is not bool for v in flags) or len(set(flags)) != 1:
        raise ValueError('Inconsistent label eligibility; do not relabel a failed baseline')
    if record['trials'] != state['trials']:
        raise ValueError('Search result and measurement ledger differ')
    if any(t.get('fallback_used') is not False or t.get('online_adaptation_used') is not False
           for t in record['trials']):
        raise ValueError('Oracle evidence contains fallback or adaptation')
    if flags[0]:
        coefficient_key = oracle._coefficient_key(record['selected_coefficients'])
        if coefficient_key != oracle._coefficient_key(selection['candidates'][selected]):
            raise ValueError('Selected coefficients differ from confirmed candidate')
        trials = [t for t in record['trials'] if t.get('surrogate_only') is not True
                  and oracle._coefficient_key(t['coefficients']) == coefficient_key]
        for partition, rhs_count in [('optimization', spec['rhs_count']),
                                     ('confirmation', spec['confirmation_rhs_count'])]:
            phases = {t['phase'] for t in trials if t['partition'] == partition}
            if not phases:
                raise ValueError('Eligible label lacks full optimization/confirmation evidence')
            for phase in phases:
                score = oracle.candidate_summary([t for t in trials if t['partition'] == partition and t['phase'] == phase],
                                                  repeats=spec['repeats'], rhs_count=rhs_count)
                if not score['eligible']:
                    raise ValueError('Failed candidate cannot become a training label')
    return record


def inspect_labels(out, settings, cfg, baseline):
    training, validation, _ = study._data(out, settings, baseline)
    summary = read(out/'search/summary.json')
    if summary.get('status') != 'complete' or len(summary['rows']) != len(training):
        raise ValueError('Complete the source search/retry first; incomplete cases cannot be excluded')
    imported = read(out/'import_manifest.json')
    included, excluded, xs, ys = [], [], [], []
    family_counts, family_grid_counts = {}, {}
    for example, row in zip(training, summary['rows']):
        if example.research_split != 'train':
            raise ValueError('Supervision must be training-only')
        record = verify_search_record(out, row, example, cfg, settings, imported['transfer_signature'])
        eligible = record['label_eligible']
        family = example.case.family
        for mapping, key in [(family_counts, family), (family_grid_counts, f'{family}/n{example.n}')]:
            counts = mapping.setdefault(key, dict(total=0, included=0, excluded=0))
            counts['total'] += 1
            counts['included' if eligible else 'excluded'] += 1
        entry = dict(case=example.name, normalized_operator_digest=example.group_digest,
                     family=family, n=example.n, result_path=row['path'], result_sha256=row['sha256'],
                     offline_search_seconds=row['offline_seconds'])
        if eligible:
            features, coefficients = np.asarray(record['features']), np.asarray(record['selected_coefficients'])
            if (features.ndim != 2 or features.shape[1] != 12 or not len(features)
                    or coefficients.shape != (len(features), 2)
                    or not np.isfinite(features).all() or not np.isfinite(coefficients).all()
                    or (coefficients < settings['bounds'][0]).any() or (coefficients > settings['bounds'][1]).any()):
                raise ValueError('Invalid eligible feature/target rows')
            xs.extend(features)
            ys.extend(coefficients)
            included.append(dict(entry, level_rows=len(features)))
        else:
            excluded.append(dict(entry, reason='no_successful_search_label',
                selection_outcome=record['selection_outcome'],
                stop_reasons=dict(Counter(t['stop_reason'] for t in record['trials'])),
                trial_count=len(record['trials']),
                residual_histories_preserved_in=row['path']))
    if summary.get('failed_cases') != [r['case'] for r in excluded]:
        raise ValueError('Search summary failure list differs from verified records')
    if summary.get('total_cases') != len(training) or summary.get('eligible_cases') != len(included):
        raise ValueError('Search summary counts differ from verified records')
    if not included:
        raise ValueError('No valid training labels remain; cannot fit either model')
    plan = read(out/'data/dataset_plan.json')
    report = dict(version=VERSION, policy=POLICY, selection_split='train',
        total_training_operators=len(training), included_operators=len(included), excluded_operators=len(excluded),
        included_level_rows=len(xs), included=included, excluded=excluded,
        by_family=family_counts, by_family_grid=family_grid_counts,
        normalization='Only included training level rows; identical scaler for MLP and ridge',
        p_training_operators=len(imported['training_operator_digests']),
        p_training_disjoint=True, p_checkpoint_unchanged=True,
        evaluation_policy='No operator filtering; retain failures and fallback costs',
        validation_operators=len(validation), holdout_counts={s: plan['splits'][s]['count'] for s in study.HOLDOUT},
        search_summary_sha256=digest_file(out/'search/summary.json'),
        offline_search_seconds=summary['offline_seconds'],
        source_search_failure_fraction=len(excluded)/len(training))
    return np.asarray(xs), np.asarray(ys), report


def source_inventory(source):
    names = {'configuration.json', 'fixed_classical.json', 'light_manifest.json', 'import_manifest.json',
             'data/dataset_plan.json', 'data/development_manifest.json'}
    names.update(r['path'] for r in read(source/'import_manifest.json')['files'].values())
    for folder in ('search', 'retry'):
        names.update(str(p.relative_to(source)) for p in (source/folder).rglob('*.json'))
    return {name: digest_file(safe_file(source, name)) for name in sorted(names)}


def verify_snapshot(out, manifest):
    for name, sha in manifest['source_files'].items():
        if digest_file(safe_file(out, name)) != sha:
            raise ValueError(f'Preserved source evidence changed: {name}')


def verify_retry_source(source):
    """Accept completed retry artifacts without re-running candidates or changing them."""
    if not (source/'retry/manifest.json').exists():
        return
    manifest = signed_read(source/'retry/manifest.json')
    status = signed_read(source/'retry/status.json')
    summary = read(source/'search/summary.json')
    if (status['status'] != 'complete' or status['summary_sha256'] != digest_file(source/'search/summary.json')
            or summary.get('retry_manifest') != manifest):
        raise ValueError('Retry must be complete and unchanged before importing its labels')
    for entry in manifest['source_files'].values():
        if digest_file(safe_file(source, entry['path'])) != entry['sha256']:
            raise ValueError('Retry parent evidence changed')


def prepare(output, *, source_run=None, resume=False):
    out = Path(output).resolve()
    manifest_path = out/'subset/manifest.json'
    if manifest_path.exists():
        if not resume:
            raise FileExistsError('Subset run exists; use --resume')
    else:
        if resume or source_run is None:
            raise ValueError('New subset run requires --source-run and a separate empty --run-dir')
        source = Path(source_run).resolve()
        if out == source or out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError('Use a separate sibling run; never modify the source run')
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Subset preparation requires an empty directory')
        with study._lease(source):
            _, settings, cfg, baseline = study.load(source)
            study._development_mutable(source)
            if (source/'subset/manifest.json').exists():
                raise ValueError('Use the original completed search or completed retry as the parent')
            verify_retry_source(source)
            _, _, selection = inspect_labels(source, settings, cfg, baseline)
            manifest = dict(version=VERSION, policy=POLICY, source_run=str(source),
                runner_sha256=digest_file(Path(__file__)), source_files=source_inventory(source),
                training_selection_digest=_hash(selection), evaluation_filtering=False)
            out.mkdir(parents=True, exist_ok=True)
            signed_write(manifest_path, manifest)
    with study._lease(out):
        manifest = signed_read(manifest_path)
        if (manifest['version'] != VERSION or manifest['policy'] != POLICY
                or manifest['runner_sha256'] != digest_file(Path(__file__))):
            raise ValueError('Subset execution source/policy changed')
        if source_run is not None and str(Path(source_run).resolve()) != manifest['source_run']:
            raise ValueError('Subset parent changed')
        marker = out/'subset/prepared.json'
        if not marker.exists():
            source = Path(manifest['source_run'])
            with study._lease(source):
                study._development_mutable(source)
                for name, sha in manifest['source_files'].items():
                    path = safe_file(source, name)
                    if digest_file(path) != sha:
                        raise ValueError('Source changed during import; use a new run')
                    target = out/name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
            verify_snapshot(out, manifest)
            _, settings, cfg, baseline = study.load(out)
            _, _, selection = inspect_labels(out, settings, cfg, baseline)
            if _hash(selection) != manifest['training_selection_digest']:
                raise ValueError('Training selection changed during snapshot')
            write_json(out/'subset/training_selection.json', selection)
            write_json(marker, dict(manifest_digest=manifest['digest'],
                                   selection_sha256=digest_file(out/'subset/training_selection.json')))
        _, _, _, selection = load(out)
        study._development_mutable(out)
    print(f'Valid-label study: train {selection["included_operators"]}/{selection["total_training_operators"]}; '
          f'excluded {selection["excluded_operators"]}; validation ALL {selection["validation_operators"]}. '
          'All search failures retained. No search or P training repeated.', flush=True)
    return selection


def load(output):
    out, settings, cfg, baseline = study.load(output)
    manifest = signed_read(out/'subset/manifest.json')
    if (manifest['version'] != VERSION or manifest['policy'] != POLICY
            or manifest['runner_sha256'] != digest_file(Path(__file__)) or manifest['evaluation_filtering']):
        raise ValueError('Subset source/policy changed')
    verify_snapshot(out, manifest)
    marker = read(out/'subset/prepared.json')
    selection = read(out/'subset/training_selection.json')
    if (marker != dict(manifest_digest=manifest['digest'], selection_sha256=digest_file(out/'subset/training_selection.json'))
            or _hash(selection) != manifest['training_selection_digest']):
        raise ValueError('Training exclusion list/report changed')
    return out, settings, cfg, selection


def training_inputs(out, settings, cfg, selection):
    baseline = study.load_baseline(out/'fixed_classical.json')
    x, y, current = inspect_labels(out, settings, cfg, baseline)
    if current != selection:
        raise ValueError('Frozen supervision selection changed')
    provenance = dict(split='train', policy=POLICY,
        operators=[r['normalized_operator_digest'] for r in selection['included']],
        excluded_operators=[r['normalized_operator_digest'] for r in selection['excluded']],
        training_selection_sha256=digest_file(out/'subset/training_selection.json'),
        subset_manifest_sha256=digest_file(out/'subset/manifest.json'),
        source_manifest=_hash(read(out/'light_manifest.json')),
        search_summary_sha256=digest_file(out/'search/summary.json'),
        transfer_signature=read(out/'import_manifest.json')['transfer_signature'])
    return x, y, provenance


def fit(output, *, resume=False):
    out, settings, cfg, selection = load(output)
    with study._lease(out):
        study._development_mutable(out)
        x, y, provenance = training_inputs(out, settings, cfg, selection)
        result = fit_models(x, y, out/'fit', dict(settings['fit'], bounds=settings['bounds']), provenance, resume=resume)
    print(f'Fitted MLP (242) and ridge (26): same {selection["included_operators"]} A / {len(x)} level rows; '
          f'{selection["excluded_operators"]} failed A excluded only from supervision and scaler.', flush=True)
    return result


def arms(out, settings, cfg, baseline, selection, *, include_online=False):
    x, y, provenance = training_inputs(out, settings, cfg, selection)
    spec = study.validate_fit_settings(dict(settings['fit'], bounds=settings['bounds']))
    predictors = load_fitted(out/'fit', fit_manifest(x, y, spec, provenance, Standardizer.fit(x)))
    p = study._p_model(out)
    emcfg = replace(cfg, mg=replace(cfg.mg, strategy_name=study.FIXED_STRATEGY_NAME,
        em_iterations=5, em_fixed_iterations=True, em_rtol=1e-8))
    result = {'fixed_C': dict(model=None, branch='C', selector=False),
              'em5_C': dict(model=None, branch='C', selector=False, reference_only=True,
                           config=emcfg, rules=study.FixedClassicalBaseline.from_config(emcfg))}
    for name, key in zip(study.BASE_ARMS, (None, 'mlp', 'regression')):
        result[name] = dict(model=p, branch='H_P', selector=False, light=True,
            predictor=predictors[key] if key else None, online=None, bounds=settings['bounds'])
    if include_online:
        for name, key, method in zip(study.ONLINE_ARMS, ('mlp', 'regression'), ('mlp_ttt', 'regression_online')):
            result[name] = dict(model=p, branch='H_P', selector=False, light=True,
                predictor=predictors[key], bounds=settings['bounds'], online=dict(settings['online'], method=method, enabled=True))
    if read(out/'import_manifest.json')['frozen_line']['included']:
        frozen = study.Components.load(out/'imported/em_line/checkpoints/frozen_P_line/candidate.pt')
        if frozen.expert_signature('transfer') != p.expert_signature('transfer'):
            raise ValueError('CNN-line P differs from common P')
        result['frozen_P_line'] = dict(model=frozen, branch='H_SP', selector=False,
            config=study.AdaptiveConfig.from_dict(read(out/'imported/em_line/configuration.json')['solver']))
    return result


def artifacts(out):
    result = study._artifacts(out)
    extra = ['subset/manifest.json', 'subset/prepared.json', 'subset/training_selection.json']
    extra += list(signed_read(out/'subset/manifest.json')['source_files'])
    for name in extra:
        result[name] = dict(path=str(out/name), sha256=digest_file(out/name))
    return result


def report(result, target, selection):
    study.report(result, target)
    write_json(target/'training_policy.json', selection)
    text = (f'# Training-label policy\n\n'
        f'{selection["total_training_operators"]}개 학습 A 중 유효한 탐색 정답을 얻은 '
        f'{selection["included_operators"]}개로 MLP와 ridge를 학습했습니다. '
        f'{selection["excluded_operators"]}개는 정답 생성 실패로 지도학습 및 표준화에서 제외했습니다.\n\n'
        '실패 목록·잔차·시간은 원본 기록에 보존했습니다. Validation/final/OOD는 제외 없이 평가하며, '
        '평가 실패와 fallback 비용도 기존 지표에 포함합니다.\n\n'
        '| Family | 전체 | 학습 포함 | 제외 |\n| --- | ---: | ---: | ---: |\n')
    for family, counts in selection['by_family'].items():
        text += f'| {family} | {counts["total"]} | {counts["included"]} | {counts["excluded"]} |\n'
    (target/'training_policy.md').write_text(text, encoding='utf-8')


def benchmark(output, *, resume=False, include_online=False, **protocol):
    protocol = study._validate_protocol(protocol)
    out, settings, cfg, selection = load(output)
    baseline = study.load_baseline(out/'fixed_classical.json')
    with study._lease(out):
        study._ensure_development_open(out)
        if 'multiple' not in protocol['regimes'] or settings['target_rhs_count'] not in protocol['rhs_counts']:
            raise ValueError('Include setup-inclusive multiple and target_rhs_count')
        _, examples, _ = study._data(out, settings, baseline)  # Never apply the training mask here.
        models = arms(out, settings, cfg, baseline, selection, include_online=include_online)
        target = out/'benchmarks/validation'
        result = study.evaluate_light(examples, models, cfg, baseline, target, resume=resume, **protocol)
        report(result, target, selection)
        candidates = tuple(name for name in models if name not in ('fixed_C', 'em5_C'))
        decision = study.select_light_candidate(result, rhs_count=settings['target_rhs_count'], candidate_arms=candidates)
        write_json(target/'selection.json', decision)
        files = [target/name for name in ('raw_results.json', 'cycle_diagnostics.json', 'selection.json',
                                         'paired_effects.json', 'training_policy.json', 'training_policy.md')]
        evidence = dict(version=VERSION, artifacts=artifacts(out),
            files={str(p.relative_to(out)): dict(path=str(p), sha256=digest_file(p)) for p in files},
            include_online=include_online, selection=decision, performance_certified=False)
        write_json(out/'benchmark_evidence.json', evidence)
        write_json(out/'selected/deployment.json', study._deployment(out, settings, decision, include_online))
    print(f'Validation ALL {len(examples)} operators. Recommendation: {decision["recommended_deployment_arm"]}.', flush=True)
    return result


def freeze(output, **protocol):
    protocol = study._validate_protocol(protocol, final=True)
    out, settings, cfg, selection = load(output)
    baseline = study.load_baseline(out/'fixed_classical.json')
    with study._lease(out):
        study._ensure_development_open(out)
        evidence = read(out/'benchmark_evidence.json')
        if evidence['version'] != VERSION or evidence['artifacts'] != artifacts(out):
            raise ValueError('Subset training/search evidence changed after validation')
        for item in evidence['files'].values():
            if digest_file(item['path']) != item['sha256']:
                raise ValueError('Validation evidence changed')
        if 'multiple' not in protocol['regimes'] or settings['target_rhs_count'] not in protocol['rhs_counts']:
            raise ValueError('Final must include setup-inclusive target workload')
        models = arms(out, settings, cfg, baseline, selection, include_online=evidence['include_online'])
        selected = read(out/'selected/deployment.json')
        if selected != study._deployment(out, settings, evidence['selection'], evidence['include_online']):
            raise ValueError('Deployment differs from validation selection')
        files = {k: r['path'] for k, r in {**evidence['artifacts'], **evidence['files']}.items()}
        files.update(evidence=out/'benchmark_evidence.json', deployment=out/'selected/deployment.json')
        evaluation = dict(version=VERSION, policy=POLICY, solver=cfg.to_dict(), arms=list(models),
            training_selection_sha256=digest_file(out/'subset/training_selection.json'),
            include_online=evidence['include_online'], selected_arm=selected['arm'],
            target_rhs_count=settings['target_rhs_count'], **protocol)
        return study.freeze_research(out/'data', dict(common_P=out/'imported/em_line/checkpoints/em_P/candidate.pt',
            mlp=out/'fit/mlp.json', regression=out/'fit/regression.json'),
            out/'fixed_classical.json', files, sources(), evaluation)


def final(output, *, resume=False):
    out, settings, cfg, selection = load(output)
    baseline = study.load_baseline(out/'fixed_classical.json')
    path = out/'data/research_freeze.json'
    evaluation = read(path)['config']
    models = arms(out, settings, cfg, baseline, selection, include_online=evaluation['include_online'])
    if (evaluation['version'] != VERSION or evaluation['policy'] != POLICY
            or evaluation['solver'] != json_safe(cfg.to_dict()) or evaluation['arms'] != list(models)
            or evaluation['training_selection_sha256'] != digest_file(out/'subset/training_selection.json')
            or evaluation['target_rhs_count'] != settings['target_rhs_count']):
        raise ValueError('Frozen subset policy/protocol changed')
    with study.claim_final_evaluation(out/'data', path, evaluation, resume=resume) as claim:
        splits, _ = study.materialize_final_data(out/'data', claim, baseline)
        outputs = []
        for name in evaluation['splits']:
            target = out/'final'/name
            result = study.evaluate_light(splits[name], models, cfg, baseline, target,
                resume=resume and (target/'run_manifest.json').exists(),
                **{key: evaluation[key] for key in ('repeats', 'warmups', 'rhs_counts', 'regimes')})
            report(result, target, selection)
            outputs.extend(target/key for key in ('raw_results.json', 'cycle_diagnostics.json',
                           'paired_effects.json', 'training_policy.json', 'training_policy.md'))
        study.complete_final_evaluation(out/'data', claim, outputs)
    print('Final/OOD full cohorts sealed; no exclusions or final-set reselection.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'fit', 'benchmark', 'freeze', 'final'):
        command = commands.add_parser(name)
        command.add_argument('--run-dir', required=True)
        if name != 'freeze':
            command.add_argument('--resume', action='store_true')
        if name == 'prepare':
            command.add_argument('--source-run', help='Completed original search or completed retry; never modified')
        if name == 'benchmark':
            command.add_argument('--include-online', action='store_true')
        if name in ('benchmark', 'freeze'):
            command.add_argument('--repeats', type=int, default=3)
            command.add_argument('--warmups', type=int, default=1)
            command.add_argument('--rhs-counts', nargs='+', type=int, default=[1, 4, 16, 64])
            command.add_argument('--regimes', nargs='+', choices=('cold', 'warm', 'multiple'), default=['cold', 'warm', 'multiple'])
        if name == 'freeze':
            command.add_argument('--splits', nargs='+', choices=study.HOLDOUT, default=list(study.HOLDOUT))
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        return prepare(args.run_dir, source_run=args.source_run, resume=args.resume)
    if args.command in ('fit', 'final'):
        return globals()[args.command](args.run_dir, resume=args.resume)
    protocol = {k: getattr(args, k) for k in ('repeats', 'warmups', 'rhs_counts', 'regimes')}
    if args.command == 'freeze':
        return freeze(args.run_dir, splits=args.splits, **protocol)
    return benchmark(args.run_dir, resume=args.resume, include_online=args.include_online, **protocol)


if __name__ == '__main__':
    main()
