"""Fixed imported EM-P, scalar line relaxation: MLP versus non-neural regression.

This is a separate experiment: no existing run's source or freeze manifest is
rewritten. Search uses training operators only; final data require a sealed
validation selection. Online adapters are episodic, not checkpoint mutations.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np

from ..provenance import hardware_environment, json_safe
from .config import AdaptiveConfig
from .direct_pruning_study import _ensure_development_open, _validate_protocol
from .em_line_study import _checkpoint as em_checkpoint, validate_settings as validate_em
from .fixed_classical import (BilinearClassicalBaseline, FixedClassicalBaseline,
                              FIXED_STRATEGY_NAME, load_baseline)
from .joint_transfer_study import _read, _initialize, _development, _lease
from .joint_transfer_evaluation import report
from .limited import digest_file
from .models import Components
from .research_data import (HOLDOUT, _hash, _write_json as write_json,
    freeze_research, claim_final_evaluation, materialize_final_data,
    complete_final_evaluation, _digests)
from .light_smoother_models import (FEATURE_VERSION, FEATURE_NAMES,
    FEATURE_DEFINITIONS, Standardizer)
from .light_smoother_search import search_case, validate_search_settings
from .light_smoother_fit import (fit_models, validate_fit_settings, load_fitted,
                               _manifest as fit_manifest)
from .light_smoother_runtime import validate_online
from .light_smoother_evaluation import evaluate_light, select_light_candidate

VERSION = 'fixed-p-light-line-v1'
PROJECT = Path(__file__).resolve().parents[3]
BASE_ARMS = ('learned_P_line', 'learned_P_MLP', 'learned_P_regression')
ONLINE_ARMS = ('learned_P_MLP_TTT', 'learned_P_regression_online')


def validate_settings(settings):
    if settings.get('version') != VERSION:
        raise ValueError('Require a separate light-smoother configuration')
    cfg = AdaptiveConfig.from_dict(settings['solver'])
    BilinearClassicalBaseline.from_config(cfg)
    if (cfg.mode != 'research' or cfg.branch != 'H_P' or not cfg.use_transfer
            or cfg.use_smoother or cfg.transfer_levels != 'all' or cfg.spatial
            or cfg.gate_mode != 'open' or cfg.use_learned_controller
            or cfg.mg.pre_steps != 2 or cfg.mg.post_steps != 2):
        raise ValueError('Require fixed all-level learned P and pre/post=2 line_alt')
    if type(settings['smoke']) is not bool or type(settings['include_frozen_line']) is not bool:
        raise ValueError('smoke/include_frozen_line must be boolean')
    for key in ('seed', 'torch_threads', 'training_per_family', 'validation_per_family', 'target_rhs_count'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('training_sizes', 'validation_sizes'):
        values = settings[key]
        if (not values or len(set(values)) != len(values)
                or any(type(n) is not int or n <= cfg.mg.coarsest_n or n & (n+1) for n in values)):
            raise ValueError('Require distinct nonterminal 2**L-1 grids')
    bounds = np.asarray(settings['bounds'], dtype=float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 < bounds[0] < 1 < bounds[1]):
        raise ValueError('Finite positive bounds must contain 1 in their interior')
    search = validate_search_settings(dict(settings['search'], bounds=settings['bounds']))
    fit = validate_fit_settings(dict(settings['fit'], bounds=settings['bounds']))
    for method in ('mlp_ttt', 'regression_online'):
        validate_online(dict(settings['online'], method=method, enabled=True))
    if list(search['bounds']) != list(fit['bounds']):
        raise ValueError('Search and both estimators must share bounds')
    return cfg


def _sources():
    return sorted([p for p in (PROJECT/'src').rglob('*') if p.suffix in ('.py', '.cpp')]
                  + [PROJECT/'scripts/run_v6_7_light_smoother_study.py'])


def _manifest(settings, baseline, imported):
    return dict(version=VERSION, settings_digest=_hash(settings),
        baseline_digest=baseline.digest(), import_digest=_hash(imported),
        feature_version=FEATURE_VERSION, features=list(FEATURE_NAMES),
        feature_definitions=FEATURE_DEFINITIONS,
        source_digest=_hash({str(p.relative_to(PROJECT)): digest_file(p) for p in _sources()}),
        hardware=hardware_environment(refresh=True))


def _import_em(out, source, settings, cfg, baseline):
    """Validate completed historical evidence, then snapshot without rewriting it.

    A historical source hash is recorded, not relabelled as current sources.
    The source run need not be resumable under this newer checkout to import a
    completed checkpoint. Stage lineage/weights/data are still verified.
    """
    source = Path(source).resolve()
    original = _read(source/'configuration.json')
    source_cfg = validate_em(original)
    source_baseline = load_baseline(source/'fixed_classical.json')
    if (source_baseline.digest() != baseline.digest()
            or source_cfg.mg.numerical_scope() != cfg.mg.numerical_scope()):
        raise ValueError('Imported EM-P must share the fixed numerical protocol/tolerance')
    if original['smoke'] and not settings['smoke']:
        raise ValueError('A smoke-trained P cannot be imported into the research study')
    source_manifest = _read(source/'em_line_manifest.json')
    if (source_manifest['settings_digest'] != _hash(original)
            or source_manifest['baseline_digest'] != baseline.digest()):
        raise ValueError('EM source settings/baseline provenance changed')
    model = em_checkpoint(source, 'em_P', source_baseline, settings=original)
    if not model.metadata.get('transfer_trained') or model.metadata.get('final_test_seen'):
        raise ValueError('Need a completed EM-P trained without final/OOD data')
    development = _read(source/'data/development_manifest.json')
    operators = [r['normalized_operator_digest'] for r in development['splits']['train']]
    if model.metadata.get('training_operator_digests') != operators:
        raise ValueError('P training operators differ from source data')
    paths = ['configuration.json', 'fixed_classical.json', 'em_line_manifest.json',
             'initial.pt', 'initial.json', 'data/development_manifest.json', 'data/dataset_plan.json']
    paths += [f'checkpoints/em_P/{name}' for name in
              ('candidate.pt', 'status.json', 'training_manifest.json', 'training.json')]
    # Record identities only: prior final/OOD operators are never reconstructed
    # or used as labels. Flatten historical lists because the shared scanner
    # recognizes individual normalized_operator_digest records.
    excluded = set(development.get('historical_index', {}).get('normalized_operator_digests', []))
    excluded.update(_digests(development['splits']))
    if (source/'data/final_data_manifest.json').exists():
        historical_final = _read(source/'data/final_data_manifest.json')
        excluded.update(_digests(historical_final))
        excluded.update(historical_final.get('historical_index', {}).get('normalized_operator_digests', []))
        paths.append('data/final_data_manifest.json')
    optional = dict(available=False, included=False,
        reason='Not requested; enable include_frozen_line in a fresh configuration')
    if settings['include_frozen_line']:
        folder = source/'checkpoints/frozen_P_line'
        if not (folder/'status.json').exists():
            optional['reason'] = 'Completed frozen_P_line checkpoint unavailable; finish its source EM-line training in its original checkout'
        elif _read(folder/'status.json').get('status') != 'complete':
            optional['reason'] = 'frozen_P_line stage incomplete; finish its source EM-line training in its original checkout'
        else:
            frozen = em_checkpoint(source, 'frozen_P_line', source_baseline, settings=original)
            if frozen.expert_signature('transfer') != model.expert_signature('transfer'):
                raise ValueError('frozen_P_line P differs; cannot make a matched-P smoother comparison')
            paths += [f'checkpoints/frozen_P_line/{name}' for name in
                      ('candidate.pt', 'status.json', 'training_manifest.json', 'training.json')]
            optional = dict(available=True, included=True, reason=None,
                            transfer_signature=frozen.expert_signature('transfer'))
    copied = {}
    for relative in paths:
        destination = out/'imported/em_line'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, destination)
        copied[relative] = dict(path=str(destination.relative_to(out)),
                               source_path=str(source/relative), sha256=digest_file(destination))
    exclusion_path = out/'imported/exclusion_manifest.json'
    write_json(exclusion_path, dict(scope='Historical operator identities: excluded from new train/validation/final',
        operators=[dict(normalized_operator_digest=digest) for digest in sorted(excluded)]))
    copied['exclusion_manifest.json'] = dict(path=str(exclusion_path.relative_to(out)),
        source_path=None, sha256=digest_file(exclusion_path))
    provenance = dict(version=VERSION, source_run=str(source), source_manifest=source_manifest,
        files=copied, transfer_signature=model.expert_signature('transfer'),
        components_signature=model.signature(), training_operator_digests=operators,
        source_data_plan_digest=development['plan_digest'], frozen_line=optional,
        import_policy='Completed stage evidence verified; historical source fingerprint preserved, not migrated',
        joint_line_included=False)
    write_json(out/'import_manifest.json', provenance)
    return provenance


def _verify_import(out):
    provenance = _read(out/'import_manifest.json')
    for record in provenance['files'].values():
        if digest_file(out/record['path']) != record['sha256']:
            raise ValueError(f'Imported checkpoint/provenance changed: {record["path"]}')
    source = out/'imported/em_line'
    baseline = load_baseline(source/'fixed_classical.json')
    settings = _read(source/'configuration.json')
    model = em_checkpoint(source, 'em_P', baseline, settings=settings)
    if model.signature() != provenance['components_signature']:
        raise ValueError('Imported P checkpoint signature changed')
    if provenance['frozen_line']['included']:
        frozen = em_checkpoint(source, 'frozen_P_line', baseline, settings=settings)
        if frozen.expert_signature('transfer') != model.expert_signature('transfer'):
            raise ValueError('Imported frozen-line P is not identical')
    return provenance


def load(output):
    out = Path(output).resolve()
    settings = _read(out/'configuration.json')
    cfg = validate_settings(settings)
    _initialize(settings)
    baseline = load_baseline(out/'fixed_classical.json')
    baseline.validate_config(cfg)
    imported = _verify_import(out)
    if _read(out/'light_manifest.json') != json_safe(_manifest(settings, baseline, imported)):
        raise ValueError('Light study sources/settings/import/environment changed; use a new run')
    return out, settings, cfg, baseline


def _data(out, settings, baseline):
    # Include the source snapshot in historical exclusion even when imported
    # from a different checkout. No imported validation/final A can be trained.
    spec = dict(settings, historical_roots=list(dict.fromkeys(
        settings.get('historical_roots', [])+[str(out/'imported')])))
    training, validation, manifest = _development(out, spec, baseline)
    imported = _read(out/'import_manifest.json')
    forbidden = set(imported['training_operator_digests'])
    if any(e.group_digest in forbidden for e in training+validation):
        raise ValueError('New study operators must be disjoint from imported P training data')
    return training, validation, manifest


def prepare(config, output, *, em_run_dir=None, resume=False):
    settings = _read(config)
    cfg = validate_settings(settings)
    _initialize(settings)
    baseline = BilinearClassicalBaseline.from_config(cfg)
    out = Path(output).resolve()
    if (out/'light_manifest.json').exists():
        if not resume:
            raise FileExistsError('Use identical --resume or a fresh light-study directory')
        _, previous, _, _ = load(out)
        if previous != settings:
            raise ValueError('Prepare settings changed')
        if em_run_dir is not None and str(Path(em_run_dir).resolve()) != _read(out/'import_manifest.json')['source_run']:
            raise ValueError('Prepare import source changed')
        _ensure_development_open(out)
    else:
        if resume:
            raise FileNotFoundError('No committed light study to resume')
        if em_run_dir is None:
            raise ValueError('prepare requires --em-run-dir with a completed em_P stage')
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Prepare requires an empty directory')
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'configuration.json', settings)
        write_json(out/'fixed_classical.json', baseline.to_dict())
        imported = _import_em(out, em_run_dir, settings, cfg, baseline)
        write_json(out/'light_manifest.json', _manifest(settings, baseline, imported))
    with _lease(out):
        training, validation, _ = _data(out, settings, baseline)
    optional = _read(out/'import_manifest.json')['frozen_line']
    print(f'Prepared light smoother: train={len(training)}, validation={len(validation)}; '
          'fixed imported EM-P; MLP=242 / ridge=26 coefficients; online adaptation default OFF', flush=True)
    print(f'frozen_P_line: {"included (identical P)" if optional["included"] else optional["reason"]}', flush=True)


def _p_model(out):
    return Components.load(out/'imported/em_line/checkpoints/em_P/candidate.pt')


def _development_mutable(out):
    _ensure_development_open(out)
    if (out/'benchmarks/validation/run_manifest.json').exists():
        raise ValueError('Validation started; search/fit are frozen. Use a new run for changes')


def search(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    with _lease(out):
        _development_mutable(out)
        training, _, _ = _data(out, settings, baseline)
        model = _p_model(out)
        spec = dict(settings['search'], bounds=settings['bounds'])
        rows = []
        for index, example in enumerate(training):
            folder = out/'search'/example.name
            # A run-level resume safely starts cases not reached previously.
            row = search_case(example, model, cfg, spec, folder,
                              resume=resume and (folder/'progress.json').exists())
            rows.append(dict(case=example.name, normalized_operator_digest=example.group_digest,
                path=str((folder/'result.json').relative_to(out)), sha256=digest_file(folder/'result.json'),
                confirmed_improvement=row['confirmed_improvement'],
                label_eligible=row['label_eligible'],
                selection_outcome=row['selection_outcome'],
                offline_seconds=row['offline_seconds']))
            write_json(out/'search/progress.json', dict(status='running', completed_cases=index+1,
                                                      total_cases=len(training)))
            print(f'[SEARCH light] {index+1}/{len(training)} {example.name}: '
                  f'confirmed_improvement={row["confirmed_improvement"]}', flush=True)
        summary = dict(version=VERSION, status='complete', rows=rows,
            improved_cases=sum(r['confirmed_improvement'] for r in rows), total_cases=len(rows),
            eligible_cases=sum(r['label_eligible'] for r in rows),
            failed_cases=[r['case'] for r in rows if not r['label_eligible']],
            offline_seconds=sum(r['offline_seconds'] for r in rows),
            fallback_allowed=False, use='training-only restricted-search oracle, not a deployable model')
        write_json(out/'search/summary.json', summary)
        write_json(out/'search/progress.json', dict(status='complete', completed_cases=len(rows), total_cases=len(rows)))
        return summary


def _search_data(out, training):
    summary = _read(out/'search/summary.json')
    if summary.get('status') != 'complete' or len(summary['rows']) != len(training):
        raise ValueError('Complete training-only search before fit')
    xs, ys, operators = [], [], []
    for example, row in zip(training, summary['rows']):
        if (row['case'] != example.name or row['normalized_operator_digest'] != example.group_digest
                or digest_file(out/row['path']) != row['sha256']):
            raise ValueError('Search dataset provenance changed')
        record = _read(out/row['path'])
        if record['status'] != 'complete' or record['fallback_allowed']:
            raise ValueError('Search labels must be completed without fallback')
        confirmation = record['confirmation']
        selected = confirmation['selected_index']
        if (record.get('label_eligible') is not True
                or record.get('selected_candidate_success') is not True
                or not confirmation['summaries'][selected]['eligible']):
            raise ValueError(f'No successful search label for {example.name}; '
                'failed operator and trials remain recorded. Use a new search '
                'configuration/run; do not drop this case or fit its failed baseline.')
        xs.extend(record['features'])
        ys.extend(record['selected_coefficients'])
        operators.append(example.group_digest)
    if not xs:
        raise ValueError('Search has no nonterminal-level training rows')
    return np.asarray(xs), np.asarray(ys), operators


def fit(output, *, resume=False):
    out, settings, _, baseline = load(output)
    with _lease(out):
        _development_mutable(out)
        training, _, _ = _data(out, settings, baseline)
        x, y, operators = _search_data(out, training)
        provenance = _fit_provenance(out, operators)
        result = fit_models(x, y, out/'fit', dict(settings['fit'], bounds=settings['bounds']),
                            provenance, resume=resume)
        print(f'Fitted shared MLP (242) and ridge (26), using identical {len(x)} training level rows', flush=True)
        return result


def _fit_provenance(out, operators):
    return dict(split='train', operators=operators,
        source_manifest=_hash(_read(out/'light_manifest.json')),
        search_summary_sha256=digest_file(out/'search/summary.json'),
        transfer_signature=_read(out/'import_manifest.json')['transfer_signature'])


def _fitted(out, settings, baseline):
    training, _, _ = _data(out, settings, baseline)
    x, y, operators = _search_data(out, training)
    spec = validate_fit_settings(dict(settings['fit'], bounds=settings['bounds']))
    expected = fit_manifest(x, y, spec, _fit_provenance(out, operators), Standardizer.fit(x))
    return load_fitted(out/'fit', expected_manifest=expected)


def _arms(out, settings, cfg, baseline, *, include_online=False):
    # load_fitted verifies checkpoint/scaler/training status hashes.
    predictors = _fitted(out, settings, baseline)
    p = _p_model(out)
    emcfg = replace(cfg, mg=replace(cfg.mg, strategy_name=FIXED_STRATEGY_NAME,
        em_iterations=5, em_fixed_iterations=True, em_rtol=1e-8))
    arms = {'fixed_C': dict(model=None, branch='C', selector=False),
            'em5_C': dict(model=None, branch='C', selector=False, reference_only=True,
                         config=emcfg, rules=FixedClassicalBaseline.from_config(emcfg))}
    for name, key in zip(BASE_ARMS, (None, 'mlp', 'regression')):
        arms[name] = dict(model=p, branch='H_P', selector=False, light=True,
            predictor=predictors[key] if key else None, online=None, bounds=settings['bounds'])
    if include_online:
        for name, key, method in zip(ONLINE_ARMS, ('mlp', 'regression'), ('mlp_ttt', 'regression_online')):
            arms[name] = dict(model=p, branch='H_P', selector=False, light=True,
                predictor=predictors[key], bounds=settings['bounds'],
                online=dict(settings['online'], method=method, enabled=True))
    imported = _read(out/'import_manifest.json')
    if imported['frozen_line']['included']:
        frozen = Components.load(out/'imported/em_line/checkpoints/frozen_P_line/candidate.pt')
        if frozen.expert_signature('transfer') != p.expert_signature('transfer'):
            raise ValueError('CNN-line comparison P differs from common imported P')
        source_cfg = AdaptiveConfig.from_dict(_read(out/'imported/em_line/configuration.json')['solver'])
        arms['frozen_P_line'] = dict(model=frozen, branch='H_SP', selector=False, config=source_cfg)
    return arms


def _artifacts(out):
    paths = [out/name for name in ('configuration.json', 'fixed_classical.json',
        'light_manifest.json', 'import_manifest.json', 'search/summary.json',
        'fit/fit_manifest.json', 'fit/status.json', 'fit/mlp.json', 'fit/regression.json')]
    paths += [out/record['path'] for record in _read(out/'import_manifest.json')['files'].values()]
    paths += [out/record['path'] for record in _read(out/'search/summary.json')['rows']]
    return {str(p.relative_to(out)): dict(path=str(p), sha256=digest_file(p)) for p in paths}


def benchmark(output, *, resume=False, include_online=False, **protocol):
    protocol = _validate_protocol(protocol)
    out, settings, cfg, baseline = load(output)
    with _lease(out):
        _ensure_development_open(out)
        if 'multiple' not in protocol['regimes'] or settings['target_rhs_count'] not in protocol['rhs_counts']:
            raise ValueError('Include multiple and target_rhs_count for setup-inclusive selection')
        _, examples, _ = _data(out, settings, baseline)
        arms = _arms(out, settings, cfg, baseline, include_online=include_online)
        target = out/'benchmarks/validation'
        result = evaluate_light(examples, arms, cfg, baseline, target, resume=resume, **protocol)
        report(result, target)
        candidates = tuple(name for name in arms if name not in ('fixed_C', 'em5_C'))
        selection = select_light_candidate(result, rhs_count=settings['target_rhs_count'], candidate_arms=candidates)
        write_json(target/'selection.json', selection)
        evidence = dict(version=VERSION, artifacts=_artifacts(out),
            files={str(p.relative_to(out)): dict(path=str(p), sha256=digest_file(p)) for p in
                   (target/'raw_results.json', target/'cycle_diagnostics.json', target/'selection.json',
                    target/'paired_effects.json')},
            include_online=include_online, selection=selection, performance_certified=False)
        write_json(out/'benchmark_evidence.json', evidence)
        write_json(out/'selected/deployment.json',
                   _deployment(out, settings, selection, include_online))
        print(f'Setup + solve recommendation: {selection["recommended_deployment_arm"]}; '
              'warm is secondary; smoke is not speedup evidence.', flush=True)
        return result


def _deployment(out, settings, selection, include_online):
    return dict(arm=selection['recommended_deployment_arm'],
        selection=str(out/'benchmarks/validation/selection.json'),
        common_p=str(out/'imported/em_line/checkpoints/em_P/candidate.pt'),
        mlp=str(out/'fit/mlp.json'), regression=str(out/'fit/regression.json'),
        online_settings=settings['online'], include_online=include_online)


def freeze(output, **protocol):
    protocol = _validate_protocol(protocol, final=True)
    out, settings, cfg, baseline = load(output)
    with _lease(out):
        _ensure_development_open(out)
        evidence = _read(out/'benchmark_evidence.json')
        if evidence['artifacts'] != _artifacts(out):
            raise ValueError('Models/search/import provenance changed after validation')
        for item in evidence['files'].values():
            if digest_file(item['path']) != item['sha256']:
                raise ValueError('Validation evidence changed')
        if 'multiple' not in protocol['regimes'] or settings['target_rhs_count'] not in protocol['rhs_counts']:
            raise ValueError('Final must include setup-inclusive target workload')
        arms = _arms(out, settings, cfg, baseline, include_online=evidence['include_online'])
        selected = _read(out/'selected/deployment.json')
        if selected != _deployment(out, settings, evidence['selection'], evidence['include_online']):
            raise ValueError('Deployment differs from validation selection')
        files = {k: r['path'] for k, r in {**evidence['artifacts'], **evidence['files']}.items()}
        files.update(evidence=out/'benchmark_evidence.json', deployment=out/'selected/deployment.json')
        evaluation = dict(version=VERSION, solver=cfg.to_dict(), arms=list(arms),
            include_online=evidence['include_online'], selected_arm=selected['arm'],
            target_rhs_count=settings['target_rhs_count'], **protocol)
        return freeze_research(out/'data', dict(common_P=out/'imported/em_line/checkpoints/em_P/candidate.pt',
            mlp=out/'fit/mlp.json', regression=out/'fit/regression.json'),
            out/'fixed_classical.json', files, _sources(), evaluation)


def final(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    path = out/'data/research_freeze.json'
    evaluation = _read(path)['config']
    arms = _arms(out, settings, cfg, baseline, include_online=evaluation['include_online'])
    if (evaluation['version'] != VERSION or evaluation['solver'] != json_safe(cfg.to_dict())
            or evaluation['arms'] != list(arms) or evaluation['target_rhs_count'] != settings['target_rhs_count']):
        raise ValueError('Frozen light study protocol changed')
    with claim_final_evaluation(out/'data', path, evaluation, resume=resume) as claim:
        splits, _ = materialize_final_data(out/'data', claim, baseline)
        outputs = []
        for name in evaluation['splits']:
            target = out/'final'/name
            result = evaluate_light(splits[name], arms, cfg, baseline, target,
                resume=resume and (target/'run_manifest.json').exists(),
                **{key: evaluation[key] for key in ('repeats', 'warmups', 'rhs_counts', 'regimes')})
            report(result, target)
            outputs.extend(target/key for key in ('raw_results.json', 'cycle_diagnostics.json', 'paired_effects.json'))
        complete_final_evaluation(out/'data', claim, outputs)
    print('Final/OOD sealed; no final-set model reselection.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'search', 'fit', 'benchmark', 'freeze', 'final'):
        p = sub.add_parser(name)
        p.add_argument('--run-dir', required=True)
        if name != 'freeze':
            p.add_argument('--resume', action='store_true')
        if name == 'prepare':
            p.add_argument('--config', default='configs/v6_7_light_smoother_research.json')
            p.add_argument('--em-run-dir')
        if name == 'benchmark':
            p.add_argument('--include-online', action='store_true', help='Add two separately reported online arms')
        if name in ('benchmark', 'freeze'):
            p.add_argument('--repeats', type=int, default=3)
            p.add_argument('--warmups', type=int, default=1)
            p.add_argument('--rhs-counts', nargs='+', type=int, default=[1, 4, 16, 64])
            p.add_argument('--regimes', nargs='+', choices=('cold', 'warm', 'multiple'), default=['cold', 'warm', 'multiple'])
        if name == 'freeze':
            p.add_argument('--splits', nargs='+', choices=HOLDOUT, default=list(HOLDOUT))
    a = parser.parse_args(argv)
    if a.command == 'prepare':
        return prepare(a.config, a.run_dir, em_run_dir=a.em_run_dir, resume=a.resume)
    if a.command in ('search', 'fit', 'final'):
        return globals()[a.command](a.run_dir, resume=a.resume)
    protocol = dict(repeats=a.repeats, warmups=a.warmups, rhs_counts=a.rhs_counts, regimes=a.regimes)
    if a.command == 'freeze':
        return freeze(a.run_dir, splits=a.splits, **protocol)
    return benchmark(a.run_dir, resume=a.resume, include_online=a.include_online, **protocol)
