"""Frozen, single-use final/OOD evaluation for the paired CNN study.

The held-out protocol is committed before training. Freeze uses validation only,
and evaluation retains every prespecified seed rather than selecting on test data.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path

from . import bamg_energy_cnn_benchmark as validation_benchmark
from . import bamg_energy_cnn_holdout_benchmark as heldout_benchmark
from . import bamg_energy_holdout_data as heldout_data
from .bamg_energy_study import digest, file_digest, read_json, write_json

VERSION = 'bamg-energy-cnn-frozen-final-v1'


@contextmanager
def study_lock(out):
    with (out / '.cnn-comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _check_files(out, files):
    for relative, expected in files.items():
        path = (out / relative).resolve()
        if not path.is_relative_to(out) or not path.is_file() or file_digest(path) != expected:
            raise ValueError('Frozen study artifact changed or missing: ' + relative)


def _current_study(out):
    from .bamg_energy_cnn_study import _initialize, _manifest
    settings = read_json(out / 'configuration.json')
    manifest = read_json(out / 'study_manifest.json')
    _initialize(settings)
    if _manifest(settings, manifest['source'], manifest['teacher']) != manifest:
        raise ValueError('Study source/configuration/environment changed; use its original checkout')
    _check_files(out, read_json(out / 'prepared_manifest.json'))
    return settings, manifest


def checkpoint_map(out, settings):
    result = {'old_cnn': out / 'inputs/initial_weight.pt'}
    for seed in settings['model_seeds']:
        for arm in ('scratch', 'warm'):
            name = f'{arm}_seed{seed}'
            root = out / 'runs' / name
            result[name + '_supervised'] = root / 'supervised_models.pt'
            result[name + '_finetuned'] = root / 'deployment.pt'
    return result


def _groups(value):
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('normalized_operator_digest', 'group_digest', 'normalized_digest') and isinstance(child, str):
                found.add(child)
            elif key == 'normalized_operator_digests' and isinstance(child, list):
                found.update(child)
            else:
                found.update(_groups(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_groups(child))
    return found


def _verify_freeze(out):
    settings, manifest = _current_study(out)
    path = out / 'research_freeze.json'
    if not path.is_file():
        raise ValueError('Run freeze after completed validation before evaluating final/OOD')
    frozen = read_json(path)
    if frozen.get('version') != VERSION or frozen['study_digest'] != digest(manifest):
        raise ValueError('Final freeze does not describe this study')
    _check_files(out, frozen['files'])
    plan = read_json(out / 'holdout_plan.json')
    heldout_data.validate_plan(plan)
    checkpoints = checkpoint_map(out, settings)
    actual = {name: str(p.relative_to(out)) for name, p in checkpoints.items()}
    if frozen['checkpoints'] != actual:
        raise ValueError('Frozen checkpoint method set changed')
    return settings, frozen, plan, checkpoints


def _audit_final_data(final, plan, frozen, identity, *, require_complete=False):
    state = heldout_data.validate_cache(final / 'data', plan=plan,
        forbidden_digests=frozen['forbidden_operator_digests'], freeze_identity=identity)
    expected = {s['case_id'] for s in heldout_data.specifications(plan)}
    if require_complete and (not state['completed'] or set(state['cases']) != expected):
        raise ValueError('Completed benchmark lacks complete final/OOD data provenance')
    for path in (final / 'benchmark/benchmark_cases').glob('*.json'):
        recorded = read_json(path)
        actual = recorded.get('actual')
        if actual is None:
            continue
        case_id = recorded['case']['case_id']
        item = state['cases'].get(case_id)
        if (item is None or actual.get('data_record_sha256') != item['sha256']
                or actual['normalized_operator_digest'] != item['normalized_operator_digest']):
            raise ValueError('Final benchmark no longer matches its saved data case: ' + case_id)
    return state


def freeze(output):
    out = Path(output).resolve()
    with study_lock(out):
        if (out / 'research_freeze.json').exists():
            _verify_freeze(out)
            return read_json(out / 'research_freeze.json')
        if (out / 'final/claim.json').exists():
            raise ValueError('Final evaluation already claimed; cannot replace its freeze')
        settings, manifest = _current_study(out)
        if 'holdout_plan.json' not in read_json(out / 'prepared_manifest.json'):
            raise ValueError('Holdout plan was not committed before training; use a new study with holdout configuration')
        plan = read_json(out / 'holdout_plan.json')
        heldout_data.validate_plan(plan)
        if not (out / 'comparison.json').is_file():
            raise ValueError('Complete training and validation before freezing final/OOD')
        report = read_json(out / 'comparison.json')
        development = read_json(out / 'data/development_manifest.json')
        expected_report = dict(version=settings['version'], selected_lambda=manifest['teacher']['selected_lambda'],
            source_model=manifest['source']['selected_model'],
            init_weight_sha256=manifest['source']['init_weight_sha256'],
            model_seeds=settings['model_seeds'], train_cases=len(development['splits']['train']),
            validation_cases=len(development['splits']['validation']))
        if any(report.get(key) != value for key, value in expected_report.items()):
            raise ValueError('Validation report does not describe the frozen study inputs')
        summary = validation_benchmark.summarize(out / 'validation')
        if not report.get('completed') or not summary['completed'] or report['benchmark'] != summary:
            raise ValueError('Complete unchanged validation before freezing final/OOD')
        checkpoints = checkpoint_map(out, settings)
        bench_manifest = read_json(out / 'validation/benchmark_manifest.json')
        prepared = read_json(out / 'prepared_manifest.json')
        expected_provenance = dict(study=digest(manifest), prepared=prepared,
            teacher_manifest_sha256=file_digest(out / 'teacher/train/manifest.json'),
            init_weight_sha256=manifest['source']['init_weight_sha256'])
        if bench_manifest['identity'] != expected_provenance:
            raise ValueError('Validation data/teacher/study provenance changed')
        observed_cases = [(r['case_id'], r['normalized_operator_digest']) for r in bench_manifest['cases']]
        expected_cases = [(r['operator_digest'], r['normalized_operator_digest'])
                          for r in development['splits']['validation']]
        if observed_cases != expected_cases:
            raise ValueError('Validation cases do not match the prepared split')
        expected_hashes = {name: file_digest(p) for name, p in checkpoints.items()}
        if bench_manifest['checkpoint_hashes'] != expected_hashes:
            raise ValueError('Validation did not evaluate the current checkpoint set')
        if bench_manifest['settings_digest'] != digest(settings):
            raise ValueError('Validation settings do not match the study')
        expected_methods = {'fixed_C', 'em5_C', *checkpoints}
        if set(summary['methods']) != expected_methods:
            raise ValueError('Validation method set is incomplete')
        files = dict(read_json(out / 'prepared_manifest.json'))
        required = ['configuration.json', 'study_manifest.json', 'prepared_manifest.json',
                    'fixed_classical.json', 'comparison.json', 'teacher/train/manifest.json',
                    'validation/benchmark_manifest.json']
        required += [str(p.relative_to(out)) for p in (out / 'validation/benchmark_cases').glob('*.json')]
        for name in report['runs']:
            root = out / 'runs' / name
            completion = read_json(root / 'completion.json')
            if completion.get('status') != 'complete' or report['runs'][name] != completion:
                raise ValueError('All prespecified training arms must finish before freezing')
            arm_identity = read_json(root / 'identity.json')
            from .bamg_training import _digest
            if completion['identity_sha256'] != _digest(arm_identity):
                raise ValueError('Completed arm identity changed')
            for key, expected in completion['artifact_sha256'].items():
                path = Path(completion['paths'][key]).resolve()
                if not path.is_relative_to(root) or file_digest(path) != expected:
                    raise ValueError('Completed arm artifact changed: ' + name + '/' + key)
                required.append(str(path.relative_to(out)))
            required += [str((root / name).relative_to(out)) for name in ('completion.json', 'identity.json')]
        expected_runs = {f'{arm}_seed{seed}' for seed in settings['model_seeds'] for arm in ('scratch', 'warm')}
        if set(report['runs']) != expected_runs:
            raise ValueError('Prespecified seed/arm set is incomplete')
        for name in required:
            files[name] = file_digest(out / name)
        for path in checkpoints.values():
            files[str(path.relative_to(out))] = file_digest(path)
        forbidden = _groups(read_json(out / 'inputs/exclusion_manifest.json'))
        forbidden.update(_groups(read_json(out / 'data/development_manifest.json')))
        # Freeze validation choice, but report every seed on final/OOD.
        eligible_learned = [n for n, v in summary['methods'].items()
                            if n in checkpoints and v['eligible']]
        selected_learned = min(eligible_learned,
            key=lambda n: (summary['methods'][n]['failures'], summary['methods'][n]['warm_seconds'], n)) if eligible_learned else None
        frozen = dict(version=VERSION, study_digest=digest(manifest), files=files,
            checkpoints={name: str(p.relative_to(out)) for name, p in checkpoints.items()},
            selection=dict(scope='development_validation_only',
                selected_method=summary['selected_method'], selected_learned_model=selected_learned,
                selected_lambda=report['selected_lambda'], model_seeds=settings['model_seeds'],
                primary_comparison=['scratch_finetuned', 'warm_finetuned'],
                all_prespecified_seeds_retained=True, holdout_selection_permitted=False),
            forbidden_operator_digests=sorted(forbidden),
            evaluation_scope='smoke_only' if settings['smoke'] else 'preregistered_q1_spd_holdout',
            plan_digest=digest(plan), final_operators_materialized=False)
        write_json(out / 'research_freeze.json', frozen)
        print(f'Frozen validation choices and {len(heldout_data.specifications(plan))} final/OOD operator specifications.', flush=True)
        return frozen


def evaluate_final(output, *, resume=False, max_cases=None):
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError('max_cases must be a positive integer')
    out = Path(output).resolve()
    with study_lock(out):
        settings, frozen, plan, checkpoints = _verify_freeze(out)
        final = out / 'final'
        path = final / 'claim.json'
        freeze_sha = file_digest(out / 'research_freeze.json')
        identity = dict(freeze_sha256=freeze_sha,
                        claim_fingerprint=digest(dict(freeze_sha256=freeze_sha, command='evaluate-final', version=VERSION)))
        if path.exists():
            claim = read_json(path)
            if claim['identity'] != identity:
                raise ValueError('Only the identical frozen final evaluation may resume')
            if not resume:
                raise FileExistsError('Final evaluation already claimed; use --resume for this same evaluation')
            if claim['status'] == 'completed':
                _check_files(out, claim['result_files'])
                _audit_final_data(final, plan, frozen, identity, require_complete=True)
                return read_json(final / 'report.json')
        else:
            if resume:
                raise ValueError('Cannot resume an unclaimed final evaluation')
            if final.exists() and any(final.iterdir()):
                raise ValueError('Unclaimed final output must be empty')
            claim = dict(version=VERSION, status='claimed', identity=identity)
            write_json(path, claim)
        # Audit cached metadata before skipping any already measured case.
        _audit_final_data(final, plan, frozen, identity)
        cases = heldout_data.specifications(plan)
        def load_case(record):
            return heldout_data.load_case(final / 'data', record, plan=plan,
                forbidden_digests=frozen['forbidden_operator_digests'],
                freeze_identity=identity, resume=True)
        summary = heldout_benchmark.benchmark(final / 'benchmark', cases=cases,
            load_example=load_case, settings=settings, checkpoints=checkpoints,
            identity=identity, resume=resume, max_cases=max_cases)
        result = dict(version=VERSION, completed=summary['completed'],
                      evaluation_scope=frozen['evaluation_scope'], selection=frozen['selection'],
                      holdout_selection_permitted=False, benchmark=summary,
                      interpretation='Independent operators from declared Q1 SPD distributions; not a universal generalization guarantee.')
        write_json(final / 'report.json', result)
        if summary['completed']:
            _audit_final_data(final, plan, frozen, identity, require_complete=True)
            # Seal all compact evidence; completed resume never retimes.
            paths = [p for p in final.rglob('*.json') if p != path]
            claim.update(status='completed', result_files={str(p.relative_to(out)): file_digest(p) for p in paths})
            write_json(path, claim)
        return result


def report_final(output, *, group=None):
    out = Path(output).resolve()
    report_path = out / 'final/report.json'
    if not (out / 'final/benchmark/benchmark_manifest.json').exists():
        raise FileNotFoundError('No saved final/OOD results; run freeze then evaluate-final')
    if report_path.exists():
        result = read_json(report_path)
    else:
        frozen = read_json(out / 'research_freeze.json')
        result = dict(evaluation_scope=frozen['evaluation_scope'], selection=frozen['selection'])
    claim = read_json(out / 'final/claim.json')
    if claim['status'] == 'completed':
        _check_files(out, claim['result_files'])
    # Partial summaries can be refreshed from existing scalar measurements only.
    summary = heldout_benchmark.summarize(out / 'final/benchmark')
    print('Final/OOD scope:', result['evaluation_scope'])
    print('Validation-selected lambda:', result['selection']['selected_lambda'])
    print('Validation-selected model:', result['selection']['selected_method'])
    heldout_benchmark.print_summary(summary, group=group)
    return summary
