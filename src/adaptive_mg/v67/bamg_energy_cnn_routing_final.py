"""Freeze validation-only grid routing before one final/OOD evaluation.

All fixed-lambda CNNs and all prespecified seeds remain visible. The router is
evaluated as an actual deployment using the original finest grid, never as a
minimum over final timings or a mixture of CNNs at different MG levels.
"""
from __future__ import annotations

from pathlib import Path

from . import bamg_energy_cnn_routing_benchmark as validation_benchmark
from . import bamg_energy_cnn_routing_holdout_benchmark as heldout_benchmark
from . import bamg_energy_cnn_routing as routing
from . import bamg_energy_holdout_data as heldout_data
from .bamg_energy_cnn_final import (
    _audit_final_data, _check_files, _groups, study_lock,
)
from .bamg_energy_study import digest, file_digest, read_json, write_json

VERSION = 'bamg-energy-cnn-grid-routing-frozen-final-v1'


def _current_study(out):
    from . import bamg_energy_cnn_routing_study as study
    settings = read_json(out / 'configuration.json')
    manifest = read_json(out / 'study_manifest.json')
    study._initialize(settings)
    if study._manifest(settings, manifest['source'], manifest['teacher']) != manifest:
        raise ValueError('Study source/configuration/environment changed; use its original checkout')
    _check_files(out, read_json(out / 'prepared_manifest.json'))
    return settings, manifest


def _checkpoint_declaration(out, checkpoints):
    """Bind routes and every alternative's study-relative location, not just files."""
    def relative(value):
        if isinstance(value, dict):
            if value.get('kind') != 'grid_router_v1':
                raise ValueError('Unknown frozen checkpoint descriptor')
            return dict(kind=value['kind'], policy=value['policy'],
                        checkpoints={key: relative(path)
                                     for key, path in sorted(value['checkpoints'].items())})
        path = Path(value).resolve()
        if not path.is_relative_to(out):
            raise ValueError('Frozen checkpoint must be inside the study')
        return str(path.relative_to(out))
    return {name: relative(value) for name, value in sorted(checkpoints.items())}


def _verify_freeze(out):
    from . import bamg_energy_cnn_routing_study as study
    settings, manifest = _current_study(out)
    path = out / 'research_freeze.json'
    if not path.is_file():
        raise ValueError('Run freeze after completed validation before evaluating final/OOD')
    frozen = read_json(path)
    if frozen.get('version') != VERSION or frozen.get('study_digest') != digest(manifest):
        raise ValueError('Final freeze does not describe this routing study')
    _check_files(out, frozen['files'])
    selection = read_json(out / 'routing_selection.json')
    if frozen['selection'] != selection:
        raise ValueError('Frozen grid routing selection changed')
    plan = read_json(out / 'holdout_plan.json')
    heldout_data.validate_plan(plan)
    if frozen['plan_digest'] != digest(plan):
        raise ValueError('Frozen holdout plan changed')
    checkpoints = study.checkpoint_map(out, settings, selection=selection)
    if frozen['checkpoints'] != _checkpoint_declaration(out, checkpoints):
        raise ValueError('Frozen checkpoint method set or grid routing changed')
    if frozen['checkpoint_hashes'] != validation_benchmark.checkpoint_hashes(checkpoints):
        raise ValueError('Frozen checkpoint content or grid routing changed')
    return settings, frozen, plan, checkpoints


def _audit_validation(out, name, settings, checkpoints, identity, expected_cases):
    root = out / name
    summary = validation_benchmark.summarize(root)
    manifest = read_json(root / 'benchmark_manifest.json')
    if not summary['completed']:
        raise ValueError('Complete fixed and routed validation before freezing final/OOD')
    if manifest['identity'] != identity:
        raise ValueError('Validation data/teacher/routing/study provenance changed: ' + name)
    observed = [(row['case_id'], row['normalized_operator_digest']) for row in manifest['cases']]
    if observed != expected_cases:
        raise ValueError('Validation cases do not match the prepared split: ' + name)
    if manifest['checkpoint_hashes'] != validation_benchmark.checkpoint_hashes(checkpoints):
        raise ValueError('Validation did not evaluate the current checkpoint set: ' + name)
    if manifest['settings_digest'] != digest(settings):
        raise ValueError('Validation settings do not match the study: ' + name)
    methods = set(settings['numerical_baselines']) | set(checkpoints)
    if set(summary['methods']) != methods or set(manifest['methods']) != methods:
        raise ValueError('Validation method set is incomplete: ' + name)
    return summary


def freeze(output):
    from . import bamg_energy_cnn_routing_study as study
    from .bamg_training import _digest
    out = Path(output).resolve()
    with study_lock(out):
        if (out / 'research_freeze.json').exists():
            _verify_freeze(out)
            return read_json(out / 'research_freeze.json')
        if (out / 'final/claim.json').exists():
            raise ValueError('Final evaluation already claimed; cannot replace its freeze')
        settings, manifest = _current_study(out)
        prepared = read_json(out / 'prepared_manifest.json')
        if 'holdout_plan.json' not in prepared:
            raise ValueError('Holdout plan was not committed before training; use a new study')
        plan = read_json(out / 'holdout_plan.json')
        heldout_data.validate_plan(plan)
        if not (out / 'comparison.json').is_file():
            raise ValueError('Complete training and validation before freezing final/OOD')
        report = read_json(out / 'comparison.json')
        development = read_json(out / 'data/development_manifest.json')
        expected_report = dict(version=settings['version'], lambdas=settings['lambdas'],
            numerical_selected_lambda=manifest['teacher']['selected_lambda'],
            source_model=manifest['source']['selected_model'],
            init_weight_sha256=manifest['source']['init_weight_sha256'],
            model_seeds=settings['model_seeds'], train_cases=len(development['splits']['train']),
            validation_cases=len(development['splits']['validation']))
        if any(report.get(key) != value for key, value in expected_report.items()):
            raise ValueError('Validation report does not describe the frozen study inputs')
        fixed = study.checkpoint_map(out, settings, include_routed=False)
        identity = study.provenance(out, manifest)
        cases = [(row['operator_digest'], row['normalized_operator_digest'])
                 for row in development['splits']['validation']]
        summary = _audit_validation(out, 'validation', settings, fixed, identity, cases)
        if not report.get('completed') or report.get('benchmark') != summary:
            raise ValueError('Complete unchanged fixed validation before freezing final/OOD')
        selection = routing.select_routes(summary, settings)
        if (read_json(out / 'routing_selection.json') != selection
                or report.get('routing_selection') != selection):
            raise ValueError('Grid routing must be recomputed from unchanged development validation')
        checkpoints = study.checkpoint_map(out, settings, selection=selection)
        routed = {name: value for name, value in checkpoints.items() if name not in fixed}
        routed_identity = dict(identity,
            routing_selection_sha256=file_digest(out / 'routing_selection.json'))
        routed_summary = _audit_validation(out, 'validation_routed', settings, routed,
                                           routed_identity, cases)
        if report.get('routed_benchmark') != routed_summary:
            raise ValueError('Complete unchanged routed validation before freezing final/OOD')
        run_specifications = {routing.run_name(strength, arm, seed): (strength, arm, seed)
                              for strength in settings['lambdas']
                              for seed in settings['model_seeds'] for arm in ('scratch', 'warm')}
        expected_runs = set(run_specifications)
        if set(report['runs']) != expected_runs:
            raise ValueError('Prespecified lambda/seed/arm set is incomplete')
        required = ['configuration.json', 'study_manifest.json', 'prepared_manifest.json',
                    'fixed_classical.json', 'comparison.json', 'routing_selection.json',
                    'teacher_bank_manifest.json']
        for folder in ('validation', 'validation_routed'):
            required.append(folder + '/benchmark_manifest.json')
            required += [str(path.relative_to(out)) for path in (out / folder / 'benchmark_cases').glob('*.json')]
        expected_train = {row['operator_digest'] for row in development['splits']['train']}
        bank_audit = read_json(out / 'teacher_bank_manifest.json')
        if bank_audit.get('study') != digest(manifest) or set(bank_audit.get('records', {})) != expected_train:
            raise ValueError('Shared teacher vector-bank audit changed')
        for strength in settings['lambdas']:
            relative = 'labels/lambda_' + routing.lambda_key(strength) + '/teacher/train/manifest.json'
            label_manifest = read_json(out / relative)
            expected_label_identity = dict(study=digest(manifest), prepared=prepared,
                                           lambda_value=float(strength))
            if (label_manifest['identity'] != expected_label_identity
                    or set(label_manifest['records']) != expected_train):
                raise ValueError('TRAIN teacher cache identity or operator cohort changed')
            for case_id, expected in label_manifest['records'].items():
                path = (out / relative).parent / (case_id + '.pt')
                if not path.is_file() or file_digest(path) != expected:
                    raise ValueError('TRAIN teacher cache changed before freeze: ' + case_id)
            required.append(relative)
        for name in sorted(expected_runs):
            root = out / 'runs' / name
            strength, arm, seed = run_specifications[name]
            completion = read_json(root / 'completion.json')
            if completion.get('status') != 'complete' or report['runs'][name] != completion:
                raise ValueError('All prespecified training arms must finish before freezing')
            arm_identity = read_json(root / 'identity.json')
            if completion['identity_sha256'] != _digest(arm_identity):
                raise ValueError('Completed arm identity changed')
            if arm_identity['settings'] != settings:
                raise ValueError('Completed arm settings differ from the routing study')
            if (arm_identity['arm'] != arm or arm_identity['seed'] != seed
                    or arm_identity['provenance'] != study.arm_provenance(out, manifest, strength)):
                raise ValueError('Completed arm lambda/data/initialization provenance changed')
            for key, expected in completion['artifact_sha256'].items():
                path = Path(completion['paths'][key]).resolve()
                if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != expected:
                    raise ValueError('Completed arm artifact changed: ' + name + '/' + key)
                required.append(str(path.relative_to(out)))
            required += [str((root / name).relative_to(out)) for name in ('completion.json', 'identity.json')]
        files = dict(prepared)
        for name in required:
            files[name] = file_digest(out / name)
        for path in validation_benchmark.checkpoint_files(checkpoints):
            path = path.resolve()
            if not path.is_relative_to(out):
                raise ValueError('Checkpoint escaped the routing study')
            files[str(path.relative_to(out))] = file_digest(path)
        forbidden = _groups(read_json(out / 'inputs/exclusion_manifest.json'))
        forbidden.update(_groups(development))
        frozen = dict(version=VERSION, study_digest=digest(manifest), files=files,
            checkpoints=_checkpoint_declaration(out, checkpoints),
            checkpoint_hashes=validation_benchmark.checkpoint_hashes(checkpoints),
            selection=selection, holdout_selection_permitted=False,
            primary_comparison=['routed_scratch_finetuned', 'routed_warm_finetuned'],
            all_prespecified_seeds_retained=True,
            forbidden_operator_digests=sorted(forbidden),
            evaluation_scope='smoke_only' if settings['smoke'] else 'preregistered_q1_spd_holdout',
            plan_digest=digest(plan), final_operators_materialized=False)
        write_json(out / 'research_freeze.json', frozen)
        print(f'Frozen validation grid routing and {len(heldout_data.specifications(plan))} final/OOD operator specifications.', flush=True)
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
        identity = dict(freeze_sha256=freeze_sha, routing_selection=frozen['selection'],
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
            interpretation='Frozen finest-grid routing on independent Q1 SPD operators; no final/OOD winner selection.')
        write_json(final / 'report.json', result)
        if summary['completed']:
            _audit_final_data(final, plan, frozen, identity, require_complete=True)
            paths = [p for p in final.rglob('*.json') if p != path]
            claim.update(status='completed', result_files={str(p.relative_to(out)): file_digest(p) for p in paths})
            write_json(path, claim)
        return result


def report_final(output, *, group=None):
    out = Path(output).resolve()
    if not (out / 'final/benchmark/benchmark_manifest.json').exists():
        raise FileNotFoundError('No saved final/OOD results; run freeze then evaluate-final')
    _, frozen, plan, _ = _verify_freeze(out)
    claim = read_json(out / 'final/claim.json')
    freeze_sha = file_digest(out / 'research_freeze.json')
    identity = dict(freeze_sha256=freeze_sha, routing_selection=frozen['selection'],
        claim_fingerprint=digest(dict(freeze_sha256=freeze_sha, command='evaluate-final', version=VERSION)))
    if claim['identity'] != identity:
        raise ValueError('Final report does not match its frozen evaluation claim')
    if claim['status'] == 'completed':
        _check_files(out, claim['result_files'])
    _audit_final_data(out / 'final', plan, frozen, identity,
                      require_complete=claim['status'] == 'completed')
    summary = heldout_benchmark.summarize(out / 'final/benchmark')
    print('Final/OOD scope:', frozen['evaluation_scope'])
    for arm, policy in frozen['selection']['arms'].items():
        print(f'Frozen {arm} routing: {policy["grid_to_lambda"]}; '
              f'unseen grids: {policy["unseen_grid_policy"]}; '
              f'global fixed lambda: {policy["global_lambda"]}')
    print('All prespecified seeds retained. No final/OOD model selection.')
    heldout_benchmark.print_summary(summary, group=group)
    return summary
