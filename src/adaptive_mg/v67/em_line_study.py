"""EM-imitation P -> frozen-P line learning -> joint full-V-cycle fine-tuning.

EM labels are offline and conditioned on each actual student level operator.
Deployment generates P and shared pre/post line factors once per level; model
selection measures setup plus all solves at the identical residual tolerance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import torch

from ..provenance import hardware_environment, json_safe
from .config import AdaptiveConfig
from .direct_pruning_study import _ensure_development_open, _validate_protocol
from .em_line_models import make_em_line_components
from .em_line_training import em_imitation_objective, line_vcycle_objective
from .fixed_classical import (BilinearClassicalBaseline, FixedClassicalBaseline,
                              FIXED_STRATEGY_NAME, load_baseline)
from .joint_transfer_study import _read, _initialize, _development, _lease, _save_state
from .joint_transfer_training import curriculum_cycles
from .joint_transfer_evaluation import raw_contractions, report, select_measured_candidate
from .limited import digest_file, forced_config
from .models import Components
from .research_data import (freeze_research, claim_final_evaluation,
    materialize_final_data, complete_final_evaluation, HOLDOUT, _hash,
    _write_json as write_json)
from .research_evaluation import evaluate_research

VERSION = 'em-imitation-shared-line-v1'
PROJECT = Path(__file__).resolve().parents[3]
STAGES = {'em_P': ('H_P', ('transfer',)),
          'frozen_P_line': ('H_SP', ('smoother',)),
          'joint_line': ('H_SP', ('transfer', 'smoother'))}
MODULES = ('smoother', 'transfer', 'detector', 'controller')


def validate_settings(settings):
    if settings.get('version') != VERSION:
        raise ValueError('Require a new EM/shared-line study configuration')
    cfg = AdaptiveConfig.from_dict(settings['solver'])
    BilinearClassicalBaseline.from_config(cfg)
    if (cfg.branch != 'H_SP' or cfg.mode != 'research' or not cfg.use_transfer
            or not cfg.use_smoother or cfg.smoother_levels != 'all'
            or cfg.transfer_levels != 'all' or cfg.replace_pre != 2 or cfg.replace_post != 2
            or cfg.replacement_group_pre != 1 or cfg.replacement_group_post != 1
            or cfg.spatial or cfg.gate_mode != 'open' or cfg.use_learned_controller
            or not cfg.lazy or cfg.application != 'replace'):
        raise ValueError('Require all-level P and shared line replacement, pre=2/post=2')
    for key in ('seed', 'hidden', 'torch_threads', 'training_per_family',
                'validation_per_family', 'raw_cycles', 'raw_probes', 'target_rhs_count'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(settings['smoke']) is not bool:
        raise ValueError('smoke must be a boolean')
    for key in ('training_sizes', 'validation_sizes'):
        values = settings[key]
        if (not values or len(set(values)) != len(values)
                or any(type(n) is not int or n <= cfg.mg.coarsest_n or n & (n+1) for n in values)):
            raise ValueError('Require distinct 2**L-1 grids with a nonterminal level')
    if set(settings['stages']) != set(STAGES):
        raise ValueError('All three sequential stages are required')
    em = settings['em_teacher']
    if em != dict(iterations=5, rtol=1e-8, fixed_iterations=True):
        raise ValueError('EM teacher uses an explicit fixed five-iteration budget per level')
    for name, spec in settings['stages'].items():
        if type(spec['updates']) is not int or spec['updates'] < 1:
            raise ValueError(f'{name}.updates must be positive')
        for key in ('learning_rate', 'gradient_clip'):
            if not np.isfinite(spec[key]) or spec[key] <= 0:
                raise ValueError(f'{name}.{key} must be finite and positive')
        if name != 'em_P':
            for key in ('start_cycles', 'end_cycles', 'probes_per_kind'):
                if type(spec[key]) is not int or spec[key] < 1:
                    raise ValueError(f'{name}.{key} must be a positive integer')
            if spec['end_cycles'] < spec['start_cycles']:
                raise ValueError('Cycle curriculum cannot decrease')
            for key in ('hard_smoothing', 'hard_cycles'):
                if type(spec[key]) is not int or spec[key] < 0:
                    raise ValueError(f'{name}.{key} must be a nonnegative integer')
            if not np.isfinite(spec['lambda_stability']) or spec['lambda_stability'] < 0:
                raise ValueError('Invalid stability-loss coefficient')
    return cfg


def _sources():
    return sorted([p for p in (PROJECT/'src').rglob('*') if p.suffix in ('.py', '.cpp')]
                  + [PROJECT/'scripts/run_v6_7_em_line_study.py'])


def _manifest(settings, baseline):
    return dict(version=VERSION, settings_digest=_hash(settings), baseline_digest=baseline.digest(),
        source_digest=_hash({str(p.relative_to(PROJECT)): digest_file(p) for p in _sources()}),
        hardware=hardware_environment(refresh=True))


def load(output):
    out = Path(output).resolve()
    settings = _read(out/'configuration.json')
    cfg = validate_settings(settings)
    _initialize(settings)
    baseline = load_baseline(out/'fixed_classical.json')
    baseline.validate_config(cfg)
    if _read(out/'em_line_manifest.json') != json_safe(_manifest(settings, baseline)):
        raise ValueError('Sources/settings/environment changed; use a new EM-line run')
    return out, settings, cfg, baseline


def prepare(config, output, *, resume=False):
    settings = _read(config)
    cfg = validate_settings(settings)
    _initialize(settings)
    baseline = BilinearClassicalBaseline.from_config(cfg)
    out = Path(output).resolve()
    if (out/'em_line_manifest.json').exists():
        if not resume:
            raise FileExistsError('Run exists; use identical --resume or a new directory')
        _, stored, _, _ = load(out)
        if stored != settings:
            raise ValueError('Prepare settings differ')
        _ensure_development_open(out)
    else:
        if resume:
            raise FileNotFoundError('No EM-line run to resume')
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Prepare requires an empty directory')
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'configuration.json', settings)
        write_json(out/'fixed_classical.json', baseline.to_dict())
        model = make_em_line_components(hidden=settings['hidden'], seed=settings['seed'])
        model.save(out/'initial.pt')
        write_json(out/'initial.json', dict(sha256=digest_file(out/'initial.pt')))
        write_json(out/'em_line_manifest.json', _manifest(settings, baseline))
    training, validation, _ = _development(out, settings, baseline)
    print(f'Prepared EM-line study: train={len(training)}, validation={len(validation)}; '
          'EM5 offline labels, bilinear fallback; em_P -> frozen_P_line -> joint_line', flush=True)


def _initial(out):
    path = out/'initial.pt'
    if digest_file(path) != _read(out/'initial.json')['sha256']:
        raise ValueError('Initial checkpoint changed')
    return Components.load(path)


def _training_manifest(name, initial, cfg, baseline, settings, operators):
    branch, names = STAGES[name]
    return dict(version=VERSION, stage=name, config=forced_config(cfg, branch).to_dict(),
        settings=settings['stages'][name], em_teacher=settings['em_teacher'],
        baseline_digest=baseline.digest(), initial_signature=initial.signature(),
        frozen_component_signatures={key: initial.expert_signature(key)
                                     for key in MODULES if key not in names},
        operators=list(operators))


def _checkpoint(out, name, baseline, *, initial=None, settings=None):
    settings = _read(out/'configuration.json') if settings is None else settings
    cfg = validate_settings(settings)
    if initial is None:
        index = tuple(STAGES).index(name)
        initial = (_initial(out) if index == 0 else
                   _checkpoint(out, tuple(STAGES)[index-1], baseline, settings=settings))
    operators = [record['normalized_operator_digest'] for record in
                 _read(out/'data/development_manifest.json')['splits']['train']]
    expected = _training_manifest(name, initial, cfg, baseline, settings, operators)
    folder = out/'checkpoints'/name
    status, training = _read(folder/'status.json'), _read(folder/'training_manifest.json')
    path = folder/'candidate.pt'
    if (training != json_safe(expected)
            or status.get('status') != 'complete' or not status.get('all_updates_completed')
            or status.get('skipped') != 0 or status.get('updates') != training['settings']['updates']
            or training.get('stage') != name or training.get('baseline_digest') != baseline.digest()
            or status.get('baseline_digest') != baseline.digest()
            or status.get('checkpoint_sha256') != digest_file(path)
            or status.get('frozen_component_signatures') != expected['frozen_component_signatures']):
        raise ValueError(f'Incomplete or changed stage: {name}')
    model = Components.load(path)
    if (model.metadata.get('em_line_stage') != name
            or model.metadata.get('training_baseline_digest') != baseline.digest()
            or model.metadata.get('previous_signature') != initial.signature()
            or model.metadata.get('previous_stage') != initial.metadata.get('em_line_stage')
            or model.metadata.get('training_branch') != STAGES[name][0]
            or model.metadata.get('trainable_components') != list(STAGES[name][1])
            or model.metadata.get('training_operator_digests') != operators):
        raise ValueError('Checkpoint lineage differs')
    for key, signature in status['frozen_component_signatures'].items():
        if model.expert_signature(key) != signature:
            raise ValueError(f'Frozen {key} changed in {name}')
    return model


def _run_training(out, name, initial, examples, cfg, baseline, settings, *, resume=False):
    branch, names = STAGES[name]
    cfg = forced_config(cfg, branch)
    spec = settings['stages'][name]
    total = spec['updates']
    folder = out/'checkpoints'/name
    frozen = {key: initial.expert_signature(key) for key in MODULES if key not in names}
    manifest = _training_manifest(name, initial, cfg, baseline, settings,
                                  [e.group_digest for e in examples])
    path = folder/'training_manifest.json'
    if path.exists():
        if not resume or _read(path) != json_safe(manifest):
            raise ValueError('Training exists; identical --resume required')
        if (folder/'status.json').exists() and _read(folder/'status.json').get('status') == 'complete':
            return _checkpoint(out, name, baseline, initial=initial, settings=settings)
    else:
        write_json(path, manifest)
    model = deepcopy(initial)
    for key in MODULES:
        module = getattr(model, key)
        module.train(key in names)
        for parameter in module.parameters():
            parameter.requires_grad_(key in names)
    params = [p for key in names for p in getattr(model, key).parameters()]
    optimizer = torch.optim.Adam(params, lr=spec['learning_rate'])
    statepath = folder/'resume.pt'
    history, start = [], 0
    if statepath.exists():
        if not resume:
            raise ValueError('Interrupted training requires --resume')
        state = torch.load(statepath, map_location='cpu', weights_only=True)
        if (state.get('manifest_digest') != _hash(manifest)
                or type(state.get('updates')) is not int or not 0 <= state['updates'] <= total
                or len(state.get('history', [])) != state['updates']
                or set(state.get('model', {})) != set(MODULES)):
            raise ValueError('Training resume belongs to a different protocol')
        for key, value in state['model'].items():
            getattr(model, key).load_state_dict(value)
        optimizer.load_state_dict(state['optimizer'])
        history, start = state['history'], state['updates']
    stage_seed = settings['seed'] + 1000003*list(STAGES).index(name)
    try:
        for update in range(start, total):
            epoch, position = divmod(update, len(examples))
            order = np.random.default_rng(stage_seed+epoch).permutation(len(examples))
            example = examples[int(order[position])]
            optimizer.zero_grad(set_to_none=True)
            if name == 'em_P':
                cycles = 0
                loss, metrics = em_imitation_objective(example, model, cfg,
                    em_iterations=settings['em_teacher']['iterations'], em_rtol=settings['em_teacher']['rtol'])
            else:
                cycles = curriculum_cycles(update, total, start=spec['start_cycles'], end=spec['end_cycles'])
                loss, metrics = line_vcycle_objective(example, model, cfg, cycles=cycles,
                    seed=stage_seed+update*104729, probes_per_kind=spec['probes_per_kind'],
                    hard_smoothing=spec['hard_smoothing'], hard_cycles=spec['hard_cycles'],
                    lambda_stability=spec['lambda_stability'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, spec['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in params):
                raise FloatingPointError('Nonfinite model after optimizer update')
            history.append(dict(update=update+1, case=example.name, cycles=cycles,
                loss=float(loss.detach()), gradient_norm=float(norm), metrics=json_safe(metrics)))
            _save_state(statepath, dict(manifest_digest=_hash(manifest), updates=update+1,
                model={key: getattr(model, key).state_dict() for key in MODULES},
                optimizer=optimizer.state_dict(), history=history))
            write_json(folder/'status.json', dict(status='running', updates=update+1, total=total))
            print(f'[TRAIN {name}] {update+1}/{total} {example.name}: '
                  f'cycles={cycles}, loss={float(loss.detach()):.6g}', flush=True)
        for key, signature in frozen.items():
            if model.expert_signature(key) != signature:
                raise ValueError(f'Frozen {key} changed during {name}')
    except (ValueError, RuntimeError, FloatingPointError) as error:
        write_json(folder/'status.json', dict(status='failed', updates=len(history), reason=str(error)))
        raise
    model.metadata.update(em_line_stage=name, training_baseline_digest=baseline.digest(),
        training_branch=branch, trainable_components=list(names),
        smoother_trained=initial.metadata.get('smoother_trained', False) or 'smoother' in names,
        transfer_trained=initial.metadata.get('transfer_trained', False) or 'transfer' in names,
        training_operator_digests=[e.group_digest for e in examples],
        previous_stage=initial.metadata.get('em_line_stage'),
        previous_signature=initial.signature(), final_test_seen=False, online_em=False,
        offline_em_iteration_budget=settings['em_teacher']['iterations'])
    model.eval().save(folder/'candidate.pt')
    write_json(folder/'training.json', history)
    write_json(folder/'status.json', dict(status='complete', updates=total, all_updates_completed=True,
        skipped=0, baseline_digest=baseline.digest(), checkpoint_sha256=digest_file(folder/'candidate.pt'),
        frozen_component_signatures=frozen))
    return model


def train(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    with _lease(out):
        _ensure_development_open(out)
        if ((out/'benchmark_evidence.json').exists()
                or (out/'benchmarks/validation/run_manifest.json').exists()):
            raise ValueError('Validation started; use a new run to change learned weights')
        examples, _, _ = _development(out, settings, baseline)
        model = _initial(out)
        for name in STAGES:
            model = _run_training(out, name, model, examples, cfg, baseline, settings, resume=resume)


def _arms(out, cfg, baseline):
    emcfg = replace(cfg, mg=replace(cfg.mg, strategy_name=FIXED_STRATEGY_NAME,
        em_iterations=5, em_fixed_iterations=True, em_rtol=1e-8))
    arms = {'fixed_C': dict(model=None, branch='C', selector=False),
            'em5_C': dict(model=None, branch='C', selector=False, reference_only=True,
                          config=emcfg, rules=FixedClassicalBaseline.from_config(emcfg))}
    previous = _initial(out)
    for name, (branch, _) in STAGES.items():
        model = _checkpoint(out, name, baseline, initial=previous)
        arms[name] = dict(model=model, branch=branch, selector=False)
        previous = model
    return arms


def benchmark(output, *, resume=False, **protocol):
    protocol = _validate_protocol(protocol)
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if 'multiple' not in protocol['regimes'] or settings['target_rhs_count'] not in protocol['rhs_counts']:
        raise ValueError('Include multiple and target_rhs_count for setup-inclusive selection')
    with _lease(out):
        _ensure_development_open(out)
        _, examples, _ = _development(out, settings, baseline)
        arms = _arms(out, cfg, baseline)
        target = out/'benchmarks/validation'
        result = evaluate_research(examples, arms, cfg, baseline, target,
            reference_arm='fixed_C', resume=resume, **protocol)
        report(result, target)
        write_json(target/'raw_contractions.json', raw_contractions(examples, arms, cfg,
            cycles=settings['raw_cycles'], probes=settings['raw_probes']))
        selection = select_measured_candidate(result, rhs_count=settings['target_rhs_count'],
            candidate_arms=tuple(STAGES))
        write_json(target/'selection.json', selection)
        files = {key: dict(path=str(target/key), sha256=digest_file(target/key))
                 for key in ('raw_results.json', 'cycle_diagnostics.json', 'raw_contractions.json', 'selection.json')}
        write_json(out/'benchmark_evidence.json', dict(files=files,
            checkpoints={n: a['model'].signature() for n, a in arms.items() if a['model']},
            selection=selection, performance_certified=False))
        # Selection itself records failure/nonselection and the classical option.
        # Export only the validated recommendation, never an unqualified last update.
        selected = selection.get('recommended_deployment_arm')
        export = dict(arm=selected, selection=str(target/'selection.json'), checkpoint=None)
        if selected in STAGES:
            path = out/'selected'/'candidate.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out/'checkpoints'/selected/'candidate.pt', path)
            export.update(checkpoint=str(path), sha256=digest_file(path), branch=STAGES[selected][0])
        write_json(out/'selected'/'deployment.json', export)
        print(f'Setup + solve selection: {selected}; see {target / "selection.json"}', flush=True)
        return result


def freeze(output, **protocol):
    protocol = _validate_protocol(protocol, final=True)
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if 'multiple' not in protocol['regimes'] or settings['target_rhs_count'] not in protocol['rhs_counts']:
        raise ValueError('Final protocol must include the setup-inclusive target RHS workload')
    with _lease(out):
        _ensure_development_open(out)
        arms = _arms(out, cfg, baseline)
        evidence = _read(out/'benchmark_evidence.json')
        if evidence['checkpoints'] != {n: a['model'].signature() for n, a in arms.items() if a['model']}:
            raise ValueError('Checkpoint changed after validation')
        files = {}
        for key, record in evidence['files'].items():
            if digest_file(record['path']) != record['sha256']:
                raise ValueError('Validation evidence changed')
            files[key] = record['path']
        if _read(files['selection.json']) != evidence['selection']:
            raise ValueError('Stored selection differs from validation evidence')
        selected = evidence['selection'].get('recommended_deployment_arm')
        deployment_path = out/'selected/deployment.json'
        deployment = _read(deployment_path)
        if (deployment.get('arm') != selected
                or deployment.get('selection') != str(out/'benchmarks/validation/selection.json')):
            raise ValueError('Exported deployment differs from validation selection')
        files.update(configuration=out/'configuration.json', manifest=out/'em_line_manifest.json',
                     evidence=out/'benchmark_evidence.json', deployment=deployment_path,
                     initial=out/'initial.pt', initial_manifest=out/'initial.json')
        checkpoints = {name: out/'checkpoints'/name/'candidate.pt' for name in STAGES}
        for name in STAGES:
            for key in ('training_manifest.json', 'status.json'):
                files[f'{name}:{key}'] = out/'checkpoints'/name/key
        if selected in STAGES:
            exported = out/'selected/candidate.pt'
            if (deployment.get('checkpoint') != str(exported)
                    or deployment.get('branch') != STAGES[selected][0]
                    or deployment.get('sha256') != digest_file(exported)
                    or digest_file(exported) != digest_file(checkpoints[selected])):
                raise ValueError('Selected checkpoint copy differs from validated stage')
            checkpoints['selected_deployment'] = exported
        elif deployment.get('checkpoint') is not None:
            raise ValueError('Classical/nonselection cannot export a learned checkpoint')
        evaluation = dict(version=VERSION, solver=cfg.to_dict(), arms=list(arms),
            selected_arm=selected, target_rhs_count=settings['target_rhs_count'], **protocol)
        return freeze_research(out/'data', checkpoints, out/'fixed_classical.json', files, _sources(), evaluation)


def final(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    path = out/'data/research_freeze.json'
    evaluation = _read(path)['config']
    arms = _arms(out, cfg, baseline)
    if (evaluation['version'] != VERSION or evaluation['solver'] != json_safe(cfg.to_dict())
            or evaluation['arms'] != list(arms) or evaluation['target_rhs_count'] != settings['target_rhs_count']):
        raise ValueError('Frozen EM-line evaluation changed')
    with claim_final_evaluation(out/'data', path, evaluation, resume=resume) as claim:
        splits, _ = materialize_final_data(out/'data', claim, baseline)
        outputs = []
        for name in evaluation['splits']:
            target = out/'final'/name
            result = evaluate_research(splits[name], arms, cfg, baseline, target, reference_arm='fixed_C',
                resume=resume and (target/'run_manifest.json').exists(),
                **{key: evaluation[key] for key in ('repeats', 'warmups', 'regimes', 'rhs_counts')})
            report(result, target)
            outputs.extend(target/key for key in ('raw_results.json', 'cycle_diagnostics.json'))
        complete_final_evaluation(out/'data', claim, outputs)
    print('Final/OOD EM-line evaluation sealed; no final-set model selection.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'train', 'benchmark', 'freeze', 'final'):
        p = sub.add_parser(name)
        p.add_argument('--run-dir', required=True)
        if name != 'freeze':
            p.add_argument('--resume', action='store_true')
        if name == 'prepare':
            p.add_argument('--config', default='configs/v6_7_em_line_research.json')
        if name in ('benchmark', 'freeze'):
            p.add_argument('--repeats', type=int, default=3)
            p.add_argument('--warmups', type=int, default=1)
            p.add_argument('--rhs-counts', nargs='+', type=int, default=[1, 4, 16, 64])
            p.add_argument('--regimes', nargs='+', choices=('cold', 'warm', 'multiple', 'warm_multiple'),
                           default=['cold', 'warm', 'multiple'])
        if name == 'freeze':
            p.add_argument('--splits', nargs='+', choices=HOLDOUT, default=list(HOLDOUT))
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        return prepare(args.config, args.run_dir, resume=args.resume)
    if args.command in ('train', 'final'):
        return globals()[args.command](args.run_dir, resume=args.resume)
    protocol = dict(repeats=args.repeats, warmups=args.warmups, rhs_counts=args.rhs_counts, regimes=args.regimes)
    if args.command == 'freeze':
        return freeze(args.run_dir, splits=args.splits, **protocol)
    return benchmark(args.run_dir, resume=args.resume, **protocol)
