"""Residual U-net P teacher -> gated CNN distillation -> measured MG study.

The direct and distilled students have identical initialization, architecture,
training data and update budgets. Teacher cost is offline; deployment timing
includes student generation, Galerkin construction and numerical solves.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math

import numpy as np
import torch

from ..provenance import write_json, json_safe
from .banks import Stats, hybrid_cycle
from ..hierarchy import classical_cycle
from .models import Components
from .strong import PreparedStrongMG, load_strong_rules
from .spatial import SpatialState
from .limited import digest_file
from .three_pillars import (calibrate as calibrate_classical, _load_run, _development,
                           _ensure_development_open, _read, _report, _sources,
                           _validate_settings)
from .research_training import train_expert, sample_config
from .research_runtime import offline_teacher_generation
from .research_evaluation import evaluate_research, manufactured_rhs
from .research_data import (freeze_research, claim_final_evaluation,
                           materialize_final_data, complete_final_evaluation)
from .unet_transfer import ResidualUNetTransferTeacher, CompactTransferStudent

VERSION = 'residual-unet-transfer-study-v1'
STAGES = ('direct', 'teacher', 'distilled')
ARMS = {'direct': 'direct_CNN', 'teacher': 'teacher', 'distilled': 'distilled_CNN'}


def validate_settings(settings):
    cfg = _validate_settings(settings)
    study = settings.get('unet_transfer', {})
    if study.get('version') != VERSION:
        raise ValueError('Use a new residual U-net transfer study configuration')
    if cfg.use_smoother or not cfg.use_transfer or cfg.transfer_levels != 'all':
        raise ValueError('This study learns all nonterminal P levels with classical smoothing')
    if cfg.branch != 'H_P' or cfg.use_learned_controller or cfg.spatial:
        raise ValueError('This study requires forced H_P without a learned policy/spatial gate')
    if study.get('reference') != 'frozen_parent' or settings['support'] != 'support_preserving':
        raise ValueError('Teacher and students must share frozen-parent sparse support')
    if study.get('parameterization', 'softmax') not in ('softmax', 'affine'):
        raise ValueError('Unknown P parameterization')
    for key in ('student_hidden', 'teacher_hidden', 'teacher_depth', 'teacher_blocks'):
        value = study[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    for name in ('training', 'teacher_training'):
        train = settings[name]
        if int(train['updates']) < 2 or int(train.get('prefix_cycles', 0)) < 1:
            raise ValueError('Require at least two updates and one learned V-cycle')
    train = settings['training']
    if not 1 <= int(train['kd_updates']) < int(train['updates']):
        raise ValueError('KD needs both distillation and task-only fine-tuning updates')
    if train.get('distillation_mode') != 'cycle_action' or float(train['lambda_kd']) <= 0:
        raise ValueError('Use positive full-cycle-action KD')
    gate = study['teacher_gate']
    if not 0 < float(gate['minimum_residual_reduction']) < 1:
        raise ValueError('Teacher must improve contraction by a positive declared margin')
    if int(gate['cycles']) < 1 or int(gate['rhs_count']) < 1 or int(gate['min_cases']) < 1:
        raise ValueError('Invalid teacher gate sampling')
    if int(gate['repeats']) < 1 or int(gate['warmups']) < 0:
        raise ValueError('Invalid teacher numerical validation')
    if float(study['deployment']['minimum_speedup']) <= 1:
        raise ValueError('Deployment needs a strictly faster setup-inclusive time')
    return cfg


def load(output):
    out, settings, cfg = _load_run(output)
    validate_settings(settings)
    return out, settings, cfg, load_strong_rules(out / 'selector_rules.json')


def calibrate(config, output, *, resume=False):
    validate_settings(_read(config))
    return calibrate_classical(config, output, resume=resume)


def initial_model(settings, stage):
    if stage not in STAGES:
        raise ValueError('Unknown training stage')
    study = settings['unet_transfer']
    model = Components.create(hidden=study['student_hidden'], seed=int(settings['seed']))
    common = dict(support=settings['support'], reference=study['reference'],
                  parameterization=study.get('parameterization', 'softmax'),
                  complexity_caps=settings['complexity_caps'])
    if stage == 'teacher':
        model.transfer = ResidualUNetTransferTeacher(hidden=study['teacher_hidden'],
            depth=study['teacher_depth'], blocks_per_stage=study['teacher_blocks'], **common)
    else:
        model.transfer = CompactTransferStudent(hidden=study['student_hidden'], **common)
    model.metadata.update(research_version=VERSION, role='teacher' if stage == 'teacher' else 'student',
        initialization_seed=int(settings['seed']), final_test_seen=False,
        transfer_only=True, attention=False, trained_against='frozen strong C*(A)')
    model.mark_policy_stale('transfer research has no adaptive deployment policy')
    return model.eval()


def _checkpoint(out, stage):
    return out / 'checkpoints' / stage / 'candidate.pt'


def expert(out, stage, rules):
    path = _checkpoint(out, stage)
    status = _read(path.parent / 'status.json')
    if not status.get('all_updates_completed') or status.get('skipped'):
        raise ValueError(f'{stage}: incomplete/skipped training updates cannot qualify')
    if status.get('feasible_task_updates', 0) < 1:
        raise ValueError(f'{stage}: no feasible learned-P task update')
    if digest_file(path) != status['checkpoint_sha256']:
        raise ValueError(f'{stage}: checkpoint changed after training')
    model = Components.load(path)
    if model.metadata.get('training_rules_digest') != rules.digest():
        raise ValueError('Checkpoint uses different frozen strong classical rules')
    if model.metadata.get('training_branch') != 'H_P':
        raise ValueError('Expected a transfer-only checkpoint')
    return model


def _gate_binding(out, cfg, rules):
    return dict(version=VERSION, rules_digest=rules.digest(), config=cfg.to_dict(),
        direct_sha256=digest_file(_checkpoint(out, 'direct')),
        teacher_sha256=digest_file(_checkpoint(out, 'teacher')))


def verified_gate(out, cfg, rules):
    gate = _read(out / 'teacher_gate.json')
    if gate['binding'] != json_safe(_gate_binding(out, cfg, rules)):
        raise ValueError('Teacher/direct checkpoint or solver changed after teacher gate')
    for record in gate['evidence'].values():
        if digest_file(record['path']) != record['sha256']:
            raise ValueError('Teacher gate evidence changed')
    return gate


def train(output, stage='all', *, resume=False):
    out, settings, cfg, rules = load(output)
    _ensure_development_open(out)
    if (out / 'teacher_gate.json').exists() and stage in ('direct', 'teacher', 'all'):
        raise ValueError('Teacher already assessed; use a new run to change experts')
    examples, _, _ = _development(out, settings, rules)
    if stage == 'all':
        return {name: train(output, name, resume=resume and
                    (out / 'checkpoints' / name / 'training_manifest.json').exists())
                for name in ('direct', 'teacher')}
    teacher = None
    if stage == 'distilled':
        gate = verified_gate(out, cfg, rules)
        decision_path = out / 'distillation_decision.json'
        if not gate['adopted']:
            if _checkpoint(out, 'distilled').exists():
                raise ValueError('Distilled checkpoint exists despite rejected teacher')
            decision = dict(status='skipped', reason='teacher_contraction_gate_not_passed',
                gate_sha256=digest_file(out / 'teacher_gate.json'), checkpoint_created=False)
            write_json(decision_path, decision)
            print('KD skipped: teacher did not pass the predeclared contraction gate.', flush=True)
            return decision
        teacher = expert(out, 'teacher', rules)
    initial = initial_model(settings, stage)
    # Fresh, identical initialization and total update budget for direct and KD.
    training = settings['teacher_training'] if stage == 'teacher' else settings['training']
    target = out / 'checkpoints' / stage
    model, status = train_expert(initial, examples, cfg, rules, training, target,
        branch='H_P', teacher=teacher, resume=resume)
    if not status['all_updates_completed'] or status['skipped'] or not status['feasible_task_updates']:
        raise RuntimeError(f'{stage}: training did not complete usable learned-P updates; see status.json')
    if stage == 'distilled':
        records = _read(target / 'training.json')
        applied = sum(r.get('kd_status') == 'applied' for r in records)
        if applied < 1:
            raise RuntimeError('No valid teacher action was distilled')
        write_json(out / 'distillation_decision.json', dict(status='complete',
            gate_sha256=digest_file(out / 'teacher_gate.json'), applied_kd_updates=applied,
            checkpoint_sha256=digest_file(target / 'candidate.pt'),
            task_only_updates=sum(r['phase'] == 'full_cycle_task' for r in records),
            identical_direct_student_initialization=True, equal_student_update_budget=True,
            teacher_in_deployment=False))
    return status


def raw_contractions(examples, models, cfg, rules, *, cycles, rhs_count):
    """Numerical bank V-cycles with no safeguard replacement or tolerance stop.

    A failed sparse proposal is a failed gate row, never a classical substitute.
    Ratios are true Euclidean residual norms after the same number of cycles.
    """
    rows = {}
    for example in examples:
        row = {}
        rhs, _ = manufactured_rhs(example, rhs_count)
        for name in ('strong_C', 'direct_CNN', 'teacher'):
            scfg = sample_config(example, cfg, rules, 'C' if name == 'strong_C' else 'H_P')
            model = None if name == 'strong_C' else models[name].frozen_inference_copy()
            try:
                with offline_teacher_generation():
                    prepared = PreparedStrongMG(example.a, example.n, model, scfg, rules)
                    stats = Stats()
                    root = prepared.classical if name == 'strong_C' else prepared.ensure_branch('H_P', stats)
                    histories = []
                    for b in rhs:
                        norm0 = float(np.linalg.norm(b))
                        if not np.isfinite(norm0) or norm0 <= 0:
                            raise FloatingPointError('Invalid residual probe')
                        x = np.zeros_like(b); history = []
                        for step in range(cycles):
                            if name == 'strong_C':
                                x = classical_cycle(root, x, b, scfg.mg, Stats())
                            else:
                                x = hybrid_cycle(root, x, b, scfg, Stats(),
                                                 SpatialState(model, scfg), step, True)
                            rho = float(np.linalg.norm(b - example.a @ x) / norm0)
                            if not np.isfinite(rho):
                                raise FloatingPointError('Nonfinite raw V-cycle residual')
                            history.append(rho)
                        histories.append(history)
                    contraction = float(np.exp(np.mean(np.log(np.maximum(
                        np.array(histories)[:, -1], 1e-300)))))
                    row[name] = dict(feasible=True, contraction=contraction, history=histories,
                        fixed_cycles=cycles, rhs_count=rhs_count, fallback_allowed=False)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                row[name] = dict(feasible=False, contraction=None, error=str(error), fallback_allowed=False)
        rows[example.name] = row
    return rows


def gate_decision(raw, numerical, *, min_reduction=.05, min_cases=5):
    if not 0 < min_reduction < 1 or min_cases < 1:
        raise ValueError('Invalid teacher gate thresholds')
    measured = {row['example']['name']: row for row in numerical['rows']}
    if not raw or set(raw) != set(measured) or len(measured) != len(numerical['rows']):
        raise ValueError('Teacher gate needs identical unique validation cases')
    names = ('strong_C', 'direct_CNN', 'teacher')
    failures, common, missing_use = [], [], []
    invalid = []
    for case, row in raw.items():
        for name in names:
            value = row[name].get('contraction')
            if (not row[name].get('feasible') or not isinstance(value, (float, int))
                    or not np.isfinite(value) or value <= 0):
                invalid.append(dict(case=case, arm=name))
        runs = {name: measured[case]['runs'][name]['cold']['1'] for name in names}
        ok = {name: bool(runs[name]) and all(r.get('successful', False) for r in runs[name]) for name in names}
        if (ok['strong_C'] or ok['direct_CNN']) and not ok['teacher']:
            failures.append(case)
        if all(ok.values()):
            common.append(case)
            if not all(r.get('actual_neural_used', False) for r in runs['teacher']):
                missing_use.append(case)
    threshold = 1 / (1 - min_reduction)
    gains = {}
    for reference in ('strong_C', 'direct_CNN'):
        values = [math.log(raw[c][reference]['contraction']) - math.log(raw[c]['teacher']['contraction'])
                  for c in common] if common and not invalid else []
        gains[reference] = float(np.exp(np.mean(values))) if values else None
    passed = (not invalid and not failures and not missing_use and len(common) >= min_cases
              and all(g is not None and np.isfinite(g) and g >= threshold for g in gains.values()))
    return dict(adopted=bool(passed), metric='fixed-cycle true residual ratio',
        improvement_vs=gains, required_improvement=threshold, min_cases=min_cases,
        common_case_ids=common, new_teacher_failure_case_ids=failures,
        infeasible_or_nonfinite=invalid, missing_actual_teacher_use_case_ids=missing_use,
        teacher_cost_scope='offline only; not a deployment speedup', performance_certified=False)


def teacher_check(output, *, resume=False):
    out, settings, cfg, rules = load(output)
    _ensure_development_open(out)
    if (out / 'teacher_gate.json').exists():
        if not resume:
            raise FileExistsError('Teacher gate already fixed; use --resume to read it')
        return verified_gate(out, cfg, rules)
    _, validation, _ = _development(out, settings, rules)
    models = {ARMS[s]: expert(out, s, rules) for s in ('direct', 'teacher')}
    gate = settings['unet_transfer']['teacher_gate']
    target = out / 'teacher_validation'
    specs = {'strong_C': dict(model=None, branch='C'),
        'direct_CNN': dict(model=models['direct_CNN'], branch='H_P'),
        'teacher': dict(model=models['teacher'], branch='H_P', training_only=True)}
    result = evaluate_research(validation, specs, cfg, rules, target,
        repeats=int(gate['repeats']), warmups=int(gate['warmups']), regimes=('cold',), rhs_counts=(1,),
        resume=resume and (target / 'run_manifest.json').exists())
    raw = raw_contractions(validation, models, cfg, rules, cycles=int(gate['cycles']), rhs_count=int(gate['rhs_count']))
    raw_path = out / 'teacher_raw_contractions.json'; write_json(raw_path, raw)
    decision = gate_decision(raw, result, min_reduction=float(gate['minimum_residual_reduction']),
                             min_cases=int(gate['min_cases']))
    evidence = {name: dict(path=str(path), sha256=digest_file(path)) for name, path in
                (('raw', raw_path), ('numerical', target / 'raw_results.json'))}
    decision.update(binding=_gate_binding(out, cfg, rules), evidence=evidence)
    write_json(out / 'teacher_gate.json', decision)
    print('Teacher gate:', 'PASS' if decision['adopted'] else 'REJECT', decision['improvement_vs'], flush=True)
    return decision


def _student_arms(out, cfg, rules):
    gate = verified_gate(out, cfg, rules)
    arms = {'strong_C': dict(model=None, branch='C'),
            'direct_CNN': dict(model=expert(out, 'direct', rules), branch='H_P')}
    if gate['adopted']:
        decision = _read(out / 'distillation_decision.json')
        if (decision.get('status') != 'complete' or
                decision['gate_sha256'] != digest_file(out / 'teacher_gate.json') or
                decision['checkpoint_sha256'] != digest_file(_checkpoint(out, 'distilled'))):
            raise ValueError('Complete, gate-bound distillation required before deployment comparison')
        arms['distilled_CNN'] = dict(model=expert(out, 'distilled', rules), branch='H_P')
    elif _checkpoint(out, 'distilled').exists():
        raise ValueError('Rejected teacher must not have a deployment student checkpoint')
    return arms


def deployment_decisions(result, *, minimum_speedup=1.03, min_cases=5):
    """Named setup-inclusive cohorts; warm-only wins cannot qualify."""
    decisions = {}
    for regime in ('cold', 'multiple'):
        for count, arms in result['summary'].get(regime, {}).items():
            for name in ('direct_CNN', 'distilled_CNN'):
                if name not in arms:
                    continue
                entry = arms[name]
                common, ratios, neural = [], [], []
                direct_ratios = []
                for row in result['rows']:
                    runs = row['runs'][name][regime][count]
                    base = row['runs']['strong_C'][regime][count]
                    if all(r['successful'] for r in runs + base):
                        common.append(row['example']['name'])
                        ratios.append(np.median([r['wall_seconds'] for r in base]) /
                                      np.median([r['wall_seconds'] for r in runs]))
                        neural.append(all(r['actual_neural_used'] for r in runs))
                    if name == 'distilled_CNN':
                        direct = row['runs']['direct_CNN'][regime][count]
                        if all(r['successful'] for r in runs + direct):
                            direct_ratios.append(np.median([r['wall_seconds'] for r in direct]) /
                                                 np.median([r['wall_seconds'] for r in runs]))
                gain = float(np.exp(np.mean(np.log(ratios)))) if ratios else None
                ci = entry['ci95']
                time_pass = (len(common) >= min_cases and not entry['new_failure_case_ids']
                    and all(neural) and gain is not None and gain >= minimum_speedup)
                confidence = bool(ci and ci[0] is not None and ci[0] > 1.)
                accuracy = not entry['accuracy_worse_vs_strong_case_ids']
                decisions[f'{regime}/rhs{count}/{name}'] = dict(
                    observed_time_target_met=bool(time_pass), confidence_supports_speedup=confidence,
                    accuracy_check_passed=accuracy, qualifies=bool(time_pass and confidence and accuracy),
                    geometric_speedup_vs_strong=gain, ci95=ci, common_case_ids=common,
                    geometric_speedup_vs_direct=float(np.exp(np.mean(np.log(direct_ratios)))) if direct_ratios else None,
                    new_failure_case_ids=entry['new_failure_case_ids'],
                    setup_included=True, teacher_deployed=False, performance_certified=False)
    return decisions


def benchmark(output, *, tag='validation', resume=False, **protocol):
    out, settings, cfg, rules = load(output)
    _ensure_development_open(out)
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('Use a simple benchmark tag')
    _, validation, _ = _development(out, settings, rules)
    target = out / 'benchmarks' / tag
    result = evaluate_research(validation, _student_arms(out, cfg, rules), cfg, rules, target,
                              resume=resume, **protocol)
    _report(result, target)
    decisions = deployment_decisions(result, **settings['unet_transfer']['deployment'])
    write_json(target / 'deployment_decisions.json', decisions)
    write_json(out / 'benchmark_evidence.json', dict(
        report=str(target / 'raw_results.json'), report_sha256=digest_file(target / 'raw_results.json'),
        decisions=str(target / 'deployment_decisions.json'),
        decisions_sha256=digest_file(target / 'deployment_decisions.json'),
        checkpoints={name: arm['model'].signature() for name, arm in _student_arms(out, cfg, rules).items() if arm['model']},
        performance_certified=False))
    return decisions


def freeze(output, **protocol):
    out, settings, cfg, rules = load(output)
    _ensure_development_open(out)
    if not set(protocol.get('regimes', ())) & {'cold', 'multiple'}:
        raise ValueError('Final evaluation must include setup-inclusive cold or multiple workloads')
    arms = _student_arms(out, cfg, rules)
    evidence = _read(out / 'benchmark_evidence.json')
    for key in ('report', 'decisions'):
        if digest_file(evidence[key]) != evidence[key + '_sha256']:
            raise ValueError('Deployment benchmark evidence changed')
    signatures = {name: arm['model'].signature() for name, arm in arms.items() if arm['model']}
    if signatures != evidence['checkpoints']:
        raise ValueError('Student changed after validation')
    paths = {'teacher_gate': out / 'teacher_gate.json', 'teacher_checkpoint': _checkpoint(out, 'teacher'),
             'validation': evidence['report'], 'decisions': evidence['decisions'],
             'benchmark_evidence': out / 'benchmark_evidence.json'}
    paths.update({name: value['path'] for name, value in verified_gate(out, cfg, rules)['evidence'].items()})
    if 'distilled_CNN' in arms:
        paths['distillation_decision'] = out / 'distillation_decision.json'
    checkpoints = {ARMS[s]: _checkpoint(out, s) for s in ('direct', 'distilled') if ARMS[s] in arms}
    evaluation = dict(version=VERSION, solver=cfg.to_dict(), arms=list(arms),
                      deployment=settings['unet_transfer']['deployment'], **protocol)
    return freeze_research(out / 'data', checkpoints, out / 'selector_rules.json', paths, _sources(), evaluation)


def final(output, *, resume=False):
    out, _, cfg, rules = load(output)
    path = out / 'data/research_freeze.json'; evaluation = _read(path)['config']
    if evaluation['version'] != VERSION or evaluation['solver'] != cfg.to_dict():
        raise ValueError('Wrong frozen evaluation')
    arms = _student_arms(out, cfg, rules)
    if list(arms) != evaluation['arms']:
        raise ValueError('Frozen student arms changed')
    with claim_final_evaluation(out / 'data', path, evaluation, resume=resume) as claim:
        splits, _ = materialize_final_data(out / 'data', claim, rules)
        outputs = []
        for name, examples in splits.items():
            target = out / 'final' / name
            result = evaluate_research(examples, arms, cfg, rules, target,
                resume=resume and (target / 'run_manifest.json').exists(),
                **{k: evaluation[k] for k in ('repeats', 'warmups', 'regimes', 'rhs_counts')})
            _report(result, target)
            write_json(target / 'deployment_decisions.json', deployment_decisions(result, **evaluation['deployment']))
            outputs.extend([target / 'raw_results.json', target / 'deployment_decisions.json'])
        complete_final_evaluation(out / 'data', claim, outputs)
    print('Final/OOD evaluation sealed; read measured outcomes, no automatic speedup claim.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('calibrate', 'train', 'teacher-check', 'distill', 'benchmark', 'freeze', 'final'):
        p = sub.add_parser(name); p.add_argument('--run-dir', required=True)
        if name != 'freeze': p.add_argument('--resume', action='store_true')
        if name == 'calibrate': p.add_argument('--config', default='configs/v6_7_unet_transfer_smoke.json')
        if name == 'train': p.add_argument('--stage', choices=('all', 'direct', 'teacher'), default='all')
        if name == 'benchmark': p.add_argument('--tag', default='validation')
        if name in ('benchmark', 'freeze'):
            p.add_argument('--repeats', type=int, default=5)
            p.add_argument('--warmups', type=int, default=1)
            p.add_argument('--rhs-counts', nargs='+', type=int, default=[1, 4, 16, 64])
            p.add_argument('--regimes', nargs='+', choices=('cold', 'warm', 'multiple', 'warm_multiple'),
                           default=['cold', 'warm', 'multiple'])
    a = parser.parse_args(argv)
    if a.command == 'calibrate': return calibrate(a.config, a.run_dir, resume=a.resume)
    if a.command == 'train': return train(a.run_dir, a.stage, resume=a.resume)
    if a.command == 'teacher-check': return teacher_check(a.run_dir, resume=a.resume)
    if a.command == 'distill': return train(a.run_dir, 'distilled', resume=a.resume)
    if a.command == 'final': return final(a.run_dir, resume=a.resume)
    protocol = dict(repeats=a.repeats, warmups=a.warmups, rhs_counts=a.rhs_counts, regimes=a.regimes)
    if a.command == 'freeze': return freeze(a.run_dir, **protocol)
    return benchmark(a.run_dir, tag=a.tag, resume=a.resume, **protocol)
