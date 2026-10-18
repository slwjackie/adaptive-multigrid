"""Residual U-net P teacher -> gated CNN distillation -> measured MG study.

The direct and distilled students have identical initialization, architecture,
training data and update budgets. Teacher cost is offline; deployment timing
includes student generation, Galerkin construction and numerical solves.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch

from ..provenance import write_json, json_safe, hardware_environment
from .banks import Stats, hybrid_cycle
from ..hierarchy import classical_cycle
from .models import Components
from .solver import PreparedAdaptiveMG
from .config import AdaptiveConfig
from .fixed_classical import FixedClassicalBaseline, load_baseline
from .spatial import SpatialState
from .limited import digest_file, initialize_timing_runtime
from .research_training import train_expert, sample_config
from .research_runtime import offline_teacher_generation
from .research_evaluation import evaluate_research, manufactured_rhs
from .research_data import (freeze_research, claim_final_evaluation,
                           materialize_final_data, complete_final_evaluation,
                           make_research_plan, prepare_development_data,
                           load_development_data, _hash)
from .unet_transfer import ResidualUNetTransferTeacher, CompactTransferStudent

VERSION = 'residual-unet-fixed-em-study-v2'
PROJECT = Path(__file__).resolve().parents[3]
STAGES = ('direct', 'teacher', 'distilled')
ARMS = {'direct': 'direct_CNN', 'teacher': 'teacher', 'distilled': 'distilled_CNN'}


def validate_settings(settings):
    if settings.get('version') != VERSION:
        raise ValueError('Use a new fixed-classical U-net configuration/run')
    removed = {'bank', 'calibration_sizes', 'calibration_per_family', 'calibration_repeats',
               'calibration_rhs', 'calibration_regime', 'minimum_leaf_cases', 'cycle_margin'}
    if removed & settings.keys():
        raise ValueError('Selector calibration settings are not part of this fixed-baseline study')
    cfg = AdaptiveConfig.from_dict(settings['solver'])
    # Strategy suffixes override these fields in MGConfig: reject contradictions
    # in the source JSON instead of silently correcting the declared experiment.
    raw_mg = settings['solver']['mg']
    for key in ('strategy_name', 'em_iterations', 'em_fixed_iterations', 'em_rtol', 'pre_steps', 'post_steps'):
        if key not in raw_mg or raw_mg[key] != getattr(cfg.mg, key):
            raise ValueError(f'Explicit, consistent fixed baseline setting required: {key}')
    FixedClassicalBaseline.from_config(cfg)
    if cfg.application != 'replace' or not cfg.lazy or cfg.mode != 'research':
        raise ValueError('Use lazy replacement in research mode')
    if settings['complexity_caps'].get('complexity_reference') != 'parent':
        raise ValueError('P complexity caps must use the fixed classical parent')
    for key in ('training_sizes', 'validation_sizes'):
        values = settings[key]
        if (not values or len(set(values)) != len(values) or
                any(isinstance(n, bool) or not isinstance(n, int) or n < 3 or n & (n + 1) for n in values)):
            raise ValueError('Use distinct 2**L-1 sizes >=3')
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
    out = Path(output).resolve()
    settings = _read(out / 'configuration.json')
    cfg = validate_settings(settings)
    _initialize(settings)
    baseline = load_baseline(out / 'fixed_classical.json')
    if not isinstance(baseline, FixedClassicalBaseline):
        raise ValueError('This study accepts only a fixed classical baseline')
    baseline.validate_config(cfg)
    if _read(out / 'unet_transfer_manifest.json') != json_safe(_manifest(settings, baseline)):
        raise ValueError('Sources/settings/baseline/environment changed; start a new run')
    return out, settings, cfg, baseline


def _read(path):
    return json.loads(Path(path).read_text())


def _sources():
    return sorted([p for p in (PROJECT / 'src').rglob('*') if p.suffix in ('.py', '.cpp')]
                  + [PROJECT / 'scripts/run_v6_7_unet_transfer_study.py'])


def _initialize(settings):
    torch.set_num_threads(int(settings.get('torch_threads', 1)))
    initialize_timing_runtime()


def _manifest(settings, baseline):
    return dict(version=VERSION, settings_digest=_hash(settings), baseline_digest=baseline.digest(),
        source_digest=_hash({str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in _sources()}), hardware=hardware_environment(refresh=True))


def _ensure_development_open(out):
    if (out / 'data/research_freeze.json').exists() or (out / 'data/final_claim.json').exists():
        raise ValueError('Run already frozen; start a new development run')


def _development(out, settings, baseline):
    data = out / 'data'
    if (data / 'development_manifest.json').exists():
        return load_development_data(data, baseline)
    plan = make_research_plan(smoke=bool(settings['smoke']), seed=int(settings['seed']) + 700001)
    for name, key, per in (('train', 'training_sizes', 'training_per_family'),
                            ('validation', 'validation_sizes', 'validation_per_family')):
        spec = plan['splits'][name]
        spec.update(sizes=settings[key], per_family=int(settings[per]))
        spec['count'] = len(spec['sizes']) * len(spec['families']) * spec['per_family']
    roots = settings.get('historical_roots') or [str(out.parent)]
    return prepare_development_data(data, baseline, plan=plan, historical_roots=roots)


def prepare(config, output, *, resume=False):
    """Pin a single baseline and fresh data; no portfolio search or calibration."""
    settings = _read(config)
    cfg = validate_settings(settings)
    baseline = FixedClassicalBaseline.from_config(cfg)
    _initialize(settings)
    out = Path(output).resolve()
    path = out / 'unet_transfer_manifest.json'
    if path.exists():
        if not resume:
            raise FileExistsError('Run exists; use --resume or a new run directory')
        _, stored, _, saved = load(out)
        if settings != stored or baseline.digest() != saved.digest():
            raise ValueError('Prepare resume settings changed')
        _ensure_development_open(out)
    else:
        if resume:
            raise FileNotFoundError('No fixed-classical run to resume')
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Prepare needs an empty run directory')
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'configuration.json', settings)
        write_json(out / 'fixed_classical.json', baseline.to_dict())
        write_json(path, _manifest(settings, baseline))
    train_data, validation, _ = _development(out, settings, baseline)
    print('Fixed classical prepared:', cfg.mg.strategy_name,
          f'; train={len(train_data)}, validation={len(validation)}; no selector', flush=True)
    return baseline.to_dict()


def _report(result, out):
    inclusive = {}
    for row in result['table']:
        key = f"{row['regime']}:{row['rhs_count']}:{row['arm']}"
        item = inclusive.setdefault(key, dict(total_case_median_wall_seconds=0., failures=0, cases=0))
        item['total_case_median_wall_seconds'] += row['wall_seconds'] or 0.
        item['failures'] += int(not row['successful'])
        item['cases'] += 1
    write_json(out / 'failure_inclusive_wall.json', dict(entries=inclusive,
        scope='sum of measured case medians including capped failures; not a successful time-to-solution ranking'))
    for regime, counts in result['summary'].items():
        for count, arms in counts.items():
            for name, values in arms.items():
                print(regime, 'RHS=' + count, name, f"{values['successes']}/{values['total']}",
                      'speedup_vs_fixed=', values['geometric_speedup'], flush=True)


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
        transfer_only=True, attention=False, trained_against='fixed line_alt + energy minimization + full coarsening')
    model.mark_policy_stale('transfer research has no adaptive deployment policy')
    return model.eval()


def _checkpoint(out, stage):
    return out / 'checkpoints' / stage / 'candidate.pt'


def expert(out, stage, baseline):
    path = _checkpoint(out, stage)
    status = _read(path.parent / 'status.json')
    if not status.get('all_updates_completed') or status.get('skipped'):
        raise ValueError(f'{stage}: incomplete/skipped training updates cannot qualify')
    if status.get('feasible_task_updates', 0) < 1:
        raise ValueError(f'{stage}: no feasible learned-P task update')
    if digest_file(path) != status['checkpoint_sha256']:
        raise ValueError(f'{stage}: checkpoint changed after training')
    model = Components.load(path)
    if model.metadata.get('training_baseline_digest') != baseline.digest():
        raise ValueError('Checkpoint uses different fixed classical baseline')
    if model.metadata.get('training_branch') != 'H_P':
        raise ValueError('Expected a transfer-only checkpoint')
    return model


def _gate_binding(out, cfg, baseline):
    return dict(version=VERSION, baseline_digest=baseline.digest(), config=cfg.to_dict(),
        direct_sha256=digest_file(_checkpoint(out, 'direct')),
        teacher_sha256=digest_file(_checkpoint(out, 'teacher')))


def verified_gate(out, cfg, baseline):
    gate = _read(out / 'teacher_gate.json')
    if gate['binding'] != json_safe(_gate_binding(out, cfg, baseline)):
        raise ValueError('Teacher/direct checkpoint or solver changed after teacher gate')
    for record in gate['evidence'].values():
        if digest_file(record['path']) != record['sha256']:
            raise ValueError('Teacher gate evidence changed')
    return gate


def train(output, stage='all', *, resume=False):
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if (out / 'teacher_gate.json').exists() and stage in ('direct', 'teacher', 'all'):
        raise ValueError('Teacher already assessed; use a new run to change experts')
    examples, _, _ = _development(out, settings, baseline)
    if stage == 'all':
        return {name: train(output, name, resume=resume and
                    (out / 'checkpoints' / name / 'training_manifest.json').exists())
                for name in ('direct', 'teacher')}
    teacher = None
    if stage == 'distilled':
        gate = verified_gate(out, cfg, baseline)
        decision_path = out / 'distillation_decision.json'
        if not gate['adopted']:
            if _checkpoint(out, 'distilled').exists():
                raise ValueError('Distilled checkpoint exists despite rejected teacher')
            decision = dict(status='skipped', reason='teacher_contraction_gate_not_passed',
                gate_sha256=digest_file(out / 'teacher_gate.json'), checkpoint_created=False)
            write_json(decision_path, decision)
            print('KD skipped: teacher did not pass the predeclared contraction gate.', flush=True)
            return decision
        teacher = expert(out, 'teacher', baseline)
    initial = initial_model(settings, stage)
    # Fresh, identical initialization and total update budget for direct and KD.
    training = settings['teacher_training'] if stage == 'teacher' else settings['training']
    target = out / 'checkpoints' / stage
    model, status = train_expert(initial, examples, cfg, baseline, training, target,
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


def raw_contractions(examples, models, cfg, baseline, *, cycles, rhs_count):
    """Numerical bank V-cycles with no safeguard replacement or tolerance stop.

    A failed sparse proposal is a failed gate row, never a classical substitute.
    Ratios are true Euclidean residual norms after the same number of cycles.
    """
    rows = {}
    for example in examples:
        row = {}
        rhs, _ = manufactured_rhs(example, rhs_count)
        for name in ('fixed_C', 'direct_CNN', 'teacher'):
            scfg = sample_config(example, cfg, baseline, 'C' if name == 'fixed_C' else 'H_P')
            model = None if name == 'fixed_C' else models[name].frozen_inference_copy()
            try:
                with offline_teacher_generation():
                    prepared = PreparedAdaptiveMG(example.a, example.n, model, scfg)
                    stats = Stats()
                    root = prepared.classical if name == 'fixed_C' else prepared.ensure_branch('H_P', stats)
                    histories = []
                    for b in rhs:
                        norm0 = float(np.linalg.norm(b))
                        if not np.isfinite(norm0) or norm0 <= 0:
                            raise FloatingPointError('Invalid residual probe')
                        x = np.zeros_like(b); history = []
                        for step in range(cycles):
                            if name == 'fixed_C':
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
    names = ('fixed_C', 'direct_CNN', 'teacher')
    failures, common, missing_use = [], [], []
    invalid = []
    for case, row in raw.items():
        for name in names:
            value = row[name].get('contraction')
            if (not row[name].get('feasible') or not isinstance(value, (float, int))
                    or not np.isfinite(value) or value <= 0
                    or row[name].get('fallback_allowed') is not False):
                invalid.append(dict(case=case, arm=name))
        runs = {name: measured[case]['runs'][name]['cold']['1'] for name in names}
        ok = {name: bool(runs[name]) and all(r.get('successful', False) for r in runs[name]) for name in names}
        if (ok['fixed_C'] or ok['direct_CNN']) and not ok['teacher']:
            failures.append(case)
        if all(ok.values()):
            common.append(case)
            if not all(r.get('actual_neural_used', False) for r in runs['teacher']):
                missing_use.append(case)
    threshold = 1 / (1 - min_reduction)
    gains = {}
    for reference in ('fixed_C', 'direct_CNN'):
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
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if (out / 'teacher_gate.json').exists():
        if not resume:
            raise FileExistsError('Teacher gate already fixed; use --resume to read it')
        return verified_gate(out, cfg, baseline)
    _, validation, _ = _development(out, settings, baseline)
    models = {ARMS[s]: expert(out, s, baseline) for s in ('direct', 'teacher')}
    gate = settings['unet_transfer']['teacher_gate']
    target = out / 'teacher_validation'
    specs = {'fixed_C': dict(model=None, branch='C', selector=False),
        'direct_CNN': dict(model=models['direct_CNN'], branch='H_P', selector=False),
        'teacher': dict(model=models['teacher'], branch='H_P', selector=False, training_only=True)}
    result = evaluate_research(validation, specs, cfg, baseline, target,
        reference_arm='fixed_C', repeats=int(gate['repeats']), warmups=int(gate['warmups']), regimes=('cold',), rhs_counts=(1,),
        resume=resume and (target / 'run_manifest.json').exists())
    raw = raw_contractions(validation, models, cfg, baseline, cycles=int(gate['cycles']), rhs_count=int(gate['rhs_count']))
    raw_path = out / 'teacher_raw_contractions.json'; write_json(raw_path, raw)
    decision = gate_decision(raw, result, min_reduction=float(gate['minimum_residual_reduction']),
                             min_cases=int(gate['min_cases']))
    evidence = {name: dict(path=str(path), sha256=digest_file(path)) for name, path in
                (('raw', raw_path), ('numerical', target / 'raw_results.json'))}
    decision.update(binding=_gate_binding(out, cfg, baseline), evidence=evidence)
    write_json(out / 'teacher_gate.json', decision)
    print('Teacher gate:', 'PASS' if decision['adopted'] else 'REJECT', decision['improvement_vs'], flush=True)
    return decision


def _student_arms(out, cfg, baseline):
    gate = verified_gate(out, cfg, baseline)
    arms = {'fixed_C': dict(model=None, branch='C', selector=False),
            'direct_CNN': dict(model=expert(out, 'direct', baseline), branch='H_P', selector=False)}
    if gate['adopted']:
        decision = _read(out / 'distillation_decision.json')
        if (decision.get('status') != 'complete' or
                decision['gate_sha256'] != digest_file(out / 'teacher_gate.json') or
                decision['checkpoint_sha256'] != digest_file(_checkpoint(out, 'distilled'))):
            raise ValueError('Complete, gate-bound distillation required before deployment comparison')
        arms['distilled_CNN'] = dict(model=expert(out, 'distilled', baseline), branch='H_P', selector=False)
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
                    base = row['runs']['fixed_C'][regime][count]
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
                accuracy = not entry['accuracy_worse_vs_fixed_case_ids']
                decisions[f'{regime}/rhs{count}/{name}'] = dict(
                    observed_time_target_met=bool(time_pass), confidence_supports_speedup=confidence,
                    accuracy_check_passed=accuracy, qualifies=bool(time_pass and confidence and accuracy),
                    geometric_speedup_vs_fixed=gain, ci95=ci, common_case_ids=common,
                    geometric_speedup_vs_direct=float(np.exp(np.mean(np.log(direct_ratios)))) if direct_ratios else None,
                    new_failure_case_ids=entry['new_failure_case_ids'],
                    setup_included=True,
                    time_scope='measured EM parent setup + CNN generation + Galerkin/factorization + safeguarded solve',
                    teacher_deployed=False, performance_certified=False)
    return decisions


def benchmark(output, *, tag='validation', resume=False, **protocol):
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('Use a simple benchmark tag')
    _, validation, _ = _development(out, settings, baseline)
    target = out / 'benchmarks' / tag
    result = evaluate_research(validation, _student_arms(out, cfg, baseline), cfg, baseline, target,
                              reference_arm='fixed_C', resume=resume, **protocol)
    _report(result, target)
    decisions = deployment_decisions(result, **settings['unet_transfer']['deployment'])
    write_json(target / 'deployment_decisions.json', decisions)
    write_json(out / 'benchmark_evidence.json', dict(
        report=str(target / 'raw_results.json'), report_sha256=digest_file(target / 'raw_results.json'),
        decisions=str(target / 'deployment_decisions.json'),
        decisions_sha256=digest_file(target / 'deployment_decisions.json'),
        checkpoints={name: arm['model'].signature() for name, arm in _student_arms(out, cfg, baseline).items() if arm['model']},
        performance_certified=False))
    return decisions


def freeze(output, **protocol):
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if not set(protocol.get('regimes', ())) & {'cold', 'multiple'}:
        raise ValueError('Final evaluation must include setup-inclusive cold or multiple workloads')
    arms = _student_arms(out, cfg, baseline)
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
    paths.update({name: value['path'] for name, value in verified_gate(out, cfg, baseline)['evidence'].items()})
    if 'distilled_CNN' in arms:
        paths['distillation_decision'] = out / 'distillation_decision.json'
    checkpoints = {ARMS[s]: _checkpoint(out, s) for s in ('direct', 'distilled') if ARMS[s] in arms}
    evaluation = dict(version=VERSION, solver=cfg.to_dict(), arms=list(arms),
                      deployment=settings['unet_transfer']['deployment'], **protocol)
    return freeze_research(out / 'data', checkpoints, out / 'fixed_classical.json', paths, _sources(), evaluation)


def final(output, *, resume=False):
    out, _, cfg, baseline = load(output)
    path = out / 'data/research_freeze.json'; evaluation = _read(path)['config']
    if evaluation['version'] != VERSION or evaluation['solver'] != cfg.to_dict():
        raise ValueError('Wrong frozen evaluation')
    arms = _student_arms(out, cfg, baseline)
    if list(arms) != evaluation['arms']:
        raise ValueError('Frozen student arms changed')
    with claim_final_evaluation(out / 'data', path, evaluation, resume=resume) as claim:
        splits, _ = materialize_final_data(out / 'data', claim, baseline)
        outputs = []
        for name, examples in splits.items():
            target = out / 'final' / name
            result = evaluate_research(examples, arms, cfg, baseline, target,
                reference_arm='fixed_C', resume=resume and (target / 'run_manifest.json').exists(),
                **{k: evaluation[k] for k in ('repeats', 'warmups', 'regimes', 'rhs_counts')})
            _report(result, target)
            write_json(target / 'deployment_decisions.json', deployment_decisions(result, **evaluation['deployment']))
            outputs.extend([target / 'raw_results.json', target / 'deployment_decisions.json'])
        complete_final_evaluation(out / 'data', claim, outputs)
    print('Final/OOD evaluation sealed; read measured outcomes, no automatic speedup claim.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'train', 'teacher-check', 'distill', 'benchmark', 'freeze', 'final'):
        p = sub.add_parser(name); p.add_argument('--run-dir', required=True)
        if name != 'freeze': p.add_argument('--resume', action='store_true')
        if name == 'prepare': p.add_argument('--config', default='configs/v6_7_unet_transfer_smoke.json')
        if name == 'train': p.add_argument('--stage', choices=('all', 'direct', 'teacher'), default='all')
        if name == 'benchmark': p.add_argument('--tag', default='validation')
        if name in ('benchmark', 'freeze'):
            p.add_argument('--repeats', type=int, default=5)
            p.add_argument('--warmups', type=int, default=1)
            p.add_argument('--rhs-counts', nargs='+', type=int, default=[1, 4, 16, 64])
            p.add_argument('--regimes', nargs='+', choices=('cold', 'warm', 'multiple', 'warm_multiple'),
                           default=['cold', 'warm', 'multiple'])
    a = parser.parse_args(argv)
    if a.command == 'prepare': return prepare(a.config, a.run_dir, resume=a.resume)
    if a.command == 'train': return train(a.run_dir, a.stage, resume=a.resume)
    if a.command == 'teacher-check': return teacher_check(a.run_dir, resume=a.resume)
    if a.command == 'distill': return train(a.run_dir, 'distilled', resume=a.resume)
    if a.command == 'final': return final(a.run_dir, resume=a.resume)
    protocol = dict(repeats=a.repeats, warmups=a.warmups, rhs_counts=a.rhs_counts, regimes=a.regimes)
    if a.command == 'freeze': return freeze(a.run_dir, **protocol)
    return benchmark(a.run_dir, tag=a.tag, resume=a.resume, **protocol)
