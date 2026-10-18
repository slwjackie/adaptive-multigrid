"""Numerical contracts for transfer action distillation and teacher selection."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from adaptive_mg import DiffusionCase, MGConfig, assemble_stiffness
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.research_training import create_research_components, full_cycle_objective
from adaptive_mg.v67.unroll import make_graph


CAPS = dict(max_row_nnz=16, max_p_ratio=1., max_ac_ratio=1.15,
            max_operator_complexity=3., complexity_reference='parent',
            max_operator_complexity_ratio=1.15)


def _config():
    return AdaptiveConfig(
        mg=MGConfig(mode='classical', pre_steps=2, post_steps=2, nn_levels=1,
                    coarsest_n=3, stencil_backend='csr',
                    strategy_name='jacobi_bilinear_full'),
        mode='research', branch='H_P', use_smoother=False, use_transfer=True,
        spatial=False, gate_mode='open', inference_dtype='float32')


def _model():
    return create_research_components(
        transfer='existing', smoother_hidden=4, transfer_hidden=4,
        support='support_preserving', complexity_caps=CAPS, seed=716031)


def _graph_pair(*, changed=False):
    torch.set_num_threads(1)
    a = assemble_stiffness(DiffusionCase(n=7, epsilon=.2, angle_deg=25.))
    cfg = _config()
    student = _model()
    teacher = deepcopy(student)
    if changed:
        with torch.no_grad():
            student.transfer.head.bias.copy_(torch.linspace(-.08, .08, 16))
    sg = make_graph(a, (7, 7), student, cfg)
    # This deliberately keeps a teacher autograd graph: KD must detach its
    # target even when a caller forgets the optional no_grad optimization.
    tg = make_graph(a, (7, 7), teacher, cfg)
    return sg, tg, student, teacher, cfg


def test_action_distillation_is_zero_for_identical_vcycles():
    from adaptive_mg.v67.research_training import cycle_action_distillation_loss

    sg, tg, student, teacher, cfg = _graph_pair()
    loss = cycle_action_distillation_loss(sg, tg, student, teacher, cfg,
                                          probes=3, seed=8201)
    assert torch.isfinite(loss)
    assert float(loss.detach()) == pytest.approx(0., abs=1e-25)


def test_action_distillation_has_student_gradient_and_detached_teacher():
    from adaptive_mg.v67.research_training import cycle_action_distillation_loss

    sg, tg, student, teacher, cfg = _graph_pair(changed=True)
    loss = cycle_action_distillation_loss(sg, tg, student, teacher, cfg,
                                          probes=3, seed=8201)
    assert torch.isfinite(loss) and float(loss.detach()) > 1e-15
    loss.backward()
    gradients = [p.grad for p in student.transfer.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(float(g.abs().max()) > 1e-12 for g in gradients)
    assert all(p.grad is None for m in teacher.modules() for p in m.parameters())


def test_action_distillation_compares_operator_effect_not_p_coordinates():
    """P and 2P give identical exact two-level correction when Ac scales by 4."""
    from adaptive_mg.v67.autograd_sparse import SparseTensor
    from adaptive_mg.v67.research_training import cycle_action_distillation_loss

    sg, tg, student, teacher, cfg = _graph_pair()
    assert tg.coarse.coarse is None
    tg.p = SparseTensor.from_scipy(2. * tg.p.numpy())
    scaled_ac = 4. * tg.coarse.a.numpy()
    tg.coarse = replace(tg.coarse, a=SparseTensor.from_scipy(scaled_ac),
                        raw_scipy=scaled_ac)
    loss = cycle_action_distillation_loss(sg, tg, student, teacher, cfg,
                                          probes=3, seed=8201)
    assert float(loss.detach()) == pytest.approx(0., abs=1e-24)


def test_full_cycle_objective_uses_action_kd_and_keeps_actual_task_loss():
    torch.set_num_threads(1)
    e = dataset([7], 7, 716032)[0]
    student = _model()
    teacher = deepcopy(student)
    with torch.no_grad():
        teacher.transfer.head.bias.copy_(torch.linspace(-.05, .05, 16))
    loss, details, x = full_cycle_objective(
        e, student, _config(), prefix=2, tail=1, teacher=teacher,
        lambda_kd=.25, distillation_mode='cycle_action')
    assert torch.isfinite(loss) and torch.isfinite(x).all()
    assert details['kd_status'] == 'applied'
    assert float(details['kd'].detach()) > 0.
    assert details['task_branch'] == 'learned_prefix'
    assert details['actions'] == ['H', 'H', 'C']
    assert len(details['history']) == 3
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0.
               for p in student.transfer.parameters())
    assert all(p.grad is None for m in teacher.modules() for p in m.parameters())


@pytest.mark.parametrize('probes', [0, -1])
def test_action_distillation_rejects_empty_probe_objective(probes):
    from adaptive_mg.v67.research_training import cycle_action_distillation_loss

    sg, tg, student, teacher, cfg = _graph_pair()
    with pytest.raises(ValueError):
        cycle_action_distillation_loss(sg, tg, student, teacher, cfg, probes=probes)


def _gate_evidence():
    raw = {}
    rows = []
    for name in ('validation-a', 'validation-b'):
        raw[name] = {
            'strong_C': dict(feasible=True, contraction=.5),
            'direct_CNN': dict(feasible=True, contraction=.4),
            'teacher': dict(feasible=True, contraction=.3),
        }
        arms = {}
        for arm in raw[name]:
            arms[arm] = {'cold': {'1': [dict(successful=True,
                actual_neural_used=arm != 'strong_C', wall_seconds=100. if arm == 'teacher' else 1.)
                for _ in range(2)]}}
        rows.append(dict(example=dict(name=name), runs=arms))
    return raw, dict(rows=rows)


def test_teacher_gate_uses_convergence_not_offline_teacher_inference_time():
    from adaptive_mg.v67.unet_transfer_study import gate_decision

    raw, numerical = _gate_evidence()
    # The slow teacher is intentionally eligible: it is never the deployment arm.
    assert gate_decision(raw, numerical, min_cases=2)['adopted']


@pytest.mark.parametrize('reference', ['strong_C', 'direct_CNN'])
def test_teacher_gate_requires_improvement_over_both_references(reference):
    from adaptive_mg.v67.unet_transfer_study import gate_decision

    raw, numerical = _gate_evidence()
    for case in raw.values():
        case[reference]['contraction'] = .2
    assert not gate_decision(raw, numerical, min_cases=2)['adopted']


@pytest.mark.parametrize('invalid', ['infeasible', 'nonfinite', 'nonpositive',
                                      'classical_fallback', 'new_failure'])
def test_teacher_gate_cannot_hide_bad_case_or_fallback(invalid):
    from adaptive_mg.v67.unet_transfer_study import gate_decision

    raw, numerical = _gate_evidence()
    teacher = raw['validation-b']['teacher']
    measured = numerical['rows'][1]['runs']['teacher']['cold']['1']
    if invalid == 'infeasible':
        teacher['feasible'] = False
    elif invalid == 'nonfinite':
        teacher['contraction'] = float('nan')
    elif invalid == 'nonpositive':
        teacher['contraction'] = 0.
    elif invalid == 'classical_fallback':
        measured[1]['actual_neural_used'] = False
    else:
        measured[1]['successful'] = False
    # min_cases=1 ensures that deleting the bad second case cannot pass the gate.
    assert not gate_decision(raw, numerical, min_cases=1)['adopted']


def test_teacher_gate_requires_declared_cohort_and_minimum_cases():
    from adaptive_mg.v67.unet_transfer_study import gate_decision

    raw, numerical = _gate_evidence()
    assert not gate_decision(raw, numerical, min_cases=3)['adopted']
    numerical['rows'][1]['example']['name'] = 'different-case'
    with pytest.raises(ValueError):
        gate_decision(raw, numerical, min_cases=1)


def test_teacher_gate_rejects_failure_swap_even_when_success_counts_match():
    from adaptive_mg.v67.unet_transfer_study import gate_decision

    raw, numerical = _gate_evidence()
    for run in numerical['rows'][0]['runs']['teacher']['cold']['1']:
        run['successful'] = False
    for run in numerical['rows'][1]['runs']['strong_C']['cold']['1']:
        run['successful'] = False
    assert not gate_decision(raw, numerical, min_cases=1)['adopted']


def _study_settings():
    path = Path(__file__).resolve().parents[1] / 'configs/v6_7_unet_transfer_smoke.json'
    return json.loads(path.read_text())


def test_direct_and_distilled_students_start_identically_with_shared_contract():
    from adaptive_mg.v67.unet_transfer_study import initial_model, validate_settings

    settings = _study_settings()
    cfg = validate_settings(settings)
    direct = initial_model(settings, 'direct')
    # Teacher initialization between them must not change the student RNG state.
    initial_model(settings, 'teacher')
    distilled = initial_model(settings, 'distilled')
    assert direct.generation_signature() == distilled.generation_signature()
    assert type(direct.transfer) is type(distilled.transfer)
    for key, value in direct.transfer.state_dict().items():
        torch.testing.assert_close(value, distilled.transfer.state_dict()[key], rtol=0, atol=0)
    assert direct.transfer.support == distilled.transfer.support == 'support_preserving'
    assert direct.transfer.reference == distilled.transfer.reference == 'frozen_parent'
    assert direct.transfer.complexity_caps == distilled.transfer.complexity_caps
    assert cfg.use_transfer and not cfg.use_smoother and cfg.transfer_levels == 'all'


def test_real_teacher_then_student_training_applies_kd_then_task_only(tmp_path):
    from adaptive_mg.v67.research_training import train_expert
    from adaptive_mg.v67.strong import StrongRules, select_strong_strategy
    from adaptive_mg.v67.unet_transfer_study import initial_model

    torch.set_num_threads(1)
    settings = _study_settings()
    settings['unet_transfer'].update(student_hidden=4, teacher_hidden=4, teacher_depth=1)
    e = dataset([7], 7, 716033)[0]
    rules = StrongRules()
    e.strong_selection = select_strong_strategy(e.a, e.n, rules).to_dict()
    e.research_split = 'train'
    training = dict(settings['training'], updates=2, kd_updates=1, prefix_cycles=1,
                    tail_cycles=0, rhs_variants=1, lambda_noharm=0.)
    teacher, teacher_status = train_expert(initial_model(settings, 'teacher'), [e],
        _config(), rules, training, tmp_path / 'teacher', branch='H_P')
    assert teacher_status['all_updates_completed'] and teacher_status['feasible_task_updates'] == 2
    assert teacher.transfer.training_only
    original_signature = teacher.generation_signature()
    student, status = train_expert(initial_model(settings, 'distilled'), [e],
        _config(), rules, training, tmp_path / 'student', branch='H_P', teacher=teacher)
    records = json.loads((tmp_path / 'student/training.json').read_text())
    assert status['all_updates_completed'] and status['feasible_task_updates'] == 2
    assert records[0]['phase'] == 'distill_plus_task' and records[0]['kd_status'] == 'applied'
    assert records[1]['phase'] == 'full_cycle_task' and records[1]['kd_status'] == 'not_requested'
    assert all(r['distillation_mode'] == 'cycle_action' and not r['skipped'] for r in records)
    assert not student.transfer.training_only
    assert teacher.generation_signature() == original_signature
    manifest = json.loads((tmp_path / 'student/training_manifest.json').read_text())
    assert manifest['settings']['distillation_mode'] == 'cycle_action'
    assert manifest['settings']['kd_probes'] == training['kd_probes']
    assert manifest['teacher_signature'] == original_signature


def _deployment_report(regime='cold'):
    entry = dict(ci95=[1.5, 2.5], new_failure_case_ids=[], accuracy_worse_vs_strong_case_ids=[])
    rows = []
    for name in ('validation-a', 'validation-b'):
        runs = {}
        for arm in ('strong_C', 'direct_CNN', 'distilled_CNN'):
            runs[arm] = {regime: {'1': [dict(successful=True,
                actual_neural_used=arm != 'strong_C',
                wall_seconds=2. if arm == 'strong_C' else 1.,
                solve_seconds=.2, total_setup_seconds=1.8 if arm == 'strong_C' else .8)]}}
        rows.append(dict(example=dict(name=name), runs=runs))
    return dict(rows=rows, summary={regime: {'1': {
        'direct_CNN': deepcopy(entry), 'distilled_CNN': deepcopy(entry)}}})


def test_deployment_decision_requires_setup_inclusive_workload():
    from adaptive_mg.v67.unet_transfer_study import deployment_decisions

    assert deployment_decisions(_deployment_report('warm'), min_cases=2) == {}
    result = _deployment_report()
    assert deployment_decisions(result, min_cases=2)['cold/rhs1/distilled_CNN']['qualifies']
    for row in result['rows']:
        run = row['runs']['distilled_CNN']['cold']['1'][0]
        run.update(solve_seconds=.01, total_setup_seconds=2.99, wall_seconds=3.)
    decision = deployment_decisions(result, min_cases=2)['cold/rhs1/distilled_CNN']
    assert not decision['observed_time_target_met'] and not decision['qualifies']
    assert decision['setup_included']


@pytest.mark.parametrize('defect', ['new_failure', 'no_neural', 'accuracy', 'confidence'])
def test_deployment_cannot_qualify_on_timing_alone(defect):
    from adaptive_mg.v67.unet_transfer_study import deployment_decisions

    result = _deployment_report()
    entry = result['summary']['cold']['1']['distilled_CNN']
    bad = result['rows'][1]['runs']['distilled_CNN']['cold']['1'][0]
    if defect == 'new_failure':
        bad['successful'] = False
        entry['new_failure_case_ids'] = ['validation-b']
    elif defect == 'no_neural':
        bad['actual_neural_used'] = False
    elif defect == 'accuracy':
        entry['accuracy_worse_vs_strong_case_ids'] = ['validation-b']
    else:
        entry['ci95'] = [.99, 2.5]
    assert not deployment_decisions(result, min_cases=1)['cold/rhs1/distilled_CNN']['qualifies']
