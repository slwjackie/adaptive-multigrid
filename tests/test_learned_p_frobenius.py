"""Fixed line_alt, unbiased Frobenius probes, P-only gradients and cache reuse."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.data import make_examples
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.learned_p import (NoNeuralSmoother, make_p_components, make_p_oracle,
    export_p_oracle, import_p_oracle, frobenius_objective, rademacher_errors,
    apply_error_operator, p_diagnostic)
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.unroll import make_graph
from adaptive_mg.v67.banks import Stats, hybrid_cycle
from adaptive_mg.v67.solver import PreparedAdaptiveMG
from adaptive_mg.v67.research_runtime import offline_teacher_generation
from adaptive_mg.v67 import joint_transfer_study as study


def inputs(n=7):
    torch.set_num_threads(1)
    example = make_examples([n], per_size=1, seed=39)[0]
    example.research_split = 'train'
    cfg = AdaptiveConfig(mg=MGConfig(strategy_name='line_alt_bilinear_full__v22',
        pre_steps=2, post_steps=2, max_cycles=150, stencil_backend='csr'),
        mode='research', branch='H_P', use_smoother=False, use_transfer=True,
        replace_pre=0, replace_post=0, smoother_levels=(), transfer_levels='all',
        spatial=False, gate_mode='open', use_learned_controller=False)
    return example, make_p_components(hidden=4, seed=19), cfg


def test_p_model_contains_no_nn_smoother_and_round_trips(tmp_path):
    example, model, cfg = inputs()
    assert isinstance(model.smoother, NoNeuralSmoother)
    assert not list(model.smoother.parameters()) and not model.smoother.state_dict()
    path = tmp_path/'p.pt'; model.save(path)
    restored = Components.load(path)
    assert restored.signature() == model.signature()
    assert isinstance(restored.smoother, NoNeuralSmoother)
    with pytest.raises(RuntimeError, match='no neural smoother'):
        restored.smoother(torch.zeros(1))
    with pytest.raises(ValueError, match='P-only'):
        frobenius_objective(example, restored, replace(cfg, use_smoother=True))


def test_complete_scaled_basis_matches_exact_frobenius_and_spectral_bound():
    example, model, cfg = inputs()
    n = example.a.shape[0]
    with torch.no_grad():
        dense_e = apply_error_operator(make_graph(example.a, (7, 7), model, cfg),
                                       torch.eye(n, dtype=torch.float64), model, cfg)
    loss, metrics = frobenius_objective(example, model, cfg,
        errors=np.sqrt(n)*torch.eye(n, dtype=torch.float64))
    assert loss.item() == pytest.approx(float(dense_e.square().sum()/n), rel=1e-12)
    assert metrics['frobenius_squared_estimate'] == pytest.approx(float(dense_e.square().sum()), rel=1e-12)
    rho = float(torch.linalg.eigvals(dense_e).abs().max())
    assert rho <= float(torch.linalg.matrix_norm(dense_e)) + 1e-12
    z = rademacher_errors(n, 4096, 131)
    estimated = apply_error_operator(make_graph(example.a, (7, 7), model, cfg), z, model, cfg).square().sum(0).mean()
    assert float(estimated.detach()) == pytest.approx(float(dense_e.square().sum()), rel=.08)


def test_all_level_p_gradients_through_galerkin_and_classical_line_smoothing():
    example, model, cfg = inputs(15)
    oracle = make_p_oracle(example, model, cfg, perturbation=.02)
    z = rademacher_errors(225, 8, 19)
    loss, metrics = frobenius_objective(example, oracle, cfg, errors=z)
    loss.backward()
    assert metrics['fallback_used'] is False and not metrics['nn_smoother']
    assert len(oracle.transfer.logits) == 2
    for p in oracle.transfer.logits:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    p = oracle.transfer.logits[0]
    index = tuple(int(v) for v in torch.nonzero(p.grad.abs() == p.grad.abs().max())[0])
    derivative, original = float(p.grad[index]), float(p.detach()[index])
    values = []
    for delta in (-.001, .001):
        with torch.no_grad(): p[index] = original+delta
        values.append(float(frobenius_objective(example, oracle, cfg, errors=z)[0].detach()))
    with torch.no_grad(): p[index] = original
    assert derivative == pytest.approx((values[1]-values[0])/.002, rel=.04, abs=1e-7)


def test_runtime_error_operator_matches_training_and_never_calls_nn_smoother(monkeypatch):
    example, model, cfg = inputs(15)
    def forbidden(*args, **kwargs):
        raise AssertionError('Neural smoother generation/application must be absent')
    monkeypatch.setattr(NoNeuralSmoother, 'forward', forbidden)
    import adaptive_mg.v67.banks as banks
    monkeypatch.setattr(banks, 'generated_stencil', forbidden)
    z = rademacher_errors(225, 2, 14)
    graph = make_graph(example.a, (15, 15), model, cfg)
    expected = apply_error_operator(graph, z, model, cfg).detach().numpy()
    prepared = PreparedAdaptiveMG(example.a, 15, model, cfg)
    root = prepared.ensure_branch('H_P', Stats())
    for column in range(2):
        actual = hybrid_cycle(root, z[:, column].numpy(), np.zeros(225), cfg, Stats(), None, 0, False)
        np.testing.assert_allclose(actual, expected[:, column], atol=1e-11, rtol=1e-10)


def test_same_a_reuses_all_p_banks_across_rhs_and_changed_a_rebuilds():
    example, model, cfg = inputs(15)
    prepared = PreparedAdaptiveMG(example.a, 15, model, cfg)
    first = Stats(); root = prepared.ensure_branch('H_P', first)
    assert first.transfer_nn_calls == 2
    assert first.smoother_nn_calls == 0
    rhs = np.stack([example.b, -example.b, .7*example.b])
    results = prepared.solve_many(rhs)
    assert all(r.converged for r in results)
    assert all(r.stats['transfer_nn_calls'] == 0 and r.stats['smoother_nn_calls'] == 0 for r in results)
    assert prepared.ensure_branch('H_P', Stats()) is root
    prepared.replace_operator(example.a*1.1)
    renewed = Stats(); newer = prepared.ensure_branch('H_P', renewed)
    assert renewed.transfer_nn_calls == 2 and newer is not root
    np.testing.assert_allclose(newer.coarse.a.toarray(),
        (newer.p.T@prepared.a@newer.p).toarray(), atol=1e-12)


def test_p_only_oracle_and_distillation_never_create_trainable_smoother(tmp_path):
    example, model, cfg = inputs()
    oracle = make_p_oracle(example, model, cfg)
    path = tmp_path/'oracle.pt'; torch.save(export_p_oracle(oracle), path)
    target = import_p_oracle(torch.load(path, weights_only=True))
    _, metrics = frobenius_objective(example, model, cfg, teacher=target)
    assert metrics['action_loss'] == 0 and metrics['operator_loss'] == 0
    assert not list(target.smoother.parameters())
    with torch.no_grad(): target.transfer.logits[0].add_(torch.randn_like(target.transfer.logits[0])*.2)
    loss, metrics = frobenius_objective(example, model, cfg, teacher=target)
    assert metrics['action_loss'] > 0 and metrics['operator_loss'] > 0
    loss.backward()
    assert model.transfer.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.transfer.parameters())
    example.research_split = 'validation'
    with pytest.raises(ValueError, match='TRAIN'):
        frobenius_objective(example, model, cfg)


def test_fixed_error_diagnostic_has_identical_before_after_probes():
    example, model, cfg = inputs()
    diagnostic = p_diagnostic(example, model, deepcopy(model), cfg, probes=4, repeats=1, warmups=0)
    assert diagnostic['frobenius_improved'] is False
    assert diagnostic['models']['before']['frobenius_per_unknown'] == diagnostic['models']['after']['frobenius_per_unknown']
    assert diagnostic['models']['after']['measured']['samples'][0]['smoother_nn_calls'] == 0


def test_p_only_small_workflow_resume_and_four_arm_benchmark(tmp_path):
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root/'configs/v6_7_learned_p_smoke.json').read_text())
    settings['training']['updates'] = 2
    settings['search'].update(starts=2, updates=2, repeats=1)
    settings['distillation'].update(updates=2, finetune_updates=2)
    path = tmp_path/'config.json'; path.write_text(json.dumps(settings))
    out = tmp_path/'run'
    study.prepare(path, out)
    study.train(out)
    assert set(p.name for p in (out/'checkpoints').iterdir()) == {'learned_P'}
    study.train(out, resume=True)
    diagnostic = study.diagnose(out)
    assert len(diagnostic['rows']) == 14
    assert study.diagnose(out, resume=True) == diagnostic
    study.search(out)
    study.distill(out)
    study.search(out, resume=True)
    study.distill(out, resume=True)
    _, settings, cfg, baseline = study.load(out)
    arms = study._arms(out, settings, cfg, baseline)
    assert set(arms) == {'fixed_C', 'em5_C', 'learned_P', 'distilled_P'}
    for arm in ('learned_P', 'distilled_P'):
        model = arms[arm]['model']
        assert isinstance(model.smoother, NoNeuralSmoother)
        assert model.metadata['trainable_components'] == ['transfer']
        history = study._read(out/'checkpoints'/arm/'training.json')
        assert all('frobenius_per_unknown' in r['metrics'] for r in history)
        assert all(not r['metrics']['fallback_used'] for r in history)
    result = study.benchmark(out, repeats=1, warmups=0, rhs_counts=[1], regimes=['multiple'])
    assert set(result['manifest']['arms']) == set(arms)
    assert all(row['runs'][name]['multiple']['1'][0]['successful'] for row in result['rows'] for name in arms)
    assert all(row['runs'][name]['multiple']['1'][0]['counters'].get('smoother_nn_calls', 0) == 0
               for row in result['rows'] for name in ('learned_P', 'distilled_P'))
    study.benchmark(out, resume=True, repeats=1, warmups=0, rhs_counts=[1], regimes=['multiple'])


def test_config_rejects_nn_smoother_and_wrong_frobenius_target():
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root/'configs/v6_7_learned_p_smoke.json').read_text())
    for change in ('nn', 'cycle', 'objective'):
        s = deepcopy(original)
        if change == 'nn': s['solver']['use_smoother'] = True
        elif change == 'cycle': s['training']['end_cycles'] = 3
        else: s['objective']['kind'] = 'residual'
        with pytest.raises(ValueError): study.validate_settings(s)


def test_p_only_exact_optimizer_resume(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root/'configs/v6_7_learned_p_smoke.json').read_text())
    settings['training']['updates'] = 2
    path = tmp_path/'config.json'; path.write_text(json.dumps(settings))
    out = tmp_path/'run'; study.prepare(path, out)
    original = study.frobenius_objective
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError('interrupted')
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(study, 'frobenius_objective', interrupted)
        with pytest.raises(RuntimeError, match='interrupted'): study.train(out)
    study.train(out, resume=True)
    out, settings, cfg, baseline = study.load(out)
    train, _, _ = study._development(out, settings, baseline)
    actual = study._checkpoint(out, 'learned_P', baseline)
    target = tmp_path/'reference'; target.mkdir()
    expected = study._run_training(target, 'learned_P', study._initial(out), train,
        cfg, baseline, settings, resume=False)
    assert actual.signature() == expected.signature()
    assert study._read(out/'checkpoints/learned_P/training.json') == study._read(target/'checkpoints/learned_P/training.json')


def test_p_only_freeze_and_final_keep_no_nn_smoother(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root/'configs/v6_7_learned_p_smoke.json').read_text())
    settings.update(smoke=False, seed=settings['seed']+100003)
    settings['training']['updates'] = 2
    settings['search'].update(starts=2, updates=1, repeats=1)
    settings['distillation'].update(updates=1, finetune_updates=1)
    original = study.make_research_plan
    def plan(**kwargs):
        value = original(**kwargs)
        for split in value['splits'].values():
            split.update(sizes=[7], families=split['families'][:1], per_family=1, count=1)
        return value
    monkeypatch.setattr(study, 'make_research_plan', plan)
    path = tmp_path/'config.json'; path.write_text(json.dumps(settings))
    out = tmp_path/'run'; study.prepare(path, out)
    study.train(out); study.search(out); study.distill(out)
    study.benchmark(out, repeats=1, warmups=0, rhs_counts=[1], regimes=['multiple'])
    study.freeze(out, repeats=1, warmups=0, rhs_counts=[1], regimes=['multiple'], splits=['final', 'ood_grid'])
    def forbidden(*args, **kwargs):
        raise AssertionError('Final evaluation must not select or train')
    monkeypatch.setattr(study, 'select_measured_candidate', forbidden)
    study.final(out)
    assert study._read(out/'data/final_claim.json')['status'] == 'completed'
    for split in ('final', 'ood_grid'):
        manifest = study._read(out/'final'/split/'run_manifest.json')
        assert set(manifest['arms']) == {'fixed_C', 'em5_C', 'learned_P', 'distilled_P'}
        assert all(not a['config']['use_smoother'] for a in manifest['arms'].values())
