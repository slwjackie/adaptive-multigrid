"""Whole-cycle joint gradients and raw offline search correctness."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.data import make_examples
from adaptive_mg.transfer import coarse_fine_indices
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.joint_transfer_training import (
    InfeasibleCandidate, curriculum_cycles, export_oracle, import_oracle,
    joint_objective, make_joint_components, make_oracle, measure_raw_candidate,
    mixed_probes,
)
from adaptive_mg.v67.research_runtime import offline_teacher_generation
from adaptive_mg.v67.unroll import make_graph


def inputs(n=7):
    torch.set_num_threads(1)
    example = make_examples([n], per_size=1, seed=32)[0]
    cfg = AdaptiveConfig(mg=MGConfig(strategy_name='line_alt_bilinear_full',
        pre_steps=2, post_steps=2, stencil_backend='csr', max_cycles=50),
        mode='research', branch='H_SP', replace_pre=2, replace_post=2,
        smoother_levels='all', transfer_levels='all', spatial=False,
        gate_mode='open', use_learned_controller=False)
    return example, make_joint_components(hidden=4, seed=18), cfg


def test_raw_objective_trains_p_and_both_smoothers_through_recursive_coarse_a():
    example, model, cfg = inputs(15)
    oracle = make_oracle(example, model, cfg, perturbation=.01)
    loss, report = joint_objective(example, oracle, cfg, cycles=3,
                                  hard_smoothing=1, hard_cycles=1)
    assert not report['fallback_used'] and report['feasible']
    assert report['probe_labels'] == ['random', 'classical_smoothed', 'learned_hard']
    loss.backward()
    assert len(oracle.transfer.logits) == 2
    for parameter in oracle.transfer.logits:
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    for directions in oracle.smoother.directions:
        assert directions.grad[:, 0].abs().sum() > 0
        assert directions.grad[:, 1].abs().sum() > 0
    # Differentiate one fine P entry through rebuilt Ac and terminal solve.
    parameter = oracle.transfer.logits[0]
    index = tuple(int(i) for i in torch.nonzero(parameter.grad.abs() == parameter.grad.abs().max())[0])
    analytical = float(parameter.grad[index])
    original = float(parameter.detach()[index]); numerical = []
    # Hold the probe residuals fixed: objective's detached hard-mode generator is
    # intentionally not differentiated, so test the graph on common random rhs.
    from adaptive_mg.v67.unroll import cycle
    b = torch.randn(example.a.shape[0], dtype=torch.float64)
    def value():
        root = make_graph(example.a, (15, 15), oracle, cfg)
        x = cycle(root, torch.zeros_like(b), b, oracle, cfg)
        return (b-root.a.apply(x)).square().sum()
    oracle.transfer.zero_grad(); oracle.smoother.zero_grad()
    with offline_teacher_generation():
        scalar = value(); scalar.backward(); analytical = float(parameter.grad[index])
        for delta in (-1e-3, 1e-3):
            with torch.no_grad(): parameter[index] = original + delta
            numerical.append(float(value().detach()))
    with torch.no_grad(): parameter[index] = original
    assert analytical == pytest.approx((numerical[1]-numerical[0])/.002, rel=.03, abs=2e-5)


def test_oracle_projection_preserves_support_boundary_rows_and_injection():
    example, model, cfg = inputs(15)
    oracle = make_oracle(example, model, cfg, perturbation=1.)
    with offline_teacher_generation():
        level = make_graph(example.a, (15, 15), oracle, cfg)
    while level.coarse is not None:
        weights = level.interpolation_weights.detach().numpy()
        assert np.all(weights[level.base_weights == 0] == 0)
        np.testing.assert_allclose(weights.sum(1), level.base_weights.sum(1), atol=2e-15)
        assert np.any(level.base_weights.sum(1) != 1.)
        rows = coarse_fine_indices(level.pattern)
        np.testing.assert_array_equal(weights[rows], level.base_weights[rows])
        expected = level.p.numpy().T @ level.raw_scipy @ level.p.numpy()
        np.testing.assert_allclose(level.coarse.raw_scipy.toarray(), expected.toarray(), atol=1e-12)
        level = level.coarse


def test_oracle_round_trip_and_action_operator_distillation(tmp_path):
    example, model, cfg = inputs()
    oracle = make_oracle(example, model, cfg)
    target = tmp_path/'oracle.pt'; torch.save(export_oracle(oracle), target)
    restored = import_oracle(torch.load(target, weights_only=True))
    _, report = joint_objective(example, model, cfg, cycles=2,
        hard_smoothing=1, hard_cycles=1, teacher=restored)
    assert report['action_loss'] == 0. and report['operator_loss'] == 0.
    with torch.no_grad(): restored.smoother.directions[0][:, 1, 1] += .1
    loss, report = joint_objective(example, model, cfg, cycles=2,
        hard_smoothing=1, hard_cycles=1, teacher=restored)
    assert report['action_loss'] > 0 and report['operator_loss'] > 0
    loss.backward()
    assert model.smoother.stencil_heads[1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in restored.smoother.parameters())


def test_training_and_oracle_are_train_only_and_bound_to_operator():
    example, model, cfg = inputs()
    oracle = make_oracle(example, model, cfg)
    changed = deepcopy(example); changed.a = example.a * 2
    with pytest.raises(ValueError, match='different operator'):
        joint_objective(changed, oracle, cfg)
    example.research_split = 'validation'
    with pytest.raises(ValueError, match='TRAIN'):
        make_oracle(example, model, cfg)
    with pytest.raises(ValueError, match='TRAIN'):
        joint_objective(example, model, cfg)


def test_hard_probes_detached_normalized_and_seeded():
    example, model, cfg = inputs()
    first, labels = mixed_probes(example, model, cfg, seed=11, per_kind=2)
    second, _ = mixed_probes(example, model, cfg, seed=11, per_kind=2)
    assert first.shape == (49, 6) and not first.requires_grad
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(torch.linalg.vector_norm(first, dim=0), torch.ones(6, dtype=torch.float64))
    assert labels.count('learned_hard') == 2


def test_raw_candidate_timing_includes_setup_and_no_classical_recovery(monkeypatch):
    example, model, cfg = inputs()
    oracle = make_oracle(example, model, cfg)
    import adaptive_mg.v67.banks as banks
    def forbidden(*args, **kwargs):
        raise AssertionError('classical recovery ran during raw timing')
    monkeypatch.setattr(banks, 'classical_step', forbidden)
    result = measure_raw_candidate(example, oracle, cfg, target_rhs_count=4,
        cycles=2, hard_smoothing=1, hard_cycles=1, probe_model=model)
    assert result['successful'] and result['raw_no_fallback'] and result['em_iterations'] == 0
    assert result['rhs_count'] == 4 and result['cycles'] >= 4
    assert result['seconds'] == pytest.approx(result['setup_seconds']+result['solve_seconds'])
    assert result['final_residual'] <= cfg.mg.tolerance
    assert result['setup_seconds'] > 0 and not result['fallback_used']
    short = replace(cfg, mg=replace(cfg.mg, max_cycles=1))
    failure = measure_raw_candidate(example, oracle, short, hard_smoothing=1, hard_cycles=1)
    assert not failure['successful'] and not failure['fallback_used']
    assert failure['reason'] == 'raw candidate failed tolerance'


def test_nonfinite_proposals_rejected_without_training_fallback():
    example, model, cfg = inputs()
    oracle = make_oracle(example, model, cfg)
    with torch.no_grad(): oracle.smoother.raw_gains[0].fill_(float('nan'))
    with pytest.raises(InfeasibleCandidate, match='nonfinite'):
        joint_objective(example, oracle, cfg)
    result = measure_raw_candidate(example, oracle, cfg)
    assert not result['successful'] and not result['fallback_used']


def test_curriculum_uses_3_then_6_full_cycles():
    assert [curriculum_cycles(i, 4) for i in range(4)] == [3, 3, 6, 6]
    with pytest.raises(ValueError): curriculum_cycles(4, 4)


def test_zero_free_stencil_has_finite_gradients():
    example, model, cfg = inputs()
    oracle = make_oracle(example, model, cfg)
    with torch.no_grad(): oracle.smoother.directions[0].zero_()
    features = torch.zeros((1, 10, 7, 7))
    directions, gains = oracle.smoother.paired_directions_and_gains(features)
    (directions.sum()+gains.sum()).backward()
    assert torch.isfinite(oracle.smoother.directions[0].grad).all()
