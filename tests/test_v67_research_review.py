"""Independent regressions for cross-module research/runtime contracts."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from adaptive_mg.config import MGConfig
from adaptive_mg.pde import DiffusionCase, assemble_stiffness
from adaptive_mg.transfer import coarse_fine_indices
from adaptive_mg.v67.banks import Stats
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.research_controls import ControlledTransferCNN
from adaptive_mg.v67.research_runtime import offline_teacher_generation
from adaptive_mg.v67.research_smoothers import make_research_smoother
from adaptive_mg.v67.research_training import validation_adoption
from adaptive_mg.v67.solver import PreparedAdaptiveMG
from adaptive_mg.v67.strong import PreparedStrongMG, load_strong_rules
from adaptive_mg.v67.unroll import make_graph


def _problem(n=7):
    a = assemble_stiffness(DiffusionCase(n=n, epsilon=1., angle_deg=0., contrast=1.))
    b = np.random.default_rng(716).normal(size=n * n)
    return a, b


def _config(branch='H_P'):
    return AdaptiveConfig(mg=MGConfig(mode='classical', pre_steps=2, post_steps=2,
        nn_levels=1, stencil_backend='csr', max_cycles=10, strategy_name='jacobi_bilinear_full'),
        branch=branch, mode='research', spatial=False, gate_mode='open',
        use_smoother=branch in {'H_S', 'H_SP'}, use_transfer=branch in {'H_P', 'H_SP'})


class _ExplicitExpandedLogits(ControlledTransferCNN):
    def __init__(self, n=7):
        super().__init__(hidden=4, support='expanded', complexity_caps=dict(
            max_row_nnz=3, max_p_ratio=10., max_ac_ratio=10., max_operator_complexity=10.))
        self.logits = nn.Parameter(torch.zeros(1, 36, n, n))

    def forward(self, _features):
        return self.logits


def test_fp64_runtime_repair_preserves_topk_and_unroll_straight_through_gradient():
    a, _ = _problem()
    model = Components.create(hidden=4)
    model.transfer = _ExplicitExpandedLogits()
    cfg = _config()
    graph = make_graph(a, (7, 7), model, cfg)
    prepared = PreparedAdaptiveMG(a, 7, model, cfg)
    runtime = prepared.ensure_branch('H_P', Stats())
    difference = (runtime.p - graph.p.numpy()).tocsr()
    assert np.max(np.abs(difference.data), initial=0.) < 1e-14
    actual_nnz = np.diff(runtime.p.indptr)
    assert np.max(actual_nnz) <= 3
    assert np.allclose(np.asarray(runtime.p.sum(1)).ravel(), graph.base_weights.sum(1), atol=1e-14)
    injection = coarse_fine_indices(graph.pattern)
    assert np.allclose(np.asarray(runtime.p[injection].sum(1)).ravel(), 1.)
    numerical = graph.interpolation_weights.detach().numpy()
    dropped = (graph.pattern.columns >= 0) & (numerical == 0)
    dropped[injection] = False
    row, column = np.argwhere(dropped)[0]
    assert graph.interpolation_weights[row, column] == 0
    # A zero/pruned candidate must still receive the promised surrogate task
    # derivative; otherwise support expansion cannot directly learn new edges.
    graph.interpolation_weights[row, column].backward()
    assert model.transfer.logits.grad is not None
    assert float(model.transfer.logits.grad.abs().sum()) > 1e-8
    assert torch.isfinite(model.transfer.logits.grad).all()


def test_cached_teacher_bank_cannot_escape_its_offline_context():
    a, b = _problem()
    model = Components.create(hidden=4)
    model.smoother = make_research_smoother('teacher_cnn', hidden=4, depth=1)
    cfg = _config('H_S')
    prepared = PreparedStrongMG(a, 7, model, cfg, load_strong_rules())
    calls = []
    hook = model.smoother.body.register_forward_pre_hook(lambda *_: calls.append(1))
    with offline_teacher_generation():
        root = prepared.ensure_branch('H_S', Stats())
        assert root is prepared.learned
    assert len(calls) == 1
    # Already generated operators are still teacher artifacts, not students.
    with pytest.raises(RuntimeError, match='training-only'):
        prepared.solve(b)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match='training-only'):
        prepared.solve_many([b, b * .3])
    assert len(calls) == 1
    prepared.config = replace(cfg, branch='C', mode='classical')
    prepared.solve(b)
    assert len(calls) == 1  # Unused teacher does not block pure classical solving.
    hook.remove()


@pytest.mark.parametrize('metric', ['seconds', 'contraction'])
@pytest.mark.parametrize('invalid', [-1., 0., float('nan'), float('inf')])
def test_adoption_rejects_invalid_success_metrics(metric, invalid):
    reference = {'a': {'success': True, 'seconds': 2., 'contraction': .4}}
    candidate = {'a': {'success': True, 'seconds': 1., 'contraction': .2}}
    candidate['a'][metric] = invalid
    kwargs = {'task_reduction': .05} if metric == 'contraction' else {}
    with pytest.raises(ValueError, match='finite and positive'):
        validation_adoption(candidate, reference, min_cases=1, **kwargs)


def test_cnn_transfer_projection_contract_survives_checkpoint_and_changes_cache_key(tmp_path):
    a, _ = _problem()
    model = Components.create(hidden=4)
    model.smoother = make_research_smoother('o2', hidden=4, depth=1)
    model.transfer = ControlledTransferCNN(hidden=4, support='expanded')
    model.save(tmp_path / 'controlled.pt')
    clone = Components.load(tmp_path / 'controlled.pt')
    assert clone.generation_signature() == model.generation_signature()
    assert clone.transfer.research_transfer_spec() == model.transfer.research_transfer_spec()
    prepared = PreparedAdaptiveMG(a, 7, clone, _config())
    first = prepared.ensure_branch('H_P', Stats())
    signature = clone.generation_signature()
    clone.transfer.complexity_caps['max_row_nnz'] = 3
    assert clone.generation_signature() != signature
    stats = Stats()
    second = prepared.ensure_branch('H_P', stats)
    assert second is not first and stats.transfer_nn_calls == 1
    assert np.max(np.diff(second.p.indptr)) <= 3


@pytest.mark.parametrize('change_p', [False, True])
def test_same_actual_level_smoother_is_generated_once_across_hs_and_hsp(change_p):
    a, _ = _problem(n=15)
    model = Components.create(hidden=4)
    model.smoother = make_research_smoother('student_cnn', hidden=4)
    cfg = _config('H_S')
    cfg = replace(cfg, mg=replace(cfg.mg, nn_levels=2))
    prepared = PreparedStrongMG(a, 15, model, cfg, load_strong_rules())
    first_stats = Stats()
    hs = prepared.ensure_branch('H_S', first_stats)
    assert first_stats.smoother_nn_calls == 2
    if change_p:
        with torch.no_grad():
            model.transfer.head.bias.copy_(torch.linspace(-.15, .15, 16))
    joint_stats = Stats()
    joint = prepared.ensure_branch('H_SP', joint_stats)
    assert joint.neural_stencil is hs.neural_stencil  # A0 did not change.
    assert joint_stats.smoother_nn_calls == (1 if change_p else 0)
    assert joint_stats.smoother_operator_cache_hits == (1 if change_p else 2)
    if change_p:
        assert joint.coarse.neural_stencil is not hs.coarse.neural_stencil
    else:
        assert joint.coarse.neural_stencil is hs.coarse.neural_stencil
    repeat_stats = Stats()
    assert prepared.ensure_branch('H_SP', repeat_stats) is joint
    assert repeat_stats.neural_setup_calls == 0
