"""Independent checks of the real MLP TTT path and its first warm-call cost."""
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import torch
import pytest

from adaptive_mg.data import make_examples
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.em_line_models import make_em_line_components
from adaptive_mg.v67.fixed_classical import BilinearClassicalBaseline
from adaptive_mg.v67 import light_smoother_evaluation as evaluation
from adaptive_mg.v67 import light_smoother_runtime as runtime
from adaptive_mg.v67.light_smoother_models import MLPPredictor, Standardizer, TinyMLP


def inputs():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    cfg = AdaptiveConfig.from_dict(json.loads((root/'configs/v6_7_em_line_smoke.json').read_text())['solver'])
    example = make_examples([15], per_size=4, seed=32)[-1]
    p = make_em_line_components(hidden=8, seed=27)
    with torch.no_grad():
        p.transfer.head.bias.normal_(0, .02)
    mlp = TinyMLP((.8, 1.2))
    with torch.no_grad():
        mlp.output.bias.fill_(-.2)
    predictor = MLPPredictor(mlp, Standardizer.fit(np.zeros((1, 12))))
    online = dict(enabled=True, method='mlp_ttt', trigger_window=1, trigger_rho=1e-8,
        near_tolerance_factor=1., min_remaining_cycles=.1, max_calls=1,
        time_budget_seconds=10., gradient_steps=2)
    return example, p, cfg, predictor, online


def test_real_mlp_ttt_uses_frozen_factors_without_dense_matrices_or_model_mutation(monkeypatch):
    example, p, cfg, predictor, online = inputs()
    prepared = runtime.PreparedLightMG(example.a, example.n, p, cfg, predictor, online)
    initial_model = predictor.to_dict()
    p_signature = p.expert_signature('transfer')
    factor_count = prepared._factor_count()
    array_ids = [(id(level.a), id(level.p), id(level.r), id(level.cache)) for level in prepared.levels]

    def forbidden(*args, **kwargs):
        raise AssertionError('TTT must use cached sparse operators and compiled factorizations')

    import scipy.sparse.linalg as sla
    monkeypatch.setattr(sla, 'splu', forbidden)
    for cls in (sp.csr_matrix, sp.csc_matrix):
        monkeypatch.setattr(cls, 'toarray', forbidden)
        monkeypatch.setattr(cls, 'todense', forbidden)
    result = prepared.solve(example.b)
    assert result.stats['online_adaptation_calls'] == 1
    assert result.stats['online_gradient_steps'] == 2
    assert result.stats['online_graph_builds'] == 1
    assert result.stats['online_probe_cycles'] == 2
    assert result.stats['online_adaptation_seconds'] <= result.solve_seconds
    assert prepared._factor_count() == factor_count
    assert array_ids == [(id(level.a), id(level.p), id(level.r), id(level.cache)) for level in prepared.levels]
    assert predictor.to_dict() == initial_model
    assert p.expert_signature('transfer') == p_signature
    assert all(parameter.grad is None for parameter in predictor.model.parameters())
    assert all(parameter.grad is None for parameter in p.transfer.parameters())


def test_first_ttt_graph_preparation_is_inside_warm_solve_time(monkeypatch):
    example, p, cfg, predictor, online = inputs()
    baseline = BilinearClassicalBaseline.from_config(cfg)
    example.classical_selection = baseline.selection_record()
    captures, graph_events = [], []
    original_construct = evaluation._construct_light
    original_graph = runtime._fixed_torch_graph

    def capture(*args):
        prepared = original_construct(*args)
        captures.append(prepared)
        assert prepared._graph is None
        return prepared

    def costly_graph(level):
        if level.index == 0:
            prepared = captures[-1]
            # Its priming solve has consumed setup but could not adapt/build
            # this graph. The first graph is created by the timed warm solve.
            graph_events.append((prepared.online_enabled, prepared.setup_pending))
            time.sleep(.01)
        return original_graph(level)

    monkeypatch.setattr(evaluation, '_construct_light', capture)
    monkeypatch.setattr(runtime, '_fixed_torch_graph', costly_graph)
    arm = dict(model=p, predictor=predictor, online=online,
               branch='H_P', selector=False, light=True, bounds=[.8, 1.2])
    record = evaluation.measured_light(example, arm, cfg, baseline, regime='warm')
    assert graph_events == [(True, False)]
    assert record['warm_prime_counters']['online_adaptation_calls'] == 0
    assert record['online_adaptation_calls'] == 1
    assert record['online_adaptation_seconds'] >= .01
    assert record['wall_seconds'] >= record['online_adaptation_seconds']
    assert record['total_setup_seconds'] == 0.
    assert record['solve_seconds'] == record['wall_seconds']
    assert record['online_adaptation_accepted']+record['online_adaptation_rejected'] == 1


@pytest.mark.parametrize('failure', ['probe', 'gradient'])
def test_failed_online_work_is_counted_and_previous_coefficients_restored(failure, monkeypatch):
    example, p, cfg, predictor, online = inputs()
    prepared = runtime.PreparedLightMG(example.a, example.n, p, cfg, predictor, online)
    before = prepared.get_coefficients()

    def failed(*args, **kwargs):
        raise RuntimeError('injected partial cycle failure')

    monkeypatch.setattr(prepared, 'raw_cycle' if failure == 'probe' else 'differentiable_cycle', failed)
    prepared._adapt(np.zeros_like(example.b), example.b)
    event = prepared.online_events[-1]
    counters = event['counters']
    assert not event['accepted']
    assert counters['online_adaptation_rejected'] == 1
    assert event['seconds'] > 0
    np.testing.assert_array_equal(prepared.get_coefficients(), before)
    assert counters['online_probe_cycles'] == 1
    if failure == 'probe':
        assert counters['online_probe_completed_cycles'] == 0
        trial = event['trials'][0]
        assert trial['attempted_cycles'] == 1 and trial['completed_cycles'] == 0
        assert not trial['success'] and trial['residual'] is None
        assert 'injected partial cycle failure' in trial['error']
        assert trial['seconds'] > 0
    else:
        assert counters['online_probe_completed_cycles'] == 1
        assert counters['online_gradient_vcycles'] == 1
        assert counters['online_gradient_completed_vcycles'] == 0
        assert counters['online_gradient_steps'] == 0
