"""Check that both new normalizations remain differentiable through A_c=P^TAP."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from adaptive_mg.config import MGConfig
from adaptive_mg.data import make_examples
from adaptive_mg.v67.bamg_solver import (ReferenceModels, build_graph, build_runtime,
                                        graph_cycle, residual_loss)
from bamg_feature9 import Feature9WeightCNN
from bamg_feature9_adapter import feature9_mode


@pytest.mark.parametrize('mode', ['row9', 'symmetric9'])
def test_actual_coarse_features_gradients_and_runtime_parity(mode):
    torch.set_num_threads(1)
    torch.manual_seed(73)
    example = make_examples([15], per_size=4, seed=32)[-1]
    model = Feature9WeightCNN(hidden=3, depth=1, activation='silu', feature_mode=mode).double()
    cfg = MGConfig(mode='classical', strategy_name='line_alt_bilinear_full',
                   pre_steps=2, post_steps=2, max_cycles=30, stencil_backend='csr')
    reference = ReferenceModels(None, None, mg_config=cfg, support_mode='fixed_bilinear')
    masks = reference.masks_for(example.a, (15, 15))
    captured = []

    def capture(module, args):
        assert args[0].shape[1] == 9
        if args[0].requires_grad:
            args[0].retain_grad()
            captured.append(args[0])

    rhs = np.random.default_rng(5).normal(size=225)
    with feature9_mode():
        handle = model.register_forward_pre_hook(capture)
        try:
            graph = build_graph(example.a, (15, 15), model, masks, cfg)
            loss, _, _ = residual_loss(graph, rhs, cycles=2)
            loss.backward()
        finally:
            handle.remove()
        assert len(captured) == 1 and captured[0].grad.abs().sum() > 0
        bias = model.head.bias
        index = int(bias.grad.abs().argmax())
        actual = float(bias.grad[index])
        assert abs(actual) > 1e-8
        old, step, values = float(bias.detach()[index]), 2e-5, []
        try:
            for direction in (1., -1.):
                with torch.no_grad():
                    bias[index] = old + direction * step
                candidate = build_graph(example.a, (15, 15), model, masks, cfg)
                value, _, _ = residual_loss(candidate, rhs, cycles=2)
                values.append(float(value.detach()))
        finally:
            with torch.no_grad():
                bias[index] = old
        assert actual == pytest.approx((values[0] - values[1]) / (2 * step), rel=2e-4, abs=2e-7)
        with torch.no_grad():
            bank = build_runtime(example.a, (15, 15), model, reference, cfg)
            graph = build_graph(example.a, (15, 15), model, masks, cfg)
            expected = graph_cycle(graph, torch.zeros(225, dtype=torch.float64), torch.tensor(rhs))
            np.testing.assert_allclose(bank.cycle(np.zeros(225), rhs), expected.numpy(), rtol=1e-10, atol=1e-12)
