"""Independent numerical checks beyond regular isotropic Poisson examples."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from neural_amg.fem import generate_mesh, assemble_p1
from neural_amg.model import GraphAttentionTransfer
from neural_amg.solver import (
    _runtime_cycle, build_hierarchy, multicycle_loss, vcycle,
)


def irregular_grounded_graph():
    """Connected weighted graph with irregular degrees and positive grounding."""
    rng = np.random.default_rng(304)
    n = 47
    pairs = {(i, i + 1) for i in range(n - 1)}
    pairs.update(tuple(sorted(p)) for p in rng.integers(0, n, size=(70, 2))
                 if p[0] != p[1])
    rows, cols = map(np.array, zip(*sorted(pairs)))
    weights = rng.uniform(.1, 2., len(rows))
    adjacency = sp.coo_matrix((np.r_[weights, weights],
        (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)).tocsr()
    return (sp.diags(np.asarray(adjacency.sum(1)).ravel()
                    + rng.uniform(.01, .1, n)) - adjacency).tocsr()


def rotated_anisotropic_p1():
    angle = np.deg2rad(31.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tensor = rotation @ np.diag([1., .02]) @ rotation.T
    return assemble_p1(generate_mesh(7, 123), tensor).a


@pytest.mark.parametrize('problem', [irregular_grounded_graph, rotated_anisotropic_p1])
def test_multilevel_irregular_and_rotated_runtime_parity(problem):
    torch.manual_seed(127)
    a = problem()
    model = GraphAttentionTransfer(hidden=8)
    hierarchy = build_hierarchy(a, model, max_coarse=3)
    assert len(hierarchy.levels) >= 3
    for index, level in enumerate(hierarchy.levels[:-1]):
        p = level.p.numpy()
        expected = (p.T @ level.a.numpy() @ p).toarray()
        np.testing.assert_allclose(hierarchy.levels[index + 1].a.numpy().toarray(),
                                   expected, rtol=1e-12, atol=1e-12)
        np.linalg.cholesky(expected)
    rng = np.random.default_rng(57)
    b = rng.normal(size=(a.shape[0], 2))
    initial = rng.normal(size=b.shape)
    actual = vcycle(hierarchy, torch.tensor(b), torch.tensor(initial)).detach().numpy()
    np.testing.assert_allclose(actual, _runtime_cycle(hierarchy, b, initial),
                               rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('problem', [irregular_grounded_graph, rotated_anisotropic_p1])
def test_frozen_structure_multilevel_parameter_gradient_matches_finite_difference(problem):
    # Freeze discrete C/S through both perturbed forwards. The derivative must
    # still include every downstream A=P^T A P, its features and its smoother.
    torch.manual_seed(127)
    a = problem()
    model = GraphAttentionTransfer(hidden=8)
    hierarchy = build_hierarchy(a, model, max_coarse=3)
    structures = [level.structure for level in hierarchy.levels[:-1]]
    probes = np.random.default_rng(57).normal(size=(a.shape[0], 2))
    loss = multicycle_loss(hierarchy, probes, cycles=2)
    loss.backward()
    parameter = max((p for p in model.parameters() if p.grad is not None),
                    key=lambda p: p.grad.abs().max().item())
    offset = int(parameter.grad.abs().argmax())
    analytic = parameter.grad.flatten()[offset].item()
    assert abs(analytic) > 1e-6
    base = parameter.flatten()[offset].item()
    step = 1e-5
    perturbed = []
    with torch.no_grad():
        for delta in (step, -step):
            parameter.flatten()[offset] = base + delta
            h = build_hierarchy(a, model, max_coarse=3, structures=structures)
            perturbed.append(multicycle_loss(h, probes, cycles=2).item())
        parameter.flatten()[offset] = base
    numerical = (perturbed[0] - perturbed[1]) / (2 * step)
    assert analytic == pytest.approx(numerical, rel=2e-5, abs=2e-8)
