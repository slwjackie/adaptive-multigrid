import numpy as np
import pyamg
import pytest
import scipy.sparse as sp
import torch

from neural_amg.model import GraphAttentionTransfer
from neural_amg.solver import build_hierarchy, solve, vcycle, multicycle_loss, _runtime_cycle


def test_sparse_multilevel_runtime_autograd_and_galerkin_agree():
    torch.manual_seed(9)
    a = pyamg.gallery.poisson((7, 7), format='csr')
    model = GraphAttentionTransfer(hidden=8)
    h = build_hierarchy(a, model, max_coarse=3)
    assert len(h.levels) >= 3
    for i, level in enumerate(h.levels[:-1]):
        p = level.p.numpy()
        expected = p.T @ level.a.numpy() @ p
        np.testing.assert_allclose(expected.toarray(), h.levels[i+1].a.numpy().toarray(), atol=1e-12)
        np.linalg.cholesky(expected.toarray())
    b = np.random.default_rng(7).normal(size=49)
    np.testing.assert_allclose(vcycle(h, torch.tensor(b)).detach().numpy(), _runtime_cycle(h, b), rtol=1e-12, atol=1e-12)
    result = solve(h, b, max_cycles=200)
    assert result['converged']
    np.testing.assert_allclose(a @ result['x'], b, atol=1e-7)
    loss = multicycle_loss(h, np.random.default_rng(8).normal(size=(49, 2)), cycles=3)
    loss.backward()
    assert np.isfinite(float(loss.detach()))
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0


def test_classical_and_teacher_use_actual_sparse_hierarchy():
    a = pyamg.gallery.poisson((5,5), format='csr')
    for method in ['classical', 'teacher']:
        h = build_hierarchy(a, method=method, max_coarse=3)
        result = solve(h, np.ones(25), max_cycles=150)
        assert result['converged']
        assert h.setup_seconds > 0
        assert h.operator_complexity >= 1


def test_zero_rhs_and_explicit_failures():
    a = pyamg.gallery.poisson((3,3), format='csr')
    h = build_hierarchy(a, method='classical')
    result = solve(h, np.zeros(9))
    assert result['converged'] and result['cycles'] == 0
    with pytest.raises(ValueError, match='max_levels'):
        build_hierarchy(a, method='classical', max_levels=1, max_coarse=2)
    with pytest.raises(ValueError, match='symmetric'):
        build_hierarchy(sp.csr_matrix([[2., 1.], [0., 2.]]), method='classical')
    with pytest.raises(ValueError, match='nonzero'):
        multicycle_loss(h, np.zeros((9, 1)))
    assert not solve(build_hierarchy(pyamg.gallery.poisson((5,5),format='csr'),method='classical'),np.ones(25),max_cycles=0)['converged']
