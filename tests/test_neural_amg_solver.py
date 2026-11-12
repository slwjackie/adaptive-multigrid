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


def test_nonfinite_interpolation_stops_at_last_valid_level():
    from neural_amg.autograd_sparse import SparseTensor

    class FailsOnSecondLevel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = GraphAttentionTransfer(hidden=8)
            self.calls = 0

        def forward(self, a, structure):
            self.calls += 1
            p = self.model(a, structure)
            if self.calls == 2:
                return SparseTensor(p.pattern, p.values * float('nan'))
            return p

    torch.manual_seed(9)
    a = pyamg.gallery.poisson((7, 7), format='csr')
    model = FailsOnSecondLevel()
    hierarchy = build_hierarchy(a, model, max_coarse=3)
    assert len(hierarchy.levels) == 2
    assert hierarchy.levels[0].p is not None
    assert hierarchy.levels[1].p is None
    assert hierarchy.guard_events[0]['stage'] == 'setup'
    assert hierarchy.guard_events[0]['level'] == 1
    assert 'Nonfinite interpolation' in hierarchy.guard_events[0]['reason']
    loss = multicycle_loss(hierarchy, np.random.default_rng(57).normal(size=(49, 2)), cycles=2)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().max() > 0 for p in model.parameters())
    assert solve(hierarchy, np.ones(49), max_cycles=150)['converged']


def test_first_actual_cycle_truncates_failing_coarse_level_in_both_paths():
    a = pyamg.gallery.poisson((7, 7), format='csr')
    rhs = np.random.default_rng(32).normal(size=49)
    answers = []
    for differentiable in (False, True):
        h = build_hierarchy(a, method='classical', max_coarse=3)
        assert len(h.levels) > 2
        # Force a faulty smoother at level 1. Level 0 must retain its transfer.
        h.levels[1].omega = torch.tensor(100., dtype=torch.float64)
        h.runtime[1]['omega'] = 100.
        if differentiable:
            x = vcycle(h, torch.tensor(rhs)).detach().numpy()
        else:
            x = _runtime_cycle(h, rhs)
        answers.append(x)
        assert len(h.levels) == 2
        assert h.levels[0].p is not None and h.levels[1].p is None
        assert h.guard_events[0]['stage'] == 'first_cycle'
        assert h.guard_events[0]['level'] == 1
        assert np.linalg.norm(rhs - a @ x) < np.linalg.norm(rhs)
        assert solve(h, rhs, max_cycles=150)['converged']
    np.testing.assert_allclose(*answers, rtol=1e-12, atol=1e-12)


def test_terminal_factor_failure_walks_back_to_valid_parent(monkeypatch):
    import neural_amg.solver as solver

    original = solver.sla.splu
    failed_sizes = []
    def reject_smallest(matrix):
        if matrix.shape[0] <= 3:
            failed_sizes.append(matrix.shape[0])
            raise RuntimeError('injected terminal factor failure')
        return original(matrix)

    monkeypatch.setattr(solver.sla, 'splu', reject_smallest)
    a = pyamg.gallery.poisson((7, 7), format='csr')
    h = build_hierarchy(a, method='classical', max_coarse=3)
    assert failed_sizes
    assert h.levels[-1].a.shape[0] > 3
    assert h.levels[-1].p is None
    assert any(event.get('action') == 'reject_terminal' for event in h.guard_events)
    assert solve(h, np.ones(49), max_cycles=150)['converged']


def test_factorization_is_shared_with_autograd_and_no_setup_trial_rhs():
    a = pyamg.gallery.poisson((3, 3), format='csr')
    h = build_hierarchy(a, method='classical', max_coarse=9)
    terminal = h.levels[-1].a
    assert terminal.cache['factorizations'] == 1
    assert h.autograd_checked == set() and h.runtime_checked == set()
    rhs = torch.ones(9, dtype=torch.float64, requires_grad=True)
    result = vcycle(h, rhs)
    result.square().sum().backward()
    assert torch.isfinite(rhs.grad).all()
    assert terminal.cache['factorizations'] == 1
    assert terminal.cache['lu'] is h.terminal_lu


def test_nonfinite_galerkin_operator_is_rejected_before_use(monkeypatch):
    from neural_amg.autograd_sparse import SparseTensor

    original = SparseTensor.symmetrize
    def broken_product(self):
        result = original(self)
        return SparseTensor(result.pattern, result.values * float('nan'))

    monkeypatch.setattr(SparseTensor, 'symmetrize', broken_product)
    a = pyamg.gallery.poisson((5, 5), format='csr')
    h = build_hierarchy(a, method='classical', max_coarse=3)
    assert len(h.levels) == 1
    assert 'Nonfinite Galerkin' in h.guard_events[0]['reason']
    result = solve(h, np.ones(25))
    assert result['converged'] and result['cycles'] == 1


@pytest.mark.parametrize('differentiable', [False, True])
def test_actual_terminal_solve_failure_truncates_parent(monkeypatch, differentiable):
    from neural_amg.autograd_sparse import SparseTensor

    a = pyamg.gallery.poisson((7, 7), format='csr')
    h = build_hierarchy(a, method='classical', max_coarse=3)
    previous_terminal = h.levels[-1].a.shape[0]
    if differentiable:
        original = SparseTensor.solve
        def fail_smallest(self, b):
            if self.shape[0] == previous_terminal:
                raise RuntimeError('injected LU solve failure')
            return original(self, b)
        monkeypatch.setattr(SparseTensor, 'solve', fail_smallest)
        x = vcycle(h, torch.ones(49, dtype=torch.float64)).detach().numpy()
    else:
        class FailedSolve:
            def solve(self, b):
                raise RuntimeError('injected LU solve failure')
        h.terminal_lu = FailedSolve()
        x = _runtime_cycle(h, np.ones(49))
    assert np.isfinite(x).all()
    assert h.levels[-1].a.shape[0] > previous_terminal
    assert any(event['stage'] == 'cycle_solve' for event in h.guard_events)
    assert solve(h, np.ones(49), max_cycles=150)['converged']


def test_nonfinite_first_cycle_diagnostics_are_json_serializable():
    import json
    from neural_amg.solver import _first_cycle_bad

    bad, details = _first_cycle_bad(sp.eye(2, format='csr'), np.ones(2),
                                   np.zeros(2), np.array([np.nan, np.inf]))
    assert bad and details['residual_after'] == [None]
    json.dumps(details, allow_nan=False)


def test_first_cycle_growth_budget_allows_transient_growth_and_is_configurable():
    from neural_amg.solver import _first_cycle_bad

    a = sp.eye(2, format='csr')
    rhs, initial, result = np.ones(2), np.zeros(2), 3 * np.ones(2)
    # A factor-two initial growth is allowed by the default numerical guard.
    bad, details = _first_cycle_bad(a, rhs, initial, result)
    assert not bad and details['first_cycle_max_growth'] == 10.
    assert _first_cycle_bad(a, rhs, initial, result, max_growth=1.)[0]
    assert _first_cycle_bad(a, rhs, initial, 12 * np.ones(2))[0]
    hierarchy = build_hierarchy(a, method='classical', first_cycle_max_growth=2.)
    assert hierarchy.first_cycle_max_growth == 2.
    for value in (0.9, np.inf, np.nan, True):
        with pytest.raises(ValueError, match='first_cycle_max_growth'):
            build_hierarchy(a, method='classical', first_cycle_max_growth=value)


def test_interpolation_cannot_change_fixed_support():
    from neural_amg.autograd_sparse import SparseTensor
    from neural_amg.solver import NumericalHierarchyError, _check_interpolation
    from neural_amg.structure import Structure

    reference = sp.csr_matrix([[1., 0.], [0., 1.], [1., 0.]])
    structure = Structure(np.array([0, 1]), reference)
    # Row sums and injection are unchanged, but row 2 now selects another node.
    modified = SparseTensor.from_scipy(sp.csr_matrix([[1., 0.], [0., 1.], [0., 1.]]))
    with pytest.raises(NumericalHierarchyError, match='fixed support'):
        _check_interpolation(modified, structure)
