"""Fixed-P invariants; no speedup or combustion claim follows from these tests."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.pde import DiffusionCase, assemble_stiffness
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.h2_fixed_p import backend as mod
from adaptive_mg.v67.h2_fixed_p.backend import FixedPSolver, hierarchy_p_digest
from adaptive_mg.v67.research_training import create_research_components
from adaptive_mg.v67.world_model.backend import levels
from adaptive_mg.v67.world_model.data import snapshot


def config(**kwargs):
    mg = MGConfig(mode='classical', strategy_name='line_alt_energymin_full__em5__v11',
                  max_cycles=60, stencil_backend='csr')
    return AdaptiveConfig(mg=replace(mg, **kwargs), mode='research', branch='H_S',
                          use_transfer=False, use_smoother=True, spatial=False,
                          gate_mode='open', use_learned_controller=False)


def state(index=0, *, changed=False, boundary='dirichlet', mesh='grid', layout=None):
    n = 7
    a = assemble_stiffness(DiffusionCase(n, epsilon=.8, angle_deg=20., contrast=2., pattern='channel'))
    if changed:
        a = a.copy()
        a.setdiag(a.diagonal() + np.linspace(.2, .8, n*n))
    exact = np.sin(np.arange(n*n) + .3)
    return snapshot(a, a @ exact, shape=(n,n), time=.1*index, index=index,
                    mesh_id=mesh, boundary_id=boundary,
                    context={} if layout is None else dict(layout=layout))


@pytest.fixture
def hs():
    # Tiny numerical test fixture, NOT a research performance checkpoint. Do one
    # real gradient step so tests do not admit an untouched initialized network.
    torch.set_num_threads(1)
    m = create_research_components(smoother='student_cnn', smoother_hidden=4,
                                   transfer_hidden=4, seed=41)
    opt = torch.optim.SGD(m.smoother.parameters(), lr=1e-3)
    opt.zero_grad()
    sum((p.square().sum() for p in m.smoother.parameters())).backward()
    opt.step()
    m.metadata.update(training_branch='H_S', optimizer_updates=1, smoother_trained=True)
    return m.eval()


def test_changed_A_keeps_all_P_refreshes_Ac_factors_and_old_bank(monkeypatch):
    solver = FixedPSolver(config())
    r0 = solver.step(state())
    old = solver.classical_root
    old_levels = list(levels(old))
    old_matrices = [l.a.copy() for l in old_levels]
    p0 = hierarchy_p_digest(old)
    import adaptive_mg.v67.hs_world.backend as parent
    monkeypatch.setattr(parent, 'build_fixed_hierarchy', lambda *a, **k: pytest.fail('P rebuilt after first system'))
    s1 = state(1, changed=True)
    r1 = solver.step(s1)
    assert r0.success and r1.success
    assert hierarchy_p_digest(solver.classical_root) == p0
    assert r1.stats['hierarchy_p_digest_initial'] == r1.stats['hierarchy_p_digest_final'] == p0
    new_levels = list(levels(solver.classical_root))
    for previous, new, matrix in zip(old_levels, new_levels, old_matrices):
        np.testing.assert_array_equal(previous.a.toarray(), matrix.toarray())
        assert previous is not new
        if new.coarse is not None:
            np.testing.assert_array_equal(previous.p.toarray(), new.p.toarray())
            np.testing.assert_allclose(new.coarse.a.toarray(), (new.p.T @ new.a @ new.p).toarray(), atol=1e-12)
            assert new.cache is not previous.cache
        else:
            assert new.lu is not previous.lu
            np.testing.assert_allclose(new.a @ new.lu.solve(np.ones(new.a.shape[0])), 1., atol=1e-12)
    assert r1.stats['builds'][0]['numeric_refactorized']
    assert np.linalg.norm(s1.b - s1.a @ r1.x) <= r1.threshold


def test_classical_and_HS_share_exact_P_and_refresh_current_smoother(hs):
    c = FixedPSolver(config())
    h = FixedPSolver(config(), hs)
    c0, h0 = c.step(state()), h.step(state())
    old = h.smoother_root
    assert c0.stats['hierarchy_p_digest_initial'] == h0.stats['hierarchy_p_digest_initial']
    assert h0.stats['neural_trial_cycles'] > 0
    assert h0.stats['actual_neural_used']
    h1 = h.step(state(1, changed=True))
    assert h1.stats['hierarchy_p_preserved']
    assert h.smoother_root is not old
    assert h1.stats['builds'][1]['smoother_nn_calls'] > 0
    for base, neural in zip(levels(h.classical_root), levels(h.smoother_root)):
        assert base.p is neural.p and base.a is neural.a
        assert not getattr(neural, 'learned_transfer', False)


def test_rejection_rolls_back_only_last_trial_same_P_original_tolerance_total_budget(hs, monkeypatch):
    cfg = config(max_cycles=3, tolerance=1e-9)
    s = state()
    exact = np.linalg.solve(s.a.toarray(), s.b)
    neural_inputs, classical_inputs = [], []

    def neural(root, x, b, cfg, stats, *args):
        neural_inputs.append(x.copy())
        stats.neural_apply_calls += 1
        if len(neural_inputs) == 1:
            return .5 * exact
        x[:] = 1e99  # Trial may mutate its input; accepted iterate must survive.
        return np.full_like(x, np.nan)

    def classical(root, x, b, cfg, stats):
        classical_inputs.append((x.copy(), hierarchy_p_digest(root)))
        return exact.copy()

    monkeypatch.setattr(mod, 'hybrid_cycle', neural)
    monkeypatch.setattr(mod, 'classical_cycle', classical)
    solver = FixedPSolver(cfg, hs)
    r = solver.step(s)
    assert r.success and r.fallback and r.reason == 'residual_rejection'
    assert r.cycles == cfg.mg.max_cycles == 3
    assert r.stats['neural_trial_cycles'] == 2
    assert r.stats['accepted_neural_cycles'] == 1
    assert r.stats['rejected_neural_cycles'] == 1
    assert r.stats['classical_trial_cycles'] == 1
    np.testing.assert_allclose(classical_inputs[0][0], .5 * exact)
    assert classical_inputs[0][1] == solver.initial_p_digest
    assert r.threshold == pytest.approx(cfg.mg.tolerance * np.linalg.norm(s.b))
    assert len(r.residuals) == 3  # Rejected residual is not an accepted state.
    assert r.actual_action == 'H_S_THEN_C'


def test_failure_retains_P_and_does_not_reset_budget_or_rebuild(monkeypatch):
    cfg = config(max_cycles=1, tolerance=1e-14)
    solver = FixedPSolver(cfg)
    monkeypatch.setattr(mod, 'classical_cycle', lambda root, x, *args: x)
    r0 = solver.step(state())
    assert not r0.success and r0.cycles == 1 and solver.bank is not None
    r1 = solver.step(state(1, changed=True))
    assert not r1.success and r1.cycles == 1
    assert r1.stats['hierarchy_p_digest_final'] == r0.stats['hierarchy_p_digest_initial']
    assert r1.stats['builds'][0]['base_hierarchy_builds'] == 0


def test_stagnation_uses_same_P_recovery(hs, monkeypatch):
    s = state()
    exact = np.linalg.solve(s.a.toarray(), s.b)
    solver = FixedPSolver(config(stagnation_patience=1), hs)

    def neural(root, x, b, cfg, stats, *args):
        stats.neural_apply_calls += 1
        return x

    monkeypatch.setattr(mod, 'hybrid_cycle', neural)
    monkeypatch.setattr(mod, 'classical_cycle', lambda root, x, *a: exact.copy())
    result = solver.step(s)
    assert result.success and result.reason == 'stagnation_recovery'
    assert result.stats['neural_trial_cycles'] == result.stats['accepted_neural_cycles'] == 1
    assert result.stats['hierarchy_p_preserved']
    assert result.cycles == 2


def test_same_matrix_new_RHS_reuses_numeric_factors(hs):
    solver = FixedPSolver(config(), hs)
    solver.step(state())
    base, neural = solver.classical_root, solver.smoother_root
    s = state(1)
    s = snapshot(s.a, 1.2*s.b, shape=s.shape, time=s.time, index=s.index,
                 mesh_id=s.mesh_id, boundary_id=s.boundary_id)
    result = solver.step(s)
    assert solver.classical_root is base and solver.smoother_root is neural
    assert all(b.get('smoother_nn_calls', 0) == 0 for b in result.stats['builds'])
    assert all(b.get('exact_matrix_cache_hit') for b in result.stats['builds'])


def test_changed_sparsity_rejected_even_when_shape_mesh_boundary_match():
    solver = FixedPSolver(config())
    solver.step(state())
    s = state(1)
    a = s.a.copy()
    i, j = np.argwhere((a.toarray() != 0) & ~np.eye(a.shape[0], dtype=bool))[0]
    a[i, j] = a[j, i] = 0
    a.eliminate_zeros()
    changed = snapshot(a, s.b, shape=s.shape, time=s.time, index=s.index,
                       mesh_id=s.mesh_id, boundary_id=s.boundary_id)
    with pytest.raises(ValueError, match='topology'):
        solver.step(changed)


@pytest.mark.parametrize('kwargs', [dict(boundary='new'), dict(mesh='new'), dict(layout='unstructured')])
def test_incompatible_trajectory_rejected_not_reset(kwargs):
    solver = FixedPSolver(config())
    solver.step(state())
    old = solver.bank
    with pytest.raises(ValueError, match='layout|boundary|topology'):
        solver.step(state(1, **kwargs))
    assert solver.bank is old


def test_order_and_reset_are_explicit():
    solver = FixedPSolver(config())
    solver.step(state())
    with pytest.raises(ValueError, match='order'):
        solver.step(state())
    solver.reset()
    assert solver.bank is None and solver.initial_p_digest is None
    assert solver.step(state(boundary='new-trajectory')).success


def test_untrained_transfer_nonfinite_and_disabled_HS_rejected(hs):
    untrained = create_research_components(smoother='student_cnn', smoother_hidden=4, transfer_hidden=4)
    with pytest.raises(ValueError, match='trained H_S'):
        FixedPSolver(config(), untrained)
    with pytest.raises(ValueError, match='transfer'):
        FixedPSolver(replace(config(), use_transfer=True))
    hs.metadata['training_branch'] = 'H_P'
    with pytest.raises(ValueError, match='trained H_S'):
        FixedPSolver(config(), hs)
    hs.metadata['training_branch'] = 'H_S'
    with pytest.raises(ValueError, match='enable actual'):
        FixedPSolver(replace(config(), replace_pre=0), hs)
    with torch.no_grad():
        next(hs.smoother.parameters()).fill_(float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        FixedPSolver(config(), hs)


def test_setup_failure_keeps_first_classical_P(hs, monkeypatch):
    import adaptive_mg.v67.hs_world.backend as parent
    monkeypatch.setattr(parent, 'prepare_smoother_bank', lambda *a, **k: (_ for _ in ()).throw(ValueError('bad stencil')))
    solver = FixedPSolver(config(), hs)
    r0 = solver.step(state())
    r1 = solver.step(state(1, changed=True))
    assert r0.success and r0.fallback and r0.stats['neural_trial_cycles'] == 0
    assert r1.success and r1.fallback
    assert r0.stats['hierarchy_p_digest_initial'] == r1.stats['hierarchy_p_digest_final']
    assert r1.stats['builds'][0]['base_hierarchy_builds'] == 0


def test_timing_phases_are_disjoint_and_no_neural_work_in_C():
    r = FixedPSolver(config()).step(state())
    keys = ['feature_seconds', 'setup_seconds', 'solve_seconds', 'verification_seconds',
            'recovery_seconds', 'orchestration_seconds']
    assert all(r.stats[k] >= 0 for k in keys)
    assert sum(r.stats[k] for k in keys) == pytest.approx(r.total_seconds)
    assert r.stats['neural_trial_cycles'] == r.stats['neural_apply_calls'] == 0
    assert r.stats['smoother_nn_calls'] == 0
    assert r.stats['recovery_seconds'] == 0
    assert r.record()['hierarchy_p_digest'] == r.stats['hierarchy_p_digest_final']
