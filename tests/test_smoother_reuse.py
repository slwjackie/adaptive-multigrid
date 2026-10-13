"""Numerical/control fixtures only; these tests do not establish CFD speedups."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_h2_fixed_p_backend import config, state, hs
from test_h2_fixed_p_study import settings, make_fixture_recording
from adaptive_mg.v67.banks import Stats
from adaptive_mg.v67.h2_fixed_p import backend, study as base
from adaptive_mg.v67.h2_fixed_p.backend import overlay_cached_smoother
from adaptive_mg.v67.h2_fixed_p.data import read
from adaptive_mg.v67.multistage import ResidualCascadeBank
from adaptive_mg.v67.smoother_reuse import VERSION, ACTIONS
from adaptive_mg.v67.smoother_reuse import study, world
from adaptive_mg.v67.smoother_reuse.policy import ReuseSolver, observation, FEATURES
from adaptive_mg.v67.world_model.backend import levels
from adaptive_mg.v67.world_model.data import write_json


def reuse_settings():
    return dict(schema=VERSION, policies=['refresh','reuse','periodic_2','drift_0.001'],
                seed=13, measurement_repeats=1, baseline_repeats=1, behavior_max_age=8,
                probe_horizon=2, hidden=4, epochs=2, minimum_gain=.05, fit_world=True)


def test_changed_A_reuses_only_stencil_not_old_Galerkin_or_factors(hs, monkeypatch):
    solver = ReuseSolver(config(), hs, 'reuse')
    r0 = solver.step(state())
    old = solver.core.cached_smoother
    previous_matrices = [l.a.copy() for l in levels(old)]
    old_p = solver.p_digest
    import adaptive_mg.v67.hs_world.backend as parent
    monkeypatch.setattr(parent, 'prepare_smoother_bank', lambda *a: pytest.fail('NN regenerated on reuse'))
    changed = state(1, changed=True)
    result = solver.step(changed)
    assert r0.success and result.success and result.stats['smoother_reused']
    assert not result.stats['smoother_refreshed'] and solver.p_digest == old_p
    assert result.stats['smoother_age'] == 2
    for current, cached, original in zip(levels(solver.core.smoother_root), levels(old), previous_matrices):
        np.testing.assert_array_equal(cached.a.toarray(), original.toarray())
        assert current.neural_stencil is cached.neural_stencil
        assert current.a is not cached.a
        if current.coarse:
            np.testing.assert_allclose(current.coarse.a.toarray(), (current.p.T@current.a@current.p).toarray())
            assert current.cache is not cached.cache
        else:
            assert current.lu is not cached.lu
    np.testing.assert_array_equal(solver.core.smoother_root.a.toarray(), changed.a.toarray())
    assert np.linalg.norm(changed.b-changed.a@result.x) <= result.threshold
    assert result.total_seconds >= result.stats['backend_total_seconds']


def test_refresh_same_A_really_runs_NN_and_periodic_age_counts_pressure_calls(hs):
    refresh = ReuseSolver(config(), hs, 'refresh')
    assert refresh.step(state()).stats['smoother_refreshed']
    second = refresh.step(state(1))
    assert second.stats['smoother_refreshed']
    assert sum(s.get('smoother_nn_calls', 0) for s in second.stats['builds']) > 0
    periodic = ReuseSolver(config(), hs, 'periodic_2')
    choices = [periodic.step(state(i, changed=bool(i))).stats['smoother_action_selected'] for i in range(4)]
    assert choices == ['refresh','reuse','refresh','reuse']


def test_rejected_stale_stencil_clears_cache_then_forces_regeneration(hs, monkeypatch):
    solver = ReuseSolver(config(), hs, 'reuse')
    solver.step(state())
    hybrid = backend.hybrid_cycle
    monkeypatch.setattr(backend, 'hybrid_cycle', lambda root,x,*a: np.full_like(x, np.nan))
    bad = solver.step(state(1, changed=True))
    assert bad.success and bad.fallback and solver.core.cached_smoother is None
    monkeypatch.setattr(backend, 'hybrid_cycle', hybrid)
    recovered = solver.step(state(2, changed=True))
    assert recovered.success and recovered.stats['smoother_refreshed']
    assert recovered.stats['smoother_action_selected'] == 'refresh'


def test_counterfactual_fork_does_not_advance_parent_or_mutate_its_bank(hs):
    parent = ReuseSolver(config(), hs, 'reuse')
    parent.step(state())
    digest, bank = parent.p_digest, parent.core.bank
    one, two = parent.fork(), parent.fork()
    one.step(state(1, changed=True), action='refresh')
    two.step(state(1, changed=True), action='reuse')
    assert parent.previous.index == 0 and parent.core.smoother_age == 1
    assert parent.core.bank is bank
    assert one.p_digest == two.p_digest == digest
    assert one.core.cached_smoother is not parent.core.cached_smoother
    assert two.core.cached_smoother is parent.core.cached_smoother


def test_cascade_reuses_stages_but_rebinds_residual_matvec_to_current_A():
    import scipy.sparse as sp
    class Stage:
        def apply(self, r, stats):
            return .1*r
    before, current = sp.eye(3,format='csr'), 2*sp.eye(3,format='csr')
    cached = SimpleNamespace(shape=(3,1), index=0, coarse=None,
                             neural_stencil=ResidualCascadeBank((Stage(),Stage()),before))
    live = SimpleNamespace(shape=(3,1), index=0, coarse=None, neural_stencil=None, a=current)
    overlay = overlay_cached_smoother(live,cached)
    assert cached.neural_stencil.a is before and overlay.neural_stencil.a is current
    assert overlay.neural_stencil.stages is cached.neural_stencil.stages
    np.testing.assert_allclose(overlay.neural_stencil.apply(np.ones(3),Stats()), .18)


def test_observation_excludes_future_schedule_time_and_case_labels(hs):
    solver = ReuseSolver(config(),hs)
    original = state()
    secret = replace(original, time=900., index=123, context={'future_forcing':1e20, 'split':'test'})
    np.testing.assert_array_equal(observation(original,solver), observation(secret,solver))
    assert not any('time' in key or 'future' in key or 'split' in key for key in FEATURES)
    with pytest.raises(ValueError, match='validation/test'):
        study.collect_episode({'split':'test'},[original],config(),hs,reuse_settings())


def test_live_session_uses_temporal_solver_and_verifies_native_residual(hs):
    from test_h2_fixed_p_service import contract, request
    from adaptive_mg.v67.h2_fixed_p.service import Session
    cfg = config(max_cycles=80)
    solver = ReuseSolver(cfg,hs,'reuse')
    session = Session(cfg,contract(),mode='hs',expert=hs,solver=solver)
    for i in range(2):
        wire, a, exact = request(i,mode='hs')
        response = session.handle(wire)
        assert response['success'] and response['final_residual']<=response['threshold']
        assert response['step']['stats']['smoother_reused'] is bool(i)
    with pytest.raises(ValueError,match='hs service arm'):
        Session(cfg,contract(),mode='classical',solver=solver)


def test_probe_accounts_for_delayed_payback_and_retains_failed_windows():
    def rows(cost):
        return [dict(total_seconds=cost, success=True, pressure_calls=4) for _ in range(3)]
    ep = dict(available=[[True,True]]*24, probe_horizon=4,
              # Immediate refresh is ALWAYS more expensive in this fixture.
              measurements=[{'0':rows(2.),'1':rows(1.)}]*24,
              horizon_measurements=[{'0':rows(3.),'1':rows(5.)}]*12+ [{'0':rows(5.),'1':rows(3.)}]*12)
    report = study.tradeoff_report([ep],.05)
    assert report['world_model_candidate'] and report['refresh_wins'] == report['reuse_wins'] == 12
    assert report['nonoverlapping_wins'] == dict(refresh_wins=3, reuse_wins=3)
    assert not report['world_model_benefit_established']
    ep = deepcopy(ep)
    ep['horizon_measurements'][0]['0'][0]['success'] = False
    assert not study.tradeoff_report([ep],.05)['world_model_candidate']


@pytest.mark.parametrize('fit_world,memory,head', [
    (False, 'shared', 'linear'), (True, 'shared', 'linear'),
    (True, 'shared', 'role_mlp'), (True, 'independent_roles', 'role_mlp')])
def test_real_numerical_collection_training_freeze_and_sealed_test(tmp_path, monkeypatch, fit_world, memory, head):
    # Explicit test-only matrices; not recorded OpenFOAM or flame evidence.
    monkeypatch.setattr(base,'source_digest',lambda:'a'*64)
    config_path = tmp_path/'settings.json'
    cfg = settings()
    cfg['smoother_reuse'] = reuse_settings()
    cfg['smoother_reuse']['fit_world'] = fit_world
    cfg['smoother_reuse'].update(world_memory=memory, world_head=head, world_head_hidden=3)
    write_json(config_path,cfg)
    inputs = [make_fixture_recording(tmp_path/split,split,offset) for split,offset in
              [('train',0),('tune',7),('validation',15),('test',30)]]
    run = tmp_path/'run'
    base.prepare(config_path,inputs,run)
    base.train(run)
    # Gate injection tests the fit/persistence path; no performance assertion.
    probe = study.tradeoff_report
    monkeypatch.setattr(study,'tradeoff_report',lambda episodes,margin:
                        dict(probe(episodes,margin),world_model_candidate=True))
    collection = study.collect(run)
    assert not collection['test_read'] and not collection['validation_read']
    assert len(collection['episodes']) == 2
    first = read(run/'reuse'/next(iter(collection['episodes'])))
    assert first['available'][0] == [True,False]
    assert first['horizon_measurements'][0]['0'][0]['pressure_calls'] == 2
    rollout = first['horizon_measurements'][0]['0'][0]
    assert len(rollout['calls']) == 2
    assert rollout['total_seconds'] == pytest.approx(sum(r['total_seconds'] for r in rollout['calls']))
    assert all({'cycles', 'fallback', 'nn_setup_seconds', 'neural_apply_seconds'} <= r.keys()
               for r in rollout['calls'])
    trained = study.train(run)
    assert bool(trained['models']) is fit_world
    if fit_world:
        assert trained['model_details']['world']['architecture'] == dict(
            memory=memory, head=head, head_hidden=3)
        assert trained['model_details']['world']['parameter_count'] > 0
    assert not trained['world_model_benefit_established']
    with pytest.raises(FileNotFoundError):
        study.evaluate(run,'test',1)
    base.evaluate(run,split='validation',repeats=1)
    valid = study.evaluate(run,'validation',1)
    assert valid['success'] and valid['world_model_enabled'] is fit_world
    assert not valid['full_cfd_wall_clock_measured']
    base.freeze(run)
    study.freeze(run)
    assert study.evaluate(run,'test',1)['success']
    assert study.check_freeze(run)
    with pytest.raises(ValueError,match='closed'):
        study.train(run)
    label = run/'reuse'/next(iter(collection['episodes']))
    label.write_text(label.read_text()+' ')
    with pytest.raises(ValueError,match='changed'):
        study.check_freeze(run)


def model_episode(split='train'):
    rng=np.random.default_rng(15)
    return dict(split=split, obs=rng.normal(size=(5,len(FEATURES))).tolist(),
        available=[[True,True]]*5, log_cost=np.log([[.02,.01]]*5).tolist(),
        success=[[True,True]]*5, next_obs=rng.normal(size=(5,2,len(FEATURES))).tolist(),
        behavior=[0,1,1,0,1])


def test_world_fit_uses_only_train_and_prediction_rejects_nonfinite():
    with pytest.raises(ValueError,match='training'):
        world.fit([model_episode('test')],epochs=1)
    artifact = world.fit([model_episode()],hidden=4,epochs=2)
    with pytest.raises(ValueError,match='tune'):
        world.calibrate(artifact,[model_episode('train')])
    predictor = world.Predictor(artifact)
    raw = model_episode()['obs'][0]
    result, _ = predictor.choose(raw,'refresh')
    assert result in ACTIONS
    with torch.no_grad():
        predictor.net.head.weight.fill_(float('nan'))
    action, info = predictor.choose(raw,'reuse')
    assert action == 'reuse' and info['reason'] == 'nonfinite_prediction'
    assert predictor.state is None
