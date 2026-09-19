from dataclasses import replace
import importlib
import numpy as np
import pytest
from adaptive_mg import *
from adaptive_mg.hierarchy import classical_cycle, neural_cycle, hierarchy_identity, WorkStats
from adaptive_mg.provenance import operator_digest
solver_module=importlib.import_module('adaptive_mg.solver')


def cfg(**kw):
    d=dict(mode='fixed',fixed_k=2,max_cycles=6,tolerance=1e-30,absolute_tolerance=0.,near_tolerance_factor=0.,stagnation_patience=50,stencil_backend='csr')
    d.update(kw);return MGConfig(**d)


@pytest.mark.parametrize('strategy',[s.name for s in STRATEGIES])
def test_zero_is_exact_classical(problem,model,strategy):
    a,b,n=problem;c=cfg(strategy_name=strategy,fixed_k=0,max_cycles=3)
    expected=PreparedTemporalMGSolver(a,n,model,replace(c,mode='classical')).solve(b)
    prepared=PreparedTemporalMGSolver(a,n,model,c);actual=prepared.solve(b)
    np.testing.assert_array_equal(expected.x,actual.x)
    assert expected.residual_history==actual.residual_history
    assert actual.stats['neural_setup_calls']==actual.stats['neural_apply_calls']==0
    assert actual.stats['predictor_calls']==0 and actual.switch_events==[]
    assert prepared.root.neural_stencil is None


@pytest.mark.parametrize('k',[1,2,3,5])
def test_prefix_and_permanent_tail(problem,model,k):
    a,b,n=problem;conf=cfg(fixed_k=k)
    p=PreparedTemporalMGSolver(a,n,model,conf);identity=hierarchy_identity(p.root)
    result=p.solve(b)
    assert result.cycle_path==['neural']*k+['classical']*(6-k)
    assert len(result.switch_events)==1
    assert result.switch_events[0]['first_classical_cycle']==k+1
    assert identity==hierarchy_identity(p.root)
    assert result.stats['switch_hierarchy_builds']==0
    assert result.stats['detector_calls']==0 and result.stats['spatial_gate_calls']==0
    assert result.stats['accepted_neural_cycles']==k
    assert result.stats['neural_setup_calls']==2
    # Continue explicitly from the NN prefix using the SAME classical algorithm.
    x=np.zeros_like(b);counter=WorkStats()
    manual=PreparedTemporalMGSolver(a,n,model,conf)
    for _ in range(k):x=neural_cycle(manual.root,x,b,model.smoother,conf,counter)
    for _ in range(6-k):x=classical_cycle(manual.root,x,b,conf,counter)
    np.testing.assert_array_equal(x,result.x)


@pytest.mark.parametrize('kind',['growth','nan','infinity'])
def test_bad_neural_trial_rolled_back(problem,model,monkeypatch,kind):
    a,b,n=problem;conf=cfg()
    called=[]
    def bad(level,x,b,*args):
        called.append(1)
        return x-np.ones_like(x)*1e12 if kind=='growth' else np.full_like(x,np.nan if kind=='nan' else np.inf)
    monkeypatch.setattr(solver_module,'neural_cycle',bad)
    r=PreparedTemporalMGSolver(a,n,model,conf).solve(b)
    c=PreparedTemporalMGSolver(a,n,model,replace(conf,mode='classical')).solve(b)
    np.testing.assert_array_equal(r.x,c.x)
    assert r.residual_history==c.residual_history
    assert len(called)==1
    assert r.stats['attempted_neural_cycles']==1 and r.stats['rejected_neural_cycles']==1
    assert r.stats['classical_recovery_cycles']==1 and r.stats['classical_cycles']==6
    assert r.cycle_path==['classical_recovery']+['classical']*5
    assert len(r.switch_events)==1
    assert r.attempts[0]['rejected']


def test_stagnation_switches_only_once(problem,model,monkeypatch):
    a,b,n=problem
    monkeypatch.setattr(solver_module,'neural_cycle',lambda level,x,*args:x.copy())
    conf=cfg(fixed_k=5,stagnation_patience=2)
    r=PreparedTemporalMGSolver(a,n,model,conf).solve(b)
    assert r.cycle_path==['neural']*2+['classical']*4
    assert r.switch_events[0]['reason']=='stagnation'
    c=PreparedTemporalMGSolver(a,n,model,replace(conf,mode='classical',max_cycles=4)).solve(b)
    np.testing.assert_array_equal(r.x,c.x)


def test_initial_solution_skips_predictor_and_nn(problem,model):
    a,b,n=problem;x=np.ones_like(b)
    p=PreparedTemporalMGSolver(a,n,model,cfg())
    r=p.solve(a@x,x0=x)
    assert r.converged and r.mg_cycles==0 and r.k_nn==0
    assert r.stats['neural_setup_calls']==r.stats['predictor_calls']==0


def test_near_tolerance_switch_skips_bank(problem,model):
    a,b,n=problem
    r=PreparedTemporalMGSolver(a,n,model,cfg(tolerance=.1,near_tolerance_factor=20.)).solve(b)
    assert r.switch_events[0]['reason']=='near_tolerance'
    assert r.stats['neural_setup_calls']==0
    assert all(path=='classical' for path in r.cycle_path)


def test_reserve_tail_and_budget_one(problem,model):
    a,b,n=problem
    r=PreparedTemporalMGSolver(a,n,model,cfg(fixed_k=6,max_cycles=1)).solve(b)
    assert r.k_nn==0 and r.decision['requested_k']==6
    assert r.cycle_path==['classical']


def test_finite_tolerance_and_residual_reporting(problem,model):
    a,b,n=problem
    r=PreparedTemporalMGSolver(a,n,model,MGConfig(mode='fixed',fixed_k=2)).solve(b)
    measured=np.linalg.norm(b-a@r.x)
    assert measured==pytest.approx(r.final_true_residual,rel=1e-12)
    assert r.converged and measured<=r.stopping_threshold
    assert r.stats['accepted_neural_cycles']+r.stats['classical_cycles']==r.mg_cycles


@pytest.mark.parametrize('change',[
    {'mode':'adaptive'},{'fixed_k':-1},{'max_cycles':0},{'max_nn_cycles':-1},
    {'tolerance':float('nan')},{'tolerance':0.},{'safety_growth':.5},
    {'stagnation_patience':0},{'nn_levels':-2},{'device':'cuda'},
    {'native_threads':0},{'reserve_classical_cycles':0},{'fixed_k':1.5}])
def test_invalid_configuration(change):
    with pytest.raises(ValueError):MGConfig(**change)


def test_no_forced_final_classical_when_nn_already_converged(problem,model,monkeypatch):
    import scipy.sparse.linalg as spla
    a,b,n=problem
    exact=spla.spsolve(a,b)
    monkeypatch.setattr(solver_module,'neural_cycle',lambda *args:exact.copy())
    r=PreparedTemporalMGSolver(a,n,model,MGConfig(mode='fixed',fixed_k=4)).solve(b)
    assert r.converged and r.cycle_path==['neural'] and r.switch_events==[]


def test_base_matrices_unchanged(problem,model):
    a,b,n=problem;p=PreparedTemporalMGSolver(a,n,model,cfg())
    root=p.root;before=[]
    while root is not None:
        before.append((operator_digest(root.a),None if root.p is None else operator_digest(root.p)))
        root=root.coarse
    p.solve(b);after=[];root=p.root
    while root is not None:
        after.append((operator_digest(root.a),None if root.p is None else operator_digest(root.p)))
        root=root.coarse
    assert before==after


@pytest.mark.parametrize('setting',[{'nn_levels':0},{'smoother_gain_multiplier':0.}])
def test_no_configured_neural_work_is_exact_classical(problem,model,setting):
    a,b,n=problem;configuration=cfg(**setting)
    r=PreparedTemporalMGSolver(a,n,model,configuration).solve(b)
    c=PreparedTemporalMGSolver(a,n,model,replace(configuration,mode='classical')).solve(b)
    assert r.k_nn==0 and r.stats['neural_apply_calls']==0
    np.testing.assert_array_equal(r.x,c.x)


def test_rejected_cycles_are_not_hidden(problem,model,monkeypatch):
    a,b,n=problem;monkeypatch.setattr(solver_module,'neural_cycle',lambda level,x,*args:np.full_like(x,np.nan))
    r=PreparedTemporalMGSolver(a,n,model,cfg()).solve(b)
    assert r.mg_cycles==6 and r.executed_mg_cycles==7
    assert r.to_dict()['executed_mg_cycles']==7
