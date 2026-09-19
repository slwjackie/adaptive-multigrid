"""Numerical and cache contracts for the limited C/S/P/SP patch."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from adaptive_mg import DiffusionCase, MGConfig, assemble_stiffness
from adaptive_mg.provenance import module_signature, hardware_environment
from adaptive_mg.solver import PreparedTemporalMGSolver
from adaptive_mg.v67 import AdaptiveConfig, Components, PreparedAdaptiveMG
from adaptive_mg.v67.banks import Stats, hybrid_cycle
from adaptive_mg.v67.spatial import SpatialState
import adaptive_mg.v67.solver as solver_module


def problem(n=7, strategy='jacobi_bilinear_full', **kw):
    a=assemble_stiffness(DiffusionCase(n=n,epsilon=.7,contrast=1.))
    b=np.random.default_rng(19).normal(size=n*n)
    mg=MGConfig(mode='classical',strategy_name=strategy,pre_steps=2,post_steps=2,
        max_cycles=5,tolerance=1e-14,absolute_tolerance=0.,nn_levels=2,near_tolerance_factor=0.,stencil_backend='csr')
    cfg=AdaptiveConfig(mg=mg,mode='research',branch='H_SP',spatial=False,gate_mode='open',**kw)
    return a,b,Components.create(hidden=4),cfg


def fail(*args,**kwargs):
    raise AssertionError('forbidden learned work')


@pytest.mark.parametrize('cold',['C','closed','disabled','uncertified'])
def test_cold_bypass_matches_original_classical_without_state_objects(monkeypatch,cold):
    a,b,c,cfg=problem()
    if cold=='C':cfg=replace(cfg,branch='C')
    elif cold=='closed':cfg=replace(cfg,gate_mode='closed')
    elif cold=='disabled':cfg=replace(cfg,branch='auto',use_smoother=False,use_transfer=False)
    else:cfg=replace(cfg,branch='auto',mode='production')
    monkeypatch.setattr(c,'signature',fail)
    monkeypatch.setattr(c.controller,'compile',fail)
    monkeypatch.setattr(c.detector,'compile',fail)
    monkeypatch.setattr(solver_module,'OnlineHistory',fail)
    monkeypatch.setattr(solver_module,'SpatialState',fail)
    x0=b*.01
    old=PreparedTemporalMGSolver(a,7,None,cfg.mg).solve(b,x0=x0)
    result=PreparedAdaptiveMG(a,7,c,cfg).solve(b,x0)
    np.testing.assert_array_equal(result.x,old.x)
    assert result.residual_history==old.residual_history
    assert result.converged==old.converged
    assert result.stats['controller_calls']==result.stats['nn_forward_calls']==0
    assert result.stats['classical_bypass_cycles']==result.cycles


@pytest.mark.parametrize('branch',['H_S','H_P','H_SP'])
def test_forced_branch_not_overridden_by_false_certificate(branch):
    a,b,c,cfg=problem();cfg=replace(cfg,branch=branch)
    r=PreparedAdaptiveMG(a,7,c,cfg).solve(b)
    assert not c.metadata['certificate']['validated']
    assert r.stats['branch_'+branch+'_attempts']>0
    assert r.stats['learned_operator_applications']>0
    assert r.stats['controller_calls']==r.stats['controller_nn_calls']==0
    assert r.branch_policy_status=='forced_component_with_safety'


def test_s_only_never_generates_transfer_and_shares_classical_hierarchy(monkeypatch):
    a,b,c,cfg=problem(n=15);cfg=replace(cfg,branch='H_S')
    monkeypatch.setattr(c.transfer,'forward',fail)
    p=PreparedAdaptiveMG(a,15,c,cfg);r=p.solve(b)
    assert r.stats['transfer_nn_calls']==r.stats['transfer_bank_builds']==0
    current,base=p.learned,p.classical
    while current is not None:
        assert current.a is base.a and current.p is base.p and current.r is base.r
        assert current.cache is base.cache and current.lu is base.lu
        assert base.neural_stencil is None
        current,base=current.coarse,base.coarse


def test_p_only_never_generates_smoother_or_detector(monkeypatch):
    a,b,c,cfg=problem();cfg=replace(cfg,branch='H_P',spatial=True,gate_mode='learned')
    monkeypatch.setattr(c.smoother,'direction_and_gain',fail)
    monkeypatch.setattr(c.detector,'compile',fail)
    r=PreparedAdaptiveMG(a,7,c,cfg).solve(b)
    assert r.stats['smoother_nn_calls']==r.stats['smoother_bank_builds']==r.stats['detector_calls']==0
    assert r.stats['learned_transfer_apply_calls']>0
    assert r.p_bank=='learned'


def test_sp_prepared_once_no_inner_heavy_forward_or_signature(monkeypatch):
    a,b,c,cfg=problem();p=PreparedAdaptiveMG(a,7,c,cfg);st=Stats()
    p.ensure_branch('H_SP',st)
    assert st.transfer_bank_builds==st.smoother_bank_builds==1
    monkeypatch.setattr(c.smoother,'direction_and_gain',fail)
    monkeypatch.setattr(c.transfer,'forward',fail)
    calls=[];signature=c.signature
    monkeypatch.setattr(c,'signature',lambda:(calls.append(1),signature())[1])
    with patch.object(solver_module,'module_signature',side_effect=fail):
        r=p.solve(b)
    assert len(calls)==1
    assert r.stats['smoother_nn_calls']==r.stats['transfer_nn_calls']==0


def test_shared_learned_p_and_actual_coarse_factor_dependency():
    a,b,c,cfg=problem(n=15,strategy='line_alt_bilinear_full')
    with torch.no_grad():c.transfer.head.bias.copy_(torch.linspace(-.3,.3,16))
    p=PreparedAdaptiveMG(a,15,c,cfg);st=Stats()
    hp=p.ensure_branch('H_P',st);hsp=p.ensure_branch('H_SP',st);hs=p.ensure_branch('H_S',st)
    assert st.transfer_bank_builds==1
    assert hsp.p is hp.p and hsp.r is hp.r and hsp.coarse.a is hp.coarse.a
    assert np.max(np.abs((hp.coarse.a-p.classical.coarse.a).data))>1e-9
    assert hsp.coarse.cache is hp.coarse.cache
    assert hsp.coarse.cache is not hs.coarse.cache
    level=hp
    while level.coarse is not None:
        assert level.p.shape==(level.a.shape[0],level.coarse.a.shape[0])
        np.testing.assert_allclose(level.r.toarray(),level.p.T.toarray(),atol=0.)
        np.testing.assert_allclose(level.coarse.a.toarray(),(level.r@level.a@level.p).toarray(),rtol=1e-12,atol=1e-12)
        assert level.a.dtype==level.p.dtype==level.r.dtype==np.float64
        assert level.cache.a is level.a
        for block in level.cache.get('x'):
            rhs=np.arange(len(block.indices),dtype=float)
            mat=level.a[block.indices][:,block.indices]
            np.testing.assert_allclose(mat@block.solve(rhs),rhs,atol=1e-10)
        level=level.coarse


def test_component_changes_invalidate_only_dependent_banks():
    a,b,c,cfg=problem();p=PreparedAdaptiveMG(a,7,c,cfg);st=Stats()
    hs=p.ensure_branch('H_S',st);hp=p.ensure_branch('H_P',st);hsp=p.ensure_branch('H_SP',st);base=p.classical
    with torch.no_grad():c.smoother.gain_head.bias.add_(.01)
    p.solve(b)
    assert p.classical is base and p.learned is not hsp
    assert p.ensure_branch('H_P',st) is hp
    hs2=p.ensure_branch('H_S',st);assert hs2 is not hs
    with torch.no_grad():c.transfer.head.bias.add_(.01)
    p.solve(b)
    assert p.ensure_branch('H_S',st) is hs2
    assert p.ensure_branch('H_P',st) is not hp
    assert p.classical is base


@pytest.mark.parametrize('change',['dtype','device','gain','nn_levels','native_threads','A','base_config'])
def test_generation_dependency_changes_never_reuse_stale_banks(change):
    a,b,c,cfg=problem();p=PreparedAdaptiveMG(a,7,c,cfg);p.solve(b)
    old=p.learned;oldp=old.p;base=p.classical
    if change=='dtype':p.config=replace(cfg,inference_dtype='float64')
    elif change=='device':p.config=replace(cfg,inference_device='auto') # resolves to MPS on this Mac
    elif change=='gain':p.config=replace(cfg,mg=replace(cfg.mg,smoother_gain_multiplier=.8))
    elif change=='nn_levels':p.config=replace(cfg,mg=replace(cfg.mg,nn_levels=1))
    elif change=='native_threads':p.config=replace(cfg,mg=replace(cfg.mg,native_threads=2))
    elif change=='A':a.data*=1.01
    else:p.config=replace(cfg,mg=replace(cfg.mg,operator_sweeps=cfg.mg.operator_sweeps+1))
    if change=='device':
        # Inject another resolved generation device key without requiring accelerator availability.
        with patch.object(solver_module,'resolve_device',return_value=torch.device('cpu:0')):p.solve(b)
    else:p.solve(b)
    assert p.learned is not old
    if change in {'gain','native_threads'}:assert p.learned.p is oldp
    if change in {'A','base_config'}:assert p.classical is not base
    else:assert p.classical is base


def test_p_bank_survives_policy_and_replacement_budget_changes():
    a,b,c,cfg=problem();p=PreparedAdaptiveMG(a,7,c,cfg);p.solve(b);old=p.learned
    p.config=replace(cfg,replace_pre=2,replace_post=1,gate_refresh=2,record_trace=False)
    r=p.solve(b)
    assert p.learned is old
    assert r.stats['smoother_nn_calls']==r.stats['transfer_nn_calls']==0


def test_generation_does_not_mutate_checkpoint_dtype():
    a,b,c,cfg=problem(inference_dtype='float64');signature=c.signature()
    p=PreparedAdaptiveMG(a,7,c,cfg);p.solve(b);p.solve(b)
    assert c.signature()==signature
    assert next(c.smoother.parameters()).dtype==torch.float32
    assert p.learned_builds_total==1


def test_lock_preserves_accepted_iterate_remaining_budget_and_stops_policy(monkeypatch):
    a,b,c,cfg=problem();cfg=replace(cfg,branch='auto',mode='hybrid');calls=[]
    def select(self,*args):
        calls.append(1)
        assert len(calls)<=2
        return ('HYBRID' if len(calls)==1 else 'CLASSICAL_LOCK'),{'reason':'test_lock'}
    monkeypatch.setattr(solver_module.OnlineHistory,'select',select)
    p=PreparedAdaptiveMG(a,7,c,cfg);p.ensure_learned(Stats())
    first=hybrid_cycle(p.learned,np.zeros_like(b),b,cfg,Stats(),SpatialState(c,cfg),1)
    tail=PreparedAdaptiveMG(a,7,c,replace(cfg,branch='C',mg=replace(cfg.mg,max_cycles=4))).solve(b,first)
    r=p.solve(b)
    assert len(calls)==2 and r.stats['classical_lock_cycles']==4
    assert r.abstention['locked_after_attempt']==1
    assert r.executed_cycles==5 and r.stats['accepted_neural_cycles']==1
    np.testing.assert_array_equal(r.x,tail.x)
    assert r.residual_history[1:]==tail.residual_history


def test_rejected_attempt_consumes_budget_and_no_later_learned_work(monkeypatch):
    a,b,c,cfg=problem();cfg=replace(cfg,mg=replace(cfg.mg,smoother_gain_multiplier=1e80,max_cycles=4))
    calls=[];original=c.smoother.direction_and_gain
    monkeypatch.setattr(c.smoother,'direction_and_gain',lambda f:(calls.append(1),original(f))[1])
    r=PreparedAdaptiveMG(a,7,c,cfg).solve(b)
    q=PreparedAdaptiveMG(a,7,c,replace(cfg,branch='C',mg=replace(cfg.mg,max_cycles=3))).solve(b)
    assert len(calls)==1 and r.stats['rejected_neural_cycles']==1
    assert r.executed_cycles==4 and r.cycles==3
    assert r.abstention['locked_after_attempt']==1
    assert r.stats['classical_lock_cycles']==2 and r.stats['classical_recovery_cycles']==1
    assert r.stats['nn_setup_seconds']>0 and r.stats['operator_generation_seconds']>0
    np.testing.assert_array_equal(r.x,q.x)


def test_multiple_rhs_share_setup_but_reset_state():
    a,b,c,cfg=problem();p=PreparedAdaptiveMG(a,7,c,cfg)
    results=p.solve_many(np.stack((b,2*b,-b)))
    assert sum(r.stats['transfer_bank_builds'] for r in results)==1
    assert sum(r.stats['smoother_bank_builds'] for r in results)==1
    for rhs,r in zip((b,2*b,-b),results):
        single=PreparedAdaptiveMG(a,7,c,cfg).solve(rhs)
        np.testing.assert_array_equal(single.x,r.x)
        assert r.stats['hybrid_entries']==single.stats['hybrid_entries']


@pytest.mark.parametrize('branch',['H_S','H_P','H_SP'])
def test_single_coarse_cycle_and_replacement_budget(branch):
    a,b,c,cfg=problem();cfg=replace(cfg,branch=branch,use_smoother=branch!='H_P',use_transfer=branch!='H_S')
    p=PreparedAdaptiveMG(a,7,c,cfg);root=p.ensure_branch(branch,Stats());stats=Stats()
    hybrid_cycle(root,np.zeros_like(b),b,cfg,stats,SpatialState(c,cfg),1)
    assert stats.coarse_solves==stats.coarse_path_count==1
    assert stats.replacement_calls==(0 if branch=='H_P' else 1)
    assert stats.classical_smooth_calls==(4 if branch=='H_P' else 3)
    assert stats.learned_transfer_apply_calls==(0 if branch=='H_S' else 2)


@pytest.mark.parametrize('smoother,transfer',[(True,False),(False,True)])
def test_component_auto_is_explicitly_uncalibrated(smoother,transfer):
    a,b,c,cfg=problem();c.metadata['controller_trained']=True
    cfg=replace(cfg,branch='auto',use_smoother=smoother,use_transfer=transfer)
    r=PreparedAdaptiveMG(a,7,c,cfg).solve(b)
    assert r.branch_policy_status=='uncalibrated_branch'
    assert r.stats['controller_calls']==r.stats['nn_forward_calls']==0


def test_stale_expert_signature_bypasses_auto():
    a,b,c,cfg=problem();c.metadata['controller_trained']=True
    c.metadata['controller_expert_signatures']={k:module_signature(getattr(c,k)) for k in ('smoother','transfer')}
    with torch.no_grad():c.smoother.gain_head.bias.add_(.01)
    r=PreparedAdaptiveMG(a,7,c,replace(cfg,branch='auto')).solve(b)
    assert r.branch_policy_status=='stale_expert_policy'
    assert r.stats['controller_calls']==0


def test_default_certificate_scope_stays_backward_compatible():
    cfg=AdaptiveConfig();scope=cfg.certification_scope()
    assert 'branch' not in scope
    assert replace(cfg,branch='H_S').certification_scope()['branch']=='H_S'


@pytest.mark.parametrize('field',['policy_generation_signature','controller_generation_signature','detector_generation_signature'])
def test_stale_generation_provenance_bypasses_auto(field):
    a,b,c,cfg=problem();c.metadata['controller_trained']=True
    c.metadata[field]=c.generation_signature()
    with torch.no_grad():c.transfer.head.bias.add_(.01)
    result=PreparedAdaptiveMG(a,7,c,replace(cfg,branch='auto')).solve(b)
    assert result.branch_policy_status=='stale_expert_policy'
    assert result.stats['nn_forward_calls']==0


def test_temporary_classical_decisions_can_reenter_hybrid(monkeypatch):
    a,b,c,cfg=problem();cfg=replace(cfg,branch='auto',mode='hybrid',mg=replace(cfg.mg,max_cycles=6))
    sequence=iter(('CLASSICAL','HYBRID','CLASSICAL','HYBRID','CLASSICAL_LOCK'))
    seen=[]
    def select(self,*args):
        state=next(sequence);seen.append(state);return state,{'reason':'test_temporal_reentry'}
    monkeypatch.setattr(solver_module.OnlineHistory,'select',select)
    result=PreparedAdaptiveMG(a,7,c,cfg).solve(b)
    assert seen==['CLASSICAL','HYBRID','CLASSICAL','HYBRID','CLASSICAL_LOCK']
    assert result.stats['hybrid_entries']==2
    assert result.stats['accepted_neural_cycles']==2
    assert result.stats['classical_lock_cycles']==2


@pytest.mark.parametrize('eager',[False,True])
def test_failed_generation_retains_setup_time_and_actual_forward_counts(monkeypatch,eager):
    a,b,c,cfg=problem();cfg=replace(cfg,lazy=not eager,mg=replace(cfg.mg,max_cycles=1))
    monkeypatch.setattr(c.transfer,'forward',lambda f:torch.full((1,16,7,7),float('nan')))
    result=PreparedAdaptiveMG(a,7,c,cfg).solve(b)
    assert result.stats['setup_failures']==1
    assert result.stats['transfer_nn_calls']==1 and result.stats['smoother_nn_calls']==0
    assert result.stats['nn_setup_seconds']>0 and result.stats['operator_generation_seconds']>0
    assert result.solve_seconds>result.stats['nn_setup_seconds']
    assert result.stats['branch_setup_seconds']>=result.stats['nn_setup_seconds']
    if eager:
        assert result.stats['attempted_neural_cycles']==0
        assert result.stats['classical_bypass_cycles']==0 and result.stats['classical_lock_cycles']==1
        assert result.abstention['locked_after_attempt']==0
    else:
        assert result.stats['attempted_neural_cycles']==result.stats['rejected_neural_cycles']==1
        assert result.stats['classical_recovery_cycles']==0 and result.executed_cycles==1
        assert result.stop_reason=='cycle_budget_after_rejected_trial'
        np.testing.assert_array_equal(result.x,np.zeros_like(b))
        assert result.abstention['locked_after_attempt']==1
