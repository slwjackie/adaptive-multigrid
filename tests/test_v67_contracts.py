import numpy as np
import torch
import scipy.sparse as sp
import pytest
from dataclasses import replace
from copy import deepcopy
from adaptive_mg import DiffusionCase,assemble_stiffness,MGConfig
from adaptive_mg.strategy import STRATEGIES
from adaptive_mg.hierarchy import classical_cycle
from adaptive_mg.data import fixed_benchmark_examples
from adaptive_mg.provenance import stable_norm
from adaptive_mg.v67 import AdaptiveConfig,Components,PreparedAdaptiveMG
from adaptive_mg.v67.banks import Stats,prepare_learned_bank,hybrid_cycle,replacement_step
from adaptive_mg.v67.spatial import SpatialState
from adaptive_mg.v67.native import block_statistics,apply_rows
from adaptive_mg.v67.controller import break_even,OnlineHistory
from adaptive_mg.v67.autograd_sparse import SparseTensor
from adaptive_mg.v67.unroll import rollout,make_graph,cycle
from adaptive_mg.native_stencil import native_available


def make(strategy='jacobi_bilinear_full',n=7,**kwargs):
    a=assemble_stiffness(DiffusionCase(n=n,epsilon=.7,contrast=1.))
    b=np.random.default_rng(99).normal(size=n*n)
    mg=MGConfig(mode='classical',strategy_name=strategy,pre_steps=2,post_steps=2,max_cycles=5,
                tolerance=1e-14,absolute_tolerance=0.,nn_levels=1)
    cfg=AdaptiveConfig(mg=mg,mode='hybrid',spatial=False,gate_mode='open',**kwargs)
    return a,b,Components.create(hidden=4),cfg


def test_new_checkpoint_roundtrip(tmp_path):
    c=Components.create(hidden=4);p=tmp_path/'model.pt';c.save(p)
    d=Components.load(p);assert c.signature()==d.signature()
    assert not p.with_name(p.name+'.tmp').exists()

@pytest.mark.parametrize('strategy',[s.name for s in STRATEGIES])
def test_closed_exact_classical(strategy):
    a,b,c,cfg=make(strategy)
    old=PreparedAdaptiveMG(a,7,c,replace(cfg,mode='classical')).solve(b)
    p=PreparedAdaptiveMG(a,7,c,replace(cfg,gate_mode='closed',spatial=True))
    r=p.solve(b)
    assert np.array_equal(old.x,r.x)
    assert old.residual_history==r.residual_history
    assert p.learned is None and r.stats['smoother_nn_calls']==0 and r.stats['detector_calls']==0


def test_uncertified_default_no_heavy_work():
    a,b,c,cfg=make();p=PreparedAdaptiveMG(a,7,c,replace(cfg,mode='production',spatial=True,gate_mode='learned'))
    r=p.solve(b);assert p.learned is None
    assert r.stats['controller_calls']==r.stats['smoother_nn_calls']==r.stats['transfer_nn_calls']==r.stats['detector_calls']==0


def test_bank_cache_and_no_inner_torch(monkeypatch):
    a,b,c,cfg=make();p=PreparedAdaptiveMG(a,7,c,cfg);p.ensure_learned(Stats())
    root=p.learned
    def fail(*a,**k):raise AssertionError('heavy NN in the cycle')
    monkeypatch.setattr(c.smoother,'direction_and_gain',fail)
    monkeypatch.setattr(c.transfer,'forward',fail)
    r=p.solve(b);s=p.solve(2*b)
    assert p.learned is root and p.learned_builds_total==1
    assert r.stats['smoother_nn_calls']==s.stats['smoother_nn_calls']==0


def test_dtype_change_does_not_rebuild_each_rhs():
    a,b,c,cfg=make(inference_dtype='float64');p=PreparedAdaptiveMG(a,7,c,cfg)
    p.solve(b);p.solve(b);assert p.learned_builds_total==1


def test_same_A_multi_rhs_one_freshness():
    a,b,c,cfg=make();p=PreparedAdaptiveMG(a,7,c,cfg)
    results=p.solve_many(np.stack((b,2*b,-b)))
    assert sum(r.stats['freshness_checks'] for r in results)==1
    assert sum(r.stats['learned_hierarchy_builds'] for r in results)==1
    for rhs,r in zip((b,2*b,-b),results):
        q=PreparedAdaptiveMG(a,7,c,cfg).solve(rhs)
        assert np.array_equal(q.x,r.x)


def test_A_and_model_freshness():
    a,b,c,cfg=make();p=PreparedAdaptiveMG(a,7,c,cfg);p.solve(b)
    root=p.classical;old=p.learned
    with torch.no_grad():c.smoother.gain_head.bias.add_(.01)
    p.solve(b);assert p.classical is root and p.learned is not old
    a.data*=1.01;p.solve(b);assert p.classical is not root


def test_galerkin_FP64_injection():
    a,b,c,cfg=make();p=PreparedAdaptiveMG(a,7,c,cfg);l=p.ensure_learned(Stats())
    from adaptive_mg.transfer import coarse_fine_indices
    while l.coarse is not None:
        assert l.a.dtype==l.p.dtype==l.r.dtype==np.float64
        assert np.max(np.abs((l.r-l.p.T).data),initial=0)==0
        ac=l.p.T@l.a@l.p
        np.testing.assert_allclose(l.coarse.a.toarray(),ac.toarray(),rtol=1e-12,atol=1e-12)
        ix=coarse_fine_indices(l.pattern)
        np.testing.assert_allclose(l.p[ix].toarray(),np.eye(l.p.shape[1]),atol=1e-12)
        l=l.coarse


def test_one_coarse_path_not_auxiliary():
    a,b,c,cfg=make();p=PreparedAdaptiveMG(a,7,c,cfg);st=Stats();root=p.ensure_learned(st)
    q=Stats();spatial=SpatialState(c,cfg)
    hybrid_cycle(root,np.zeros_like(b),b,cfg,q,spatial,1)
    assert q.coarse_solves==1 and q.coarse_path_count==1
    assert q.classical_smooth_calls==3 # two pre + two post; one pre replaced
    assert q.replacement_calls==1


def test_growth_rollback_and_lock():
    a,b,c,cfg=make();cfg=replace(cfg,mg=replace(cfg.mg,smoother_gain_multiplier=1e80,max_cycles=4))
    p=PreparedAdaptiveMG(a,7,c,cfg);r=p.solve(b)
    assert r.stats['rollback_count']==1 and r.stats['accepted_neural_cycles']==0
    assert r.stats['attempted_neural_cycles']==1
    assert r.executed_cycles<=4
    q=PreparedAdaptiveMG(a,7,c,replace(cfg,mode='classical',mg=replace(cfg.mg,max_cycles=r.cycles))).solve(b)
    assert np.array_equal(q.x,r.x)


def test_invalid_P_rollback(monkeypatch):
    a,b,c,cfg=make()
    def bad(f):return torch.full((1,16,7,7),float('nan'))
    monkeypatch.setattr(c.transfer,'forward',bad)
    r=PreparedAdaptiveMG(a,7,c,cfg).solve(b)
    assert r.stats['rollback_count']==1 and all(v!='hybrid' for v in r.cycle_path)


def test_current_true_residual():
    a,b,c,cfg=make();r=PreparedAdaptiveMG(a,7,c,cfg).solve(b)
    assert r.final_true_residual==stable_norm(b-a@r.x)
    assert np.isfinite(r.x).all()

@pytest.mark.parametrize('strategy',['jacobi_bilinear_full','line_x_bilinear_full','line_alt_bilinear_full'])
def test_unroll_runtime_parity(strategy):
    a,b,c,cfg=make(strategy)
    p=PreparedAdaptiveMG(a,7,c,cfg);st=Stats();root=p.ensure_learned(st)
    xn=hybrid_cycle(root,np.zeros_like(b),b,cfg,st,SpatialState(c,cfg),1)
    tg=make_graph(a,(7,7),c,cfg,learned=True)
    xt=cycle(tg,torch.zeros(49,dtype=torch.double),torch.tensor(b),c,cfg).detach().numpy()
    np.testing.assert_allclose(xt,xn,atol=2e-10,rtol=2e-8)


def test_both_heads_have_gradients():
    a,b,c,cfg=make();e=fixed_benchmark_examples([7])[0]
    loss,_,_=rollout(e,c,cfg,1,1);loss.backward()
    for m in (c.smoother,c.transfer):
        assert sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)>0


def test_hierarchical_closed_forward_with_ste_gradients():
    a,b,c,cfg=make();e=fixed_benchmark_examples([7])[0]
    cfg=replace(cfg,spatial=True,gate_mode='learned')
    loss,_,x=rollout(e,c,cfg,1,1,hierarchical=True)
    pc=PreparedAdaptiveMG(e.a,7,c,replace(cfg,mode='classical'))
    xc=np.zeros_like(e.b)
    for i in range(2):xc=classical_cycle(pc.classical,xc,e.b,cfg.mg,Stats())
    np.testing.assert_allclose(x.detach().numpy(),xc,atol=1e-12)
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in c.detector.parameters() if p.grad is not None)>0
    assert sum(p.grad.abs().sum().item() for p in c.smoother.parameters() if p.grad is not None)>0

@pytest.mark.parametrize('operation',['apply','solve','product'])
def test_sparse_autograd_gradcheck(operation):
    a=SparseTensor.from_scipy(sp.csr_matrix([[3.,1.],[1.,2.]]),True)
    x=torch.tensor([1.,2.],dtype=torch.double,requires_grad=True)
    if operation=='product':
        b=SparseTensor.from_scipy(sp.csr_matrix([[1.,2.],[3.,4.]]),True)
        f=lambda av,bv:SparseTensor(a.pattern,av).product(SparseTensor(b.pattern,bv)).values
        assert torch.autograd.gradcheck(f,(a.values,b.values))
    else:
        f=lambda av,xx:getattr(SparseTensor(a.pattern,av),operation)(xx)
        assert torch.autograd.gradcheck(f,(a.values,x))


def test_break_even_charges_setup():
    a=break_even(10.,.5,.01,2.,1.,0.,4,.05)
    b=break_even(10.,.5,.01,2.,1.,1.,4,.05)
    assert a['use_hybrid'] and not b['use_hybrid']


def test_temporal_reentry_and_lock():
    a,b,c,cfg=make();cfg=replace(cfg,mode='research',min_dwell=2)
    h=OnlineHistory();h.update('CLASSICAL',.5,.01);h.update('CLASSICAL',.5,.01)
    net=lambda f:np.array([np.log(2.),0.,0.,20.])
    state,_=h.select(cfg,net,True,np.zeros(18),1.,1e-8,1.,3,True)
    assert state=='HYBRID'
    h.mode='CLASSICAL';h.dwell=2;h.entries=1
    state,_=h.select(cfg,net,True,np.zeros(18),1e-2,1e-8,1.,8,True)
    assert state=='HYBRID'
    h.mode='CLASSICAL_LOCK'
    state,_=h.select(cfg,None,False,np.zeros(18),1.,1e-8,1.,9,True)
    assert state=='CLASSICAL_LOCK'


def test_numpy_controller_parity():
    c=Components.create(hidden=4);x=np.random.default_rng(3).normal(size=(20,28)).astype(np.float32)
    np.testing.assert_allclose(c.controller.compile()(x),c.controller(torch.tensor(x)).detach().numpy(),atol=1e-6)


def test_spatial_refresh_cache():
    a,b,c,cfg=make();cfg=replace(cfg,spatial=True,gate_mode='learned',gate_refresh=4)
    with torch.no_grad():c.detector.fc2.bias.fill_(20.)
    p=PreparedAdaptiveMG(a,7,c,cfg);root=p.ensure_learned(Stats());state=SpatialState(c,cfg);st=Stats()
    for i in range(1,9):state.gate(root,b,i,st)
    assert st.detector_calls==2


def test_fused_restriction_and_rows():
    if not native_available():pytest.skip('native backend not built')
    a,b,c,cfg=make();cfg=replace(cfg,mg=replace(cfg.mg,stencil_backend='native'))
    p=PreparedAdaptiveMG(a,7,c,cfg);root=p.ensure_learned(Stats())
    rc,stats,blocks=block_statistics(b,root.shape,3,root.p)
    np.testing.assert_allclose(rc,root.p.T@b,atol=1e-13)
    assert abs(stats[:,0].sum()-b@b)<1e-12
    bank=root.neural_stencil;rows=np.arange(0,len(b),3)
    np.testing.assert_allclose(apply_rows(bank.native,b,rows),(bank.csr@b)[rows],atol=1e-12)


def test_partial_line_not_masking_full_classical():
    a,b,c,cfg=make('line_x_bilinear_full');p=PreparedAdaptiveMG(a,7,c,cfg);root=p.ensure_learned(Stats())
    mask=np.zeros((7,7),bool);mask[:,2]=True;mask=mask.ravel();st=Stats()
    replacement_step(root,b,mask,cfg,st)
    assert st.replaced_line_solves==1 and st.remaining_line_solves==6
    assert st.classical_smooth_calls==0


def test_production_config_rejects_additive_and_mps_fp64():
    with pytest.raises(ValueError):AdaptiveConfig(application='additive')
    with pytest.raises(ValueError):AdaptiveConfig(inference_device='mps',inference_dtype='float64')


def test_classical_bypass_never_compiles_micro_models(monkeypatch):
    a,b,c,cfg=make()
    def bad():raise AssertionError("unnecessary NN preparation on classical bypass")
    monkeypatch.setattr(c.detector,"compile",bad)
    monkeypatch.setattr(c.controller,"compile",bad)
    r=PreparedAdaptiveMG(a,7,c,replace(cfg,mode="production")).solve(b)
    assert r.stats["detector_calls"]==r.stats["controller_calls"]==0

def test_spatial_compiled_model_reused_across_rhs(monkeypatch):
    a,b,c,cfg=make()
    cfg=replace(cfg,spatial=True,gate_mode="learned")
    with torch.no_grad():c.detector.fc2.bias.fill_(20.)
    original=c.detector.compile;calls=[]
    def count():calls.append(1);return original()
    monkeypatch.setattr(c.detector,"compile",count)
    p=PreparedAdaptiveMG(a,7,c,cfg)
    p.solve(b);p.solve(2*b)
    assert len(calls)==1

def test_disabled_learned_controller_bypasses_predictor():
    a,b,c,cfg=make()
    cfg=replace(cfg,mode="research",use_learned_controller=False)
    c.metadata["controller_trained"]=True
    p=PreparedAdaptiveMG(a,7,c,cfg)
    def bad(*args):raise AssertionError("disabled controller was invoked")
    p.compiled_controller=bad
    r=p.solve(b)
    assert p.learned is None and r.stats["accepted_neural_cycles"]==0


def test_eager_learned_setup_failure_falls_back(monkeypatch):
    a,b,c,cfg=make(lazy=False)
    def bad(f):return torch.full((1,16,7,7),float('nan'))
    monkeypatch.setattr(c.transfer,'forward',bad)
    result=PreparedAdaptiveMG(a,7,c,cfg).solve(b)
    reference=PreparedAdaptiveMG(a,7,c,replace(cfg,mode='classical')).solve(b)
    assert result.stats['setup_failures']==1
    assert result.stats['accepted_neural_cycles']==0
    assert np.array_equal(result.x,reference.x)
