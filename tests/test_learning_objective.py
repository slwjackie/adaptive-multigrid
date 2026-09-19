from dataclasses import replace
import numpy as np
import pytest
import torch
from adaptive_mg import *
from adaptive_mg.measurement import attach_objective
from adaptive_mg.data import make_examples,split_manifest
from adaptive_mg.learning import prepare_torch_hierarchy,torch_cycle,prefix_rollout
from adaptive_mg.hierarchy import WorkStats,classical_cycle,neural_cycle
from adaptive_mg.provenance import operator_digest,write_json
from adaptive_mg.train_switch import fit_switch
from adaptive_mg.policy import N_FEATURES


@pytest.mark.parametrize('strategy',['jacobi_bilinear_full','line_alt_bilinear_full','line_diag45_bilinear_full','chebyshev_bilinear_full','line_x_operator_semi_y'])
@pytest.mark.parametrize('neural',[False,True])
def test_torch_runtime_parity(model,problem,strategy,neural):
    a,b,n=problem;cfg=MGConfig(mode='fixed',fixed_k=2,strategy_name=strategy,stencil_backend='csr')
    p=PreparedTemporalMGSolver(a,n,model,cfg)
    t=prepare_torch_hierarchy(a,(n,n),model.smoother,cfg)
    tx=torch_cycle(t,torch.zeros(len(b),dtype=torch.float64),torch.tensor(b),cfg,neural=neural)
    nx=neural_cycle(p.root,np.zeros_like(b),b,model.smoother,cfg,WorkStats()) if neural else classical_cycle(p.root,np.zeros_like(b),b,cfg,WorkStats())
    np.testing.assert_allclose(tx.detach().numpy(),nx,rtol=3e-10,atol=3e-10)


def test_prefix_tail_gradients(model,problem):
    a,b,n=problem;cfg=MGConfig(mode='fixed',fixed_k=2)
    t=prepare_torch_hierarchy(a,(n,n),model.smoother,cfg)
    loss,x,history=prefix_rollout(t,torch.tensor(b),cfg,k_nn=2,tail_cycles=2)
    loss.backward()
    g=sum(float(p.grad.abs().sum()) for p in model.smoother.parameters() if p.grad is not None)
    assert len(history)==4 and np.isfinite(g) and g>0


def act(k,t,ok):
    return dict(k=k,median_seconds=t,converged=ok,nonfinite_attempts=0,rejected_attempts=0,growth_log=0.)


def test_fast_failure_never_is_a_win():
    row=dict(actions=[act(0,1.,True),act(1,.001,False),act(2,.7,True)])
    attach_objective(row)
    assert row['oracle_k']==2
    assert row['actions'][1]['objective']>100
    assert row['actions'][1]['time_to_tolerance_seconds'] is None


def test_all_failures_abstain_not_min_runtime():
    row=dict(actions=[act(0,1.,False),act(1,.001,False)])
    attach_objective(row)
    assert row['oracle_k']==0 and row['training_costs']==[0.,1.]


def test_oracle_includes_zero_and_prefers_non_noisy_tie():
    row=dict(actions=[act(0,1.,True),act(1,.999,True)])
    attach_objective(row)
    assert row['oracle_k']==0


def test_normalized_operator_disjoint():
    a=make_examples([15],per_size=6,seed=7);b=make_examples([15],per_size=6,seed=8)
    assert split_manifest(train=a,audit=b)['normalized_operator_disjoint']
    with pytest.raises(ValueError):split_manifest(train=a,audit=a)
    assert operator_digest(a[0].a,scale_invariant=True)==operator_digest(4*a[0].a,scale_invariant=True)


def test_json_has_no_nan(tmp_path):
    p=tmp_path/'x.json';write_json(p,{'bad':float('nan'),'infinity':float('inf')})
    assert 'NaN' not in p.read_text() and 'Infinity' not in p.read_text()


def test_measured_cost_training_learns_nonzero_without_gates(model):
    # Synthetic deterministic cost table verifies the training algorithm, not
    # a claim of empirical speedup in the real PDE benchmark.
    rows=[]
    for sign in [-1.,1.]*8:
        f=[0.]*N_FEATURES;f[0]=sign
        costs=[1.,.5] if sign>0 else [.5,1.]
        rows.append(dict(features=f,training_costs=costs,actions=[{'k':0},{'k':2}],oracle_k=2 if sign>0 else 0,oracle_status='test'))
    rec=fit_switch(model,rows,rows,epochs=100,hidden=8,seed=5)
    features=torch.tensor([r['features'] for r in rows],dtype=torch.float64)
    choices=model.switch(features).argmax(-1).tolist()
    assert choices==[0,1]*8
    assert rec['history'][-1]['gradient_norm']>=0
