from dataclasses import replace
import numpy as np
import pytest
import torch
from adaptive_mg import *
from adaptive_mg.models import SwitchNet,switch_numpy
from adaptive_mg.policy import N_FEATURES,initial_features
from adaptive_mg.provenance import module_signature,hardware_environment
from adaptive_mg.checkpoint import CONTRACT
from adaptive_mg.hierarchy import hierarchy_identity
from adaptive_mg.strategy import global_context_features


def with_switch(model,k=2):
    model.switch=SwitchNet(N_FEATURES,(0,1,2,3),hidden=8).double()
    with torch.no_grad():
        model.switch.body[-1].weight.zero_();model.switch.body[-1].bias.fill_(-10);model.switch.body[-1].bias[k]=10
    return model


def conf(**kw):
    d=dict(mode='fixed',fixed_k=2,max_cycles=5,tolerance=1e-20,absolute_tolerance=0.,near_tolerance_factor=0.,stagnation_patience=50,stencil_backend='csr')
    d.update(kw);return MGConfig(**d)


def test_predictor_once_per_solve(problem,model):
    a,b,n=problem;model=with_switch(model)
    r=PreparedTemporalMGSolver(a,n,model,conf(mode='predicted')).solve(b)
    assert r.stats['predictor_calls']==1 and r.k_nn==2
    assert r.cycle_path[:3]==['neural','neural','classical']


def test_unvalidated_deployment_does_not_claim_learning_win(problem,model):
    a,b,n=problem;model=with_switch(model)
    p=PreparedTemporalMGSolver(a,n,model,conf(mode='temporal'))
    r=p.solve(b)
    assert r.k_nn==0 and r.stats['predictor_calls']==0
    assert r.decision['reason']=='policy_not_validated'
    assert p.operator_context is None


def test_audited_policy_and_hardware_invalidation(problem,model):
    a,b,n=problem;model=with_switch(model);cfg=conf(mode='temporal')
    model.metadata['temporal_certificate']=dict(validated=True,contract=CONTRACT,
        smoother_signature=module_signature(model.smoother),switch_signature=module_signature(model.switch),
        environment=hardware_environment(),scope=cfg.numerical_scope(),min_n=15,max_n=31,cache_regimes=[0,1])
    p=PreparedTemporalMGSolver(a,n,model,cfg);r=p.solve(b)
    assert r.k_nn==2 and r.stats['predictor_calls']==1
    model.metadata['temporal_certificate']['environment']={'changed':True}
    q=p.solve(b)
    assert q.k_nn==0 and q.decision['reason']=='hardware_or_threads_changed'


def test_model_mutation_invalidates_nn_only(problem,model):
    a,b,n=problem;p=PreparedTemporalMGSolver(a,n,model,conf())
    p.solve(b);identity=hierarchy_identity(p.root);bank=p.root.neural_stencil
    model.smoother.gain_head.bias.data.add_(.001)
    r=p.solve(b)
    assert p.root.neural_stencil is not bank
    assert hierarchy_identity(p.root)==identity and p.cache_rebuilds==1
    assert r.stats['neural_setup_calls']>0


def test_operator_mutation_rebuilds(problem,model):
    a,b,n=problem;p=PreparedTemporalMGSolver(a,n,model,conf())
    p.solve(b);old=id(p.root)
    a.data*=1.01
    r=p.solve(b)
    assert id(p.root)!=old and p.cache_rebuilds==2
    other=PreparedTemporalMGSolver(a,n,model,conf()).solve(b)
    np.testing.assert_array_equal(r.x,other.x)


def test_replacing_config_rebuilds(problem,model):
    a,b,n=problem;p=PreparedTemporalMGSolver(a,n,model,conf())
    p.solve(b);p.config=replace(p.config,strategy_name='jacobi_bilinear_full')
    r=p.solve(b)
    assert p.cache_rebuilds==2 and r.strategy_name=='jacobi_bilinear_full'


def test_rhs_reuse_resets_temporal_state(problem,model):
    a,b,n=problem
    bs=np.stack([b,2*b,-b,np.ones_like(b)])
    p=PreparedTemporalMGSolver(a,n,model,conf());rs=p.solve_many(bs)
    assert [r.stats['freshness_checks'] for r in rs]==[1,0,0,0]
    assert [r.stats['neural_setup_calls'] for r in rs]==[2,0,0,0]
    assert rs[0].setup_seconds>0 and all(r.setup_seconds==0 for r in rs[1:])
    for rhs,r in zip(bs,rs):
        z=PreparedTemporalMGSolver(a,n,model,conf()).solve(rhs)
        np.testing.assert_array_equal(r.x,z.x)
        assert r.cycle_path==z.cycle_path


def test_features_finite_and_rhs_dependent(problem):
    a,b,n=problem;c=global_context_features(a,(n,n))
    f=initial_features(c,(n,n),b,1e-8,'line_alt_bilinear_full')
    g=initial_features(c,(n,n),np.ones_like(b),1e-8,'line_alt_bilinear_full')
    assert len(f)==N_FEATURES and np.isfinite(f).all() and not np.array_equal(f,g)


def test_numpy_torch_policy_parity(model):
    model=with_switch(model);torch.manual_seed(88)
    with torch.no_grad():model.switch.body[-1].weight.normal_()
    x=torch.randn(N_FEATURES,dtype=torch.float64)
    y=model.switch(x).detach().numpy()
    z=switch_numpy(model.switch.numpy_weights(),x.numpy())
    np.testing.assert_allclose(z,y,rtol=1e-12,atol=1e-12)


def test_checkpoint_roundtrip(tmp_path,model,problem):
    a,b,n=problem;model=with_switch(model);path=tmp_path/'model.pt';model.save(path)
    m=TemporalComponents.load(path)
    assert module_signature(m.smoother)==module_signature(model.smoother)
    assert module_signature(m.switch)==module_signature(model.switch)
    r=PreparedTemporalMGSolver(a,n,m,conf(mode='predicted')).solve(b)
    s=PreparedTemporalMGSolver(a,n,model,conf(mode='predicted')).solve(b)
    np.testing.assert_array_equal(r.x,s.x)


def test_legacy_requires_trust():
    with pytest.raises(ValueError,match='trusted'):
        TemporalComponents.migrate_v65('does_not_exist.pt')


def test_nan_prediction_abstains(problem,model):
    a,b,n=problem;model=with_switch(model)
    model.switch.body[-1].bias.data.fill_(float('nan'))
    r=PreparedTemporalMGSolver(a,n,model,conf(mode='predicted')).solve(b)
    assert r.k_nn==0 and r.decision['reason']=='nonfinite_prediction'


def test_operator_rebuild_time_is_not_double_charged(problem,model):
    from time import perf_counter
    a,b,n=problem;p=PreparedTemporalMGSolver(a,n,model,conf())
    p.solve(b);a.data*=1.01
    t=perf_counter();r=p.solve(b);wall=perf_counter()-t
    assert r.setup_seconds>0
    # The elapsed result must fit inside the externally timed post-mutation
    # solve, because its rebuild is counted once, not in both setup and solve.
    assert r.elapsed_seconds <= wall+1e-6
