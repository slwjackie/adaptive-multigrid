"""Contracts for EM plans, sparse batched differentiation, affine P, policy."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg import DiffusionCase,assemble_stiffness,MGConfig
from adaptive_mg.energymin import energy_weights,row_sum_basis
from adaptive_mg.transfer import build_transfer_pattern,scipy_prolongation_from_weights,coarse_fine_indices
from adaptive_mg.strategy import STRATEGIES,get_strategy
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.banks import Stats,hybrid_cycle
from adaptive_mg.v67.p_headroom import AffineSupport,energy_minimize
from adaptive_mg.v67.strong import StrongRules,PreparedStrongMG,classical_bank
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.research_training import create_research_components,transfer_feasibility
from adaptive_mg.v67.research_transfer import make_graph_transfer,project_transfer_weights
from adaptive_mg.v67.autograd_sparse import SparseTensor
from adaptive_mg.v67.unroll import make_graph,cycle,_line_batches
from adaptive_mg.v67.spatial import SpatialState

CAPS=dict(max_p_ratio=1.,max_ac_ratio=1.15,complexity_reference='parent',max_operator_complexity_ratio=1.15)
PLAN='line_alt_energymin_full__em10__v11'


def config(plan=PLAN):
    return AdaptiveConfig(mg=MGConfig(mode='classical',strategy_name=plan,max_cycles=80,pre_steps=2,post_steps=2,
                          nn_levels=1,stencil_backend='csr'),mode='research',branch='H_P',use_smoother=False,
                          spatial=False,gate_mode='open',transfer_levels='all')


def model():
    m=create_research_components(smoother='student_cnn',transfer='small_gnn',smoother_hidden=4,transfer_hidden=4)
    m.transfer=make_graph_transfer(width=4,support='support_preserving',parameterization='affine',reference='frozen_parent',
                                  support_only=True,complexity_caps=CAPS)
    return m


def rules(plan=PLAN):
    r=StrongRules();return r.replace_strategies({k:plan for k in r.rule_ids})


def test_opt_in_plan_preserves_legacy_classifiers_and_resolves_schedule():
    assert len(STRATEGIES)==16 and len(classical_bank())==16
    assert len(classical_bank('em_schedule'))>16
    for plan in classical_bank('em_schedule'):
        c=MGConfig(strategy_name=plan.name)
        if plan.pre_steps is not None:assert (c.pre_steps,c.post_steps)==(plan.pre_steps,plan.post_steps)
    assert config().mg.pre_steps==1
    assert AdaptiveConfig.from_dict(config().to_dict())==config()
    with pytest.raises(ValueError):replace(config(),replace_pre=2)


@pytest.mark.parametrize('n',[7,15,31])
def test_energy_fast_matches_general_constraints_and_preserves_support(n):
    a=assemble_stiffness(DiffusionCase(n,epsilon=.03,angle_deg=35))
    pattern=build_transfer_pattern(n);p=scipy_prolongation_from_weights(pattern,pattern.bilinear_weights)
    fixed=coarse_fine_indices(pattern)
    z=row_sum_basis(p,fixed)
    np.testing.assert_allclose((z.T@z).toarray(),np.eye(z.shape[1]),atol=2e-14)
    general=AffineSupport.build(p,fixed,coarse_modes=np.ones((p.shape[1],1)))
    expected,report=energy_minimize(a,general,maxiter=300)
    w,r=energy_weights(a,pattern,maxiter=300)
    got=scipy_prolongation_from_weights(pattern,w)
    np.testing.assert_allclose(got.toarray(),expected.toarray(),atol=1e-7,rtol=1e-6)
    assert r['energy_after']<=r['energy_before']+1e-10
    AffineSupport.build(p,fixed).validate(got)
    np.testing.assert_allclose((got@np.ones(p.shape[1])),p@np.ones(p.shape[1]),atol=1e-14)


@pytest.mark.parametrize('plan',[PLAN,'jacobi_bilinear_full','line_diag45_bilinear_full','line_x_operator_semi_y'])
@pytest.mark.parametrize('perturb',[False,True])
def test_affine_all_levels_train_runtime_parity(plan,perturb):
    n=15;a=assemble_stiffness(DiffusionCase(n,epsilon=.05,angle_deg=27));c=config(plan);m=model()
    if perturb:
        with torch.no_grad():m.transfer.delta_head.weight.fill_(.02)
    prepared=PreparedStrongMG(a,n,m,c,rules(plan));root=prepared.ensure_branch('H_P',Stats())
    graph=make_graph(a,(n,n),m,c);reference=make_graph(a,(n,n),m,c,learned=False)
    node,gn,rn=root,graph,prepared.classical
    while node.coarse is not None:
        np.testing.assert_allclose(node.p.toarray(),gn.p.numpy().toarray(),atol=2e-8,rtol=2e-6)
        np.testing.assert_array_equal(node.base_weights,rn.base_weights)
        if not perturb:np.testing.assert_array_equal(node.p.toarray(),rn.p.toarray())
        node,gn,rn=node.coarse,gn.coarse,rn.coarse
    rng=np.random.default_rng(3);r=rng.normal(size=n*n)
    actual=hybrid_cycle(root,np.zeros_like(r),r,c,Stats(),SpatialState(m,c),0,True)
    expected=cycle(graph,torch.zeros(n*n,dtype=torch.double),torch.tensor(r),m,c)
    np.testing.assert_allclose(actual,expected.detach(),rtol=4e-6,atol=2e-8)
    batch=torch.tensor(np.stack((r,r*.7,r[::-1]),axis=1).copy())
    together=cycle(graph,torch.zeros_like(batch),batch,m,c)
    columns=torch.stack([cycle(graph,torch.zeros_like(batch[:,i]),batch[:,i],m,c) for i in range(3)],1)
    torch.testing.assert_close(together,columns,rtol=2e-9,atol=1e-10)
    together.square().sum().backward()
    assert m.transfer.delta_head.weight.grad is not None and torch.isfinite(m.transfer.delta_head.weight.grad).all()
    assert transfer_feasibility(graph,m,c,reference)[0]['feasible']


def test_affine_constraints_and_forbidden_gradient_and_large_head():
    m=model();pat=build_transfer_pattern(7);base=pat.bilinear_weights
    raw=torch.randn(1,16,7,7,dtype=torch.double,requires_grad=True)
    out=project_transfer_weights(m.transfer,pat,raw*100,base)
    torch.testing.assert_close(out.sum(1),torch.tensor(base.sum(1)),rtol=0,atol=1e-13)
    assert out.abs().sum(1).max()<=8+1e-12
    out.square().sum().backward()
    gradient=raw.grad[0].permute(1,2,0).reshape(base.shape)
    assert torch.count_nonzero(gradient[torch.tensor(base==0)])==0
    assert torch.count_nonzero(gradient[torch.tensor(coarse_fine_indices(pat))])==0


def test_sparse_lu_reuse_mutation_and_batched_gradient():
    p=sp.csr_matrix(np.array([[4.,1.],[1.,3.]]));a=SparseTensor.from_scipy(p,requires_grad=True)
    b=torch.randn(2,3,dtype=torch.double,requires_grad=True)
    x=a.solve(b);y=a.solve(2*b);assert a.cache['factorizations']==1
    (x.square().sum()+y.square().sum()).backward()
    v=a.values.detach().clone().requires_grad_();bb=b.detach().clone().requires_grad_()
    dense=torch.zeros(2,2,dtype=torch.double).index_put((torch.tensor(a.rows),torch.tensor(a.cols)),v)
    xx=torch.linalg.solve(dense,bb);yy=torch.linalg.solve(dense,2*bb);(xx.square().sum()+yy.square().sum()).backward()
    torch.testing.assert_close(a.values.grad,v.grad);torch.testing.assert_close(b.grad,bb.grad)
    with torch.no_grad():a.values.add_(.01)
    a.solve(b.detach());assert a.cache['factorizations']==2
    other=SparseTensor.from_scipy(p);other.solve(b.detach());assert other.cache['factorizations']==1


def test_parent_gradient_direction_matches_finite_difference():
    from adaptive_mg.v67.hp_training import probe_loss
    a=assemble_stiffness(DiffusionCase(7,epsilon=.05,angle_deg=22));c=replace(config(),inference_dtype='float64');m=model()
    probes=torch.randn(49,3,dtype=torch.double)
    def value():return probe_loss(make_graph(a,(7,7),m,c),m,c,probes,cycles=3)[0]
    loss=value();loss.backward();parameter=m.transfer.delta_head.bias
    # Row-common bias vanishes; check a nontrivial direction in the weight vector instead.
    parameter=m.transfer.delta_head.weight;direction=torch.randn_like(parameter);analytic=float((parameter.grad*direction).sum())
    original=parameter.detach().clone();h=1e-5
    with torch.no_grad():parameter.copy_(original+h*direction)
    plus=float(value().detach())
    with torch.no_grad():parameter.copy_(original-h*direction)
    minus=float(value().detach())
    np.testing.assert_allclose((plus-minus)/(2*h),analytic,rtol=.025,atol=2e-5)


def test_affine_checkpoint_schema_roundtrip(tmp_path):
    m=model();path=tmp_path/'m.pt';m.save(path);loaded=Components.load(path)
    assert loaded.transfer.parameterization=='affine' and loaded.transfer.reference=='frozen_parent'
    assert loaded.transfer.support_only and loaded.signature()==m.signature()


def test_warm_calibration_records_actual_plan_and_excludes_setup():
    from adaptive_mg.v67.research_data import _generate_split
    from adaptive_mg.v67.strong_calibration import measure_classical_portfolio
    e=_generate_split('selector_train',dict(sizes=[7],families=['near_isotropic'],per_family=1,seed=72),set(),rules())[0][0]
    # Real bank, tiny N; setup excluded, different prime RHS.
    rows=measure_classical_portfolio([e],config(),rules(),repeats=1,rhs_count=2,bank='em_schedule',regime='warm_multiple')
    for name,runs in rows[0]['runs'].items():
        run=runs[0]
        if run['success']:
            assert run['preparation_seconds']>0
            c=AdaptiveConfig.from_dict(run['measurement_config']);plan=get_strategy(name)
            if plan.pre_steps is not None:assert c.mg.pre_steps==plan.pre_steps
    assert rows[0]['regime']=='warm_multiple'


def hp_policy(m,cfg,rr,allow=True):
    from adaptive_mg.provenance import hardware_environment
    from adaptive_mg.v67.cost_policy import ContinuousCostPolicy,config_scope,fit_cost_model
    def rows(prefix):
        return [dict(operator=f'{prefix}-{i}',C_success=True,H_success=True,C_seconds=.01,H_seconds=.005,
            neural_used=True,context=dict(N=225,nnz=1500,depth=3,complexity=1.4,rhs_count=k,cached=cached,
                strategy=PLAN,operator_features={})) for i in range(4) for k in (1,4) for cached in (False,True)]
    fitted=fit_cost_model(rows('fit'),rows('tune'))
    fitted['beta']=[float(np.log(2.) if allow else -1.)]+[0.]*(len(fitted['beta'])-1)
    fitted['N_range']=[1,5000];fitted['rhs_range']=[1,64]
    return ContinuousCostPolicy(fitted,m,rr.digest(),config_scope(cfg,plan_selection=True),hardware_environment(),
                                m.signature(),{},branch='H_P')


@pytest.mark.parametrize('allow',[False,True])
def test_hp_policy_branch_parent_scope_batch_decision_and_zero_warm_setup(allow):
    from adaptive_mg.v67.cost_policy import PreparedCostMG,ContinuousCostPolicy
    n=15;a=assemble_stiffness(DiffusionCase(n,epsilon=.04,angle_deg=31));m=model().frozen_inference_copy();c=config();rr=rules()
    policy=hp_policy(m,c,rr,allow)
    # Deliberately original config V22 -> selected runtime V11; no false stale policy.
    c=replace(c,mg=MGConfig(mode='classical',pre_steps=2,post_steps=2,nn_levels=1,max_cycles=80,stencil_backend='csr'))
    p=PreparedCostMG(a,n,policy=policy,config=c,rules=rr)
    assert p._policy_valid and p.config.mg.pre_steps==1
    bs=np.random.default_rng(7).normal(size=(4,n*n))
    if allow:p.prepare_warm()
    result=p.solve_many(bs)
    assert sum(r.stats['controller_calls'] for r in result)==1
    assert sum(r.stats['policy_batch_decisions'] for r in result)==1
    if allow:
        assert all(r.stats['nn_forward_calls']==0 for r in result)
        assert any(r.stats['learned_transfer_apply_calls']>0 for r in result)
        assert all(r.abstention['chosen_branch']=='H_P' for r in result)
    else:
        assert not p.transfer_banks
        ref=PreparedStrongMG(a,n,None,replace(c,branch='C',mode='classical'),rr).solve_many(bs)
        for x,y in zip(result,ref):
            np.testing.assert_array_equal(x.x,y.x)
            assert x.residual_history==y.residual_history


def test_hp_policy_mutation_invalidates_transfer_banks():
    from adaptive_mg.v67.cost_policy import PreparedCostMG
    n=7;a=assemble_stiffness(DiffusionCase(n));m=model().frozen_inference_copy();c=config();rr=rules()
    p=PreparedCostMG(a,n,policy=hp_policy(m,c,rr),config=c,rules=rr);p.prepare_warm()
    assert p.transfer_banks
    with torch.no_grad():m.transfer.delta_head.weight.add_(.01)
    result=p.solve(np.ones(n*n))
    assert not p.transfer_banks and result.abstention['chosen_branch']=='C'
    assert result.abstention['cost_policy']['reason']=='stale_policy'


def test_asymptotic_training_resume_matches_uninterrupted_last_weights(tmp_path,monkeypatch):
    import adaptive_mg.v67.hp_training as train
    from adaptive_mg.v67.research_data import _generate_split
    rr=rules();forbidden=set()
    tr=_generate_split('train',dict(sizes=[7],families=['channel'],per_family=1,seed=801),forbidden,rr)[0]
    va=_generate_split('validation',dict(sizes=[7],families=['channel'],per_family=1,seed=802),forbidden,rr)[0]
    # Exact-resume contract independent of noisy elapsed time in model selection.
    monkeypatch.setattr(train,'validate_checkpoint',lambda *a,**k:dict(eligible=True,geometric_speedup=1.,rows=[],new_failures=[]))
    cfg=config();m=model();settings=dict(updates=3,random_probes=2,slow_probes=2,power_cycles=1,probe_cycles=3,
                                    validate_every=1,seed=52,learning_rate=.001)
    train.train_asymptotic(m,tr,va,cfg,rr,settings,tmp_path/'full')
    r=train.train_asymptotic(m,tr,va,cfg,rr,settings,tmp_path/'resume',max_updates=1)
    assert not r['all_updates_completed']
    train.train_asymptotic(m,tr,va,cfg,rr,settings,tmp_path/'resume',resume=True)
    first=Components.load(tmp_path/'full/last.pt');second=Components.load(tmp_path/'resume/last.pt')
    for name,tensor in first.transfer.state_dict().items():
        torch.testing.assert_close(tensor,second.transfer.state_dict()[name],rtol=0,atol=0)
    a=torch.load(tmp_path/'full/resume.pt',weights_only=True)['extra'];b=torch.load(tmp_path/'resume/resume.pt',weights_only=True)['extra']
    torch.testing.assert_close(a['rng'],b['rng'],rtol=0,atol=0)
    for key in a['probes']:torch.testing.assert_close(a['probes'][key],b['probes'][key],rtol=0,atol=0)


def test_actual_numerical_checkpoint_validator(tmp_path):
    from adaptive_mg.v67.hp_training import validate_checkpoint
    from adaptive_mg.v67.research_data import _generate_split
    rr=rules();va=_generate_split('validation',dict(sizes=[7],families=['near_isotropic'],per_family=1,seed=72),set(),rr)[0]
    report=validate_checkpoint(model(),va,config(),rr,rhs_count=2,repeats=1,tail_cycles=6)
    assert report['eligible'] and not report['new_failures'] and report['rows'][0]['C_success']
    assert report['rows'][0]['tail_rho']['H_P'] is not None


def test_small_em_workflow_and_synthetic_final(tmp_path,monkeypatch):
    """Only a declared tiny TEST holdout is opened; research OOD remains untouched."""
    from adaptive_mg.v67 import em_transfer_study as flow,three_pillars as tp,strong
    original_bank=strong.classical_bank
    def small_bank(name='controlled'):
        if name=='em_schedule':
            allowed={'line_alt_bilinear_full__v22','jacobi_bilinear_full__v11','line_alt_energymin_full__em5__v11'}
            return tuple(s for s in original_bank(name) if s.name in allowed)
        return original_bank(name)
    monkeypatch.setattr(strong,'classical_bank',small_bank)
    original=tp.make_research_plan
    def tiny_plan(**kwargs):
        plan=original(**kwargs)
        for key in tp.HOLDOUT:plan['splits'][key].update(sizes=[7],per_family=1,families=['channel'],count=1)
        return plan
    monkeypatch.setattr(tp,'make_research_plan',tiny_plan)
    settings=json.loads((tp.PROJECT/'configs/v6_7_em_hp_smoke.json').read_text())
    settings.update(smoke=False,seed=192831,training_sizes=[7],validation_sizes=[7],historical_roots=[str(tmp_path)])
    settings['training'].update(updates=2,validate_every=1,validation_tail_cycles=4)
    for i,spec in enumerate(settings['study']['policy_splits'].values()):spec['seed']=81071+i*10001
    path=tmp_path/'config.json';path.write_text(json.dumps(settings));out=tmp_path/'run'
    flow.calibrate(path,out);flow.train(out,max_updates=1);flow.train(out,resume=True)
    protocol=dict(repeats=1,warmups=0,rhs_counts=[1],regimes=['warm_multiple'])
    flow.benchmark(out,**protocol);flow.benchmark(out,resume=True,**protocol);flow.select(out)
    with pytest.raises(ValueError,match='fixed'):flow.train(out,resume=True)
    p=flow.fit_policy(out);flow.fit_policy(out,resume=True)
    assert p.branch=='H_P' and p.models.keys()=={'H_P'}
    flow.policy_validation(out,**protocol);flow.policy_validation(out,resume=True,**protocol)
    with pytest.raises(ValueError,match='retune'):flow.fit_policy(out,resume=True)
    flow.freeze(out,**protocol)
    assert not (out/'data/final_data_manifest.json').exists()
    flow.final(out)
    assert json.loads((out/'data/final_claim.json').read_text())['status']=='completed'
    with pytest.raises((ValueError,FileExistsError)):flow.final(out)
