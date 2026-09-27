"""Contracts for warm scope, numerical cascades, learning and continuous policy."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.banks import Stats, hybrid_cycle, selected_level
from adaptive_mg.v67.multistage import make_multistage
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.strong import StrongRules, PreparedStrongMG, select_strong_strategy
from adaptive_mg.v67.research_training import create_research_components, full_cycle_objective, noharm_penalty
from adaptive_mg.v67.research_data import _generate_split
from adaptive_mg.v67.unroll import make_graph, cycle
from adaptive_mg.v67.spatial import SpatialState
from adaptive_mg.v67.research_evaluation import measured_research
from adaptive_mg.v67.cost_policy import fit_cost_model,predict_cost


def example(n=7):
    return _generate_split('train',dict(sizes=[n],per_family=1,families=['near_isotropic'],seed=728191),set(),StrongRules())[0][0]


def config(**kwargs):
    return AdaptiveConfig(mg=MGConfig(mode='classical',max_cycles=40,pre_steps=2,post_steps=2,
                           nn_levels=1,stencil_backend='csr'),mode='research',branch='H_S',
                          spatial=False,gate_mode='open',**kwargs)


def model(stages=2,kind='cascade_cnn',level_count=1):
    m=create_research_components(smoother='student_cnn',smoother_hidden=4,transfer_hidden=4)
    m.smoother=make_multistage(kind=kind,stages=stages,hidden=4,level_count=level_count)
    return m


@pytest.mark.parametrize('stages',[1,2,3])
@pytest.mark.parametrize('kind',['cascade_cnn','tiny_polynomial'])
@pytest.mark.parametrize('group',[1,2])
def test_multistage_runtime_matches_differentiable_full_cycle(stages,kind,group):
    torch.set_num_threads(1)
    e=example();m=model(stages,kind)
    cfg=config(replace_pre=group,replacement_group_pre=group)
    selected=select_strong_strategy(e.a,e.n,StrongRules())
    cfg=replace(cfg,mg=replace(cfg.mg,strategy_name=selected.strategy_name))
    with torch.no_grad():
        for p in m.smoother.parameters():p.add_(.001)
    prepared=PreparedStrongMG(e.a,e.n,m,cfg,StrongRules());st=Stats()
    bank=prepared.ensure_branch('H_S',st)
    got=hybrid_cycle(bank,np.zeros_like(e.b),e.b,cfg,st,SpatialState(m,cfg),1,root_gate=np.ones_like(e.b,dtype=bool))
    graph=make_graph(e.a,(e.n,e.n),m,cfg)
    expected=cycle(graph,torch.zeros(e.b.size,dtype=torch.float64),torch.tensor(e.b),m,cfg)
    np.testing.assert_allclose(got,expected.detach().numpy(),rtol=2e-6,atol=2e-8)
    expected.square().sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.smoother.parameters())
    if stages>1:
        assert st.multistage_residual_matvecs==st.multistage_applications*(stages-1)
    assert st.replaced_classical_slots==group


def test_cascade_is_not_product_of_inverse_stencils():
    e=example();m=model();cfg=config();p=PreparedStrongMG(e.a,e.n,m,cfg,StrongRules());st=Stats()
    bank=p.ensure_branch('H_S',st).neural_stencil
    r=e.b;one=bank.stages[0].apply(r,Stats())
    expected=one+bank.stages[1].apply(r-e.a@one,Stats())
    np.testing.assert_allclose(bank.apply(r,Stats()),expected)
    assert not np.allclose(expected,bank.stages[1].apply(one,Stats()))


def test_multistage_checkpoint_roundtrip_and_shared_size_contract(tmp_path):
    m=model(3,level_count=2);path=tmp_path/'candidate.pt';m.save(path)
    restored=Components.load(path)
    assert restored.signature()==m.signature()
    for n in (7,15,31):
        d,g=restored.smoother.stage_directions_and_gains(torch.randn(1,10,n,n),1)
        assert d.shape==(1,3,9,n,n) and g.shape==(1,3)
    restored.smoother.train_only_level(1)
    assert not any(p.requires_grad for p in restored.smoother.levels[0].parameters())
    assert all(p.requires_grad for p in restored.smoother.levels[1].parameters())


def test_true_noharm_penalty_changes_gradient_and_detaches_reference():
    h=torch.tensor([.5,.2],requires_grad=True);c=torch.tensor([.1,.1],requires_grad=True)
    p=noharm_penalty(h,c,1e-8);p.backward()
    assert p>0 and h.grad.abs().sum()>0 and c.grad is None
    assert noharm_penalty(c.detach()*.5,c,1e-8)==0


def test_coarse_complement_and_noharm_objective_are_differentiable():
    e=example(15);m=model(2,level_count=2);cfg=config(smoother_levels=(0,1))
    cfg=replace(cfg,mg=replace(cfg.mg,strategy_name=e.strong_selection['strategy_name']))
    loss,details,_=full_cycle_objective(e,m,cfg,prefix=2,tail=1,lambda_noharm=.1,lambda_coarse=.1,coarse_probes=1)
    loss.backward()
    assert details['reference_history'].numel()==3
    assert torch.isfinite(loss) and any(p.grad is not None for p in m.smoother.parameters())


def test_warm_multiple_uses_independent_prime_zero_guess_and_no_generation_in_timer():
    e=example();m=model();cfg=config();arm=dict(model=m,branch='H_S')
    result=measured_research(e,arm,cfg,StrongRules(),regime='warm_multiple',rhs_count=4)
    assert result['success'] and result['independent_warm_prime']
    assert result['classical_setup_seconds']==0 and result['neural_setup_seconds']==0
    assert result['counters']['nn_forward_calls']==0 and result['warm_prime_counters']['nn_forward_calls']==1
    assert result['warm_prime_results'][0]['rhs_digest'] not in result['rhs_digests']
    assert len(set(result['rhs_digests']))==4
    assert len({r['x0_digest'] for r in result['rhs_results']})==1


def policy_rows(split,ids,ratio=1.5):
    result=[]
    for i in ids:
        for N in (225,961):
            for cached in (False,True):
                c=dict(N=N,nnz=7*N,depth=4,complexity=1.5,rhs_count=4,cached=cached,
                       strategy='line_alt_bilinear_full',operator_features={})
                result.append(dict(operator=f'{split}-{i}-{N}',context=c,C_success=True,H_success=True,
                    C_seconds=N*.001,H_seconds=N*.001/ratio,neural_used=True))
    return result


def test_policy_predicts_unseen_size_without_exact_n_table_and_abstains_outside_scope():
    tr=policy_rows('fit',range(4));tu=policy_rows('tune',range(3))
    m=fit_cost_model(tr,tu,settings={'max_size_extrapolation':20.,'extrapolation_penalty':.01})
    c=dict(tr[0]['context'],N=63**2,nnz=7*63**2)
    branch,detail=predict_cost(m,c)
    assert branch=='H_S' and detail['out_of_training_size']
    assert 'n=' not in json.dumps(m)
    assert predict_cost(m,dict(c,N=10**9))[0]=='C'
    with pytest.raises(ValueError,match='disjoint'):fit_cost_model(tr,tr)


def test_policy_blocks_tune_failures_and_uses_operator_not_repeat_coverage():
    tr=policy_rows('fit',range(4));tu=policy_rows('tune',range(3));tu[0]['H_success']=False
    m=fit_cost_model(tr,tu)
    assert predict_cost(m,tr[0]['context'])[0]=='C'
    assert len(m['tune_optimism_by_operator'])==6


def test_explicit_level_settings_and_group_validation():
    cfg=config(smoother_levels=[1],transfer_levels=[0])
    assert not selected_level(0,cfg,'smoother') and selected_level(1,cfg,'smoother')
    assert selected_level(0,cfg,'transfer')
    assert AdaptiveConfig.from_dict(cfg.to_dict())==cfg
    with pytest.raises(ValueError):config(replace_pre=1,replacement_group_pre=2)


def runtime_policy(m,cfg,rules,probes=0):
    from adaptive_mg.v67.cost_policy import ContinuousCostPolicy,config_scope
    from adaptive_mg.provenance import hardware_environment
    rows=policy_rows('fit',range(4));tune=policy_rows('tune',range(3))
    fitted=fit_cost_model(rows,tune,settings={'max_size_extrapolation':100.,'extrapolation_penalty':0.})
    # This test artifact forces the known strategy, independent of machine timings.
    fitted['strategies']=['jacobi_bilinear_full']
    fitted['tune_strategy_operators']={'jacobi_bilinear_full':6}
    fitted['beta']=[np.log(2.)]+[0.]*(len(fitted['beta'])-1)
    fitted['N_range']=[1,10000];fitted['rhs_range']=[1,64]
    return ContinuousCostPolicy(fitted,m,rules.digest(),config_scope(cfg),hardware_environment(),m.signature(),{},probes)


@pytest.mark.parametrize('probes',[0,1,2])
def test_continuous_runtime_reuses_parent_counts_probe_and_keeps_threshold(probes):
    from adaptive_mg.v67.cost_policy import PreparedCostMG
    e=example();m=model();cfg=config();rules=StrongRules();policy=runtime_policy(m,cfg,rules,probes)
    p=PreparedCostMG(e.a,e.n,policy=policy,config=cfg,rules=rules)
    initial=np.ones_like(e.b)*.01
    result=p.solve(e.b,initial)
    assert result.converged and result.requested_branch=='auto'
    assert result.stopping_threshold==max(cfg.mg.absolute_tolerance,cfg.mg.tolerance*np.linalg.norm(e.b-e.a@initial))
    assert result.executed_cycles<=cfg.mg.max_cycles
    assert result.stats['policy_probe_cycles']==probes
    assert result.stats['controller_calls']>=1
    assert result.abstention['selected_classical_strategy']==p.selection.strategy_name
    p.prepare_warm();total=p.learned_builds_total
    p.solve_many(np.stack([e.b,e.b*2]))
    assert p.learned_builds_total==total


def test_continuous_policy_rejects_changed_expert(tmp_path):
    from adaptive_mg.v67.cost_policy import ContinuousCostPolicy
    e=example();m=model();cfg=config();policy=runtime_policy(m,cfg,StrongRules())
    path=tmp_path/'policy.json';policy.save(path)
    with torch.no_grad():next(m.smoother.parameters()).add_(.01)
    with pytest.raises(ValueError,match='expert changed'):ContinuousCostPolicy.load(path,m)


def test_new_policy_cannot_affect_forced_neural_arm(monkeypatch):
    import adaptive_mg.v67.cost_policy as policies
    monkeypatch.setattr(policies,'predict_cost',lambda *a:pytest.fail('forced expert consulted policy'))
    e=example();cfg=config();r=measured_research(e,dict(branch='H_S',model=model()),cfg,StrongRules(),regime='cold')
    assert r['actual_neural_used']


def test_policy_feature_uses_actual_tensor_anisotropy_name_and_config_json_roundtrip(tmp_path):
    from adaptive_mg.v67.cost_policy import feature_vector,ContinuousCostPolicy,config_scope,PreparedCostMG
    ctx=policy_rows('fit',[0])[0]['context']
    a=feature_vector(ctx,[ctx['strategy']])
    b=feature_vector(dict(ctx,operator_features={'tensor_anisotropy_ratio':100.}),[ctx['strategy']])
    assert a[6]!=b[6]
    e=example(15);m=model(2,level_count=2).frozen_inference_copy();cfg=config(smoother_levels=(0,1))
    rules=StrongRules();policy=runtime_policy(m,cfg,rules)
    file=tmp_path/'policy.json';policy.save(file);loaded=ContinuousCostPolicy.load(file,m)
    assert loaded.solver_scope==config_scope(cfg)
    PreparedCostMG(e.a,e.n,policy=loaded,config=cfg,rules=rules)


def test_native_multistage_matches_csr_when_available():
    from adaptive_mg.native_stencil import native_available
    if not native_available():pytest.skip('optional native library not built')
    e=example(15);m=model(3);csr=config();native=replace(csr,mg=replace(csr.mg,stencil_backend='native',native_min_cells=1))
    pc=PreparedStrongMG(e.a,e.n,m,csr,StrongRules());pn=PreparedStrongMG(e.a,e.n,m,native,StrongRules())
    bc=pc.ensure_branch('H_S',Stats());bn=pn.ensure_branch('H_S',Stats())
    np.testing.assert_allclose(bc.neural_stencil.apply(e.b,Stats()),bn.neural_stencil.apply(e.b,Stats()),rtol=1e-12,atol=1e-12)


def test_new_loss_and_policy_settings_reject_invalid_values():
    with pytest.raises(ValueError):noharm_penalty(torch.ones(1),torch.ones(1),1e-8,float('nan'))
    with pytest.raises(ValueError):fit_cost_model(policy_rows('fit',[0,1]),policy_rows('tune',[0,1]),settings={'ridge':-1})


def test_full_warm_study_workflow_pins_expert_policy_and_synthetic_final(tmp_path,monkeypatch):
    """Tiny declared TEST plan only; never creates the real research holdout."""
    from adaptive_mg.v67 import three_pillars as old, warm_study as flow
    original=old.make_research_plan
    def tiny_plan(**kwargs):
        plan=original(**kwargs)
        for key in old.HOLDOUT:
            plan['splits'][key].update(sizes=[7],per_family=1,families=['channel'],count=1)
        return plan
    monkeypatch.setattr(old,'make_research_plan',tiny_plan)
    settings=json.loads((old.PROJECT/'configs/v6_7_warm_study_smoke.json').read_text())
    settings.update(smoke=False,seed=618273,historical_roots=[str(tmp_path)])
    settings['training'].update(updates=2,seed=618273)
    for i,spec in enumerate(settings['study']['policy_splits'].values()):spec['seed']=998129+i*10001
    path=tmp_path/'config.json';path.write_text(json.dumps(settings));out=tmp_path/'run'
    flow.calibrate(path,out);data=flow.prepare_data(out)
    all_groups=[e.group_digest for rows in data.values() for e in rows]
    assert len(all_groups)==len(set(all_groups))
    with pytest.raises(FileNotFoundError):flow.fit_policy(out)
    flow.train_variants(out,['H2_L']);flow.train_variants(out,['H2_L'],resume=True)
    protocol=dict(repeats=1,warmups=0,rhs_counts=(1,),regimes=('warm_multiple',))
    flow.benchmark_experts(out,['H2_L'],**protocol)
    flow.select_expert(out,name='H2_L')
    with pytest.raises(ValueError,match='selected'):flow.train_variants(out,['H2_L'],resume=True)
    policy=flow.fit_policy(out);flow.fit_policy(out,resume=True)
    assert policy.model['train_operator_ids']!=policy.model['tune_operator_ids']
    with pytest.raises(FileNotFoundError):flow.freeze(out,**protocol)
    report=flow.validate_policy(out,probes=(0,1),**protocol)
    flow.validate_policy(out,probes=(0,1),resume=True,**protocol)
    coverage=json.loads((out/'benchmarks/policy_validation/policy_coverage.json').read_text())
    assert coverage['rows'] and all(not r['certificate'] for r in coverage['rows'])
    flow.freeze(out,probes=0,**protocol)
    assert not (out/'data/final_data_manifest.json').exists()
    with pytest.raises(ValueError,match='frozen'):flow.fit_policy(out,resume=True)
    flow.final(out)
    claim=json.loads((out/'data/final_claim.json').read_text())
    assert claim['status']=='completed'
    with pytest.raises((ValueError,FileExistsError)):flow.final(out)
    with pytest.raises((ValueError,FileExistsError)):flow.final(out,resume=True)
