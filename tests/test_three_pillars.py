"""Regression coverage for baseline integrity, replacement cost and safe P."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg import DiffusionCase, assemble_stiffness, MGConfig
from adaptive_mg.hierarchy import (WorkStats, StencilBank, stencil_from_coefficients,
                                   build_fixed_hierarchy)
from adaptive_mg.smoothers import LineSmootherCache, _directional, classical_smoothing_correction
from adaptive_mg.strategy import get_strategy
from adaptive_mg.transfer import (build_transfer_pattern, coarse_fine_indices,
                                 weights_from_sparse_matrix)
from adaptive_mg.v67.banks import Stats, prepare_smoother_bank, resolve_device
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.strong import StrongRules, select_strong_strategy, PreparedStrongMG
from adaptive_mg.v67.strong_calibration import calibrate_multisize
from adaptive_mg.v67.research_training import create_research_components, full_cycle_objective
from adaptive_mg.v67.research_transfer import project_transfer_weights, TransferComplexityError

CAPS=dict(max_row_nnz=16,max_p_ratio=1.,max_ac_ratio=1.15,max_operator_complexity=3.,
          complexity_reference='parent',max_operator_complexity_ratio=1.15)


def config(**kw):
    return AdaptiveConfig(mg=MGConfig(mode='classical',max_cycles=30,pre_steps=2,post_steps=2,
                                    nn_levels=1,stencil_backend='csr'),
                          mode='research',branch='H_P',spatial=False,gate_mode='open',**kw)


def student(transfer='small_gnn'):
    return create_research_components(smoother='student_cnn',transfer=transfer,
        smoother_hidden=4,transfer_hidden=4,support='support_preserving',complexity_caps=CAPS)


@pytest.mark.parametrize('shape',[(7,15),(15,7),(15,31)])
@pytest.mark.parametrize('direction',['x','y','diag45'])
@pytest.mark.parametrize('reverse',[False,True])
def test_rectangular_batched_sweep_and_work_match(shape,direction,reverse):
    nx,ny=shape
    tx=sp.diags([-np.ones(nx-1),2*np.ones(nx),-np.ones(nx-1)],[-1,0,1])
    ty=sp.diags([-np.ones(ny-1),2*np.ones(ny),-np.ones(ny-1)],[-1,0,1])
    a=(sp.kron(tx,sp.eye(ny))+sp.kron(sp.eye(nx),ty)).tocsr()
    r=np.random.default_rng(193).normal(size=nx*ny)
    slow=LineSmootherCache(a,shape);fast=LineSmootherCache(a,shape)
    ws,wf=WorkStats(),WorkStats()
    expected,_=_directional(np.zeros_like(r),r.copy(),slow.get(direction),reverse=reverse,work=ws,safety=False)
    got=classical_smoothing_correction(a,shape,r,'line_'+direction,cache=fast,reverse=reverse,work=wf)
    np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-13)
    assert wf.work_flops==ws.work_flops
    assert fast.factorization_count<=2
    count=fast.factorization_count;fast.prepare(direction)
    assert fast.factorization_count==count


def test_wide_same_colour_coupling_rejects_batch_before_any_factor():
    shape=(7,7);a=sp.eye(49,format='lil')*8
    # x-lines have fixed y, so y=0 and y=2 are in the same zebra colour.
    a[0,2]=a[2,0]=-.1
    cache=LineSmootherCache(a.tocsr(),shape)
    assert cache.zebra_batches('x') is None
    assert cache.factorization_count==0
    cache.prepare('x')
    assert cache.factorization_count==7


def test_selector_v2_uses_joint_direction_heterogeneity_and_coverage_fallback():
    a=assemble_stiffness(DiffusionCase(n=15,epsilon=1e-3,angle_deg=45,contrast=100,pattern='channel'))
    rules=StrongRules(require_coverage=True)
    selected=select_strong_strategy(a,15,rules)
    assert selected.rule_evidence['heterogeneous']
    assert selected.rule_id in ('diagonal_positive_heterogeneous','heterogeneous_orientation')
    assert selected.strategy_name==rules.fallback_strategy_name
    with pytest.raises(ValueError,match='stale'):
        StrongRules.from_dict({**rules.to_dict(),'version':'a_only_moment_rules_v1'})
    poisson=assemble_stiffness(DiffusionCase(n=15,epsilon=1.))
    fitted=replace(rules,coverage_by_rule=(('near_isotropic',7,15),))
    assert select_strong_strategy(poisson,15,fitted).strategy_name=='jacobi_bilinear_full'
    assert select_strong_strategy(assemble_stiffness(DiffusionCase(n=31,epsilon=1.)),31,fitted).strategy_name==rules.fallback_strategy_name


def calibration_row(i,n,rules,*,bad=False,slow=False):
    def run(t,success=True,cycles=10):
        return dict(seconds=t,success=success,cycles=[cycles,cycles])
    return dict(normalized_operator_digest=f'group-{i}',n=n,rule_id='near_isotropic',
                rules_digest=rules.digest(),rhs_count=2,runs={
                'line_alt_bilinear_full':[run(2.)],
                'jacobi_bilinear_full':[run(1.,not bad,149 if slow else 10)]})


@pytest.mark.parametrize('case',['missing_size','tune_failure','low_headroom','good'])
def test_calibration_requires_multisize_tune_and_every_rhs(case):
    rules=StrongRules(require_coverage=True)
    train=[calibration_row(0,15,rules),calibration_row(1,31,rules)]
    tune=[calibration_row(2,15,rules),calibration_row(3,31 if case!='missing_size' else 15,rules,
            bad=case=='tune_failure',slow=case=='low_headroom')]
    result,_=calibrate_multisize(train,tune,rules,required_sizes=[15,31],minimum_leaf_cases=2)
    assert dict(result.strategy_by_rule)['near_isotropic']==('jacobi_bilinear_full' if case=='good' else 'line_alt_bilinear_full')
    assert ('near_isotropic',15,31) in result.coverage_by_rule if case=='good' else not result.coverage_by_rule
    with pytest.raises(ValueError,match='overlap'):
        calibrate_multisize(train,train,rules,required_sizes=[15,31],minimum_leaf_cases=2)


@pytest.mark.parametrize('coarse',[(3,3),(3,7),(7,3)])
def test_preserving_projection_has_no_external_support_or_gradient(coarse):
    pattern=build_transfer_pattern((7,7),coarse)
    model=student().transfer
    delta=(1e-6*torch.randn(1,16,7,7,dtype=torch.float64)).requires_grad_()
    w=project_transfer_weights(model,pattern,delta,pattern.bilinear_weights)
    base=pattern.bilinear_weights
    np.testing.assert_allclose(w.detach().numpy().sum(1),base.sum(1),rtol=0,atol=2e-15)
    assert np.all(w.detach().numpy()[base==0]==0)
    coarse_rows=coarse_fine_indices(pattern)
    np.testing.assert_array_equal(w.detach().numpy()[coarse_rows],base[coarse_rows])
    (w.square().sum()).backward()
    grad=delta.grad.permute(0,2,3,1).reshape_as(w).numpy()
    assert np.all(grad[base==0]==0)
    assert np.isfinite(grad).all()


@pytest.mark.parametrize('angle',[0.,90.])
@pytest.mark.parametrize('n',[15,31])
def test_identity_p_survives_high_complexity_semicoarsening(n,angle):
    a=assemble_stiffness(DiffusionCase(n=n,epsilon=1e-3,angle_deg=angle))
    prepared=PreparedStrongMG(a,n,student(),config(),StrongRules())
    st=Stats();bank=prepared.ensure_branch('H_P',st)
    assert bank.p is prepared.classical.p
    assert bank.coarse.a is prepared.classical.coarse.a
    assert bank.research_complexity['operator_complexity_ratio']==1.
    assert st.transfer_identity_levels>=1
    assert st.setup_failures==0


@pytest.mark.parametrize('transfer',['existing','small_gnn','edge_mlp'])
def test_perturbed_safe_p_train_and_runtime_share_projection(transfer):
    from adaptive_mg.v67.data import dataset
    from adaptive_mg.v67.unroll import make_graph
    from adaptive_mg.v67.research_training import sample_config, transfer_feasibility
    e=dataset([7],7,193324)[0];rules=StrongRules();model=student(transfer)
    e.strong_selection=select_strong_strategy(e.a,e.n,rules).to_dict()
    with torch.no_grad():
        for parameter in model.transfer.parameters():parameter.add_(.001)
    cfg=sample_config(e,config(),rules,'H_P')
    deployed=PreparedStrongMG(e.a,e.n,model,cfg,rules)
    bank=deployed.ensure_branch('H_P',Stats())
    trained=make_graph(e.a,(e.n,e.n),model,cfg,learned=True)
    report,_=transfer_feasibility(trained,model,cfg)
    assert report['feasible']
    np.testing.assert_allclose(trained.p.numpy().toarray(),bank.p.toarray(),rtol=1e-6,atol=1e-9)
    w=weights_from_sparse_matrix(deployed.classical.pattern,bank.p)
    assert np.all(w[deployed.classical.base_weights==0]==0)


def test_infeasible_fixed_support_skips_inference_and_numeric_galerkin(monkeypatch):
    import adaptive_mg.v67.banks as banks
    a=assemble_stiffness(DiffusionCase(n=15,epsilon=1.))
    m=student();m.transfer.complexity_caps['max_row_nnz']=1
    p=PreparedStrongMG(a,15,m,config(),StrongRules())
    def forbidden(*args,**kwargs):raise AssertionError('expensive operation before precheck')
    monkeypatch.setattr(banks,'generated_p',forbidden)
    monkeypatch.setattr(banks,'galerkin_coarse_operator',forbidden)
    with pytest.raises(TransferComplexityError,match='precheck'):
        p.ensure_branch('H_P',Stats())


def test_inference_model_prepared_once_and_checkpoint_not_mutated():
    a=assemble_stiffness(DiffusionCase(n=15,epsilon=1.))
    m=student();m.smoother.double().train()
    cfg=replace(config(),branch='H_S',mg=replace(config().mg,nn_levels=2))
    p=PreparedStrongMG(a,15,m,cfg,StrongRules());st=Stats()
    p.ensure_branch('H_S',st)
    assert st.smoother_nn_calls==2
    assert st.inference_model_preparations==1
    assert m.smoother.training and next(m.smoother.parameters()).dtype==torch.float64


def test_native_bank_does_not_materialize_csr_until_requested():
    class Native:
        def apply(self,r):return r[:,None]
    calls=[]
    def factory():calls.append(1);return sp.eye(7,format='csr')
    bank=StencilBank(None,Native(),csr_factory=factory,work_nnz=7)
    np.testing.assert_array_equal(bank.apply(np.arange(7),Stats()),np.arange(7))
    assert not calls
    assert bank.csr.nnz==7 and len(calls)==1
    assert bank.csr.nnz==7 and len(calls)==1


def test_stencil_numeric_arrays_are_not_shared():
    values=np.ones((9,7,7));diag=np.ones(49)
    first=stencil_from_coefficients(values,1.,diag,(7,7))
    second=stencil_from_coefficients(values*2,1.,diag,(7,7))
    before=second.data.copy();first.data[:]=0;first.eliminate_zeros()
    np.testing.assert_array_equal(second.data,before)


def test_frozen_signature_cache_detects_normal_torch_updates():
    original=student();snapshot=original.frozen_inference_copy()
    sig=snapshot.signature()
    assert snapshot.signature()==sig
    with torch.no_grad():next(snapshot.smoother.parameters()).add_(.1)
    assert snapshot.signature()!=sig
    assert original.signature()==sig


def test_size_aware_auto_device_stays_cpu_for_small_workloads(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    cfg=replace(config(),inference_device='auto',auto_device_min_cells=4096)
    assert str(resolve_device(cfg,cells=31*31))=='cpu'
    assert str(resolve_device(cfg,cells=127*127))=='cuda'


def test_end_to_end_smoke_from_scratch(tmp_path):
    from adaptive_mg.v67 import three_pillars as workflow
    settings=json.loads((workflow.PROJECT/'configs/v6_7_three_pillars_smoke.json').read_text())
    settings['hidden']=4;settings['training']['updates']=1
    settings['seed']=settings['training']['seed']=733102
    settings['historical_roots']=[str(tmp_path)]
    config_path=tmp_path/'settings.json';config_path.write_text(json.dumps(settings))
    out=tmp_path/'run'
    workflow.calibrate(config_path,out)
    workflow.train(out,['H_S','H_P'])
    result=workflow.benchmark(out,['H_S','H_P'],repeats=1,warmups=0,
                              rhs_counts=(1,),regimes=('cold',))
    assert result['summary']['cold']['1']['strong_C']['total']==14
    assert (out/'checkpoints/H_P/candidate.pt').exists()
    assert not (out/'data/final_data_manifest.json').exists()
    with pytest.raises(ValueError,match='Smoke'):
        workflow.freeze(out,['H_S','H_P'],repeats=1,warmups=0,rhs_counts=(1,),regimes=('cold',))


def test_small_final_workflow_seals_exact_frozen_inputs(tmp_path, monkeypatch):
    """Exercise CLI final wiring on a tiny declared plan, not the real final set."""
    from adaptive_mg.v67 import three_pillars as workflow
    original_plan=workflow.make_research_plan
    def tiny_plan(**kwargs):
        plan=original_plan(**kwargs)
        for name in workflow.HOLDOUT:
            plan['splits'][name].update(sizes=[7],per_family=1,families=['channel'],count=1)
        return plan
    monkeypatch.setattr(workflow,'make_research_plan',tiny_plan)
    settings=json.loads((workflow.PROJECT/'configs/v6_7_three_pillars_smoke.json').read_text())
    settings.update(smoke=False,hidden=4,seed=1718301,historical_roots=[str(tmp_path)])
    settings['training']['seed']=1718301;settings['training']['updates']=1
    config_path=tmp_path/'settings.json';config_path.write_text(json.dumps(settings))
    out=tmp_path/'run'
    workflow.calibrate(config_path,out)
    workflow.train(out,['H_P'])
    workflow.freeze(out,['H_P'],repeats=1,warmups=0,rhs_counts=(1,),regimes=('cold',))
    assert not (out/'data/final_data_manifest.json').exists()
    workflow.final(out)
    claim=json.loads((out/'data/final_claim.json').read_text())
    assert claim['status']=='completed' and len(claim['results'])==len(workflow.HOLDOUT)
    with pytest.raises(ValueError,match='frozen'):
        workflow.train(out,['H_P'],resume=True)
    with pytest.raises((ValueError,FileExistsError)):
        workflow.final(out)


def test_partial_calibration_resume_keeps_committed_dataset(tmp_path,monkeypatch):
    from adaptive_mg.v67 import three_pillars as workflow
    settings=json.loads((workflow.PROJECT/'configs/v6_7_three_pillars_smoke.json').read_text())
    settings['historical_roots']=[str(tmp_path)]
    path=tmp_path/'cfg.json';path.write_text(json.dumps(settings));out=tmp_path/'run'
    original=workflow.measure_classical_portfolio
    def interrupted(*args,**kwargs):raise RuntimeError('synthetic interruption')
    monkeypatch.setattr(workflow,'measure_classical_portfolio',interrupted)
    with pytest.raises(RuntimeError,match='synthetic'):
        workflow.calibrate(path,out)
    before=(out/'calibration_manifest.json').read_bytes()
    def forbidden(*args,**kwargs):raise AssertionError('resume must not regenerate/rescan')
    monkeypatch.setattr(workflow,'historical_operator_index',forbidden)
    monkeypatch.setattr(workflow,'_generate_split',forbidden)
    monkeypatch.setattr(workflow,'measure_classical_portfolio',original)
    workflow.calibrate(path,out,resume=True)
    assert (out/'calibration_manifest.json').read_bytes()==before
