"""Fixed-parent comparison and H_S-only world-model numerical contracts."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.research_training import create_research_components
from adaptive_mg.v67.research_data import _generate_split
from adaptive_mg.v67.research_evaluation import evaluate_research, measured_research
from adaptive_mg.v67.strong import StrongRules
from adaptive_mg.v67.banks import Stats
from adaptive_mg.v67.world_model.data import snapshot
from adaptive_mg.v67.hs_world import tuning, study, temporal, learning
from adaptive_mg.v67.hs_world.backend import HSSmoothingBackend,ACTIONS,levels


def cfg():
    return AdaptiveConfig(mg=MGConfig(mode='classical',strategy_name='line_alt_bilinear_full__v11',
        max_cycles=80,stencil_backend='csr'),mode='research',branch='H_S',use_smoother=True,
        use_transfer=False,spatial=False,gate_mode='open',use_learned_controller=False)


def rules():return tuning.fixed_rules(cfg().mg.strategy_name)


def examples(n=7):
    return _generate_split('validation',dict(sizes=[n],per_family=1,families=['near_isotropic'],seed=775341),set(),rules())[0]


def trained():
    # Unit-test fixture only; smoke/integration below performs genuine training.
    m=create_research_components(smoother='student_cnn',smoother_hidden=4,transfer_hidden=4,seed=17)
    m.metadata.update(training_branch='H_S',optimizer_updates=1,training_rules_digest=rules().digest())
    return m


def state(e,t=0,factor=1.):
    a=e.a*factor
    return snapshot(a,a@e.exact,shape=(e.n,e.n),time=t*.1,index=t,mesh_id='mesh',boundary_id='dirichlet')


@pytest.mark.parametrize('n',[7,15,31])
def test_one_global_plan_is_constant_across_operators(n):
    from adaptive_mg.v67.strong import select_strong_strategy
    e=examples(n)[0];s=select_strong_strategy(e.a,n,rules())
    assert s.strategy_name==cfg().mg.strategy_name
    assert not s.rule_evidence['fallback_for_coverage']


def test_global_selection_uses_fit_not_tune_and_never_picks_tune_runner_up():
    def rows(prefix,seconds,success=None):
        return [dict(normalized_operator_digest=prefix+str(i),runs={p:[dict(success=(success or {}).get(p,True),seconds=v)] for p,v in seconds.items()}) for i in range(3)]
    fit=rows('fit',{'A':3.,'B':1.,'C':2.});tune=rows('tune',{'A':5.,'B':10.,'C':.01})
    result=tuning.fit_global(fit,tune,['A','B','C'],'A')
    assert result['plan']=='B'
    tune=rows('tune',{'A':5.,'B':10.,'C':.01},{'B':False})
    result=tuning.fit_global(fit,tune,['A','B','C'],'A')
    assert result['plan']=='A' and result['fit_winner']=='B'
    with pytest.raises(ValueError):tuning.fit_global(fit,fit,['A','B','C'],'A')


def test_named_reference_and_secondary_strong_parent_are_not_mislabelled(tmp_path):
    e=examples();c=cfg();r=rules();strong=tuning.fixed_rules('jacobi_bilinear_full__v22')
    arms={'C_tuned':dict(model=None,branch='C'),'H0':dict(model=trained(),branch='H_S'),
          'strong_C':dict(model=None,branch='C',rules=strong,robustness_only=True)}
    report=evaluate_research(e,arms,c,r,tmp_path,repeats=1,warmups=0,rhs_counts=(1,),
                             regimes=('warm_multiple',),reference_arm='C_tuned')
    assert report['reference_arm']=='C_tuned'
    assert all('speedup_vs_reference' in row and 'speedup_vs_strong' not in row for row in report['table'])
    rr=report['rows'][0]['runs']['strong_C']['warm_multiple']['1'][0]
    assert rr['selection']['strategy_name']=='jacobi_bilinear_full__v22'
    h=report['rows'][0]['runs']['H0']['warm_multiple']['1'][0]
    assert h['selection']['strategy_name']==c.mg.strategy_name
    report2=evaluate_research(e,arms,c,r,tmp_path,repeats=1,warmups=0,rhs_counts=(1,),
                             regimes=('warm_multiple',),reference_arm='C_tuned',resume=True)
    assert report['table']==report2['table']
    arms['H0']['rules']=strong
    with pytest.raises(ValueError,match='robustness'):
        evaluate_research(e,arms,c,r,tmp_path/'bad',reference_arm='C_tuned')


@pytest.mark.parametrize('action',ACTIONS)
def test_all_hs_actions_never_call_transfer_generator_and_use_current_A(monkeypatch,action):
    import adaptive_mg.v67.banks as banks
    import adaptive_mg.v67.world_model.backend as old
    monkeypatch.setattr(banks,'generated_p',lambda *a:pytest.fail('learned P called'))
    monkeypatch.setattr(old,'generated_p',lambda *a:pytest.fail('learned P called'))
    e=examples()[0];backend=HSSmoothingBackend(cfg(),rules(),trained())
    first=backend.solve(state(e),None,'REBUILD_HS');s=state(e,1,1.07)
    result=backend.solve(s,first.bank,action)
    assert result.success and not result.bank.neural_transfer
    assert np.linalg.norm(s.b-s.a@result.x)<=result.threshold
    active=any(l.neural_stencil is not None for l in levels(result.bank.root))
    if not result.fallback:assert active==action.endswith('HS')
    for level in levels(result.bank.root):
        assert not getattr(level,'learned_transfer',False)
        if level.coarse is not None:
            np.testing.assert_allclose(level.coarse.a.toarray(),(level.p.T@level.a@level.p).toarray(),rtol=1e-12,atol=1e-12)
    assert result.stats['builds'][0]['numeric_refactorized']


def test_reuse_toggle_S_without_stale_bank_and_candidate_mutation():
    e=examples()[0];s=state(e);b=HSSmoothingBackend(cfg(),rules(),trained());sel,c=b.select(s)
    first,st=b.build(s,None,'REBUILD_HS',sel,c)
    pure,st=b.build(s,first,'REUSE_C',sel,c)
    assert pure.root is first.classical_root and pure.smoother_root is first.smoother_root
    active,st=b.build(s,pure,'REUSE_HS',sel,c)
    assert active.root is first.root and st['exact_matrix_cache_hit']
    assert st['smoother_nn_calls']==0
    changed=state(e,1,1.01);sel,c=b.select(changed)
    newer,st=b.build(changed,active,'REUSE_HS',sel,c)
    assert st['numeric_refactorized'] and st['smoother_nn_calls']>0
    assert first.matrix_digest==s.matrix_digest
    assert not b.compatible(replace(changed,mesh_id='different'),newer,sel.strategy_name)


def test_HP_checkpoint_rejected_and_classical_action_mask():
    m=trained();m.metadata['training_branch']='H_P'
    with pytest.raises(ValueError,match='H_S'):HSSmoothingBackend(cfg(),rules(),m)
    b=HSSmoothingBackend(cfg(),rules());e=examples()[0];s=state(e)
    assert b.available(s,None).tolist()==[True,False,False,False]


def test_policy_abstains_to_tuned_reuse_not_rebuild():
    times=np.ones((2,4));probs=np.ones((2,4));rolls=np.ones((2,4))
    cal=dict(margin=[[1.]*4 for _ in range(4)],episode_coverage=[[3]*4 for _ in range(4)],failure_counts=[[0]*4 for _ in range(4)])
    a,d=learning.choose((times,probs,rolls),[True]*4,cal,1)
    assert a==1 and d['reference']=='REUSE_C'
    cal['margin'][1][3]=0;times[:,3]=.5;rolls[:,3]=.5
    assert learning.choose((times,probs,rolls),[True]*4,cal,1)[0]==3


def test_trajectory_orchestration_keeps_real_neural_count_and_order():
    e=examples()[0];b=HSSmoothingBackend(cfg(),rules(),trained())
    w=dict(max_age=8,max_matrix_change=.5,horizon=2)
    solver=temporal.WorldSolver(b,w,heuristic_name='reuse',neural=True)
    _,r=solver.step(state(e));assert r['actual_neural_used'] and not r['learned_transfer']
    _,r=solver.step(state(e,1,1.04));assert r['actual_action']=='REUSE_HS' or r['fallback']
    assert r['total_seconds']>=r['setup_seconds']
    with pytest.raises(ValueError,match='order'):solver.step(state(e,1))
    solver.reset();solver.step(state(e))


def small_config(tmp_path):
    root=Path(__file__).parents[1]
    c=json.loads((root/'configs/v6_7_hs_world_smoke.json').read_text())
    c['historical_roots']=[str(tmp_path)];c['calibration_sizes']=[7,15];c['training_sizes']=[7,15]
    c['families']=['near_isotropic'];c['training']['updates']=2
    c['training']['prefix_cycles']=1;c['training']['tail_cycles']=1
    c['study']['variants']={k:c['study']['variants'][k] for k in ('H0','H1')}
    c['thesis']['classical_candidates']=['line_alt_bilinear_full__v22','line_alt_bilinear_full__v11']
    c['world'].update(counts=[2,2,2,2],steps=3,sizes=[7],epochs=2,ensemble=1,hidden=4,
                      heuristics=['rebuild','reuse'],max_age=4)
    p=tmp_path/'config.json';p.write_text(json.dumps(c));return p


def test_full_fixed_HS_temporal_workflow_and_frozen_test(tmp_path):
    config=small_config(tmp_path);out=tmp_path/'run'
    tuning.calibrate(config,out);tuning.calibrate(config,out,resume=True)
    es=study.data(out)
    assert not set(e.group_digest for e in es['train'])&set(e.group_digest for e in es['validation'])
    study.train(out,['H0','H1']);study.train(out,['H1'],resume=True)
    kw=dict(repeats=1,warmups=0,rhs_counts=(1,2),regimes=('warm_multiple',))
    report=study.benchmark(out,['H0','H1'],**kw)
    assert 'strong_C' not in report['summary']['warm_multiple']['1']
    study.select(out,name='H0')
    with pytest.raises(ValueError):study.train(out,['H0'])
    temporal.prepare(out);temporal.collect(out);temporal.collect(out,resume=True)
    temporal.train(out)
    report=temporal.evaluate(out,repeats=1)
    assert set(('World_C','World_HS','HS_matched_reuse','C_tuned_reuse')).issubset(report['summary'])
    assert report['learned_transfer'] is False and report['combustion_validated'] is False
    for row in report['rows']:
        assert all(not r['neural_systems'] for r in row['runs']['World_C'])
    with pytest.raises(ValueError):temporal.evaluate(out,repeats=1,resume=True)
    temporal.freeze(out)
    with pytest.raises(ValueError):study.train(out,['H0'])
    test=temporal.evaluate(out,split='test',repeats=1)
    assert test['protocol']['split']=='test'
    with pytest.raises(ValueError):temporal.evaluate(out,split='test',repeats=1,resume=True)
    # Test artifact tampering is not silently accepted.
    path=out/'temporal/world_HS.pt';blob=path.read_bytes();path.write_bytes(blob+b'changed')
    with pytest.raises(ValueError,match='changed'):temporal.check_freeze(out)


def test_bad_neural_trial_rolls_back_to_same_current_C_and_original_tolerance(monkeypatch):
    import adaptive_mg.v67.world_model.backend as core
    monkeypatch.setattr(core,'hybrid_cycle',lambda level,x,*a,**kw:np.full_like(x,np.nan))
    e=examples()[0];s=state(e);b=HSSmoothingBackend(cfg(),rules(),trained())
    r=b.solve(s,None,'REBUILD_HS')
    assert r.success and r.fallback and r.actual_action=='REBUILD_C'
    assert r.stats['neural_trial_cycles']==1 and r.stats['accepted_neural_cycles']==0
    assert r.cycles<=cfg().mg.max_cycles
    assert r.threshold==max(cfg().mg.absolute_tolerance,cfg().mg.tolerance*np.linalg.norm(s.b))
    assert r.bank.plan==cfg().mg.strategy_name


def test_classical_heuristics_do_not_pay_for_NN_feature_extraction(monkeypatch):
    monkeypatch.setattr(temporal,'observation',lambda *a,**kw:pytest.fail('unneeded NN feature scan'))
    e=examples()[0];b=HSSmoothingBackend(cfg(),rules(),trained())
    w=dict(max_age=8,max_matrix_change=.5,horizon=2)
    for name in ('rebuild','reuse','drift_0.05','periodic_2'):
        r=temporal.run_episode([state(e),state(e,1,1.01)],b,w,heuristic_name=name)
        assert r['success'] and r['neural_systems']==0


def test_static_and_temporal_freeze_rejects_reselection_and_config_tampering(tmp_path):
    config=small_config(tmp_path);out=tmp_path/'run'
    tuning.calibrate(config,out);study.data(out)
    path=out/'selector_rules.json';r=json.loads(path.read_text());r['provenance']='changed'
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='changed'):tuning.load(out)


def test_world_checkpoint_action_version_is_not_legacy_compatible(tmp_path):
    import adaptive_mg.v67.world_model.learning as legacy
    artifact=dict(models=[learning.WorldNet(3,4)],xmean=np.zeros(3,np.float32),xscale=np.ones(3,np.float32),
        ymean=np.zeros(4,np.float32),yscale=np.ones(4,np.float32),hidden=4,version='old',actions=list(ACTIONS))
    path=tmp_path/'model.pt';legacy.save_model(path,artifact)
    with pytest.raises(ValueError,match='action'):learning.load_model(path)


def test_optional_native_hs_actions_agree_with_csr():
    from adaptive_mg.native_stencil import native_available
    if not native_available():pytest.skip('optional native kernel not built')
    c=cfg();c=replace(c,mg=replace(c.mg,stencil_backend='native',native_min_cells=1))
    e=examples()[0];s=state(e);m=trained()
    a=HSSmoothingBackend(cfg(),rules(),m).solve(s,None,'REBUILD_HS')
    b=HSSmoothingBackend(c,rules(),m).solve(s,None,'REBUILD_HS')
    assert a.success and b.success and a.cycles==b.cycles
    np.testing.assert_allclose(a.x,b.x,rtol=1e-11,atol=1e-12)


def test_optional_strong_audit_uses_only_declared_calibration_data(tmp_path,monkeypatch):
    config=small_config(tmp_path);out=tmp_path/'run';tuning.calibrate(config,out)
    seen=[]
    def measure(out,label,examples,cfg,rules,settings,**kwargs):
        assert label.startswith('strong_selector_') and kwargs['bank']=='em_schedule'
        seen.extend(e.research_split for e in examples)
        return []
    def calibrate(*a,**kw):return StrongRules(),{'audit_only':True}
    monkeypatch.setattr(tuning,'measure_split',measure)
    monkeypatch.setattr(tuning,'calibrate_multisize',calibrate)
    before=(out/'selector_rules.json').read_bytes()
    tuning.strong_audit(out)
    assert set(seen)=={'selector_train','selector_tune'}
    assert (out/'selector_rules.json').read_bytes()==before
    assert (out/'strong_audit_rules.json').exists()


def test_gamg_comparator_rejects_synthetic_and_mismatched_rhs(tmp_path,monkeypatch):
    from adaptive_mg.v67.hs_world import cfd_compare as mod
    from adaptive_mg.v67.world_model.data import write_json
    header={'source_kind':'synthetic_elliptic'}
    monkeypatch.setattr(mod,'load',lambda out:(tmp_path,{},header,None))
    with pytest.raises(ValueError,match='synthetic'):mod.compare(tmp_path,'missing.json')
    header.update(source_kind='external_cfd',data_sha256='data')
    sys=dict(step=0,matrix_digest='A',rhs_digest='b',x0_digest='x0',threshold=1e-8,final_true_residual=1e-9,success=True)
    hw=dict(machine='x',cpu_model='y',affinity_count=1,torch_threads=1)
    report=dict(hardware=hw,rows=[dict(trajectory='case',runs={'World_HS':[dict(success=True,total_seconds=1.,rows=[sys])]})])
    write_json(tmp_path/'temporal/validation/report.json',report)
    ref=dict(schema='openfoam-gamg-sequence-v1',solver='GAMG',data_sha256='data',time_scope=mod.SCOPE,hardware=hw,
        execution_threads=1,openfoam_version='unit-test-fixture',configuration_sha256='config',
        trajectories=[dict(trajectory='case',runs=[dict(total_seconds=2.,systems=[dict(sys,rhs_digest='WRONG')])])])
    path=tmp_path/'gamg.json';write_json(path,ref)
    with pytest.raises(ValueError,match='mismatch'):mod.compare(tmp_path,path)
    ref['trajectories'][0]['runs'][0]['systems'][0]['rhs_digest']='b';write_json(path,ref)
    out=mod.compare(tmp_path,path)
    assert out['geometric_speedup']==2. and not out['full_cfd_wall_clock_measured']
