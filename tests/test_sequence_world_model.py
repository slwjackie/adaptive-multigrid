from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg import DiffusionCase,assemble_stiffness,MGConfig
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.strong import StrongRules,PreparedStrongMG
from adaptive_mg.v67.world_model import data as D
from adaptive_mg.v67.world_model.backend import SequenceBackend,ACTIONS,levels,hierarchy_signature
from adaptive_mg.v67.world_model.learning import fit_world,calibrate,Predictor,choose,save_model,load_model,observation
from adaptive_mg.v67.world_model import study as W

torch.set_num_threads(1)
PLAN='line_alt_energymin_full__em5__v11'

def make(index=0,n=7,angle=30.,contrast=3.,boundary='fixed',mesh=None):
    a=assemble_stiffness(DiffusionCase(n,epsilon=.3,angle_deg=angle,contrast=contrast,pattern='channel'))
    b=a@np.random.default_rng(index+128).normal(size=n*n)
    return D.snapshot(a,b,shape=(n,n),time=index*.01,index=index,mesh_id=mesh or f'grid-{n}',boundary_id=boundary)

def backend(expert=None,branch='H_P'):
    cfg=AdaptiveConfig(mg=MGConfig(mode='classical',strategy_name=PLAN,max_cycles=80,stencil_backend='csr'),
        mode='research',branch='H_P',spatial=False,gate_mode='open',transfer_levels='all',record_trace=False)
    r=StrongRules();r=r.replace_strategies({k:PLAN for k in r.rule_ids})
    return SequenceBackend(cfg,r,expert,expert_branch=branch)

def model():
    from adaptive_mg.v67.research_training import create_research_components
    from adaptive_mg.v67.research_transfer import make_graph_transfer
    m=create_research_components(smoother='student_cnn',transfer='small_gnn',smoother_hidden=4,transfer_hidden=4)
    m.transfer=make_graph_transfer(width=4,support='support_preserving',parameterization='affine',reference='frozen_parent',support_only=True,
        complexity_caps=dict(max_p_ratio=1.,max_ac_ratio=1.15,complexity_reference='parent',max_operator_complexity_ratio=1.15))
    return m


def test_reuse_updates_actual_galerkin_and_never_stale_numeric_factors():
    b=backend();s0=make(n=15);r0=b.solve(s0,None,'REBUILD_C');assert r0.success
    s1=make(1,n=15,angle=31.);r1=b.solve(s1,r0.bank,'REUSE_P');assert r1.success
    for old,new in zip(levels(r0.bank.root),levels(r1.bank.root)):
        assert old is not new
        if new.p is not None:
            np.testing.assert_array_equal(old.p.toarray(),new.p.toarray())
            np.testing.assert_allclose(new.coarse.a.toarray(),(new.p.T@new.a@new.p).toarray(),atol=1e-12)
            assert old.cache is not new.cache
        else:assert old.lu is not new.lu
    np.testing.assert_allclose(r1.bank.root.a.toarray(),s1.a.toarray())
    assert r1.bank.p_age==1


def test_exact_A_reuses_whole_bank_but_new_rhs_is_solved():
    b=backend();s=make();r=b.solve(s,None,'REBUILD_C')
    nxt=D.snapshot(s.a,s.b*2,shape=s.shape,time=.01,index=1,mesh_id=s.mesh_id,boundary_id=s.boundary_id)
    r2=b.solve(nxt,r.bank,'REUSE_P')
    assert r2.success and r.bank.root is r2.bank.root
    assert r2.stats['builds'][0]['exact_matrix_cache_hit']
    np.testing.assert_allclose(r2.x,2*r.x,rtol=1e-7,atol=1e-10)

@pytest.mark.parametrize('change',['boundary','mesh','shape','plan'])
def test_structural_or_plan_change_cannot_reuse(change):
    b=backend();r=b.solve(make(),None,'REBUILD_C')
    s=make(1,boundary='new' if change=='boundary' else 'fixed',mesh='new' if change=='mesh' else None,n=15 if change=='shape' else 7)
    if change=='plan':r.bank=replace(r.bank,plan='jacobi_bilinear_full')
    out=b.solve(s,r.bank,'REUSE_P')
    assert out.success and out.actual_action=='REBUILD_C' and out.fallback


def test_refresh_fine_preserves_deeper_P_and_parent_snapshot():
    b=backend();r=b.solve(make(n=15),None,'REBUILD_C');signature=hierarchy_signature(r.bank.root)
    new=b.solve(make(1,n=15,angle=40.,contrast=4.),r.bank,'REFRESH_FINE')
    assert new.success
    assert hierarchy_signature(r.bank.root)==signature
    for old,newl in zip(list(levels(r.bank.root))[1:],list(levels(new.bank.root))[1:]):
        if old.p is not None:np.testing.assert_array_equal(old.p.toarray(),newl.p.toarray())

@pytest.mark.parametrize('branch',['H_P','H_S','H_SP'])
def test_existing_neural_generators_are_real_numerical_paths(branch):
    b=backend(model(),branch);s=make(n=15);r=b.solve(s,None,'REBUILD_H');assert r.success
    if branch in ('H_P','H_SP'):assert r.stats['learned_transfer_apply_calls']>0
    if branch in ('H_S','H_SP'):assert r.stats['neural_apply_calls']>0
    rebuilt=b.solve(make(1,n=15,angle=32.),r.bank,'REUSE_P');assert rebuilt.success
    if branch in ('H_S','H_SP'):
        assert rebuilt.stats['builds'][0]['smoother_nn_calls']>0


def test_C_matches_existing_classical_solver():
    b=backend();s=make(n=15)
    got=b.solve(s,None,'REBUILD_C')
    ref=PreparedStrongMG(s.a,s.shape,None,replace(b.cfg,mode='classical',branch='C'),b.rules).solve(s.b,s.x0)
    assert got.cycles==ref.executed_cycles
    np.testing.assert_allclose(got.x,ref.x,rtol=0,atol=1e-12)


def test_bad_reused_trial_rolls_back_with_original_threshold(monkeypatch):
    import adaptive_mg.v67.world_model.backend as module
    b=backend();s=make();r=b.solve(s,None,'REBUILD_C');target=make(1)
    original=module.classical_cycle;calls=[0]
    def bad(root,x,rhs,cfg,st):
        calls[0]+=1
        return np.full_like(x,np.nan) if calls[0]==1 else original(root,x,rhs,cfg,st)
    monkeypatch.setattr(module,'classical_cycle',bad)
    got=b.solve(target,r.bank,'REUSE_P')
    assert got.success and got.fallback and got.actual_action=='REBUILD_C'
    assert got.threshold==max(b.cfg.mg.absolute_tolerance,b.cfg.mg.tolerance*np.linalg.norm(target.b))
    assert got.cycles<=b.cfg.mg.max_cycles and got.cycles>=len(got.residuals)

@pytest.mark.parametrize('bad',['nonsymmetric','indefinite','nullspace','nan'])
def test_foreign_invalid_systems_are_rejected(bad):
    a=sp.eye(49,format='lil')
    if bad=='nonsymmetric':a[0,1]=.1
    if bad=='indefinite':a[0,1]=a[1,0]=2.
    if bad=='nullspace':a[0,1]=a[1,0]=-1.
    if bad=='nan':a[0,0]=np.nan
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        D.snapshot(a,np.ones(49),shape=(7,7),time=0,index=0,mesh_id='a',boundary_id='a')


def test_sequence_files_integrity_and_group_order(tmp_path):
    D.generate(tmp_path/'data',counts=(1,1,1,1),steps=3,sizes=(7,))
    tr,ss=D.load_trajectories(tmp_path/'data','train')[0]
    assert ss[0].matrix_digest==ss[1].matrix_digest
    assert not ss[0].a.data.flags.writeable
    p=tmp_path/'data'/tr['snapshots'][0]['path'];p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='hash'):D.load_trajectories(tmp_path/'data','train')


def test_case_group_cannot_cross_splits(tmp_path):
    m=D.generate(tmp_path/'data',counts=(1,1,1,1),steps=2,sizes=(7,))
    m['trajectories'][1]['case_group']=m['trajectories'][0]['case_group'];D.write_json(tmp_path/'data/sequence_manifest.json',m)
    with pytest.raises(ValueError,match='crosses'):D.load_manifest(tmp_path/'data')
    with pytest.raises(ValueError,match='escapes'):D.inside(tmp_path,'../outside')


def episode(ident='x'):
    states=[make(0),make(1,angle=32.),make(2,angle=34.)];b=backend();settings=W.default_settings();settings['measurement_repeats']=1
    return W.collect_episode(dict(id=ident,case_group=ident,split='train'),states,b,settings,W.plans_for(b))


def test_action_conditioned_dynamics_training_and_roundtrip(tmp_path):
    ep=episode();artifact=fit_world([ep],ensemble=2,hidden=8,epochs=3)
    artifact.update(plans=W.plans_for(backend()),expert_signature=None,backend_contract=backend().contract)
    p=Predictor(artifact);times,success,rollout=p.predict(ep['obs'][0]);assert times.shape==(2,4)
    assert np.isfinite(times).all() and np.all(rollout>=times)
    assert not np.allclose(times[:,0],times[:,1])
    artifact['calibration']=calibrate(artifact,[ep]);save_model(tmp_path/'world.pt',artifact)
    other=Predictor(load_model(tmp_path/'world.pt'))
    np.testing.assert_array_equal(other.predict(ep['obs'][0])[0],times)
    assert len(artifact['training_loss'])==2


def test_world_gate_abstains_without_evidence_and_for_failure():
    p=(np.array([[.1,.01,.02,.02]]),np.ones((1,4)),np.array([[.2,.02,.04,.04]]))
    c=dict(margin=[0,0,0,0],episode_coverage=[3,3,3,3],failure_counts=[0,0,0,0])
    choice,_=choose(p,np.array([1,1,1,0],bool),c);assert choice==1
    c['failure_counts'][1]=1;assert choose(p,np.array([1,1,0,0],bool),c)[0]==0
    c['failure_counts'][1]=0;c['episode_coverage'][1]=0
    assert choose(p,np.array([1,1,0,0],bool),c)[0]==0


def test_online_rejects_time_reversal_and_does_not_require_future():
    b=backend();solver=W.WorldMGSolver(b,W.default_settings(),baseline='reuse')
    solver.step(make(0));solver.step(make(1));assert solver.t==2
    with pytest.raises(ValueError,match='order'):solver.step(make(0))
    solver.reset();solver.step(make(0));assert solver.t==1


def test_external_LDU_import_witness_permutation_and_boundary_contract(tmp_path):
    src=tmp_path/'input';src.mkdir();s=make();coo=sp.triu(s.a,k=1).tocoo();rng=np.random.default_rng(5)
    probes=rng.normal(size=(49,2));p=src/'a.npz'
    np.savez(p,diag=s.a.diagonal(),lower_addr=coo.row,upper_addr=coo.col,upper=coo.data,lower=coo.data,
             b=s.b,x0=s.x0,probe_vectors=probes,probe_products=s.a@probes,structured_to_native=np.arange(49))
    entries=[dict(path='a.npz',sha256=D.file_hash(p),boundary_finalized=True,nullspace='none',coupled_interfaces=0,
          layout='structured_2d_xmajor',shape=[7,7],time=i*.01,index=i,mesh_id='grid',boundary_id='anchored') for i in range(2)]
    m=dict(schema='finalized-ldu-sequence-v1',physics='user-declared reacting CFD',trajectories=[dict(id='case',case_group='case',split='train',snapshots=entries)])
    D.write_json(src/'ldu_sequence.json',m);D.import_finalized_ldu(src,tmp_path/'output')
    _,ss=D.load_trajectories(tmp_path/'output','train')[0];np.testing.assert_array_equal(ss[0].a.toarray(),s.a.toarray())
    assert not D.load_manifest(tmp_path/'output')['combustion_verified']
    m['trajectories'][0]['snapshots'][0]['boundary_finalized']=False;D.write_json(src/'ldu_sequence.json',m)
    with pytest.raises(ValueError,match='boundary'):D.import_finalized_ldu(src,tmp_path/'rejected')


def test_complete_sequence_workflow_leakage_free_and_single_test_use(tmp_path):
    settings=W.default_settings();settings.update(counts=[2,2,1,1],steps=3,sizes=[7],epochs=2,ensemble=2,hidden=8)
    cfg=tmp_path/'config.json';D.write_json(cfg,settings);out=tmp_path/'run'
    header=W.prepare(out,cfg);assert header['source_kind']=='synthetic_elliptic'
    with pytest.raises(FileNotFoundError):W.train(out)
    W.collect(out);W.collect(out,resume=True)
    artifact=W.train(out)
    assert 'reference_baseline' in artifact
    with pytest.raises(ValueError,match='closed'):W.collect(out,resume=True)
    with pytest.raises(FileNotFoundError):W.freeze(out)
    with pytest.raises(FileNotFoundError):W.evaluate(out,split='test',repeats=1)
    val=W.evaluate(out,repeats=1);assert not val['full_cfd_wall_clock_measured']
    assert val['summary']['classical_rebuild']['successes']==1
    with pytest.raises(ValueError):W.evaluate(out,repeats=1,resume=True)
    W.freeze(out);report=W.evaluate(out,split='test',repeats=1)
    assert report['source_kind']=='synthetic_elliptic'
    with pytest.raises(ValueError):W.evaluate(out,split='test',repeats=1,resume=True)


def test_bad_neural_setup_falls_back_current_C_not_old_matrix(monkeypatch):
    import adaptive_mg.v67.world_model.backend as module
    b=backend(model());s=make();old=b.solve(s,None,'REBUILD_C')
    monkeypatch.setattr(module,'prepare_transfer_bank',lambda *args:(_ for _ in ()).throw(ValueError('forced setup failure')))
    current=make(1,angle=35.);r=b.solve(current,old.bank,'REBUILD_H')
    assert r.success and r.fallback and r.actual_action=='REBUILD_C'
    np.testing.assert_array_equal(r.bank.root.a.toarray(),current.a.toarray())
    assert r.setup_seconds>0


def test_collection_counterfactuals_share_previous_bank_but_not_future_online_inputs():
    e=episode('counterfactual');assert np.asarray(e['targets']).shape==(3,4,4)
    assert not e['available'][0][1] and e['available'][1][1]
    assert np.any(np.asarray(e['next_obs'][0])[0]!=0)
    for step in e['measurements']:
        for records in step.values():
            for row in records:assert row['total_seconds']>=row['setup_seconds']


def test_recorder_explicit_external_contract_and_solution_mapping(tmp_path):
    from adaptive_mg.v67.world_model.adapter import FinalizedLduRecorder,solution_to_native
    s=make();coo=sp.triu(s.a,k=1).tocoo();probes=np.random.default_rng(10).normal(size=(49,2))
    recorder=FinalizedLduRecorder(tmp_path/'native',producer='unit-test fake finalized LDU',physics='synthetic test, NOT combustion')
    kw=dict(trajectory='c1',case_group='g1',split='train',shape=(7,7),mesh_id='grid',boundary_id='fixed',
        diag=s.a.diagonal(),lower_addr=coo.row,upper_addr=coo.col,lower=coo.data,upper=coo.data,b=s.b,x0=s.x0,
        probe_vectors=probes,probe_products=s.a@probes,structured_to_native=np.arange(49),
        boundary_finalized=True,nullspace='none',coupled_interfaces=0)
    for t in range(2):recorder.record(index=t,time=t*.1,**kw)
    D.import_finalized_ldu(tmp_path/'native',tmp_path/'imported')
    assert len(D.load_trajectories(tmp_path/'imported','train')[0][1])==2
    with pytest.raises(ValueError,match='nonmonotone'):recorder.record(index=1,time=.1,**kw)
    order=np.array([2,0,1]);np.testing.assert_array_equal(solution_to_native([10,20,30],order),[20,30,10])


def test_nonfinite_prediction_abstains():
    p=(np.full((2,4),np.nan),np.ones((2,4)),np.ones((2,4)))
    assert choose(p,np.ones(4,bool),{})[0]==0
