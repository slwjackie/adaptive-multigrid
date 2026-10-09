"""Sequence data -> real action rollouts -> world model -> frozen evaluation."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import torch
from ...config import MGConfig
from ...provenance import hardware_environment,json_safe
from ..config import AdaptiveConfig
from ..models import Components
from ..strong import StrongRules,load_strong_rules
from ..three_pillars import _source_digest
from . import VERSION
from .data import (generate,import_finalized_ldu,load_manifest,load_trajectories,
                   write_json,file_hash,digest)
from .backend import ACTIONS,SequenceBackend
from .learning import (FEATURES,feedback,observation,fit_world,calibrate,save_model,
                       load_model,Predictor,choose)


def read(path): return json.loads(Path(path).read_text())

def default_settings(smoke=True):
    return dict(version=VERSION,seed=20261010,steps=5 if smoke else 16,
                counts=[4,3,2,2] if smoke else [20,8,8,8],sizes=[7,15] if smoke else [15,31,63],
                torch_threads=1,ensemble=3,hidden=24,epochs=12 if smoke else 100,
                measurement_repeats=1 if smoke else 3,minimum_episodes=2,minimum_gain=.03,
                horizon=2,max_age=8,max_matrix_change=.5,max_size_extrapolation=4.,max_complexity=8.,
                expert_branch='H_P',smoke=smoke,
                solver=AdaptiveConfig(mg=MGConfig(mode='classical',strategy_name='line_alt_energymin_full__em5__v11',
                    max_cycles=150,nn_levels=1,stencil_backend='csr'),mode='research',branch='H_P',
                    use_smoother=False,spatial=False,gate_mode='open',use_learned_controller=False,
                    transfer_levels='all',record_trace=False).to_dict())


def prepare(output,config=None,*,source_run=None,expert_checkpoint=None,input_ldu=None):
    out=Path(output).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('new world-study directory required')
    settings=read(config) if config else default_settings()
    if settings.get('version')!=VERSION:raise ValueError('wrong world configuration version')
    torch.set_num_threads(int(settings['torch_threads']))
    if source_run:
        cfg=AdaptiveConfig.from_dict(read(Path(source_run)/'configuration.json')['solver'])
        rules=load_strong_rules(Path(source_run)/'selector_rules.json')
        baseline_scope='user-supplied frozen classical selector; not recalibrated on this sequence test'
        settings['solver']=cfg.to_dict()
    else:
        cfg=AdaptiveConfig.from_dict(settings['solver']);r=StrongRules()
        rules=r.replace_strategies({leaf:cfg.mg.strategy_name for leaf in r.rule_ids})
        baseline_scope='fixed EM plan for sequence smoke; not a calibrated strongest portfolio'
    out.mkdir(parents=True,exist_ok=True);write_json(out/'settings.json',settings);write_json(out/'rules.json',rules.to_dict())
    expert_hash=None
    if expert_checkpoint:
        expert=Components.load(expert_checkpoint)
        if expert.metadata.get('training_rules_digest')!=rules.digest():raise ValueError('expert trained under different classical rules')
        if expert.metadata.get('training_branch')!=settings['expert_branch']:raise ValueError('expert branch mismatch')
        if not expert.metadata.get('optimizer_updates',0)>0:raise ValueError('supply a genuinely trained expert')
        expert.save(out/'expert.pt');expert_hash=file_hash(out/'expert.pt')
    if input_ldu:import_finalized_ldu(input_ldu,out/'data')
    else:generate(out/'data',seed=settings['seed'],counts=settings['counts'],steps=settings['steps'],sizes=settings['sizes'])
    manifest=load_manifest(out/'data')
    header=dict(version=VERSION,source_digest=_source_digest(),settings_digest=digest(settings),rules_digest=rules.digest(),
        expert_sha256=expert_hash,data_sha256=file_hash(out/'data/sequence_manifest.json'),hardware=hardware_environment(refresh=True),
        baseline_scope=baseline_scope,source_kind=manifest['source_kind'],physics=manifest['physics'],
        online_cfd_coupled=False,combustion_validated=False)
    write_json(out/'run_manifest.json',header);return header


def load(output):
    out=Path(output).resolve();settings=read(out/'settings.json');header=read(out/'run_manifest.json')
    torch.set_num_threads(int(settings['torch_threads']))
    if header['source_digest']!=_source_digest() or header['settings_digest']!=digest(settings):raise ValueError('source/settings changed: use new run')
    if header['hardware']!=hardware_environment(refresh=True):raise ValueError('timing environment changed: use new run')
    if header['data_sha256']!=file_hash(out/'data/sequence_manifest.json'):raise ValueError('dataset manifest changed')
    rules=load_strong_rules(out/'rules.json')
    if rules.digest()!=header['rules_digest']:raise ValueError('classical rules changed')
    expert=None
    if header['expert_sha256']:
        if file_hash(out/'expert.pt')!=header['expert_sha256']:raise ValueError('expert changed: refit sequence model')
        expert=Components.load(out/'expert.pt')
    cfg=AdaptiveConfig.from_dict(settings['solver'])
    backend=SequenceBackend(cfg,rules,expert,expert_branch=settings['expert_branch'],max_complexity=settings['max_complexity'])
    return out,settings,header,backend


def plans_for(backend):
    return sorted(set(dict(backend.rules.strategy_by_rule).values())|{backend.rules.fallback_strategy_name})


def collect_episode(tr,states,backend,settings,plans):
    rng=np.random.default_rng(settings['seed']+int(digest(tr['id'])[:8],16));bank=None;previous=None;previous_result=None
    obs=[];targets=[];successes=[];availability=[];next_observations=[];behavior=[];feedbacks=[];raw_records=[]
    for t,s in enumerate(states):
        selection,cfg=backend.select(s);available=backend.available(s,bank,selection)
        obs.append(observation(s,bank,previous,previous_result,selection,cfg,plans).tolist())
        outputs={};records={}
        for a in rng.permutation(np.flatnonzero(available)):
            attempts=[backend.solve(s,bank,ACTIONS[a],selection=selection,cfg=cfg) for _ in range(settings['measurement_repeats'])]
            results=np.array([r.total_seconds for r in attempts]);idx=int(np.argsort(results)[len(results)//2])
            outputs[int(a)]=attempts[idx];records[int(a)]=[r.record() for r in attempts]
        ys=np.zeros((4,4));ok=np.zeros(4);nxt=np.zeros((4,len(obs[-1])))
        for a,r in outputs.items():
            # Repetition medians for cost labels; no repetition counted as an
            # independent trajectory. Same numerical result is deterministic.
            measured=records[a]
            ys[a]=feedback(r)[:4]
            ys[a,0]=np.log(max(float(np.median([z['setup_seconds'] for z in measured])),1e-8))
            ys[a,1]=np.log(max(float(np.median([z['solve_seconds'] for z in measured])),1e-8))
            ok[a]=all(z['success'] for z in measured)
            if t+1<len(states):
                ns,ncfg=backend.select(states[t+1])
                nxt[a]=observation(states[t+1],r.bank,s,r,ns,ncfg,plans)
        # Random behavior avoids collecting only always-rebuilt hierarchies.
        good=[a for a,r in outputs.items() if r.success]
        a=int(rng.choice(good if good else list(outputs)))
        chosen=outputs[a];actual=ACTIONS.index(chosen.actual_action)
        targets.append(ys.tolist());successes.append(ok.tolist());availability.append(available.tolist())
        next_observations.append(nxt.tolist());behavior.append(actual);feedbacks.append(feedback(chosen).tolist())
        raw_records.append({ACTIONS[k]:v for k,v in records.items()})
        bank=chosen.bank;previous=s;previous_result=chosen
    return dict(trajectory=tr['id'],case_group=tr['case_group'],split=tr['split'],obs=obs,targets=targets,
                success=successes,available=availability,next_obs=next_observations,
                behavior=behavior,feedback=feedbacks,measurements=raw_records)


def collect(output,*,resume=False):
    out,settings,header,backend=load(output)
    if (out/'world.pt').exists() or (out/'freeze.json').exists():raise ValueError('collection closed after model fitting')
    plans=plans_for(backend);result={};contract=digest(header)
    for split in ('train','tune'):
        result[split]=[]
        for tr,states in load_trajectories(out/'data',split):
            target=out/'transitions'/split/(tr['id']+'.json')
            if target.exists():
                if not resume:raise FileExistsError('use --resume for existing episodes')
                entry=read(target)
                if entry['contract']!=contract:raise ValueError('stale transition labels')
                episode=entry['episode']
            else:
                episode=collect_episode(tr,states,backend,settings,plans)
                write_json(target,dict(contract=contract,episode=episode))
            result[split].append(episode);print('[collect]',split,tr['id'],len(states),'systems',flush=True)
    write_json(out/'collection.json',dict(contract=contract,plans=plans,
        inputs={str(p.relative_to(out)):file_hash(p) for p in sorted((out/'transitions').rglob('*.json'))},
        source_kind=header['source_kind'],actual_action_timings=True,test_seen=False))
    return result


def heuristic(name,obs,available,t):
    action=0
    if name=='reuse':action=1
    elif name.startswith('periodic_'):action=0 if t%int(name.split('_')[1])==0 else 1
    elif name.startswith('drift_'):action=0 if obs[9]>np.log1p(float(name.split('_')[1])) else 1
    elif name=='neural_rebuild':action=3
    elif name!='classical_rebuild':raise ValueError('unknown baseline')
    return action if available[action] else 0


class WorldMGSolver:
    """Public sequential solve interface, suitable for a verified in-process adapter.

    Pressure matrix snapshots only; no call advances chemistry/Navier-Stokes.
    A stream must call reset at independent case boundaries.
    """
    def __init__(self,backend,settings,artifact=None,baseline='world'):
        self.backend=backend;self.settings=settings;self.artifact=artifact;self.baseline=baseline
        self.plans=artifact['plans'] if artifact is not None else plans_for(backend)
        if baseline=='world' and artifact is None:raise ValueError('trained world artifact required')
        if artifact is not None and artifact['backend_contract']!=backend.contract:raise ValueError('world/backend contract mismatch')
        self.predictor=Predictor(artifact) if artifact is not None else None;self.reset()

    def reset(self):
        self.bank=None;self.previous=None;self.result=None;self.t=0
        if self.predictor:self.predictor.reset()

    def step(self,s):
        start=perf_counter()
        if self.previous is not None and (s.index<=self.previous.index or s.time<self.previous.time):
            raise ValueError('out-of-order snapshot; reset between independent trajectories')
        selection,cfg=self.backend.select(s);available=self.backend.available(s,self.bank,selection)
        obs=observation(s,self.bank,self.previous,self.result,selection,cfg,self.plans)
        reason='baseline';detail={};action=0;policy_start=perf_counter()
        if self.baseline!='world':action=heuristic(self.baseline,obs,available,self.t)
        else:
            incompatible=self.previous is not None and not available[1]
            if incompatible:self.predictor.reset()
            predictions=self.predictor.predict(obs,horizon=self.settings['horizon'])
            trained=self.artifact['N_range'];N=s.a.shape[0]
            scale=max(N/trained[1],trained[0]/N,1.)
            if s.source_kind!=self.artifact['source_kind']:reason='untrained_data_domain'
            elif self.bank is None:reason='first_system_rebuild'
            elif incompatible:reason='structural_or_plan_change'
            elif self.bank.p_age>=self.settings['max_age']:reason='hierarchy_age_limit'
            elif obs[9]>np.log1p(self.settings['max_matrix_change']):reason='large_matrix_change'
            elif selection.strategy_name not in self.plans:reason='unseen_plan'
            elif scale>self.settings['max_size_extrapolation']:reason='size_outside_support'
            else:
                action,detail=choose(predictions,available,self.artifact['calibration'],
                    minimum_gain=self.settings['minimum_gain'],minimum_episodes=self.settings['minimum_episodes'])
                reason='world_plan' if action else 'empirical_abstention'
        policy_seconds=perf_counter()-policy_start
        result=self.backend.solve(s,self.bank,ACTIONS[action],selection=selection,cfg=cfg)
        if self.predictor:self.predictor.update(ACTIONS.index(result.actual_action),result)
        self.bank=result.bank;self.previous=s;self.result=result;self.t+=1
        end=perf_counter();record=result.record()
        # Timer includes feature extraction, hashing, selection, world inference,
        # numeric refresh, failed attempts and classical recovery. Disk I/O and
        # the diagnostic serialization below are excluded for all baselines.
        record.update(total_seconds=end-start,policy_seconds=policy_seconds,
           orchestration_seconds=max(0.,end-start-result.total_seconds),decision_reason=reason,
           decision=detail,step=s.index,time=s.time,source_kind=s.source_kind,
           N=s.a.shape[0],selected_plan=selection.strategy_name,
           classical_coverage_fallback=selection.rule_evidence.get('fallback_for_coverage',False),
           scope='matrix-sequence solver, not full CFD wall-clock')
        return result.x,record


def run_episode(states,backend,settings,*,artifact=None,baseline='classical_rebuild'):
    solver=WorldMGSolver(backend,settings,artifact,baseline);rows=[]
    for s in states:
        _,row=solver.step(s);rows.append(row)
    return dict(rows=rows,success=all(r['success'] for r in rows),
          total_seconds=sum(r['total_seconds'] for r in rows),
          setup_seconds=sum(r['setup_seconds'] for r in rows),cycles=sum(r['cycles'] for r in rows),
          fallback_count=sum(r['fallback'] for r in rows),
          actions={a:sum(r['actual_action']==a for r in rows) for a in ACTIONS})


def train(output):
    out,settings,header,backend=load(output)
    if (out/'world.pt').exists():raise FileExistsError('model already fitted; use new run for changes')
    collection=read(out/'collection.json')
    if collection['contract']!=digest(header):raise ValueError('collection contract changed')
    for p,h in collection['inputs'].items():
        if file_hash(out/p)!=h:raise ValueError('training labels changed')
    episodes={s:[read(p)['episode'] for p in sorted((out/'transitions'/s).glob('*.json'))] for s in ('train','tune')}
    if set(e['case_group'] for e in episodes['train'])&set(e['case_group'] for e in episodes['tune']):raise ValueError('group leakage')
    artifact=fit_world(episodes['train'],ensemble=settings['ensemble'],hidden=settings['hidden'],epochs=settings['epochs'],seed=settings['seed'])
    artifact.update(version=VERSION,plans=collection['plans'],backend_contract=backend.contract,source_kind=header['source_kind'],
                    expert_signature=backend.expert_signature,training_groups=[e['case_group'] for e in episodes['train']],
                    N_range=[int(round(np.exp(min(v[0] for e in episodes['train'] for v in e['obs'])))),
                             int(round(np.exp(max(v[0] for e in episodes['train'] for v in e['obs']))))])
    artifact['calibration']=calibrate(artifact,episodes['tune'])
    # Practical non-neural reuse comparator is selected ONLY on tune trajectories.
    candidates=['classical_rebuild','reuse','periodic_2','periodic_4','drift_0.01','drift_0.05','drift_0.2']
    scores={};tune=load_trajectories(out/'data','tune')
    for name in candidates:
        rows=[run_episode(states,backend,settings,baseline=name) for _,states in tune]
        scores[name]=dict(total_seconds=sum(r['total_seconds'] for r in rows),successes=sum(r['success'] for r in rows),
                          trajectory_success=[r['success'] for r in rows])
    required=scores['classical_rebuild']['trajectory_success']
    valid=[n for n in candidates if all(not c or h for c,h in zip(required,scores[n]['trajectory_success']))]
    artifact['reference_baseline']=min(valid,key=lambda n:(-scores[n]['successes'],scores[n]['total_seconds']))
    artifact['tune_baselines']=scores
    artifact['run_contract']=digest(header);artifact['collection_sha256']=file_hash(out/'collection.json')
    save_model(out/'world.pt',artifact)
    write_json(out/'training.json',dict(training_loss=artifact['training_loss'],calibration=artifact['calibration'],
        reference_baseline=artifact['reference_baseline'],tune_baselines=scores,
        trained_parameters=sum(p.numel() for m in artifact['models'] for p in m.parameters()),
        source_kind=header['source_kind'],trained_neural_mg_expert=bool(backend.expert),test_seen=False,
        evidence='training execution is not performance validation',performance_certified=False))
    return artifact


def evaluate(output,*,split='validation',repeats=3,resume=False):
    out,settings,header,backend=load(output)
    if split not in ('validation','test') or repeats<1:raise ValueError('invalid evaluation protocol')
    artifact=load_model(out/'world.pt')
    if artifact['run_contract']!=digest(header):raise ValueError('world model trained in different run')
    if artifact['collection_sha256']!=file_hash(out/'collection.json'):raise ValueError('training collection changed')
    model_hash=file_hash(out/'world.pt');protocol=dict(split=split,repeats=repeats,model=model_hash,run=digest(header))
    target=out/split;target.mkdir(exist_ok=True)
    if split=='test':
        frozen=read(out/'freeze.json')
        if frozen['model']!=model_hash or frozen['run']!=digest(header):raise ValueError('frozen model/run changed')
        if file_hash(out/'validation/report.json')!=frozen['validation']:raise ValueError('validation changed after freeze')
    progress=target/'progress.json'
    if progress.exists():
        old=read(progress)
        if not resume or old['protocol']!=protocol or old['complete']:raise ValueError('complete/changed evaluation cannot be reopened')
    elif resume:raise ValueError('no evaluation to resume')
    write_json(progress,dict(protocol=protocol,complete=False))
    names=list(dict.fromkeys(['classical_rebuild',artifact['reference_baseline'],'reuse','world']))
    if backend.expert:names.insert(-1,'neural_rebuild')
    rows=[];rng=np.random.default_rng(settings['seed']+2901)
    for tr,states in load_trajectories(out/'data',split):
        per={name:[] for name in names}
        for rep in range(repeats):
            for name in rng.permutation(names):
                path=target/f'{tr["id"]}_{name}_{rep}.json'
                if path.exists():
                    if not resume:raise FileExistsError(path)
                    entry=read(path)
                    if entry['protocol']!=protocol:raise ValueError('stale evaluation row')
                    result=entry['result']
                else:
                    result=run_episode(states,backend,settings,artifact=artifact if name=='world' else None,baseline=name)
                    write_json(path,dict(protocol=protocol,result=result))
                per[name].append(result)
        rows.append(dict(trajectory=tr['id'],case_group=tr['case_group'],runs=per))
        print('[evaluate]',split,tr['id'],flush=True)
    summary={};ref=artifact['reference_baseline']
    if ref not in names:raise AssertionError('tune reference missing')
    for name in names:
        ratios=[];success=0;new=[];actions={a:0 for a in ACTIONS};total=0.
        for row in rows:
            rr=row['runs'][ref];nr=row['runs'][name];good=all(r['success'] for r in nr);base_ok=all(r['success'] for r in rr)
            success+=good;total+=float(np.median([r['total_seconds'] for r in nr]))
            if base_ok and not good:new.append(row['trajectory'])
            if base_ok and good:ratios.append(float(np.median([r['total_seconds'] for r in rr])/np.median([r['total_seconds'] for r in nr])))
            for a in ACTIONS:actions[a]+=nr[0]['actions'][a]
        if ratios:
            logs=np.log(ratios);boot=np.exp(logs[rng.integers(0,len(logs),size=(2000,len(logs)))].mean(1))
            gm=float(np.exp(logs.mean()));ci=np.quantile(boot,[.025,.975]).tolist()
        else:gm=None;ci=None
        summary[name]=dict(successes=success,total=len(rows),speedup_vs_tune_selected_baseline=gm,
            ci95=ci,new_failure_trajectories=new,first_repeat_action_counts=actions,all_case_wall_seconds=total)
    report=dict(protocol=protocol,reference_baseline=ref,summary=summary,rows=rows,
        source_kind=header['source_kind'],physics=header['physics'],trained_neural_mg_expert=bool(backend.expert),
        confidence_scope='bootstrap independent trajectory median ratios; not repetitions or timesteps',
        wall_scope='all sequential validation/features/policy/setup/solve/recovery, excludes input disk I/O',
        full_cfd_wall_clock_measured=False,combustion_validated=False,performance_certified=False)
    write_json(target/'report.json',report);write_json(progress,dict(protocol=protocol,complete=True))
    return report


def freeze(output):
    out,settings,header,backend=load(output)
    if (out/'freeze.json').exists():raise FileExistsError('already frozen')
    if not read(out/'validation/progress.json')['complete']:raise ValueError('independent sequence validation required')
    report=read(out/'validation/report.json');model=file_hash(out/'world.pt')
    if report['protocol']['model']!=model:raise ValueError('model changed after validation')
    write_json(out/'freeze.json',dict(run=digest(header),model=model,validation=file_hash(out/'validation/report.json'),
              source_kind=header['source_kind'],test_seen=False,performance_certified=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','collect','train','evaluate','freeze'):
        p=sub.add_parser(name);p.add_argument('--run-dir',required=True)
        if name=='prepare':
            p.add_argument('--config',default='configs/v6_7_world_model_smoke.json');p.add_argument('--source-run')
            p.add_argument('--expert-checkpoint');p.add_argument('--input-ldu')
        if name in ('collect','evaluate'):p.add_argument('--resume',action='store_true')
        if name=='evaluate':p.add_argument('--split',choices=('validation','test'),default='validation');p.add_argument('--repeats',type=int,default=3)
    a=parser.parse_args(argv)
    if a.command=='prepare':result=prepare(a.run_dir,a.config,source_run=a.source_run,expert_checkpoint=a.expert_checkpoint,input_ldu=a.input_ldu)
    elif a.command=='collect':collect(a.run_dir,resume=a.resume);return
    elif a.command=='train':train(a.run_dir);print('World model fitted. Independent sequence validation is still required.');return
    elif a.command=='freeze':freeze(a.run_dir);print('Frozen. Synthetic test is NOT hydrogen combustion validation.');return
    else:result=evaluate(a.run_dir,split=a.split,repeats=a.repeats,resume=a.resume)
    print(json.dumps(json_safe(result.get('summary',result)),indent=2))
