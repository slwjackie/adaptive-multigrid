"""Time-varying H_S-only study; no hydrogen flow solver is implemented here."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np

from ..world_model.data import (generate, import_finalized_ldu, load_manifest, load_trajectories,
                                file_hash, write_json, digest)
from ..research_data import _digests
from .tuning import read, assert_open, VERSION as THESIS_VERSION
from .backend import HSSmoothingBackend, ACTIONS
from .learning import (VERSION, observation, feedback, fit_world, Predictor, choose,
                       calibrate, save_model, load_model)


def prepare(output,*,input_ldu=None):
    from .study import selected
    out,settings,cfg,rules,expert,selection=selected(output);assert_open(out)
    target=out/'temporal'
    if target.exists() and any(target.iterdir()):raise FileExistsError('temporal data already prepared')
    w=dict(settings['world']);target.mkdir(parents=True,exist_ok=True)
    if input_ldu:import_finalized_ldu(input_ldu,target/'data')
    else:generate(target/'data',seed=w['seed'],counts=w['counts'],steps=w['steps'],sizes=w['sizes'])
    m=load_manifest(target/'data');groups=[t['case_group'] for t in m['trajectories']]
    # Whole independent cases are the statistical unit. No slicing one CFD run
    # into apparently independent trajectories. Merge consecutive chunks first.
    if len(groups)!=len(set(groups)):raise ValueError('one complete trajectory per independent physical case required')
    excluded=_digests(read(out/'calibration_manifest.json'))|_digests(read(out/'static_data_manifest.json'))
    temporal={v['normalized_matrix_digest'] for t in m['trajectories'] for v in t['snapshots']}
    if temporal&excluded:raise ValueError('sequence data overlaps classical/HS development operators')
    header=dict(version=THESIS_VERSION,action_version=VERSION,actions=list(ACTIONS),
        selection_sha256=file_hash(out/'expert_selection.json'),data_sha256=file_hash(target/'data/sequence_manifest.json'),
        settings=w,source_kind=m['source_kind'],physics=m['physics'],
        primary_classical_plan=selection['solver']['mg']['strategy_name'],
        learned_transfer=False,online_cfd_coupled=False,combustion_validated=False)
    write_json(target/'run_manifest.json',header)
    return header


def load(output):
    from .study import selected
    out,settings,cfg,rules,expert,selection=selected(output)
    h=read(out/'temporal/run_manifest.json')
    if h['version']!=THESIS_VERSION or h['actions']!=list(ACTIONS) or h['action_version']!=VERSION:
        raise ValueError('temporal action contract changed')
    if h['selection_sha256']!=file_hash(out/'expert_selection.json') or h['settings']!=settings['world']:
        raise ValueError('selected H_S/world settings changed')
    if h['data_sha256']!=file_hash(out/'temporal/data/sequence_manifest.json'):
        raise ValueError('sequence manifest changed')
    backend=HSSmoothingBackend(cfg,rules,expert,max_complexity=settings['world']['max_complexity'])
    return out,settings['world'],h,backend


def heuristic(name,obs,t):
    if name=='rebuild':return 0
    if name=='reuse':return 1
    if name.startswith('periodic_'):
        n=int(name.split('_')[1])
        if n<1:raise ValueError('positive rebuild period required')
        return int(t%n!=0)
    if name.startswith('drift_'):
        bound=float(name.split('_')[1])
        if not np.isfinite(bound) or bound<0:raise ValueError('invalid drift threshold')
        return int(obs[9]<=np.log1p(bound))
    raise ValueError('unknown predeclared reuse heuristic')


def available_now(s,bank,selection,obs,backend,settings,allow_smoother):
    available=backend.available(s,bank,selection).copy()
    if bank is not None and (bank.p_age>=settings['max_age'] or obs[9]>np.log1p(settings['max_matrix_change'])):
        available[[1,3]]=False
    if not allow_smoother:available[2:]=False
    return available


def collect_episode(tr,states,backend,w,allow_smoother):
    plans=[backend.cfg.mg.strategy_name];rng=np.random.default_rng(w['seed']+int(digest(tr['id'])[:8],16)+int(allow_smoother))
    bank=None;previous=None;last=None
    data={key:[] for key in ('obs','targets','success','available','next_obs','behavior','feedback','measurements')}
    for t,s in enumerate(states):
        sel,cfg=backend.select(s);obs=observation(s,bank,previous,last,sel,cfg,plans)
        available=available_now(s,bank,sel,obs,backend,w,allow_smoother)
        outputs={};records={}
        for a in rng.permutation(np.flatnonzero(available)):
            trials=[backend.solve(s,bank,ACTIONS[a],selection=sel,cfg=cfg) for _ in range(w['measurement_repeats'])]
            chosen=trials[int(np.argsort([v.total_seconds for v in trials])[len(trials)//2])]
            outputs[int(a)]=chosen;records[int(a)]=[v.record() for v in trials]
        targets=np.zeros((4,4));success=np.zeros(4);nextobs=np.zeros((4,len(obs)))
        for a,r in outputs.items():
            targets[a]=feedback(r)[:4]
            targets[a,0]=np.log(max(np.median([v['setup_seconds'] for v in records[a]]),1e-8))
            targets[a,1]=np.log(max(np.median([v['solve_seconds'] for v in records[a]]),1e-8))
            success[a]=all(v['success'] for v in records[a])
            if t+1<len(states):
                nxt,ncfg=backend.select(states[t+1])
                nextobs[a]=observation(states[t+1],r.bank,s,r,nxt,ncfg,plans)
        good=[a for a in outputs if outputs[a].success]
        a=int(rng.choice(good or list(outputs)));chosen=outputs[a];actual=ACTIONS.index(chosen.actual_action)
        for key,value in dict(obs=obs.tolist(),targets=targets.tolist(),success=success.tolist(),
                available=available.tolist(),next_obs=nextobs.tolist(),behavior=actual,
                feedback=feedback(chosen).tolist(),measurements={ACTIONS[a]:v for a,v in records.items()}).items():
            data[key].append(value)
        bank=chosen.bank;previous=s;last=chosen
    return dict(data,trajectory=tr['id'],case_group=tr['case_group'],split=tr['split'],
                allow_smoother=allow_smoother,actions=list(ACTIONS))


def collect(output,*,resume=False):
    out,w,h,backend=load(output);assert_open(out);target=out/'temporal'
    if any((target/(m+'.pt')).exists() for m in ('world_C','world_HS')):
        raise ValueError('labels are closed after world-model fitting')
    contract=digest(h);inputs={}
    for mode in ('world_C','world_HS'):
        for split in ('train','tune'):
            for tr,states in load_trajectories(target/'data',split):
                path=target/'transitions'/mode/split/(tr['id']+'.json')
                if path.exists():
                    entry=read(path)
                    if not resume or entry['contract']!=contract or digest(entry['episode'])!=entry['episode_digest']:
                        raise ValueError('stale labels or missing --resume')
                else:
                    ep=collect_episode(tr,states,backend,w,mode=='world_HS')
                    write_json(path,dict(contract=contract,episode=ep,episode_digest=digest(ep)))
                inputs[str(path.relative_to(target))]=file_hash(path)
                print('[collect]',mode,split,tr['id'],flush=True)
    write_json(target/'collection.json',dict(contract=contract,inputs=inputs,actual_action_timings=True))


class WorldSolver:
    """Public step(snapshot) API; no future snapshots accepted at decision time."""
    def __init__(self,backend,settings,*,artifact=None,heuristic_name='rebuild',neural=False,horizon=None):
        self.backend=backend;self.settings=settings;self.artifact=artifact
        self.heuristic_name=heuristic_name;self.neural=neural
        self.plans=artifact['plans'] if artifact else [backend.cfg.mg.strategy_name]
        self.horizon=horizon or settings['horizon']
        if artifact and artifact['backend_contract']!=backend.contract:raise ValueError('world/backend contract mismatch')
        self.predictor=Predictor(artifact) if artifact else None;self.reset()

    def reset(self):
        self.bank=None;self.previous=None;self.last=None;self.t=0
        if self.predictor:self.predictor.reset()

    def step(self,s):
        start=perf_counter()
        if self.previous is not None and (s.index<=self.previous.index or s.time<self.previous.time):
            raise ValueError('out-of-order snapshot; reset between independent cases')
        selection,cfg=self.backend.select(s)
        if self.predictor:
            obs=observation(s,self.bank,self.previous,self.last,selection,cfg,self.plans)
        else:
            # A non-neural heuristic must not pay for NN-only feature scans.
            obs=np.zeros(28,np.float32)
            if self.heuristic_name!='rebuild' and self.previous is not None and self.previous.a.shape==s.a.shape:
                obs[9]=np.log1p(np.linalg.norm((s.a-self.previous.a).data)/max(np.linalg.norm(self.previous.a.data),1e-100))
        enabled=self.artifact['allow_smoother'] if self.artifact else self.neural
        available=available_now(s,self.bank,selection,obs,self.backend,self.settings,enabled)
        ref=heuristic(self.heuristic_name,obs,self.t);ref=ref if available[ref] else 0
        policy_start=perf_counter();detail={};reason='non_neural_heuristic'
        if not self.predictor:
            action=ref+2 if self.neural else ref
            if not available[action]:action=ref
        else:
            incompatible=self.bank is not None and not self.backend.compatible(s,self.bank,selection.strategy_name)
            if incompatible:self.predictor.reset()
            predictions=self.predictor.predict(obs,self.horizon)
            lo,hi=self.artifact['N_range'];N=s.a.shape[0];scale=max(N/hi,lo/N,1.)
            action=ref
            if s.source_kind!=self.artifact['source_kind']:reason='untrained_source_domain'
            elif incompatible:reason='incompatible_hierarchy'
            elif scale>self.settings['max_size_extrapolation']:reason='size_outside_support'
            else:
                action,detail=choose(predictions,available,self.artifact['calibration'],ref,
                    minimum_gain=self.settings['minimum_gain'],minimum_episodes=self.settings['minimum_episodes'])
                reason='world_choice' if action!=ref else 'tuned_classical_abstention'
        result=self.backend.solve(s,self.bank,ACTIONS[action],selection=selection,cfg=cfg)
        if self.predictor:self.predictor.update(ACTIONS.index(result.actual_action),result)
        self.bank=result.bank;self.previous=s;self.last=result;self.t+=1
        end=perf_counter();row=result.record()
        # Actual solver data for optional external GAMG matching.
        row.update(total_seconds=end-start,policy_and_orchestration_seconds=max(0.,end-start-result.total_seconds),
            decision_reason=reason,decision=detail,step=s.index,time=s.time,source_kind=s.source_kind,
            matrix_digest=s.matrix_digest,rhs_digest=digest(s.b.tolist()),x0_digest=digest(s.x0.tolist()),
            selected_plan=selection.strategy_name,learned_transfer=False,
            actual_neural_used=result.stats.get('neural_trial_cycles',0)>0,
            accepted_neural_cycles=result.stats.get('accepted_neural_cycles',0),
            scope='sequential linear systems, includes setup/decision/recovery, not full CFD')
        return result.x,row


def run_episode(states,backend,w,*,artifact=None,heuristic_name='rebuild',neural=False,horizon=None):
    solver=WorldSolver(backend,w,artifact=artifact,heuristic_name=heuristic_name,neural=neural,horizon=horizon)
    rows=[solver.step(s)[1] for s in states]
    return dict(rows=rows,success=all(r['success'] for r in rows),total_seconds=sum(r['total_seconds'] for r in rows),
        setup_seconds=sum(r['setup_seconds'] for r in rows),cycles=sum(r['cycles'] for r in rows),
        neural_systems=sum(r['actual_neural_used'] for r in rows),accepted_neural_cycles=sum(r['accepted_neural_cycles'] for r in rows),
        fallbacks=sum(r['fallback'] for r in rows),actions={a:sum(r['actual_action']==a for r in rows) for a in ACTIONS})


def select_heuristics(tune,backend,w):
    rng=np.random.default_rng(w['seed']+419);scores={};names=w['heuristics']
    if 'rebuild' not in names or 'reuse' not in names:raise ValueError('must retain rebuild and always-reuse baselines')
    for neural in (False,True):
        observations={n:[] for n in names}
        for tr,states in tune:
            runs={n:[] for n in names}
            for _ in range(w['baseline_repeats']):
                for name in rng.permutation(names):runs[name].append(run_episode(states,backend,w,heuristic_name=name,neural=neural))
            for name in names:
                observations[name].append(dict(trajectory=tr['id'],success=all(r['success'] for r in runs[name]),
                     seconds=float(np.median([r['total_seconds'] for r in runs[name]]))))
        anchor=observations['rebuild']
        eligible=[n for n in names if all(not c['success'] or h['success'] for c,h in zip(anchor,observations[n]))]
        winner=min(eligible,key=lambda n:(-sum(r['success'] for r in observations[n]),sum(r['seconds'] for r in observations[n])))
        scores['H_S' if neural else 'classical']=dict(selected=winner,rows=observations)
    return scores


def train(output):
    out,w,h,backend=load(output);assert_open(out);target=out/'temporal'
    if any((target/(m+'.pt')).exists() for m in ('world_C','world_HS')):raise FileExistsError('world already trained; new run required')
    collection=read(target/'collection.json')
    if collection['contract']!=digest(h):raise ValueError('changed collection')
    for p,sha in collection['inputs'].items():
        if file_hash(target/p)!=sha:raise ValueError('training labels changed')
    baselines=select_heuristics(load_trajectories(target/'data','tune'),backend,w)
    for mode in ('world_C','world_HS'):
        eps={s:[read(p)['episode'] for p in sorted((target/'transitions'/mode/s).glob('*.json'))] for s in ('train','tune')}
        if set(e['case_group'] for e in eps['train'])&set(e['case_group'] for e in eps['tune']):raise ValueError('case leakage')
        if any(e['allow_smoother']!=(mode=='world_HS') for s in eps for e in eps[s]):raise ValueError('mixed action dataset')
        a=fit_world(eps['train'],ensemble=w['ensemble'],hidden=w['hidden'],epochs=w['epochs'],seed=w['seed'])
        N=[v[0] for ep in eps['train'] for v in ep['obs']]
        a.update(version=VERSION,actions=list(ACTIONS),allow_smoother=mode=='world_HS',
            plans=[backend.cfg.mg.strategy_name],backend_contract=backend.contract,expert_signature=backend.expert_signature,
            source_kind=h['source_kind'],N_range=[int(round(np.exp(min(N)))),int(round(np.exp(max(N))))],
            run_contract=digest(h),collection_sha256=file_hash(target/'collection.json'),
            reference_heuristic=baselines['classical']['selected'])
        a['calibration']=calibrate(a,eps['tune']);save_model(target/(mode+'.pt'),a)
    write_json(target/'training.json',dict(baselines=baselines,world_C_sha256=file_hash(target/'world_C.pt'),
        world_HS_sha256=file_hash(target/'world_HS.pt'),source_kind=h['source_kind'],trained_H_S=True,
        independent_performance_validated=False,learned_transfer=False))
    return baselines


def _artifacts(target,h):
    status=read(target/'training.json');models={}
    for mode in ('world_C','world_HS'):
        path=target/(mode+'.pt')
        if file_hash(path)!=status[mode+'_sha256']:raise ValueError('world checkpoint changed')
        a=load_model(path)
        if a['run_contract']!=digest(h) or a['collection_sha256']!=file_hash(target/'collection.json'):
            raise ValueError('world data/expert contract changed')
        models[mode]=a
    return status,models


def summarize(rows,reference,seed):
    rng=np.random.default_rng(seed);names=list(rows[0]['runs']);summary={}
    for name in names:
        success=0;ratios=[];new=[];total=0.;actions={a:0 for a in ACTIONS};neural=0
        for row in rows:
            rr=row['runs'][reference];cur=row['runs'][name]
            ok=all(r['success'] for r in cur);ref_ok=all(r['success'] for r in rr);success+=ok
            t=float(np.median([r['total_seconds'] for r in cur]));total+=t
            if ref_ok and ok:ratios.append(float(np.median([r['total_seconds'] for r in rr]))/t)
            if ref_ok and not ok:new.append(row['case_group'])
            neural+=cur[0]['neural_systems']
            for a in actions:actions[a]+=cur[0]['actions'][a]
        if ratios:
            logs=np.log(ratios);gm=float(np.exp(logs.mean()))
            samples=np.exp(logs[rng.integers(0,len(logs),(2000,len(logs)))].mean(1))
            ci=np.quantile(samples,[.025,.975]).tolist()
        else:gm=None;ci=None
        summary[name]=dict(successes=success,total=len(rows),geometric_speedup=gm,ci95=ci,
            new_failure_case_groups=new,all_case_seconds=total,first_repeat_actions=actions,neural_systems=neural)
    return summary


def evaluate(output,*,split='validation',repeats=3,resume=False):
    out,w,h,backend=load(output)
    if split not in ('validation','test') or repeats<1:raise ValueError('invalid evaluation')
    if split=='validation':assert_open(out)
    else:check_freeze(out)
    target=out/'temporal';training,models=_artifacts(target,h)
    c=training['baselines']['classical']['selected'];hs=training['baselines']['H_S']['selected']
    protocol=dict(split=split,repeats=repeats,run=digest(h),training=file_hash(target/'training.json'))
    dest=target/split;dest.mkdir(exist_ok=True);progress=dest/'progress.json'
    if progress.exists():
        old=read(progress)
        if not resume or old['protocol']!=protocol or old['complete']:raise ValueError('completed/changed evaluation cannot reopen')
    elif resume:raise FileNotFoundError('no evaluation to resume')
    write_json(progress,dict(protocol=protocol,complete=False))
    specs={'C_rebuild':dict(heuristic_name='rebuild'), 'C_tuned_reuse':dict(heuristic_name=c),
           'HS_rebuild':dict(heuristic_name='rebuild',neural=True),
           'HS_matched_reuse':dict(heuristic_name=c,neural=True),
           'HS_tuned_reuse':dict(heuristic_name=hs,neural=True),
           'World_C':dict(heuristic_name=c,artifact=models['world_C']),
           'World_HS':dict(heuristic_name=c,artifact=models['world_HS']),
           'World_HS_horizon1':dict(heuristic_name=c,artifact=models['world_HS'],horizon=1)}
    rows=[];rng=np.random.default_rng(w['seed']+991)
    for tr,states in load_trajectories(target/'data',split):
        runs={n:[] for n in specs}
        for rep in range(repeats):
            for name in rng.permutation(list(specs)):
                path=dest/f'{tr["id"]}_{name}_{rep}.json'
                if path.exists():
                    entry=read(path)
                    if not resume or entry['protocol']!=protocol or digest(entry['result'])!=entry['result_digest']:
                        raise ValueError('changed partial evaluation row')
                    result=entry['result']
                else:
                    result=run_episode(states,backend,w,**specs[name])
                    write_json(path,dict(protocol=protocol,result=result,result_digest=digest(result)))
                runs[name].append(result)
        rows.append(dict(trajectory=tr['id'],case_group=tr['case_group'],runs=runs))
        print('[sequence]',split,tr['id'],flush=True)
    # Main and component contrasts use measured paired case ratios, never a
    # product of aggregate speedups from different cohorts.
    contrasts={name:summarize(rows,name,w['seed']) for name in ('C_tuned_reuse','HS_matched_reuse','World_C')}
    report=dict(protocol=protocol,reference='C_tuned_reuse',summary=contrasts['C_tuned_reuse'],
        contrasts=contrasts,rows=rows,heuristic_C=c,heuristic_HS=hs,source_kind=h['source_kind'],physics=h['physics'],
        input_data_sha256=h['data_sha256'],hardware=read(out/'thesis_manifest.json')['hardware'],
        primary_classical_plan=h['primary_classical_plan'],learned_transfer=False,trained_H_S=True,
        confidence_scope='paired bootstrap of independent physical-case trajectory medians; no timestep/repeat pseudoreplication',
        time_scope='all setup/refactor/selection/inference/solve/recovery; input disk I/O excluded',
        full_cfd_wall_clock_measured=False,combustion_validated=False,performance_certified=False)
    write_json(dest/'report.json',report);write_json(progress,dict(protocol=protocol,complete=True))
    for name,s in report['summary'].items():print(name,s['successes'],'/',s['total'],s['geometric_speedup'],flush=True)
    return report


def freeze(output):
    out,w,h,backend=load(output);assert_open(out);target=out/'temporal'
    if not read(target/'validation/progress.json')['complete']:raise ValueError('independent temporal validation required')
    training,models=_artifacts(target,h)
    report=read(target/'validation/report.json')
    if report['protocol']['training']!=file_hash(target/'training.json'):raise ValueError('model changed since validation')
    paths=['configuration.json','thesis_manifest.json','selector_rules.json','tuned_classical.json',
           'expert_selection.json','static_data_manifest.json','temporal/run_manifest.json','temporal/collection.json',
           'temporal/training.json','temporal/world_C.pt','temporal/world_HS.pt','temporal/validation/report.json',
           'temporal/data/sequence_manifest.json']
    if (out/'strong_audit_rules.json').exists():paths+=['strong_audit_rules.json','strong_audit_evidence.json']
    write_json(out/'freeze.json',dict(version=THESIS_VERSION,files={p:file_hash(out/p) for p in paths},
        numerical_scope='H_S + solver-state world model; not full CFD',source_kind=h['source_kind'],test_seen=False))


def check_freeze(output):
    out=Path(output);f=read(out/'freeze.json')
    if f['version']!=THESIS_VERSION:raise ValueError('wrong freeze version')
    for p,sha in f['files'].items():
        if file_hash(out/p)!=sha:raise ValueError('frozen artifact changed: '+p)
    return f


def solver_from_run(output, *, mode='World_HS', require_frozen=True):
    """Construct a stateful single-stream Python bridge. This does not invoke
    OpenFOAM; the caller must supply verified current snapshots and reset at
    physical-case boundaries. Finalized-LDU adapters remain in world_model.
    """
    out,w,h,backend=load(output)
    if require_frozen:check_freeze(out)
    training,models=_artifacts(out/'temporal',h)
    base=training['baselines']['classical']['selected']
    if mode in ('World_C','World_HS'):
        return WorldSolver(backend,w,artifact=models['world_C' if mode=='World_C' else 'world_HS'],heuristic_name=base)
    if mode=='C_tuned_reuse':return WorldSolver(backend,w,heuristic_name=base)
    if mode=='HS_matched_reuse':return WorldSolver(backend,w,heuristic_name=base,neural=True)
    raise ValueError('unknown frozen inference mode')
