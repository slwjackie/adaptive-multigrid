"""Six-stage, resumable training with separate measured acceptance and audit.

No training loop runs silently in a solve. Every stage writes candidate weights,
raw timings, acceptance decisions, and a resume checkpoint before continuing.
"""
from __future__ import annotations
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import json
import math
import numpy as np
import torch
from ..provenance import write_json,json_safe,hardware_environment,stable_norm
from ..hierarchy import classical_cycle,classical_step
from ..strategy import global_context_features
from .models import Components
from .solver import PreparedAdaptiveMG
from .banks import Stats
from .data import dataset,split_manifest
from .unroll import rollout
from .native import block_statistics
from .spatial import features_from_statistics
from .controller import controller_features
from .evaluation import measured,paired_suite,summarize,save_summary,certify
from .acceptance import evaluate_stage_acceptance


def _append(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(json_safe(record),ensure_ascii=False)+'\n')


def operator_stage(model,train,cfg,settings,stage,out,resume=False):
    """Stage 1/2/3/6: real backprop through both S and P when selected."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'candidate.pt').exists() and not resume:
        raise FileExistsError('candidate already preserved; use a new output directory or resume')
    if not (out/'entry.pt').exists():
        model.save(out/'entry.pt',extra={'role':'entry','stage':stage})
    candidate=deepcopy(model);epochs=int(settings['epochs'].get(stage,1))
    trained_modules={'smoother':['smoother'],'transfer':['transfer'],
                     'joint':['smoother','transfer'],'hierarchical':['smoother','transfer','detector']}[stage]
    stagecfg=replace(cfg,mode='hybrid',use_smoother=stage!='transfer',use_transfer=stage!='smoother',
        spatial=stage=='hierarchical',gate_mode='learned' if stage=='hierarchical' else 'open')
    start_epoch=0;start_pos=0;payload=None
    if resume and (out/'resume.pt').is_file():
        candidate=Components.load(out/'resume.pt')
        payload=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)['extra']
        start_epoch=int(payload.get('epoch',0));start_pos=int(payload.get('next_position',0))
    for n in ('smoother','transfer','detector','controller'):
        for p in getattr(candidate,n).parameters():p.requires_grad_(n in trained_modules)
    # Move heavy models before optimizer construction; numerical graph is CPU FP64.
    from .banks import resolve_device
    dev=resolve_device(cfg);dt=torch.float32 if cfg.inference_dtype=='float32' else torch.float64
    candidate.smoother.to(dev,dtype=dt);candidate.transfer.to(dev,dtype=dt)
    parameters=[p for n in trained_modules for p in getattr(candidate,n).parameters()]
    opt=torch.optim.Adam(parameters,lr=float(settings.get('operator_lr',1e-4)))
    if payload and 'optimizer' in payload:opt.load_state_dict(payload['optimizer'])
    records=[];total_steps=0
    prior_steps=int(payload.get("cumulative_updates",0)) if payload else 0
    # Hardware-calibrated scalar proxy. For fixed S/P schedules it is a CONSTANT;
    # action policy training, not a fictitious timing gradient, optimizes costs.
    probe_cfg=replace(stagecfg,spatial=False,gate_mode='open')
    try:
        e=train[0];c=measured(e,candidate,replace(probe_cfg,mode='classical'))
        h=measured(e,candidate,probe_cfg)
        tc=np.median(c['cycle_seconds']) if c['cycle_seconds'] else 1.
        times=np.array(h['cycle_seconds'],float)
        if len(times):times[0]=max(1e-9,times[0]-h['stats']['nn_setup_seconds'])
        ratio=float(np.median(times)/max(tc,1e-9)) if len(times) else 1.
    except (ValueError,RuntimeError,FloatingPointError):ratio=1.
    write_json(out/'compute_calibration.json',{'hybrid_classical_cycle_ratio':ratio,
        'proxy_constant_for_fixed_schedule':stage!='hierarchical','hardware':hardware_environment()})
    for epoch in range(start_epoch,epochs):
        order=np.random.default_rng(settings.get('seed',71)+epoch).permutation(len(train))
        for position in range(start_pos if epoch==start_epoch else 0,len(order)):
            e=train[int(order[position])];opt.zero_grad(set_to_none=True)
            try:
                schedule=None;exploration=False
                if stage=='hierarchical':
                    length=int(settings.get('unroll_prefix',2))+int(settings.get('unroll_tail',2))
                    replay_cfg=replace(stagecfg,mode='research',mg=replace(stagecfg.mg,max_cycles=length))
                    replay=PreparedAdaptiveMG(e.a,e.n,candidate,replay_cfg).solve(e.b)
                    schedule=['H' if p=='hybrid' else 'C' for p in replay.cycle_path]
                    schedule=(schedule+['C']*length)[:length]
                    # Offline exploration creates training states, never win labels.
                    rg=np.random.default_rng(settings.get('seed',71)+epoch*len(train)+position)
                    if rg.random()<float(settings.get('hierarchical_exploration',.25)):
                        schedule[int(rg.integers(max(1,length-1)))]='H';exploration=True
                    schedule[-1]='C'
                loss,details,_=rollout(e,candidate,stagecfg,
                    prefix=int(settings.get('unroll_prefix',2)),tail=int(settings.get('unroll_tail',2)),
                    compute_ratio=ratio,lambda_compute=float(settings.get('lambda_compute',.02)),hierarchical=stage=='hierarchical',schedule=schedule)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                if not loss.requires_grad:raise FloatingPointError('no active differentiable neural path')
                loss.backward()
                norms={n:float(sum(p.grad.detach().double().square().sum().item() for p in getattr(candidate,n).parameters() if p.grad is not None)**.5) for n in trained_modules}
                g=torch.nn.utils.clip_grad_norm_(parameters,float(settings.get('gradient_clip',2.)))
                if not torch.isfinite(g):raise FloatingPointError('nonfinite gradient')
                opt.step();total_steps+=1
                record={'epoch':epoch,'position':position,'operator':e.name,'loss':float(loss.detach()),
                    'residual_loss':float(details['residual'].detach()),'compute_proxy':float(details['cost'].detach()),
                    'gradient_norms':norms,'history':[float(v.detach()) for v in details['history']],'skipped':False,'action_schedule':details['actions'],'offline_exploration':exploration}
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                # Not accepted silently: persistent skip reasons are required.
                record={'epoch':epoch,'position':position,'operator':e.name,'skipped':True,'reason':str(exc)}
            records.append(record);_append(out/'training.jsonl',record)
            if (position+1)%int(settings.get('checkpoint_every',8))==0 or position+1==len(order):
                candidate.save(out/'resume.pt',extra={'epoch':epoch,'next_position':position+1,'optimizer':opt.state_dict(),'cumulative_updates':prior_steps+total_steps})
                write_json(out/'progress.json',{'stage':stage,'epoch':epoch,'completed_in_epoch':position+1,
                    'operators_per_epoch':len(order),'status':'running'})
        start_pos=0
        candidate.save(out/'resume.pt',extra={'epoch':epoch+1,'next_position':0,'optimizer':opt.state_dict(),'cumulative_updates':prior_steps+total_steps})
        print(f'[{stage}] epoch {epoch+1}/{epochs}, operators {len(train)}, updates {total_steps}',flush=True)
    for name in trained_modules:
        if name in {'smoother','transfer','detector'} and prior_steps+total_steps>0:candidate.metadata[name+'_trained']=True
    for module in candidate.modules():
        module.cpu()
        for p in module.parameters():p.requires_grad_(True)
    candidate.metadata['certificate']={'validated':False,'reason':'weights_changed_requires_new_audit'}
    if (out/'candidate.pt').exists():
        preserved=Components.load(out/'candidate.pt')
        if candidate.signature()!=preserved.signature() or candidate.metadata!=preserved.metadata:
            raise FileExistsError('resume would overwrite a different preserved candidate; use a new output directory')
    else:
        candidate.save(out/'candidate.pt',extra={'role':'candidate','stage':stage,'config':stagecfg.to_dict()})
    report={'stage':stage,'epochs':epochs,'cumulative_updates':prior_steps+total_steps,'updates_this_invocation':total_steps,'samples_seen_this_invocation':len(records),
        'skips_this_invocation':sum(r['skipped'] for r in records),'performance_certified':False}
    write_json(out/'progress.json',dict(report,status='training_complete_pending_validation'))
    return candidate,stagecfg,report


def accept_operator(candidate,previous,examples,cfg,settings,out,*,stage=None):
    """Time+accuracy acceptance against same-architecture incumbent, NOT loss only."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    repeats=int(settings.get('validation_repeats',3));raw=[]
    if repeats < 1:raise ValueError('validation_repeats must be positive')
    rng=np.random.default_rng(181)
    for e in examples:
        runs={'previous':[],'candidate':[]}
        # Warm-up both arms, but cold constructor+solve is measured each time.
        for model in (previous,candidate):measured(e,model,cfg)
        for _ in range(repeats):
            for arm in rng.permutation(['previous','candidate']):
                run=measured(e,candidate if arm=='candidate' else previous,cfg)
                run['measurement_config']=cfg.to_dict()
                runs[arm].append(run)
        raw.append({'example':e.manifest(),'runs':runs})
        write_json(out/'raw.json',raw)
    report=evaluate_stage_acceptance(raw,candidate,previous,cfg,settings,stage=stage)
    write_json(out/'acceptance.json',report)
    return report['accepted'],report


def train_detector(model,examples,cfg,settings,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);features=[];labels=[];records=[]
    force=replace(cfg,mode='hybrid',spatial=False,gate_mode='open',use_smoother=True)
    for e in examples:
        p=PreparedAdaptiveMG(e.a,e.n,model,force);st=Stats()
        try:
            root=p.ensure_learned(st);x=np.zeros_like(e.b)
            for state in range(int(settings.get('detector_states',2))):
                residual=e.b-root.a@x
                scale=max(np.max(np.abs(residual),initial=0),1e-300)
                _,raw,blocks=block_statistics(residual/scale,root.shape,cfg.detector_max_side)
                f=features_from_statistics(raw,root.shape,blocks)
                from time import perf_counter
                tc=[];tn=[]
                for _ in range(int(settings.get('timing_repeats',3))):
                    t=perf_counter();dc=classical_step(root,residual,cfg.mg,Stats(),reverse=False);tc.append(perf_counter()-t)
                    t=perf_counter();dn=root.neural_stencil.apply(residual,Stats());tn.append(perf_counter()-t)
                _,rawc,_=block_statistics((residual-root.a@dc)/scale,root.shape,cfg.detector_max_side)
                _,rawn,_=block_statistics((residual-root.a@dn)/scale,root.shape,cfg.detector_max_side)
                e0=np.maximum(raw[:,0],1e-30)
                ec=-.5*np.log(np.maximum(rawc[:,0],1e-30)/e0)/max(np.median(tc),1e-9)
                en=-.5*np.log(np.maximum(rawn[:,0],1e-30)/e0)/max(np.median(tn),1e-9)
                target=((en>0)&(en>(1+cfg.win_margin)*ec)&(rawn[:,0]<rawc[:,0])).astype(np.float32)
                features.extend(f);labels.extend(target)
                records.append({'operator':e.name,'state':state,'blocks':len(target),'positives':int(target.sum()),
                    'classical_sweep_seconds':float(np.median(tc)),'neural_sweep_seconds':float(np.median(tn))})
                x=classical_cycle(p.classical,x,e.b,cfg.mg,Stats())
        except (ValueError,RuntimeError,FloatingPointError) as exc:records.append({'operator':e.name,'error':str(exc)})
    write_json(out/'oracle.json',{'records':records,'label_scope':'local smoothing efficiency proxy; final global acceptance separate'})
    if not features:
        write_json(out/'status.json',{'trained':False,'reason':'no_finite_spatial_examples'});return model
    x=torch.tensor(np.asarray(features),dtype=torch.float32);y=torch.tensor(np.asarray(labels),dtype=torch.float32)
    m=model.detector.cpu().float();m.normalize_on(x)
    pos=torch.where(y>0)[0];neg=torch.where(y==0)[0]
    if len(pos)==0:
        with torch.no_grad():m.fc2.weight.zero_();m.fc2.bias.fill_(-20)
        model.metadata['detector_trained']=False
        write_json(out/'status.json',{'trained':False,'reason':'no_genuine_neural_positive','examples':len(y),'positives':0})
        model.save(out/'detector.pt');return model
    opt=torch.optim.Adam(m.parameters(),lr=float(settings.get('policy_lr',.002)))
    generator=torch.Generator().manual_seed(settings.get('seed',71))
    for epoch in range(int(settings['epochs'].get('detector',100))):
        # Oversample actual positives; audit distribution is never rebalanced.
        k=min(len(y),int(settings.get('policy_batch',256)))
        ip=pos[torch.randint(len(pos),(k//2,),generator=generator)]
        inn=neg[torch.randint(len(neg),(k-k//2,),generator=generator)] if len(neg) else ip
        idx=torch.cat((ip,inn));opt.zero_grad(set_to_none=True)
        logits=m(x[idx]).ravel()
        bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[idx])
        rank=torch.relu(.2-logits[:len(ip),None]+logits[len(ip):][None,:]).mean() if len(pos) and len(neg) else logits.sum()*0
        loss=bce+.1*rank;loss.backward();opt.step()
    model.metadata['detector_trained']=True
    signatures=model.component_signatures()
    model.metadata['detector_expert_signatures']={name:signatures[name] for name in ('smoother','transfer')}
    np.savez_compressed(out/'spatial_data.npz',features=x.numpy(),targets=y.numpy())
    write_json(out/'status.json',{'trained':True,'examples':len(y),'positives':len(pos),'epochs':int(settings['epochs'].get('detector',100))})
    model.save(out/'detector.pt');return model


def temporal_table(model,examples,cfg,settings,out):
    """Offline paired C vs H-burst+tail from IDENTICAL residual states."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);path=out/'action_table.json'
    rows=[]
    for e in examples:
        p=PreparedAdaptiveMG(e.a,e.n,model,replace(cfg,mode='classical'))
        x0=np.zeros_like(e.b)
        for state in range(int(settings.get('controller_states',2))):
            # One actual C cycle yields observable, not counterfactual, history.
            from time import perf_counter
            t=perf_counter();x1=classical_cycle(p.classical,x0,e.b,cfg.mg,Stats());tc_cycle=perf_counter()-t
            rbefore=e.b-p.a@x0;r=e.b-p.a@x1
            rho=stable_norm(r)/max(stable_norm(rbefore),1e-300)
            norm0=stable_norm(e.b);target=max(cfg.mg.absolute_tolerance,cfg.mg.tolerance*norm0)
            if stable_norm(r)<=target:break
            # Fix the ORIGINAL stopping threshold across both branches.
            mg=replace(cfg.mg,absolute_tolerance=target)
            configs={'C':replace(cfg,mg=mg,mode='classical'),'H':replace(cfg,mg=mg,mode='burst')}
            arms={'C':[],'H':[]};rng=np.random.default_rng(settings.get('seed',71)+state)
            for _ in range(int(settings.get('timing_repeats',3))):
                for arm in rng.permutation(['C','H']):arms[arm].append(measured(e,model,configs[arm],x0=x1))
            c=arms['C'];h=arms['H'];cs=all(v['converged'] for v in c);hs=all(v['converged'] for v in h)
            tc=float(np.median([v['wall_seconds'] for v in c]));th=float(np.median([v['wall_seconds'] for v in h]))
            hr=h[len(h)//2];first=next((v for v in hr['trace'] if v.get('path')=='hybrid'),None)
            setup=float(np.median([v['stats']['nn_setup_seconds'] for v in h]))
            tr=max(tc_cycle,1e-9)
            if first and 0<rho<1:
                rh=max(float(first['rho']),1e-16);hc=max(float(first['seconds'])-hr['stats']['nn_setup_seconds'],1e-9)
                rate_ratio=max(-math.log(rh),1e-6)/max(-math.log(rho),1e-6)
                time_ratio=hc/tr
            else:rate_ratio=.01;time_ratio=10.
            eta_ratio=rate_ratio/max(time_ratio,1e-9)
            instability=(100*int(not hs)+25*int(cs and not hs)+np.mean([v['stats']['rollback_count'] for v in h]))
            j=th/max(tc,1e-9)+float(settings.get('lambda_instability',2))*instability+float(settings.get('lambda_eta',.1))*max(0,1-eta_ratio)
            genuine=bool(cs and hs and first is not None and th<(1-cfg.win_margin)*tc and j<1-cfg.win_margin)
            if not cs and not hs:j=max(j,100.)
            f=controller_features(global_context_features(e.a,(e.n,e.n)),stable_norm(r)/max(norm0,1e-300),rho,tr,'CLASSICAL',False,state+2,target/max(norm0,1e-300),0,int(rho>=cfg.mg.stagnation_rho))
            targets=[math.log(max(rate_ratio,1e-6)),math.log(max(time_ratio,1e-6)),math.log1p(setup/tr)]
            row={'operator':e.manifest(),'state':state,'features':f.tolist(),'targets':targets,'J_C':1.,'J_H':float(j),
                'hybrid_positive':genuine,'classical_converged':cs,'hybrid_converged':hs,'eta_ratio':eta_ratio,'runs':arms}
            rows.append(row);write_json(path,rows)
            # A warm-cache feature row uses *measured NN setup* removed, not a
            # fictional setup saving in the reported cold benchmark.
            warmf=f.copy();warmf[22]=1. # operator18 + dynamic index4 (cached)
            warmtime=max(th-setup,1e-9)
            warmj=warmtime/max(tc,1e-9)+float(settings.get('lambda_instability',2))*instability+float(settings.get('lambda_eta',.1))*max(0,1-eta_ratio)
            warm=dict(row,features=warmf.tolist(),J_H=float(warmj),
                hybrid_positive=bool(cs and hs and first is not None and warmj<1-cfg.win_margin),
                regime='setup-subtracted training estimate; NOT a warm benchmark')
            rows.append(warm);write_json(path,rows)
            x0=x1
    return rows


def fit_controller(model,rows,settings,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not rows:
        write_json(out/'status.json',{'trained':False,'reason':'no_controller_states'});return model
    x=torch.tensor([r['features'] for r in rows],dtype=torch.float32)
    target=torch.tensor([r['targets'] for r in rows],dtype=torch.float32).clamp(-12,12)
    y=torch.tensor([r['hybrid_positive'] for r in rows],dtype=torch.float32)
    risk=torch.tensor([min(r['J_H'],100.) for r in rows],dtype=torch.float32)
    m=model.controller.cpu().float();m.normalize_on(x)
    opt=torch.optim.Adam(m.parameters(),lr=float(settings.get('policy_lr',.002)))
    for epoch in range(int(settings['epochs'].get('controller',200))):
        opt.zero_grad(set_to_none=True);v=m(x);prob=torch.sigmoid(v[:,3])
        regression=torch.nn.functional.smooth_l1_loss(v[:,:3],target)
        bce=torch.nn.functional.binary_cross_entropy_with_logits(v[:,3],y)
        expected=((1-prob)+prob*risk).mean()
        loss=regression+.2*bce+.2*expected
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
    model.metadata['controller_trained']=True;model.metadata['certificate']={'validated':False}
    signatures=model.component_signatures()
    experts={name:signatures[name] for name in ('smoother','transfer')}
    model.metadata['controller_expert_signatures']=experts
    detector_current=(not model.metadata.get('detector_trained')
                      or model.metadata.get('detector_expert_signatures')==experts)
    if detector_current:
        model.metadata['policy_stale']=False
        model.metadata.pop('policy_stale_reason',None)
    else:model.mark_policy_stale('detector_expert_signature_missing_or_mismatched')
    report={'trained':True,'states':len(rows),'genuine_positive_states':int(y.sum()),
        'objective':'regression + BCE + expected(T/Tc + lambda_I I + lambda_eta max(0,1-etaH/etaC))',
        'no_online_shadow_probes':True,'epochs':int(settings['epochs'].get('controller',200))}
    write_json(out/'status.json',report);model.save(out/'controller.pt');return model


def pipeline(config_path,output,initial=None,resume=False):
    settings=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    existing_config=out/'configuration.json'
    if resume and existing_config.exists() and json.loads(existing_config.read_text())!=settings:
        raise ValueError('resume configuration differs; use the original config or a new output directory')
    if not resume and (out/'current.pt').exists():
        raise FileExistsError('output already contains a run; use --resume or a different output directory')
    if resume and (out/'pipeline_progress.json').exists():
        old=json.loads((out/'pipeline_progress.json').read_text())
        if old.get('status')=='complete' and (out/'checkpoint_v6_7_final.pt').exists():
            return Components.load(out/'checkpoint_v6_7_final.pt'),json.loads((out/'audit/summary.json').read_text())
    from .config import AdaptiveConfig
    cfg=AdaptiveConfig.from_dict(settings['solver'])
    torch.set_num_threads(int(settings.get('torch_threads',1)))
    write_json(out/'configuration.json',settings);write_json(out/'environment.json',hardware_environment())
    seed=settings.get('seed',71)
    train=dataset(settings['train_sizes'],settings['train_per_size'],seed)
    tune=dataset(settings['tune_sizes'],settings['tune_per_size'],seed+100001)
    audit=dataset(settings['audit_sizes'],settings['audit_per_size'],seed+200002)
    write_json(out/'split_manifest.json',split_manifest(train=train,tune=tune,audit=audit))
    if resume and (out/'current.pt').is_file():model=Components.load(out/'current.pt')
    elif initial:
        try:model=Components.load(initial)
        except ValueError:model=Components.migrate_v66(initial)
    else:model=Components.create(hidden=int(settings.get('hidden',16)),seed=seed)
    donepath=out/'pipeline_progress.json'
    done=json.loads(donepath.read_text()).get('completed',[]) if resume and donepath.exists() else []
    stages=['smoother','transfer','joint','detector','controller','hierarchical']
    validation=tune[:int(settings.get('stage_validation_cases',len(tune)))]
    for stage in stages:
        if stage in done:continue
        stageout=out/stage
        write_json(donepath,{'status':'running','current_stage':stage,'completed':done})
        if stage in {'smoother','transfer','joint','hierarchical'}:
            candidate,stagecfg,report=operator_stage(model,train,cfg,settings,stage,stageout,resume)
            validation_cfg=replace(stagecfg,mode='research') if stage=='hierarchical' else stagecfg
            accepted,acceptance=accept_operator(candidate,model,validation,validation_cfg,settings,stageout/'validation',stage=stage)
            if accepted:
                model=candidate
                model.mark_policy_stale('stage_promotion_requires_policy_refit')
                model.metadata['stage_promoted']=True
                model.metadata['direct_classical_speedup_observed']=None
                if not (stageout/'promoted.pt').exists():
                    model.save(stageout/'promoted.pt',extra={'role':'promoted','stage':stage,'acceptance':acceptance})
                elif Components.load(stageout/'promoted.pt').signature()!=model.signature():
                    raise FileExistsError('promotion would overwrite a different checkpoint; use a new output directory')
            # Both the rejected candidate and accepted incumbent are persisted.
            status=dict(report);status.update(acceptance);status['acceptance']=acceptance
            status['checkpoints']={'entry':str(stageout/'entry.pt'),'candidate':str(stageout/'candidate.pt'),
                                   'promoted':str(stageout/'promoted.pt') if accepted else None}
            write_json(stageout/'status.json',status)
            if stage=='hierarchical' and accepted:
                model=train_detector(model,train,cfg,settings,stageout/'refresh_detector')
                table=temporal_table(model,train,cfg,settings,stageout/'refresh_actions')
                model=fit_controller(model,table,settings,stageout/'refresh_controller')
        elif stage=='detector':model=train_detector(model,train,cfg,settings,stageout)
        else:
            table=temporal_table(model,train,cfg,settings,stageout)
            model=fit_controller(model,table,settings,stageout)
        model.save(out/'current.pt');done.append(stage)
        write_json(donepath,{'status':'running','completed':done})
    # Audit is untouched by all checkpoint choices. A failed audit does not
    # trigger more tuning on these same cases; defaults stay classical.
    configs={'classical':replace(cfg,mode='classical'),'adaptive':replace(cfg,mode='research')}
    rows=paired_suite(audit,model,configs,repeats=int(settings.get('audit_repeats',3)),
        warmups=int(settings.get('warmups',1)),output=out/'audit')
    result=summarize(rows);save_summary(out/'audit',result)
    certificate=certify(model,cfg,rows,min_cases=int(settings.get('audit_min_cases',20)),margin=cfg.win_margin)
    model.metadata['certificate']=certificate
    model.save(out/'checkpoint_v6_7_final.pt');write_json(out/'certificate.json',certificate)
    write_json(donepath,{'status':'complete','completed':stages,'performance_certified':certificate['validated']})
    return model,result
