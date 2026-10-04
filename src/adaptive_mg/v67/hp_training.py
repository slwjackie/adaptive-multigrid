"""TRAIN-only all-level P learning, persistent slow probes, measured selection.

Frozen parent interpolation is a deliberate constant reference. Learned Ac and
smoothing remain differentiable. Raw m-cycle probe loss is a training surrogate;
checkpoints are selected with the existing safeguarded NumPy solver, never final.
"""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import hashlib
import json
import math
import numpy as np
import torch
from torch.nn import functional as F

from ..provenance import hardware_environment, stable_norm, write_json
from .banks import Stats, hybrid_cycle
from .limited import digest_file, forced_config
from .models import Components
from .research_data import _hash
from .research_training import sample_config, transfer_feasibility, _levels
from .strong import PreparedStrongMG
from .spatial import SpatialState
from .unroll import make_graph, cycle

VERSION='asymptotic-p-training-v1'


def energy_norm(graph, errors):
    return (errors*graph.a.apply(errors)).sum(dim=0).clamp_min(1e-200).sqrt()


def normalized_probes(graph,values):
    norms=energy_norm(graph,values)
    if not bool(torch.isfinite(norms).all()) or not bool((norms>1e-90).all()):
        raise FloatingPointError('degenerate/nonfinite probe; do not report as good contraction')
    return values/norms[None,:]


def probe_loss(graph, model, cfg, probes, *, cycles=4, tail_weight=.3, temperature=.15, stability_weight=.1):
    """Actual recursive V-cycles, mean log contraction + log-mean-exp tail.

    This is not an exact spectral radius, norm bound or no-harm certificate.
    Vector norms, not squared norms, define all reported contraction factors.
    """
    if cycles<2 or temperature<=0 or tail_weight<0 or stability_weight<0:
        raise ValueError('invalid asymptotic loss controls')
    e=probes;initial=energy_norm(graph,e);history=[initial]
    for k in range(cycles):
        e=cycle(graph,e,torch.zeros_like(e),model,cfg,k)
        history.append(energy_norm(graph,e))
    h=torch.stack(history).clamp_min(1e-100)
    factors=h[1:]/h[:-1]
    bulk=(torch.log(h[-1]/h[0])/cycles).mean()
    logs=torch.log(factors[-1].clamp_min(1e-100))
    tail=temperature*(torch.logsumexp(logs/temperature,0)-math.log(logs.numel()))
    stability=F.relu(factors-1.).square().mean()
    loss=bulk+tail_weight*tail+stability_weight*stability
    return loss,dict(bulk=bulk,tail=tail,stability=stability,
                     history=h,tail_factor=factors[-1],objective_scope='raw full V-cycles; not deployed policy')


def _snapshot(model,path,extra=None):
    model.save(path,extra=extra or {})


def validate_checkpoint(model,examples,cfg,rules,*,rhs_count=3,repeats=3,seed=1,tail_cycles=20):
    """Development-only checkpoint selection. All solves use true FP64 residuals.

    Identical zero guesses and fixed random exact solutions generate diverse RHS.
    Pair order alternates; the independent final benchmark must remeasure winners.
    Tail diagnostic uses raw V-cycles, kept separate from safeguarded timings.
    """
    if not examples or any(e.research_split not in ('validation','hp_validation') for e in examples):
        raise ValueError('checkpoint selection needs independent development validation')
    if rhs_count<1 or repeats<1 or tail_cycles<3:raise ValueError('invalid validation protocol')
    rows=[]
    for e in examples:
        scfg=sample_config(e,cfg,rules,'H_P')
        rng=np.random.default_rng(int(seed)^int(e.group_digest[:8],16))
        exact=rng.normal(size=(rhs_count+1,e.a.shape[0]));bs=(e.a@exact.T).T
        calls={};checks={};counts={};used={};setup_ok=True
        # Each candidate retains its own prepared caches across repeats only.
        solvers={name:PreparedStrongMG(e.a,e.n,model.frozen_inference_copy() if name=='H_P' else None,
                    replace(scfg,branch=name,mode='classical' if name=='C' else 'research'),rules)
                 for name in ('C','H_P')}
        for name,p in solvers.items():
            if name=='H_P':
                try:p.ensure_branch('H_P',Stats())
                except (ValueError,RuntimeError,FloatingPointError):setup_ok=False  # still time the real fallback
            p.solve(bs[-1]);calls[name]=[];checks[name]=[];counts[name]=[];used[name]=[]
        for repeat in range(repeats):
            order=('C','H_P') if repeat%2==0 else ('H_P','C')
            for name in order:
                p=solvers[name];t=perf_counter();results=p.solve_many(bs[:-1]);elapsed=perf_counter()-t
                ok=all(r.converged and r.executed_cycles<=scfg.mg.max_cycles and
                       np.isfinite(stable_norm(b-e.a@r.x)) and stable_norm(b-e.a@r.x)<=r.stopping_threshold
                       for b,r in zip(bs[:-1],results))
                calls[name].append(elapsed);checks[name].append(ok)
                counts[name].append(sum(r.executed_cycles for r in results))
                used[name].append(any(r.stats.get('accepted_neural_cycles',0)>0 for r in results))
        rho={}
        for name,p in solvers.items():
            err=rng.normal(size=e.a.shape[0]) if name=='C' else err0.copy()
            if name=='C':err0=err.copy()
            ratios=[]
            try:
                root=p.ensure_branch(name,Stats());spatial=SpatialState(p.components,p.config) if name=='H_P' else None
                from ..hierarchy import classical_cycle
                for k in range(tail_cycles):
                    before=np.sqrt(max(float(err@(e.a@err)),1e-200))
                    if before<1e-90:break
                    err/=before
                    err=(classical_cycle(root,err,np.zeros_like(err),p.config.mg,Stats()) if name=='C' else
                         hybrid_cycle(root,err,np.zeros_like(err),p.config,Stats(),spatial,k,True))
                    ratios.append(np.sqrt(max(float(err@(e.a@err)),1e-200)))
                rho[name]=float(np.exp(np.log(np.maximum(ratios[-min(8,len(ratios)):],1e-100)).mean())) if ratios else None
            except (ValueError,RuntimeError,FloatingPointError):rho[name]=None
        c_ok=all(checks['C']);h_ok=all(checks['H_P'])
        rows.append(dict(operator=e.group_digest,name=e.name,n=e.n,C_success=c_ok,H_success=h_ok,
            C_seconds=float(np.median(calls['C'])),H_seconds=float(np.median(calls['H_P'])),
            C_cycles=float(np.median(counts['C'])),H_cycles=float(np.median(counts['H_P'])),
            neural_used=any(used['H_P']),proposal_setup_ok=setup_ok,tail_rho=rho,selected_plan=scfg.mg.strategy_name,
            speedup=float(np.median(calls['C'])/np.median(calls['H_P'])) if c_ok and h_ok else None))
    losses=[r['operator'] for r in rows if r['C_success'] and not r['H_success']]
    common=[r['speedup'] for r in rows if r['speedup'] is not None]
    return dict(rows=rows,new_failures=losses,eligible=not losses and bool(common) and all(r['proposal_setup_ok'] for r in rows),
                geometric_speedup=float(np.exp(np.log(common).mean())) if common else None,
                source='independent development operators; measured prepared multi-RHS',certified=False)


def train_asymptotic(initial,examples,validation,cfg,rules,settings,out,*,resume=False,max_updates=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not examples or any(e.research_split not in ('train','hp_train') for e in examples):
        raise ValueError('weights may use TRAIN operators only')
    if not validation or any(e.research_split not in ('validation','hp_validation') for e in validation) or set(e.group_digest for e in examples)&set(e.group_digest for e in validation):
        raise ValueError('operator-disjoint validation required')
    if getattr(initial.transfer,'reference',None)!='frozen_parent' or getattr(initial.transfer,'parameterization',None)!='affine':
        raise ValueError('asymptotic HP workflow requires affine frozen-parent expert')
    settings=dict(settings);steps=int(settings['updates']);q=int(settings.get('random_probes',8));slow_count=int(settings.get('slow_probes',8))
    power=int(settings.get('power_cycles',3));m=int(settings.get('probe_cycles',4));interval=int(settings.get('validate_every',40))
    if min(steps,q,slow_count,power,interval)<1 or m<2:raise ValueError('invalid training counts')
    manifest=dict(version=VERSION,settings=settings,config=cfg.to_dict(),rules_digest=rules.digest(),
        initial_signature=initial.generation_signature(),train=[e.group_digest for e in examples],
        validation=[e.group_digest for e in validation],hardware=hardware_environment(),
        objective='full raw V-cycle random+persistent error probes',checkpoint_selection='safeguarded warm_multi_rhs',final_seen=False)
    path=out/'training_manifest.json'
    if path.exists():
        if not resume or json.loads(path.read_text())!=json.loads(json.dumps(manifest)):
            raise ValueError('changed training contract or missing --resume')
    else:
        if resume or any(out.iterdir()):raise FileExistsError('new empty expert directory required')
        write_json(path,manifest)
    model=deepcopy(initial).eval();start=0;records=[];probes={};best=None;best_step=None;validations=[]
    rng=torch.Generator(device='cpu').manual_seed(int(settings['seed']))
    order=np.random.default_rng(int(settings['seed'])).permutation(len(examples))
    if resume:
        payload=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
        model=Components.load(out/'resume.pt');extra=payload['extra']
        start=extra['next_update'];records=extra['records'];probes=extra['probes'];rng.set_state(extra['rng'])
        best=extra['best'];best_step=extra['best_step'];validations=extra['validations']
    for module in model.modules():
        for p in module.parameters():p.requires_grad_(module is model.transfer)
    optimizer=torch.optim.Adam(model.transfer.parameters(),lr=float(settings.get('learning_rate',.001)))
    if resume:optimizer.load_state_dict(extra['optimizer'])
    reference_cache=OrderedDict();begin=perf_counter()
    def validate(step):
        nonlocal best,best_step
        score=validate_checkpoint(model,validation,cfg,rules,rhs_count=int(settings.get('validation_rhs',3)),
                repeats=int(settings.get('validation_repeats',3)),seed=int(settings['seed'])+19001,
                tail_cycles=int(settings.get('validation_tail_cycles',20)))
        score['step']=step;write_json(out/'validation'/f'step_{step:06d}.json',score)
        validations.append(dict(step=step,eligible=score['eligible'],speedup=score['geometric_speedup']))
        if score['eligible'] and (best is None or score['geometric_speedup']>best):
            best=score['geometric_speedup'];best_step=step
            model.metadata.update(training_rules_digest=rules.digest(),training_branch='H_P',optimizer_updates=step,
                                  training_kind=VERSION,transfer_trained=step>0,smoother_trained=False,final_test_seen=False)
            _snapshot(model,out/'best.pt',dict(role='development_selected',step=step,validation_speedup=best,performance_certified=False))
    if start==0:validate(0)
    if max_updates is not None and (isinstance(max_updates,bool) or max_updates<1):raise ValueError('max_updates must be positive')
    stop=min(steps,start+max_updates) if max_updates is not None else steps
    for step in range(start,stop):
        e=examples[int(order[step%len(order)])];scfg=sample_config(e,cfg,rules,'H_P');t=perf_counter()
        key=e.group_digest
        if key not in reference_cache:
            reference_cache[key]=make_graph(e.a,(e.n,e.n),model,scfg,learned=False)
            if len(reference_cache)>int(settings.get('reference_cache_size',8)):reference_cache.popitem(last=False)
        reference=reference_cache[key];reference_cache.move_to_end(key)
        optimizer.zero_grad(set_to_none=True)
        graph=make_graph(e.a,(e.n,e.n),model,scfg,reference_root=reference)
        feasible,repair=transfer_feasibility(graph,model,scfg,reference)
        new=torch.randn(e.a.shape[0],q,generator=rng,dtype=torch.float64)
        with torch.no_grad():
            v=probes.get(key,torch.randn(e.a.shape[0],slow_count,generator=rng,dtype=torch.float64))
            v=normalized_probes(graph,v)
            for k in range(power):
                v=cycle(graph,v,torch.zeros_like(v),model,scfg,k)
                try:v=normalized_probes(graph,v)
                except FloatingPointError:v=normalized_probes(graph,torch.randn(v.shape,generator=rng,dtype=torch.float64))
            probes[key]=v.detach().clone()
            batch=normalized_probes(graph,torch.cat((new,v),1))
        if feasible['feasible']:
            loss,details=probe_loss(graph,model,scfg,batch,cycles=m,tail_weight=float(settings.get('lambda_tail',.3)),
                    temperature=float(settings.get('tail_temperature',.15)),stability_weight=float(settings.get('lambda_stability',.1)))
            task='learned_probe'
        else:loss=repair;details={};task='feasibility_repair_only'
        if not loss.requires_grad or not bool(torch.isfinite(loss)):raise FloatingPointError('inactive/nonfinite P objective')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.transfer.parameters(),float(settings.get('gradient_clip',2.)))
        if not bool(torch.isfinite(grad)):raise FloatingPointError('nonfinite transfer gradient')
        changes=[]
        for level in _levels(graph):
            w=level.interpolation_weights.detach();base=torch.as_tensor(level.base_weights,dtype=w.dtype)
            changes.append(dict(level=level.index,relative_delta=float(torch.linalg.vector_norm(w-base)/torch.linalg.vector_norm(base).clamp_min(1e-30)),
                                max_row_abs_sum=float(w.abs().sum(1).max())))
        optimizer.step()
        rec=dict(step=step+1,operator=key,name=e.name,loss=float(loss.detach()),gradient_norm=float(grad),
                 task=task,feasible=feasible['feasible'],levels=changes,train_seconds=perf_counter()-t,
                 **{k:float(v.detach()) for k,v in details.items() if torch.is_tensor(v) and v.ndim==0})
        records.append(rec)
        if (step+1)%interval==0 or step+1==steps:validate(step+1)
        model.metadata.update(training_rules_digest=rules.digest(),training_branch='H_P',optimizer_updates=step+1,
                              training_kind=VERSION,transfer_trained=True,final_test_seen=False)
        _snapshot(model,out/'resume.pt',dict(next_update=step+1,optimizer=optimizer.state_dict(),records=records,
                  probes=probes,rng=rng.get_state(),best=best,best_step=best_step,validations=validations))
        write_json(out/'training.json',records)
        print(f'[HP asymptotic] {step+1}/{steps} {e.name} {task} loss={rec["loss"]:.5g}',flush=True)
    complete=stop==steps
    if complete:
        _snapshot(model,out/'last.pt')
        if best_step is None:raise RuntimeError('no development checkpoint preserves baseline successes')
        selected=Components.load(out/'best.pt')
        selected.mark_policy_stale('new HP expert and parent require actual-RHS policy refit')
        _snapshot(selected,out/'candidate.pt',dict(selected_step=best_step,completed_updates=steps,
                   validation_speedup=best,selection_is_not_certification=True))
    report=dict(version=VERSION,updates=len(records),requested_updates=steps,skipped=0,all_updates_completed=complete,
        best_step=best_step,selected_warm_speedup=best,validation=validations,
        checkpoint=str(out/'candidate.pt') if complete else None,
        checkpoint_sha256=digest_file(out/'candidate.pt') if complete else None,
        training_seconds_this_invocation=perf_counter()-begin,final_test_seen=False,performance_certified=False)
    write_json(out/'status.json',report)
    return report
