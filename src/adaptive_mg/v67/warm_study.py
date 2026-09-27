"""Sequential warm-first experiment: expert development, THEN policy fitting.

This CLI never claims a performance improvement automatically. H0/H1/H2/H3
are ablations; named paper ideas are inspirations, not reproductions. Old
three-pillars runs remain readable for diagnostics but cannot be resumed under
changed source. Each stage pins data, checkpoints, rules and measured evidence.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch

from ..provenance import hardware_environment, write_json, stable_norm
from .banks import Stats
from .config import AdaptiveConfig
from .models import Components
from .multistage import make_multistage
from .solver import PreparedAdaptiveMG
from .strong import classical_bank, load_strong_rules
from .limited import digest_file
from .research_data import (_generate_split, _restore, _hash, _digests, _write_json,
    historical_operator_index, freeze_research, claim_final_evaluation,
    materialize_final_data, complete_final_evaluation)
from .research_training import create_research_components, train_expert
from .research_evaluation import evaluate_research, manufactured_rhs, REGIMES
from .cost_policy import (ContinuousCostPolicy, fit_cost_model, config_scope,
                          context_from_record)
from .three_pillars import (calibrate, _load_run, _development, _read, _sources,
                            _report, _ensure_development_open)

VERSION='warm-study-v1'


def _load(output):
    out,settings,cfg=_load_run(output)
    if settings.get('study',{}).get('version')!=VERSION:
        raise ValueError('use a warm-study config and a NEW run directory')
    return out,settings,cfg,load_strong_rules(out/'selector_rules.json')


def _tag(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',value):raise ValueError('invalid experiment tag')
    return value


def prepare_data(output):
    out,settings,cfg,rules=_load(output)
    train,validation,development=_development(out,settings,rules)
    path=out/'study_split_manifest.json';specs=settings['study']['policy_splits']
    if path.exists():
        saved=_read(path)
        if saved['plan_digest']!=_hash(specs) or saved['rules_digest']!=rules.digest():
            raise ValueError('policy split plan changed')
    else:
        _ensure_development_open(out)
        history=historical_operator_index(settings.get('historical_roots') or [str(out.parent)],exclude=[out])
        forbidden=set(history['normalized_operator_digests'])|_digests(development)|_digests(_read(out/'calibration_manifest.json'))
        records={}
        for name in ('policy_fit','policy_tune','policy_validation'):
            spec=specs[name]
            _,records[name],_=_generate_split(name,spec,forbidden,rules)
        saved=dict(version=VERSION,plan_digest=_hash(specs),rules_digest=rules.digest(),splits=records,
                   never_expert_training=True,normalized_operator_disjoint=True,final_materialized=False)
        _write_json(path,saved,exclusive=True)
    seen=_digests(development['splits'])|_digests(_read(out/'calibration_manifest.json'))
    for records in saved['splits'].values():
        for row in records:
            group=row['normalized_operator_digest']
            if group in seen:raise ValueError('study split overlap')
            seen.add(group)
    result={'train':train,'architecture_validation':validation}
    result.update({name:_restore(rows,rules) for name,rows in saved['splits'].items()})
    return result


def variant_config(cfg,spec):
    changes=spec.get('solver',{})
    allowed={'smoother_levels','transfer_levels','replace_pre','replace_post',
             'replacement_group_pre','replacement_group_post'}
    if set(changes)-allowed:raise ValueError('variant may change only the declared learned schedule')
    return replace(cfg,**changes)


def _initial(settings,spec):
    model=create_research_components(smoother='student_cnn',transfer='small_gnn',
        smoother_hidden=int(settings['hidden']),transfer_hidden=int(settings['hidden']),
        support=settings['support'],complexity_caps=settings['complexity_caps'],seed=int(settings['seed']))
    if spec['kind']!='current':
        model.smoother=make_multistage(kind=spec['kind'],stages=int(spec.get('stages',1)),
            hidden=int(settings['hidden']),level_count=int(spec.get('level_count',1)))
    model.mark_policy_stale('warm study expert must precede policy fitting')
    return model


def train_variants(output,names,*,resume=False):
    out,settings,cfg,rules=_load(output);_ensure_development_open(out)
    if (out/'expert_selection.json').exists():raise ValueError('expert already selected; use a new study to retrain')
    data=prepare_data(out);specs=settings['study']['variants']
    for name in names:
        if name not in specs:raise ValueError('unknown variant '+name)
        spec=specs[name];chosen=variant_config(cfg,spec)
        train_settings={**settings['training'],**spec.get('training',{})}
        model,status=train_expert(_initial(settings,spec),data['train'],chosen,rules,train_settings,
                                  out/'experts'/name,branch='H_S',resume=resume)
        if not status['all_updates_completed']:
            raise RuntimeError('skipped updates in '+name+'; inspect status before benchmarking')
        print(name,'updates=',status['updates'],'parameters=',status['parameters']['smoother'],flush=True)


def _expert(out,settings,rules,name):
    path=out/'experts'/name/'candidate.pt';status=_read(path.parent/'status.json')
    if not status.get('all_updates_completed') or digest_file(path)!=status['checkpoint_sha256']:
        raise ValueError('incomplete or modified expert')
    model=Components.load(path)
    if model.metadata.get('training_rules_digest')!=rules.digest() or model.metadata.get('training_branch')!='H_S':
        raise ValueError('expert trained against a different parent/branch')
    return model.frozen_inference_copy()


def expert_arms(out,settings,cfg,rules,names):
    arms={'fixed_C':dict(model=None,branch='C',selector=False),
          'strong_C':dict(model=None,branch='C',selector=True)}
    for name in names:
        spec=settings['study']['variants'][name]
        arms[name]=dict(model=_expert(out,settings,rules,name),branch='H_S',
                       config=variant_config(cfg,spec),schedule_ablation=True)
    return arms


def benchmark_experts(output,names,*,tag='architecture',resume=False,**protocol):
    out,settings,cfg,rules=_load(output);_ensure_development_open(out)
    if (out/'expert_selection.json').exists():raise ValueError('architecture selected; do not retune against policy data')
    examples=prepare_data(out)['architecture_validation']
    target=out/'benchmarks'/_tag(tag)
    report=evaluate_research(examples,expert_arms(out,settings,cfg,rules,names),cfg,rules,target,
                            resume=resume,**protocol)
    _report(report,target)
    return report


def select_expert(output,*,tag='architecture',name=None):
    out,settings,cfg,rules=_load(output);_ensure_development_open(out)
    target=out/'benchmarks'/_tag(tag);report=_read(target/'comparison.json');manifest=_read(target/'run_manifest.json')
    if _read(target/'progress.json')['status']!='complete':raise ValueError('complete architecture evidence first')
    candidates=[]
    for arm,metadata in manifest['arms'].items():
        if arm not in settings['study']['variants']:continue
        rows=[r for r in report['table'] if r['arm']==arm and r['regime'] in ('warm','warm_multiple')]
        baseline={ (r['case'],r['regime'],r['rhs_count']):r for r in report['table'] if r['arm']=='strong_C'}
        losses=[r for r in rows if baseline[r['case'],r['regime'],r['rhs_count']]['successful'] and not r['successful']]
        ratios=[r['speedup_vs_strong'] for r in rows if r['speedup_vs_strong'] is not None]
        if not losses and ratios:
            candidates.append((float(np.exp(np.mean(np.log(ratios)))),arm))
    if not candidates:raise ValueError('no no-new-failure expert with measured warm evidence')
    if name is None:name=max(candidates)[1]
    if name not in {v[1] for v in candidates}:raise ValueError('selected expert lost baseline successes or lacks warm evidence')
    model=_expert(out,settings,rules,name)
    if manifest['arms'][name]['model_signature']!=model.signature():raise ValueError('expert changed after benchmark')
    chosen=variant_config(cfg,settings['study']['variants'][name])
    value=dict(version=VERSION,variant=name,checkpoint=str(out/'experts'/name/'candidate.pt'),
        checkpoint_sha256=digest_file(out/'experts'/name/'candidate.pt'),expert_signature=model.signature(),
        solver=chosen.to_dict(),rules_digest=rules.digest(),selection_evidence=str(target/'comparison.json'),
        selection_evidence_sha256=digest_file(target/'comparison.json'),
        observed_warm_score=dict((n,s) for s,n in candidates)[name],
        selection_scope='development architecture selection; not a performance certificate')
    _write_json(out/'expert_selection.json',value,exclusive=True)
    print('Expert fixed:',name,'; policy fitting may now start.',flush=True)
    return value


def _selected(out,settings,cfg,rules):
    value=_read(out/'expert_selection.json')
    if digest_file(value['checkpoint'])!=value['checkpoint_sha256'] or rules.digest()!=value['rules_digest']:
        raise ValueError('selected expert or rules changed')
    if digest_file(value['selection_evidence'])!=value['selection_evidence_sha256']:
        raise ValueError('architecture evidence changed')
    expert=_expert(out,settings,rules,value['variant'])
    chosen=AdaptiveConfig.from_dict(value['solver'])
    if chosen!=variant_config(cfg,settings['study']['variants'][value['variant']]) or expert.signature()!=value['expert_signature']:
        raise ValueError('selected schedule/model changed; refit in a new run')
    return value,expert,chosen


def _labels(report,examples):
    mapping={e.group_digest:e for e in examples};labels=[]
    for row in report['rows']:
        group=row['example']['normalized_operator_digest'];e=mapping[group]
        for regime,counts in row['runs']['strong_C'].items():
            for k,cruns in counts.items():
                hruns=row['runs']['H_S'][regime][k]
                labels.append(dict(operator=group,split=e.research_split,
                    context=context_from_record(e,cruns[0],int(k),regime in ('warm','warm_multiple')),
                    C_success=all(r['successful'] for r in cruns),H_success=all(r['successful'] for r in hruns),
                    C_seconds=float(np.median([r['wall_seconds'] for r in cruns])),
                    H_seconds=float(np.median([r['wall_seconds'] for r in hruns])),
                    neural_used=any(r['actual_neural_used'] for r in hruns)))
    return labels


def fit_policy(output,*,resume=False):
    out,settings,cfg,rules=_load(output);_ensure_development_open(out)
    selected,expert,chosen=_selected(out,settings,cfg,rules);data=prepare_data(out)
    ps=settings['study']['policy'];protocol=ps['measurement']
    arms={'strong_C':dict(model=None,branch='C'), 'H_S':dict(model=expert,branch='H_S')}
    labels={};evidence={}
    for split in ('policy_fit','policy_tune'):
        target=out/'policy'/split
        report=evaluate_research(data[split],arms,chosen,rules,target,
                                resume=resume and (target/'run_manifest.json').exists(),**protocol)
        labels[split]=_labels(report,data[split]);evidence[split]=digest_file(target/'comparison.json')
    fitted=fit_cost_model(labels['policy_fit'],labels['policy_tune'],settings=ps)
    policy=ContinuousCostPolicy(fitted,expert,rules.digest(),config_scope(chosen),hardware_environment(),
                                expert.signature(),dict(expert_selection_sha256=digest_file(out/'expert_selection.json'),
                                measured_evidence=evidence,labels_scope='actual independent-RHS batches; no repeated-RHS extrapolated labels'))
    path=out/'policy/policy.json'
    if path.exists() and _read(path)!=policy.to_dict():raise ValueError('existing policy differs; use a new study')
    policy.save(path)
    write_json(out/'policy/labels.json',labels)
    print('Policy saved:',path,'; it is still a development candidate.',flush=True)
    return policy


def _policy(out,expert,probes=0):
    policy=ContinuousCostPolicy.load(out/'policy/policy.json',expert)
    if probes not in (0,1,2):raise ValueError('probe count must be 0,1,2')
    policy=replace(policy,probe_cycles=probes)
    return policy


def policy_arms(out,expert,probes):
    arms={'strong_C':dict(model=None,branch='C'), 'H_S':dict(model=expert,branch='H_S')}
    for probe in probes:
        name='adaptive' if probe==0 else 'adaptive_probe'+str(probe)
        arms[name]=dict(model=None,branch='auto',policy=_policy(out,expert,probe))
    return arms


def policy_coverage(report):
    """Descriptive decisions by grid/branch/workload; repeats are not samples."""
    rows=[]
    for record in report['rows']:
        for arm,regimes in record['runs'].items():
            if not arm.startswith('adaptive'):continue
            for regime,counts in regimes.items():
                for rhs,runs in counts.items():
                    for repeat,run in enumerate(runs):
                        for index,result in enumerate(run['rhs_results']):
                            info=result.get('abstention',{});ctx=info.get('cost_policy_context',{})
                            detail=info.get('cost_policy',{})
                            rows.append(dict(operator=record['example']['normalized_operator_digest'],arm=arm,
                                regime=regime,rhs_count=int(rhs),repeat=repeat,rhs_index=index,
                                N=ctx.get('N'),strategy=ctx.get('strategy'),
                                chosen_branch=info.get('chosen_branch'),reason=detail.get('reason'),
                                size_extrapolation=detail.get('size_extrapolation'),
                                classical_coverage_fallback=ctx.get('classical_coverage_fallback'),
                                success=result['verified_success'],certificate=False))
    return dict(rows=rows,scope='decision diagnostics; repeats/RHS are NOT independent operators')


def validate_policy(output,*,probes=(0,),tag='policy_validation',resume=False,**protocol):
    out,settings,cfg,rules=_load(output);_ensure_development_open(out)
    _,expert,chosen=_selected(out,settings,cfg,rules)
    data=prepare_data(out)
    target=out/'benchmarks'/_tag(tag)
    report=evaluate_research(data['policy_validation'],policy_arms(out,expert,probes),chosen,rules,target,
                            resume=resume,**protocol)
    _report(report,target)
    write_json(target/'policy_coverage.json',policy_coverage(report))
    for probe in probes:
        policy=_policy(out,expert,probe);path=out/'policy'/f'validated_probe{probe}.json'
        policy.save(path)
    write_json(out/'policy_validation_evidence.json',dict(tag=tag,probes=list(probes),
        report=str(target/'comparison.json'),sha256=digest_file(target/'comparison.json'),
        expert_selection_sha256=digest_file(out/'expert_selection.json'),
        policy_sha256=digest_file(out/'policy/policy.json'),performance_certified=False))
    return report


def freeze(output,*,probes=0,**protocol):
    out,settings,cfg,rules=_load(output)
    selected,expert,chosen=_selected(out,settings,cfg,rules)
    evidence=_read(out/'policy_validation_evidence.json')
    if probes not in evidence['probes'] or digest_file(evidence['report'])!=evidence['sha256']:
        raise ValueError('selected policy needs completed unchanged independent policy-validation')
    if digest_file(out/'policy/policy.json')!=evidence['policy_sha256'] or digest_file(out/'expert_selection.json')!=evidence['expert_selection_sha256']:
        raise ValueError('expert/policy changed after independent validation')
    validation=_read(evidence['report'])
    name='adaptive' if probes==0 else 'adaptive_probe'+str(probes)
    if not any(name in arms for counts in validation['summary'].values() for arms in counts.values()):
        raise ValueError('chosen policy absent from independent validation')
    _policy(out,expert,probes)
    evaluation=dict(version=VERSION,variant=selected['variant'],probes=probes,
                    solver=chosen.to_dict(),**protocol)
    paths={'expert_selection':out/'expert_selection.json','policy':out/'policy/policy.json',
           'policy_validation':out/'policy_validation_evidence.json','split_manifest':out/'study_split_manifest.json',
           'independent_evidence':evidence['report'],'architecture_evidence':selected['selection_evidence']}
    path=freeze_research(out/'data',{'H_S':selected['checkpoint']},out/'selector_rules.json',paths,_sources(),evaluation)
    print('Frozen:',path,'; no final/OOD operators have been inspected.',flush=True)


def final(output,*,resume=False):
    out,settings,cfg,rules=_load(output)
    _,expert,chosen=_selected(out,settings,cfg,rules)
    path=out/'data/research_freeze.json';evaluation=_read(path)['config']
    if evaluation.get('version')!=VERSION or AdaptiveConfig.from_dict(evaluation['solver'])!=chosen:
        raise ValueError('wrong frozen experiment')
    with claim_final_evaluation(out/'data',path,evaluation,resume=resume) as claim:
        splits,_=materialize_final_data(out/'data',claim,rules)
        forbidden=_digests(_read(out/'study_split_manifest.json'))
        paths=[]
        for split,examples in splits.items():
            if any(e.group_digest in forbidden for e in examples):raise ValueError('final/policy data overlap')
            target=out/'final'/split
            kwargs={k:evaluation[k] for k in ('repeats','warmups','rhs_counts','regimes')}
            report=evaluate_research(examples,policy_arms(out,expert,[evaluation['probes']]),chosen,rules,target,
                                    resume=resume and (target/'run_manifest.json').exists(),**kwargs)
            _report(report,target);write_json(target/'policy_coverage.json',policy_coverage(report))
            paths.append(target/'raw_results.json')
        complete_final_evaluation(out/'data',claim,paths)
    print('Final sealed. Inspect speed, failures AND neural coverage; completion is not superiority.',flush=True)


def diagnose(source_run,case_name,output,*,rhs_count=64,rhs_index=None,repeats=3):
    """Replay the exact declared development A/RHS; never pick an easier demo."""
    target=Path(output)
    if target.exists():raise FileExistsError(target)
    source=Path(source_run).resolve();settings=_read(source/'configuration.json')
    cfg=AdaptiveConfig.from_dict(settings['solver']);rules=load_strong_rules(source/'selector_rules.json')
    manifest=_read(source/'data/development_manifest.json')
    records=[r for values in manifest['splits'].values() for r in values if r['name']==case_name]
    if len(records)!=1:raise ValueError('case is not uniquely present in the declared development manifest')
    e=_restore(records,rules)[0]
    rhs,exacts=manufactured_rhs(e,rhs_count)
    indices=range(rhs_count) if rhs_index is None else [rhs_index]
    if any(not 0<=i<rhs_count for i in indices) or repeats<1:raise ValueError('invalid RHS/repeat selection')
    rows=[]
    for strategy in classical_bank('controlled'):
        name=strategy.name if hasattr(strategy,'name') else str(strategy)
        for index in indices:
            for repeat in range(repeats):
                chosen=replace(cfg,mode='classical',branch='C',mg=replace(cfg.mg,strategy_name=name))
                start=perf_counter();error=None
                try:
                    solver=PreparedAdaptiveMG(e.a,e.n,None,chosen);result=solver.solve(rhs[index])
                    elapsed=perf_counter()-start;true=stable_norm(rhs[index]-e.a@result.x)
                    good=bool(result.converged and np.isfinite(true) and true<=result.stopping_threshold)
                    cycles=result.executed_cycles
                except (ValueError,RuntimeError,FloatingPointError) as exc:
                    elapsed=perf_counter()-start;good=False;true=None;cycles=None;error=str(exc)
                rows.append(dict(strategy=name,rhs_index=index,repeat=repeat,success=good,
                                 true_residual=true,executed_cycles=cycles,wall_seconds=elapsed,error=error))
        print('[diagnostic]',name,flush=True)
    report=dict(version=VERSION,case=case_name,operator_digest=e.digest,
        rhs_digests=[hashlib.sha256(rhs[i].tobytes()).hexdigest() for i in indices],rows=rows,
        source_manifest_sha256=digest_file(source/'data/development_manifest.json'),
        numerical_protocol=cfg.mg.to_dict(),scope='offline portfolio diagnostic, NOT deployed selection; no rule updates')
    target=Path(output)
    if target.exists():raise FileExistsError('diagnostic output already exists')
    write_json(target,report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('calibrate','prepare','train','benchmark','select','policy-fit','policy-validate','freeze','final'):
        p=sub.add_parser(name);p.add_argument('--run-dir',required=True)
        if name in ('calibrate','train','benchmark','policy-fit','policy-validate','final'):p.add_argument('--resume',action='store_true')
        if name=='calibrate':p.add_argument('--config',default='configs/v6_7_warm_study_smoke.json')
        if name in ('train','benchmark'):p.add_argument('--variants',nargs='+',default=['H0','H1','H2','H2_NH'])
        if name in ('benchmark','policy-validate','freeze'):
            p.add_argument('--repeats',type=int,default=5);p.add_argument('--warmups',type=int,default=1)
            p.add_argument('--rhs-counts',nargs='+',type=int,default=[1,4,16,64])
            p.add_argument('--regimes',nargs='+',choices=REGIMES,default=['warm_multiple'])
        if name in ('benchmark','select'):p.add_argument('--tag',default='architecture')
        if name=='select':p.add_argument('--variant')
        if name=='policy-validate':
            p.add_argument('--tag',default='policy_validation');p.add_argument('--probes',nargs='+',type=int,default=[0])
        if name=='freeze':p.add_argument('--probes',type=int,default=0)
    p=sub.add_parser('diagnose');p.add_argument('--source-run',required=True);p.add_argument('--case',required=True)
    p.add_argument('--output',required=True);p.add_argument('--rhs-count',type=int,default=64)
    p.add_argument('--rhs-index',type=int);p.add_argument('--repeats',type=int,default=3)
    a=parser.parse_args(argv)
    if a.command=='diagnose':return diagnose(a.source_run,a.case,a.output,rhs_count=a.rhs_count,rhs_index=a.rhs_index,repeats=a.repeats)
    if a.command=='calibrate':calibrate(a.config,a.run_dir,resume=a.resume);return prepare_data(a.run_dir)
    if a.command=='prepare':return prepare_data(a.run_dir)
    if a.command=='train':return train_variants(a.run_dir,a.variants,resume=a.resume)
    if a.command=='select':return select_expert(a.run_dir,tag=a.tag,name=a.variant)
    if a.command=='policy-fit':return fit_policy(a.run_dir,resume=a.resume)
    if a.command=='final':return final(a.run_dir,resume=a.resume)
    kw=dict(repeats=a.repeats,warmups=a.warmups,rhs_counts=a.rhs_counts,regimes=a.regimes)
    if a.command=='benchmark':return benchmark_experts(a.run_dir,a.variants,tag=a.tag,resume=a.resume,**kw)
    if a.command=='policy-validate':return validate_policy(a.run_dir,probes=a.probes,tag=a.tag,resume=a.resume,**kw)
    return freeze(a.run_dir,probes=a.probes,**kw)
