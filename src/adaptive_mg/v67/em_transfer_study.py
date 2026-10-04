"""EM/schedule -> all-level affine P -> independent cost policy workflow.

Reuses immutable operator-disjoint splits, safeguarded measurement, and final
sealing from warm_study. Existing H_S architectures/policies remain unchanged.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from ..provenance import write_json,hardware_environment
from .config import AdaptiveConfig
from .models import Components
from .strong import load_strong_rules
from .limited import digest_file
from .three_pillars import calibrate,_load_run,_ensure_development_open,_read,_report,_sources
from .warm_study import prepare_data,policy_coverage,_tag
from .hp_training import train_asymptotic
from .research_training import create_research_components
from .research_transfer import make_graph_transfer
from .research_evaluation import evaluate_research,REGIMES
from .cost_policy import ContinuousCostPolicy,fit_cost_model,config_scope,context_from_record
from .research_data import freeze_research,claim_final_evaluation,materialize_final_data,complete_final_evaluation,_digests

VERSION='em-affine-hp-study-v1'


def load(output):
    out,settings,cfg=_load_run(output)
    if settings.get('em_hp_version')!=VERSION:raise ValueError('use a NEW EM transfer run/config')
    if settings.get('calibration_regime')!='warm_multiple' or settings.get('bank')!='em_schedule':
        raise ValueError('EM study requires the warm EM/schedule bank')
    if cfg.transfer_levels!='all':raise ValueError('EM study requires every nonterminal transfer level')
    return out,settings,cfg,load_strong_rules(out/'selector_rules.json')


def initial_model(settings):
    model=create_research_components(smoother='student_cnn',transfer='small_gnn',
        smoother_hidden=int(settings['hidden']),transfer_hidden=int(settings['hidden']),
        support='support_preserving',complexity_caps=settings['complexity_caps'],seed=int(settings['seed']))
    model.transfer=make_graph_transfer('small_gnn',width=int(settings['hidden']),support='support_preserving',
        parameterization='affine',reference='frozen_parent',support_only=True,complexity_caps=settings['complexity_caps'])
    return model


def train(output,*,resume=False,max_updates=None):
    out,settings,cfg,rules=load(output);_ensure_development_open(out)
    if (out/'hp_selection.json').exists():raise ValueError('expert fixed: use a NEW run for retraining')
    data=prepare_data(out)
    return train_asymptotic(initial_model(settings),data['train'],data['architecture_validation'],cfg,rules,
                           settings['training'],out/'experts/H_P',resume=resume,max_updates=max_updates)


def expert(out,rules):
    path=out/'experts/H_P/candidate.pt';status=_read(path.parent/'status.json')
    if not status['all_updates_completed'] or digest_file(path)!=status['checkpoint_sha256']:
        raise ValueError('incomplete/changed selected numerical checkpoint')
    model=Components.load(path)
    if model.metadata.get('training_rules_digest')!=rules.digest() or model.metadata.get('training_branch')!='H_P':
        raise ValueError('checkpoint belongs to different parent/branch')
    return model.frozen_inference_copy()


def arms(model,policy=None):
    values={'fixed_C':dict(model=None,branch='C',selector=False),
            'strong_C':dict(model=None,branch='C'), 'H_P':dict(model=model,branch='H_P')}
    if policy is not None:values['adaptive_P']=dict(model=None,branch='auto',policy=policy)
    return values


def benchmark(output,*,tag='hp_validation',resume=False,**protocol):
    out,settings,cfg,rules=load(output);_ensure_development_open(out)
    if (out/'hp_selection.json').exists():raise ValueError('expert selected; use policy-validation instead')
    data=prepare_data(out);target=out/'benchmarks'/_tag(tag)
    result=evaluate_research(data['architecture_validation'],arms(expert(out,rules)),cfg,rules,target,resume=resume,**protocol)
    _report(result,target);return result


def select(output,tag='hp_validation'):
    out,settings,cfg,rules=load(output);_ensure_development_open(out)
    target=out/'benchmarks'/_tag(tag);comparison=_read(target/'comparison.json');manifest=_read(target/'run_manifest.json')
    if _read(target/'progress.json')['status']!='complete':raise ValueError('complete numerical validation required')
    model=expert(out,rules)
    if manifest['arms']['H_P']['model_signature']!=model.signature():raise ValueError('expert changed after benchmark')
    selected=[a['H_P'] for regime,counts in comparison['summary'].items() if regime in ('warm','warm_multiple') for a in counts.values()]
    if not selected or any(a['new_failure_case_ids'] for a in selected):raise ValueError('no warm evidence or new baseline failure')
    value=dict(version=VERSION,expert_signature=model.signature(),checkpoint_sha256=digest_file(out/'experts/H_P/candidate.pt'),
        rules_digest=rules.digest(),config=cfg.to_dict(),evidence=str(target/'comparison.json'),
        evidence_sha256=digest_file(target/'comparison.json'),performance_certified=False)
    from .research_data import _write_json
    _write_json(out/'hp_selection.json',value,exclusive=True)
    print('H_P fixed. Slower than C is allowed: cost policy may abstain.',flush=True)
    return value


def selected(out,cfg,rules):
    record=_read(out/'hp_selection.json');model=expert(out,rules)
    if record['expert_signature']!=model.signature() or record['rules_digest']!=rules.digest() or AdaptiveConfig.from_dict(record['config'])!=cfg:
        raise ValueError('selected expert/parent/config changed')
    if digest_file(record['evidence'])!=record['evidence_sha256'] or digest_file(out/'experts/H_P/candidate.pt')!=record['checkpoint_sha256']:
        raise ValueError('selected inputs changed')
    return model


def labels(report,examples):
    mapping={e.group_digest:e for e in examples};result=[]
    for row in report['rows']:
        key=row['example']['normalized_operator_digest'];e=mapping[key]
        for regime,counts in row['runs']['strong_C'].items():
            for k,cr in counts.items():
                hr=row['runs']['H_P'][regime][k]
                result.append(dict(operator=key,context=context_from_record(e,cr[0],int(k),regime in ('warm','warm_multiple')),
                    C_success=all(r['successful'] for r in cr),H_success=all(r['successful'] for r in hr),
                    C_seconds=float(np.median([r['wall_seconds'] for r in cr])),H_seconds=float(np.median([r['wall_seconds'] for r in hr])),
                    neural_used=any(r['actual_neural_used'] for r in hr)))
    return result


def fit_policy(output,*,resume=False):
    out,settings,cfg,rules=load(output);_ensure_development_open(out)
    if (out/'hp_policy_validation.json').exists():raise ValueError('policy independently inspected: do not retune in this run')
    model=selected(out,cfg,rules);data=prepare_data(out);values={};evidence={};ps=settings['study']['policy']
    for split in ('policy_fit','policy_tune'):
        target=out/'hp_policy'/split
        result=evaluate_research(data[split],arms(model),cfg,rules,target,
                   resume=resume and (target/'run_manifest.json').exists(),**ps['measurement'])
        values[split]=labels(result,data[split]);evidence[split]=digest_file(target/'comparison.json')
    fitted=fit_cost_model(values['policy_fit'],values['policy_tune'],settings=ps)
    policy=ContinuousCostPolicy(fitted,model,rules.digest(),config_scope(cfg,plan_selection=True),hardware_environment(),
                   model.signature(),dict(selection=digest_file(out/'hp_selection.json'),measurements=evidence),branch='H_P')
    target=out/'hp_policy/policy.json'
    if target.exists() and _read(target)!=policy.to_dict():raise ValueError('policy changed; use a new run')
    policy.save(target);write_json(out/'hp_policy/labels.json',values)
    print('C/H_P empirical policy saved; no speed guarantee.',flush=True)
    return policy


def policy_validation(output,*,resume=False,**protocol):
    out,settings,cfg,rules=load(output);_ensure_development_open(out)
    model=selected(out,cfg,rules);policy=ContinuousCostPolicy.load(out/'hp_policy/policy.json',model)
    data=prepare_data(out);target=out/'benchmarks/hp_policy_validation'
    result=evaluate_research(data['policy_validation'],arms(model,policy),cfg,rules,target,resume=resume,**protocol)
    _report(result,target);write_json(target/'policy_coverage.json',policy_coverage(result))
    write_json(out/'hp_policy_validation.json',dict(report=str(target/'comparison.json'),report_sha256=digest_file(target/'comparison.json'),
        policy_sha256=digest_file(out/'hp_policy/policy.json'),selection_sha256=digest_file(out/'hp_selection.json'),performance_certified=False))
    return result


def freeze(output,**protocol):
    out,settings,cfg,rules=load(output);selected(out,cfg,rules)
    ev=_read(out/'hp_policy_validation.json')
    if ev['policy_sha256']!=digest_file(out/'hp_policy/policy.json') or ev['report_sha256']!=digest_file(ev['report']) or ev['selection_sha256']!=digest_file(out/'hp_selection.json'):
        raise ValueError('independently validated evidence changed')
    config=dict(version=VERSION,solver=cfg.to_dict(),**protocol)
    selection=_read(out/'hp_selection.json')
    paths={'policy':out/'hp_policy/policy.json','selection':out/'hp_selection.json',
           'policy_validation':out/'hp_policy_validation.json','policy_report':ev['report'],
           'architecture_evidence':selection['evidence'],'splits':out/'study_split_manifest.json'}
    return freeze_research(out/'data',{'H_P':out/'experts/H_P/candidate.pt'},out/'selector_rules.json',paths,_sources(),config)


def final(output,*,resume=False):
    out,settings,cfg,rules=load(output);model=selected(out,cfg,rules)
    freeze_path=out/'data/research_freeze.json';evaluation=_read(freeze_path)['config']
    if evaluation['version']!=VERSION or AdaptiveConfig.from_dict(evaluation['solver'])!=cfg:raise ValueError('wrong frozen study')
    policy=ContinuousCostPolicy.load(out/'hp_policy/policy.json',model)
    with claim_final_evaluation(out/'data',freeze_path,evaluation,resume=resume) as claim:
        splits,_=materialize_final_data(out/'data',claim,rules);outputs=[]
        forbidden=_digests(_read(out/'study_split_manifest.json'))
        for name,examples in splits.items():
            if any(e.group_digest in forbidden for e in examples):raise ValueError('policy/final operator overlap')
            target=out/'final'/name
            report=evaluate_research(examples,arms(model,policy),cfg,rules,target,
                resume=resume and (target/'run_manifest.json').exists(),**{k:evaluation[k] for k in ('repeats','warmups','regimes','rhs_counts')})
            _report(report,target);write_json(target/'policy_coverage.json',policy_coverage(report));outputs.append(target/'raw_results.json')
        complete_final_evaluation(out/'data',claim,outputs)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('calibrate','train','benchmark','select','policy-fit','policy-validate','freeze','final'):
        p=sub.add_parser(name);p.add_argument('--run-dir',required=True)
        if name=='calibrate':p.add_argument('--config',default='configs/v6_7_em_hp_smoke.json')
        if name in ('calibrate','train','benchmark','policy-fit','policy-validate','final'):p.add_argument('--resume',action='store_true')
        if name=='train':p.add_argument('--max-updates',type=int)
        if name in ('benchmark','select'):p.add_argument('--tag',default='hp_validation')
        if name in ('benchmark','policy-validate','freeze'):
            p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmups',type=int,default=1)
            p.add_argument('--rhs-counts',nargs='+',type=int,default=[1,4,16,64])
            p.add_argument('--regimes',nargs='+',choices=REGIMES,default=['warm_multiple','multiple'])
    args=parser.parse_args(argv);out=args.run_dir
    if args.command=='calibrate':calibrate(args.config,out,resume=args.resume);return prepare_data(Path(out).resolve())
    if args.command=='train':return train(out,resume=args.resume,max_updates=args.max_updates)
    if args.command=='select':return select(out,args.tag)
    if args.command=='policy-fit':return fit_policy(out,resume=args.resume)
    if args.command=='final':return final(out,resume=args.resume)
    kw={k:getattr(args,k) for k in ('repeats','warmups','rhs_counts','regimes')}
    if args.command=='benchmark':return benchmark(out,tag=args.tag,resume=args.resume,**kw)
    if args.command=='policy-validate':return policy_validation(out,resume=args.resume,**kw)
    return freeze(out,**kw)
