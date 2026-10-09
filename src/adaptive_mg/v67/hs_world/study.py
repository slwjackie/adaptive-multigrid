"""Primary thesis workflow: C_tuned -> H_S -> sequence world model.

H_P/H_SP stay in the repository as historical ablations, never as active
experts here. Synthetic experiments do not establish combustion CFD results.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np

from ..research_data import FAMILIES, _generate_split, _restore, _digests, historical_operator_index
from ..research_evaluation import evaluate_research, REGIMES
from ..research_training import train_expert
from ..warm_study import _initial, variant_config, _tag
from ..models import Components
from ..config import AdaptiveConfig
from ..strong import load_strong_rules
from ..world_model.data import write_json, file_hash, digest
from .tuning import VERSION, load, read, assert_open, calibrate, strong_audit


def data(output):
    out,settings,cfg,rules=load(output)
    path=out/'static_data_manifest.json'
    if not path.exists():
        assert_open(out)
        roots=settings.get('historical_roots') or [str(out.parent)]
        history=historical_operator_index(roots,exclude=[out])
        excluded=set(history['normalized_operator_digests'])|_digests(read(out/'calibration_manifest.json'))
        splits={}
        for i,(name,key,per) in enumerate((('train','training_sizes','training_per_family'),
                                         ('validation','validation_sizes','validation_per_family'))):
            spec=dict(sizes=settings[key],families=settings.get('families',list(FAMILIES)),
                      per_family=settings[per],seed=settings['seed']+700001+110001*i)
            _,splits[name],_=_generate_split(name,spec,excluded,rules)
        write_json(path,dict(splits=splits,rules_digest=rules.digest(),final_seen=False))
    saved=read(path)
    if saved['rules_digest']!=rules.digest():raise ValueError('static data parent changed')
    result={n:_restore(rows,rules) for n,rows in saved['splits'].items()}
    ids=[e.group_digest for rows in result.values() for e in rows]
    if len(ids)!=len(set(ids)) or set(ids)&_digests(read(out/'calibration_manifest.json')):
        raise ValueError('static/classical operator leakage')
    return result


def train(output,names,*,resume=False):
    out,settings,cfg,rules=load(output);assert_open(out)
    if (out/'expert_selection.json').exists():raise ValueError('expert selected; retraining needs a new run')
    examples=data(out)['train']
    for name in names:
        if name not in settings['study']['variants']:raise ValueError('unknown H_S variant')
        spec=settings['study']['variants'][name];chosen=variant_config(cfg,spec)
        if chosen.use_transfer or chosen.branch!='H_S':raise ValueError('H_S only')
        _,status=train_expert(_initial(settings,spec),examples,chosen,rules,
            {**settings['training'],**spec.get('training',{})},out/'experts'/name,branch='H_S',resume=resume)
        if not status['all_updates_completed']:raise RuntimeError('skipped updates; inspect training logs')


def expert(out,settings,rules,name):
    if name not in settings['study']['variants']:raise ValueError('unknown smoother')
    path=out/'experts'/name/'candidate.pt';status=read(path.parent/'status.json')
    if not status['all_updates_completed'] or status['checkpoint_sha256']!=file_hash(path):
        raise ValueError('incomplete/modified smoother')
    model=Components.load(path)
    if model.metadata.get('training_branch')!='H_S' or model.metadata.get('training_rules_digest')!=rules.digest():
        raise ValueError('H_S trained with a different fixed classical parent')
    if model.metadata.get('optimizer_updates',0)<1:raise ValueError('genuinely trained H_S required')
    return model.frozen_inference_copy()


def arms(out,settings,cfg,rules,names,*,audit=True):
    result={'C_tuned':dict(model=None,branch='C')}
    for name in names:
        result[name]=dict(model=expert(out,settings,rules,name),branch='H_S',
            config=variant_config(cfg,settings['study']['variants'][name]),schedule_ablation=True)
    if audit and (out/'strong_audit_rules.json').exists():
        result['strong_C']=dict(model=None,branch='C',rules=load_strong_rules(out/'strong_audit_rules.json'),robustness_only=True)
    return result


def benchmark(output,names,*,tag='architecture',resume=False,**protocol):
    out,settings,cfg,rules=load(output);assert_open(out)
    if (out/'expert_selection.json').exists():raise ValueError('expert already fixed; no architecture retuning')
    target=out/'benchmarks'/_tag(tag)
    report=evaluate_research(data(out)['validation'],arms(out,settings,cfg,rules,names),cfg,rules,target,
                            reference_arm='C_tuned',resume=resume,**protocol)
    for regime,counts in report['summary'].items():
        for rhs,methods in counts.items():
            for name,row in methods.items():
                print(regime,rhs,name,f"{row['successes']}/{row['total']}",
                      'speedup_vs_C_tuned=',row['geometric_speedup'],flush=True)
    return report


def select(output,*,tag='architecture',name=None):
    out,settings,cfg,rules=load(output);assert_open(out)
    destination=out/'expert_selection.json'
    if destination.exists():raise FileExistsError('expert is frozen; new run required to reselect')
    base=out/'benchmarks'/_tag(tag)
    report=read(base/'comparison.json');manifest=read(base/'run_manifest.json')
    if not read(base/'progress.json')['status']=='complete' or report.get('reference_arm')!='C_tuned':
        raise ValueError('completed C_tuned benchmark required')
    baseline={(r['case'],r['regime'],r['rhs_count']):r for r in report['table'] if r['arm']=='C_tuned'}
    scores={}
    for variant in settings['study']['variants']:
        rows=[r for r in report['table'] if r['arm']==variant and r['regime'] in ('warm','warm_multiple')]
        loss=any(baseline[r['case'],r['regime'],r['rhs_count']]['successful'] and not r['successful'] for r in rows)
        ratios=[r['speedup_vs_reference'] for r in rows if r['speedup_vs_reference'] is not None]
        neural=any(r['actual_neural_used'] for r in rows)
        if rows and not loss and ratios and neural:
            scores[variant]=float(np.exp(np.log(ratios).mean()))
    if not scores:raise ValueError('no real neural candidate preserving baseline successes')
    name=name or max(scores,key=scores.get)
    if name not in scores:raise ValueError('selected variant is not eligible')
    model=expert(out,settings,rules,name)
    if model.signature()!=manifest['arms'][name]['model_signature']:raise ValueError('expert changed after benchmark')
    chosen=variant_config(cfg,settings['study']['variants'][name])
    record=dict(version=VERSION,variant=name,solver=chosen.to_dict(),rules_digest=rules.digest(),
        checkpoint=str(out/'experts'/name/'candidate.pt'),checkpoint_sha256=file_hash(out/'experts'/name/'candidate.pt'),
        expert_signature=model.signature(),benchmark=str(base/'comparison.json'),benchmark_sha256=file_hash(base/'comparison.json'),
        static_data_sha256=file_hash(out/'static_data_manifest.json'),tuning_sha256=file_hash(out/'tuned_classical.json'),
        audit_rules_sha256=file_hash(out/'strong_audit_rules.json') if (out/'strong_audit_rules.json').exists() else None,
        primary_reference='C_tuned',observed_warm_speedup=scores[name],all_scores=scores,
        superior_in_observed_gm=scores[name]>1.,performance_certified=False)
    write_json(destination,record)
    print('Fixed H_S:',name,'observed warm speedup:',scores[name],'; not a speed certificate',flush=True)
    return record


def selected(output):
    out,settings,cfg,rules=load(output);s=read(out/'expert_selection.json')
    for path,key in ((s['checkpoint'],'checkpoint_sha256'),(s['benchmark'],'benchmark_sha256'),
                     (out/'static_data_manifest.json','static_data_sha256'),(out/'tuned_classical.json','tuning_sha256')):
        if file_hash(path)!=s[key]:raise ValueError('selected expert/evidence changed')
    if s['rules_digest']!=rules.digest():raise ValueError('selected parent changed')
    if s['audit_rules_sha256'] is not None and file_hash(out/'strong_audit_rules.json')!=s['audit_rules_sha256']:
        raise ValueError('strong audit changed after selection')
    model=expert(out,settings,rules,s['variant'])
    chosen=variant_config(cfg,settings['study']['variants'][s['variant']])
    if chosen!=AdaptiveConfig.from_dict(s['solver']) or model.signature()!=s['expert_signature']:
        raise ValueError('selected H_S schedule changed')
    return out,settings,chosen,rules,model,s


def static_test(output,*,repeats=5,warmups=1,rhs_counts=(1,4,16,64),regimes=('warm_multiple',),resume=False):
    from .temporal import check_freeze
    out,settings,cfg,rules,model,selection=selected(output);check_freeze(out)
    path=out/'static_test_manifest.json'
    if not path.exists():
        excluded=_digests(read(out/'static_data_manifest.json'))|_digests(read(out/'calibration_manifest.json'))
        if (out/'temporal/data/sequence_manifest.json').exists():
            excluded|={e['normalized_matrix_digest'] for tr in read(out/'temporal/data/sequence_manifest.json')['trajectories'] for e in tr['snapshots']}
        spec=dict(sizes=settings['thesis']['static_test_sizes'],families=settings.get('families',list(FAMILIES)),
                  per_family=settings['thesis']['static_test_per_family'],seed=settings['seed']+91000013)
        _,rows,_=_generate_split('hs_test',spec,excluded,rules)
        write_json(path,dict(splits={'hs_test':rows},rules_digest=rules.digest()))
    examples=_restore(read(path)['splits']['hs_test'],rules)
    return evaluate_research(examples,arms(out,settings,cfg,rules,[selection['variant']]),cfg,rules,
          out/'static_test',reference_arm='C_tuned',repeats=repeats,warmups=warmups,rhs_counts=rhs_counts,
          regimes=regimes,resume=resume)


def main(argv=None):
    from . import temporal
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for cmd in ('calibrate','train','benchmark','select','strong-audit','world-prepare','collect','world-train',
                'world-validate','freeze','test','gamg-compare'):
        p=sub.add_parser(cmd);p.add_argument('--run-dir',required=True)
        if cmd in ('calibrate','train','benchmark','strong-audit','collect','world-validate','test'):
            p.add_argument('--resume',action='store_true')
        if cmd=='calibrate':p.add_argument('--config',default='configs/v6_7_hs_world_smoke.json')
        if cmd in ('train','benchmark'):p.add_argument('--variants',nargs='+',default=['H0','H1','H2','H2_NH'])
        if cmd in ('benchmark','select'):p.add_argument('--tag',default='architecture')
        if cmd=='select':p.add_argument('--variant')
        if cmd in ('benchmark','test'):
            p.add_argument('--repeats',type=int,default=5);p.add_argument('--warmups',type=int,default=1)
            p.add_argument('--rhs-counts',type=int,nargs='+',default=[1,4,16,64])
            p.add_argument('--regimes',nargs='+',choices=REGIMES,default=['warm_multiple','multiple'])
        if cmd=='test':p.add_argument('--component',choices=['sequence','static'],default='sequence')
        if cmd=='world-prepare':p.add_argument('--input-ldu')
        if cmd=='world-validate':p.add_argument('--repeats',type=int,default=3)
        if cmd=='gamg-compare':
            p.add_argument('--reference',required=True);p.add_argument('--split',choices=['validation','test'],default='validation')
    a=parser.parse_args(argv);out=a.run_dir
    if a.command=='calibrate':return calibrate(a.config,out,resume=a.resume)
    if a.command=='train':return train(out,a.variants,resume=a.resume)
    if a.command=='strong-audit':return strong_audit(out,resume=a.resume)
    if a.command=='select':return select(out,tag=a.tag,name=a.variant)
    if a.command=='world-prepare':return temporal.prepare(out,input_ldu=a.input_ldu)
    if a.command=='collect':return temporal.collect(out,resume=a.resume)
    if a.command=='world-train':return temporal.train(out)
    if a.command=='freeze':return temporal.freeze(out)
    if a.command=='gamg-compare':
        from .cfd_compare import compare
        return compare(out,a.reference,split=a.split)
    if a.command=='world-validate':return temporal.evaluate(out,split='validation',repeats=a.repeats,resume=a.resume)
    if a.command=='test' and a.component=='sequence':return temporal.evaluate(out,split='test',repeats=a.repeats,resume=a.resume)
    kw=dict(repeats=a.repeats,warmups=a.warmups,rhs_counts=a.rhs_counts,regimes=a.regimes,resume=a.resume)
    if a.command=='test':return static_test(out,**kw)
    return benchmark(out,a.variants,tag=a.tag,**kw)
