"""From-scratch, reproducible workflow for the three-pillar solver fixes.

No historical checkpoints are required. Calibration and NN datasets are
operator-disjoint. Full research is opt-in; smoke never consumes final data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from ..provenance import hardware_environment, json_safe, write_json
from .config import AdaptiveConfig
from .models import Components
from .strong import StrongRules, PreparedStrongMG, load_strong_rules
from .strong_calibration import measure_classical_portfolio, calibrate_multisize
from .research_data import (FAMILIES, HOLDOUT, _generate_split, _restore, _hash,
    historical_operator_index, make_research_plan, prepare_development_data,
    load_development_data, freeze_research, claim_final_evaluation,
    materialize_final_data, complete_final_evaluation)
from .research_training import create_research_components, train_expert
from .research_evaluation import evaluate_research
from .limited import initialize_timing_runtime

VERSION='three-pillars-v1'
PROJECT=Path(__file__).resolve().parents[3]


def _read(path):
    return json.loads(Path(path).read_text())


def _sources():
    return sorted([p for p in (PROJECT/'src').rglob('*') if p.suffix in ('.py','.cpp')]
                  +[PROJECT/'scripts/run_v6_7_three_pillars.py'])


def _source_digest():
    return _hash({str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in _sources()})


def _validate_settings(settings):
    if settings.get('version')!=VERSION:
        raise ValueError('unsupported three-pillar config')
    cfg=AdaptiveConfig.from_dict(settings['solver'])
    if cfg.application!='replace' or not cfg.lazy or cfg.mode!='research':
        raise ValueError('primary workflow requires lazy replacement in research mode')
    if settings['support']!='support_preserving':
        raise ValueError('primary P training must use support_preserving')
    if settings['complexity_caps'].get('complexity_reference')!='parent':
        raise ValueError('primary P must use parent-relative complexity caps')
    for key in ('calibration_sizes','training_sizes','validation_sizes'):
        values=settings[key]
        if not values or len(set(values))!=len(values) or any(not isinstance(n,int) or n<3 or n&(n+1) for n in values):
            raise ValueError('sizes must be distinct 2**L-1 integers >=3')
    if len(settings['calibration_sizes'])<2:
        raise ValueError('calibration requires multiple sizes')
    return cfg


def _load_run(out):
    out=Path(out).resolve()
    settings=_read(out/'configuration.json');cfg=_validate_settings(settings)
    torch.set_num_threads(int(settings.get('torch_threads',1)))
    initialize_timing_runtime()
    manifest=_read(out/'three_pillars_manifest.json')
    if manifest['source_digest']!=_source_digest() or manifest['settings_digest']!=_hash(settings):
        raise ValueError('sources/settings changed; start a new run, do not reuse old evidence')
    if manifest['hardware']!=hardware_environment(refresh=True):
        raise ValueError('hardware/library/thread environment changed; start a new run')
    return out,settings,cfg


def _ensure_development_open(out):
    if (out/'data/research_freeze.json').exists() or (out/'data/final_claim.json').exists():
        raise ValueError('run already frozen; start a new development run')


def calibrate(config_path, output, *, resume=False):
    settings=_read(config_path);cfg=_validate_settings(settings)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(settings.get('torch_threads',1)))
    initialize_timing_runtime()
    header=dict(version=VERSION,source_digest=_source_digest(),settings_digest=_hash(settings),
                hardware=hardware_environment(refresh=True))
    path=out/'three_pillars_manifest.json'
    if path.exists():
        if not resume or _read(path)!=json_safe(header):
            raise ValueError('use --resume with identical sources/config/hardware, or a new run directory')
    else:
        if resume or any(out.iterdir()):raise FileExistsError('calibration output must be new/empty')
        write_json(out/'configuration.json',settings);write_json(path,header)
    _ensure_development_open(out)
    if (out/'selector_rules.json').exists():
        print('Calibration already complete:',out/'selector_rules.json',flush=True)
        return
    initial=StrongRules(require_coverage=True,fallback_strategy_name=cfg.mg.strategy_name)
    manifest_path=out/'calibration_manifest.json'
    if manifest_path.exists():
        # Resume from immutable recorded operators, not a rescanned history that
        # may have gained unrelated runs since this calibration was interrupted.
        saved=_read(manifest_path)
        if saved['settings_digest']!=_hash(settings) or saved['rules']!=json_safe(initial.to_dict()):
            raise ValueError('stale calibration manifest')
        groups={name:_restore(saved['splits'][name],initial)
                for name in ('selector_train','selector_tune')}
    else:
        roots=settings.get('historical_roots') or [str(out.parent)]
        history=historical_operator_index(roots,exclude=[out])
        forbidden=set(history['normalized_operator_digests'])
        groups={};spec_records={}
        for index,name in enumerate(('selector_train','selector_tune')):
            spec=dict(sizes=settings['calibration_sizes'],per_family=int(settings['calibration_per_family']),
                      families=list(FAMILIES),seed=int(settings['seed'])+100003*index)
            examples,records,_=_generate_split(name,spec,forbidden,initial)
            groups[name]=examples;spec_records[name]=records
        write_json(manifest_path,dict(rules=initial.to_dict(),historical_index=history,
                   splits=spec_records,settings_digest=_hash(settings),final_used=False))
    measured={}
    for name,examples in groups.items():
        rows=[]
        for index,e in enumerate(examples):
            target=out/'calibration_records'/name/(e.group_digest+'.json')
            request=_hash(dict(header=header,operator=e.digest,rules=initial.digest(),name=name))
            if target.exists():
                value=_read(target)
                if not resume or value['request_digest']!=request:
                    raise ValueError('incompatible partial calibration record')
                row=value['row']
            else:
                row=measure_classical_portfolio([e],cfg,initial,
                    repeats=int(settings['calibration_repeats']),rhs_count=int(settings['calibration_rhs']),
                    bank=settings.get('bank','controlled'),seed=int(settings['seed'])+index)[0]
                write_json(target,dict(request_digest=request,row=row))
            rows.append(row)
            print(f'[calibration {name}] {index+1}/{len(examples)} {e.name}',flush=True)
        measured[name]=rows
    rules,evidence=calibrate_multisize(measured['selector_train'],measured['selector_tune'],initial,
        required_sizes=settings['calibration_sizes'],fixed_strategy=cfg.mg.strategy_name,
        minimum_leaf_cases=int(settings['minimum_leaf_cases']),max_cycles=cfg.mg.max_cycles,
        cycle_margin=float(settings['cycle_margin']))
    write_json(out/'selector_evidence.json',evidence)
    write_json(out/'selector_rules.json',rules.to_dict())
    print('Frozen rules:',out/'selector_rules.json',flush=True)
    print('Admitted leaves:',len(rules.coverage_by_rule),'; other leaves/sizes use',rules.fallback_strategy_name,flush=True)


def _development(out,settings,rules):
    data=out/'data'
    if (data/'development_manifest.json').exists():
        return load_development_data(data,rules)
    plan=make_research_plan(smoke=bool(settings['smoke']),seed=int(settings['seed'])+700001)
    for name,key,per in (('train','training_sizes','training_per_family'),
                          ('validation','validation_sizes','validation_per_family')):
        spec=plan['splits'][name]
        spec.update(sizes=settings[key],per_family=int(settings[per]))
        spec['count']=len(spec['sizes'])*len(spec['families'])*spec['per_family']
    # Includes this run's calibration manifest: calibration and NN data cannot overlap.
    roots=settings.get('historical_roots') or [str(out.parent)]
    return prepare_development_data(data,rules,plan=plan,historical_roots=roots)


def train(output, branches, *, resume=False):
    out,settings,cfg=_load_run(output);_ensure_development_open(out)
    rules=load_strong_rules(out/'selector_rules.json')
    examples,_,_=_development(out,settings,rules)
    for branch in branches:
        if branch not in ('H_S','H_P','H_SP'):raise ValueError('unknown training branch')
        initial=create_research_components(smoother='student_cnn',transfer='small_gnn',
            smoother_hidden=int(settings['hidden']),transfer_hidden=int(settings['hidden']),
            support=settings['support'],complexity_caps=settings['complexity_caps'],seed=int(settings['seed']))
        if branch=='H_SP':
            initial.smoother=Components.load(out/'checkpoints/H_S/candidate.pt').smoother
            initial.transfer=Components.load(out/'checkpoints/H_P/candidate.pt').transfer
            initial.mark_policy_stale('joint fine-tuning from independent strong-aware experts')
        print('[train]',branch,'operators=',len(examples),'updates=',settings['training']['updates'],flush=True)
        _,status=train_expert(initial,examples,cfg,rules,settings['training'],out/'checkpoints'/branch,
                              branch=branch,resume=resume)
        print(branch,status['updates'],'updates; checkpoint:',status['checkpoint'],flush=True)
        if not status['all_updates_completed']:
            raise RuntimeError('some optimizer updates were skipped; inspect status.json before benchmarking')


def _arms(out,branches,rules):
    arms={'fixed_C':dict(model=None,branch='C',selector=False),
          'strong_C':dict(model=None,branch='C',selector=True)}
    for branch in branches:
        model=Components.load(out/'checkpoints'/branch/'candidate.pt')
        if model.metadata.get('training_rules_digest')!=rules.digest() or model.metadata.get('training_branch')!=branch:
            raise ValueError('checkpoint is not trained against this frozen parent/branch')
        # Loaded-model preparation is an explicitly separate session cost, not
        # per-A solver work. Operator construction/NN inference stay in the timer.
        arms[branch]=dict(model=model.frozen_inference_copy(),branch=branch)
    return arms


def _report(result,out):
    inclusive={}
    for row in result['table']:
        key=f"{row['regime']}:{row['rhs_count']}:{row['arm']}"
        item=inclusive.setdefault(key,dict(total_case_median_wall_seconds=0.,failures=0,cases=0))
        item['total_case_median_wall_seconds']+=row['wall_seconds'] or 0.
        item['failures']+=int(not row['successful']);item['cases']+=1
    write_json(out/'failure_inclusive_wall.json',dict(entries=inclusive,
        scope='sum of actual per-case median elapsed times, including capped failures; not a successful time-to-solution ranking'))
    for regime,counts in result['summary'].items():
        for count,arms in counts.items():
            for name,v in arms.items():
                print(regime,'RHS='+count,name,f"{v['successes']}/{v['total']}",
                      'speedup_vs_strong=',v['geometric_speedup'],
                      'neural_cases=',len(v['actual_neural_case_ids']),flush=True)


def benchmark(output,branches,*,tag='validation',repeats=5,warmups=1,rhs_counts=(1,4,16,64),
              regimes=('cold','warm','multiple'),resume=False):
    out,settings,cfg=_load_run(output)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',tag):raise ValueError('invalid benchmark tag')
    rules=load_strong_rules(out/'selector_rules.json')
    _,validation,_=_development(out,settings,rules)
    target=out/'benchmarks'/tag
    result=evaluate_research(validation,_arms(out,branches,rules),cfg,rules,target,
        repeats=repeats,warmups=warmups,rhs_counts=rhs_counts,regimes=regimes,resume=resume)
    _report(result,target)
    return result


def demo(output,branch,*,family='channel',n=31):
    out,settings,cfg=_load_run(output);rules=load_strong_rules(out/'selector_rules.json')
    spec=dict(sizes=[n],per_family=1,families=[family],seed=int(settings['seed'])+9900001)
    examples,_,_=_generate_split('demo',spec,set(),rules);e=examples[0]
    model=None if branch=='C' else _arms(out,[branch],rules)[branch]['model']
    chosen=replace(cfg,branch=branch,mode='classical' if branch=='C' else 'research')
    prepared=PreparedStrongMG(e.a,n,model,chosen,rules);result=prepared.solve(e.b)
    target=out/'demos'/f'{family}_{n}_{branch}.json'
    write_json(target,result.to_dict())
    print('Selected classical parent:',prepared.selection.strategy_name)
    print('Branch:',branch,'success:',result.converged,'relative residual:',result.relative_residual)
    print('Cycles:',result.executed_cycles,'setup+solve seconds:',result.elapsed_seconds)
    print('Result:',target)
    return result


def freeze(output,branches,*,repeats=5,warmups=1,rhs_counts=(1,4,16,64),regimes=('cold','warm','multiple')):
    out,settings,cfg=_load_run(output);rules=load_strong_rules(out/'selector_rules.json')
    _development(out,settings,rules);_arms(out,branches,rules)
    evaluation=dict(branches=list(branches),repeats=repeats,warmups=warmups,
                    rhs_counts=list(rhs_counts),regimes=list(regimes),solver=cfg.to_dict())
    path=freeze_research(out/'data',{b:out/'checkpoints'/b/'candidate.pt' for b in branches},
                         out/'selector_rules.json',{},_sources(),evaluation)
    print('Frozen, no final operators inspected:',path,flush=True)


def final(output,*,resume=False):
    out,_,cfg=_load_run(output);rules=load_strong_rules(out/'selector_rules.json')
    freeze_path=out/'data/research_freeze.json';evaluation=_read(freeze_path)['config']
    if evaluation['solver']!=cfg.to_dict():raise ValueError('frozen solver changed')
    with claim_final_evaluation(out/'data',freeze_path,evaluation,resume=resume) as claim:
        splits,_=materialize_final_data(out/'data',claim,rules)
        paths=[]
        for split,examples in splits.items():
            target=out/'final'/split
            result=evaluate_research(examples,_arms(out,evaluation['branches'],rules),cfg,rules,target,
                repeats=evaluation['repeats'],warmups=evaluation['warmups'],
                rhs_counts=evaluation['rhs_counts'],regimes=evaluation['regimes'],resume=resume)
            _report(result,target)
            paths.append(target/'raw_results.json')
        complete_final_evaluation(out/'data',claim,paths)
    print('Final/OOD evaluation sealed. This does not assert a speedup or issue an adaptive policy certificate.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('calibrate','train','benchmark','demo','freeze','final'):
        p=sub.add_parser(name);p.add_argument('--run-dir',required=True)
        if name in ('calibrate','train','benchmark','final'):p.add_argument('--resume',action='store_true')
        if name=='calibrate':p.add_argument('--config',default='configs/v6_7_three_pillars_smoke.json')
        if name in ('train','benchmark','freeze'):
            p.add_argument('--branches',nargs='+',choices=('H_S','H_P','H_SP'),default=['H_S','H_P'])
        if name in ('benchmark','freeze'):
            p.add_argument('--repeats',type=int,default=5);p.add_argument('--warmups',type=int,default=1)
            p.add_argument('--rhs-counts',nargs='+',type=int,default=[1,4,16,64])
            p.add_argument('--regimes',nargs='+',choices=('cold','warm','multiple'),default=['cold','warm','multiple'])
        if name=='benchmark':p.add_argument('--tag',default='validation')
        if name=='demo':
            p.add_argument('--branch',choices=('C','H_S','H_P','H_SP'),default='H_S')
            p.add_argument('--family',choices=FAMILIES,default='channel');p.add_argument('--n',type=int,default=31)
    a=parser.parse_args(argv)
    if a.command=='calibrate':return calibrate(a.config,a.run_dir,resume=a.resume)
    if a.command=='train':return train(a.run_dir,a.branches,resume=a.resume)
    if a.command=='demo':return demo(a.run_dir,a.branch,family=a.family,n=a.n)
    if a.command=='final':return final(a.run_dir,resume=a.resume)
    kw=dict(repeats=a.repeats,warmups=a.warmups,rhs_counts=a.rhs_counts,regimes=a.regimes)
    if a.command=='freeze':return freeze(a.run_dir,a.branches,**kw)
    return benchmark(a.run_dir,a.branches,tag=a.tag,resume=a.resume,**kw)
