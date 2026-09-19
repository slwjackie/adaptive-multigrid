#!/usr/bin/env python3
"""Continue verification from saved smoke students; no restart or final access."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_mg.provenance import write_json
from adaptive_mg.v67.limited import read_settings,digest_file
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.strong import load_strong_rules
from adaptive_mg.v67.research_data import load_development_data
from adaptive_mg.v67.research_evaluation import evaluate_research,evaluate_classical_reference
from adaptive_mg.v67.research_training import train_expert
from adaptive_mg.v67.research_policy import ResearchPolicy


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args(argv)
    project=Path(__file__).resolve().parents[1]
    run=Path(args.run_dir).resolve();out=Path(args.output_dir).resolve()
    if (project/'artifacts/my_v6_7_strong_aware').resolve() not in out.parents:
        raise ValueError('Use a new strong-aware artifact subdirectory')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Verification records are immutable; choose a new directory')
    settings=json.loads((run/'configuration.json').read_text())
    _,_,cfg=read_settings(project/settings['base_research']);cfg=replace(cfg,mode='research')
    rules=load_strong_rules(run/'frozen_rules.json')
    train,validation,data=load_development_data(run,rules)
    out.mkdir(parents=True,exist_ok=True);started=perf_counter()
    source={str(p.relative_to(project)):digest_file(p) for p in sorted((project/'src/adaptive_mg').rglob('*.py'))}
    write_json(out/'verification_manifest.json',dict(source_run=str(run),source=source,
        scope='continued smoke verification using existing trained checkpoints; no architecture reselection',
        existing_training_not_restarted=True,final_test_seen=False))
    # Exercise repaired feasibility-aware training starting from the existing
    # GNN checkpoint. Keep this diagnostic continuation out of branch selection.
    initial=Components.load(run/'checkpoints/gnn_direct/candidate.pt')
    local_settings=dict(settings['training'],updates=2,kd_updates=0)
    _,training_status=train_expert(initial,train,cfg,rules,local_settings,
        out/'gnn_repair_continuation',branch='H_P')
    models={b:Components.load(run/'selected_students'/(b+'.pt')) for b in ('H_S','H_P','H_SP')}
    policy=ResearchPolicy.load(run/'policy/deployable_policy')
    arms=dict(fixed_C=dict(model=None,branch='C',selector=False),strong_C=dict(model=None,branch='C'),
        **{b:dict(model=model,branch=b) for b,model in models.items()},
        adaptive=dict(policy=policy,branch='auto'),
        adaptive_rhs_blind=dict(policy=policy,branch='auto',expected_rhs_mode='blind',reference_only=True))
    result=evaluate_research(validation,arms,cfg,rules,out/'branches',repeats=args.repeats,warmups=1,
        regimes=('cold',),rhs_counts=(1,))
    print('Continued cold comparison:',[(n,x['successes'],x['geometric_speedup']) for n,x in result['summary']['cold']['1'].items()],flush=True)
    subset=[next(e for e in validation if e.n==n and e.research_family=='near_isotropic') for n in (15,31)]
    batch=evaluate_research(subset,arms,cfg,rules,out/'cold_warm_multiple_rhs',repeats=args.repeats,warmups=1,
        regimes=('cold','warm','multiple'),rhs_counts=(1,4,16,64))
    reference=evaluate_classical_reference([validation[0]],cfg,rules,out/'classical_reference',
        research=project/settings['base_research'],repeats=args.repeats,warmups=1,bank='controlled')
    report=dict(status='complete',source_run=str(run),training_continuation=training_status,
        branches=result['summary'],amortization_crossover=batch['amortization_crossover'],
        classical_reference=reference['summary'],existing_checkpoints_unchanged=True,
        final_test_seen=False,performance_certified=False,wall_seconds=perf_counter()-started)
    write_json(out/'verification_report.json',report)
    print('Verification complete:',out,flush=True)


if __name__=='__main__':main()
