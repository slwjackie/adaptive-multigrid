#!/usr/bin/env python3
"""Repeatable CPU/MPS/CUDA generation on saved development examples only."""
from pathlib import Path
import argparse
from dataclasses import replace
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.limited import read_settings, digest_file
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.strong import load_strong_rules
from adaptive_mg.v67.research_data import load_development_data
from adaptive_mg.v67.research_generation import benchmark_generation
from adaptive_mg.provenance import write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--devices',nargs='+',choices=('cpu','mps','cuda'),default=['cpu','mps','cuda'])
    parser.add_argument('--all-validation',action='store_true')
    parser.add_argument('--include-teachers',action='store_true')
    args=parser.parse_args(argv)
    project=Path(__file__).resolve().parents[1]
    run=Path(args.run_dir).resolve();out=Path(args.output_dir).resolve()
    allowed=(project/'artifacts/my_v6_7_strong_aware').resolve()
    if allowed not in out.parents:raise ValueError('Use a new strong-aware artifact subdirectory')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Preserve previous measurements; choose a new output')
    settings=json.loads((run/'configuration.json').read_text())
    _,_,cfg=read_settings(project/settings['base_research'])
    cfg=replace(cfg,mode='research')
    rules=load_strong_rules(run/'frozen_rules.json')
    _,validation,_=load_development_data(run,rules)
    if args.all_validation:examples=validation
    else:
        examples=[next(e for e in validation if e.n==n and
                       e.research_family==family)
                  for n,family in ((15,'near_isotropic'),(31,'rotated_anisotropy'))]
    groups={
        'smoother':('H_S',['ordinary_direct','compact_direct','d4_direct','o2_direct']),
        'transfer':('H_P',['p_cnn_direct','gnn_direct','edge_direct'])}
    if args.include_teachers:
        groups['smoother'][1].append('s_teacher');groups['transfer'][1].append('p_teacher')
    out.mkdir(parents=True,exist_ok=True)
    checkpoints={}
    for _,names in groups.values():
        for name in names:
            path=run/'checkpoints'/name/'candidate.pt'
            checkpoints[name]=dict(path=str(path),sha256=digest_file(path))
    write_json(out/'provenance.json',dict(run=str(run),checkpoints=checkpoints,
        split='validation',example_names=[e.name for e in examples],
        final_test_seen=False,selection_unchanged=True,
        source={str(p.relative_to(project)):digest_file(p) for p in sorted((project/'src/adaptive_mg').rglob('*.py'))},
        note='Independent device diagnostic; original run manifests/results are preserved'))
    for group,(branch,names) in groups.items():
        models={name:Components.load(checkpoints[name]['path']) for name in names}
        result=benchmark_generation(models,examples,cfg,rules,out/group,
            repeats=args.repeats,branches=(branch,),devices=args.devices,warmups=1,split='validation')
        print(group,[(r['model'],r['example'],r['device'],r['status']) for r in result['summary']],flush=True)
    print('Measured generation only; no final certificate:',out,flush=True)


if __name__=='__main__':main()
