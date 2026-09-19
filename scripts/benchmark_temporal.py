#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import csv
import json
from dataclasses import replace
import numpy as np
import torch
from adaptive_mg import MGConfig,TemporalComponents
from adaptive_mg.data import fixed_benchmark_examples
from adaptive_mg.measurement import run_measured,geometric_mean
from adaptive_mg.provenance import write_json,hardware_environment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--sizes',default='15,31,63,127,255')
    p.add_argument('--strategy',default=None);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-cycles',type=int,default=None);p.add_argument('--tolerance',type=float,default=None)
    p.add_argument('--modes',default='classical,temporal,predicted,fixed2')
    p.add_argument('--threads',type=int,default=1);p.add_argument('--regime',choices=['cold','warm'],default='cold')
    p.add_argument('--output-dir',default='artifacts/temporal_benchmark');a=p.parse_args()
    torch.set_num_threads(a.threads)
    model=TemporalComponents.load(a.checkpoint)
    scope=model.metadata.get('temporal_certificate',{}).get('scope',{})
    cfg=MGConfig(**scope)
    if a.strategy: cfg=replace(cfg,strategy_name=a.strategy)
    if a.max_cycles is not None: cfg=replace(cfg,max_cycles=a.max_cycles)
    if a.tolerance is not None: cfg=replace(cfg,tolerance=a.tolerance)
    modes=a.modes.split(',');out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    examples=fixed_benchmark_examples([int(n) for n in a.sizes.split(',')]);rng=np.random.default_rng(781)
    raw=[];summary=[]
    write_json(out/'environment.json',hardware_environment());write_json(out/'solver_config.json',cfg.to_dict())
    for e in examples:
        bymode={m:[] for m in modes}
        def conf(m): return replace(cfg,mode='fixed',fixed_k=int(m[5:])) if m.startswith('fixed') else replace(cfg,mode=m)
        for m in modes: run_measured(e,model,conf(m),regime=a.regime)
        for repeat in range(a.repeats):
            for m in rng.permutation(modes).tolist():
                r=run_measured(e,model,conf(m),regime=a.regime)
                raw.append(dict(name=e.name,n=e.n,mode=m,repeat=repeat,**r));bymode[m].append(r)
        base=bymode.get('classical',[])
        tc=float(np.median([r['wall_seconds'] for r in base])) if base else None
        cb=bool(base) and all(r['converged'] for r in base)
        for m,records in bymode.items():
            tt=float(np.median([r['wall_seconds'] for r in records]));ok=all(r['converged'] for r in records)
            r=records[0]
            row=dict(name=e.name,n=e.n,mode=m,converged=ok,wall_seconds=tt,
                speedup=tc/tt if cb and ok else None,cycles=r['mg_cycles'],executed_mg_cycles=r['executed_mg_cycles'],K_NN=r['decision']['k'],
                neural_cycles=r['stats']['accepted_neural_cycles'],rejections=r['stats']['rejected_neural_cycles'],
                true_relative_residual=r['relative_residual'],relative_solution_error=r['relative_solution_error'],
                switches=len(r['switch_events']),predictor_calls=r['stats']['predictor_calls'])
            summary.append(row)
        print('[benchmark]',e.name,e.n,{m:(bymode[m][0]['mg_cycles'],bymode[m][0]['decision']['k']) for m in modes},flush=True)
        write_json(out/'raw_results.json',raw);write_json(out/'summary_rows.json',summary)
    totals={m:dict(converged=sum(r['converged'] for r in summary if r['mode']==m),cases=len(examples),
                  common_success=sum(r['speedup'] is not None for r in summary if r['mode']==m),
                  geometric_speedup=geometric_mean([r['speedup'] for r in summary if r['mode']==m]),
                  nonzero_k_cases=sum(r['K_NN']>0 for r in summary if r['mode']==m)) for m in modes}
    write_json(out/'benchmark_summary.json',totals)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    text='# Standalone temporal-switch benchmark\n\nTime scope: '+a.regime+'; speedup = Classical / candidate. Failures are NOT counted as speed wins.\n\n|Mode|Converged|Common-success speedup|Nonzero K|\n|---|---:|---:|---:|\n'
    for m,r in totals.items(): text+=f"|{m}|{r['converged']}/{r['cases']}|{r['geometric_speedup']}|{r['nonzero_k_cases']}|\n"
    text+='\nAn uncertified temporal policy abstains to K=0. `predicted` bypasses the empirical certificate for diagnostics only; numerical rollback/one-way safeguards remain enabled.\n'
    (out/'benchmark_summary.md').write_text(text)
    print(json.dumps(totals,indent=2))


if __name__=='__main__': main()
