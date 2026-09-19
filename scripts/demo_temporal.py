#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
from dataclasses import replace
import torch
from adaptive_mg import *
from adaptive_mg.data import fixed_benchmark_examples
from adaptive_mg.provenance import write_json
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True);p.add_argument('--n',type=int,default=63)
p.add_argument('--case',default='poisson');p.add_argument('--mode',choices=['temporal','predicted','fixed','classical'],default='temporal')
p.add_argument('--k',type=int,default=2);p.add_argument('--output',default='artifacts/demo_trace.json')
a=p.parse_args();torch.set_num_threads(1);model=TemporalComponents.load(a.checkpoint)
scope=model.metadata.get('temporal_certificate',{}).get('scope',{})
cfg=replace(MGConfig(**scope),mode=a.mode,fixed_k=a.k,verbose=True)
e=next(e for e in fixed_benchmark_examples([a.n]) if e.name==a.case)
r=PreparedTemporalMGSolver(e.a,e.n,model,cfg).solve(e.b)
write_json(a.output,r.to_dict())
print('Decision:',r.decision)
print('Transitions:',r.switch_events)
print('Converged:',r.converged,'accepted updates:',r.mg_cycles,'executed V-cycles:',r.executed_mg_cycles)
print('NN setup calls:',r.stats['neural_setup_calls'],'detector calls:',r.stats['detector_calls'])
