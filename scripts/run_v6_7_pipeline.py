#!/usr/bin/env python3
"""Run/resume the six stages and independent audit. No background service."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_mg.v67.training import pipeline

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/v6_7_research.json')
    p.add_argument('--output-dir',default='artifacts/v6_7_research')
    p.add_argument('--initial-checkpoint')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    model,result=pipeline(args.config,args.output_dir,args.initial_checkpoint,args.resume)
    s=result['summary']['adaptive'];c=result['summary']['classical']
    print('Classical 대비 end-to-end speedup:',s['geometric_mean_speedup'])
    print('Classical cycles (common-success mean):',s['mean_classical_cycles_common'])
    print('Adaptive cycles (common-success mean):',s['mean_cycles_common'])
    print(f"Accuracy/tolerance: Classical {c['successes']}/{c['total']}; Neural {s['successes']}/{s['total']}")
    print('Stability improvement:', 'Yes (observed)' if s['successes']>c['successes'] and s['new_failures']==0 else 'No (not established)')
    print('NN setup amortized: No (cold audit)')
    print('Total-time improvement certified:',model.metadata['certificate']['validated'])
if __name__=='__main__':main()
