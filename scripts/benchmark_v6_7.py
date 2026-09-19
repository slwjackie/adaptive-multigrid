#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys,json
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from adaptive_mg.v67 import *
from adaptive_mg.v67.evaluation import paired_suite,summarize,save_summary
from adaptive_mg.data import fixed_benchmark_examples
from adaptive_mg.config import MGConfig
from adaptive_mg.provenance import write_json,hardware_environment


def ablations(cfg):
    raw=replace(cfg,mode='hybrid',spatial=False,gate_mode='open',break_even=False)
    temporal=replace(cfg,mode='research',spatial=False,gate_mode='open',break_even=False)
    return {
        'A_classical':replace(cfg,mode='classical'),
        'B_smoother_only':replace(raw,use_transfer=False),
        'C_transfer_only':replace(raw,use_smoother=False),
        'D_both_replacement':raw,
        'E_temporal':temporal,
        'F_temporal_spatial':replace(temporal,spatial=True,gate_mode='learned'),
        'G_break_even':replace(cfg,mode='research',lazy=False),
        'H_lazy_cache_fp64':replace(cfg,mode='research',inference_dtype='float64'),
        'I_mixed_precision':replace(cfg,mode='research',inference_dtype='float32'),
        'J_production_certified':replace(cfg,mode='production'),
        'additive':replace(raw,application='additive'),
        'fine_only':replace(raw,mg=replace(cfg.mg,nn_levels=1)),
        'first_two':replace(raw,mg=replace(cfg.mg,nn_levels=2)),
        'all_levels':replace(raw,mg=replace(cfg.mg,nn_levels=-1)),
        'detector_every_cycle':replace(cfg,mode='research',gate_refresh=1),
        'detector_refresh_cached':replace(cfg,mode='research',gate_refresh=4),
        'fp64_generator':replace(raw,inference_dtype='float64'),
        'fp32_generator':replace(raw,inference_dtype='float32'),
        'pre_neural_post_classical':replace(raw,replace_pre=cfg.mg.pre_steps,replace_post=0),
        'pre_classical_post_neural':replace(raw,replace_pre=0,replace_post=cfg.mg.post_steps),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoints/initial_v6_7.pt')
    p.add_argument('--sizes',default='15,31,63,127,255')
    p.add_argument('--strategy',default='line_alt_bilinear_full')
    p.add_argument('--max-cycles',type=int,default=100)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--regime',choices=['cold','warm'],default='cold')
    p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto'])
    p.add_argument('--all-ablations',action='store_true')
    p.add_argument('--output-dir',default='artifacts/v6_7_benchmark')
    args=p.parse_args();torch.set_num_threads(1)
    c=Components.load(args.checkpoint)
    cfg=AdaptiveConfig(mg=MGConfig(mode='classical',strategy_name=args.strategy,max_cycles=args.max_cycles,
        pre_steps=2,post_steps=2),inference_device=args.device)
    configs=ablations(cfg) if args.all_ablations else {
        'classical':replace(cfg,mode='classical'),'adaptive_research':replace(cfg,mode='research'),
        'adaptive_production':cfg,'forced_hybrid':replace(cfg,mode='hybrid',spatial=False,gate_mode='open')}
    if args.device=='mps':configs={k:v for k,v in configs.items() if v.inference_dtype!='float64'}
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'environment.json',hardware_environment());write_json(out/'configurations.json',{k:v.to_dict() for k,v in configs.items()})
    rows=paired_suite(fixed_benchmark_examples([int(v) for v in args.sizes.split(',')]),c,configs,
        repeats=args.repeats,warmups=args.warmups,regime=args.regime,output=out)
    result=summarize(rows,baseline='A_classical' if args.all_ablations else 'classical');save_summary(out,result)
    print((out/'summary.md').read_text())
if __name__=='__main__':main()
