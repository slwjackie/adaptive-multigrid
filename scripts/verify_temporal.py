#!/usr/bin/env python3
"""Large-grid state/cache contracts (not a time-to-tolerance certificate)."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
from dataclasses import replace
from time import perf_counter
import numpy as np
import torch
from adaptive_mg import *
from adaptive_mg.hierarchy import hierarchy_identity
from adaptive_mg.data import fixed_benchmark_examples
from adaptive_mg.provenance import write_json,stable_norm


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--large-sizes',default='127,255');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output-dir',default='artifacts/temporal_contracts');args=p.parse_args()
    torch.set_num_threads(1);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    model=TemporalComponents.load(args.checkpoint)
    cfg=MGConfig(mode='fixed',fixed_k=2,max_cycles=6,tolerance=1e-30,absolute_tolerance=0.,near_tolerance_factor=0.)
    records=[]
    for e in fixed_benchmark_examples([int(n) for n in args.large_sizes.split(',')]):
        prepared=PreparedTemporalMGSolver(e.a,e.n,model,cfg);before=hierarchy_identity(prepared.root)
        r=prepared.solve(e.b)
        first_c=next((i for i,v in enumerate(r.cycle_path) if v!='neural'),len(r.cycle_path))
        invariant=(all(path!='neural' for path in r.cycle_path[first_c:]) and len(r.switch_events)<=1
                   and before==hierarchy_identity(prepared.root)
                   and r.stats['switch_hierarchy_builds']==r.stats['detector_calls']==0
                   and np.isfinite(r.final_true_residual))
        measured=stable_norm(e.b-e.a@r.x)
        assert invariant and np.isclose(measured,r.final_true_residual,rtol=1e-12,atol=0.)
        c0=replace(cfg,fixed_k=0,max_cycles=2)
        c=PreparedTemporalMGSolver(e.a,e.n,model,replace(c0,mode='classical')).solve(e.b)
        z=PreparedTemporalMGSolver(e.a,e.n,model,c0).solve(e.b)
        identical=np.array_equal(c.x,z.x) and c.residual_history==z.residual_history
        assert identical
        records.append(dict(name=e.name,n=e.n,one_way_contract=invariant,zero_equals_classical=identical,
                            phase_path=r.cycle_path,true_relative_residual=r.relative_residual,
                            interpretation='six-cycle contract test, NOT full tolerance convergence',stats=r.stats))
        write_json(out/'large_grid_contracts.json',records)
        print('[large contract]',e.name,e.n,r.cycle_path,flush=True)
    # Repeated RHS: time includes setup once, and each RHS starts a NEW temporal
    # state. Identical A and smoother reuse their banks, not solution vectors.
    e=fixed_benchmark_examples([63])[0]
    rng=np.random.default_rng(22);bs=np.stack([e.b*(1+.1*j)+.001*rng.standard_normal(len(e.b)) for j in range(6)])
    options=replace(cfg,max_cycles=100,tolerance=1e-8,absolute_tolerance=1e-12)
    times={'new':[],'reuse':[]};last=None;exact_equal=True
    # Untimed warmup primes library allocations, never reuses a solver in the
    # measured 'new' arm.
    PreparedTemporalMGSolver(e.a,e.n,model,options).solve(bs[0])
    for repeat in range(args.repeats):
        outcomes={}
        for arm in (['new','reuse'] if repeat%2==0 else ['reuse','new']):
            t=perf_counter()
            if arm=='new':rs=[PreparedTemporalMGSolver(e.a,e.n,model,options).solve(b) for b in bs]
            else:rs=PreparedTemporalMGSolver(e.a,e.n,model,options).solve_many(bs)
            times[arm].append(perf_counter()-t);outcomes[arm]=rs
        exact_equal &= all(np.array_equal(a.x,b.x) for a,b in zip(outcomes['new'],outcomes['reuse']))
        last=outcomes['reuse']
    assert exact_equal and last and all(r.converged for r in last)
    assert sum(r.stats['freshness_checks'] for r in last)==1
    assert all(r.stats['neural_setup_calls']==0 for r in last[1:])
    report=dict(n=63,rhs_count=6,repeats=args.repeats,times=times,
                speedup=float(np.median(times['new'])/np.median(times['reuse'])),
                bitwise_equal=bool(exact_equal),all_converged=True,
                setup_calls_per_rhs=[r.stats['neural_setup_calls'] for r in last],
                predictor_calls_per_rhs=[r.stats['predictor_calls'] for r in last],
                phase_paths=[r.cycle_path for r in last],freshness_checks=sum(r.stats['freshness_checks'] for r in last),
                interpretation='same fixed-K solver with/without setup reuse; NOT neural-vs-classical speedup')
    write_json(out/'rhs_reuse.json',report)
    write_json(out/'contracts_summary.json',dict(large_grid_cases=len(records),all_large_contracts_passed=all(r['one_way_contract'] and r['zero_equals_classical'] for r in records),rhs_reuse=report))
    print('All contracts passed.',flush=True)


if __name__=='__main__':main()
