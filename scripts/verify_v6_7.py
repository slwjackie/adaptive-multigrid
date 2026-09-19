#!/usr/bin/env python3
"""Additional stress/cache/native microbenchmarks, separate from accuracy audit."""
import argparse,sys
from pathlib import Path
from dataclasses import replace
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from adaptive_mg import MGConfig,DiffusionCase,assemble_stiffness
from adaptive_mg.v67 import *
from adaptive_mg.v67.banks import Stats
from adaptive_mg.v67.native import apply_rows
from adaptive_mg.v67.data import dataset,split_manifest
from adaptive_mg.provenance import write_json,hardware_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/initial_v6_7.pt')
    p.add_argument('--sizes',default='15,31,63,127,255');p.add_argument('--output-dir',default='artifacts/v6_7_contracts')
    args=p.parse_args();torch.set_num_threads(1);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    model=Components.load(args.checkpoint);records=[]
    for n in [int(x) for x in args.sizes.split(',')]:
        for case in [DiffusionCase(n=n,epsilon=1.,contrast=1.),DiffusionCase(n=n,epsilon=.01,angle_deg=45.,contrast=10.,pattern='local_patch')]:
            a=assemble_stiffness(case);b=np.random.default_rng(n).normal(size=n*n)
            mg=MGConfig(mode='classical',strategy_name='jacobi_bilinear_full',nn_levels=1,pre_steps=2,post_steps=2,max_cycles=4,
                        tolerance=1e-14,absolute_tolerance=0.,stencil_backend='auto')
            cfg=AdaptiveConfig(mg=mg,mode='hybrid',spatial=False,gate_mode='open')
            prepared=PreparedAdaptiveMG(a,n,model,cfg);initial_identity=id(prepared.classical)
            r=prepared.solve(b);learned_identity=id(prepared.learned);s=prepared.solve(-b)
            closed=PreparedAdaptiveMG(a,n,model,replace(cfg,gate_mode='closed',spatial=True)).solve(b)
            classical=PreparedAdaptiveMG(a,n,model,replace(cfg,mode='classical')).solve(b)
            records.append(dict(n=n,case=case.pattern,finite=bool(np.isfinite(r.x).all()),
                classical_hierarchy_reused=id(prepared.classical)==initial_identity,
                learned_bank_reused=id(prepared.learned)==learned_identity,
                second_rhs_heavy_calls=s.stats['neural_setup_calls'],
                closed_exact=bool(np.array_equal(closed.x,classical.x)),
                residual_hierarchy_rebuilds=r.stats['residual_hierarchy_rebuilds'],
                cycles=r.cycles,attempts=r.executed_cycles,
                purpose='four-cycle contract check, NOT full time-to-tolerance validation'))
            write_json(out/'large_grid_contracts.json',records)
    # Same-A RHS reuse for BOTH arms. Do not label these as neural-vs-classical speedups.
    n=31;a=assemble_stiffness(DiffusionCase(n=n,epsilon=1.));bs=np.random.default_rng(3).normal(size=(6,n*n));reuse={}
    for mode in ['classical','hybrid']:
        cfg=AdaptiveConfig(mg=MGConfig(mode='classical',strategy_name='jacobi_bilinear_full',max_cycles=30,pre_steps=2,post_steps=2,nn_levels=1),mode=mode,spatial=False,gate_mode='open')
        t=perf_counter();separate=[PreparedAdaptiveMG(a,n,model,cfg).solve(b) for b in bs];tc=perf_counter()-t
        t=perf_counter();prepared=PreparedAdaptiveMG(a,n,model,cfg);shared=prepared.solve_many(bs);tn=perf_counter()-t
        reuse[mode]=dict(separate_seconds=tc,shared_seconds=tn,setup_reuse_speedup=tc/tn,
            exact_parity=all(np.array_equal(x.x,y.x) for x,y in zip(separate,shared)),
            freshness_checks=sum(x.stats['freshness_checks'] for x in shared),
            nn_setups=[x.stats['neural_setup_calls'] for x in shared],
            all_converged=all(x.converged for x in shared))
    write_json(out/'rhs_reuse.json',reuse)
    # Backend dispatch threshold calibration. All methods preserve SAME active rows.
    cfg=AdaptiveConfig(mg=MGConfig(mode='classical',nn_levels=1),mode='hybrid',spatial=False,gate_mode='open')
    a=assemble_stiffness(DiffusionCase(n=63,epsilon=1.));prepared=PreparedAdaptiveMG(a,63,model,cfg)
    root=prepared.ensure_learned(Stats());bank=root.neural_stencil;r=np.random.default_rng(6).normal(size=63*63)
    calibration=[]
    for fraction in [.05,.1,.25,.5,.75,1.]:
        rows=np.sort(np.random.default_rng(6).choice(len(r),max(1,int(len(r)*fraction)),replace=False))
        csr=bank.csr[rows]
        timings={}
        for backend in ['csr_rows','csr_full','native_rows','native_full']:
            if backend.startswith('native') and bank.native is None:continue
            def apply():
                if backend=='csr_rows':return csr@r
                if backend=='csr_full':return (bank.csr@r)[rows]
                if backend=='native_rows':return apply_rows(bank.native,r,rows)
                return bank.native.apply(r)[:,0][rows]
            apply();samples=[]
            for _ in range(7):
                t=perf_counter()
                for __ in range(20):z=apply()
                samples.append((perf_counter()-t)/20)
            assert np.allclose(z,csr@r,rtol=1e-12,atol=1e-12)
            timings[backend]=float(np.median(samples))
        calibration.append(dict(active_fraction=fraction,seconds=timings,fastest=min(timings,key=timings.get)))
    write_json(out/'backend_calibration.json',dict(n=63,results=calibration,
        note='Microbenchmark only. active_skip changes algorithm and needs independent solver validation.'))
    # Validate generation and disjointness (do not call this model training).
    groups={k:dataset([7,15],7,seed) for k,seed in [('train',17),('tune',37),('audit',67)]}
    write_json(out/'small_split_check.json',split_manifest(**groups))
    write_json(out/'environment.json',hardware_environment())
    passed=all(x['finite'] and x['closed_exact'] and x['classical_hierarchy_reused'] and x['learned_bank_reused'] and x['second_rhs_heavy_calls']==0 for x in records)
    write_json(out/'status.json',{'contract_checks_passed':passed,'cases':len(records),
        'accuracy_audit_completed':False,'M2_Pro_measured':False,'A100_measured':False})
    print('Contracts passed:',passed,'cases:',len(records))
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()
