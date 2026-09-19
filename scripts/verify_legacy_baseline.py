#!/usr/bin/env python3
"""Optional explicit parity check against the supplied v6.5 source tree."""
from pathlib import Path
import sys,subprocess,os,tempfile,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from adaptive_mg import *
from adaptive_mg.provenance import write_json
p=argparse.ArgumentParser();p.add_argument('--legacy-root',required=True);p.add_argument('--output',default='artifacts/legacy_baseline_parity.json')
a=p.parse_args();root=Path(a.legacy_root).resolve()
if not (root/'src/adaptive_mg/config.py').exists():raise ValueError('legacy source root not found')
code=r'''
import sys,numpy as np,torch
from adaptive_mg.config import MGConfig
from adaptive_mg.solver import PreparedNeuralMGSolver
from adaptive_mg.pde import DiffusionCase,assemble_stiffness
from adaptive_mg.strategy import STRATEGIES
torch.set_num_threads(1)
A=assemble_stiffness(DiffusionCase(n=15,epsilon=.03,angle_deg=37.))
b=np.random.default_rng(93).standard_normal(225)
result={}
for s in STRATEGIES:
 c=MGConfig(mode='classical',global_selection='fixed',strategy_name=s.name,max_cycles=3,tolerance=1e-30,absolute_tolerance=0.)
 r=PreparedNeuralMGSolver(A,15,None,c).solve(b)
 result[s.name]=r.x
 result[s.name+'__residual']=np.array(r.residual_history)
np.savez(sys.argv[1],**result)
'''
torch.set_num_threads(1)
with tempfile.TemporaryDirectory() as tmp:
    file=Path(tmp)/'legacy.npz'
    env=dict(os.environ,PYTHONPATH=str(root/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,'-c',code,str(file)],cwd=tmp,env=env,check=True)
    legacy=np.load(file);A=assemble_stiffness(DiffusionCase(n=15,epsilon=.03,angle_deg=37.));b=np.random.default_rng(93).standard_normal(225)
    records=[]
    for s in STRATEGIES:
        r=PreparedTemporalMGSolver(A,15,config=MGConfig(mode='classical',strategy_name=s.name,max_cycles=3,tolerance=1e-30,absolute_tolerance=0.)).solve(b)
        xequal=np.array_equal(r.x,legacy[s.name])
        oldr=legacy[s.name+'__residual']; newr=np.asarray(r.residual_history)
        residual_error=float(np.max(np.abs(newr-oldr)/np.maximum(np.abs(oldr),1e-300)))
        requal=np.allclose(newr,oldr,rtol=2e-14,atol=0.0)
        print(s.name, 'x_equal',xequal,'x_maxdiff',np.max(np.abs(r.x-legacy[s.name])), 'residual_rel_diff',residual_error)
        assert xequal and requal,s.name
        records.append(dict(strategy=s.name,solution_bitwise_equal=bool(xequal),residual_within_roundoff=bool(requal),residual_relative_max_difference=residual_error))
write_json(a.output,dict(legacy_version='6.5',new_version='6.6',cases=len(records),all_solutions_bitwise_equal=True,all_residuals_within_roundoff=True,residual_note='v6.6 uses overflow-resistant stable_norm; residual diagnostic differs from v6.5 np.linalg.norm at rounding level only.',records=records))
print('v6.5/v6.6 classical parity:',len(records),'strategies, solutions bitwise identical; residual diagnostics within roundoff')
