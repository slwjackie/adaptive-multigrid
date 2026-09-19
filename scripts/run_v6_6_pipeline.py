#!/usr/bin/env python3
"""Sequential reproducible driver; no timer measurements run concurrently."""
from pathlib import Path
import sys
import argparse
import subprocess
import os
p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/v6_6_verification.json')
p.add_argument('--initial-checkpoint',default='checkpoints/initial_temporal.pt')
p.add_argument('--output-dir',default='artifacts/v6_6_run')
p.add_argument('--sizes',default='15,31,63,127,255');p.add_argument('--repeats',type=int,default=3)
a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ)
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env.setdefault(key,'1')
def run(script,*arguments):
    subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*map(str,arguments)],check=True,env=env)
run('train_temporal.py','--config',a.config,'--initial-checkpoint',a.initial_checkpoint,'--output-dir',out/'training')
checkpoint=out/'training/checkpoint_v6_6_final.pt'
run('benchmark_temporal.py','--checkpoint',checkpoint,'--sizes',a.sizes,'--repeats',a.repeats,'--output-dir',out/'benchmark')
run('verify_temporal.py','--checkpoint',checkpoint,'--output-dir',out/'contracts','--repeats',a.repeats)
