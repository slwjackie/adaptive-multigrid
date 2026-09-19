#!/usr/bin/env python3
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_mg.native_stencil import build_native
p=argparse.ArgumentParser();p.add_argument('--no-openmp',action='store_true');p.add_argument('--force',action='store_true');a=p.parse_args()
print(json.dumps(build_native(openmp=not a.no_openmp,force=a.force),indent=2))
