#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
from adaptive_mg import TemporalComponents
p=argparse.ArgumentParser()
p.add_argument('--input',required=True);p.add_argument('--output',required=True)
p.add_argument('--trust-legacy-checkpoint',action='store_true')
a=p.parse_args()
m=TemporalComponents.migrate_v65(a.input,trusted=a.trust_legacy_checkpoint)
m.save(a.output)
print('Migrated smoother only. Spatial detector, selector, router, learned-transfer extra cycle and prior certificates NOT imported.')
