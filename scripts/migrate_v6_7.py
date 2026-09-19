#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_mg.v67 import Components
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
a=p.parse_args();c=Components.migrate_v66(a.source);c.save(a.output)
print('Migrated smoother only. Learned P/controller/detector initialized anew; certificates reset.')
