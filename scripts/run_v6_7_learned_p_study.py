#!/usr/bin/env python3
"""Fixed line_alt smoother and Frobenius-trained sparse learned P."""
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.joint_transfer_study import main

if __name__ == '__main__':
    main()
