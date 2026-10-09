#!/usr/bin/env python3
"""Independent fixed-P H2 CFD workflow; leaves existing studies untouched."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.h2_fixed_p.study import main

if __name__ == '__main__':
    main()
