#!/usr/bin/env python3
"""Fixed-support, two-CNN BAMG study (CPU sparse FP64 reference backend)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from adaptive_mg.v67.bamg_study import main

if __name__ == '__main__':
    main()
