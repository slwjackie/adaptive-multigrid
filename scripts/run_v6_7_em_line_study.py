#!/usr/bin/env python3
"""Run sequential EM-imitation P and shared line-smoother training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from adaptive_mg.v67.em_line_study import main

if __name__ == '__main__':
    main()
