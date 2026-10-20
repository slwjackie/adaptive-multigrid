#!/usr/bin/env python3
"""Independent fixed-P lightweight line-smoother experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from adaptive_mg.v67.light_smoother_study import main

if __name__ == '__main__':
    main()
