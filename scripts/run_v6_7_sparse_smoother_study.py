#!/usr/bin/env python3
"""Run direct CNN with learned sparse approximate-inverse smoothing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.sparse_smoother_study import main

if __name__ == '__main__':
    main()
