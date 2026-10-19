#!/usr/bin/env python3
"""Run direct-CNN channel pruning and charged online adaptation comparisons."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.direct_pruning_study import main

if __name__ == '__main__':
    main()
