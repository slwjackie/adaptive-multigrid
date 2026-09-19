#!/usr/bin/env python3
"""No-training paired comparison against deterministic A-only classical MG."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.strong_evaluation import benchmark_main

if __name__ == '__main__':
    benchmark_main()
