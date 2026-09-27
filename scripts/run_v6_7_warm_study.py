#!/usr/bin/env python3
"""Warm multi-RHS experts -> frozen expert -> continuous-size policy -> final."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.warm_study import main
if __name__ == '__main__':
    main()
