#!/usr/bin/env python3
"""Run the isolated numerical-teacher comparison without CNN training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.bamg_energy_study import main

if __name__ == '__main__':
    main()
