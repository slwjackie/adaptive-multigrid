#!/usr/bin/env python3
"""Compare scratch and warm-start Weight CNNs on shared hybrid TRAIN labels."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from adaptive_mg.v67.bamg_energy_cnn_source_compat import main

if __name__ == '__main__':
    main()
