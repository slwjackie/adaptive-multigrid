#!/usr/bin/env python3
"""C_tuned + H_S primary study; strong_C is an optional robustness audit."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.hs_world.study import main
if __name__ == '__main__':
    main()
