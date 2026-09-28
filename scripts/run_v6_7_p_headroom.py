#!/usr/bin/env python3
"""Development-only P headroom / classical interpolation / 2x2 diagnostic."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_mg.v67.p_headroom_study import main
if __name__=='__main__':
    main()
