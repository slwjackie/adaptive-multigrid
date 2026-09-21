#!/usr/bin/env python3
"""Calibrate, train, benchmark and run the corrected standalone MG solver."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_mg.v67.three_pillars import main
if __name__=='__main__':main()
