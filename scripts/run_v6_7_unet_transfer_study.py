#!/usr/bin/env python3
"""Run the isolated residual U-net teacher / CNN transfer study."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.unet_transfer_study import main

if __name__ == '__main__':
    main()
