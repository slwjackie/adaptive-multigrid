#!/usr/bin/env python3
"""Copy a verified interrupted 66e9f6d run for the teacher L1-start fix.

Successful cached teachers already started inside the same feasible set, so
their numerical labels, learned weights, optimizer and RNG state are preserved.
The original folder and any earlier bootstrap-recovery provenance are retained.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recover_neural_amg_bootstrap import recover_teacher_l1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Original interrupted run, left unchanged')
    parser.add_argument('--output-dir', required=True, help='New, nonexistent run directory')
    args = parser.parse_args()
    recover_teacher_l1(args.run_dir, args.output_dir)
