#!/usr/bin/env python3
"""Freeze deterministic leaf choices from a completed train/tune bank benchmark."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.strong import load_strong_rules
from adaptive_mg.v67.strong_calibration import calibrate_from_run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__ + ' NO neural training or new solves; audit/fresh/smoke input rejected.')
    p.add_argument('--input-run', required=True, help='completed --split train/tune --oracle-bank controlled/all run')
    p.add_argument('--research-dir', default='artifacts/my_v6_7_research')
    p.add_argument('--rules', default='configs/v6_7_strong_rules.json')
    p.add_argument('--output-dir', default='artifacts/my_v6_7_strong_baseline/calibration')
    p.add_argument('--minimum-leaf-cases', type=int, default=3)
    p.add_argument('--tie-fraction', type=float, default=.01)
    a = p.parse_args(argv)
    frozen, evidence = calibrate_from_run(a.input_run, a.research_dir, load_strong_rules(a.rules), a.output_dir,
        minimum_leaf_cases=a.minimum_leaf_cases, tie_fraction=a.tie_fraction)
    print('NO NN TRAINING / NO NEW SOLVES. Frozen rules:', Path(a.output_dir).resolve() / 'selector_rules.json')
    print('Rules digest:', frozen.digest())
    for leaf, result in evidence['leaves'].items():
        print(leaf, result['samples'], result['previous_strategy'], '->', result['selected_strategy'], result['status'])
    print('Strength requires a separate held-out comparison; final certificate unchanged.')


if __name__ == '__main__': main()
