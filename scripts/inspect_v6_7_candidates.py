#!/usr/bin/env python3
"""Read preserved research artifacts; write diagnosis only to a separate output."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from adaptive_mg.v67.inspection import inspect_candidates, verify_preservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research', type=Path, default=Path('artifacts/my_v6_7_research'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/my_v6_7_limited_fix/inspection'))
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--verify-hashes', type=Path, help='Check a saved preservation manifest without writing artifacts')
    args = parser.parse_args()
    if args.verify_hashes:
        result = verify_preservation(args.verify_hashes)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['all_unchanged'] else 1
    result = inspect_candidates(args.research, args.output, args.project_root)
    print(json.dumps({'output': result['output'], **result['preservation'],
                      'initial_entry_reconstruction_verified': result['registry']['reconstructed_initial_entry']['verified']},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
