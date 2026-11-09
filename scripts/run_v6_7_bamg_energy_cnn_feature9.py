#!/usr/bin/env python3
"""Opt-in row9/symmetric9 routing study: run, resume, freeze and final/OOD.

Use a feature9 config and a new output directory. Original ten-channel outputs
must still use their original runner. Teacher labels load one case at a time.
"""
import argparse
import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent / 'src'))

from adaptive_mg.v67 import bamg_energy_cnn_routing_study as routing
from adaptive_mg.v67.bamg_energy_cnn_source_compat import source_reader
from bamg_energy_cnn_memory import memory_mode
from bamg_feature9 import validate_mode
from bamg_feature9_adapter import feature9_mode


def _mode(path):
    with Path(path).open(encoding='utf-8') as stream:
        settings = json.load(stream)
    return validate_mode(settings['models']['weight'].get('feature_mode'))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if '-h' in argv or '--help' in argv:
        return routing.main(argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('command', choices=('run', 'report', 'freeze', 'evaluate-final', 'report-final'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    args, _ = parser.parse_known_args(argv)
    saved = Path(args.output) / 'configuration.json'
    if args.command == 'run':
        if not args.config:
            parser.error('run requires a feature9 --config')
        requested = _mode(args.config)
        if saved.exists() and _mode(saved) != requested:
            raise ValueError('Different feature mode; use a new output directory')
    else:
        _mode(saved)
    with source_reader(), memory_mode(), feature9_mode():
        return routing.main(argv)


if __name__ == '__main__':
    main()
