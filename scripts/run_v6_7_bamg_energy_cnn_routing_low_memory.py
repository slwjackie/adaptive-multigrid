#!/usr/bin/env python3
"""Run or resume grid-routing CNNs with teacher labels loaded one case at a time.

All original study arguments are forwarded unchanged. The adapter lives outside
the recorded numerical source tree, and separately binds its execution sources
before use and in the eventual final freeze. Existing study identities and
cached mathematical labels are preserved.
"""
import argparse
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent / 'src'))

from adaptive_mg.v67 import bamg_energy_cnn_routing_study as routing
from adaptive_mg.v67.bamg_energy_cnn_source_compat import source_reader
from bamg_energy_cnn_memory import memory_mode
from bamg_energy_cnn_memory_audit import execution_audit


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('command', choices=('run', 'report', 'freeze', 'evaluate-final', 'report-final'))
    parser.add_argument('--output', required=True)
    if '-h' in argv or '--help' in argv:
        return routing.main(argv)
    args, _ = parser.parse_known_args(argv)
    adapter_paths = [Path(__file__).resolve(), SCRIPTS / 'bamg_energy_cnn_memory.py',
                     SCRIPTS / 'bamg_energy_cnn_memory_audit.py']
    with source_reader(), memory_mode(), execution_audit(args.output, args.command, adapter_paths):
        return routing.main(argv)


if __name__ == '__main__':
    main()
