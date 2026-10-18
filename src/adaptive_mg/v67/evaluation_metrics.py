"""Shared measurement diagnostics, independent of classical strategy selection."""
from pathlib import Path
import hashlib
import json

import numpy as np

from ..provenance import json_safe


def _hash(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _hierarchy(level):
    rows = []
    while level is not None:
        rows.append(dict(grid_shape=list(level.shape), operator_shape=list(level.a.shape),
                         operator_nnz=int(level.a.nnz),
                         prolongation_shape=list(level.p.shape) if level.p is not None else None,
                         prolongation_nnz=int(level.p.nnz) if level.p is not None else 0))
        if getattr(level,"transfer_setup_report",None) is not None:
            rows[-1]["transfer_setup_report"]=dict(level.transfer_setup_report)
        level = level.coarse
    return rows


def _speed_summary(ratios):
    if not ratios:
        return dict(geometric_speedup=None, ci95=[None, None], cases=0)
    logs = np.log(ratios)
    samples = np.exp(np.random.default_rng(93).choice(logs, (2000, len(logs)), replace=True).mean(axis=1))
    return dict(geometric_speedup=float(np.exp(logs.mean())), ci95=np.quantile(samples, [.025, .975]).tolist(), cases=len(logs))


def _source_signature():
    source_root = Path(__file__).resolve().parents[1]
    files = sorted(source_root.rglob('*.py'))
    return _hash({str(path.relative_to(source_root)): digest_file(path) for path in files})
