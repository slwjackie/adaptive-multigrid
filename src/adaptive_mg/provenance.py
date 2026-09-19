"""Exact cache freshness and hardware-scoped performance evidence."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import numpy as np
import scipy
import scipy.sparse as sp
import torch
from threadpoolctl import threadpool_info


def operator_digest(a: sp.spmatrix, *, scale_invariant=False) -> str:
    a = a.tocsr(copy=True).astype(np.float64)
    a.sum_duplicates(); a.sort_indices(); a.eliminate_zeros()
    data = a.data
    if scale_invariant:
        # A rounded normalized fingerprint also groups operators which only
        # differ in an irrelevant uniform scale, unlike the old split check.
        data = np.round(data / max(float(np.max(np.abs(data), initial=0)), 1e-300), 11)
    h = hashlib.sha256(repr(a.shape).encode())
    for v in (a.indptr.astype(np.int64), a.indices.astype(np.int64), data):
        h.update(v.tobytes())
    return h.hexdigest()


def module_signature(module) -> str:
    if module is None:
        return "none"
    h = hashlib.sha256(type(module).__qualname__.encode())
    attrs = {k: getattr(module, k) for k in ("body_kind", "basis_count", "coefficient_scale", "split_direction_gain", "direction_epsilon", "k_values") if hasattr(module, k)}
    h.update(json.dumps(attrs, sort_keys=True).encode())
    for k, v in sorted(module.state_dict().items()):
        value = v.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(value.dtype).encode()); h.update(repr(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def hardware_environment():
    cpu = platform.processor()
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip(); break
    except OSError:
        pass
    pools = [{k: p.get(k) for k in ('internal_api', 'num_threads', 'version', 'architecture')} for p in threadpool_info()]
    return dict(machine=platform.machine(), system=platform.system(), cpu_model=cpu,
                affinity_count=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
                numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__),
                torch_threads=torch.get_num_threads(), pools=pools, runtime='scipy_fp64_cpu')


def stable_norm(x):
    """An overflow-resistant L2 norm; nonfinite vectors are never accepted."""
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(np.abs(x), initial=0.0))
    if not np.isfinite(m):
        return float('inf')
    if m == 0:
        return 0.0
    with np.errstate(over='ignore', invalid='ignore'):
        return float(m * np.linalg.norm(x / m))


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)
