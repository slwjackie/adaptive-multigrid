"""Stable sparse operator fingerprints for disjoint data splits."""
import hashlib
import numpy as np
import scipy.sparse as sp


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

