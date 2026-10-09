"""Persistent, local-only pressure-solver bridge for the OpenFOAM 13 plugin.

Framing: uint64 big-endian byte count, then UTF-8 JSON. No pickle, shell,
remote listening socket or per-solve Python process. Native Amul witnesses
validate the finalized system before solving; native code rechecks the answer.
"""
from __future__ import annotations

import json
import os
import socket
import struct
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as graph

from ...provenance import stable_norm
from ..world_model.adapter import FinalizedLduRecorder, solution_to_native
from ..world_model.data import Snapshot, digest, snapshot, write_json
from .data import validate_case_contract

SCHEMA = 'h2-fixed-p-live-v1'
MAX_FRAME = 128 * 1024 * 1024


def _exact(sock, size):
    blocks = []
    while size:
        block = sock.recv(min(size, 1024 * 1024))
        if not block:
            raise EOFError('connection closed inside frame')
        blocks.append(block)
        size -= len(block)
    return b''.join(blocks)


def receive(sock):
    size = struct.unpack('!Q', _exact(sock, 8))[0]
    if size < 2 or size > MAX_FRAME:
        raise ValueError('invalid bridge frame size')
    value = json.loads(_exact(sock, size).decode('utf-8'))
    if not isinstance(value, dict):
        raise ValueError('object request required')
    return value


def send(sock, value):
    payload = json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(payload) > MAX_FRAME:
        raise ValueError('bridge response too large')
    sock.sendall(struct.pack('!Q', len(payload)) + payload)


def _array(value, name, shape=None, integer=False):
    a = np.asarray(value)
    if integer and a.dtype.kind not in 'iu':
        raise ValueError(name + ' must contain integers')
    a = np.asarray(a, dtype=np.int64 if integer else np.float64)
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has wrong shape')
    if not np.isfinite(a).all():
        raise ValueError(name + ' contains nonfinite values')
    return a


def _anchored_m_matrix(a):
    """Cheap sufficient numerical SPD admission for symmetric FV M matrices.

Each connected component needs a positive diagonal-dominance anchor. Generic
SPD matrices use the existing Cholesky/eigenvalue admission instead. This is
a floating-point check, not a rigorous spectral certificate.
"""
    off = a.copy()
    off.setdiag(0)
    off.eliminate_zeros()
    if off.nnz and np.max(off.data) > 0:
        return False
    surplus = a.diagonal() - np.asarray(abs(off).sum(axis=1)).ravel()
    if np.any(surplus < 0):
        return False
    count, labels = graph.connected_components(off, directed=False)
    anchors = np.bincount(labels, weights=(surplus > 0).astype(float), minlength=count)
    return bool(np.all(anchors > 0))


def decode_system(request, contract):
    if request.get('schema') != SCHEMA:
        raise ValueError('unsupported bridge schema')
    if request.get('source_kind') != 'external_cfd':
        raise ValueError('explicit external_cfd producer declaration required')
    if request.get('boundary_finalized') is not True or request.get('coupled_interfaces') != 0:
        raise ValueError('finalized uncoupled serial LDU required')
    if request.get('nullspace') != 'none':
        raise ValueError('pressure reference/Dirichlet anchor required')
    diag = _array(request['diag'], 'diag')
    if diag.ndim != 1 or len(diag) < 9:
        raise ValueError('invalid diagonal')
    n = len(diag)
    lo = _array(request['lower_addr'], 'lower_addr', integer=True)
    up = _array(request['upper_addr'], 'upper_addr', lo.shape, integer=True)
    if lo.ndim != 1 or np.any(lo < 0) or np.any(up >= n) or np.any(lo >= up):
        raise ValueError('invalid LDU addressing')
    lower = _array(request['lower'], 'lower', lo.shape)
    upper = _array(request['upper'], 'upper', lo.shape)
    b = _array(request['b'], 'b', (n,))
    x0 = _array(request['x0'], 'x0', (n,))
    order = _array(request['structured_to_native'], 'structured_to_native', (n,), integer=True)
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError('invalid cell permutation')
    native = sp.coo_matrix((np.r_[diag, upper, lower],
        (np.r_[np.arange(n), lo, up], np.r_[np.arange(n), up, lo])), shape=(n, n)).tocsr()
    probes = _array(request['probe_vectors'], 'probe_vectors')
    products = _array(request['probe_products'], 'probe_products', probes.shape)
    if probes.ndim != 2 or probes.shape[0] != n or probes.shape[1] < 2:
        raise ValueError('two native Amul witnesses required')
    if np.linalg.matrix_rank(probes) < 2 or not np.allclose(native @ probes, products, rtol=1e-10, atol=1e-12):
        raise ValueError('CSR disagrees with native finalized Amul')
    raw_shape = request['shape']
    if (not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) for n in raw_shape)):
        raise ValueError('two-integer structured grid shape required')
    shape = tuple(raw_shape)
    if shape != tuple(contract['shape']):
        raise ValueError('case grid shape changed')
    a = native[order][:, order].tocsr()
    a.sum_duplicates(); a.eliminate_zeros(); a.sort_indices()
    context = dict(request.get('context', {}), layout='structured_2d_xmajor',
                   case_contract_digest=digest(contract), physics=contract['physics'])
    if isinstance(request['index'], bool) or not isinstance(request['index'], int) or request['index'] < 0:
        raise ValueError('solve index must be a nonnegative integer')
    if any(not isinstance(request[k], str) or not request[k] for k in ('mesh_id', 'boundary_id')):
        raise ValueError('nonempty mesh/boundary identities required')
    kwargs = dict(shape=shape, time=float(request['time']), index=request['index'],
                  mesh_id=request['mesh_id'], boundary_id=request['boundary_id'],
                  source_kind='external_cfd', context=context)
    if _anchored_m_matrix(a):
        s = Snapshot(a, b[order].copy(), x0[order].copy(), **kwargs)
        s.validate(spd_check=False)
        for arr in (s.a.data, s.a.indices, s.a.indptr, s.b, s.x0):
            arr.flags.writeable = False
    else:
        s = snapshot(a, b[order], x0=x0[order], **kwargs)
    raw = dict(diag=diag, lower_addr=lo, upper_addr=up, lower=lower, upper=upper,
               b=b, x0=x0, probe_vectors=probes, probe_products=products,
               structured_to_native=order)
    return s, native, raw


class Session:
    def __init__(self, cfg, contract, *, mode, expert=None, recording=None, evidence=None):
        from .backend import FixedPSolver
        self.contract = validate_case_contract(contract)
        if mode not in ('native', 'classical', 'hs'):
            raise ValueError('mode must be native/classical/hs')
        if (mode == 'hs') != (expert is not None):
            raise ValueError('H_S requires a trained checkpoint; other arms cannot use one')
        if cfg.mg.residual_reference != 'initial':
            raise ValueError('live bridge uses the initial raw L2 residual')
        self.cfg, self.mode = cfg, mode
        self.solver = FixedPSolver(cfg, expert) if mode != 'native' else None
        self.recorder = None
        if recording is not None:
            if mode != 'native':
                raise ValueError('collect training recordings with the native arm; no recording I/O in timing arms')
            self.recorder = FinalizedLduRecorder(recording,
                producer={'solver': 'OpenFOAM-13', 'bridge': SCHEMA}, physics=contract['physics'])
            write_json(Path(recording) / 'case_contract.json', contract)
        self.last_index, self.last_time, self.topology = -1, -float('inf'), None
        self.evidence = Path(evidence) if evidence else None
        if self.evidence:
            self.evidence.parent.mkdir(parents=True, exist_ok=True)
            if self.evidence.exists():
                raise FileExistsError('fresh evidence file required')

    def handle(self, request):
        started = perf_counter()
        if request.get('mode') != self.mode:
            raise ValueError('plugin/service arm mismatch')
        expected_op = 'record' if self.mode == 'native' else 'solve'
        if request.get('op') != expected_op:
            raise ValueError('unexpected operation for this arm')
        for key, value in (('rtol', self.cfg.mg.tolerance), ('atol', self.cfg.mg.absolute_tolerance),
                           ('max_cycles', self.cfg.mg.max_cycles)):
            if request.get(key) != value:
                raise ValueError('plugin/service stopping criterion mismatch: ' + key)
        s, native, raw = decode_system(request, self.contract)
        if s.index <= self.last_index or s.time < self.last_time:
            raise ValueError('nonmonotone solve stream')
        if self.topology is not None and self.topology != s.topology_key:
            raise ValueError('mesh/boundary/sparsity changed in fixed-P stream')
        initial_product = s.a @ s.x0
        initial = stable_norm(s.b - initial_product)
        threshold = max(self.cfg.mg.absolute_tolerance, self.cfg.mg.tolerance * initial)
        if not np.isfinite(initial) or not np.isfinite(threshold):
            raise ValueError('nonfinite raw residual or stopping threshold')
        # CSR and native LDU sum in different orders. When b and A*x0 nearly
        # cancel, their residual discrepancy is governed by those large terms,
        # not by the small residual itself. This allowance applies ONLY to the
        # diagnostic agreement test; neither solve's stopping target is relaxed.
        residual_roundoff = 128*np.finfo(float).eps*(stable_norm(s.b) + stable_norm(initial_product))
        checks = (('native_initial_residual', initial, residual_roundoff + 1e-10*initial),
                  ('native_threshold', threshold, self.cfg.mg.tolerance*residual_roundoff + 1e-12*threshold))
        for key, expected, allowance in checks:
            if key in request and (not np.isfinite(request[key]) or abs(request[key] - expected) > allowance):
                raise ValueError('native/Python raw residual contract mismatch: ' + key)
        admission_seconds = perf_counter() - started
        if self.mode == 'native':
            xn = _array(request['x_native_reference'], 'x_native_reference', s.b.shape)
            final = stable_norm(raw['b'] - native @ xn)
            if not np.isfinite(final) or final > threshold:
                raise ValueError('native reference failed the shared raw L2 threshold')
            result = dict(success=True, cycles=int(request.get('reference_cycles', 0)),
                          threshold=threshold, initial_residual=initial, final_residual=final,
                          p_digest=None)
            if self.recorder:
                self.recorder.record(trajectory=self.contract['case_id'], case_group=self.contract['case_group'],
                    split=self.contract['split'], index=s.index, time=s.time, shape=s.shape,
                    mesh_id=s.mesh_id, boundary_id=s.boundary_id, boundary_finalized=True,
                    nullspace='none', coupled_interfaces=0, context=s.context, **raw)
        else:
            step = self.solver.step(s)
            xn = solution_to_native(step.x, raw['structured_to_native'])
            final = stable_norm(raw['b'] - native @ xn)
            result = dict(success=bool(step.success and final <= threshold), cycles=step.cycles,
                threshold=threshold, initial_residual=initial, final_residual=final,
                p_digest=self.solver.p_digest, step=step.record())
        self.last_index, self.last_time, self.topology = s.index, s.time, s.topology_key
        result.update(schema=SCHEMA, mode=self.mode, index=s.index, time=s.time,
                      admission_seconds=admission_seconds,
                      service_seconds=perf_counter() - started,
                      matrix_digest=s.matrix_digest, error='')
        if self.evidence:
            with self.evidence.open('a') as f:
                f.write(json.dumps(result, allow_nan=False) + '\n')
        return dict(result, x_native=xn.tolist())


def serve(path, session, *, timeout=300.0):
    """One producer connection per process; fail closed and retain evidence."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('refuse to replace an existing socket/path')
    if len(os.fsencode(path)) >= 104:
        raise ValueError('Unix socket path too long; use a short scratch path')
    path.parent.mkdir(parents=True, exist_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path)); os.chmod(path, 0o600)
        listener.listen(1); listener.settimeout(timeout)
        print('READY ' + str(path), flush=True)
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(timeout)
            while True:
                try:
                    request = receive(conn)
                except EOFError:
                    break
                try:
                    response = session.handle(request)
                except Exception as exc:
                    send(conn, dict(schema=SCHEMA, success=False, error=str(exc), x_native=[]))
                    raise
                send(conn, response)
                if not response['success']:
                    raise RuntimeError('pressure solve failed; CFD must not advance')
    finally:
        listener.close()
        if path.is_socket():
            path.unlink()
