"""Immutable ordered synthetic elliptic-system snapshots with explicit data provenance."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import os
import tempfile
from functools import cached_property
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import scipy.sparse.linalg as sla
from ...grid import as_shape, validate_root_shape
from ...solver import validate_matrix
from ...provenance import json_safe, operator_digest

SPLITS = ('train', 'tune', 'validation', 'test')

def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, allow_nan=False).encode()).hexdigest()

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(json_safe(value), f, indent=2, allow_nan=False); f.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def inside(root, relative):
    root = Path(root).resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ValueError('path escapes dataset root')
    return path

@dataclass(frozen=True)
class Snapshot:
    a: sp.csr_matrix
    b: np.ndarray
    x0: np.ndarray
    shape: tuple[int, int]
    time: float
    index: int
    mesh_id: str
    boundary_id: str
    source_kind: str = 'synthetic_elliptic'
    context: dict = field(default_factory=dict)

    def validate(self, *, spd_check=True):
        validate_root_shape(self.shape, 3); validate_matrix(self.a, self.shape)
        if self.b.shape != (self.a.shape[0],) or self.x0.shape != self.b.shape:
            raise ValueError('RHS/x0 shape mismatch')
        if not np.isfinite(self.b).all() or not np.isfinite(self.x0).all():
            raise ValueError('nonfinite RHS/x0')
        if not np.isfinite(self.time) or self.index < 0 or not self.mesh_id or not self.boundary_id:
            raise ValueError('missing time/mesh/boundary contract')
        if self.source_kind != 'synthetic_elliptic':
            raise ValueError('only synthetic_elliptic snapshots are supported')
        if spd_check:
            if self.a.shape[0] <= 1024:
                la.cholesky(self.a.toarray(), lower=True, check_finite=True)
            else:
                # Numerical admission check, NOT a rigorous eigenvalue certificate.
                v = np.ones(self.a.shape[0]); v /= np.linalg.norm(v)
                eig = sla.eigsh(self.a, k=1, which='SA', return_eigenvectors=False,
                               v0=v, tol=1e-7)[0]
                if eig <= 1e-12 * np.max(self.a.diagonal()):
                    raise ValueError('non-SPD or unresolved near-nullspace')
        return self

    @cached_property
    def matrix_digest(self): return operator_digest(self.a)
    @cached_property
    def normalized_matrix_digest(self):
        a=self.a.copy();a.data/=float(np.max(np.abs(a.data)))
        return operator_digest(a)

    @cached_property
    def topology_key(self):
        return digest(dict(shape=self.shape,mesh=self.mesh_id,boundary=self.boundary_id,
                           indptr=hashlib.sha256(self.a.indptr.tobytes()).hexdigest(),
                           indices=hashlib.sha256(self.a.indices.tobytes()).hexdigest()))

def snapshot(a, b, *, shape, time, index, mesh_id, boundary_id, x0=None,
             source_kind='synthetic_elliptic', context=None):
    a = sp.csr_matrix(a, dtype=np.float64, copy=True)
    a.sum_duplicates(); a.eliminate_zeros(); a.sort_indices()
    b = np.array(b, dtype=np.float64, copy=True)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    s = Snapshot(a,b,x,as_shape(shape),float(time),int(index),str(mesh_id),str(boundary_id),source_kind,context or {})
    s.validate()
    for arr in (a.data,a.indices,a.indptr,b,x): arr.flags.writeable = False
    return s

def save_snapshot(path, s):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    meta=dict(shape=s.shape,time=s.time,index=s.index,mesh_id=s.mesh_id,boundary_id=s.boundary_id,
              source_kind=s.source_kind,context=s.context)
    with path.open('wb') as f:
        np.savez_compressed(f, data=s.a.data, indices=s.a.indices, indptr=s.a.indptr,
                            b=s.b,x0=s.x0,metadata=np.array(json.dumps(json_safe(meta),allow_nan=False)))
    return dict(path=path.name,sha256=file_hash(path),matrix_digest=s.matrix_digest,normalized_matrix_digest=s.normalized_matrix_digest)

def load_snapshot(path):
    with np.load(path,allow_pickle=False) as d:
        meta=json.loads(str(d['metadata']));n=len(d['b'])
        return snapshot(sp.csr_matrix((d['data'],d['indices'],d['indptr']),shape=(n,n)),
                        d['b'],x0=d['x0'],**meta)

def load_manifest(root):
    root=Path(root);m=json.loads((root/'sequence_manifest.json').read_text())
    if m.get('schema')!='world-sequence-v1':raise ValueError('unknown sequence schema')
    if m.get('source_kind')!='synthetic_elliptic':raise ValueError('only synthetic_elliptic datasets are supported')
    groups={};ids=set();operators={}
    for tr in m['trajectories']:
        if tr['split'] not in SPLITS or tr['id'] in ids:raise ValueError('bad split/duplicate trajectory')
        if not tr['case_group'] or not tr['id']:raise ValueError('empty case/trajectory identity')
        ids.add(tr['id']);g=tr['case_group']
        for e in tr['snapshots']:
            key=e.get('normalized_matrix_digest',e['matrix_digest'])
            if key in operators and operators[key]!=tr['split']:raise ValueError('normalized operator crosses splits')
            operators[key]=tr['split']
        if g in groups and groups[g]!=tr['split']:raise ValueError('physical case crosses data splits')
        groups[g]=tr['split']
    return m

def load_trajectories(root, split):
    if split not in SPLITS:raise ValueError('invalid split')
    m=load_manifest(root);result=[]
    for tr in m['trajectories']:
        if tr['split']!=split:continue
        states=[]
        for entry in tr['snapshots']:
            path=inside(root,entry['path'])
            if file_hash(path)!=entry['sha256']:raise ValueError('snapshot hash changed')
            s=load_snapshot(path)
            if s.matrix_digest!=entry['matrix_digest']:raise ValueError('matrix digest changed')
            if states and (s.index<=states[-1].index or s.time<states[-1].time):
                raise ValueError('trajectory must preserve solve ordering and nondecreasing time')
            states.append(s)
        if len(states)<2:raise ValueError('a trajectory needs at least two systems')
        result.append((tr,states))
    if not result:raise ValueError('empty sequence split '+split)
    return result

def generate(root, *, seed=20261010, counts=(6,3,3,3), steps=6, sizes=(7,15)):
    """Coefficient trajectories only: NEVER invent T/rho or call them H2 CFD."""
    from ...pde import DiffusionCase, assemble_stiffness
    root=Path(root)
    if root.exists() and any(root.iterdir()):raise FileExistsError('new dataset directory required')
    if len(counts)!=4 or min(counts)<1 or steps<2:raise ValueError('invalid dataset sizes')
    root.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed);trajectories=[]
    for split,count in zip(SPLITS,counts):
        for j in range(count):
            ident=f'{split}_{j:03d}';n=int(sizes[j%len(sizes)]);theta=float(rng.uniform(15,75))
            contrast=float(rng.uniform(2,20));eps=float(rng.uniform(.1,.6))
            family=('channel','checkerboard','local_patch')[j%3];entries=[]
            mesh=f'rectangle-{n}-standard';prev=None
            for t in range(steps):
                theta=float(np.clip(theta+rng.normal(0,1.5),5,85));contrast*=float(np.exp(rng.normal(0,.04)))
                if t==steps//2 and j%2==0:contrast*=1.5
                a=assemble_stiffness(DiffusionCase(n,epsilon=eps,angle_deg=theta,contrast=contrast,pattern=family))
                # Includes exact-A reuse without fabricating an evolving fluid state.
                if t==1 and prev is not None:a=prev.a
                exact=rng.normal(size=n*n);b=a@exact
                s=snapshot(a,b,shape=(n,n),time=t*.01,index=t,mesh_id=mesh,boundary_id='dirichlet-eliminated',
                           context={'synthetic_family':family})
                name=f'{ident}/{t:05d}.npz';entry=save_snapshot(root/name,s);entry['path']=name;entries.append(entry);prev=s
            trajectories.append(dict(id=ident,case_group=ident,split=split,snapshots=entries))
    m=dict(schema='world-sequence-v1',seed=seed,source_kind='synthetic_elliptic',
           physics='time-varying diffusion coefficients; no Navier-Stokes or chemistry',
           trajectories=trajectories)
    # For generated data, disallow cross-trajectory duplicate operators. Same-A
    # consecutive RHS in ONE trajectory is intentional.
    seen={}
    for tr in trajectories:
        for e in tr['snapshots']:
            key=e['normalized_matrix_digest']
            if key in seen and seen[key]!=tr['id']:raise ValueError('duplicate operator across trajectories')
            seen[key]=tr['id']
    write_json(root/'sequence_manifest.json',m);return m
