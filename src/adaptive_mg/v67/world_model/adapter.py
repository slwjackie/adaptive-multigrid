"""Python-side finalized-LDU recorder and native/structured solution mapping.

Call AFTER the CFD code has assembled boundary/reference contributions. This
module does not hook into OpenFOAM automatically and does not advance a flame.
Native matrix-vector witnesses must come from the external solver, not CSR.
"""
from __future__ import annotations
from pathlib import Path
import json
import re
import numpy as np
from .data import file_hash,write_json

class FinalizedLduRecorder:
    def __init__(self,root,*,producer,physics):
        self.root=Path(root)
        if self.root.exists() and any(self.root.iterdir()):raise FileExistsError('new recorder directory required')
        self.root.mkdir(parents=True,exist_ok=True)
        self.manifest=dict(schema='finalized-ldu-sequence-v1',producer=producer,physics=physics,trajectories=[])
        self._flush()

    def _flush(self):write_json(self.root/'ldu_sequence.json',self.manifest)

    def record(self,*,trajectory,case_group,split,index,time,shape,mesh_id,boundary_id,
               diag,lower_addr,upper_addr,lower,upper,b,x0,probe_vectors,probe_products,
               structured_to_native,boundary_finalized,nullspace,coupled_interfaces,context=None):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',trajectory) or not case_group:raise ValueError('invalid case identity')
        if split not in ('train','tune','validation','test'):raise ValueError('invalid split')
        if not boundary_finalized or nullspace!='none' or coupled_interfaces!=0:
            raise ValueError('only boundary-finalized anchored serial scalar systems supported')
        tr=next((v for v in self.manifest['trajectories'] if v['id']==trajectory),None)
        if tr is None:
            tr=dict(id=trajectory,case_group=case_group,split=split,snapshots=[]);self.manifest['trajectories'].append(tr)
        if tr['case_group']!=case_group or tr['split']!=split:raise ValueError('trajectory identity changed')
        if tr['snapshots'] and (index<=tr['snapshots'][-1]['index'] or time<tr['snapshots'][-1]['time']):
            raise ValueError('nonmonotone solve index/time')
        for other in self.manifest['trajectories']:
            if other['case_group']==case_group and other['split']!=split:raise ValueError('case group leakage')
        path=self.root/trajectory/f'{index:07d}.npz';path.parent.mkdir(exist_ok=True)
        if path.exists():raise FileExistsError(path)
        with path.open('wb') as f:
            np.savez_compressed(f,diag=diag,lower_addr=lower_addr,upper_addr=upper_addr,lower=lower,upper=upper,
                 b=b,x0=x0,probe_vectors=probe_vectors,probe_products=probe_products,structured_to_native=structured_to_native)
        tr['snapshots'].append(dict(path=str(path.relative_to(self.root)),sha256=file_hash(path),index=int(index),time=float(time),
            shape=list(shape),mesh_id=str(mesh_id),boundary_id=str(boundary_id),layout='structured_2d_xmajor',
            boundary_finalized=True,nullspace='none',coupled_interfaces=0,context=context or {}))
        self._flush()


def solution_to_native(structured,structured_to_native):
    x=np.asarray(structured,dtype=np.float64);order=np.asarray(structured_to_native)
    if x.ndim!=1 or order.dtype.kind not in 'iu' or not np.array_equal(np.sort(order),np.arange(len(x))):
        raise ValueError('invalid vector/permutation')
    result=np.empty_like(x);result[order]=x;return result
