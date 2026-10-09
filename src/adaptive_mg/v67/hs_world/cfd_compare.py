"""Optional externally measured OpenFOAM GAMG evidence comparison.

Does not run OpenFOAM or invent a GAMG result. Requires exact system/threshold,
thread/hardware and setup-inclusive scope matches. This compares extracted
linear systems, NEVER full combustion CFD or matched nonlinear trajectories.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from .tuning import read
from .temporal import load
from ..world_model.data import file_hash, write_json

SCOPE='linear_sequence_setup_solve_recovery'


def compare(output,reference_file,*,split='validation'):
    if split not in ('validation','test'):raise ValueError('comparison split must be validation/test')
    out,w,h,_=load(output)
    if h['source_kind']!='external_cfd':raise ValueError('synthetic elliptic data cannot be labelled OpenFOAM CFD')
    report=read(out/'temporal'/split/'report.json');ref=read(reference_file)
    if ref.get('schema')!='openfoam-gamg-sequence-v1' or ref.get('solver')!='GAMG':
        raise ValueError('explicit external GAMG measurements required')
    if ref.get('data_sha256')!=h['data_sha256'] or ref.get('time_scope')!=SCOPE:
        raise ValueError('same imported data and setup-inclusive linear-solve scope required')
    expected=report['hardware']
    # Different solver libraries are expected; timing machine/threads are not.
    timing_keys=('machine','cpu_model','affinity_count')
    if any(ref.get('hardware',{}).get(k)!=expected.get(k) for k in timing_keys):
        raise ValueError('GAMG timing hardware differs')
    # Explicit producer attestation: native code need not import PyTorch.
    # Serial bridge only; this is not a parallel scalability comparison.
    if ref.get('execution_threads')!=1 or expected.get('torch_threads')!=1:
        raise ValueError('this serial comparison requires one execution thread')
    if not ref.get('openfoam_version') or not ref.get('configuration_sha256'):
        raise ValueError('native solver version/config provenance required')
    measured={r['trajectory']:r for r in ref.get('trajectories',[])}
    if len(measured)!=len(ref.get('trajectories',[])) or set(measured)!={r['trajectory'] for r in report['rows']}:
        raise ValueError('GAMG needs the exact paired trajectory cohort')
    pairs=[]
    for row in report['rows']:
        external=measured[row['trajectory']];nr=row['runs']['World_HS'];gr=external['runs']
        if not gr:raise ValueError('missing native repeats')
        systems=nr[0]['rows']
        gs=[]
        for run in gr:
            if len(run['systems'])!=len(systems):raise ValueError('GAMG system count mismatch')
            if not np.isfinite(run['total_seconds']) or run['total_seconds']<=0:raise ValueError('invalid native timing')
            ok=True
            for given,actual in zip(run['systems'],systems):
                for key in ('step','matrix_digest','rhs_digest','x0_digest'):
                    if given.get(key)!=actual[key]:raise ValueError('native matrix/RHS/guess/order mismatch')
                tol=given.get('threshold')
                if tol is None or not np.isclose(tol,actual['threshold'],rtol=1e-12,atol=0.):
                    raise ValueError('native stopping threshold mismatch')
                residual=given['final_true_residual']
                if not np.isfinite(residual) or residual<0:raise ValueError('native true residual required')
                ok &= bool(residual<=tol and given['success'])
            gs.append((ok,float(run['total_seconds'])))
        ngood=all(r['success'] for r in nr);ggood=all(a for a,_ in gs)
        speed=float(np.median([v for _,v in gs])/np.median([r['total_seconds'] for r in nr])) if ngood and ggood else None
        pairs.append(dict(trajectory=row['trajectory'],gamg_success=ggood,world_HS_success=ngood,speedup=speed))
    ratios=[p['speedup'] for p in pairs if p['speedup'] is not None]
    output=dict(source='external supplied OpenFOAM GAMG measurement',reference_sha256=file_hash(reference_file),
        paired_rows=pairs,geometric_speedup=float(np.exp(np.log(ratios).mean())) if ratios else None,
        time_scope=SCOPE,full_cfd_wall_clock_measured=False,physical_observables_validated=False)
    destination=out/f'gamg_comparison_{split}.json'
    if destination.exists():raise FileExistsError('comparison evidence already exists')
    write_json(destination,output);return output
