"""Import real CFD recordings without inventing combustion provenance."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from ...grid import validate_root_shape
from ..world_model.data import (
    SPLITS, digest, file_hash, inside, import_finalized_ldu,
    load_manifest, load_trajectories, write_json,
)


def read(path):
    return json.loads(Path(path).read_text())


def validate_case_contract(value):
    """Validate a producer declaration, not independently certify flame physics."""
    if value.get('schema') != 'h2-fixed-p-case-v1':
        raise ValueError('h2-fixed-p-case-v1 case contract required')
    for key in ('case_id', 'case_group'):
        pattern = r'[A-Za-z0-9_-]+' if key == 'case_id' else r'[A-Za-z0-9_.-]+'
        if not isinstance(value.get(key), str) or not re.fullmatch(pattern, value[key]):
            raise ValueError('invalid ' + key)
    if value.get('split') not in SPLITS:
        raise ValueError('explicit trajectory split required')
    shape = value.get('shape')
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) for n in shape)):
        raise ValueError('explicit two-integer structured grid shape required')
    validate_root_shape(tuple(shape), 3)
    physics = value.get('physics', {})
    if not isinstance(physics, dict):
        raise ValueError('physics declaration must be an object')
    expected = dict(fuel='H2', oxidizer='air', dimension=2, fixed_grid=True)
    if any(physics.get(k) != v for k, v in expected.items()):
        raise ValueError('need an actual fixed-grid 2D H2-air CFD declaration')
    if not re.fullmatch(r'[0-9a-f]{64}', str(value.get('chemistry_sha256', ''))):
        raise ValueError('hash the actual chemistry/thermo/transport inputs')
    if value.get('synthetic', False):
        raise ValueError('synthetic fields cannot be admitted as combustion data')
    return value


def _check_recording_contract(raw, contract):
    if len(raw.get('trajectories', [])) != 1:
        raise ValueError('each recording must contain exactly one physical trajectory')
    tr = raw['trajectories'][0]
    if (tr['case_group'] != contract['case_group'] or tr['split'] != contract['split']
            or tr['id'] != contract['case_id'] or raw.get('physics') != contract['physics']):
        raise ValueError('recording and case contract disagree')
    if len(tr['snapshots']) < 2:
        raise ValueError('a recording needs at least two pressure systems')
    for entry in tr['snapshots']:
        context = entry.get('context', {})
        if (entry.get('shape') != list(contract['shape'])
                or context.get('case_contract_digest') != digest(contract)
                or context.get('physics') != contract['physics']):
            raise ValueError('snapshot and case contract disagree')
    return tr


def import_recordings(inputs, output):
    """Copy producer recordings, validate native witnesses, preserve case splits.

Each input contains ldu_sequence.json and case_contract.json emitted by the
live service. Every physical case stays in a single split. No random timestep
split and no manufactured RHS is provided by this workflow.
"""
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('new dataset directory required')
    if not inputs:
        raise ValueError('at least one CFD recording required')
    out.mkdir(parents=True, exist_ok=True)
    merged = dict(schema='world-sequence-v1', source_kind='external_cfd',
                  physics='producer-declared fixed-grid 2D H2-air reacting CFD',
                  combustion_verified=False, trajectories=[], producers=[])
    ids, groups = set(), {}
    for i, source in enumerate(inputs):
        src = Path(source).resolve()
        contract = validate_case_contract(read(src / 'case_contract.json'))
        raw = read(src / 'ldu_sequence.json')
        tr = _check_recording_contract(raw, contract)
        if tr['id'] in ids:
            raise ValueError('duplicate case id')
        ids.add(tr['id'])
        if contract['case_group'] in groups and groups[contract['case_group']] != tr['split']:
            raise ValueError('physical case crosses splits')
        groups[contract['case_group']] = tr['split']
        dest = out / 'sources' / f'{i:04d}'
        dest.mkdir(parents=True)
        for entry in tr['snapshots']:
            origin = inside(src, entry['path'])
            if file_hash(origin) != entry['sha256']:
                raise ValueError('recording changed')
            target = inside(dest, entry['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
        write_json(dest / 'ldu_sequence.json', raw)
        write_json(dest / 'case_contract.json', contract)
        converted = out / 'converted' / f'{i:04d}'
        m = import_finalized_ldu(dest, converted)
        for trajectory in m['trajectories']:
            for entry in trajectory['snapshots']:
                entry['path'] = str((converted / entry['path']).relative_to(out))
            trajectory['case_contract_sha256'] = file_hash(dest / 'case_contract.json')
            trajectory['chemistry_sha256'] = contract['chemistry_sha256']
            merged['trajectories'].append(trajectory)
        merged['producers'].append(dict(case_id=tr['id'],
            contract_path=str((dest / 'case_contract.json').relative_to(out)),
            contract_sha256=file_hash(dest / 'case_contract.json'),
            raw_manifest_sha256=file_hash(dest / 'ldu_sequence.json'),
            raw_manifest_path=str((dest / 'ldu_sequence.json').relative_to(out)),
            producer=raw.get('producer')))
    write_json(out / 'sequence_manifest.json', merged)
    load_manifest(out)  # includes cross-split normalized-operator leakage checks
    for split in {tr['split'] for tr in merged['trajectories']}:
        load_trajectories(out, split)  # witnesses were checked during import; check ordering
    return merged


def verify_dataset(root):
    root = Path(root)
    m = load_manifest(root)
    if m.get('source_kind') != 'external_cfd' or not m.get('producers'):
        raise ValueError('use imported H2 CFD recordings')
    for p in m['producers']:
        path = inside(root, p['contract_path'])
        if file_hash(path) != p['contract_sha256']:
            raise ValueError('case provenance changed')
        contract = validate_case_contract(read(path))
        raw_path = inside(root, p['raw_manifest_path'])
        if file_hash(raw_path) != p['raw_manifest_sha256']:
            raise ValueError('source manifest changed')
        raw = read(raw_path)
        tr = _check_recording_contract(raw, contract)
        for entry in tr['snapshots']:
            if file_hash(inside(raw_path.parent, entry['path'])) != entry['sha256']:
                raise ValueError('source recording changed')
        matches = [v for v in m['trajectories'] if v['id'] == contract['case_id']]
        if (p['case_id'] != contract['case_id'] or len(matches) != 1
                or matches[0]['case_group'] != contract['case_group']
                or matches[0]['split'] != contract['split']
                or matches[0].get('case_contract_sha256') != p['contract_sha256']
                or matches[0].get('chemistry_sha256') != contract['chemistry_sha256']):
            raise ValueError('dataset trajectory and producer contract disagree')
    if {p['case_id'] for p in m['producers']} != {tr['id'] for tr in m['trajectories']}:
        raise ValueError('missing producer contract for dataset trajectory')
    return m
