"""Import reacting or nonreacting CFD without inventing physical provenance."""
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


H2_KIND = 'h2_combustion'
VARIABLE_DENSITY_KIND = 'variable_density_nonreacting'
FIELD_PROFILES = {
    H2_KIND: ('T', 'H2', 'U', 'p'),
    VARIABLE_DENSITY_KIND: ('T', 'He', 'N2', 'rho', 'U', 'p'),
}


def experiment_kind(value):
    """Select physics from a producer contract, preserving legacy H2 contracts."""
    if value.get('schema') == 'variable-density-case-v1':
        return VARIABLE_DENSITY_KIND
    return value.get('experiment_kind', H2_KIND)


def input_fingerprint(value):
    """Return the explicitly named input digest; inert gases have no chemistry."""
    key = 'thermophysical_sha256' if experiment_kind(value) == VARIABLE_DENSITY_KIND else 'chemistry_sha256'
    return key, value.get(key)


def validate_case_contract(value):
    """Validate a producer declaration, not independently certify flame physics."""
    if value.get('schema') not in ('h2-fixed-p-case-v1', 'variable-density-case-v1'):
        raise ValueError('h2-fixed-p-case-v1 or variable-density-case-v1 case contract required')
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
    kind = experiment_kind(value)
    if value.get('experiment_kind', kind) != kind:
        raise ValueError('case schema and experiment kind disagree')
    if value['schema'] == 'h2-fixed-p-case-v1':
        if kind != H2_KIND:
            raise ValueError('case schema and experiment kind disagree')
        expected = dict(fuel='H2', oxidizer='air', dimension=2, fixed_grid=True)
        if any(physics.get(k) != v for k, v in expected.items()):
            raise ValueError('need an actual fixed-grid 2D H2-air CFD declaration')
    else:
        expected = dict(kind=VARIABLE_DENSITY_KIND, reacting=False, dimension=2, fixed_grid=True)
        if (any(physics.get(k) != v for k, v in expected.items())
                or physics.get('reacting') is not False or physics.get('fixed_grid') is not True):
            raise ValueError('need an actual fixed-grid 2D variable-density nonreacting CFD declaration')
        if (value.get('species') != ['He', 'N2'] or type(value.get('reaction_count')) is not int
                or value['reaction_count'] != 0):
            raise ValueError('nonreacting benchmark requires He/N2 and zero reactions')
        files = value.get('thermophysical_files')
        if (not isinstance(files, list) or not files
                or any(not isinstance(p, str) or not p or Path(p).is_absolute() or '..' in Path(p).parts for p in files)):
            raise ValueError('explicit relative thermophysical input files required')
    key, fingerprint = input_fingerprint(value)
    if not re.fullmatch(r'[0-9a-f]{64}', str(fingerprint or '')):
        raise ValueError('hash the actual inputs: ' + key)
    if value.get('synthetic', False):
        raise ValueError('synthetic fields cannot be admitted as physical CFD data')
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
    # Validate the cohort before copying so a mixed primary/supplementary study
    # cannot leave a partially usable dataset behind.
    sources = [(Path(source).resolve(), validate_case_contract(read(Path(source) / 'case_contract.json')))
               for source in inputs]
    kinds = {experiment_kind(contract) for _, contract in sources}
    if len(kinds) != 1:
        raise ValueError('cannot mix experiment kinds in one dataset')
    kind = next(iter(kinds))
    out.mkdir(parents=True, exist_ok=True)
    merged = dict(schema='world-sequence-v1', source_kind='external_cfd', experiment_kind=kind,
                  physics=('producer-declared fixed-grid 2D He/N2 variable-density nonreacting CFD'
                           if kind == VARIABLE_DENSITY_KIND else 'producer-declared fixed-grid 2D H2-air reacting CFD'),
                  physics_verified=False, combustion_verified=False, trajectories=[], producers=[])
    ids, groups = set(), {}
    for i, (src, contract) in enumerate(sources):
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
            input_key, input_hash = input_fingerprint(contract)
            trajectory[input_key] = input_hash
            trajectory['experiment_kind'] = kind
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
        raise ValueError('use imported physical CFD recordings')
    kinds = set()
    for p in m['producers']:
        path = inside(root, p['contract_path'])
        if file_hash(path) != p['contract_sha256']:
            raise ValueError('case provenance changed')
        contract = validate_case_contract(read(path))
        kind = experiment_kind(contract)
        kinds.add(kind)
        input_key, input_hash = input_fingerprint(contract)
        if m.get('experiment_kind', H2_KIND) != kind:
            raise ValueError('dataset experiment kind and producer contract disagree')
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
                or matches[0].get(input_key) != input_hash
                or matches[0].get('experiment_kind', H2_KIND) != kind):
            raise ValueError('dataset trajectory and producer contract disagree')
    if len(kinds) != 1:
        raise ValueError('cannot mix experiment kinds in one dataset')
    if {p['case_id'] for p in m['producers']} != {tr['id'] for tr in m['trajectories']}:
        raise ValueError('missing producer contract for dataset trajectory')
    return m
