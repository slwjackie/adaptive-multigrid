"""Opt-in fixed-P H2 study: import -> train H_S -> replay -> coupled CFD.

World Model training is intentionally absent. Validation first determines
whether H_S offers state-dependent speed benefits worth controlling.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from ...provenance import hardware_environment, operator_digest
from ..config import AdaptiveConfig
from ..limited import initialize_timing_runtime
from ..models import Components
from ..world_model.data import digest, file_hash, inside, load_manifest, load_trajectories, write_json
from . import VERSION
from .data import import_recordings, read, verify_dataset


_FROZEN_ARTIFACTS = (
    'run_manifest.json', 'configuration.json', 'data/sequence_manifest.json',
    'expert/initial.pt', 'expert/candidate.pt', 'expert/status.json',
    'expert/training_manifest.json', 'expert/training.json', 'validation/replay.json',
)


def source_digest():
    package = Path(__file__).resolve().parents[2]
    repo = package.parents[1]
    h = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        relative = path.relative_to(package)
        # Reports/checkpoints and Python caches are not executable package
        # sources. In particular, creating a run must not invalidate itself.
        excluded = {'__pycache__', 'artifacts', 'outputs', 'runs', 'build', 'dist'}
        if any(part in excluded or part.startswith('.') for part in relative.parts):
            continue
        h.update(('package/' + relative.as_posix()).encode() + b'\0')
        h.update(path.read_bytes())
    # Pin the coupled bridge and case runners, never their generated cases,
    # compiler outputs, logs or downloaded dependencies. Config is pinned by
    # configuration_digest separately. These paths are absent in wheel installs.
    native_sources = (
        'scripts/run_v6_7_h2_fixed_p.py',
        'integrations/openfoam13/Allwmake',
        'integrations/openfoam13/AdaptiveFixedP/AdaptiveFixedP.C',
        'integrations/openfoam13/AdaptiveFixedP/AdaptiveFixedP.H',
        'integrations/openfoam13/AdaptiveFixedP/Wire.hpp',
        'integrations/openfoam13/AdaptiveFixedP/Make/files',
        'integrations/openfoam13/AdaptiveFixedP/Make/options',
        'integrations/openfoam13/case/generate.py',
        'integrations/openfoam13/case/run_case.py',
        'integrations/openfoam13/case/Allrun',
        'integrations/openfoam13/case/Allclean',
    )
    for name in native_sources:
        path = repo / name
        if path.is_file():
            h.update(name.encode() + b'\0')
            h.update(path.read_bytes())
    return h.hexdigest()


def configuration(path):
    settings = read(path)
    if settings.get('schema') != VERSION:
        raise ValueError('use an h2-fixed-p-study-v1 configuration')
    cfg = AdaptiveConfig.from_dict(settings['solver'])
    if (cfg.use_transfer or cfg.application != 'replace' or not cfg.use_smoother
            or cfg.branch != 'H_S' or cfg.spatial or cfg.use_learned_controller):
        raise ValueError('fixed-P H_S requires no learned transfer/spatial gate/controller')
    if settings.get('world_model_enabled', False) is not False:
        raise ValueError('World Model is not part of this fixed-P H_S study')
    if cfg.inference_dtype != 'float32':
        raise ValueError('this study supports FP32 neural checkpoints with FP64 numerical cycles only')
    if (cfg.replace_pre + cfg.replace_post < 1 or cfg.mg.smoother_gain_multiplier <= 0
            or cfg.smoother_levels == () or (cfg.smoother_levels is None and cfg.mg.nn_levels == 0)):
        raise ValueError('fixed-P H_S study requires active neural smoothing')
    if cfg.mg.residual_reference != 'initial':
        raise ValueError('shared raw L2 initial-residual stopping criterion required')
    threads = settings.get('torch_threads', 1)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('positive torch_threads required')
    torch.set_num_threads(threads)
    initialize_timing_runtime()
    return settings, cfg


def prepare(config, inputs, run_dir):
    settings, cfg = configuration(config)
    out = Path(run_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('new study run directory required')
    out.mkdir(parents=True, exist_ok=True)
    imported = import_recordings(inputs, out / 'data')
    splits = {t['split'] for t in imported['trajectories']}
    if not {'train', 'validation'}.issubset(splits):
        raise ValueError('independent train and validation CFD cases are required')
    write_json(out / 'configuration.json', settings)
    manifest = dict(schema=VERSION, source_digest=source_digest(),
                    configuration_digest=digest(settings),
                    dataset_sha256=file_hash(out / 'data/sequence_manifest.json'),
                    source_kind='external_cfd', combustion_verified=False,
                    world_model_enabled=False, test_seen=False,
                    split_counts={s:sum(t['split'] == s for t in imported['trajectories']) for s in splits})
    write_json(out / 'run_manifest.json', manifest)
    return manifest


def load_run(run_dir):
    out = Path(run_dir).resolve()
    settings, cfg = configuration(out / 'configuration.json')
    m = read(out / 'run_manifest.json')
    if (m.get('schema') != VERSION or m['source_digest'] != source_digest()
            or m['configuration_digest'] != digest(settings)
            or m['dataset_sha256'] != file_hash(out / 'data/sequence_manifest.json')):
        raise ValueError('source/config/data manifest changed; create a new run')
    verify_dataset(out / 'data')
    return out, settings, cfg, m


def _recorded_training_identity(root, entry):
    """Stream one admitted snapshot's identity without repeating SPD admission.

    prepare/train already perform the matrix admission. A pinned file hash
    proves these are the same bytes; service startup only needs identity
    binding, not another Cholesky/eigensolve or an in-memory CFD trajectory.
    All arrays are released when this function returns its small digest dict.
    """
    path = inside(root, entry['path'])
    if file_hash(path) != entry['sha256']:
        raise ValueError('training snapshot hash changed')
    with np.load(path, allow_pickle=False) as raw:
        meta = json.loads(str(raw['metadata']))
        b, x0 = raw['b'], raw['x0']
        if (meta.get('source_kind') != 'external_cfd' or b.ndim != 1 or x0.shape != b.shape):
            raise ValueError('invalid recorded training snapshot identity')
        a = sp.csr_matrix((raw['data'], raw['indices'], raw['indptr']), shape=(len(b), len(b)))
        matrix_hash = operator_digest(a)
        if matrix_hash != entry['matrix_digest']:
            raise ValueError('training matrix digest changed')
        return dict(index=meta['index'], time=meta['time'], matrix_digest=matrix_hash,
                    rhs_digest=hashlib.sha256(np.asarray(b, dtype=np.float64).tobytes()).hexdigest(),
                    x0_digest=hashlib.sha256(np.asarray(x0, dtype=np.float64).tobytes()).hexdigest())


def load_expert(out, cfg):
    path = Path(out) / 'expert/candidate.pt'
    status = read(path.parent / 'status.json')
    if status.get('status') != 'complete' or status['checkpoint_sha256'] != file_hash(path):
        raise ValueError('incomplete or modified H_S checkpoint')
    model = Components.load(path)
    updates = model.metadata.get('optimizer_updates', 0)
    if (model.metadata.get('fixed_p_contract') != 'h2-fixed-classical-p-v1'
            or model.metadata.get('training_source_kind') != 'external_cfd'
            or model.metadata.get('training_branch') != 'H_S'
            or isinstance(updates, bool) or not isinstance(updates, int) or updates < 1):
        raise ValueError('this workflow needs H_S trained on recorded CFD with fixed P')
    trained = read(path.parent / 'training_manifest.json')
    expected = replace(cfg, mode='research', branch='H_S', use_smoother=True,
                       use_transfer=False, spatial=False, gate_mode='open', use_learned_controller=False)
    actual = AdaptiveConfig.from_dict(trained['config'])
    # Offline training resolves 'auto' once for optimizer-state consistency.
    # It is not a change to interpolation, the numerical cycle, or NN dtype.
    if expected.inference_device == 'auto' and actual.inference_device in ('cpu', 'cuda', 'mps'):
        actual = replace(actual, inference_device='auto')
    if actual != expected:
        raise ValueError('checkpoint solver/parent configuration mismatch')
    saved_fingerprint = trained.pop('training_fingerprint')
    if digest(trained) != saved_fingerprint or model.metadata['training_fingerprint'] != saved_fingerprint:
        raise ValueError('training evidence changed')
    records = read(path.parent / 'training.json')
    if (status.get('training_fingerprint') != saved_fingerprint
            or status.get('optimizer_updates') != updates or len(records) != updates
            or len(trained.get('schedule', [])) != updates):
        raise ValueError('optimizer-update evidence mismatch')
    final = model.component_signatures()
    initial = Components.load(path.parent / 'initial.pt').component_signatures()
    if (status.get('final_signatures') != final or status.get('initial_signatures') != initial
            or trained.get('initial_signatures') != initial or initial['smoother'] == final['smoother']
            or any(initial[key] != final[key] for key in ('transfer', 'detector', 'controller'))):
        raise ValueError('only H_S may change during this training workflow')
    # Bind imported training data to the model evidence. A valid candidate from
    # a different run with the same solver configuration cannot be substituted.
    root = Path(out) / 'data'
    run_manifest = read(Path(out) / 'run_manifest.json')
    if run_manifest['dataset_sha256'] != file_hash(root / 'sequence_manifest.json'):
        raise ValueError('study data manifest changed')
    dataset = load_manifest(root)  # metadata only: does not open snapshot arrays
    expected_trajectories = {meta['id']: meta for meta in dataset['trajectories'] if meta['split'] == 'train'}
    if {tr['id'] for tr in trained['trajectories']} != set(expected_trajectories):
        raise ValueError('checkpoint training cohort differs from this study')
    samples = {}
    for tr in trained['trajectories']:
        meta = expected_trajectories[tr['id']]
        if tr['case_group'] != meta['case_group'] or len(tr['samples']) != len(meta['snapshots']):
            raise ValueError('checkpoint training cases differ from this study')
        for evidence, entry in zip(tr['samples'], meta['snapshots']):
            expected_sample = _recorded_training_identity(root, entry)
            if any(evidence.get(k) != v for k, v in expected_sample.items()):
                raise ValueError('checkpoint training snapshot differs from imported CFD recording')
            samples[(tr['id'], expected_sample['index'])] = evidence
    for step, record in enumerate(records):
        scheduled = trained['schedule'][step]
        evidence = samples.get((record.get('trajectory'), record.get('snapshot_index')))
        gradient = record.get('gradient_norm')
        if (record.get('step') != step or evidence is None
                or scheduled != dict(trajectory=record.get('trajectory'), index=record.get('snapshot_index'))
                or not isinstance(gradient, (int, float)) or not np.isfinite(gradient) or gradient <= 0
                or record.get('matrix_digest') != evidence['matrix_digest']
                or record.get('rhs_digest') != evidence['rhs_digest']
                or record.get('exported_x0_digest') != evidence['x0_digest']
                or record.get('fixed_p_digest') != evidence['fixed_p_digest']):
            raise ValueError('optimizer record does not match actual training data')
    return model


def train(run_dir):
    from .training import train_smoother
    out, settings, cfg, _ = load_run(run_dir)
    if (out / 'freeze.json').exists():
        raise ValueError('frozen study cannot retrain')
    _, status = train_smoother(load_trajectories(out / 'data', 'train'), cfg, out / 'expert',
                              **settings.get('training', {}))
    return status


def evaluate(run_dir, *, split='validation', repeats=3):
    from .evaluation import evaluate_replay
    if split not in ('validation', 'test'):
        raise ValueError('paired evaluation requires validation or test')
    out, settings, cfg, m = load_run(run_dir)
    target = out / split / 'replay.json'
    if target.exists():
        raise FileExistsError('evaluation already exists; preserve it and use a new study for retuning')
    if split == 'test':
        check_freeze(out)
    elif (out / 'freeze.json').exists():
        raise ValueError('development evaluations closed after freeze')
    model = load_expert(out, cfg)
    result = evaluate_replay(load_trajectories(out / 'data', split), cfg, model,
                            repeats=repeats, split=split,
                            seed=int(settings.get('seed', 20261010)), source_metadata=m)
    result['evidence'] = dict(checkpoint_sha256=file_hash(out / 'expert/candidate.pt'),
        dataset_sha256=m['dataset_sha256'], configuration_digest=m['configuration_digest'],
        source_digest=m['source_digest'], hardware=hardware_environment(refresh=True))
    write_json(target, result)
    return result


def freeze(run_dir):
    out, _, cfg, manifest = load_run(run_dir)
    load_expert(out, cfg)
    path = out / 'freeze.json'
    if path.exists():
        raise FileExistsError('already frozen')
    validation = out / 'validation/replay.json'
    if not validation.exists():
        raise ValueError('run independent validation before freezing')
    report = read(validation)
    evidence = report.get('evidence', {})
    if (report.get('schema') != 'h2-fixed-p-replay-evaluation-v1' or report.get('split') != 'validation'
            or evidence.get('checkpoint_sha256') != file_hash(out / 'expert/candidate.pt')
            or any(evidence.get(k) != manifest[k] for k in
                   ('dataset_sha256', 'configuration_digest', 'source_digest'))):
        raise ValueError('validation evidence is not bound to this configuration/checkpoint')
    value = dict(schema=VERSION, world_model_enabled=False,
                 artifacts={name: file_hash(out / name) for name in _FROZEN_ARTIFACTS},
                 source_digest=source_digest(),
                 claim='configuration/checkpoint frozen; no speed or physical-validity guarantee')
    write_json(path, value)
    return value


def check_freeze(run_dir):
    out = Path(run_dir)
    f = read(out / 'freeze.json')
    if (f.get('schema') != VERSION or f.get('world_model_enabled') is not False
            or set(f.get('artifacts', {})) != set(_FROZEN_ARTIFACTS)):
        raise ValueError('incomplete or invalid freeze manifest')
    if f['source_digest'] != source_digest():
        raise ValueError('source changed after freeze')
    for name, sha in f['artifacts'].items():
        if file_hash(out / name) != sha:
            raise ValueError('frozen artifact changed: ' + name)
    return f


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    s = subs.add_parser('prepare')
    s.add_argument('--run-dir', required=True)
    s.add_argument('--config', default='configs/v6_7_h2_fixed_p.json')
    s.add_argument('--input-ldu', nargs='+', required=True)
    for name in ('train', 'freeze'):
        s = subs.add_parser(name); s.add_argument('--run-dir', required=True)
    s = subs.add_parser('evaluate'); s.add_argument('--run-dir', required=True)
    s.add_argument('--split', choices=['validation', 'test'], default='validation')
    s.add_argument('--repeats', type=int, default=3)
    s = subs.add_parser('serve')
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument('--run-dir'); g.add_argument('--config')
    s.add_argument('--case-contract', required=True)
    s.add_argument('--socket', required=True)
    s.add_argument('--mode', choices=['native', 'classical', 'hs'], required=True)
    s.add_argument('--record-dir'); s.add_argument('--evidence')
    s.add_argument('--timeout', type=float, default=300.)
    s = subs.add_parser('coupled')
    s.add_argument('--run-dir', required=True); s.add_argument('--case', required=True)
    s.add_argument('--output', required=True); s.add_argument('--repeats', type=int, default=3)
    s.add_argument('--timeout', type=float, default=3600.)
    a = p.parse_args(argv)
    if a.command == 'prepare': result = prepare(a.config, a.input_ldu, a.run_dir)
    elif a.command == 'train': result = train(a.run_dir)
    elif a.command == 'evaluate': result = evaluate(a.run_dir, split=a.split, repeats=a.repeats)
    elif a.command == 'freeze': result = freeze(a.run_dir)
    elif a.command == 'coupled':
        from .fullrun import run_paired
        result = run_paired(a.run_dir, a.case, a.output, repeats=a.repeats, timeout=a.timeout)
    else:
        from .service import Session, serve
        from .data import validate_case_contract
        contract = validate_case_contract(read(a.case_contract))
        if a.run_dir:
            out, settings, cfg, _ = load_run(a.run_dir)
            if contract['split'] == 'test': check_freeze(out)
        else:
            settings, cfg = configuration(a.config)
            if a.mode == 'hs': raise ValueError('H_S service requires --run-dir with trained expert')
        expert = load_expert(out, cfg) if a.mode == 'hs' else None
        return serve(a.socket, Session(cfg, contract, mode=a.mode, expert=expert,
                     recording=a.record_dir, evidence=a.evidence), timeout=a.timeout)
    print(__import__('json').dumps(result, indent=2, allow_nan=False))
    return result
