#!/usr/bin/env python3
"""Initialize a new cancellation-regularized experiment from a verified v2 run.

Only the latest model weights and development data are reused. Fine-tuning,
optimizer state, random streams and validation selection start afresh. The
source run remains untouched and its audit records are copied into the new run.
"""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from neural_amg import study


def _verify_source(run, config, previous, pins):
    if config.get('version') != 'neural-amg-p1-study-v2':
        raise ValueError('Initialization requires the pinned v2 experiment')
    if previous.get('source') != pins['source']:
        raise ValueError('Initialization supports only the pinned d5257e2 source')
    if previous.get('config') != study.digest(config):
        raise ValueError('Source configuration identity differs')
    if previous.get('runtime') != study.identity(config)['runtime']:
        raise ValueError('Source runtime versions changed')
    if any((run / name).exists() for name in
           ('freeze.json', 'frozen_model.pt', 'evaluation.json')):
        raise ValueError('Frozen/evaluated sources are not development-only runs')
    plan = study.read_json(run / 'data_plan.json')
    if study.digest(plan) != study.digest(study._specifications(config)):
        raise ValueError('Source data plan changed')
    for split in plan:
        if split not in ('train', 'validation') and (
                (run / f'data_{split}.json').exists() or (run / 'data' / split).exists()):
            raise ValueError('Source contains holdout data')
    data, groups = {}, set()
    for split in ('train', 'validation'):
        records = study.read_json(run / f'data_{split}.json')['records']
        if len(records) != len(plan[split]):
            raise ValueError('Source development data are incomplete')
        for record, spec in zip(records, plan[split]):
            if study.digest(record['spec']) != study.digest(spec):
                raise ValueError('Source data specification changed')
            case = study._load_case(run, record)
            group = study.operator_digest(case.a, scale_invariant=True)
            if (study.operator_digest(case.a) != record['operator_digest']
                    or group != record['normalized_operator_digest']):
                raise ValueError('Source operator identity changed')
            if group in groups:
                raise ValueError('Duplicate source development operator')
            groups.add(group)
        data[split] = records
    return plan, data


def start(run, output, config_path):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError('New output must be separate from the source run')
    if output.exists():
        raise FileExistsError('Output exists; resume it or select a new directory')
    config = study.load_config(config_path)
    if config['finetune']['epochs'] < 1:
        raise ValueError('Initialization requires at least one new fine-tuning epoch')
    previous_config = study.read_json(run / 'config.json')
    previous_identity = study.read_json(run / 'identity.json')
    pins = study.read_json(Path(__file__).with_name('cancellation_start_sources.json'))
    plan, data = _verify_source(run, previous_config, previous_identity, pins)
    if study.digest(study._specifications(config)) != study.digest(plan):
        raise ValueError('New experiment must retain the development and holdout specifications')
    old_architecture = {k: v for k, v in previous_config['model'].items() if k != 'l1_cap'}
    new_architecture = {k: v for k, v in config['model'].items() if k != 'l1_cap'}
    if old_architecture != new_architecture:
        raise ValueError('Model architecture differs from the source checkpoint')
    checkpoint = run / 'training.pt'
    checkpoint_hash = study.file_hash(checkpoint)
    old = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (old.get('identity') != previous_identity
            or old.get('development_digest') != study.digest(data)):
        raise ValueError('Source checkpoint identity or development data differ')
    if old.get('phase') not in (1, 2):
        raise ValueError('Finish supervised training before initializing fine-tuning')
    if any(type(old.get(key)) is not int or old[key] < 0 for key in ('updates', 'epoch', 'offset')):
        raise ValueError('Source training position is invalid')
    if old['updates'] < previous_config['supervised']['epochs'] * len(data['train']):
        raise ValueError('Source checkpoint has not completed supervised updates')
    completed_supervised = previous_config['supervised']['epochs'] * len(data['train'])
    if old['phase'] == 1:
        if old['epoch'] > previous_config['finetune']['epochs'] or old['offset'] > len(data['train']):
            raise ValueError('Source fine-tuning position exceeds the schedule')
        expected_updates = completed_supervised + old['epoch'] * len(data['train']) + old['offset']
    else:
        expected_updates = completed_supervised + previous_config['finetune']['epochs'] * len(data['train'])
    if old['updates'] != expected_updates:
        raise ValueError('Source update count disagrees with its training position')
    if study.digest(study.read_json(run / 'validation.json')) != study.digest(old['history']):
        raise ValueError('Source validation history differs from the checkpoint')
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    model = study._model(config)
    model.load_state_dict(old['model'], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Source model weights are nonfinite')
    audit_names = ['config.json', 'identity.json', 'validation.json']
    audit_names += [name for name in ('bootstrap_depth_recovery.json', 'teacher_l1_recovery.json')
                    if (run / name).exists()]
    snapshots = {name: study.file_hash(run / name) for name in audit_names}
    expected_identity = previous_identity
    for name in ('teacher_l1_recovery.json', 'bootstrap_depth_recovery.json'):
        if name in snapshots:
            audit = study.read_json(run / name)
            if audit.get('new_identity') != expected_identity:
                raise ValueError('Source recovery audit chain differs from checkpoint')
            expected_identity = audit['previous_identity']
    current_identity = study.identity(config)
    report = dict(
        kind='cancellation_finetune_initialization', source_commit=pins['commit'],
        source_run=str(run), source_checkpoint_sha256=checkpoint_hash,
        source_updates=old['updates'], source_position={
            key: old[key] for key in ('phase', 'epoch', 'offset', 'complete')},
        previous_identity=previous_identity, new_identity=current_identity,
        reused_model='latest', optimizer_reset=True, validation_reset=True,
        new_finetune_epochs=config['finetune']['epochs'],
        archived_files={f'initialization/{name}': sha for name, sha in snapshots.items()})
    phase, epoch, offset = old['phase'], old['epoch'], old['offset']
    if phase == 1 and epoch < previous_config['finetune']['epochs'] and offset < len(data['train']):
        order = np.random.default_rng(np.random.SeedSequence(
            [previous_config['seed'], phase, epoch])).permutation(len(data['train']))
        report['source_next_case'] = data['train'][int(order[offset])]['spec']['name']
    state = dict(
        identity=current_identity, development_digest=study.digest(data), phase=1,
        epoch=0, offset=0, updates=0, complete=False, best_score=None,
        best_model=None, history=[], model=study._state(model), optimizer=None,
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
        python_rng=random.getstate(), cancellation_start_digest=study.digest(report))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.initializing-', dir=output.parent))
    try:
        study.atomic_json(staging / 'config.json', config)
        study.atomic_json(staging / 'identity.json', current_identity)
        study.atomic_json(staging / 'data_plan.json', plan)
        for split in ('train', 'validation'):
            shutil.copytree(run / 'data' / split, staging / 'data' / split)
            shutil.copy2(run / f'data_{split}.json', staging / f'data_{split}.json')
        (staging / 'initialization').mkdir()
        for name, expected in snapshots.items():
            shutil.copy2(run / name, staging / 'initialization' / name)
            if study.file_hash(staging / 'initialization' / name) != expected:
                raise ValueError('Source audit records changed during copying')
        study.atomic_json(staging / 'validation.json', [])
        study.atomic_json(staging / 'cancellation_start.json', report)
        study.atomic_torch(staging / 'training.pt', state)
        copied_data = study.prepare(config, staging, resume=True)
        if study.digest(copied_data) != state['development_digest']:
            raise ValueError('Development manifests changed during copying')
        if study.file_hash(checkpoint) != checkpoint_hash:
            raise ValueError('Source checkpoint changed; stop its training before initialization')
        if any(study.file_hash(run / name) != sha for name, sha in snapshots.items()):
            raise ValueError('Source audit records changed during copying')
        if output.exists():
            raise FileExistsError('Output appeared during initialization')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(dict(output_dir=str(output), source_updates=old['updates'],
                          new_updates=0, new_finetune_epochs=config['finetune']['epochs'])))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Original v2 run, preserved')
    parser.add_argument('--output-dir', required=True, help='New experiment directory')
    parser.add_argument('--config', required=True, help='Cancellation-regularized configuration')
    args = parser.parse_args()
    start(args.run_dir, args.output_dir, args.config)
