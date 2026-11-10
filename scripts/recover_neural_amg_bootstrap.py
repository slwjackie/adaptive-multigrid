#!/usr/bin/env python3
"""Copy a verified interrupted e9bfa4c run for the bootstrap-depth bug fix.

This is a narrowly pinned migration, not an option to bypass resume checks.
Completed old labels needed at most ten levels and are numerically unchanged
by raising only this safety cap. Model updates, RNG/optimizer state, validation
history and data are preserved. The original run is never modified.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from neural_amg import study


def recover(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError('Recovery output must be separate from the original run')
    if output.exists():
        raise FileExistsError('Recovery output already exists; resume that run or choose a new path')
    pins = study.read_json(Path(__file__).with_name('bootstrap_depth_recovery_sources.json'))
    config = study.load_config(run / 'config.json')
    previous = study.read_json(run / 'identity.json')
    current = study.identity(config)
    if previous.get('source') != pins['before'] or current['source'] != pins['after']:
        raise ValueError('Recovery supports only the pinned e9bfa4c -> bootstrap-depth fix sources')
    if previous.get('config') != current['config'] or previous.get('runtime') != current['runtime']:
        raise ValueError('Configuration or runtime changed; recovery rejected')
    cap = config['teacher'].get('bootstrap_max_levels', config['hierarchy'].get('max_levels', 10))
    if type(cap) is not int or cap < 10:
        raise ValueError('Recovery must not reduce the old bootstrap depth cap of ten')
    if (run / 'freeze.json').exists() or (run / 'frozen_model.pt').exists() or (run / 'evaluation.json').exists():
        raise ValueError('Frozen/evaluated runs cannot be recovered as interrupted training')
    plan = study.read_json(run / 'data_plan.json')
    if study.digest(plan) != study.digest(study._specifications(config)):
        raise ValueError('Data plan changed')
    for split in plan:
        if split not in ('train', 'validation') and ((run / f'data_{split}.json').exists()
                                                    or (run / 'data' / split).exists()):
            raise ValueError('Holdout data must not exist before recovery')
    data, seen = {}, set()
    for split in ('train', 'validation'):
        records = study.read_json(run / f'data_{split}.json')['records']
        if len(records) != len(plan[split]):
            raise ValueError('Development data are incomplete')
        for record, spec in zip(records, plan[split]):
            if study.digest(record['spec']) != study.digest(spec):
                raise ValueError('Cached data specification changed')
            case = study._load_case(run, record)
            if (study.operator_digest(case.a) != record['operator_digest']
                    or study.operator_digest(case.a, scale_invariant=True) != record['normalized_operator_digest']):
                raise ValueError('Cached operator identity changed')
            group = record['normalized_operator_digest']
            if group in seen:
                raise ValueError('Duplicate development operator')
            seen.add(group)
        data[split] = records
    checkpoint = run / 'training.pt'
    checkpoint_hash = study.file_hash(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['identity'] != previous or state['development_digest'] != study.digest(data):
        raise ValueError('Training checkpoint identity or development data differ')
    if state['complete']:
        raise ValueError('Recovery is for interrupted training only')
    if study.digest(study.read_json(run / 'validation.json')) != study.digest(state['history']):
        raise ValueError('Validation history differs from training checkpoint')
    valid_names = {r['spec']['name'] for r in data['train']}
    labels = {}
    for checksum in sorted((run / 'labels').glob('*.json')):
        if checksum.stem not in valid_names:
            raise ValueError('Unknown teacher cache')
        label = checksum.with_suffix('.pt')
        expected = study.read_json(checksum)['sha256']
        if study.file_hash(label) != expected:
            raise ValueError('Teacher cache checksum mismatch')
        labels[label.name] = expected
    print(f'Verified {state["updates"]} updates and {len(labels)} teacher caches; copying run.', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.recovering-', dir=output.parent))
    try:
        shutil.copytree(run, staging, dirs_exist_ok=True)
        if study.file_hash(staging / 'training.pt') != checkpoint_hash:
            raise ValueError('Checkpoint changed during recovery; stop training first')
        state['identity'] = current
        study.atomic_torch(staging / 'training.pt', state)
        study.atomic_json(staging / 'identity.json', current)
        report = dict(repair='bootstrap_depth_limit', original_commit=pins['commit'],
                      original_run=str(run), original_checkpoint_sha256=checkpoint_hash,
                      previous_identity=previous, new_identity=current,
                      preserved_updates=state['updates'], preserved_teacher_caches=labels,
                      old_bootstrap_max_levels=10, new_bootstrap_max_levels=cap)
        study.atomic_json(staging / 'bootstrap_depth_recovery.json', report)
        # Exercise the normal resume checks on the copied data before publishing.
        study.prepare(config, staging, resume=True)
        for name, expected in labels.items():
            if study.file_hash(staging / 'labels' / name) != expected:
                raise ValueError('Teacher cache changed during recovery')
        if output.exists():
            raise FileExistsError('Recovery output appeared during copying')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(dict(output_dir=str(output), preserved_updates=state['updates'], complete=True)))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Original interrupted run, left unchanged')
    parser.add_argument('--output-dir', required=True, help='New, nonexistent run directory')
    args = parser.parse_args()
    recover(args.run_dir, args.output_dir)
