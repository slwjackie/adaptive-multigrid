"""Paired scratch/warm CNN arms on one immutable numerical-label dataset.

The latest optimizer/RNG checkpoint is authoritative for resuming.  Deployment
weights come from development warm validation only after the fixed training
budget completes; finishing that budget does not certify convergence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..config import MGConfig
from ..provenance import operator_digest
from .bamg_solver import ReferenceModels
from .bamg_study import _file_hash, _initialize_weight, _model_package, _write
from .bamg_training import (
    FinetuneSettings, SupervisedSettings, _aligned_labels, _atomic_save, _digest,
    _field, _operator, _plain, _settings, _shape, _teacher_cases,
    load_training_checkpoint, train_finetune, train_supervised,
)
from .bamg_transfer import WeightCNN
from .bamg_validation import WarmCheckpointSelector


VERSION = 'bamg-energy-cnn-arm-v1'


def _example_identity(examples):
    return [dict(case_id=e.get('case_id'), family=e.get('family'),
                 shape=list(_shape(e)), operator=operator_digest(_operator(e)),
                 rhs=_digest(e.get('b')), split=e.get('split'),
                 research_split=e.get('research_split'),
                 normalized_operator_digest=e.get('normalized_operator_digest'))
            for e in examples]


def _label_identity(results):
    groups = []
    for levels in _teacher_cases(results):
        group = []
        for level in levels:
            pattern, mask, target = _aligned_labels(level)
            group.append(dict(operator=operator_digest(_operator(level)),
                              shape=list(_shape(level)), columns=_digest(pattern.columns),
                              labels=_digest([mask, target]),
                              vectors=_digest(_field(level, 'vectors'))))
        groups.append(group)
    return groups


def _stage_state(path, phase, settings, manifest, spec):
    if not path.exists():
        return None
    state = load_training_checkpoint(path)
    if state['phase'] != phase or _digest(state['manifest']) != _digest(manifest):
        raise ValueError('training checkpoint arm identity mismatch')
    if _digest(state['settings']) != _digest(settings) or state['specs'] != dict(weight=spec):
        raise ValueError('training checkpoint settings/model mismatch')
    if any(state['signature'].get(key + '_hash') != _digest(state[key])
           for key in ('version', 'phase', 'settings', 'source', 'manifest', 'specs', 'placement')):
        raise ValueError('training checkpoint signature mismatch')
    if state['status'] == 'failed':
        raise ValueError('a failed training checkpoint cannot be silently resumed')
    update = state['update']
    if (type(update) is not int or not 0 <= update <= settings['updates'] or
            len(state['history']) != update or
            (state['status'] == 'complete') != (update == settings['updates'])):
        raise ValueError('inconsistent training checkpoint progress')
    return state


def _compact_history(history):
    keys = ('update', 'loss', 'gradient_norm', 'case', 'cycles', 'probe_kinds',
            'fallback_used', 'fixed_support', 'normalization', 'teacher_forcing',
            'hard_topk_gradient', 'mask_learned', 'levels', 'warm_validation')
    return [{key: record[key] for key in keys if key in record} for record in history]


def train_arm(output, *, arm, seed, settings, train, validation, teacher_results,
              init_weight, provenance, resume=False, stop_after=None):
    """Run one paired arm, or at most ``stop_after`` additional total updates.

    ``provenance['init_weight_sha256']`` must bind the verified source deployment
    file.  Both arms use that same source identity, but only ``warm`` imports its
    Weight CNN tensors.  ``seed`` controls initialization and the paired training
    sampling/probe streams, independently of the already prepared shared data.

    Outputs have ``status`` (``paused``/``complete``), ``completed_updates``,
    ``paths`` and optional ``selection``.  Resume validates inputs even when an
    arm has already finished.  A zero invocation budget is permitted.
    """
    if arm not in ('scratch', 'warm'):
        raise ValueError('arm must be scratch or warm')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError('stop_after must be a nonnegative invocation budget')
    settings = deepcopy(settings)
    if settings.get('support_mode', 'fixed_bilinear') != 'fixed_bilinear':
        raise ValueError('paired training requires fixed bilinear support')
    fine_values = settings['finetune']
    if (fine_values.get('cycle_loss_weighting', 'uniform') != 'uniform' or
            any(fine_values.get(k) is not None for k in
                ('custom_weights', 'time_weights', 'rate_objective', 'time_profile'))):
        raise ValueError('paired training requires the uniform residual objective')
    supervised = asdict(_settings(dict(settings['supervised'], seed=seed,
                                      l1_cap=settings['l1_cap']), SupervisedSettings))
    finetune = asdict(_settings(dict(fine_values, seed=seed, l1_cap=settings['l1_cap'],
                                    cycle_loss_weighting='uniform'), FinetuneSettings))
    if supervised['alpha_mask'] != 0:
        raise ValueError('paired Weight CNN training requires alpha_mask=0')
    init_weight = Path(init_weight).resolve()
    init_hash = _file_hash(init_weight)
    if provenance.get('init_weight_sha256') != init_hash:
        raise ValueError('initialization checkpoint SHA256 does not match provenance')
    train_identity = _example_identity(train)
    label_identity = _label_identity(teacher_results)
    if (len(train_identity) != len(label_identity) or
            any(example['operator'] != labels[0]['operator']
                for example, labels in zip(train_identity, label_identity))):
        raise ValueError('shared teacher labels must align with every TRAIN operator in order')
    identity = _plain(dict(version=VERSION, arm=arm, seed=seed, settings=settings,
        supervised=supervised, finetune=finetune,
        train=train_identity, validation=_example_identity(validation),
        teacher=label_identity, provenance=deepcopy(provenance),
        init_weight_sha256=init_hash, training_budget='fixed_updates',
        convergence_certified=False))
    identity_hash = _digest(identity)
    output = Path(output).resolve()
    filenames = dict(identity='identity.json', initialization='initialization.json',
                     supervised_checkpoint='pretrain.pt', supervised_models='supervised_models.pt',
                     supervised_history='pretrain_history.json',
                     finetune_checkpoint='finetune.pt', best_warm='best_warm.pt',
                     finetune_history='finetune_history.json',
                     warm_validation='warm_validation.json', deployment='deployment.pt',
                     completion='completion.json')
    paths = {key: output / name for key, name in filenames.items()}
    if paths['identity'].exists():
        old = json.loads(paths['identity'].read_text())
        if old != identity:
            raise ValueError('arm provenance/configuration mismatch; use a new output directory')
        if not resume:
            raise ValueError('arm output exists; use resume or a new output directory')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('nonempty arm output has no identity manifest')
        output.mkdir(parents=True, exist_ok=True)
        _write(paths['identity'], identity)
    manifest = dict(arm_identity_sha256=identity_hash, arm=arm, seed=seed)

    # Initialization occurs before the training helpers seed their sampling RNG.
    torch.manual_seed(seed)
    weight = WeightCNN(**settings['models']['weight']).double()
    transfer = (dict(_initialize_weight(weight, init_weight), seed=seed) if arm == 'warm'
                else dict(transfer='random_initialization', seed=seed,
                          optimizer_reused=False, training_updates_reused=False))
    if arm == 'warm' and transfer['sha256'] != init_hash:
        raise ValueError('initialization checkpoint changed before weight transfer')
    transfer.update(all_layers_trainable=all(p.requires_grad for p in weight.parameters()),
                    initial_weight_state_sha256=_digest(weight.state_dict()),
                    source_checkpoint_sha256=init_hash)
    if not transfer['all_layers_trainable']:
        raise ValueError('all Weight CNN layers must remain trainable')
    if paths['initialization'].exists():
        if json.loads(paths['initialization'].read_text()) != _plain(transfer):
            raise ValueError('CNN initialization differs from the saved arm')
    else:
        _write(paths['initialization'], transfer)

    latest_supervised = _stage_state(paths['supervised_checkpoint'], 'supervised',
                                     supervised, manifest, weight.spec)
    latest_fine = _stage_state(paths['finetune_checkpoint'], 'finetune',
                              finetune, manifest, weight.spec)
    if latest_fine is not None and (latest_supervised is None or
                                    latest_supervised['status'] != 'complete'):
        raise ValueError('fine-tuning checkpoint lacks a completed supervised stage')
    if paths['completion'].exists():
        completed = json.loads(paths['completion'].read_text())
        if (completed.get('identity_sha256') != identity_hash or latest_fine is None or
                latest_fine['status'] != 'complete'):
            raise ValueError('inconsistent completed arm')
        for key, expected in completed['artifact_sha256'].items():
            if not paths[key].is_file() or _file_hash(paths[key]) != expected:
                raise ValueError('completed arm artifact mismatch: ' + key)
        return completed

    remaining = stop_after
    progress = dict(supervised=latest_supervised['update'] if latest_supervised else 0,
                    finetune=latest_fine['update'] if latest_fine else 0)

    def paused(stage):
        return dict(version=VERSION, status='paused', arm=arm, seed=seed, stage=stage,
                    identity_sha256=identity_hash, completed_updates=progress.copy(),
                    paths={k: str(v) for k, v in paths.items()}, selection=None)

    if latest_supervised is None or latest_supervised['status'] != 'complete':
        before = progress['supervised']
        history = train_supervised(None, weight, teacher_results, settings=supervised,
            checkpoint_path=paths['supervised_checkpoint'],
            resume=latest_supervised is not None, manifest=manifest, stop_after=remaining)
        progress['supervised'] = len(history)
        if remaining is not None:
            remaining -= len(history) - before
        _write(paths['supervised_history'], _compact_history(history))
        if len(history) < supervised['updates']:
            return paused('supervised')
        latest_supervised = load_training_checkpoint(paths['supervised_checkpoint'])
    else:
        weight.load_state_dict(latest_supervised['models']['weight'], strict=True)
    if not paths['supervised_history'].exists():
        _write(paths['supervised_history'], _compact_history(latest_supervised['history']))
    supervised_package = dict(version=VERSION, identity_sha256=identity_hash,
                              **_model_package(None, weight))
    if paths['supervised_models'].exists():
        package = torch.load(paths['supervised_models'], map_location='cpu', weights_only=True)
        if _digest(package) != _digest(supervised_package):
            raise ValueError('supervised model export differs from latest checkpoint')
    else:
        _atomic_save(supervised_package, paths['supervised_models'])
    if remaining == 0 and latest_fine is None:
        return paused('finetune')

    cfg = MGConfig(**settings['mg'])
    reference = ReferenceModels(None, None, k=settings['k'], l1_cap=settings['l1_cap'],
                               mg_config=cfg, caps=settings.get('caps'),
                               support_mode='fixed_bilinear')
    reference_state = reference.state_dict()
    selector = WarmCheckpointSelector(validation, train,
        ReferenceModels.from_state_dict(reference_state), cfg,
        settings=settings['warm_validation'], best_path=paths['best_warm'])
    if latest_fine is None or latest_fine['status'] != 'complete':
        history = train_finetune(weight, reference, train, settings=finetune, mg_config=cfg,
            checkpoint_path=paths['finetune_checkpoint'], resume=latest_fine is not None,
            manifest=manifest, stop_after=remaining, warm_selector=selector)
        progress['finetune'] = len(history)
        _write(paths['finetune_history'], _compact_history(history))
        _write(paths['warm_validation'], selector.report())
        if len(history) < finetune['updates']:
            return paused('finetune')
        latest_fine = load_training_checkpoint(paths['finetune_checkpoint'])
    else:
        selector.load_state_dict(latest_fine['warm_selection_state'], latest_fine['update'])
        selector.export_best()  # Repair best export after a latest/best save interruption.
        _write(paths['finetune_history'], _compact_history(latest_fine['history']))
        _write(paths['warm_validation'], selector.report())
    weight.load_state_dict(selector.best['weight_state'], strict=True)
    selection = dict(mode='best_warm', update=selector.best['update'],
                     best_update=selector.best['update'],
                     metrics=deepcopy(selector.best['metrics']), contract=selector.contract,
                     eligible=bool(selector.best['metrics']['eligible']),
                     budget_updates=finetune['updates'],
                     best_at_budget_boundary=selector.best['update'] == finetune['updates'],
                     training_budget='fixed_updates', convergence_certified=False)
    _atomic_save(dict(version=VERSION, arm=arm, seed=seed, identity_sha256=identity_hash,
                     reference=reference_state, weight_spec=weight.spec,
                     weight_state=weight.state_dict(), settings=settings,
                     weighting='uniform', selection=selection), paths['deployment'])
    if _file_hash(init_weight) != init_hash:
        raise ValueError('initialization checkpoint changed during training')
    progress = dict(supervised=supervised['updates'], finetune=finetune['updates'])
    completed = dict(version=VERSION, status='complete', arm=arm, seed=seed,
                     identity_sha256=identity_hash, completed_updates=progress,
                     paths={k: str(v) for k, v in paths.items()}, selection=selection,
                     artifact_sha256={k: _file_hash(v) for k, v in paths.items()
                                      if k not in ('completion', 'identity')})
    _write(paths['completion'], completed)
    return completed
