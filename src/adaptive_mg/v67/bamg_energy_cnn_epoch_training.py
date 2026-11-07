"""Epoch-shuffled orchestration for paired CNN training only.

Numerical losses, Adam updates, frozen-mask checks, warm selection and checkpoint
serialization are shared with bamg_training. The small orchestration functions
are kept separate so historical numerical-teacher source hashes remain valid.
A permutation is a pure function of (algorithm, seed, stage, epoch, case count).
The checkpoint's completed-update counter uniquely restores its epoch and cursor;
no mutable shuffle RNG is shared with residual probes or validation timing.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ..config import MGConfig
from ..provenance import operator_digest
from .bamg_training import (
    FinetuneSettings, SupervisedSettings, TrainingFailure, _aligned_labels,
    _digest, _field, _operator, _plain, _probes, _run, _settings, _shape,
    _teacher_cases, _train_only, _validate_time_profile, curriculum_cycles,
    load_training_checkpoint, supervised_objective,
)


class EpochShuffle:
    """One visit per case per epoch, reproducible from saved update progress."""
    VERSION = 'paired-epoch-shuffle-pcg64-v1'
    STAGES = {'supervised': 0, 'finetune': 1}

    def __init__(self, count, seed, stage):
        if type(count) is not int or count < 1:
            raise ValueError('Epoch shuffle requires a positive case count')
        if type(seed) is not int or seed < 0 or stage not in self.STAGES:
            raise ValueError('Invalid epoch shuffle seed or stage')
        self.count, self.seed, self.stage = count, seed, stage
        self.contract = dict(version=self.VERSION, count=count, seed=seed, stage=stage,
            cursor='divmod(checkpoint.update, count)',
            order='PCG64(SeedSequence([seed, stage_id, epoch, 0x45504f43])).permutation(count)')
        self._epoch, self._order, self._hash = None, None, None

    def select(self, update):
        if type(update) is not int or update < 0:
            raise ValueError('Epoch shuffle update must be nonnegative')
        epoch, position = divmod(update, self.count)
        if epoch != self._epoch:
            generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                [self.seed, self.STAGES[self.stage], epoch, 0x45504f43])))
            self._order = generator.permutation(self.count)
            self._hash = _digest(self._order.tolist())
            self._epoch = epoch
        return int(self._order[position]), dict(sampling='epoch_shuffle',
            epoch=epoch + 1, epoch_position=position + 1, epoch_size=self.count,
            permutation_sha256=self._hash)

    def require_budget(self, updates):
        if type(updates) is not int or updates < self.count:
            raise ValueError('Epoch-shuffled stage budget must cover at least one full TRAIN epoch')

    def log_epoch(self, sampling, updates):
        if sampling['epoch_position'] == 1:
            total = (updates + self.count - 1) // self.count
            print(f"TRAIN {self.stage} epoch {sampling['epoch']}/{total}: "
                  f"{self.count} operators, seed={self.seed}", flush=True)

    def coverage(self, updates):
        if type(updates) is not int or updates < 0:
            raise ValueError('Coverage progress must be nonnegative')
        epochs, partial = divmod(updates, self.count)
        return dict(sampling='epoch_shuffle', cases=self.count, updates=updates,
            completed_epochs=epochs, partial_epoch_cases=partial,
            minimum_visits=epochs, maximum_visits=epochs + int(partial > 0),
            distinct_cases_seen=min(updates, self.count), all_cases_seen=epochs > 0)


def train_supervised(mask_model, weight_model, teacher_results, *, settings=None,
                     checkpoint_path=None, resume=False, manifest=None, stop_after=None):
    settings = _settings(settings, SupervisedSettings)
    groups = _teacher_cases(teacher_results)
    sampler = EpochShuffle(len(groups), settings.seed, "supervised")
    sampler.require_budget(settings.updates)
    source = []
    for levels in groups:
        group = []
        for level in levels:
            pattern, mask, target = _aligned_labels(level)
            group.append(dict(operator=operator_digest(_operator(level)), shape=_shape(level),
                              columns=_digest(pattern.columns), labels=_digest([mask, target]),
                              vectors=_digest(_field(level, "vectors"))))
        source.append(group)

    def objective(update, rng):
        case, sampling = sampler.select(update)
        sampler.log_epoch(sampling, settings.updates)
        loss, report = supervised_objective(mask_model, weight_model, groups[case], settings=settings)
        return loss, dict(case=case, **sampling, **report)

    models = dict(weight=weight_model) if mask_model is None else dict(mask=mask_model, weight=weight_model)
    return _run(models, settings, phase="supervised",
                source=dict(cases=source, sampler=sampler.contract), objective=objective, checkpoint_path=checkpoint_path,
                resume=resume, manifest=manifest, stop_after=stop_after)


def train_finetune(weight_model, reference, examples, *, settings=None, mg_config=None,
                   checkpoint_path=None, resume=False, manifest=None,
                   validation_callback=None, stop_after=None, warm_selector=None):
    """Train only theta on fixed reference masks; return all completed updates.

    ``validation_callback(update, weight_model)`` may return logging data.  Model
    choice can use ``warm_selector`` for periodic full-tolerance warm evaluation.
    Its best weights/metrics are saved inside latest before exporting a separate
    best snapshot; validation RNG use is isolated from TRAIN updates.
    ``stop_after`` pauses after that many additional updates and supports exact
    resume without changing the total-budget configuration.
    """
    from .bamg_solver import build_graph, residual_loss
    settings = _settings(settings, FinetuneSettings)
    if not examples:
        raise ValueError("at least one TRAIN example is required")
    parameter = next(weight_model.parameters())
    if parameter.device.type != "cpu" or parameter.dtype != torch.float64:
        raise ValueError("BAMG residual training uses the CPU FP64 sparse backend")
    _validate_time_profile(settings, reference, weight_model)
    if resume:
        if checkpoint_path is None or not Path(checkpoint_path).is_file():
            raise ValueError("resume requires an existing checkpoint")
        saved = load_training_checkpoint(checkpoint_path)
        saved_reference = saved.get("extra", {}).get("reference")
        if saved_reference is None or _digest(reference.state_dict()) != saved["extra"]["reference_hash"]:
            raise ValueError("checkpoint reference provenance mismatch")
        # Replay the serialized, provenance-checked masks instead of running top-k
        # on a newly instantiated reference when resuming an interrupted process.
        reference = type(reference).from_state_dict(saved_reference)
    mg_config = mg_config or reference.mg_config
    if isinstance(mg_config, Mapping):
        mg_config = MGConfig(**mg_config)
    if settings.l1_cap != reference.l1_cap:
        raise ValueError("student and frozen reference must use the same weight constraint")
    examples = list(examples)
    sampler = EpochShuffle(len(examples), settings.seed, "finetune")
    sampler.require_budget(settings.updates)
    masks, source, geometry_masks = [], [], {}
    for example in examples:
        _train_only(example)
        a, shape = _operator(example), _shape(example)
        if a.shape != (shape[0] * shape[1],) * 2:
            raise ValueError("operator/grid mismatch")
        if reference.support_mode == "fixed_bilinear" and shape in geometry_masks:
            case_masks = geometry_masks[shape]
        else:
            case_masks = [m.detach().cpu().clone() for m in reference.masks_for(a, shape)]
            if reference.support_mode == "fixed_bilinear":
                geometry_masks[shape] = case_masks
        masks.append(case_masks)
        source.append(dict(operator=operator_digest(a), shape=shape, dtype=str(a.dtype),
                           case_id=_field(example, "case_id", _field(example, "group_digest")),
                           masks=_digest(case_masks)))
    frozen_state = deepcopy(reference.state_dict())
    frozen_hash = _digest(frozen_state)
    mask_hash = _digest(masks)
    train_storage = {p.untyped_storage().data_ptr() for p in weight_model.parameters()}
    for model in (reference.mask_model, reference.weight_model):
        if model is None:
            continue
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("reference CNNs must be frozen and in eval mode")
        if train_storage & {p.untyped_storage().data_ptr() for p in model.parameters()}:
            raise ValueError("trainable weights must not share frozen-reference storage")

    def verify():
        reference.check_immutable()
        if _digest(reference.state_dict()) != frozen_hash or _digest(masks) != mask_hash:
            raise TrainingFailure("reference checkpoint or saved structural masks changed")

    def objective(update, rng):
        case, sampling = sampler.select(update)
        sampler.log_epoch(sampling, settings.updates)
        a, shape = _operator(examples[case]), _shape(examples[case])
        graph = build_graph(a, shape, weight_model, masks[case], mg_config,
                            l1_cap=settings.l1_cap, caps=reference.caps)
        x0, kinds = _probes(a, shape, graph, settings, rng, parameter)
        count = curriculum_cycles(update, settings)
        custom = settings.custom_weights
        times = settings.time_weights
        if custom is not None:
            if len(custom) < count:
                raise ValueError("custom_weights is shorter than the rollout")
            custom = custom[:count]
        if times is not None:
            if len(times) < count:
                raise ValueError("time_weights is shorter than the rollout")
            times = times[:count]
        loss, report, _ = residual_loss(graph, torch.zeros_like(x0), x0=x0, cycles=count,
                                       weighting=settings.cycle_loss_weighting,
                                       custom_weights=custom, eps=settings.eps, time_weights=times,
                                       rate_objective=settings.rate_objective)
        ratios = _plain(report.get("ratios", report.get("history", [])))
        flat = np.asarray(ratios, dtype=float)
        if flat.size and (not np.isfinite(flat).all() or np.max(flat) > settings.divergence_factor):
            raise TrainingFailure("raw V-cycle diverged; no fallback or sample omission is permitted")
        report = {key: value for key, value in report.items() if key != "residual"}
        return loss, dict(case=case, **sampling, cycles=count, probe_kinds=kinds, fallback_used=False,
                          fixed_support=True, **{k: v for k, v in report.items() if k != "cycles"})

    source_contract = dict(cases=source, reference_hash=frozen_hash, mg_config=asdict(mg_config),
                           sampler=sampler.contract)
    extra = dict(reference=reference.state_dict(include_cache=True), masks=masks, mask_hash=mask_hash,
                 reference_hash=frozen_hash, deployment_requires_reference=True)
    return _run(dict(weight=weight_model), settings, phase="finetune", source=source_contract,
                objective=objective, checkpoint_path=checkpoint_path, resume=resume,
                manifest=manifest, stop_after=stop_after, extra=extra, verify=verify,
                validation_callback=validation_callback, warm_selector=warm_selector)

