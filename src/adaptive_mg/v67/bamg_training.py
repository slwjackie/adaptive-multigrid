"""Offline two-stage direct-P learning with reproducible frozen support.

Stage one uses numerical teacher labels; stage two differentiates raw classical
V-cycles.  Neither entry point performs online adaptation or changes a mask
while optimizing the residual objective.  Checkpoints are deliberately separate
from validation model selection: incompatible objectives are never ranked here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Mapping

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F

from ..config import MGConfig
from ..grid import as_shape
from ..provenance import hardware_environment, operator_digest
from ..smoothers import LineSmootherCache, classical_smoothing_correction
from ..transfer import coarse_fine_indices
from .bamg_transfer import (
    MaskCNN, WeightCNN, direct_features, direct_pattern, geometry_constraints,
    project_weights, row_slots, sparse_from_weights,
    fixed_bilinear_mask,
)


TRAINING_VERSION = "bamg-two-stage-v1"


@dataclass(frozen=True)
class SupervisedSettings:
    updates: int = 100
    learning_rate: float = 1e-3
    seed: int = 1
    gradient_clip: float = 10.0
    alpha_mask: float = 1.0
    alpha_weight: float = 1.0
    alpha_action: float = 0.0
    normalize_weights: bool = True
    l1_cap: float = 8.0
    checkpoint_every: int = 1


@dataclass(frozen=True)
class FinetuneSettings:
    updates: int = 100
    learning_rate: float = 1e-4
    seed: int = 1
    gradient_clip: float = 10.0
    cycles: int = 3
    final_cycles: int = 6
    curriculum_step: int | None = None
    cycle_loss_weighting: str = "uniform"
    custom_weights: tuple[float, ...] | None = None
    time_weights: tuple[float, ...] | None = None
    rate_objective: str | None = None
    time_profile: dict | None = None
    probe_kinds: tuple[str, ...] = ("random", "smoothed", "hard")
    probes_per_kind: int = 1
    smoothing_sweeps: int = 4
    hard_cycles: int = 2
    eps: float = 1e-12
    l1_cap: float = 8.0
    checkpoint_every: int = 1
    divergence_factor: float = 1e12


class TrainingFailure(RuntimeError):
    """A raw rollout or gradient failed; the offending update was not omitted."""


def _settings(value, cls):
    result = cls() if value is None else cls(**value) if isinstance(value, Mapping) else value
    if not isinstance(result, cls):
        raise TypeError(f"settings must be {cls.__name__} or a mapping")
    for key in ("updates", "checkpoint_every"):
        if type(getattr(result, key)) is not int or getattr(result, key) < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "gradient_clip", "l1_cap"):
        if not np.isfinite(getattr(result, key)) or getattr(result, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if type(result.seed) is not int or result.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(result, SupervisedSettings):
        alphas = [result.alpha_mask, result.alpha_weight, result.alpha_action]
        if any(not np.isfinite(v) or v < 0 for v in alphas) or not any(alphas):
            raise ValueError("loss coefficients must be nonnegative with positive total")
    else:
        for key in ("cycles", "final_cycles", "probes_per_kind", "smoothing_sweeps", "hard_cycles"):
            if type(getattr(result, key)) is not int or getattr(result, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if result.curriculum_step is not None and (
            type(result.curriculum_step) is not int or not 0 <= result.curriculum_step <= result.updates
        ):
            raise ValueError("curriculum_step must lie between zero and updates")
        if result.cycle_loss_weighting not in {"uniform", "terminal", "custom"}:
            raise ValueError("unknown cycle_loss_weighting")
        if not result.probe_kinds or not set(result.probe_kinds) <= {"random", "smoothed", "hard"}:
            raise ValueError("probe_kinds must contain random, smoothed or hard")
        if len(set(result.probe_kinds)) != len(result.probe_kinds):
            raise ValueError("duplicate probe kind")
        if not 0 < result.eps < 1 or not np.isfinite(result.divergence_factor) or result.divergence_factor <= 1:
            raise ValueError("invalid epsilon or divergence factor")
        for name in ("custom_weights", "time_weights"):
            values = getattr(result, name)
            if values is not None and (not values or any(not np.isfinite(x) or x < 0 for x in values)
                                       or sum(values) <= 0):
                raise ValueError(f"invalid {name}")
        if result.cycle_loss_weighting == "custom" and result.custom_weights is None:
            raise ValueError("custom weighting requires custom_weights")
        if (result.time_weights is None) != (result.rate_objective is None):
            raise ValueError("rate_objective and detached deployment time_weights are required together")
        if result.rate_objective not in {None, "relative", "log"}:
            raise ValueError("rate_objective must be relative or log")
        if result.time_weights is not None and any(x <= 0 for x in result.time_weights):
            raise ValueError("deployment time_weights must be strictly positive")
    return result


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _operator(obj):
    value = _field(obj, "A", _field(obj, "a"))
    if value is None or not sp.issparse(value):
        raise ValueError("training example requires a sparse A (or a)")
    return value.tocsr()


def _shape(obj):
    value = _field(obj, "shape", _field(obj, "n"))
    if value is None:
        raise ValueError("training example requires shape (or n)")
    return as_shape(value)


def _train_only(obj):
    for key in ("split", "research_split"):
        value = _field(obj, key)
        if value is not None and str(value).lower() != "train":
            raise ValueError("training and label refitting require TRAIN examples")


def _digest(value):
    """Stable digest including tensor dtype/shape and all scalar metadata."""
    h = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            v = item.detach().cpu().contiguous()
            h.update(f"tensor:{v.dtype}:{tuple(v.shape)}:".encode())
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(f"array:{item.dtype}:{item.shape}:".encode())
            h.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, Mapping):
            h.update(b"mapping{")
            for key in sorted(item, key=str):
                visit(str(key)); visit(item[key])
            h.update(b"}")
        elif isinstance(item, (tuple, list)):
            h.update(b"sequence[")
            for child in item:
                visit(child)
            h.update(b"]")
        elif isinstance(item, np.generic):
            visit(item.item())
        else:
            h.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
            h.update(b";")

    visit(value)
    return h.hexdigest()


def _plain(value):
    if torch.is_tensor(value):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _model_contract(models):
    tensors = []
    for name, model in models.items():
        expected = MaskCNN if name == "mask" else WeightCNN
        if not isinstance(model, expected):
            raise TypeError(f"P-only training requires {expected.__name__} for {name}")
        if not all(p.requires_grad for p in model.parameters()):
            raise ValueError(f"all {name} CNN parameters must be trainable")
        tensors.extend(model.parameters())
    storage = [p.untyped_storage().data_ptr() for p in tensors]
    if len(storage) != len(set(storage)):
        raise ValueError("the mask and weight CNNs must not share parameter storage")
    if len({(p.dtype, p.device) for p in tensors}) != 1:
        raise ValueError("all trainable CNN parameters must have the same dtype/device")
    return tensors


def _rng_state(rng):
    np_state = np.random.get_state()
    return dict(local=deepcopy(rng.bit_generator.state), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=[np_state[0], np_state[1].tolist(), *np_state[2:]],
                python=random.getstate())


def _restore_rng(state, rng):
    rng.bit_generator.state = state["local"]
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    random.setstate(state["python"])


def _atomic_save(value, path):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_training_checkpoint(path):
    """Load only tensor/primitive checkpoint data; never unpickle model objects."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != TRAINING_VERSION:
        raise ValueError("unsupported BAMG training checkpoint version")
    return state


def _run(models, settings, *, phase, source, objective, checkpoint_path, resume,
         manifest, stop_after, extra=None, verify=None, validation_callback=None):
    parameters = _model_contract(models)
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError("stop_after must be a nonnegative invocation budget")
    specs = {key: deepcopy(model.spec) for key, model in models.items()}
    placement = {key: dict(dtype=str(next(model.parameters()).dtype),
                           device=str(next(model.parameters()).device))
                 for key, model in models.items()}
    contract = dict(version=TRAINING_VERSION, phase=phase, settings=asdict(settings),
                    source=source, manifest=_plain(manifest or {}), specs=specs,
                    placement=placement)
    signature = {key + "_hash": _digest(value) for key, value in contract.items()}
    optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)
    rng = np.random.default_rng(settings.seed)
    history, update = [], 0
    if resume:
        if checkpoint_path is None or not Path(checkpoint_path).is_file():
            raise ValueError("resume requires an existing checkpoint")
        saved = load_training_checkpoint(checkpoint_path)
        if saved["signature"] != signature:
            changed = [k for k in signature if saved["signature"].get(k) != signature[k]]
            raise ValueError("checkpoint provenance/configuration mismatch: " + ", ".join(changed))
        if saved["status"] == "failed":
            raise ValueError("a failed training checkpoint cannot be silently resumed")
        for key, model in models.items():
            model.load_state_dict(saved["models"][key], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"], rng)
        history, update = saved["history"], saved["update"]
        if not 0 <= update <= settings.updates or len(history) != update:
            raise ValueError("inconsistent checkpoint progress")
    else:
        torch.manual_seed(settings.seed); np.random.seed(settings.seed); random.seed(settings.seed)
    for model in models.values():
        model.train()
    if verify:
        verify()

    def save(status, failure=None):
        state = dict(**contract, signature=signature, status=status, update=update,
                     history=history, optimizer=optimizer.state_dict(), rng=_rng_state(rng),
                     models={key: model.state_dict() for key, model in models.items()},
                     extra=extra or {}, failure=failure)
        _atomic_save(state, checkpoint_path)

    end = settings.updates if stop_after is None else min(settings.updates, update + stop_after)
    try:
        while update < end:
            optimizer.zero_grad(set_to_none=True)
            loss, report = objective(update, rng)
            if not torch.is_tensor(loss) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise TrainingFailure("nonfinite or nonscalar raw training loss")
            loss.backward()
            grads = [p.grad for p in parameters if p.grad is not None]
            if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                raise TrainingFailure("missing or nonfinite P-CNN gradients")
            norm = torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise TrainingFailure("nonfinite CNN parameters after optimizer step")
            if verify:
                verify()
            update += 1
            record = dict(update=update, loss=float(loss.detach()), gradient_norm=float(norm),
                          **_plain(report))
            if validation_callback is not None:
                # Validation may sample probes, but it must not perturb the TRAIN RNG.
                before = _rng_state(rng)
                try:
                    record["validation"] = _plain(validation_callback(update, models["weight"]))
                finally:
                    _restore_rng(before, rng)
                    for model in models.values():
                        model.train()
            history.append(record)
            if update % settings.checkpoint_every == 0:
                save("complete" if update == settings.updates else "running")
        save("complete" if update == settings.updates else "paused")
    except Exception as exc:
        save("failed", dict(type=type(exc).__name__, message=str(exc), attempted_update=update + 1))
        raise
    return history


def _teacher_cases(results):
    if not results:
        raise ValueError("at least one TRAIN teacher result is required")
    if _field(results, "levels") is not None:
        results = [results]
    elif _field(results[0], "A", _field(results[0], "a")) is not None:
        results = [dict(levels=results)]
    groups = []
    for result in results:
        _train_only(result)
        levels = _field(result, "levels")
        if levels is None:
            raise ValueError("teacher result requires levels")
        active = [level for level in levels if _field(level, "mask") is not None]
        if not active:
            raise ValueError("teacher has no trainable interpolation levels")
        groups.append(active)
    return groups


def _aligned_labels(level):
    """Join labels by actual coarse column, never by selected-slot position."""
    _train_only(level)
    teacher_pattern = _field(level, "pattern")
    if teacher_pattern is None:
        raise ValueError("teacher candidate/coarsening metadata mismatch")
    pattern = direct_pattern(_shape(level), teacher_pattern.coarse_shape)
    source_columns = np.asarray(teacher_pattern.columns)
    source_mask = np.asarray(_plain(_field(level, "mask")), dtype=bool)
    source_weights = np.asarray(_plain(_field(level, "weights")), dtype=np.float64)
    if source_mask.shape != source_columns.shape or source_weights.shape != source_columns.shape:
        raise ValueError("teacher labels have incompatible shapes")
    if np.any(source_mask & (source_columns < 0)) or not np.isfinite(source_weights).all():
        raise ValueError("invalid teacher mask or nonfinite teacher weight")
    if np.any(source_weights[~source_mask] != 0):
        raise ValueError("teacher values outside structural mask")
    target = np.zeros_like(pattern.columns, dtype=np.float64)
    mask = np.zeros_like(pattern.columns, dtype=bool)
    matched = np.zeros_like(source_mask)
    for slot in range(pattern.n_candidates):
        matches = ((source_columns == pattern.columns[:, slot, None]) & source_mask
                   & (pattern.columns[:, slot, None] >= 0))
        if np.any(matches.sum(1) > 1):
            raise ValueError("duplicate coarse-column labels")
        mask[:, slot] = matches.any(1)
        target[:, slot] = (source_weights * matches).sum(1)
        matched |= matches
    if np.any(source_mask & ~matched):
        raise ValueError("teacher selected a coarse column outside the CNN candidate set")
    return pattern, mask, target


def supervised_objective(mask_model, weight_model, levels, *, settings=None):
    """Mean over levels of means over learnable rows; logits bypass hard top-k.

    BCE averages valid candidates in each non-injection row.  Weight MSE averages
    selected candidates per row, optionally dividing errors by max(1, teacher
    row L1 norm).  The action option averages squared action error divided by
    each test vector's mean-square amplitude.  No teacher gradients are built.
    """
    settings = _settings(settings, SupervisedSettings)
    if mask_model is None and (settings.alpha_mask != 0 or settings.alpha_weight + settings.alpha_action <= 0):
        raise ValueError("Weight-only supervision requires alpha_mask=0 and a weight/action loss")
    parameter = next(weight_model.parameters())
    terms, reports = [], []
    for index, level in enumerate(levels):
        pattern, mask_np, target_np = _aligned_labels(level)
        if mask_model is None and not np.array_equal(mask_np, fixed_bilinear_mask(pattern).numpy()):
            raise ValueError("Weight-only teacher labels must use fixed bilinear support")
        _, injection, valid_np = geometry_constraints(pattern)
        dtype, device = parameter.dtype, parameter.device
        target = torch.as_tensor(target_np, dtype=dtype, device=device)
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device)
        free = torch.as_tensor(injection < 0, device=device) & valid.any(1)
        if mask_model is None:
            free &= mask.sum(1) > 1  # One-slot rows are fixed by their row sum.
        if not bool(free.any()):
            continue
        features = direct_features(_operator(level), _shape(level), dtype=dtype, device=device)
        logits = row_slots(mask_model(features), pattern) if mask_model is not None else None
        predicted = project_weights(row_slots(weight_model(features, mask), pattern), mask,
                                    pattern, l1_cap=settings.l1_cap)
        if logits is None:
            bce = predicted.sum() * 0.
        else:
            bce = F.binary_cross_entropy_with_logits(logits, mask.to(dtype), reduction="none")
            bce = ((bce * valid).sum(1) / valid.sum(1).clamp_min(1))[free].mean()
        scale = target.abs().sum(1).clamp_min(1.) if settings.normalize_weights else torch.ones_like(target[:, 0])
        mse = ((((predicted - target) / scale[:, None]).square() * mask).sum(1)
               / mask.sum(1).clamp_min(1))[free].mean()
        action = predicted.sum() * 0.
        if settings.alpha_action:
            vectors = _field(level, "vectors")
            if vectors is None:
                raise ValueError("action loss requires TRAIN teacher test vectors")
            z = torch.as_tensor(np.asarray(vectors), dtype=dtype, device=device)
            if z.ndim != 2 or z.shape[0] != pattern.n_fine_unknowns or z.shape[1] == 0:
                raise ValueError("teacher vectors must have shape N x m")
            zc = z[torch.as_tensor(coarse_fine_indices(pattern), device=device)]
            difference = sparse_from_weights(pattern, predicted - target, mask).apply(zc)
            scale_z = z.square().mean(0).clamp_min(torch.finfo(dtype).tiny)
            action = (difference[free].square() / scale_z[None, :]).mean()
        term = settings.alpha_mask * bce + settings.alpha_weight * mse + settings.alpha_action * action
        terms.append(term)
        selected = ((logits.detach() > 0) if logits is not None else mask) & valid & free[:, None]
        true = mask & free[:, None]
        tp = int((selected & true).sum())
        reports.append(dict(level=index, mask_bce=float(bce.detach()), weight_mse=float(mse.detach()),
                            action_loss=float(action.detach()), rows=int(free.sum()),
                            precision=tp / max(int(selected.sum()), 1),
                            recall=tp / max(int(true.sum()), 1)))
    if not terms:
        raise ValueError("no learnable teacher interpolation rows")
    return torch.stack(terms).mean(), dict(levels=reports, normalization="equal levels, equal rows",
                                          teacher_forcing=True, hard_topk_gradient=False,
                                          mask_learned=mask_model is not None)


def train_supervised(mask_model, weight_model, teacher_results, *, settings=None,
                     checkpoint_path=None, resume=False, manifest=None, stop_after=None):
    settings = _settings(settings, SupervisedSettings)
    groups = _teacher_cases(teacher_results)
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
        case = int(rng.integers(len(groups)))
        loss, report = supervised_objective(mask_model, weight_model, groups[case], settings=settings)
        return loss, dict(case=case, **report)

    models = dict(weight=weight_model) if mask_model is None else dict(mask=mask_model, weight=weight_model)
    return _run(models, settings, phase="supervised",
                source=source, objective=objective, checkpoint_path=checkpoint_path,
                resume=resume, manifest=manifest, stop_after=stop_after)


def curriculum_cycles(update, settings):
    settings = _settings(settings, FinetuneSettings)
    if type(update) is not int or not 0 <= update < settings.updates:
        raise ValueError("update outside the configured training budget")
    switch = (settings.updates + 1) // 2 if settings.curriculum_step is None else settings.curriculum_step
    return settings.cycles if update < switch else settings.final_cycles


def _probes(a, shape, graph, settings, rng, parameter):
    from .bamg_solver import graph_cycle
    columns, labels = [], []
    cache = LineSmootherCache(a, shape)
    for kind in settings.probe_kinds:
        for _ in range(settings.probes_per_kind):
            e = rng.standard_normal(a.shape[0])
            if kind == "smoothed":
                for __ in range(settings.smoothing_sweeps):
                    e += classical_smoothing_correction(a, shape, -a @ e, "line_alt", cache=cache)
            value = torch.as_tensor(e, dtype=parameter.dtype, device=parameter.device)
            if kind == "hard":
                with torch.no_grad():
                    for __ in range(settings.hard_cycles):
                        value = graph_cycle(graph, value, torch.zeros_like(value))
                value = value.detach()
            norm = torch.linalg.vector_norm(value)
            if not bool(torch.isfinite(norm)):
                raise TrainingFailure(f"nonfinite {kind} error probe")
            # Exact zero is retained and explicitly counted by residual_loss.
            if bool(norm > 0):
                value = value / norm
            columns.append(value.detach()); labels.append(kind)
    return torch.stack(columns, dim=1), labels


def _validate_time_profile(settings, reference, weight_model):
    """Bind optional detached seconds to the measured deployment configuration."""
    if settings.time_weights is None:
        if settings.time_profile is not None:
            raise ValueError("time_profile requires a time-weighted objective")
        return
    profile = settings.time_profile
    if not isinstance(profile, Mapping) or not profile.get("snapshot_id"):
        raise ValueError("detached-time training requires a deployment time_profile snapshot")
    expected = dict(backend="scipy_fp64_cpu", dtype="torch.float64", k=reference.k,
                    mask_spec=reference.mask_model.spec if reference.mask_model is not None else None,
                    weight_spec=weight_model.spec,
                    hardware=hardware_environment(), cycle_seconds=list(settings.time_weights))
    for key, value in expected.items():
        if _digest(profile.get(key)) != _digest(value):
            raise ValueError(f"deployment time_profile mismatch: {key}")
    if profile.get("measurement") != "deployment_cycle":
        raise ValueError("time_profile must measure deployment cycles, not autograd execution")
    # A static profile is permitted, but its declared validity ends here. A new
    # snapshot needs a separately identified experiment, not an unlogged refresh.
    if profile.get("valid_updates", 0) < settings.updates:
        raise ValueError("time_profile validity does not cover the training budget")


def train_finetune(weight_model, reference, examples, *, settings=None, mg_config=None,
                   checkpoint_path=None, resume=False, manifest=None,
                   validation_callback=None, stop_after=None):
    """Train only theta on fixed reference masks; return all completed updates.

    ``validation_callback(update, weight_model)`` may return logging data.  Model
    choice remains the caller's warm-time decision; callback RNG use is isolated.
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
        case = int(rng.integers(len(examples)))
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
        return loss, dict(case=case, cycles=count, probe_kinds=kinds, fallback_used=False,
                          fixed_support=True, **{k: v for k, v in report.items() if k != "cycles"})

    source_contract = dict(cases=source, reference_hash=frozen_hash, mg_config=asdict(mg_config))
    extra = dict(reference=reference.state_dict(include_cache=True), masks=masks, mask_hash=mask_hash,
                 reference_hash=frozen_hash, deployment_requires_reference=True)
    return _run(dict(weight=weight_model), settings, phase="finetune", source=source_contract,
                objective=objective, checkpoint_path=checkpoint_path, resume=resume,
                manifest=manifest, stop_after=stop_after, extra=extra, verify=verify,
                validation_callback=validation_callback)


def refit_fixed_mask_labels(levels, masks, *, teacher_config=None):
    """Prepare optional TRAIN-only weight warmup labels before residual training.

    This helper never changes the reference checkpoints or their masks.  Call it
    on TRAIN vectors for the corresponding actual operator; use the returned
    labels in ``warmup_fixed_masks`` before the main residual stage.
    """
    from .bamg_teacher import local_fit
    if len(levels) != len(masks):
        raise ValueError("one fixed structural mask is required per nonterminal level")
    result = []
    for level, mask in zip(levels, masks):
        _train_only(level)
        vectors = _field(level, "vectors")
        if vectors is None:
            raise ValueError("fixed-mask label refit requires TRAIN test vectors")
        teacher_pattern = _field(level, "pattern")
        pattern = direct_pattern(_shape(level), teacher_pattern.coarse_shape if teacher_pattern is not None else None)
        requested = np.asarray(_plain(mask), dtype=bool)
        weights, fitted_mask, diagnostics = local_fit(_operator(level), pattern, vectors,
            config=teacher_config, fixed_mask=requested)
        if not np.array_equal(requested, fitted_mask):
            raise TrainingFailure("fixed-mask LS changed a structural connection")
        result.append(dict(A=_operator(level), shape=_shape(level), pattern=pattern,
                           mask=requested.copy(), weights=weights, vectors=vectors,
                           diagnostics=diagnostics, split="train"))
    return result


def warmup_fixed_masks(weight_model, reference, teacher_results, *, settings=None,
                        checkpoint_path=None, resume=False, manifest=None, stop_after=None):
    """Optional stage-one preparation with frozen mask labels and weight-only Adam.

    Pass labels returned by ``refit_fixed_mask_labels``; the reference models are
    neither updated nor replaced.  The distinct phase is recorded in checkpoints.
    """
    settings = _settings(settings, SupervisedSettings)
    if settings.alpha_mask != 0 or settings.alpha_weight + settings.alpha_action <= 0:
        raise ValueError("fixed-mask warmup requires alpha_mask=0 and a weight/action objective")
    groups = _teacher_cases(teacher_results)
    frozen = _digest(reference.state_dict())
    source = []
    for levels in groups:
        source.append([dict(operator=operator_digest(_operator(level)), shape=_shape(level),
                            labels=_digest(_aligned_labels(level)[1:]),
                            vectors=_digest(_field(level, "vectors"))) for level in levels])

    def verify():
        reference.check_immutable()
        if _digest(reference.state_dict()) != frozen:
            raise TrainingFailure("warmup changed the frozen reference")

    def objective(update, rng):
        case = int(rng.integers(len(groups)))
        # Frozen mask logits contribute coefficient zero, never an optimizer slot.
        loss, report = supervised_objective(reference.mask_model, weight_model, groups[case], settings=settings)
        return loss, dict(case=case, **report)

    return _run(dict(weight=weight_model), settings, phase="fixed_mask_warmup",
                source=dict(labels=source, reference_hash=frozen), objective=objective,
                checkpoint_path=checkpoint_path, resume=resume, manifest=manifest,
                stop_after=stop_after, verify=verify)
