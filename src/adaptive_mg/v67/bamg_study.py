"""Reproducible BAMG labels, independent CNNs, frozen masks and warm evaluation.

This is an additive research entry point. Existing bilinear/EM studies and their
immutable source manifests are not reinterpreted. Serialized .pt files are local
trusted run artifacts, never an interface for loading untrusted checkpoints.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import platform
from pathlib import Path
import random
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from ..config import MGConfig
from ..grid import next_shape, terminal
from ..hierarchy import baseline_kwargs
from ..provenance import stable_norm
from ..transfer import build_transfer_pattern, baseline_weights, scipy_prolongation_from_weights
from .fixed_classical import BilinearClassicalBaseline
from .research_data import (
    FAMILIES, make_research_plan, prepare_development_data, load_development_data,
    freeze_research, claim_final_evaluation, materialize_final_data,
    complete_final_evaluation,
)
from .bamg_transfer import (
    MaskCNN, WeightCNN, model_from_spec, CANDIDATE_VERSION,
    FEATURE_VERSION, CONSTRAINT_VERSION,
)
from .bamg_solver import (
    ReferenceModels, build_runtime, runtime_from_prolongations,
    progress_rates, mask_digest,
)
from .bamg_teacher import TeacherConfig, build_teacher
from .bamg_training import train_supervised, train_finetune


VERSION = 'bamg-two-cnn-frozen-support-v1'
PROJECT = Path(__file__).resolve().parents[3]
METHODS = ('fixed_C', 'em5_C', 'ls_once_C', 'bootstrap_C',
           'bootstrap_two_cnn_supervised', 'bootstrap_two_cnn_finetuned_uniform',
           'bootstrap_two_cnn_finetuned_terminal')


def methods_for(settings):
    if settings.get('support_mode', 'learned') == 'fixed_bilinear':
        return tuple(m.replace('bootstrap_two_cnn_', 'bootstrap_weight_cnn_') for m in METHODS)
    return METHODS


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if k != 'x'}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, torch.Tensor):
        return _clean(value.detach().cpu().tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_clean(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def _save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary); temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text())


def _load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _sources():
    paths = list((PROJECT / 'src/adaptive_mg').rglob('*.py'))
    paths += list((PROJECT / 'src/adaptive_mg').rglob('*.cpp'))
    paths.append(PROJECT / 'scripts/run_v6_7_bamg_study.py')
    return {str(p.relative_to(PROJECT)): _file_hash(p) for p in sorted(paths)}


def _environment():
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, torch=str(torch.__version__),
                platform=platform.platform(), processor=platform.processor(),
                device='cpu', dtype='float64',
                threadpools=[{key: pool.get(key) for key in
                             ('internal_api', 'num_threads', 'version', 'architecture')}
                             for pool in threadpool_info()])


def validate_settings(settings):
    if settings.get('version') != VERSION:
        raise ValueError('Unsupported BAMG study version')
    cfg = MGConfig(**settings['mg'])
    BilinearClassicalBaseline.from_config(SimpleNamespace(mg=cfg))
    if cfg.tolerance != 1e-8 or cfg.absolute_tolerance != 1e-12 or cfg.residual_reference != 'initial':
        raise ValueError('BAMG comparison uses max(1e-12, 1e-8*initial residual)')
    if cfg.device != 'cpu' or settings.get('dtype') != 'float64':
        raise ValueError('This implementation uses the CPU sparse FP64 reference backend')
    if cfg.stencil_backend != 'csr':
        raise ValueError('Declare csr explicitly for the direct-P reference benchmark')
    if settings.get('support_mode', 'learned') not in ('learned', 'fixed_bilinear'):
        raise ValueError('Unknown reference support ablation')
    if settings.get('discretization') not in ('p1', 'q1'):
        raise ValueError('Declare p1 or q1 discretization explicitly')
    if settings.get('k') not in (4, 8) or not np.isfinite(settings.get('l1_cap', 0)) or settings.get('l1_cap', 0) < 1:
        raise ValueError('Expected k=4 or 8 and a feasible positive L1 cap')
    if settings.get('torch_threads', 0) < 1:
        raise ValueError('Positive thread budget required')
    mode = settings.get('support_mode', 'learned')
    if settings['teacher'].get('support_mode', mode) != mode:
        raise ValueError('Teacher and student must use the same support_mode')
    if mode == 'fixed_bilinear' and settings['supervised'].get('alpha_mask', 1.) != 0:
        raise ValueError('Fixed bilinear support requires alpha_mask=0')
    _teacher_config(settings)
    bench = settings['benchmark']
    if bench.get('selection_workload') != 'warm_rhs_1':
        raise ValueError('Default study selection must explicitly be warm_rhs_1')
    if (type(bench['repetitions']) is not int or bench['repetitions'] < 1 or
        1 not in bench['rhs_counts'] or any(type(n) is not int or n < 1 for n in bench['rhs_counts']) or
        len(set(bench['rhs_counts'])) != len(bench['rhs_counts'])):
        raise ValueError('Invalid benchmark repetitions or RHS counts')
    if not 0 <= bench['allowed_failure_rate'] <= 1 or bench['failure_penalty_seconds'] <= 0:
        raise ValueError('Declare a finite failure rule before evaluation')
    if (not np.isfinite(bench['failure_penalty_seconds']) or
        not np.isfinite(bench['timeout_seconds']) or bench['timeout_seconds'] <= 0):
        raise ValueError('Invalid timing budget')
    return cfg


def _initialize(settings):
    global _THREAD_LIMIT
    random.seed(settings['seed']); np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed']); torch.set_num_threads(settings['torch_threads'])
    _THREAD_LIMIT = threadpool_limits(limits=settings['torch_threads'])


def _manifest(settings):
    return dict(version=VERSION, settings_digest=_hash(settings), sources=_sources(),
                environment=_environment(), candidate_version=CANDIDATE_VERSION,
                feature_version=FEATURE_VERSION, constraint_version=CONSTRAINT_VERSION)


def _open(output, *, mutable=False):
    out = Path(output).resolve(); settings = _read(out / 'configuration.json')
    cfg = validate_settings(settings); _initialize(settings)
    if _read(out / 'study_manifest.json') != _manifest(settings):
        raise ValueError('Source/configuration/environment changed: use a new study directory')
    if mutable and (out / 'data/research_freeze.json').exists():
        raise ValueError('Research frozen; training and selection are closed')
    baseline = BilinearClassicalBaseline.from_config(SimpleNamespace(mg=cfg))
    return out, settings, cfg, baseline


def _examples(out, baseline):
    train, validation, manifest = load_development_data(out / 'data', baseline)
    def adapt(e):
        return dict(A=e.a, shape=(e.n, e.n), case_id=e.digest, b=e.b,
                    family=e.research_family, name=e.name,
                    normalized_operator_digest=e.group_digest)
    return [adapt(e) for e in train], [adapt(e) for e in validation], manifest


def prepare(config, output, *, resume=False):
    settings = _read(config); cfg = validate_settings(settings); _initialize(settings)
    out = Path(output).resolve()
    if (out / 'study_manifest.json').exists():
        if not resume:
            raise FileExistsError('Study exists; use --resume')
        _, existing, _, baseline = _open(out, mutable=True)
        if settings != existing:
            raise ValueError('Resume settings differ')
        _examples(out, baseline)
        return
    if resume or (out.exists() and any(out.iterdir())):
        raise FileExistsError('Prepare requires a new empty directory')
    baseline = BilinearClassicalBaseline.from_config(SimpleNamespace(mg=cfg))
    plan = make_research_plan(smoke=settings['smoke'], seed=settings['seed'] + 7100003,
                              discretization=settings['discretization'])
    for split in ('train', 'validation'):
        spec = plan['splits'][split]
        spec.update(sizes=settings['data'][split]['sizes'],
                    per_family=settings['data'][split]['per_family'])
        spec['families'] = settings['data'].get('families', list(FAMILIES))
        spec['count'] = len(spec['sizes']) * len(spec['families']) * spec['per_family']
    roots = list(dict.fromkeys([str(out.parent)] + settings.get('historical_roots', [])))
    prepare_development_data(out / 'data', baseline, plan=plan, historical_roots=roots)
    _write(out / 'configuration.json', settings)
    _write(out / 'fixed_classical.json', baseline.to_dict())
    _write(out / 'study_manifest.json', _manifest(settings))
    print('Prepared disjoint TRAIN/validation; final/OOD operators remain unobserved.', flush=True)


def _teacher_config(settings, *, once=False):
    values = {**settings['teacher'], 'k': settings['k'], 'l1_cap': settings['l1_cap'],
              'support_mode': settings.get('support_mode', 'learned')}
    if once:
        values['setup_passes'] = 1
    return TeacherConfig(**values)


def teacher(output, *, resume=False):
    out, settings, cfg, baseline = _open(output, mutable=True)
    train, _, _ = _examples(out, baseline)
    config = _teacher_config(settings); records = []
    for example in train:
        path = out / 'teacher/train' / (example['case_id'] + '.pt')
        identity = dict(case_id=example['case_id'], shape=example['shape'],
                        config=asdict(config), manifest=_read(out / 'study_manifest.json'))
        if path.exists():
            if not resume:
                raise FileExistsError('Teacher label exists; use --resume')
            payload = _load_pt(path)
            if payload['identity'] != identity:
                raise ValueError('Teacher cache identity mismatch')
        else:
            result = build_teacher(example['A'], example['shape'], config=config, mg_config=cfg)
            _save(path, dict(identity=identity, teacher=result))
        records.append(dict(case_id=example['case_id'], path=str(path.relative_to(out)),
                            sha256=_file_hash(path)))
        print('Teacher', example['name'], flush=True)
    _write(out / 'teacher/index.json', dict(records=records, split='train',
                                           config=asdict(config)))


def _teacher_results(out, examples):
    index = _read(out / 'teacher/index.json')
    settings = _read(out / 'configuration.json')
    records = {row['case_id']: row for row in index['records']}
    results = []
    for example in examples:
        row = records[example['case_id']]; path = out / row['path']
        if _file_hash(path) != row['sha256']:
            raise ValueError('Teacher artifact changed')
        payload = _load_pt(path)
        expected = dict(case_id=example['case_id'], shape=example['shape'],
                        config=asdict(_teacher_config(settings)), manifest=_read(out / 'study_manifest.json'))
        if payload['identity'] != expected:
            raise ValueError('Teacher label belongs to another operator/configuration/source')
        result = payload['teacher']
        results.append(result)
    return results


def _models(settings):
    _initialize(settings)
    return (None if settings.get('support_mode') == 'fixed_bilinear'
            else MaskCNN(**settings['models']['mask']).double(),
            WeightCNN(**settings['models']['weight']).double())


def _model_package(mask, weight):
    return dict(mask_spec=mask.spec if mask is not None else None,
                mask_state=mask.state_dict() if mask is not None else None,
                weight_spec=weight.spec, weight_state=weight.state_dict())


def _restore_models(package):
    mask = model_from_spec(package['mask_spec']).double() if package['mask_spec'] is not None else None
    weight = model_from_spec(package['weight_spec']).double()
    if mask is not None:
        mask.load_state_dict(package['mask_state'])
    elif package['mask_state'] is not None:
        raise ValueError('Mask state without mask architecture')
    weight.load_state_dict(package['weight_state'])
    return mask, weight


def _initialize_weight(weight, path):
    """Transfer compatible tensors only; never inherit optimizer/progress/masks."""
    path = Path(path).resolve()
    before = _file_hash(path)
    package = torch.load(path, map_location='cpu', weights_only=True)
    spec = package.get('weight_spec', package.get('specs', {}).get('weight'))
    state = package.get('weight_state', package.get('models', {}).get('weight'))
    if spec != weight.spec or state is None:
        raise ValueError('Weight CNN architecture mismatch; use the source model specification')
    if _file_hash(path) != before:
        raise ValueError('Initialization checkpoint changed while reading')
    weight.load_state_dict(state, strict=True)
    return dict(path=str(path), sha256=before, weight_spec=spec,
                transfer='weights_only', optimizer_reused=False, training_updates_reused=False)


def pretrain(output, *, resume=False, init_weight=None):
    out, settings, _, baseline = _open(output, mutable=True)
    if resume and init_weight is not None:
        raise ValueError('Use --init-weight for a new training run, not --resume')
    if (out / 'pretrain.pt').exists() and not resume:
        raise FileExistsError('Supervised checkpoint exists; use --resume or a new run')
    train, _, manifest = _examples(out, baseline); mask, weight = _models(settings)
    provenance = dict(study=_read(out / 'study_manifest.json'), data=manifest)
    initial_path = out / 'weight_initialization.json'
    if resume and initial_path.exists():
        provenance['initialization'] = _read(initial_path)
    elif init_weight is not None:
        provenance['initialization'] = _initialize_weight(weight, init_weight)
        _write(initial_path, provenance['initialization'])
    elif initial_path.exists():
        raise ValueError('Initialization record already exists; use the recorded initialization explicitly')
    history = train_supervised(mask, weight, _teacher_results(out, train),
        settings={**settings['supervised'], 'l1_cap': settings['l1_cap']},
        checkpoint_path=out / 'pretrain.pt', resume=resume,
        manifest=provenance)
    _save(out / 'supervised_models.pt', _model_package(mask, weight))
    _write(out / 'pretrain_history.json', history)


def freeze_masks(output, *, resume=False):
    out, settings, cfg, baseline = _open(output, mutable=True)
    path = out / 'reference.pt'
    if path.exists():
        if not resume:
            raise FileExistsError('Reference already frozen; use --resume')
        payload = _load_pt(path)
        if payload['supervised_sha256'] != _file_hash(out / 'supervised_models.pt'):
            raise ValueError('Supervised checkpoint differs from frozen reference')
        reference = ReferenceModels.from_state_dict(payload['reference'])
        train, validation, _ = _examples(out, baseline)
        for example in train + validation:
            masks = reference.masks_for(example['A'], example['shape'])
            if not reference.last_report['cache_hit'] or mask_digest(masks) != payload['masks'][example['case_id']]:
                raise ValueError('Frozen binary masks are missing or differ from their manifest')
        return
    mask, weight = _restore_models(_load_pt(out / 'supervised_models.pt'))
    compact = settings.get('support_mode') == 'fixed_bilinear'
    reference = ReferenceModels(None if compact else mask, None if compact else weight,
                                 k=settings['k'], l1_cap=settings['l1_cap'],
                                 mg_config=cfg, caps=settings.get('caps'),
                                 support_mode=settings.get('support_mode', 'learned'))
    train, validation, _ = _examples(out, baseline); masks = {}
    for example in train + validation:
        masks[example['case_id']] = mask_digest(reference.masks_for(example['A'], example['shape']))
    _save(path, dict(reference=reference.state_dict(include_cache=True),
                    supervised_sha256=_file_hash(out / 'supervised_models.pt'), masks=masks))
    _write(out / 'frozen_masks.json', masks)


def finetune(output, *, weighting='both', resume=False):
    out, settings, cfg, baseline = _open(output, mutable=True)
    train, _, manifest = _examples(out, baseline)
    choices = ('uniform', 'terminal') if weighting == 'both' else (weighting,)
    for choice in choices:
        path = out / ('finetune_' + choice + '.pt')
        if path.exists() and not resume:
            raise FileExistsError('Fine-tuning checkpoint exists; use --resume or a new run')
        _initialize(settings)
        _, weight = _restore_models(_load_pt(out / 'supervised_models.pt'))
        frozen = _load_pt(out / 'reference.pt')
        if frozen['supervised_sha256'] != _file_hash(out / 'supervised_models.pt'):
            raise ValueError('Supervised checkpoint differs from frozen reference')
        reference = ReferenceModels.from_state_dict(frozen['reference'])
        options = {**settings['finetune'], 'cycle_loss_weighting': choice,
                   'l1_cap': settings['l1_cap']}
        history = train_finetune(weight, reference, train, settings=options, mg_config=cfg,
            checkpoint_path=path, resume=resume and path.exists(),
            manifest=dict(study=_read(out / 'study_manifest.json'), data=manifest,
                          reference_sha256=_file_hash(out / 'reference.pt')))
        _save(out / ('deployment_' + choice + '.pt'),
              dict(version=VERSION, reference=reference.state_dict(), weight_spec=weight.spec,
                   weight_state=weight.state_dict(), settings=settings,
                   reference_sha256=_file_hash(out / 'reference.pt'), weighting=choice))
        _write(out / ('finetune_' + choice + '_history.json'), history)


def _baseline_bank(example, cfg, *, em=False, caps=None):
    start = perf_counter(); a = example['A']; shape = example['shape']; ps = []
    cfg = replace(cfg, em_iterations=5, em_fixed_iterations=True, em_rtol=1e-8)
    while not terminal(shape, cfg.coarsest_n):
        coarse = next_shape(shape, 'full', cfg.coarsest_n)
        pattern = build_transfer_pattern(shape, coarse)
        weights = baseline_weights(a, shape, 'energymin' if em else 'bilinear',
                                   coarse=coarse, **baseline_kwargs(cfg))
        p = scipy_prolongation_from_weights(pattern, weights)
        ps.append(p); a = (p.T @ a @ p).tocsr(); shape = coarse
    bank = runtime_from_prolongations(example['A'], example['shape'], ps, cfg, caps=caps)
    bank.setup_report['generation_and_prepare_seconds'] = perf_counter() - start
    bank.setup_report['setup_seconds'] = perf_counter() - start
    return bank


def _factory(method, example, settings, cfg, packages):
    if method in ('fixed_C', 'em5_C'):
        return _baseline_bank(example, cfg, em=method == 'em5_C', caps=settings.get('caps'))
    if method in ('ls_once_C', 'bootstrap_C'):
        start = perf_counter()
        result = build_teacher(example['A'], example['shape'],
                               config=_teacher_config(settings, once=method == 'ls_once_C'), mg_config=cfg)
        bank = runtime_from_prolongations(example['A'], example['shape'],
            [level.P for level in result.levels if level.P is not None], cfg, caps=settings.get('caps'))
        bank.setup_report['teacher_seconds'] = result.setup_seconds
        bank.setup_report['setup_seconds'] = perf_counter() - start
        bank.setup_report['teacher_diagnostics'] = result.diagnostics
        return bank
    weight, reference = packages[method]
    reference.clear_cache()  # Fresh cold includes the full frozen-reference pass.
    return build_runtime(example['A'], example['shape'], weight, reference, cfg, caps=settings.get('caps'))


def _packages(out):
    _, supervised = _restore_models(_load_pt(out / 'supervised_models.pt'))
    frozen = _load_pt(out / 'reference.pt')
    if frozen['supervised_sha256'] != _file_hash(out / 'supervised_models.pt'):
        raise ValueError('Supervised checkpoint differs from frozen reference')
    result = {'bootstrap_two_cnn_supervised':
              (supervised, ReferenceModels.from_state_dict(frozen['reference']))}
    for choice in ('uniform', 'terminal'):
        payload = _load_pt(out / ('deployment_' + choice + '.pt'))
        if payload['reference_sha256'] != _file_hash(out / 'reference.pt'):
            raise ValueError('Compared models do not share the frozen reference')
        weight = model_from_spec(payload['weight_spec']).double()
        weight.load_state_dict(payload['weight_state'])
        result['bootstrap_two_cnn_finetuned_' + choice] = (
            weight, ReferenceModels.from_state_dict(payload['reference']))
    if frozen['reference']['support_mode'] == 'fixed_bilinear':
        result = {name.replace('bootstrap_two_cnn_', 'bootstrap_weight_cnn_'): value
                  for name, value in result.items()}
    return result


class _FallbackFailure(RuntimeError):
    def __init__(self, cause, rejected):
        super().__init__(str(cause)); self.cause = cause; self.rejected = rejected


def _fallback_cycle(fallback, x, b, rejected):
    try:
        candidate = fallback.cycle(x, b)
        return candidate, stable_norm(b - fallback.root.a @ candidate)
    except (ValueError, RuntimeError, FloatingPointError, MemoryError, np.linalg.LinAlgError) as exc:
        raise _FallbackFailure(exc, rejected) from exc


def deployed_solve(bank, fallback, b, *, timeout_seconds):
    """Explicit existing MG growth/stagnation controls, followed by classical tail.

    All attempted work is timed. A successful tail is attributed to fallback,
    never to convergence of the learned P. Both banks are already prepared.
    """
    start = perf_counter(); cfg = bank.mg_config; x = np.zeros_like(b)
    initial = stable_norm(b - bank.root.a @ x); target = max(1e-12, 1e-8 * initial)
    residuals = [initial]; times = []; fallback_used = False; rejected = 0
    fallback_cycles = 0; stale = 0; converged = initial <= target
    reason = 'initial_converged' if converged else 'max_cycles'
    if not np.isfinite(initial):
        reason = 'nonfinite_initial_residual'
    for _ in range(0 if converged or not np.isfinite(initial) else cfg.max_cycles):
        if perf_counter() - start >= timeout_seconds:
            reason = 'timeout'; break
        tick = perf_counter(); old = residuals[-1]
        try:
            candidate, norm, fallback_used, rejected_now, used_tail, stale = _deployed_step(
                bank, fallback, x, b, old, initial, target, fallback_used, stale)
            rejected += rejected_now; fallback_cycles += used_tail
        except _FallbackFailure as exc:
            times.append(perf_counter() - tick); residuals.append(float('inf'))
            reason = 'fallback_error:' + type(exc.cause).__name__; fallback_used = True
            rejected += exc.rejected; fallback_cycles += 1
            break
        x = candidate; residuals.append(norm); times.append(perf_counter() - tick)
        if not np.isfinite(norm) or norm > cfg.divergence_factor * initial:
            reason = 'diverged'; break
        if norm <= target:
            converged = True; reason = 'converged'; break
    elapsed = perf_counter() - start
    return dict(converged=converged, reason=reason, cycles=len(times), residuals=residuals,
                cycle_seconds=times, warm_seconds=elapsed, fallback_used=fallback_used,
                fallback_cycles=fallback_cycles, rejected_learned_cycles=rejected,
                tolerance=target, **progress_rates(residuals, times, tolerance=target))


def _deployed_step(bank, fallback, x, b, old, initial, target, fallback_used, stale):
    cfg = bank.mg_config; rejected = 0; fallback_cycles = 0
    if not fallback_used:
        try:
            candidate = bank.cycle(x, b); norm = stable_norm(b - bank.root.a @ candidate)
        except (ValueError, RuntimeError, FloatingPointError, MemoryError, np.linalg.LinAlgError):
            norm = float('inf'); candidate = x
        if not np.isfinite(norm) or norm > cfg.safety_growth * old + cfg.safety_rtol_slack * initial:
            fallback_used = True; rejected += 1
            candidate, norm = _fallback_cycle(fallback, x, b, rejected)
            fallback_cycles += 1
        else:
            stale = stale + 1 if norm >= cfg.stagnation_rho * old else 0
            if stale >= cfg.stagnation_patience and norm > target:
                fallback_used = True  # Next iteration starts the classical tail.
    else:
        candidate, norm = _fallback_cycle(fallback, x, b, rejected)
        fallback_cycles += 1
    return candidate, norm, fallback_used, rejected, fallback_cycles, stale


def _rhs(example, count, seed):
    rng = np.random.default_rng(seed)
    return [example['b'].copy()] + [example['A'] @ rng.standard_normal(example['A'].shape[0])
                                   for _ in range(count - 1)]


def summarize(rows, settings):
    """All cases/repeats count, including setup failures and nonconvergence."""
    config = settings['benchmark']; penalty = config['failure_penalty_seconds']
    summaries = {}
    cohorts = {r['case_id'] for r in rows}
    for method in methods_for(settings):
        selected = [r for r in rows if r['method'] == method]
        cases = []; failures = 0; total = 0
        for case_id in sorted(cohorts):
            samples = [r for r in selected if r['case_id'] == case_id]
            values = []
            for row in samples:
                solved = row.get('raw', [{}])[0]
                success = bool(solved.get('converged', False)); total += 1; failures += not success
                values.append(solved['warm_seconds'] if success else max(
                    penalty, solved.get('warm_seconds', 0.)))
            if not samples:
                total += 1; failures += 1; values.append(penalty)
            cases.append(float(np.median(values)))
        rate = failures / max(total, 1)
        summaries[method] = dict(cases=len(cases), attempts=total, failures=failures,
            failure_rate=rate, eligible=rate <= config['allowed_failure_rate'],
            mean_case_median_warm_seconds=float(np.mean(cases)) if cases else None,
            failed_samples_use_seconds_penalty=penalty)
    eligible = [m for m in methods_for(settings) if summaries[m]['eligible'] and summaries[m]['cases']]
    winner = min(eligible, key=lambda m: summaries[m]['mean_case_median_warm_seconds']) if eligible else None
    learned = [m for m in eligible if m.startswith(('bootstrap_two_cnn_', 'bootstrap_weight_cnn_'))]
    selected_model = min(learned, key=lambda m: summaries[m]['mean_case_median_warm_seconds']) if learned else None
    return dict(methods=summaries, fastest_eligible_method=winner,
                selected_learned_model=selected_model, selection_split='validation',
                selection_metric='raw warm RHS=1, full cohort, case medians then mean')


def _benchmark_cases(out, settings, cfg, examples, *, result_path, resume=False, split='validation'):
    packages = _packages(out); rows = []; config = settings['benchmark']
    identity = dict(manifest=_read(out / 'study_manifest.json'), split=split,
        checkpoints={name: _file_hash(out / name) for name in (
            'supervised_models.pt', 'reference.pt', 'deployment_uniform.pt', 'deployment_terminal.pt')},
        cases=[e['case_id'] for e in examples])
    if result_path.exists():
        if not resume:
            raise FileExistsError('Benchmark exists; use --resume')
        previous = _read(result_path)
        if previous['identity'] != identity:
            raise ValueError('Benchmark resume identity mismatch')
        rows = previous['rows']
    done = {(r['case_id'], r['method'], r['repeat']) for r in rows}
    rng = random.Random(settings['seed'] + 7301)
    for case_index, example in enumerate(examples):
        rhs = _rhs(example, max(config['rhs_counts']), settings['seed'] + case_index)
        for repeat in range(config['repetitions']):
            methods = list(methods_for(settings)); rng.shuffle(methods)
            for method in methods:
                if (example['case_id'], method, repeat) in done:
                    continue
                row = dict(case_id=example['case_id'], family=example['family'],
                           shape=example['shape'], method=method, repeat=repeat)
                start = perf_counter()
                try:
                    bank = _factory(method, example, settings, cfg, packages)
                    row['setup'] = bank.setup_report; row['hierarchy'] = bank.hierarchy_report
                    row['mask_digest'] = getattr(bank, 'mask_digest', None)
                    # Untimed hardware warmup; each measured solve still starts at x=0.
                    for _ in range(config.get('warmup_cycles', 1)):
                        bank.cycle(np.zeros_like(rhs[0]), rhs[0])
                    raw = [bank.solve(b, timeout_seconds=config['timeout_seconds']) for b in rhs]
                    row['raw'] = raw
                    fallback = _baseline_bank(example, cfg, caps=settings.get('caps'))
                    row['fallback_setup_seconds'] = fallback.setup_report['setup_seconds']
                    row['deployed'] = [deployed_solve(bank, fallback, b,
                        timeout_seconds=config['timeout_seconds']) for b in rhs]
                    setup = bank.setup_report['setup_seconds']
                    row['reuse'] = {str(n): dict(
                        raw_warm_seconds=sum(r['warm_seconds'] for r in raw[:n]),
                        raw_total_seconds=setup + sum(r['warm_seconds'] for r in raw[:n]),
                        raw_converged=all(r['converged'] for r in raw[:n]),
                        deployed_total_seconds=setup + row['fallback_setup_seconds'] +
                            sum(r['warm_seconds'] for r in row['deployed'][:n]),
                        deployed_converged=all(r['converged'] for r in row['deployed'][:n]))
                        for n in config['rhs_counts']}
                    row['status'] = 'measured'
                except (ValueError, RuntimeError, FloatingPointError, MemoryError, np.linalg.LinAlgError) as exc:
                    row.update(status='setup_or_evaluation_failure', error=type(exc).__name__ + ': ' + str(exc),
                               elapsed_seconds=perf_counter() - start)
                rows.append(_clean(row))
                _write(result_path, dict(identity=identity, rows=rows, completed=False))
                print('Benchmark', split, example['name'], method, row['status'], flush=True)
    # Assert compared neural models used exactly the same support for each A.
    for example in examples:
        digests = {r['mask_digest'] for r in rows if r['case_id'] == example['case_id']
                   and r.get('mask_digest') is not None}
        if len(digests) > 1:
            raise AssertionError('Benchmark neural methods changed frozen masks')
    report = dict(identity=identity, rows=rows, completed=True,
                  protocol=dict(model_loading_in_setup=False, fresh_reference_cache=True,
                    sequential_rhs=True, simultaneous_batched_rhs=False,
                    offline_training_excluded=True, setup_includes_numerical_teacher=True,
                    fallback='prepared bilinear bank; growth rejection/stagnation classical tail',
                    measured_backend='cpu csr float64'))
    if split == 'validation':
        report['selection'] = summarize(rows, settings)
    _write(result_path, report)
    return report


def benchmark(output, *, resume=False):
    out, settings, cfg, baseline = _open(output, mutable=True)
    _, validation, _ = _examples(out, baseline)
    return _benchmark_cases(out, settings, cfg, validation,
                            result_path=out / 'benchmark.json', resume=resume)


def diagnose(output):
    out, settings, _, _ = _open(output)
    mask, weight = _restore_models(_load_pt(out / 'supervised_models.pt'))
    def model_info(model):
        if model is None:
            return dict(spec=None, parameters=0, bytes=0, convolution_layers=0)
        return dict(spec=model.spec, parameters=sum(p.numel() for p in model.parameters()),
                    bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                    convolution_layers=sum(isinstance(m, torch.nn.Conv2d) for m in model.modules()))
    uniform = sum(np.log([.1, .05])) / 2 < sum(np.log([.8, .02])) / 2
    terminal_prefers_second = np.log(.02) < np.log(.05)
    rate_examples = [dict(rho=rho, cycle_ms=ms, relative_rate_per_ms=(1-rho)/ms,
                         log_rate_per_ms=-np.log(rho)/ms,
                         cycles=int(np.ceil(np.log(1e-8)/np.log(rho))),
                         total_ms=int(np.ceil(np.log(1e-8)/np.log(rho))) * ms)
                     for rho, ms in ((.8, 1), (.25, 4))]
    report = dict(version=VERSION, mask=model_info(mask), weight=model_info(weight),
                  deployment_parameter_copies=('one weight CNN; geometry-only masks'
                      if settings.get('support_mode') == 'fixed_bilinear'
                      else 'one mask, frozen reference weight, final weight'),
                  mu=settings['teacher']['mu'], omega_uniform='1/K',
                  omega_terminal='0,...,0,1', rate_examples=rate_examples,
                  uniform_prefers_first=bool(uniform), terminal_prefers_second=bool(terminal_prefers_second),
                  optional_parallel_execution_implemented=False,
                  neural_device='cpu', fine_operators='sparse',
                  reference_package_bytes=(out / 'reference.pt').stat().st_size)
    _write(out / 'diagnostics.json', report)
    return report


def freeze(output):
    out, settings, _, baseline = _open(output, mutable=True)
    report = _read(out / 'benchmark.json')
    if not report['completed']:
        raise ValueError('Complete validation before freezing')
    if report['selection']['selected_learned_model'] is None:
        raise ValueError('No learned model met the declared validation failure criterion')
    _, validation, _ = _examples(out, baseline)
    expected = dict(manifest=_read(out / 'study_manifest.json'), split='validation',
        checkpoints={name: _file_hash(out / name) for name in (
            'supervised_models.pt', 'reference.pt', 'deployment_uniform.pt', 'deployment_terminal.pt')},
        cases=[e['case_id'] for e in validation])
    if report['identity'] != expected:
        raise ValueError('Validation results do not describe the current checkpoints')
    paths = {name: out / name for name in ('supervised_models.pt', 'reference.pt',
             'deployment_uniform.pt', 'deployment_terminal.pt', 'benchmark.json')}
    return freeze_research(out / 'data', paths, out / 'fixed_classical.json', {},
        [PROJECT / name for name in _sources()],
        dict(settings=settings, selection=report['selection']))


def evaluate_final(output, *, resume=False):
    out, settings, cfg, baseline = _open(output)
    root = out / 'data'; freeze_path = root / 'research_freeze.json'
    with claim_final_evaluation(root, freeze_path, dict(command='bamg-evaluate-final',
            settings_digest=_hash(settings), methods=list(methods_for(settings))), resume=resume) as claim:
        splits, _ = materialize_final_data(root, claim, baseline); paths = []
        for split, data in splits.items():
            examples = [dict(A=e.a, shape=(e.n, e.n), case_id=e.digest, b=e.b,
                             family=e.research_family, name=e.name) for e in data]
            path = out / ('benchmark_' + split + '.json')
            _benchmark_cases(out, settings, cfg, examples, result_path=path, resume=resume, split=split)
            paths.append(path)
        complete_final_evaluation(root, claim, paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'teacher', 'pretrain', 'freeze-masks', 'finetune',
                    'benchmark', 'diagnose', 'freeze', 'evaluate-final', 'run'):
        p = sub.add_parser(command); p.add_argument('--output', required=True)
        if command in ('prepare', 'run'):
            p.add_argument('--config', required=True)
        if command not in ('diagnose', 'freeze'):
            p.add_argument('--resume', action='store_true')
        if command == 'finetune':
            p.add_argument('--weighting', choices=('uniform', 'terminal', 'both'), default='both')
        if command in ('pretrain', 'run'):
            p.add_argument('--init-weight', help='Compatible old BAMG weight tensors; starts new optimizer/updates')
    args = parser.parse_args(argv)
    if getattr(args, 'init_weight', None) and args.resume:
        parser.error('--init-weight initializes a new run; it cannot be combined with --resume')
    if args.command in ('prepare', 'run'):
        prepare(args.config, args.output, resume=args.resume)
    if args.command == 'run':
        teacher(args.output, resume=args.resume)
        pretrain(args.output, resume=args.resume and (Path(args.output) / 'pretrain.pt').exists(),
                 init_weight=args.init_weight)
        freeze_masks(args.output, resume=args.resume)
        finetune(args.output, resume=args.resume)
        diagnose(args.output)
        benchmark(args.output, resume=args.resume and (Path(args.output) / 'benchmark.json').exists())
    elif args.command == 'finetune':
        finetune(args.output, weighting=args.weighting, resume=args.resume)
    elif args.command == 'pretrain':
        pretrain(args.output, resume=args.resume, init_weight=args.init_weight)
    elif args.command in ('diagnose', 'freeze'):
        globals()[args.command](args.output)
    elif args.command != 'prepare':
        globals()[args.command.replace('-', '_')](args.output, resume=args.resume)


if __name__ == '__main__':
    main()
