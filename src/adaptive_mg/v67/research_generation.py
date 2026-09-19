"""Measured FP32 generation diagnostics; no training or performance certificate.

The benchmark exercises the actual selected hierarchy and cached numerical
banks. CPU/MPS/CUDA availability is reported from this process; failed device
operations are recorded rather than silently falling back to CPU.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import csv
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from ..provenance import hardware_environment, operator_digest
from .banks import Stats, synchronize
from .limited import initialize_timing_runtime
from .research_runtime import is_teacher, offline_teacher_generation, research_spec
from .research_smoothers import O2Fields, stencil_to_o2_fields, transform_o2_fields
from .strong import PreparedStrongMG
from .strong_evaluation import _atomic_json


def available_generation_devices():
    mps = bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    cuda = bool(torch.cuda.is_available())
    return {
        'cpu': {'available': True, 'hardware': 'CPU', 'dtype': 'float32'},
        'mps': {'available': mps, 'hardware': 'Apple MPS' if mps else None,
                'dtype': 'float32', 'reason': None if mps else 'MPS unavailable in this process'},
        'cuda': {'available': cuda, 'hardware': torch.cuda.get_device_name(0) if cuda else None,
                 'dtype': 'float32', 'reason': None if cuda else 'CUDA unavailable in this process'},
    }


def _module_metadata(module):
    cost = module.cost_metadata() if hasattr(module, 'cost_metadata') else {}
    return dict(cost, parameters=sum(p.numel() for p in module.parameters()),
                training_only=is_teacher(module), architecture=research_spec(module),
                body_kind=getattr(module, 'body_kind', type(module).__name__))


def _hierarchy_metadata(root):
    levels = []
    level = root
    while level is not None:
        stencil = getattr(level, 'neural_stencil', None)
        levels.append(dict(index=level.index, shape=list(level.shape),
            a_digest=operator_digest(level.a), a_nnz=int(level.a.nnz),
            a_dtype=str(level.a.dtype), a_finite=bool(np.isfinite(level.a.data).all()),
            p_nnz=int(level.p.nnz) if level.p is not None else 0,
            p_dtype=str(level.p.dtype) if level.p is not None else None,
            p_finite=bool(np.isfinite(level.p.data).all()) if level.p is not None else True,
            smoother_dtype=str(stencil.csr.dtype) if stencil is not None else None,
            smoother_finite=bool(np.isfinite(stencil.csr.data).all()) if stencil is not None else True,
            smoother_native=bool(stencil is not None and stencil.native is not None)))
        level = level.coarse
    return levels


def _measure_generation(components, example, cfg, rules, branch, device):
    active = ('smoother',) if branch == 'H_S' else ('transfer',) if branch == 'H_P' else ('smoother', 'transfer')
    teacher = any(is_teacher(getattr(components, name)) for name in active)
    local = replace(cfg, branch=branch, mode='research', lazy=True, spatial=False,
                    gate_mode='open', inference_device=device, inference_dtype='float32',
                    mg=replace(cfg.mg, device='cpu'))
    started = perf_counter()
    prepared, stats = None, Stats()
    record = dict(status='failed', branch=branch, device=device, dtype='float32',
                  numerical_device='cpu', numerical_dtype='float64',
                  scope='offline_teacher_generation' if teacher else 'student_generation',
                  teacher=teacher, classical_setup_seconds=None, bank_setup_seconds=None)
    try:
        # An explicit offline scope never changes the production-solver guard.
        with offline_teacher_generation() if teacher else nullcontext():
            prepared = PreparedStrongMG(example.a, example.n, components, local, rules=rules)
            record['classical_setup_seconds'] = perf_counter() - started
            before_bank = perf_counter()
            root = prepared.ensure_branch(branch, stats)
            synchronize(torch.device(device))
            record['bank_setup_seconds'] = perf_counter() - before_bank
            record['cold_preparation_seconds'] = perf_counter() - started
            # The second lookup verifies the same actual bank is reused.
            warm_stats = Stats()
            before_warm = perf_counter()
            reused = prepared.ensure_branch(branch, warm_stats)
            synchronize(torch.device(device))
            record['warm_bank_lookup_seconds'] = perf_counter() - before_warm
            if reused is not root or warm_stats.neural_setup_calls:
                raise AssertionError('same-A cached bank unexpectedly generated a heavy expert')
        hierarchy = _hierarchy_metadata(root)
        if any(not row['a_finite'] or not row['p_finite'] or not row['smoother_finite'] for row in hierarchy):
            raise ValueError('nonfinite numerical bank')
        if any(row['a_dtype'] != 'float64' or row['p_dtype'] not in (None, 'float64')
               or row['smoother_dtype'] not in (None, 'float64') for row in hierarchy):
            raise ValueError('numerical bank must remain CPU FP64')
        record.update(status='complete', selected_strategy=prepared.selection.strategy_name,
            selector_seconds=prepared.selection.selector_seconds, rules_digest=prepared.selection.rules_digest,
            hierarchy=hierarchy, classical_hierarchy=_hierarchy_metadata(prepared.classical),
            warm_heavy_generation_calls=warm_stats.neural_setup_calls,
            operator_complexity=sum(row['a_nnz'] for row in hierarchy) / example.a.nnz,
            transfer_complexity=sum(row['p_nnz'] for row in hierarchy) / example.a.shape[0])
    except Exception as error:
        record.update(error=f'{type(error).__name__}: {error}',
                      failed_elapsed_seconds=perf_counter() - started)
    record.update(stats=asdict(stats), generation_seconds=stats.operator_generation_seconds,
                  cpu_device_copy_seconds=stats.cpu_device_transfer_seconds,
                  smoother_calls=stats.smoother_nn_calls, transfer_calls=stats.transfer_nn_calls)
    return record


def benchmark_generation(models, examples, cfg, rules, output, repeats=5, *,
                         branches=('H_S', 'H_P', 'H_SP'), devices=None, warmups=1,
                         order_seed=431, split='validation'):
    """Benchmark real cold banks and warm lookup on declared train/validation data.

    ``models`` maps labels to Components. ``examples`` is an Example or sequence.
    The caller's dataset manager is responsible for validating split provenance;
    this helper rejects final/OOD split declarations and never opens test data.
    Process/import/native-library loading and checkpoint loading are excluded.
    All constructor, selector, model/device conversion, generation, FP64-copy,
    sparse assembly and factorization costs in preparation are included.
    """
    if split not in {'train', 'validation', 'smoke'}:
        raise ValueError('generation optimization diagnostics cannot consume final/OOD test data')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1 or warmups < 0:
        raise ValueError('positive integer repeats and nonnegative warmups required')
    if not models or not branches or set(branches) - {'H_S', 'H_P', 'H_SP'} or len(set(branches)) != len(branches):
        raise ValueError('models and distinct learned branches required')
    examples = [examples] if hasattr(examples, 'a') else list(examples)
    if any(getattr(e,'research_split',None) not in (None,'train','validation') for e in examples):
        raise ValueError('Generation optimization cannot use final/OOD operators')
    if not examples or len({e.digest for e in examples}) != len(examples):
        raise ValueError('nonempty unique operator examples required')
    inventory = available_generation_devices()
    devices = list(devices) if devices is not None else list(inventory)
    if not devices or len(set(devices)) != len(devices) or set(devices) - set(inventory):
        raise ValueError('devices must be a distinct subset of cpu,mps,cuda')
    output = Path(output)
    if any((output / name).exists() for name in ('generation.json', 'generation.csv', 'generation_manifest.json')):
        raise FileExistsError('generation output already exists; choose a new directory')
    output.mkdir(parents=True, exist_ok=True)
    initialize_timing_runtime()
    metadata = {name: {'smoother': _module_metadata(c.smoother),
                       'transfer': _module_metadata(c.transfer),
                       'generation_signature': c.generation_signature()} for name, c in models.items()}
    manifest = dict(revision='v67-research-generation-1', split=split,
        split_provenance='caller dataset-manager validation required; helper does not open split files',
        examples=[e.manifest() for e in examples], config=cfg.to_dict(), rules=rules.to_dict(),
        rules_digest=rules.digest(), models=metadata, repeats=repeats, warmups=warmups,
        order_seed=order_seed, requested_devices=devices, availability=inventory,
        environment=hardware_environment(), production_certificate=False,
        timing_scope='selector + classical setup + real learned bank preparation; no solve in this benchmark',
        precision='FP32 generation -> CPU FP64 numerical bank',
        exclusions=['checkpoint loading', 'imports', 'native library loading'],
        a100_measurement='not measured' if not inventory['cuda']['available'] else
            'only the actually identified CUDA device above is measured')
    _atomic_json(output / 'generation_manifest.json', manifest)
    rows = []
    combinations = [(label, example, branch, device) for label in models for example in examples
                    for branch in branches for device in devices if inventory[device]['available']]
    for label in models:
        for device in devices:
            if not inventory[device]['available']:
                rows.append(dict(model=label, device=device, status='not_measured',
                                 reason=inventory[device]['reason']))
    rng = np.random.default_rng(order_seed)
    for repetition in range(-warmups, repeats):
        order = rng.permutation(len(combinations))
        for index in order:
            label, example, branch, device = combinations[int(index)]
            row = _measure_generation(models[label], example, cfg, rules, branch, device)
            row.update(model=label, example=example.name, operator_digest=example.digest,
                       repetition=max(0, repetition), warmup=repetition < 0)
            rows.append(row)
            _atomic_json(output / 'generation.json', dict(manifest=manifest, runs=rows, status='running'))
    summaries = []
    metrics = ('classical_setup_seconds', 'bank_setup_seconds', 'cold_preparation_seconds',
               'generation_seconds', 'cpu_device_copy_seconds', 'warm_bank_lookup_seconds',
               'operator_complexity', 'transfer_complexity')
    for label, example, branch, device in combinations:
        selected = [r for r in rows if r.get('model') == label and r.get('example') == example.name
                    and r.get('branch') == branch and r.get('device') == device and not r.get('warmup')]
        good = [r for r in selected if r['status'] == 'complete']
        item = dict(model=label, example=example.name, branch=branch, device=device,
                    status='complete' if len(good) == repeats else 'failed', repeats=repeats,
                    completed_repeats=len(good), failed_repeats=len(selected) - len(good),
                    teacher=any(r.get('teacher', False) for r in selected))
        # Failed preparation is never represented as a fast median.
        for metric in metrics:
            item[metric + '_median'] = float(np.median([r[metric] for r in good])) if len(good) == repeats else None
        reference = next((r for r in rows if r.get('model') == label and r.get('example') == example.name
                          and r.get('branch') == branch and r.get('device') == 'cpu'
                          and not r.get('warmup') and r['status'] == 'complete'), None)
        item['classical_level_operators_identical_to_cpu'] = (
            all([v['a_digest'] for v in r['classical_hierarchy']] ==
                [v['a_digest'] for v in reference['classical_hierarchy']] for r in good)
            if reference is not None and good else None)
        item['actual_learned_level_operators_identical_to_cpu'] = (
            all([v['a_digest'] for v in r['hierarchy']] ==
                [v['a_digest'] for v in reference['hierarchy']] for r in good)
            if reference is not None and good else None)
        summaries.append(item)
    result = dict(manifest=manifest, runs=rows, summary=summaries, status='complete',
                  performance_certified=False,
                  note='GPU FP32 roundoff can change learned coarse operators; exact A_l digest matches are reported')
    _atomic_json(output / 'generation.json', result)
    with (output / 'generation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]) if summaries else ['status'])
        writer.writeheader()
        writer.writerows(summaries)
    return result


def _resample_spatial(x, angle, reflect=False):
    # Coordinates in this project are (row,column), while grid_sample uses
    # (column,row). Its grid maps output positions to source positions.
    c, s = math.cos(angle), math.sin(angle)
    q = torch.tensor([[c, -s], [s, c]], device=x.device, dtype=x.dtype)
    if reflect:
        q = q @ torch.diag(x.new_tensor([1., -1.]))
    swap = x.new_tensor([[0., 1.], [1., 0.]])
    inverse = swap @ q.T @ swap
    # Preserve physical lattice spacing on rectangular arrays.
    h, w = x.shape[-2:]
    scale = x.new_tensor([(w - 1) / 2, (h - 1) / 2]).clamp_min(.5)
    inverse = inverse * scale[None, :] / scale[:, None]
    theta = torch.cat((inverse, x.new_zeros(2, 1)), 1)[None].expand(x.shape[0], -1, -1)
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True), grid


def _flatten_fields(fields):
    return torch.cat((fields.scalar, fields.vector.flatten(1, 2), fields.tensor.flatten(1, 2)), 1)


def rotate_feature_representation(features, angle, reflect=False):
    """Bilinear spatial resampling plus the lifted continuous channel action.

    This is a diagnostic representation action on features, not exact assembly
    of a rotated PDE. At quarter turns it agrees with the lattice D4 action.
    """
    sampled, grid = _resample_spatial(features, angle, reflect)
    basis = torch.eye(10, dtype=features.dtype, device=features.device).reshape(10, 10, 1, 1)
    # rows contain the output for each basis input; inverse maps fields->features
    lifting = _flatten_fields(stencil_to_o2_fields(basis)).reshape(10, 10)
    inverse = torch.linalg.inv(lifting)
    fields = _flatten_fields(transform_o2_fields(stencil_to_o2_fields(sampled), angle, reflect))
    return torch.einsum('bihw,ij->bjhw', fields, inverse), grid


def rotation_resampling_diagnostic(model, features, *, angles=(17., 33., 61.),
                                   reflections=(False, True), output=None):
    """Measure approximate arbitrary-angle feature/stencil covariance on CPU.

    A fixed-grid interpolation diagnostic; not an OOD solve or a proof of
    continuous spatial equivariance. Normalization is reapplied to transformed
    nine-channel directions because the arbitrary-angle channel action need not
    preserve the fixed-stencil Euclidean norm.
    """
    if features.ndim != 4 or features.shape[1] != 10 or min(features.shape[-2:]) < 5:
        raise ValueError('rotation diagnostics require (B,10,H,W) with H,W >= 5')
    if any(not math.isfinite(float(a)) for a in angles):
        raise ValueError('finite angles required')
    module = deepcopy(model).cpu().float().eval()
    features = features.detach().cpu().float()
    rows = []
    with torch.inference_mode():
        base_direction, base_gain = module.direction_and_gain(features)
        for reflection in reflections:
            for degrees in angles:
                angle = math.radians(float(degrees))
                rotated, grid = rotate_feature_representation(features, angle, reflection)
                actual_direction, actual_gain = module.direction_and_gain(rotated)
                pseudo_features = torch.cat((torch.zeros_like(base_direction[:, 0, :1]), base_direction[:, 0]), 1)
                expected, _ = rotate_feature_representation(pseudo_features, angle, reflection)
                expected = expected[:, 1:]
                expected = expected / expected.square().sum(1, keepdim=True).clamp_min(1e-12).sqrt()
                h, w = features.shape[-2:]
                margin = min(int(getattr(module, 'receptive_radius', 1)) + 1, (min(h, w) - 3) // 2)
                mask = torch.zeros((features.shape[0], h, w), dtype=torch.bool)
                mask[:, margin:h-margin, margin:w-margin] = True
                mask &= (grid[..., 0].abs() <= 1 - 2 * margin / (w - 1))
                mask &= (grid[..., 1].abs() <= 1 - 2 * margin / (h - 1))
                if not mask.any():
                    raise ValueError('no interior samples remain after rotation/cropping')
                difference = (actual_direction[:, 0] - expected).permute(0, 2, 3, 1)[mask]
                reference = expected.permute(0, 2, 3, 1)[mask]
                rows.append(dict(angle_deg=float(degrees), reflected=bool(reflection),
                    interior_pixels=int(mask.sum()), crop_margin=margin,
                    relative_direction_l2=float(torch.linalg.vector_norm(difference) /
                        torch.linalg.vector_norm(reference).clamp_min(1e-12)),
                    direction_max_abs=float(difference.abs().max()),
                    relative_gain_l2=float(torch.linalg.vector_norm(actual_gain - base_gain) /
                        torch.linalg.vector_norm(base_gain).clamp_min(1e-12))))
    result = dict(kind='fixed_lattice_bilinear_resampling_diagnostic',
        architecture=research_spec(module), samples=rows, exact_arbitrary_spatial_equivariance_claim=False,
        includes_errors=['bilinear resampling', 'finite support', 'fixed nine-point normalization', 'boundary/global pooling'],
        reference='full-rank signed-moment O2 feature action; not reassembled rotated PDE',
        numerical_rotation_ood_solve='not measured by this helper')
    if output is not None:
        output = Path(output)
        if output.exists():
            raise FileExistsError('rotation diagnostic already exists')
        _atomic_json(output, result)
    return result
