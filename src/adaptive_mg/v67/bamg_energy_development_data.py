"""Nested stratified TRAIN sizes and size-independent development validation.

Only development operators are assembled here. Geometry formulas, supports,
manufactured RHS and reconstruction use the established research-data kernels.
Each complete TRAIN block contains 20 cases per grid/family; adding blocks never
changes earlier cases. Validation is generated first so duplicate rejection also
preserves its identity when TRAIN grows. Callers must share the same immutable
historical exclusion set across the data-size comparison.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from ..provenance import json_safe
from .fixed_classical import baseline_metadata
from .research_data import (
    VERSION, FAMILIES, TRAIN_ANGLES, ResearchCase, _validate_plan, _example,
    _selection_record, _attach_selection, _hash, _write_json,
    historical_operator_index,
)

SAMPLING = 'stratified_nested_v1'
BLOCK_SIZE = dict(train=20, validation=10)


def validate_plan(plan):
    _validate_plan(plan)
    if plan.get('development_sampling') != SAMPLING or plan.get('discretization') != 'q1':
        raise ValueError('Declare stratified_nested_v1 Q1 development sampling')
    if not plan['smoke']:
        for split, block_size in BLOCK_SIZE.items():
            if plan['splits'][split]['per_family'] % block_size:
                raise ValueError('Research development counts must contain complete stratified blocks')


def _specification(split, spec, n, family, index, retry=0):
    """Metadata only; block/category assignments stay fixed during duplicate retries."""
    if split not in BLOCK_SIZE:
        raise ValueError('Only TRAIN and validation are available here')
    size = BLOCK_SIZE[split]
    block, position = divmod(index, size)
    key = [spec['seed'], n, FAMILIES.index(family), block, 0x444556]
    seed = int(np.random.SeedSequence(key + [position, retry]).generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(seed)

    def permutation(dimension, tag=0x4C4853):
        return np.random.default_rng(np.random.SeedSequence(key + [dimension, tag])).permutation(size)

    epsilon_bins, contrast_bins = permutation(0), permutation(1)
    # Pair the lowest epsilon stratum with the highest contrast stratum once
    # per heterogeneous TRAIN block. This retains every marginal LHS stratum.
    hard_position = int(np.argmin(epsilon_bins))
    if split == 'train' and family in FAMILIES[3:]:
        high = int(np.argmax(contrast_bins))
        contrast_bins[hard_position], contrast_bins[high] = contrast_bins[high], contrast_bins[hard_position]

    def u(dimension):
        bins = epsilon_bins if dimension == 0 else contrast_bins if dimension == 1 else permutation(dimension)
        return (int(bins[position]) + rng.random()) / size

    def category(dimension, options):
        # Cycling the remainder across blocks balances categories even when
        # their number (e.g. three checker sizes) does not divide the block.
        return options[(int(permutation(dimension, 0x434154)[position]) + block * size) % len(options)]

    epsilon = float(10 ** (-3 + 2 * u(0)))
    contrast = float(10 ** (1 + 2 * u(1)))
    angle = float(category(2, TRAIN_ANGLES))
    if family == 'near_isotropic':
        epsilon, contrast = .8 + .2 * u(3), 1.
    elif family == 'aligned_anisotropy':
        angle, contrast = float(category(4, (0., 90.))), 1.
    elif family == 'rotated_anisotropy':
        contrast = 1.
    center_x, center_y = .35 + .3 * u(5), .35 + .3 * u(6)
    width_x, width_y = .13 + .11 * u(7), .13 + .11 * u(8)
    case = ResearchCase(n, epsilon, angle, contrast,
        'uniform' if family in FAMILIES[:3] else family, family, 'standard',
        (center_x-width_x, center_x+width_x, center_y-width_y, center_y+width_y),
        int(category(9, (3, 4, 5))), .06 + .07 * u(10),
        float(np.log(contrast) / 2), .7 + 1.8 * u(11),
        2 * np.pi * u(12), 2 * np.pi * u(13), 'q1')
    return dict(name=f'{split}_{family}_n{n}_{index}_r{retry}', case=asdict(case),
                seed=seed, scale=float(10 ** (-.5 + u(14))), split=split,
                family=family, retry=retry,
                stratification=dict(version=SAMPLING, block=block, position=position,
                    block_size=size, paired_hard_stratum=(split == 'train' and
                        family in FAMILIES[3:] and position == hard_position)))


def specifications(plan, split):
    """Yield block-major metadata: every smaller complete design is a prefix."""
    validate_plan(plan)
    if split not in BLOCK_SIZE:
        raise ValueError('Only TRAIN and validation are available here')
    spec, size = plan['splits'][split], BLOCK_SIZE[split]
    for start in range(0, spec['per_family'], size):
        for n in spec['sizes']:
            for family in spec['families']:
                for index in range(start, min(start + size, spec['per_family'])):
                    yield _specification(split, spec, n, family, index)


def _generate_split(split, plan, forbidden, rules):
    examples, records, rejected = [], [], 0
    spec = plan['splits'][split]
    for original in specifications(plan, split):
        info = original['stratification']
        index = info['block'] * info['block_size'] + info['position']
        for retry in range(100):
            record = original if retry == 0 else _specification(
                split, spec, original['case']['n'], original['family'], index, retry)
            example = _example(record)
            if example.group_digest in forbidden:
                rejected += 1
                continue
            forbidden.add(example.group_digest)
            selection = _selection_record(example, rules)
            _attach_selection(example, selection, rules)
            example.research_split, example.research_family = split, record['family']
            example.research_ood = None
            record.update(example.manifest(), selection=selection,
                rhs_digest=hashlib.sha256(example.b.tobytes()).hexdigest(),
                exact_digest=hashlib.sha256(example.exact.tobytes()).hexdigest())
            examples.append(example)
            records.append(record)
            break
        else:
            raise ValueError('Could not generate a normalized-operator-disjoint sample after 100 attempts')
    return examples, records, rejected


def coverage(records):
    """Small audit of accepted parameters; distinct A counts, not RHS repetitions."""
    groups = {}
    for record in records:
        case = record['case']
        groups.setdefault(f'n{case["n"]}/{record["family"]}', []).append(record)
    return {key: dict(count=len(rows),
        angles=dict(sorted(Counter(str(r['case']['angle_deg']) for r in rows).items())),
        checker_cells=dict(sorted(Counter(str(r['case']['checker_cells']) for r in rows).items())),
        paired_hard_cases=sum(r['stratification']['paired_hard_stratum'] for r in rows))
        for key, rows in groups.items()}


def prepare_development_data(output_dir, rules, *, plan, historical_roots=()):
    """Write legacy-compatible immutable manifests using the new sampling plan."""
    output = Path(output_dir).resolve()
    if (output / 'dataset_plan.json').exists() or (output / 'development_manifest.json').exists():
        raise FileExistsError('Development data already committed; load it instead')
    if not historical_roots:
        raise ValueError('Explicit historical artifact roots are required for leakage exclusion')
    validate_plan(plan)
    history = historical_operator_index(historical_roots, exclude=[output])
    forbidden = set(history['normalized_operator_digests'])
    splits, records, rejected = {}, {}, {}
    # Crucial: validation is independent of TRAIN size even if a sampled A
    # collides; TRAIN, not validation, then retries within the same strata.
    for split in ('validation', 'train'):
        splits[split], records[split], rejected[split] = _generate_split(split, plan, forbidden, rules)
    manifest = dict(version=VERSION, plan_digest=_hash(plan),
        **baseline_metadata(rules, include_contract=True), historical_index=history,
        splits={split: records[split] for split in ('train', 'validation')},
        rejected_normalized_duplicates=rejected, normalized_operator_disjoint=True,
        final_operators_materialized=False, final_operators_selected=False,
        final_operators_solved=False, development_sampling=SAMPLING,
        coverage={split: coverage(rows) for split, rows in records.items()},
        nesting_contract='Same seed, baseline, support and immutable historical exclusion set; complete TRAIN blocks are prefixes; validation is shared.')
    manifest = json_safe(manifest)
    _write_json(output / 'dataset_plan.json', plan, exclusive=True)
    _write_json(output / 'development_manifest.json', manifest, exclusive=True)
    return splits['train'], splits['validation'], manifest
