"""Preregistered, streamed Q1 final/OOD cases for the paired CNN study.

This module never reads scores or selects models. A caller must validate its
freeze/claim lease before supplying ``freeze_identity``. Metadata planning does
not assemble a single operator. Only ``load_case`` materializes one case at a
time; its immutable record binds all operator, parameter and RHS fingerprints.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ..pde import assemble_q1_stiffness
from ..provenance import operator_digest
from .research_data import (
    FAMILIES, HOLDOUT, TRAIN_ANGLES, OOD_ANGLES, ResearchCase, _assemble_case,
    _file_hash, _hash, _write_json,
)

VERSION = 'bamg-energy-expanded-holdout-v1'
RHS_KINDS = ('manufactured', 'random', 'localized')
APPLICABLE = dict(final=FAMILIES, ood_grid=FAMILIES,
                  ood_rotation=FAMILIES[2:], ood_anisotropy=FAMILIES[1:],
                  ood_contrast=FAMILIES[3:], ood_topology=FAMILIES[3:])
TOPOLOGIES = {
    'local_patch': ('ellipses', 'annulus', 'l_shape'),
    'checkerboard': ('shifted_bricks', 'rotated_checks', 'stretched_checks'),
    'channel': ('sinuous_pair', 'crossing', 'branching'),
    'variable_coefficient': ('oblique_modes', 'gaussian_islands', 'radial_waves'),
}
SUPPORT = dict(epsilon=[1e-3, 1e-1], near_isotropic_epsilon=[.8, 1.],
    contrast=[10., 1000.], angles=list(TRAIN_ANGLES),
    ood_rotation_centers=list(OOD_ANGLES), ood_rotation_jitter_degrees=2.,
    ood_epsilon=[1e-5, float(10**-3.1)], ood_contrast=[float(10**3.2), 1e5],
    topology_variants={k: list(v) for k, v in TOPOLOGIES.items()})


def make_holdout_plan(*, smoke=False, seed=2026110971, discretization='q1',
                      training_sizes=None):
    """Return 1,270 research-case specifications as metadata, without assembly.

    A full default plan has 560 final + 140 grid + 150 rotation + 180 anisotropy
    + 120 contrast + 120 topology cases. Final is distribution-matched in its
    parameter support (stratified, not IID sampling). Other splits change just
    the named axis; multiple RHS are reported separately from independent A's.
    """
    training_sizes = list(training_sizes if training_sizes is not None else
                          ([7] if smoke else [15, 31, 63, 127]))
    if not _valid_sizes(training_sizes):
        raise ValueError('Training sizes must be distinct 2**L-1 integers >=3')
    common = training_sizes if smoke else [n for n in training_sizes if n >= 31]
    if not common:
        common = training_sizes
    grid_size = 2 * (max(training_sizes) + 1) - 1
    final_count, ood_count = (2, 2) if smoke else (20, 10)
    splits = {
        'final': dict(sizes=training_sizes, families=list(FAMILIES), per_family=final_count),
        'ood_grid': dict(sizes=[grid_size], families=list(FAMILIES), per_family=final_count),
        'ood_rotation': dict(sizes=common, families=list(FAMILIES[2:]), per_family=ood_count),
        'ood_anisotropy': dict(sizes=common, families=list(FAMILIES[1:]), per_family=ood_count),
        'ood_contrast': dict(sizes=common, families=list(FAMILIES[3:]), per_family=ood_count),
        'ood_topology': dict(sizes=common, families=list(FAMILIES[3:]), per_family=3 if smoke else ood_count),
    }
    for index, split in enumerate(HOLDOUT):
        values = splits[split]
        values.update(seed=seed + 1000003 * (index + 1),
                      count=len(values['sizes']) * len(values['families']) * values['per_family'])
    plan = dict(version=VERSION, smoke=bool(smoke), seed=seed,
                discretization=discretization, training_sizes=training_sizes,
                splits=splits, cases_expected=sum(s['count'] for s in splits.values()),
                rhs_kinds=list(RHS_KINDS), rhs_normalization='unit_l2',
                rhs_repetitions_are_independent_operators=False,
                sampling='independent jitter within per-parameter Latin-hypercube strata; balanced categorical values',
                final_parameter_support='same ranges, discrete angles and family formulas as development TRAIN',
                axes=dict(ood_grid='unseen grid only', ood_rotation='held-out angles only',
                          ood_anisotropy='epsilon outside TRAIN only',
                          ood_contrast='contrast outside TRAIN only',
                          ood_topology='new coefficient geometry/formula only'),
                support=deepcopy(SUPPORT),
                materialization_requires_freeze=True, selection_allowed=False,
                final_generalization_scope='declared synthetic Q1 diffusion distributions, not arbitrary PDEs')
    validate_plan(plan)
    return plan


def _valid_sizes(values):
    return (isinstance(values, list) and bool(values) and len(set(values)) == len(values)
            and all(type(n) is int and n >= 3 and not ((n + 1) & n) for n in values))


def validate_plan(plan):
    if plan.get('version') != VERSION or plan.get('discretization') != 'q1':
        raise ValueError('Expanded holdout supports this version of Q1 diffusion only')
    if type(plan.get('smoke')) is not bool or type(plan.get('seed')) is not int or plan['seed'] < 0:
        raise ValueError('Declare smoke and a nonnegative holdout seed')
    if not _valid_sizes(plan.get('training_sizes')) or set(plan.get('splits', {})) != set(HOLDOUT):
        raise ValueError('Declare all final/OOD splits and valid training sizes')
    if plan.get('rhs_kinds') != list(RHS_KINDS) or plan.get('rhs_normalization') != 'unit_l2':
        raise ValueError('The fixed RHS suite is manufactured, random and localized, unit L2')
    if plan.get('materialization_requires_freeze') is not True or plan.get('selection_allowed') is not False:
        raise ValueError('Final/OOD cannot be materialized for model selection')
    if plan.get('support') != SUPPORT:
        raise ValueError('Declared parameter support differs from the implemented generator')
    seeds = []
    for split, values in plan['splits'].items():
        families = values.get('families')
        if (not _valid_sizes(values.get('sizes')) or not isinstance(families, list)
                or not families or len(set(families)) != len(families)
                or set(families) - set(APPLICABLE[split])):
            raise ValueError('Invalid grid/family combination for ' + split)
        if type(values.get('per_family')) is not int or values['per_family'] < 1:
            raise ValueError('Positive per-family operator count required')
        if values.get('count') != len(values['sizes']) * len(families) * values['per_family']:
            raise ValueError('Declared count differs from grid/family cells')
        if type(values.get('seed')) is not int or values['seed'] < 0:
            raise ValueError('Each split needs an independent nonnegative seed')
        seeds.append(values['seed'])
        shared = set(values['sizes']) & set(plan['training_sizes'])
        if split == 'ood_grid' and shared:
            raise ValueError('Grid OOD must use unseen grid sizes')
        if split != 'ood_grid' and shared != set(values['sizes']):
            raise ValueError('Only grid OOD may change grid size')
    if len(set(seeds)) != len(seeds):
        raise ValueError('Every final/OOD split needs a distinct seed')
    if plan.get('cases_expected') != sum(v['count'] for v in plan['splits'].values()):
        raise ValueError('Total operator count differs from split counts')


def specifications(plan):
    """Stable planned IDs; no operators, exact solutions or scores are created."""
    validate_plan(plan)
    return [dict(case_id=f'{split}_{family}_n{n}_{index}',
                 name=f'{split}_{family}_n{n}_{index}', split=split, family=family,
                 n=n, index=index)
            for split in HOLDOUT for n in plan['splits'][split]['sizes']
            for family in plan['splits'][split]['families']
            for index in range(plan['splits'][split]['per_family'])]


def _parameters(plan, spec, retry):
    values = plan['splits'][spec['split']]
    count, index = values['per_family'], spec['index']
    key = [values['seed'], spec['n'], FAMILIES.index(spec['family'])]
    seed = int(np.random.SeedSequence(key + [index, retry]).generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(seed)
    def u(dimension):
        permutation = np.random.default_rng(np.random.SeedSequence(key + [dimension, 0x4C4853])).permutation(count)
        return (int(permutation[index]) + rng.random()) / count
    def category(dimension, options):
        permutation = np.random.default_rng(np.random.SeedSequence(key + [dimension, 0x434154])).permutation(count)
        return options[int(permutation[index]) % len(options)]
    split, family = spec['split'], spec['family']
    epsilon = 10 ** ((-5 + 1.9 * u(0)) if split == 'ood_anisotropy' else (-3 + 2 * u(0)))
    contrast = 10 ** ((3.2 + 1.8 * u(1)) if split == 'ood_contrast' else (1 + 2 * u(1)))
    angle = float(category(2, OOD_ANGLES if split == 'ood_rotation' else TRAIN_ANGLES))
    if split == 'ood_rotation':
        angle += 4 * u(3) - 2
    if family == 'near_isotropic':
        epsilon, contrast = .8 + .2 * u(4), 1.
    elif family == 'aligned_anisotropy':
        angle, contrast = float(category(5, (0., 90.))), 1.
    elif family == 'rotated_anisotropy':
        contrast = 1.
    center_x, center_y = .35 + .3 * u(6), .35 + .3 * u(7)
    width_x, width_y = .13 + .11 * u(8), .13 + .11 * u(9)
    case = ResearchCase(spec['n'], float(epsilon), angle, float(contrast),
        'uniform' if family in FAMILIES[:3] else family, family, 'standard',
        (center_x-width_x, center_x+width_x, center_y-width_y, center_y+width_y),
        int(category(10, (3, 4, 5))), .06 + .07 * u(11),
        float(np.log(contrast) / 2), .7 + 1.8 * u(12),
        2 * np.pi * u(13), 2 * np.pi * u(14), 'q1')
    geometry = None
    if split == 'ood_topology':
        geometry = dict(variant=category(15, TOPOLOGIES[family]),
            center=[.25 + .5 * u(16), .25 + .5 * u(17)],
            radii=[.12 + .14 * u(18), .09 + .12 * u(19)],
            orientation=180 * u(20), offset=[u(21), u(22)],
            frequency=1. + 2. * u(23), phase=2 * np.pi * u(24),
            bend=.08 + .15 * u(25), stretch=.65 + .7 * u(26))
    return dict(case=asdict(case), geometry=geometry, seed=seed,
                scale=float(10 ** (-.5 + u(27))), retry=retry)


def _topology_field(parameters, x, y):
    """Positive tensor fields with multiple genuinely different held-out shapes."""
    case, g = parameters['case'], parameters['geometry']
    theta = np.deg2rad(case['angle_deg'])
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    tensor = rotation @ np.diag([1., case['epsilon']]) @ rotation.T
    x, y = np.asarray(x), np.asarray(y)
    cx, cy = g['center']; rx, ry = g['radii']
    phi = np.deg2rad(g['orientation'])
    dx, dy = x-cx, y-cy
    xx, yy = np.cos(phi)*dx + np.sin(phi)*dy, -np.sin(phi)*dx + np.cos(phi)*dy
    variant = g['variant']
    if case['family'] == 'local_patch':
        if variant == 'ellipses':
            hard = ((xx/rx)**2 + (yy/ry)**2 < 1) | (((x-(1-cx))/(.65*rx))**2 + ((y-(1-cy))/(.8*ry))**2 < 1)
        elif variant == 'annulus':
            radius = (xx/rx)**2 + (yy/ry)**2
            hard = (radius < 1) & (radius > .28)
        else:
            hard = ((np.abs(xx) < rx) & (np.abs(yy) < .32*ry)) | ((np.abs(xx + .65*rx) < .35*rx) & (np.abs(yy) < ry))
    elif case['family'] == 'checkerboard':
        m = case['checker_cells']
        if variant == 'shifted_bricks':
            row = np.floor(y * (m+1) + g['offset'][1]).astype(int)
            col = np.floor(x * m + .5 * (row % 2) + g['offset'][0]).astype(int)
        elif variant == 'rotated_checks':
            row = np.floor((xx+.5) * m + g['offset'][0]).astype(int)
            col = np.floor((yy+.5) * m + g['offset'][1]).astype(int)
        else:
            row = np.floor(x * m * g['stretch'] + g['offset'][0]).astype(int)
            col = np.floor(y * m / g['stretch'] + g['offset'][1]).astype(int)
        hard = (row + col) % 2 == 0
    elif case['family'] == 'channel':
        width = case['channel_width']
        curve = g['bend'] * np.sin(2*np.pi*g['frequency']*x + g['phase'])
        if variant == 'sinuous_pair':
            hard = (np.abs(y - .3 - curve) < width) | (np.abs(y - .72 + .6*curve) < .7*width)
        elif variant == 'crossing':
            hard = (np.abs(yy) < width) | (np.abs(xx) < .7*width)
        else:
            trunk = (x <= cx) & (np.abs(y-cy-.35*curve) < width)
            branch = (x > cx) & ((np.abs(y-cy-.6*(x-cx)-curve) < .7*width) |
                                 (np.abs(y-cy+.6*(x-cx)+curve) < .7*width))
            hard = trunk | branch
    else:
        if variant == 'oblique_modes':
            modulation = (.65*np.sin(2*np.pi*g['frequency']*xx+g['phase'])*np.cos(4*np.pi*yy)
                          + .35*np.cos(6*np.pi*x-2*np.pi*y+case['geometry_phase']))
        elif variant == 'gaussian_islands':
            modulation = (1.4*np.exp(-(xx/rx)**2-(yy/ry)**2)
                          - .8*np.exp(-((x-(1-cx))/ry)**2-((y-(1-cy))/rx)**2) - .3)
        else:
            modulation = np.cos(2*np.pi*g['frequency']*np.sqrt((xx/rx)**2+(yy/ry)**2)+g['phase'])
        # Keep scalar coefficient bounded by the same nominal contrast range.
        coefficient = np.exp(case['amplitude'] * np.clip(modulation, -1., 1.))
        return coefficient[..., None, None] * tensor
    return np.where(hard[..., None, None], case['contrast'] * tensor, np.eye(2))


def _localized_load(n, rng):
    """Q1 quadrature load for a smooth Gaussian source, independent of A."""
    center = rng.uniform(.2, .8, 2)
    width = rng.uniform(.06, .15, 2)
    ii, jj = np.meshgrid(np.arange(n+1), np.arange(n+1), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    local = np.zeros((len(ii), 4))
    gauss = (.5-.5/np.sqrt(3.), .5+.5/np.sqrt(3.))
    for xi in gauss:
        for eta in gauss:
            x, y = (ii+xi)/(n+1), (jj+eta)/(n+1)
            f = np.exp(-.5*((x-center[0])/width[0])**2-.5*((y-center[1])/width[1])**2)
            basis = np.array([(1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta])
            local += (.25/(n+1)**2) * f[:, None] * basis
    offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    gi, gj = ii[:, None]+offsets[:, 0], jj[:, None]+offsets[:, 1]
    valid = (gi >= 1) & (gi <= n) & (gj >= 1) & (gj <= n)
    result = np.zeros(n*n)
    np.add.at(result, ((gi-1)*n+gj-1)[valid], local[valid])
    return result


def _array_hash(value):
    a = np.ascontiguousarray(value, dtype=np.float64)
    h = hashlib.sha256(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _assemble(parameters):
    values = parameters['case']
    n = values['n']
    if parameters['geometry'] is None:
        case = ResearchCase(**{**values, 'patch_bounds': tuple(values['patch_bounds'])})
        a = parameters['scale'] * _assemble_case(case)
    else:
        a = parameters['scale'] * assemble_q1_stiffness(n, lambda x, y: _topology_field(parameters, x, y))
    a = a.tocsr(); a.sum_duplicates(); a.sort_indices()
    if not np.isfinite(a.data).all() or np.any(a.diagonal() <= 0):
        raise ValueError('Generated holdout operator is nonfinite or has invalid diagonal')
    coordinate = np.arange(1, n+1)/(n+1)
    x, y = np.meshgrid(coordinate, coordinate, indexing='ij')
    exact = np.sin(np.pi*x)*np.sin(np.pi*y)
    rng = np.random.default_rng(parameters['seed'] ^ 0x4D47524853)
    for _ in range(4):
        k, l = rng.integers(1, min(n, 12)+1, size=2)
        exact += rng.uniform(-.3, .3)*np.sin(k*np.pi*x)*np.sin(l*np.pi*y)
    exact = exact.ravel()
    rhs = dict(manufactured=np.asarray(a @ exact),
               random=np.random.default_rng(parameters['seed'] ^ 0x52414E444F4D).normal(size=n*n),
               localized=_localized_load(n, np.random.default_rng(parameters['seed'] ^ 0x4C4F43414C)))
    for kind, value in rhs.items():
        norm = float(np.linalg.norm(value))
        if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError('Generated holdout RHS is invalid')
        rhs[kind] = value / norm
        if kind == 'manufactured':
            exact /= norm
    return a, rhs, exact


def _identity(plan, forbidden_digests, freeze_identity):
    validate_plan(plan)
    if not isinstance(freeze_identity, dict) or any(
            not isinstance(freeze_identity.get(k), str) or len(freeze_identity[k]) != 64
            for k in ('freeze_sha256', 'claim_fingerprint')):
        raise ValueError('A validated frozen evaluation claim identity is required')
    forbidden = sorted(set(forbidden_digests))
    if any(not isinstance(h, str) or len(h) != 64 for h in forbidden):
        raise ValueError('Forbidden normalized operator digests must be SHA256 strings')
    return dict(version=VERSION, plan_sha256=_hash(plan), forbidden_sha256=_hash(forbidden),
                freeze_identity=deepcopy(freeze_identity)), set(forbidden)


def _read(path):
    return json.loads(Path(path).read_text())


def _manifest(directory, identity, plan, *, create=False):
    directory = Path(directory)
    path = directory / 'data_manifest.json'
    if path.exists():
        state = _read(path)
        if state.get('identity') != identity or state.get('plan') != plan:
            raise ValueError('Holdout data plan/forbidden/freeze identity changed')
        return state
    if not create:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Holdout data cache has no identity manifest')
        return dict(identity=identity, plan=plan, cases={}, completed=False)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('New holdout data cache must be empty')
    state = dict(identity=identity, plan=plan, cases={}, completed=False)
    _write_json(path, state)
    return state


def validate_cache(directory, *, plan, forbidden_digests, freeze_identity):
    """Audit metadata without A assembly; recover an atomic-save orphan index."""
    identity, forbidden = _identity(plan, forbidden_digests, freeze_identity)
    state = _manifest(directory, identity, plan)
    expected = {s['case_id']: s for s in specifications(plan)}
    recovered = False
    for path in sorted((Path(directory) / 'cases').glob('*.json')):
        case_id = path.stem
        if case_id in state['cases']:
            continue
        record = _read(path)
        if (case_id not in expected or record.get('identity') != identity
                or record.get('specification') != expected[case_id]):
            raise ValueError('Unexpected orphan case in holdout data cache')
        parameters = record['parameters']
        if parameters != json.loads(json.dumps(_parameters(plan, expected[case_id], parameters['retry']))):
            raise ValueError('Orphan holdout parameters differ from deterministic specification')
        state['cases'][case_id] = dict(sha256=_file_hash(path),
            normalized_operator_digest=record['normalized_operator_digest'],
            split=expected[case_id]['split'], n=expected[case_id]['n'], family=expected[case_id]['family'])
        recovered = True
    seen = set(forbidden)
    for case_id, item in state['cases'].items():
        if case_id not in expected:
            raise ValueError('Unexpected case in holdout data cache')
        path = Path(directory) / 'cases' / (case_id + '.json')
        if not path.is_file() or _file_hash(path) != item['sha256']:
            raise ValueError('Holdout case metadata hash changed: ' + case_id)
        record = _read(path)
        if record.get('identity') != identity or record.get('specification') != expected[case_id]:
            raise ValueError('Holdout case identity differs')
        group = record['normalized_operator_digest']
        if group != item['normalized_operator_digest'] or group in seen:
            raise ValueError('Normalized-operator leakage in holdout data cache')
        seen.add(group)
    if recovered:
        state['completed'] = len(state['cases']) == plan['cases_expected']
        _write_json(Path(directory) / 'data_manifest.json', state)
    elif state.get('completed') != (len(state['cases']) == plan['cases_expected']):
        raise ValueError('Holdout data completion flag differs from case coverage')
    return state


def load_case(directory, spec, *, plan, forbidden_digests, freeze_identity, resume=True):
    """Materialize or exactly reconstruct ONE planned operator and its RHS suite.

    Root's final-evaluation lease serializes writes. Atomic case save precedes
    index publication; an orphan case record is checked/recovered on resume.
    Deduplication retries only replace duplicates, never difficult/failed solves.
    """
    identity, forbidden = _identity(plan, forbidden_digests, freeze_identity)
    expected = {s['case_id']: s for s in specifications(plan)}
    if spec.get('case_id') not in expected or spec != expected[spec['case_id']]:
        raise ValueError('Case is not part of the preregistered holdout plan')
    directory = Path(directory)
    state = _manifest(directory, identity, plan, create=True)
    case_id = spec['case_id']
    path = directory / 'cases' / (case_id + '.json')
    other_groups = {v['normalized_operator_digest'] for k, v in state['cases'].items() if k != case_id}
    forbidden |= other_groups
    if path.exists():
        if not resume:
            raise FileExistsError('Holdout case exists; use exact resume')
        if case_id in state['cases'] and _file_hash(path) != state['cases'][case_id]['sha256']:
            raise ValueError('Holdout case metadata hash changed')
        record = _read(path)
        if record.get('identity') != identity or record.get('specification') != spec:
            raise ValueError('Holdout case identity changed')
        parameters = record['parameters']
        if parameters != json.loads(json.dumps(_parameters(plan, spec, parameters['retry']))):
            raise ValueError('Holdout parameters differ from deterministic specification')
        a, rhs, exact = _assemble(parameters)
        actual = dict(operator_digest=operator_digest(a),
                      normalized_operator_digest=operator_digest(a, scale_invariant=True),
                      rhs_digests={k: _array_hash(v) for k, v in rhs.items()}, exact_digest=_array_hash(exact))
        if any(record.get(key) != value for key, value in actual.items()):
            raise ValueError('Holdout case/RHS does not reconstruct exactly')
        if actual['normalized_operator_digest'] in forbidden:
            raise ValueError('Normalized-operator leakage in reconstructed holdout')
    else:
        if case_id in state['cases']:
            raise ValueError('Indexed holdout case record disappeared')
        for retry in range(100):
            parameters = _parameters(plan, spec, retry)
            a, rhs, exact = _assemble(parameters)
            group = operator_digest(a, scale_invariant=True)
            if group in forbidden:
                continue
            record = dict(identity=identity, specification=spec, parameters=parameters,
                          operator_digest=operator_digest(a), normalized_operator_digest=group,
                          rhs_digests={k: _array_hash(v) for k, v in rhs.items()},
                          exact_digest=_array_hash(exact), duplicate_rejections=retry,
                          rhs_normalization='unit_l2', rhs_kinds=list(RHS_KINDS))
            _write_json(path, record)
            break
        else:
            raise ValueError('Cannot generate a disjoint holdout after 100 duplicate-only retries')
    item = dict(sha256=_file_hash(path), normalized_operator_digest=record['normalized_operator_digest'],
                split=spec['split'], n=spec['n'], family=spec['family'])
    if state['cases'].get(case_id) != item:
        state['cases'][case_id] = item
        state['completed'] = len(state['cases']) == plan['cases_expected']
        _write_json(directory / 'data_manifest.json', state)
    return dict(A=a, b=rhs['manufactured'], rhs=rhs, exact=exact,
                shape=(spec['n'], spec['n']), **spec,
                normalized_operator_digest=record['normalized_operator_digest'],
                operator_digest=record['operator_digest'], parameters=record['parameters'],
                data_record_sha256=item['sha256'])


def stream_cases(directory, *, plan, forbidden_digests, freeze_identity, resume=True):
    """Convenience iterator; callback-based benchmarks can instead use load_case."""
    validate_cache(directory, plan=plan, forbidden_digests=forbidden_digests,
                   freeze_identity=freeze_identity)
    for spec in specifications(plan):
        yield load_case(directory, spec, plan=plan, forbidden_digests=forbidden_digests,
                        freeze_identity=freeze_identity, resume=resume)
