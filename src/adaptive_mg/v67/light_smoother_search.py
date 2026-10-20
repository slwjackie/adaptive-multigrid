"""Training-only, measured coordinate search for directional sweep relaxation.

The oracle keeps student P, all actual Galerkin operators, and line factors
fixed.  It changes two numbers at one nonterminal level and scores the complete
raw solve, never a fallback-assisted solve.  A separate set of training RHS
confirms the final candidate against all-ones relaxation.
"""
from __future__ import annotations

from itertools import product
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..provenance import json_safe, operator_digest, stable_norm
from .research_data import _hash, _write_json
from .light_smoother_runtime import PreparedLightMG


VERSION = 'light-smoother-coordinate-search-v1'


def validate_search_settings(settings):
    """Return the complete, fingerprinted finite search protocol.

    ``repeats=1`` and single RHS are allowed for plumbing smoke tests, not for
    a claim of timing improvement.  The research defaults use three repeated
    measurements of two optimization and two confirmation RHS.
    """
    defaults = dict(grid=[.8, 1., 1.2], bounds=[.8, 1.2], passes=1, repeats=3,
                    rhs_count=2, confirmation_rhs_count=2,
                    minimum_improvement=.03, seed=20261021)
    unknown = set(settings) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown search settings: {sorted(unknown)}')
    spec = {**defaults, **settings}
    bounds = np.asarray(spec['bounds'], dtype=float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 < bounds[0] <= 1 <= bounds[1] or bounds[0] == bounds[1]):
        raise ValueError('Search bounds must be positive, finite, and contain one')
    grid = np.asarray(spec['grid'], dtype=float)
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or len(set(map(float, grid))) != len(grid)
            or np.any(grid < bounds[0]) or np.any(grid > bounds[1])):
        raise ValueError('Search grid must have distinct finite in-bounds values')
    for key in ('passes', 'repeats', 'rhs_count', 'confirmation_rhs_count'):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f'search.{key} must be a positive integer')
    if type(spec['seed']) is not int or spec['seed'] < 0:
        raise ValueError('Search seed must be a nonnegative integer')
    if (not np.isfinite(spec['minimum_improvement'])
            or not 0 <= spec['minimum_improvement'] < 1):
        raise ValueError('minimum_improvement must be in [0,1)')
    spec['grid'], spec['bounds'] = grid.tolist(), bounds.tolist()
    return spec


def _vector_digest(vector):
    return hashlib.sha256(np.asarray(vector, dtype=np.float64).tobytes()).hexdigest()


def search_rhs(example, *, count, seed, partition):
    """Independent deterministic manufactured RHS from this training A only.

    The partition string is part of the seed. No held-out A, RHS, PDE-family
    label, or stored exact solution enters coefficient features or selection.
    """
    if partition not in ('optimization', 'confirmation'):
        raise ValueError('Unknown training RHS partition')
    coordinate = np.arange(1, example.n + 1) / (example.n + 1)
    x, y = np.meshgrid(coordinate, coordinate, indexing='ij')
    result = []
    for index in range(count):
        key = _hash((VERSION, example.group_digest, seed, partition, index))
        rng = np.random.default_rng(int(key[:16], 16))
        exact = np.sin(np.pi*x) * np.sin(np.pi*y)
        for _ in range(8):
            k, l = rng.integers(1, min(example.n, 12)+1, size=2)
            exact += rng.uniform(-.5, .5) * np.sin(k*np.pi*x) * np.sin(l*np.pi*y)
        result.append(np.asarray(example.a @ exact.ravel(), dtype=np.float64))
    return np.asarray(result)


def measure_raw_solve(prepared, rhs, cfg):
    """Real wall time to the existing tolerance, with no fallback or adaptation.

    The reusable hierarchy has already been set up for this offline oracle.
    A fresh zero starting iterate and true residual are used on every trial.
    Failed and rejected trials retain their elapsed time and residual history.
    """
    started = perf_counter()
    x = np.zeros_like(rhs, dtype=np.float64)
    initial = stable_norm(rhs)
    threshold = max(cfg.mg.absolute_tolerance, cfg.mg.tolerance*initial)
    history = [initial]
    reason, error, cycles = 'max_cycles', None, 0
    try:
        while history[-1] > threshold and cycles < cfg.mg.max_cycles:
            x = prepared.raw_cycle(x, rhs)
            cycles += 1
            norm = stable_norm(rhs - prepared.a @ x)
            history.append(norm)
            if not np.isfinite(norm):
                reason = 'nonfinite_raw_cycle'
                break
            if norm > cfg.mg.divergence_factor*max(initial, threshold):
                reason = 'diverged_raw_cycle'
                break
        success = bool(np.isfinite(history[-1]) and history[-1] <= threshold)
        if success:
            reason = 'converged'
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        success = False
        reason, error = 'raw_cycle_exception', f'{type(exc).__name__}: {exc}'
    wall = perf_counter() - started
    return json_safe(dict(success=success, cycles=cycles, residual_history=history,
        stopping_threshold=threshold, final_true_residual=history[-1],
        wall_seconds=wall, stop_reason=reason, error=error,
        fallback_used=False, online_adaptation_used=False))


def candidate_summary(records, *, repeats, rhs_count):
    """A candidate is eligible only when every requested raw solve succeeded."""
    expected = repeats*rhs_count
    identities = {(r['repeat'], r['rhs_index']) for r in records}
    valid = (len(records) == expected and len(identities) == expected
             and identities == set(product(range(repeats), range(rhs_count))))
    valid = bool(valid and all(r['success'] and r['wall_seconds'] is not None
        and np.isfinite(r['wall_seconds']) and r['wall_seconds'] > 0
        and r['final_true_residual'] is not None
        and r['final_true_residual'] <= r['stopping_threshold'] for r in records))
    totals = [sum(r['wall_seconds'] for r in records if r['repeat'] == repeat)
              for repeat in range(repeats)] if valid else []
    return dict(eligible=valid, success_count=sum(bool(r['success']) for r in records),
                measurement_count=len(records), expected_measurements=expected,
                median_rhs_batch_seconds=float(np.median(totals)) if valid else None,
                repeat_rhs_batch_seconds=totals,
                total_cycles=sum(r['cycles'] for r in records))


def choose_candidate(summaries, incumbent, minimum_improvement):
    """Strict timing improvement, or successful recovery from a failed incumbent.

    Ties keep the incumbent.  Failing candidates cannot win even if they fail
    fast. This helper has no access to validation/final data.
    """
    eligible = [i for i, item in enumerate(summaries) if item['eligible']]
    if not eligible:
        return incumbent, 'no_successful_candidate'
    best = min(eligible, key=lambda i: (summaries[i]['median_rhs_batch_seconds'], i))
    old = summaries[incumbent]
    if not old['eligible']:
        return best, 'success_recovery'
    if (best != incumbent and summaries[best]['median_rhs_batch_seconds']
            < (1-minimum_improvement)*old['median_rhs_batch_seconds']):
        return best, 'measured_time_improvement'
    return incumbent, 'no_confirmed_time_improvement'


def _coefficient_key(coefficients):
    return _hash(np.asarray(coefficients, dtype=np.float64).tolist())


def search_case(example, components, cfg, settings, output, *, resume=False):
    """Search one training operator; resume after every completed measurement.

    ``output`` is a case-specific directory. ``progress.json`` is an atomic
    ledger of complete measurements; ``result.json`` is the final oracle label.
    Interrupted in-flight measurements can be repeated; completed measurements
    and previously selected coordinate steps are never re-timed on resume.
    """
    if getattr(example, 'research_split', None) != 'train':
        raise ValueError('Offline coefficient search accepts training-split A only')
    spec = validate_search_settings(settings)
    out = Path(output)
    manifest = dict(version=VERSION, settings=spec, solver=cfg.to_dict(),
        operator_digest=operator_digest(example.a), normalized_operator_digest=example.group_digest,
        example=example.name, transfer_signature=components.expert_signature('transfer'),
        split='train', fallback_allowed=False)
    path = out/'progress.json'
    if path.exists():
        state = json.loads(path.read_text())
        if not resume:
            raise FileExistsError('Search case exists; use --resume or a new run')
        if state.get('manifest') != json_safe(manifest):
            raise ValueError('Search operator/P/settings changed')
        payload = {key: value for key, value in state.items() if key != 'digest'}
        if state.get('digest') != _hash(payload):
            raise ValueError('Search progress ledger changed')
        if state.get('status') == 'complete':
            result = json.loads((out/'result.json').read_text())
            if state.get('result_digest') != _hash(result):
                raise ValueError('Completed search result changed')
            return result
    else:
        if resume and out.exists() and any(out.iterdir()):
            raise FileNotFoundError('Missing search progress ledger')
        state = dict(manifest=manifest, status='running', trials=[],
                     completed_measurements=0, offline_seconds=0., setup_seconds=0.)

    started = perf_counter()
    prior_elapsed = state['offline_seconds']

    def persist():
        state['offline_seconds'] = prior_elapsed + perf_counter()-started
        state['completed_measurements'] = len(state['trials'])
        payload = {key: value for key, value in state.items() if key != 'digest'}
        state['digest'] = _hash(payload)
        _write_json(path, state)

    persist()
    setup_started = perf_counter()
    try:
        prepared = PreparedLightMG(example.a, example.n, components, cfg, bounds=spec['bounds'])
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        state['setup_seconds'] += perf_counter()-setup_started
        state.setdefault('setup_failures', []).append(dict(
            error=f'{type(exc).__name__}: {exc}', fallback_used=False,
            outcome='Search stopped without dropping this operator or fabricating a label'))
        persist()
        raise
    state['setup_seconds'] += perf_counter()-setup_started
    ones = np.ones_like(prepared.get_coefficients())
    if ones.ndim != 2 or ones.shape[1] != 2 or not len(ones):
        raise ValueError('Search requires a nonterminal learned-P level')
    prepared.set_coefficients(ones)
    rhs = {name: search_rhs(example, count=count, seed=spec['seed'], partition=name)
           for name, count in (('optimization', spec['rhs_count']),
                               ('confirmation', spec['confirmation_rhs_count']))}
    rhs_digests = {key: [_vector_digest(b) for b in vectors] for key, vectors in rhs.items()}
    if (set(rhs_digests['optimization']) & set(rhs_digests['confirmation'])
            or any(len(set(values)) != len(values) for values in rhs_digests.values())):
        raise ValueError('Optimization and confirmation RHS must be distinct')
    journal = {item['id']: item for item in state['trials']}
    if len(journal) != len(state['trials']):
        raise ValueError('Duplicate search measurements')

    def evaluate(label, candidates, partition):
        by_candidate = [[] for _ in candidates]
        keys = [_coefficient_key(c) for c in candidates]
        seed = int(_hash((spec['seed'], example.group_digest, label))[:16], 16)
        base_order = np.random.default_rng(seed).permutation(len(candidates))
        for repeat in range(spec['repeats']):
            for rhs_index, b in enumerate(rhs[partition]):
                # Each repeat/RHS rotates a seeded permutation, distributing
                # early/late cache and thermal positions across candidates.
                order = np.roll(base_order, repeat+rhs_index)
                for candidate_index in order:
                    candidate_index = int(candidate_index)
                    identity = f'{label}/{keys[candidate_index]}/{repeat}/{rhs_index}'
                    meta = dict(id=identity, phase=label, partition=partition,
                        coefficients=np.asarray(candidates[candidate_index]).tolist(),
                        candidate_index=candidate_index, repeat=repeat, rhs_index=rhs_index,
                        rhs_digest=rhs_digests[partition][rhs_index])
                    if identity in journal:
                        item = journal[identity]
                        if any(item.get(k) != v for k, v in meta.items()):
                            raise ValueError('Search measurement provenance changed')
                    else:
                        prepared.set_coefficients(candidates[candidate_index])
                        item = {**meta, **measure_raw_solve(prepared, b, cfg)}
                        state['trials'].append(item)
                        journal[identity] = item
                        persist()
                    by_candidate[candidate_index].append(item)
        return [candidate_summary(records, repeats=spec['repeats'], rhs_count=len(rhs[partition]))
                for records in by_candidate]

    current, rounds = ones.copy(), []
    try:
        for pass_index in range(spec['passes']):
            changed = False
            for level in range(len(current)):
                candidates = [current.copy()]
                seen = {_coefficient_key(current)}
                for pair in product(spec['grid'], repeat=2):
                    candidate = current.copy()
                    candidate[level] = pair
                    key = _coefficient_key(candidate)
                    if key not in seen:
                        candidates.append(candidate)
                        seen.add(key)
                label = f'coordinate/{pass_index}/{level}'
                summaries = evaluate(label, candidates, 'optimization')
                winner, reason = choose_candidate(summaries, 0, spec['minimum_improvement'])
                changed = changed or winner != 0
                current = candidates[winner].copy()
                rounds.append(dict(pass_index=pass_index, level=level, phase=label,
                    candidates=[c.tolist() for c in candidates], summaries=summaries,
                    selected_index=winner, selected_coefficients=current.tolist(), reason=reason))
            if not changed:
                break
        proposed = current.copy()
        candidates = [ones] if np.array_equal(ones, current) else [ones, current]
        confirmation_scores = evaluate('confirmation', candidates, 'confirmation')
        winner, reason = choose_candidate(confirmation_scores, 0, spec['minimum_improvement'])
        selected = candidates[winner].copy()
        prepared.set_coefficients(selected)
        result = json_safe(dict(version=VERSION, status='complete', manifest=manifest,
            features=np.asarray(prepared.features).tolist(),
            selected_coefficients=selected.tolist(), baseline_coefficients=ones.tolist(),
            optimization_selected_coefficients=proposed.tolist(),
            confirmed_improvement=winner != 0,
            timing_improvement_confirmed=winner != 0 and reason == 'measured_time_improvement',
            optimization_rounds=rounds,
            confirmation=dict(candidates=[c.tolist() for c in candidates], summaries=confirmation_scores,
                              selected_index=winner, reason=reason),
            trials=state['trials'], rhs_digests=rhs_digests,
            setup_seconds=state['setup_seconds'],
            offline_seconds=prior_elapsed+perf_counter()-started,
            fallback_allowed=False, online_adaptation_allowed=False,
            objective='median repeated solve-to-tolerance wall time; fixed reused hierarchy',
            coefficient_scope='directional whole-sweep corrections, per nonterminal level',
            optimality='finite coordinate candidates only; not a global optimum',
            deployment_cost_scope='offline oracle only; deployment must build a fresh hierarchy'))
        _write_json(out/'result.json', result)
        state.update(status='complete', result_digest=_hash(result))
        persist()
        return result
    except BaseException:
        persist()
        raise
