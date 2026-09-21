"""A-only deterministic classical selection and a common base for C/S/P/SP.

The shipped rules are interpretable, frozen *candidates*, not an empirical claim
of optimality. Selection reads only a sparse matrix and structured grid shape.
No case label, RHS, solution, residual, timing or checkpoint enters a decision.
Features cost O(nnz + N) after sparse-format conversion. No hierarchy or trial
solve is constructed by the selector. Existing classical primitives are reused.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, fields
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp

from ..grid import as_shape
from ..strategy import ALL_STRATEGIES, STRATEGIES, get_strategy
from .config import AdaptiveConfig
from .solver import PreparedAdaptiveMG


RULE_STRATEGIES = (
    ('unsupported_stencil', 'line_alt_operator_full'),
    ('unresolved_tensor', 'line_alt_operator_full'),
    ('heterogeneous_orientation', 'line_alt_operator_alternating'),
    ('heterogeneous', 'line_alt_bilinear_full'),
    ('aligned_x_heterogeneous', 'line_alt_bilinear_full'),
    ('aligned_y_heterogeneous', 'line_alt_bilinear_full'),
    ('diagonal_positive_heterogeneous', 'line_alt_bilinear_full'),
    ('rotated_heterogeneous', 'line_alt_bilinear_full'),
    ('aligned_x', 'line_x_operator_semi_y'),
    ('aligned_y', 'line_y_operator_semi_x'),
    ('diagonal_positive', 'line_diag45_operator_full'),
    ('rotated', 'line_alt_operator_full'),
    ('near_isotropic', 'jacobi_bilinear_full'),
    ('moderate', 'chebyshev_operator_full'),
)


@dataclass(frozen=True)
class StrongRules:
    """Versioned fixed thresholds; optional leaf mapping is fitted offline only.

    ``strategy_by_rule`` is converted to immutable ordered pairs. Runtime does
    not load or modify measurements. Persist ``to_dict()`` before held-out runs.
    ``provenance`` is descriptive metadata, never a classifier input.
    """
    version: str = 'a_only_portfolio_rules_v2'
    anisotropy_min: float = 4.0
    isotropic_ratio_max: float = 1.5
    alignment_angle_deg: float = 12.0
    diagonal_angle_tolerance_deg: float = 12.0
    contrast_min: float = 8.0
    log_diagonal_std_min: float = 0.5
    orientation_variation_min: float = 0.15
    local_anisotropic_fraction_min: float = 0.10
    off_stencil_fraction_max: float = 0.01
    valid_tensor_fraction_min: float = 0.80
    strategy_by_rule: tuple = RULE_STRATEGIES
    fallback_strategy_name: str = 'line_alt_bilinear_full'
    require_coverage: bool = False
    coverage_by_rule: tuple = ()  # immutable (leaf, minimum_n, maximum_n) entries
    provenance: str = 'theory_guided_defaults; not tuned on audit; empirical strength unvalidated'

    def __post_init__(self):
        if self.version != 'a_only_portfolio_rules_v2':
            raise ValueError('unsupported/stale strong rules; recalibrate with a_only_portfolio_rules_v2')
        values = self.to_dict()
        for key, value in values.items():
            if key not in {'version', 'strategy_by_rule', 'provenance', 'fallback_strategy_name', 'require_coverage', 'coverage_by_rule'}:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'{key} must be finite numeric')
        if not 1 <= self.isotropic_ratio_max < self.anisotropy_min:
            raise ValueError('require 1 <= isotropic_ratio_max < anisotropy_min')
        if not 0 < self.alignment_angle_deg < 45 or not 0 < self.diagonal_angle_tolerance_deg < 45:
            raise ValueError('angular tolerances must be in (0,45) degrees')
        if self.contrast_min <= 1 or self.log_diagonal_std_min <= 0:
            raise ValueError('heterogeneity thresholds must be positive (contrast >1)')
        for key in ('orientation_variation_min', 'local_anisotropic_fraction_min',
                    'off_stencil_fraction_max', 'valid_tensor_fraction_min'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f'{key} must be in [0,1]')
        pairs = tuple(self.strategy_by_rule.items()) if isinstance(self.strategy_by_rule, dict) else tuple(self.strategy_by_rule)
        if len(pairs) != len(RULE_STRATEGIES) or len(dict(pairs)) != len(pairs):
            raise ValueError('strategy mapping needs every rule exactly once')
        mapping = dict(pairs)
        if set(mapping) != set(self.rule_ids):
            raise ValueError('unknown or missing rule in strategy mapping')
        for name in mapping.values():
            get_strategy(name)
        object.__setattr__(self, 'strategy_by_rule', tuple((key, mapping[key]) for key in self.rule_ids))
        get_strategy(self.fallback_strategy_name)
        if not isinstance(self.require_coverage, bool):
            raise ValueError('require_coverage must be boolean')
        coverage = tuple(tuple(row) for row in self.coverage_by_rule)
        if len({row[0] for row in coverage}) != len(coverage):
            raise ValueError('duplicate coverage leaf')
        for row in coverage:
            if (len(row) != 3 or row[0] not in self.rule_ids
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in row[1:])
                    or not 3 <= row[1] <= row[2]):
                raise ValueError('invalid size coverage')
        object.__setattr__(self, 'coverage_by_rule', coverage)
        if not isinstance(self.provenance, str):
            raise ValueError('provenance must be text')

    @property
    def rule_ids(self):
        return tuple(key for key, _ in RULE_STRATEGIES)

    def to_dict(self):
        return {**{f.name: getattr(self, f.name) for f in fields(self)},
                'strategy_by_rule': dict(self.strategy_by_rule)}

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))

    def digest(self):
        cached = getattr(self, '_digest_cache', None)
        if cached is None:
            cached = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()
            object.__setattr__(self, '_digest_cache', cached)
        return cached

    def replace_strategies(self, mapping, *, provenance=None):
        updated = dict(self.strategy_by_rule)
        updated.update(mapping)
        return replace(self, strategy_by_rule=updated,
                       provenance=self.provenance if provenance is None else provenance)


def load_strong_rules(path=None):
    return StrongRules() if path is None else StrongRules.from_dict(json.loads(Path(path).read_text()))


def classical_bank(name='controlled'):
    """Existing strategies: 16 controlled or the 48 primitive combinations.

    Smoothers: Jacobi, Chebyshev, line-x/y/alternating, and +45-degree line.
    Transfers: bilinear or operator-dependent; coarsening: full, x-first,
    y-first, alternating. Semi modes eventually reduce both dimensions. The
    bank has no -45-degree line, AMG aggregation, ILU or external solver.
    """
    if name == 'controlled':
        return STRATEGIES
    if name == 'all':
        return ALL_STRATEGIES
    raise ValueError('bank must be controlled or all')


def _tensor_metrics(xx, yy, xy):
    trace = xx + yy
    gap = np.hypot(xx - yy, 2 * xy)
    high, low = (trace + gap) * .5, (trace - gap) * .5
    floor = np.maximum(np.abs(trace) * 1e-12, np.finfo(float).tiny)
    valid = (high > floor) & (low > -floor)
    ratio = np.maximum(high, floor) / np.maximum(low, floor)
    angle = .5 * np.arctan2(2 * xy, xx - yy)
    return ratio, angle, valid, gap, trace


def operator_features(a, n):
    """Signed interior second moments plus scale-invariant sparse proxies.

    Grid indexing is x-major (i*ny+j), on a unit rectangular domain. For
    constant-coefficient P1 diffusion, -1/2 sum Aij d_ij d_ij^T recovers K up
    to a positive scale. Positive FE off-diagonals must retain their signs.
    For heterogeneous/general operators these are proxies, not identified
    physical coefficients. The outermost node ring is excluded from moments
    because removed Dirichlet couplings would otherwise bias orientations.
    """
    shape = as_shape(n)
    nx, ny = shape
    if min(shape) < 3:
        raise ValueError('selector requires at least three nodes per dimension')
    if not sp.issparse(a):
        raise TypeError('A must be a SciPy sparse matrix')
    if a.shape != (nx * ny, nx * ny):
        raise ValueError('operator/grid shape mismatch')
    csr = a.tocsr(copy=True)
    if np.iscomplexobj(csr.data):
        raise ValueError('real operator required')
    csr = csr.astype(np.float64, copy=False)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    diag = csr.diagonal()
    if not np.isfinite(csr.data).all() or np.any(diag <= 0):
        raise ValueError('finite positive diagonal required')
    scale = float(np.max(np.abs(csr.data), initial=0))
    weights = csr.data / scale
    row = np.repeat(np.arange(nx * ny), np.diff(csr.indptr))
    col = csr.indices
    dx, dy = col // ny - row // ny, col % ny - row % ny
    off = row != col
    mass = np.abs(weights)
    total = max(float(mass[off].sum()), np.finfo(float).tiny)
    local = off & (np.abs(dx) <= 1) & (np.abs(dy) <= 1)
    center = (row // ny > 0) & (row // ny < nx - 1) & (row % ny > 0) & (row % ny < ny - 1)
    edges = local & center
    interior_indices = np.arange(nx * ny).reshape(shape)[1:-1, 1:-1].ravel()
    ex, ey = dx[edges] / (nx + 1.), dy[edges] / (ny + 1.)
    signed = -.5 * weights[edges]
    def moment(value):
        return np.bincount(row[edges], weights=value, minlength=nx*ny)[interior_indices]
    xx, yy, xy = moment(signed * ex * ex), moment(signed * ey * ey), moment(signed * ex * ey)
    ratio, angle, valid, gap, trace = _tensor_metrics(xx, yy, xy)
    if valid.any():
        global_values = _tensor_metrics(xx[valid].mean(), yy[valid].mean(), xy[valid].mean())
        global_ratio, global_angle = float(global_values[0]), float(np.rad2deg(global_values[1]))
        # Gap weighting suppresses undefined eigenvector angles of isotropic rows.
        orientation_weight = np.where(valid, gap, 0.)
        denominator = float(orientation_weight.sum())
        if denominator > np.finfo(float).tiny:
            resultant = np.hypot(np.sum(orientation_weight * np.cos(2*angle)),
                                  np.sum(orientation_weight * np.sin(2*angle))) / denominator
            orientation_variation = float(np.clip(1-resultant, 0, 1))
        else:
            orientation_variation = 0.
        local_ratio_log_std = float(np.std(np.log(np.maximum(ratio[valid], 1.))))
    else:
        global_ratio, global_angle, orientation_variation, local_ratio_log_std = 1., 0., 0., 0.
    log_diag = np.log(diag) - np.log(scale)
    # Clamp only the reporting proxy in extreme dynamic range; log statistics
    # still represent contrast without overflow or an expensive quantile sort.
    log_contrast = float(log_diag.max() - log_diag.min())
    x_strength = float(mass[local & (dy == 0)].sum()) / total
    y_strength = float(mass[local & (dx == 0)].sum()) / total
    diagonal_strength = float(mass[local & (dx != 0) & (dy != 0)].sum()) / total
    return dict(
        nx=nx, ny=ny, unknowns=nx*ny, nnz=int(csr.nnz),
        moment_rows=int(len(interior_indices)), valid_tensor_fraction=float(np.mean(valid)),
        tensor_anisotropy_ratio=global_ratio, principal_angle_deg=global_angle,
        local_anisotropy_log_std=local_ratio_log_std,
        orientation_variation=orientation_variation,
        # Fixed descriptive 4x marker; rule thresholds are separately evaluated below.
        local_anisotropic_fraction=float(np.mean(valid & (ratio >= 4.))),
        diagonal_contrast_proxy=float(np.exp(min(log_contrast, 700.))),
        log_diagonal_std=float(np.std(log_diag)),
        neighbor_log_diagonal_jump=float(np.sum(mass[local]*np.abs(log_diag[row[local]]-log_diag[col[local]]))/total),
        x_coupling_fraction=x_strength, y_coupling_fraction=y_strength,
        diagonal_coupling_fraction=diagonal_strength,
        off_stencil_fraction=float(mass[off & ~local].sum()) / total,
        positive_offdiagonal_fraction=float(mass[off & (weights > 0)].sum()) / total,
    )


def _classify(features, rules):
    f = features
    anisotropic = f['tensor_anisotropy_ratio'] >= rules.anisotropy_min
    angle = f['principal_angle_deg']
    checks = dict(
        unsupported_stencil=f['off_stencil_fraction'] > rules.off_stencil_fraction_max,
        unresolved_tensor=f['valid_tensor_fraction'] < rules.valid_tensor_fraction_min,
        heterogeneous_orientation=(f['orientation_variation'] >= rules.orientation_variation_min
            and f['local_anisotropic_fraction'] >= rules.local_anisotropic_fraction_min),
        heterogeneous=(f['diagonal_contrast_proxy'] >= rules.contrast_min
            or f['log_diagonal_std'] >= rules.log_diagonal_std_min),
        aligned_x=anisotropic and abs(angle) <= rules.alignment_angle_deg,
        aligned_y=anisotropic and abs(angle) >= 90.-rules.alignment_angle_deg,
        diagonal_positive=anisotropic and abs(angle-45.) <= rules.diagonal_angle_tolerance_deg,
        rotated=anisotropic,
        near_isotropic=f['tensor_anisotropy_ratio'] <= rules.isotropic_ratio_max,
        moderate=True,
    )
    # Stencil validity/orientation mixtures remain separate safety classes.
    # Direction and coefficient variation are otherwise orthogonal axes.
    special = next((k for k in ('unsupported_stencil', 'unresolved_tensor',
                                'heterogeneous_orientation') if checks[k]), None)
    direction = next(k for k in ('aligned_x', 'aligned_y', 'diagonal_positive',
                                 'rotated', 'near_isotropic', 'moderate') if checks[k])
    heterogeneous = checks['heterogeneous']
    if special is not None:
        rule_id = special
    elif heterogeneous and direction in ('aligned_x', 'aligned_y', 'diagonal_positive', 'rotated'):
        rule_id = direction + '_heterogeneous'
    else:
        rule_id = 'heterogeneous' if heterogeneous else direction
    for key in rules.rule_ids:
        checks.setdefault(key, key == rule_id)
    return rule_id, dict(ordered_checks=checks, first_matching_rule=rule_id,
                         direction_class=direction, heterogeneous=heterogeneous,
                         classification='direction_cross_heterogeneity_v2',
                         thresholds={k: v for k, v in rules.to_dict().items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)})



@dataclass(frozen=True)
class StrongSelection:
    strategy_name: str
    rule_id: str
    features: dict
    rule_evidence: dict
    selector_seconds: float
    rules_digest: str
    status: str = 'deterministic_strong_candidate; strength_requires_held_out_measurement'

    def to_dict(self):
        return dict(self.__dict__)


def select_strong_strategy(a, n, rules=None):
    start = perf_counter()
    rules = rules or StrongRules()
    if not isinstance(rules, StrongRules):
        raise TypeError('rules must be frozen StrongRules')
    features = operator_features(a, n)
    rule_id, evidence = _classify(features, rules)
    digest = rules.digest()
    strategy = dict(rules.strategy_by_rule)[rule_id]
    coverage = {key: (lo, hi) for key, lo, hi in rules.coverage_by_rule}
    size = max(features['nx'], features['ny'])
    covered = rule_id in coverage and coverage[rule_id][0] <= size <= coverage[rule_id][1]
    if rules.require_coverage and not covered:
        strategy = rules.fallback_strategy_name
    evidence.update(size_coverage_supported=covered,
                    fallback_for_coverage=bool(rules.require_coverage and not covered))
    return StrongSelection(strategy, rule_id, features, evidence, perf_counter()-start, digest)


class PreparedStrongMG(PreparedAdaptiveMG):
    """Select once per A/rules generation, then reuse the existing v6.7 solver.

    C and H_S/H_P/H_SP share exactly the selected classical hierarchy contract,
    including safeguard recovery. Explicit neural branches are diagnostics;
    existing fixed-strategy certificates/policy calibration are never promoted
    by this adapter. A new RHS resets policy state but reuses A-only selection.
    Assign a new frozen ``rules`` value or call ``replace_operator`` to reselect.
    Prepared instances are not thread-safe, matching PreparedAdaptiveMG.
    """
    def __init__(self, a, n, components=None, config=None, rules=None):
        self.rules = rules or StrongRules()
        self.selection_count = 0
        super().__init__(a, n, components, config or AdaptiveConfig(mode='classical', branch='C'))

    def _build(self):
        start = perf_counter()
        self.selection = select_strong_strategy(self.source_a, self.shape, self.rules)
        self.selection_count += 1
        self.config = replace(self.config, mg=replace(self.config.mg, strategy_name=self.selection.strategy_name))
        self._rules_snapshot = self.selection.rules_digest
        super()._build()
        # Parent charges pending setup exactly once. Include selection, rule
        # hashing/config replacement, validation, hierarchy and factorization.
        self.initial_setup_seconds = perf_counter()-start

    def _refresh_key(self):
        super()._refresh_key()
        self.cache_key = hashlib.sha256((self.cache_key + self._rules_snapshot).encode()).hexdigest()

    def _ensure_fresh(self):
        if not isinstance(self.rules, StrongRules):
            raise TypeError('rules must be frozen StrongRules')
        if self.rules.digest() != self._rules_snapshot:
            self._build()
            return
        super()._ensure_fresh()

    def certificate_valid(self):
        # Strong selector deployment scope is not part of old certification.
        return False

    def _policy_status(self, cfg, branch):
        if cfg.branch == 'C' or cfg.mode == 'classical':
            return 'not_applicable'
        if cfg.branch != 'auto' or cfg.mode in {'hybrid', 'burst'}:
            return 'forced_component_with_safety_uncalibrated_strong_baseline'
        return 'uncalibrated_strong_baseline_policy'

    def _solve(self, b, x0=None):
        pending = self.setup_pending
        result = super()._solve(b, x0)
        result.stats['selector_calls'] = int(pending)
        result.stats['selector_seconds'] = self.selection.selector_seconds if pending else 0.
        result.stats['selector_total_calls'] = self.selection_count
        result.abstention['selected_classical_strategy'] = self.selection.strategy_name
        result.abstention['strong_rule_id'] = self.selection.rule_id
        result.abstention['strong_rules_digest'] = self.selection.rules_digest
        result.abstention['strong_policy_calibrated'] = False
        return result
