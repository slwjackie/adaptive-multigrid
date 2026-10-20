"""Frozen-P multigrid with cached, direction-wide scalar line corrections.

The learned quantity is two numbers per nonterminal level.  Each ``S_d(r)``
is the *complete existing classical directional zebra sweep*, starting from
zero correction, including its sequential residual updates.  Only the completed
correction is scaled.  This is not line SOR and it never changes line factors.

Online adaptation is an optional per-operator episode.  Its raw candidate
cycles cannot use classical recovery; the deployed solve retains the existing
forced-component residual-growth rollback and near-tolerance classical tail.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import math

import numpy as np
import torch

from ..hierarchy import classical_cycle, apply_transfer, matvec
from ..provenance import operator_digest, stable_norm
from ..smoothers import _line_sweep, _zebra_order
from .autograd_sparse import SparseTensor
from .banks import Stats
from .limited import forced_config
from .solver import AdaptiveResult, PreparedAdaptiveMG


ONLINE_DEFAULTS = dict(enabled=False, method='regression_online', trigger_window=2,
    trigger_rho=.8, near_tolerance_factor=100., min_remaining_cycles=4.,
    max_calls=1, cooldown_cycles=2, max_candidates=4, gradient_steps=2,
    learning_rate=.05, time_budget_seconds=.05, candidate_step=.1,
    minimum_improvement=.02, candidate_cycles=1, level=0)


def validate_online(settings=None):
    supplied = {} if settings is None else dict(settings)
    if set(supplied)-set(ONLINE_DEFAULTS):
        raise ValueError(f'Unknown online settings: {sorted(set(supplied)-set(ONLINE_DEFAULTS))}')
    result = dict(ONLINE_DEFAULTS, **supplied)
    if type(result['enabled']) is not bool or result['method'] not in ('mlp_ttt', 'regression_online'):
        raise ValueError('Invalid online enable/method')
    for key in ('trigger_window', 'max_calls', 'cooldown_cycles', 'max_candidates',
                'gradient_steps', 'candidate_cycles'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(result['level']) is not int or result['level'] < -1:
        raise ValueError('Online level must be -1 (round robin) or a nonnegative index')
    for key in ('near_tolerance_factor', 'min_remaining_cycles', 'learning_rate',
                'time_budget_seconds', 'candidate_step'):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if (not 0 < result['trigger_rho'] <= 1 or not 0 <= result['minimum_improvement'] < 1
            or result['near_tolerance_factor'] < 1):
        raise ValueError('Invalid adaptation threshold')
    return result


def directional_smoothing(level, residual, omega, *, reverse=False, stats=None):
    """Scale entire x/y sweep corrections, retaining compiled cached solves.

    The unscaled directional sweep uses the real matrix columns to update its
    internal residual.  After scaling we form ``r - A @ (omega * correction)``
    before the next direction.  At omega=1 its already-computed residual is
    reused, exactly matching the original line_alt operation order.
    """
    values = np.asarray(omega, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError('Expected finite x/y direction gains')
    current = np.asarray(residual, dtype=np.float64).copy()
    correction = np.zeros_like(current)
    for direction, index in ((('y', 1), ('x', 0)) if reverse else (('x', 0), ('y', 1))):
        increment, unscaled_residual = _line_sweep(np.zeros_like(current), current.copy(),
            level.cache, direction, reverse=reverse, work=stats, safety=False)
        gain = values[index]
        correction += gain*increment
        if gain == 1.:
            current = unscaled_residual
        else:
            current -= level.a @ (gain*increment)
            if stats is not None:
                stats.matvecs += 1
                stats.work_flops += 2*level.a.nnz
    return correction


class _CachedSolve(torch.autograd.Function):
    """Differentiate only a RHS through an already-factored classical block."""
    @staticmethod
    def forward(ctx, rhs, solve):
        ctx.solve = solve
        return torch.as_tensor(solve(rhs.detach().cpu().numpy()), dtype=rhs.dtype)

    @staticmethod
    def backward(ctx, derivative):
        answer = ctx.solve(derivative.detach().cpu().numpy(), trans='T')
        return torch.as_tensor(answer, dtype=derivative.dtype), None


@dataclass
class _FixedTorchLevel:
    """Autograd wrappers around immutable deployed A/P and existing LU handles."""
    a: object
    p: object = None
    r: object = None
    coarse: object = None
    solve: object = None
    groups: object = None
    index: int = 0


def _fixed_torch_graph(level):
    result = _FixedTorchLevel(SparseTensor.from_scipy(level.a), index=level.index)
    if level.coarse is None:
        result.solve = level.lu.solve
        return result
    result.p, result.r = SparseTensor.from_scipy(level.p), SparseTensor.from_scipy(level.r)
    result.groups = {}
    for direction in ('x', 'y'):
        colours = level.cache.zebra_batches(direction)
        if colours is None:
            original = level.cache.get(direction)
            groups = [original[i] for i in _zebra_order(len(original), False)]
        else:
            groups = [group for group in colours if group is not None]
        result.groups[direction] = [(torch.tensor(group.indices), group.solve,
            SparseTensor.from_scipy(group.columns)) for group in groups]
    result.coarse = _fixed_torch_graph(level.coarse)
    return result


def _torch_smooth(level, residual, omega, reverse):
    correction, current = torch.zeros_like(residual), residual
    for direction, index in ((('y', 1), ('x', 0)) if reverse else (('x', 0), ('y', 1))):
        increment, inside = torch.zeros_like(residual), current
        groups = level.groups[direction]
        for indices, solve, columns in (reversed(groups) if reverse else groups):
            value = _CachedSolve.apply(inside[indices], solve)
            increment = increment.index_add(0, indices, value)
            inside = inside-columns.apply(value)
        delta = omega[index]*increment
        correction, current = correction+delta, current-level.a.apply(delta)
    return correction


def _torch_cycle(level, x, b, coefficients, mg):
    if level.coarse is None:
        return _CachedSolve.apply(b, level.solve)
    for _ in range(mg.pre_steps):
        x = x+_torch_smooth(level, b-level.a.apply(x), coefficients[level.index], False)
    rc = level.r.apply(b-level.a.apply(x))
    ec = _torch_cycle(level.coarse, torch.zeros_like(rc), rc, coefficients, mg)
    x = x+level.p.apply(ec)
    for _ in range(mg.post_steps):
        x = x+_torch_smooth(level, b-level.a.apply(x), coefficients[level.index], True)
    return x


def _online_counters():
    return dict(online_adaptation_seconds=0., online_adaptation_calls=0,
        online_adaptation_accepted=0, online_adaptation_rejected=0,
        online_probe_cycles=0, online_gradient_vcycles=0, online_gradient_steps=0,
        online_graph_builds=0, online_budget_exhausted=0, coefficient_changes=0)


class PreparedLightMG:
    """One frozen learned-P hierarchy, two scalar gains per nonterminal level.

    A prepared object is not thread safe.  All RHS reuse P, Ac, exact line LU,
    level features and coefficients.  Only optional output adapters change;
    the supplied P and MLP checkpoints are never trained or mutated here.
    """
    def __init__(self, a, n, components, cfg, predictor=None, online=None,
                 coefficients=None, bounds=None, *, compute_features=True,
                 allow_setup_fallback=False):
        started = perf_counter()
        if type(compute_features) is not bool or type(allow_setup_fallback) is not bool:
            raise ValueError('Setup options must be booleans')
        if predictor is not None and not compute_features:
            raise ValueError('A predictor requires actual-level summary features')
        self.config = forced_config(cfg, 'H_P')
        if (self.config.mg.pre_steps != 2 or self.config.mg.post_steps != 2
                or self.config.mg.coarsest_n != 3):
            raise ValueError('Light smoother study requires pre=post=2 and terminal side=3')
        self.components, self.predictor = components, predictor
        self.online = validate_online(online)
        self.online_enabled = self.online['enabled']
        inferred = getattr(predictor, 'bounds', (.8, 1.2))
        self.bounds = tuple(float(v) for v in (inferred if bounds is None else bounds))
        if (len(self.bounds) != 2 or not np.isfinite(self.bounds).all()
                or not 0 < self.bounds[0] <= 1 <= self.bounds[1] or self.bounds[0] >= self.bounds[1]):
            raise ValueError('Require finite gain bounds 0 < low <= 1 <= high')
        if predictor is not None and tuple(inferred) != self.bounds:
            raise ValueError('Predictor and runtime coefficient bounds differ')
        if self.online_enabled and self.online['method'] == 'mlp_ttt' and getattr(predictor, 'kind', None) != 'mlp':
            raise ValueError('MLP TTT requires a frozen MLP predictor')
        self.base = PreparedAdaptiveMG(a, n, components, self.config)
        self.a, self.shape = self.base.a, self.base.shape
        self.classical = self.base.classical
        if self.classical.strategy.smoother != 'line_alt' or self.classical.strategy.transfer != 'bilinear' or self.classical.strategy.coarsening != 'full':
            raise ValueError('Require line_alt + bilinear + full fallback')
        self.setup_stats = self.base.setup_stats
        self.source_digest = operator_digest(a)
        self.p_signature = components.expert_signature('transfer')
        self.cache_key = self.base.cache_key
        self.setup_details = dict(feature_extractions=0, feature_seconds=0.,
            coefficient_prediction_seconds=0., coefficient_generations=0,
            mlp_inference_calls=0, regression_predictions=0,
            hierarchy_rebuilds=0, factor_rebuilds=0, setup_fallback=False)
        self.setup_failure = None
        self.learned, self.levels, self.features = None, [], None
        self.coefficient_changes = 0
        depth, level = 0, self.classical
        while level.coarse is not None:
            depth, level = depth+1, level.coarse
        if not depth or self.online['level'] >= depth:
            raise ValueError('Require a nonterminal selected online level')
        try:
            self._prepare_light(predictor, coefficients, compute_features)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            if not allow_setup_fallback:
                raise
            # A deployed setup failure has the same recovery destination as a
            # rejected learned cycle.  Raw search never enables this option.
            self.setup_failure = f'{type(exc).__name__}: {exc}'
            self.setup_stats.setup_failures += 1
            self.setup_details.update(setup_fallback=True, setup_failure=self.setup_failure)
            self.learned, self.levels, self.features = None, [], None
            self.coefficients = np.empty((0, 2), dtype=np.float64)
        self.base_coefficients = self.coefficients.copy()
        self._graph = None
        self.factor_snapshot = self._factor_count()
        self.reset_online()
        self.initial_setup_seconds = perf_counter()-started
        self.setup_details.update(classical_setup_seconds=self.base.initial_setup_seconds,
            light_setup_seconds=max(0., self.initial_setup_seconds-self.base.initial_setup_seconds),
            line_factor_seconds=sum(level.cache.build_seconds for level in self.levels))
        self.setup_pending = True

    def _prepare_light(self, predictor, coefficients, compute_features):
        from .light_smoother_models import extract_features
        self.learned = self.base.ensure_branch('H_P', self.setup_stats)
        level = self.learned
        while level.coarse is not None:
            self.levels.append(level)
            level = level.coarse
        if compute_features:
            started, rows = perf_counter(), []
            try:
                for level in self.levels:
                    self.setup_details['feature_extractions'] += 1
                    rows.append(extract_features(level.a, level.shape, level.index, len(self.levels)+1))
                self.features = np.stack(rows)
            finally:
                self.setup_details['feature_seconds'] = perf_counter()-started
        initial = np.ones((len(self.levels), 2))
        if predictor is not None:
            started, rows = perf_counter(), []
            try:
                for row in self.features:
                    key = 'mlp_inference_calls' if predictor.kind == 'mlp' else 'regression_predictions'
                    self.setup_details[key] += 1
                    rows.append(predictor.predict(row))
                    self.setup_details['coefficient_generations'] += 1
                initial = np.stack(rows)
            finally:
                self.setup_details['coefficient_prediction_seconds'] = perf_counter()-started
        self.coefficients = np.ones_like(initial)
        self.set_coefficients(initial if coefficients is None else coefficients)

    def _factor_count(self):
        return sum(level.cache.factorization_count for level in self.levels)

    def get_coefficients(self):
        return self.coefficients.copy()

    def set_coefficients(self, values):
        values = np.asarray(values, dtype=np.float64)
        if (values.shape != (len(self.levels), 2) or not np.isfinite(values).all()
                or np.any(values < self.bounds[0]) or np.any(values > self.bounds[1])):
            raise ValueError('Invalid level x/y coefficients')
        self.coefficient_changes += int(not np.array_equal(self.coefficients, values))
        self.coefficients = values.copy()

    def prepare_warm(self):
        """All numerical banks and predictor outputs are eagerly ready already."""
        return self

    def set_online_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('enabled must be a boolean')
        self.online_enabled = enabled

    def reset_online(self):
        """Reset the episode, retaining immutable banks and no adapted prime state."""
        self.coefficients = self.base_coefficients.copy()
        self.coefficient_changes = 0
        self.online_totals = _online_counters()
        self.online_events = []
        self._last_call_cycle = -self.online['cooldown_cycles']
        self._lifetime_cycles = 0

    def raw_cycle(self, x, b, stats=None):
        """One complete V-cycle; no adaptation, safeguards, or fallback."""
        if self.setup_failure is not None:
            raise RuntimeError('Learned setup unavailable; raw candidate cannot use classical fallback: '+self.setup_failure)
        stats = Stats() if stats is None else stats
        def visit(level, x, b):
            stats.level_visits += 1
            if level.coarse is None:
                started = perf_counter()
                result = level.lu.solve(b)
                stats.coarse_solves += 1
                stats.coarse_solve_seconds += perf_counter()-started
                return result
            for _ in range(self.config.mg.pre_steps):
                started = perf_counter()
                x = x+directional_smoothing(level, b-matvec(level.a, x, stats),
                    self.coefficients[level.index], stats=stats)
                stats.classical_smooth_calls += 1
                stats.classical_smooth_seconds += perf_counter()-started
            rc = apply_transfer(level.r, b-matvec(level.a, x, stats), stats)
            ec = visit(level.coarse, np.zeros_like(rc), rc)
            x = x+apply_transfer(level.p, ec, stats)
            stats.learned_transfer_apply_calls += 2
            for _ in range(self.config.mg.post_steps):
                started = perf_counter()
                x = x+directional_smoothing(level, b-matvec(level.a, x, stats),
                    self.coefficients[level.index], reverse=True, stats=stats)
                stats.classical_smooth_calls += 1
                stats.classical_smooth_seconds += perf_counter()-started
            return x
        return visit(self.learned, np.asarray(x, dtype=np.float64), np.asarray(b, dtype=np.float64))

    def differentiable_cycle(self, x, b, coefficients):
        """Fixed deployed hierarchy, autograd through gains and RHS only.

        The existing sparse-autograd matvec backend and the *same* cached LU
        solve handles are used.  No P CNN, Galerkin rebuild or factorization is
        performed, including in the transpose-solve backward pass.
        """
        if self.setup_failure is not None:
            raise RuntimeError('Learned setup unavailable for differentiable candidate')
        if self._graph is None:
            self._graph = _fixed_torch_graph(self.learned)
        return _torch_cycle(self._graph, x, b, coefficients, self.config.mg)

    def _trigger(self, history, threshold):
        spec = self.online
        if (not spec['enabled'] or not self.online_enabled or self.online_totals['online_adaptation_calls'] >= spec['max_calls']
                or self.online_totals['online_adaptation_seconds'] >= spec['time_budget_seconds']
                or self._lifetime_cycles-self._last_call_cycle < spec['cooldown_cycles']
                or len(history) <= spec['trigger_window']
                or history[-1] <= spec['near_tolerance_factor']*threshold):
            return False
        rho = (history[-1]/max(history[-1-spec['trigger_window']], 1e-300))**(1/spec['trigger_window'])
        if not np.isfinite(rho) or rho < spec['trigger_rho']:
            return False
        remaining = math.inf if rho >= 1 else math.log(max(threshold/history[-1], 1e-300))/math.log(max(rho, 1e-300))
        return remaining >= spec['min_remaining_cycles']

    def _adapt(self, x, b):
        spec, started = self.online, perf_counter()
        original = self.get_coefficients()
        available = spec['time_budget_seconds']-self.online_totals['online_adaptation_seconds']
        level = (self.online_totals['online_adaptation_calls'] % len(self.levels)
                 if spec['level'] == -1 else spec['level'])
        event = dict(level=level, method=spec['method'], before=original.tolist(),
            accepted=False, reason='no_improving_candidate', trials=[],
            fallback_used=False, source='same_current_x_and_b')
        counters = _online_counters()
        counters['online_adaptation_calls'] = 1
        self._last_call_cycle = self._lifetime_cycles
        def within_budget():
            return perf_counter()-started < available
        def score(values):
            self.set_coefficients(values)
            y = x.copy()
            for _ in range(spec['candidate_cycles']):
                if not within_budget():
                    raise TimeoutError('Online cooperative time budget exhausted')
                y = self.raw_cycle(y, b)
                counters['online_probe_cycles'] += 1
            residual = stable_norm(b-self.a@y)
            event['trials'].append(dict(coefficients=values.tolist(), residual=residual))
            return residual
        best, best_score = original.copy(), math.inf
        try:
            baseline = score(original)
            event['baseline_residual'] = baseline
            best_score = baseline
            if spec['method'] == 'regression_online':
                proposals = []
                for axis in (0, 1):
                    for sign in (-1., 1.):
                        candidate = original.copy()
                        candidate[level, axis] = np.clip(candidate[level, axis]+sign*spec['candidate_step'], *self.bounds)
                        if not np.array_equal(candidate, original):
                            proposals.append(candidate)
                for candidate in proposals[:spec['max_candidates']]:
                    value = score(candidate)
                    if np.isfinite(value) and value < best_score:
                        best, best_score = candidate.copy(), value
            else:
                # Only two additive *output* parameters are trainable.  The MLP
                # itself, P and every matrix/factor remain untouched.
                base = torch.tensor(original, dtype=torch.float64)
                adapter = torch.zeros(2, dtype=torch.float64, requires_grad=True)
                # Two-parameter Adam, written explicitly to avoid constructing
                # a general optimizer (and its runtime import/setup overhead)
                # inside a budgeted online call.
                first, second = torch.zeros_like(adapter), torch.zeros_like(adapter)
                xt = torch.tensor(x, dtype=torch.float64).detach()
                bt = torch.tensor(b, dtype=torch.float64).detach()
                graph_was_missing = self._graph is None
                for step in range(1, spec['gradient_steps']+1):
                    if not within_budget():
                        raise TimeoutError('Online cooperative time budget exhausted')
                    gains = base.clone()
                    gains[level] = torch.clamp(base[level]+adapter, *self.bounds)
                    y = xt
                    for _ in range(spec['candidate_cycles']):
                        if not within_budget():
                            raise TimeoutError('Online cooperative time budget exhausted')
                        y = self.differentiable_cycle(y, bt, gains)
                        counters['online_gradient_vcycles'] += 1
                        if graph_was_missing:
                            counters['online_graph_builds'] += 1
                            graph_was_missing = False
                    residual = bt-self._graph.a.apply(y)
                    loss = torch.log(residual.square().sum().clamp_min(1e-300))
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite online adapter loss')
                    loss.backward()
                    if adapter.grad is None or not bool(torch.isfinite(adapter.grad).all()):
                        raise FloatingPointError('Nonfinite online adapter gradient')
                    with torch.no_grad():
                        first = .9*first+.1*adapter.grad
                        second = .999*second+.001*adapter.grad.square()
                        update = (first/(1-.9**step))/(torch.sqrt(second/(1-.999**step))+1e-8)
                        adapter = (adapter-spec['learning_rate']*update).detach().requires_grad_(True)
                    counters['online_gradient_steps'] += 1
                candidate = original.copy()
                candidate[level] = np.clip(original[level]+adapter.detach().numpy(), *self.bounds)
                value = score(candidate)
                if np.isfinite(value) and value < best_score:
                    best, best_score = candidate, value
            if np.isfinite(best_score) and best_score < baseline*(1-spec['minimum_improvement']):
                event.update(accepted=True, reason='same_state_residual_improved', selected_residual=best_score)
        except TimeoutError as exc:
            event['reason'] = str(exc)
            counters['online_budget_exhausted'] = 1
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            event['reason'] = f'{type(exc).__name__}: {exc}'
        finally:
            self.set_coefficients(best if event['accepted'] else original)
            counters['coefficient_changes'] = int(event['accepted'])
            counters['online_adaptation_accepted'] = int(event['accepted'])
            counters['online_adaptation_rejected'] = int(not event['accepted'])
            counters['online_adaptation_seconds'] = perf_counter()-started
            if counters['online_adaptation_seconds'] > available:
                counters['online_budget_exhausted'] = 1
            event.update(after=self.coefficients.tolist(), seconds=counters['online_adaptation_seconds'],
                         counters=counters, factor_rebuilds=self._factor_count()-self.factor_snapshot)
            self.online_events.append(event)
            for key, value in counters.items():
                self.online_totals[key] += value

    def raw_solve(self, b, x0=None):
        return self._solve(b, x0, safe=False)

    def solve(self, b, x0=None):
        return self._solve(b, x0, safe=True)

    def solve_many(self, bs, x0=None):
        bs = np.asarray(bs, dtype=np.float64)
        if bs.ndim != 2 or bs.shape[1] != self.a.shape[0]:
            raise ValueError('Expected RHS shape (batch,N)')
        starts = [None]*len(bs) if x0 is None else np.asarray(x0, dtype=np.float64)
        if x0 is not None and starts.shape != bs.shape:
            raise ValueError('Initial state batch differs')
        # Exactly as the existing prepared batch solver, check frozen inputs
        # once per batch rather than hashing every matrix/checkpoint per RHS.
        started = perf_counter()
        self._check_frozen()
        freshness_seconds = perf_counter()-started
        results = [self._solve(b, x, safe=True, checked=True) for b, x in zip(bs, starts)]
        if results:
            results[0].solve_seconds += freshness_seconds
            results[0].stats.update(freshness_checks=1, freshness_seconds=freshness_seconds)
        return results

    def _check_frozen(self):
        if operator_digest(self.base.source_a) != self.source_digest or self.components.expert_signature('transfer') != self.p_signature:
            raise ValueError('Frozen operator/P changed; create a new prepared light solver')

    def _solve(self, b, x0, *, safe, checked=False):
        started, mg, stats = perf_counter(), self.config.mg, Stats()
        if not safe and self.setup_failure is not None:
            raise RuntimeError('Raw candidate cannot use setup fallback')
        if not checked:
            freshness_started = perf_counter()
            self._check_frozen()
            stats.freshness_checks = 1
            stats.freshness_seconds = perf_counter()-freshness_started
        b = np.asarray(b, dtype=np.float64)
        x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
        if b.shape != (self.a.shape[0],) or x.shape != b.shape or not np.isfinite(b).all() or not np.isfinite(x).all():
            raise ValueError('Invalid b/x0')
        norm0 = stable_norm(b-self.a@x)
        if not np.isfinite(norm0):
            raise ValueError('Nonfinite initial residual')
        reference = norm0 if mg.residual_reference == 'initial' else stable_norm(b)
        threshold = max(mg.absolute_tolerance, mg.tolerance*reference)
        denominator = max(reference, mg.absolute_tolerance, 1e-300)
        history, relative = [norm0], [norm0/denominator]
        paths, times, trace = [], [], []
        executed, locked, stop = 0, self.setup_failure is not None, 'max_cycles'
        if locked:
            stats.lock_count += 1
            trace.append(dict(path='classical_setup_fallback', error=self.setup_failure))
        online_before, event_start = dict(self.online_totals), len(self.online_events)
        while executed < mg.max_cycles and history[-1] > threshold:
            cycle_started = perf_counter()
            if safe and not locked and history[-1] <= mg.near_tolerance_factor*threshold:
                locked = True
                stats.lock_count += 1
            if safe and not locked and self._trigger(history, threshold):
                self._adapt(x, b)
            path = 'classical' if locked else 'hybrid'
            candidate = None
            error = None
            if not locked:
                stats.attempted_neural_cycles += 1
                stats.branch_H_P_attempts += 1
            try:
                candidate = (classical_cycle(self.classical, x, b, mg, stats) if locked
                             else self.raw_cycle(x, b, stats))
            except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
                if locked:
                    raise
                error = str(exc)
                candidate = np.full_like(x, np.nan)
            executed += 1
            self._lifetime_cycles += 1
            stats.trial_cycles += 1
            norm = stable_norm(b-self.a@candidate)
            stats.matvecs += 1
            rho = norm/max(history[-1], 1e-300)
            rejected = not locked and safe and (not np.isfinite(norm) or rho >= mg.safety_growth*(1+mg.safety_rtol_slack))
            if rejected:
                stats.rollback_count += 1
                stats.rejected_neural_cycles += 1
                stats.lock_count += 1
                locked = True
                if executed >= mg.max_cycles:
                    stop = 'cycle_budget_after_rejected_trial'
                    trace.append(dict(rejected=True, rho=rho, error=error))
                    break
                candidate = classical_cycle(self.classical, x, b, mg, stats)
                norm = stable_norm(b-self.a@candidate)
                executed += 1
                self._lifetime_cycles += 1
                stats.trial_cycles += 1
                stats.classical_recovery_cycles += 1
                path = 'classical_recovery'
            if path == 'hybrid':
                stats.accepted_neural_cycles += 1
                stats.branch_H_P_cycles += 1
            else:
                stats.classical_cycles += 1
                stats.branch_C_cycles += 1
                stats.classical_lock_cycles += 1
            x = candidate
            history.append(norm)
            relative.append(norm/denominator)
            paths.append(path)
            times.append(perf_counter()-cycle_started)
            if self.config.record_trace:
                trace.append(dict(path=path, rejected=rejected, residual=norm, rho=rho,
                    coefficients=self.coefficients.tolist(), error=error))
            if not np.isfinite(norm):
                stop = 'nonfinite_residual'
                break
            if norm > mg.divergence_factor*max(norm0, threshold):
                stop = 'diverged'
                break
        converged = bool(np.isfinite(history[-1]) and history[-1] <= threshold)
        if converged:
            stop = 'converged'
        values = stats.to_dict()
        adaptation = {key: self.online_totals[key]-online_before[key] for key in self.online_totals}
        values.update(adaptation)
        values.update(solver_executed_cycles=executed, accepted_cycles=len(paths),
            fallback_cycles=stats.classical_cycles, rollback_count=stats.rollback_count,
            coefficient_distance_from_ones=float(np.linalg.norm(self.coefficients-1)),
            coefficient_distance_from_prediction=float(np.linalg.norm(self.coefficients-self.base_coefficients)),
            factor_rebuilds=self._factor_count()-self.factor_snapshot, hierarchy_rebuilds=0,
            mlp_inference_calls=0, coefficient_generations=0, nn_forward_calls=0,
            online_events=self.online_events[event_start:],
            cached_coefficients=self.coefficients.tolist(), raw_no_fallback=not safe)
        values.update(setup_fallback=self.setup_failure is not None, setup_failure=self.setup_failure)
        all_executed = executed+adaptation['online_probe_cycles']+adaptation['online_gradient_vcycles']
        values['all_executed_cycles'] = all_executed
        setup = self.initial_setup_seconds if self.setup_pending else 0.
        setupstats = dict(self.setup_stats.to_dict(), **self.setup_details) if self.setup_pending else {}
        self.setup_pending = False
        return AdaptiveResult(x, converged, stop, history, relative, times, paths, trace,
            setup, perf_counter()-started, values, setupstats, threshold, executed,
            self.cache_key, requested_branch='H_P',
            actually_executed_branches=([name for name, used in (('H_P', stats.attempted_neural_cycles),
                                        ('C', stats.classical_cycles)) if used]),
            p_bank='classical' if self.setup_failure is not None else 'learned',
            branch_policy_status='fixed_light_gains_with_safety' if safe else 'raw_no_fallback')
