"""Bounded, per-operator head adaptation of a deployed transfer CNN.

Only A enters adaptation: synthetic probes are independent of evaluation RHS
and solutions.  Every probe comparison uses raw learned V-cycles, never the
runtime fallback.  Deployment retains the ordinary solver's classical safety.
All adaptation is online setup work, including rejected proposals.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np
import torch

from ..grid import as_shape
from ..provenance import operator_digest
from .autograd_sparse import SparseTensor
from .banks import Stats
from .fixed_classical import FixedClassicalBaseline
from .research_training import transfer_feasibility
from .solver import PreparedAdaptiveMG
from .unroll import TLevel, cycle, make_graph, smooth


DEFAULT_TTT_SETTINGS = dict(max_updates=2, learning_rate=1e-4,
    train_probes=2, guard_probes=2, raw_cycles=1, seed=7301,
    gradient_clip=1., proximal_weight=1e-3, smooth_probe_steps=0)


def validate_ttt_settings(settings=None):
    """Return a canonical bounded budget; reject misspelled/unknown controls."""
    if settings is not None and not isinstance(settings, dict):
        raise ValueError('TTT settings must be a dictionary')
    supplied = {} if settings is None else settings
    unknown = set(supplied) - set(DEFAULT_TTT_SETTINGS)
    if unknown:
        raise ValueError(f'Unknown TTT settings: {sorted(unknown)}')
    result = {**DEFAULT_TTT_SETTINGS, **supplied}
    bounds = dict(max_updates=(0, 16), train_probes=(1, 16), guard_probes=(1, 16),
                  raw_cycles=(1, 4), seed=(0, 2**63-1), smooth_probe_steps=(0, 4))
    for key, (lower, upper) in bounds.items():
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f'TTT {key} must be an integer in [{lower}, {upper}]')
    for key, lower, upper in (('learning_rate', 0., .01),
                              ('gradient_clip', 0., 100.),
                              ('proximal_weight', 0., 100.)):
        value = result[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not np.isfinite(value) or value > upper or
                (value < lower if key == 'proximal_weight' else value <= lower)):
            raise ValueError(f'TTT {key} is outside its finite permitted range')
        result[key] = float(value)
    return result


def _reference_graph(level):
    """Wrap an existing EM parent; no EM calculation or dense inverse."""
    result = TLevel(SparseTensor.from_scipy(level.a), level.shape, level.index,
                    level.strategy, raw_scipy=level.a)
    if level.coarse is not None:
        result.pattern = level.pattern
        result.base_weights = level.base_weights
        result.p = SparseTensor.from_scipy(level.p)
        result.interpolation_weights = torch.tensor(level.base_weights, dtype=torch.float64)
        result.coarse = _reference_graph(level.coarse)
    return result


def _probe_residuals(reference, cfg, settings):
    """Separate deterministic train/guard probes from current A only."""
    count = settings['train_probes'] + settings['guard_probes']
    generator = torch.Generator(device='cpu').manual_seed(settings['seed'])
    with torch.no_grad():
        errors = torch.randn(reference.a.shape[0], count, generator=generator,
                             dtype=torch.float64)
        for step in range(settings['smooth_probe_steps']):
            errors -= smooth(reference, reference.a.apply(errors), cfg,
                             reverse=bool(step % 2))
        residuals = reference.a.apply(errors)
        norms = torch.linalg.vector_norm(residuals, dim=0)
        if not torch.isfinite(norms).all() or bool((norms <= 1e-100).any()):
            raise FloatingPointError('Degenerate TTT synthetic probes')
        residuals = (residuals / norms).detach()
    split = settings['train_probes']
    return residuals[:, :split], residuals[:, split:]


def _raw_ratios(graph, model, cfg, residuals, raw_cycles):
    x = torch.zeros_like(residuals)
    for step in range(raw_cycles):
        x = cycle(graph, x, residuals, model, cfg, step)
    return (torch.linalg.vector_norm(residuals - graph.a.apply(x), dim=0) /
            torch.linalg.vector_norm(residuals, dim=0).clamp_min(1e-100))


def _guard_accepts(before, after):
    """No probe may worsen beyond roundoff; require aggregate improvement."""
    before, after = np.asarray(before, float), np.asarray(after, float)
    if (before.shape != after.shape or not np.isfinite(before).all() or
            not np.isfinite(after).all() or (before < 0).any() or (after < 0).any()):
        return False
    numerical_slack = 1e-12 + 1e-10 * before
    return bool(np.all(after <= before + numerical_slack) and
                np.log(np.maximum(after, 1e-100)).mean() <
                np.log(np.maximum(before, 1e-100)).mean())


class PreparedTTTMG:
    """Adapt once for each A, then reuse its learned hierarchy across RHS.

    This wrapper is deliberately not thread safe, like PreparedAdaptiveMG.
    Each instance owns its starting checkpoint and adapted clone.  Replacing
    or mutating A resets to that immutable starting checkpoint before adapting.
    """
    def __init__(self, a, n, components, config, rules, settings=None):
        initialization_started = perf_counter()
        self.settings = validate_ttt_settings(settings)
        if not isinstance(rules, FixedClassicalBaseline):
            raise ValueError('TTT requires the fixed line_alt + EM5 + full baseline')
        rules.validate_config(config)
        if (config.branch != 'H_P' or config.use_smoother or not config.use_transfer or
                config.mode != 'research' or config.spatial or config.gate_mode != 'open'):
            raise ValueError('TTT requires forced research H_P with classical smoothing')
        if (components is None or getattr(components.transfer, 'training_only', False) or
                getattr(components.transfer, 'reference', None) != 'frozen_parent' or
                not isinstance(getattr(components.transfer, 'head', None), torch.nn.Conv2d)):
            raise ValueError('TTT requires a deployable frozen-parent CNN with a convolutional head')
        self.rules = rules
        self._config = config
        self.adaptation_runs = 0
        self._starting_components = deepcopy(components).eval()
        for module in self._starting_components.modules():
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        snapshot_seconds = perf_counter() - initialization_started
        self._reset(a, n, snapshot_seconds=snapshot_seconds)

    def __getattr__(self, name):
        prepared = self.__dict__.get('_prepared')
        if prepared is None:
            raise AttributeError(name)
        return getattr(prepared, name)

    def _reset(self, a, n, *, snapshot_seconds=0.):
        reset_started = perf_counter()
        model = deepcopy(self._starting_components).eval()
        classical_started = perf_counter()
        self._prepared = PreparedAdaptiveMG(a, n, model, self._config)
        classical_seconds = perf_counter() - classical_started
        self._operator_digest = operator_digest(a)
        self.adaptation_runs += 1
        self.ttt_bank_stats = Stats().to_dict()
        self.ttt_report = dict(schema=1, settings=dict(self.settings),
            operator_digest=self._operator_digest, baseline_digest=self.rules.digest(),
            data_scope='current A only; deterministic synthetic residual probes',
            trainable_scope='transfer.head.weight and transfer.head.bias',
            guard_scope='independent probes; raw learned V-cycles; fallback forbidden',
            proposal_adopted=False, updates_applied=0, updates_retained=0,
            raw_ratios_before=None, raw_ratios_after=None, failures=[], trace=[],
            runtime_bank_setup_seconds=0., adaptation_runs=self.adaptation_runs)
        try:
            if self.settings['max_updates'] == 0:
                self.ttt_report['decision'] = 'disabled_zero_update_budget'
            else:
                self._adapt()
        finally:
            self.ttt_seconds = snapshot_seconds + perf_counter() - reset_started - classical_seconds
            self.adaptation_seconds = self.ttt_seconds
            self.ttt_report['ttt_seconds'] = self.ttt_seconds
            self.ttt_report['timing_scope'] = ('online setup: checkpoint copies, probes, graph construction, '
                'forward/backward, optimization, guard and selected learned hierarchy; '
                'ordinary classical construction counted separately')
            self._ttt_setup_pending = True

    def _adapt(self):
        prepared, settings = self._prepared, self.settings
        model, cfg = prepared.components, prepared.config
        head = model.transfer.head
        initial_head = {key: value.detach().clone() for key, value in head.state_dict().items()}
        for parameter in head.parameters():
            parameter.requires_grad_(True)
        accepted = False
        try:
            reference = _reference_graph(prepared.classical)
            train, guard = _probe_residuals(reference, cfg, settings)
            with torch.no_grad():
                initial_graph = make_graph(prepared.a, prepared.shape, model, cfg,
                                           reference_root=reference)
                feasible, _ = transfer_feasibility(initial_graph, model, cfg, reference)
                if not feasible['feasible']:
                    raise ValueError('Starting pruned transfer is infeasible on this A')
                before = _raw_ratios(initial_graph, model, cfg, guard,
                                     settings['raw_cycles']).cpu().numpy()
                if not np.isfinite(before).all():
                    raise FloatingPointError('Nonfinite initial TTT guard contraction')
            self.ttt_report['raw_ratios_before'] = before.tolist()
            del initial_graph
            optimizer = torch.optim.Adam(head.parameters(), lr=settings['learning_rate'])
            for step in range(settings['max_updates']):
                optimizer.zero_grad(set_to_none=True)
                graph = make_graph(prepared.a, prepared.shape, model, cfg,
                                   reference_root=reference)
                feasible, _ = transfer_feasibility(graph, model, cfg, reference)
                if not feasible['feasible']:
                    raise ValueError('TTT update produced an infeasible hierarchy')
                ratios = _raw_ratios(graph, model, cfg, train, settings['raw_cycles'])
                proximal = sum((parameter - initial_head[name].to(parameter)).square().sum()
                               for name, parameter in head.named_parameters())
                proximal = proximal / sum(p.numel() for p in head.parameters())
                objective = ratios.clamp_min(1e-100).log().mean()
                loss = objective + settings['proximal_weight'] * proximal
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite TTT objective')
                loss.backward()
                grads = [p.grad for p in head.parameters() if p.grad is not None]
                if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                    raise FloatingPointError('Missing/nonfinite TTT head gradients')
                grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), settings['gradient_clip'])
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError('Nonfinite TTT gradient norm')
                optimizer.step()
                self.ttt_report['updates_applied'] += 1
                self.ttt_report['trace'].append(dict(step=step + 1,
                    loss=float(loss.detach()), gradient_norm=float(grad_norm),
                    train_raw_ratios=ratios.detach().cpu().tolist()))
                del graph, ratios, loss, objective, proximal
            with torch.no_grad():
                graph = make_graph(prepared.a, prepared.shape, model, cfg,
                                   reference_root=reference)
                feasible, _ = transfer_feasibility(graph, model, cfg, reference)
                self.ttt_report['candidate_feasible'] = feasible['feasible']
                if not feasible['feasible']:
                    raise ValueError('Final TTT hierarchy is infeasible')
                after = _raw_ratios(graph, model, cfg, guard, settings['raw_cycles']).cpu().numpy()
            self.ttt_report['raw_ratios_after'] = after.tolist()
            if not np.isfinite(after).all():
                raise FloatingPointError('Nonfinite candidate TTT guard contraction')
            accepted = _guard_accepts(before, after)
            self.ttt_report['decision'] = 'guard_accepted' if accepted else 'guard_rejected'
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            self.ttt_report['decision'] = 'adaptation_failed_keep_original'
            self.ttt_report['failures'].append(f'{type(error).__name__}: {error}')
        if not accepted:
            head.load_state_dict(initial_head)
        for parameter in head.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        model.eval()
        # Refresh the model-keyed bank while retaining the already-built EM
        # parent.  Runtime generation performs its exact sparse feasibility
        # checks; a numerical discrepancy cannot sneak through the graph guard.
        stats = Stats()
        try:
            prepared._ensure_fresh()
            prepared.ensure_branch('H_P', stats)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            self.ttt_report['failures'].append(f'Runtime hierarchy: {type(error).__name__}: {error}')
            if accepted:
                accepted = False
                self.ttt_report['decision'] = 'runtime_hierarchy_rejected_keep_original'
                head.load_state_dict(initial_head)
                prepared._ensure_fresh()
            # A failed original bank is left to the existing solver's cached
            # setup-failure -> classical fallback path; no false NN success.
            try:
                prepared.ensure_branch('H_P', stats)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as original_error:
                self.ttt_report['failures'].append(f'Original hierarchy: {type(original_error).__name__}: {original_error}')
        self.ttt_bank_stats = stats.to_dict()
        self.ttt_report['runtime_bank_setup_seconds'] = stats.branch_setup_seconds
        self.ttt_report['proposal_adopted'] = accepted
        self.ttt_report['updates_retained'] = self.ttt_report['updates_applied'] if accepted else 0

    def replace_operator(self, a, n=None):
        """Reset from the checkpoint for a new A, never from the last adaptation."""
        self._reset(a, self._prepared.shape if n is None else as_shape(n))

    def _ensure_current_operator(self):
        if operator_digest(self._prepared.source_a) != self._operator_digest:
            self.replace_operator(self._prepared.source_a)

    def _account(self, results):
        if results and self._ttt_setup_pending:
            results[0].setup_seconds += self.ttt_seconds
            results[0].setup_stats['ttt_seconds'] = self.ttt_seconds
            self._ttt_setup_pending = False
        for result in results:
            result.stats['ttt_proposal_adopted'] = int(self.ttt_report['proposal_adopted'])
        return results

    def solve(self, b, x0=None):
        self._ensure_current_operator()
        return self._account([self._prepared.solve(b, x0)])[0]

    def solve_many(self, bs, x0=None):
        self._ensure_current_operator()
        return self._account(self._prepared.solve_many(bs, x0))
