"""Development-only warm checkpoint selection, separate from residual gradients."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..provenance import operator_digest
from .bamg_solver import build_runtime, mask_digest
from .bamg_training import _atomic_save, _digest


@dataclass(frozen=True)
class WarmValidationSettings:
    every: int = 100
    per_family_per_size: int = 1
    repetitions: int = 3
    warmup_cycles: int = 1
    timeout_seconds: float = 120.0
    failure_penalty_seconds: float = 120.0
    allowed_failure_rate: float = 0.0


def validation_settings(values):
    config = WarmValidationSettings(**values)
    for key in ('every', 'per_family_per_size', 'repetitions'):
        if type(getattr(config, key)) is not int or getattr(config, key) < 1:
            raise ValueError(f'warm_validation.{key} must be a positive integer')
    if type(config.warmup_cycles) is not int or config.warmup_cycles < 0:
        raise ValueError('warm_validation.warmup_cycles must be a nonnegative integer')
    for key in ('timeout_seconds', 'failure_penalty_seconds'):
        if not np.isfinite(getattr(config, key)) or getattr(config, key) <= 0:
            raise ValueError(f'warm_validation.{key} must be finite and positive')
    if not np.isfinite(config.allowed_failure_rate) or not 0 <= config.allowed_failure_rate <= 1:
        raise ValueError('invalid warm_validation.allowed_failure_rate')
    return config


def development_subset(validation, train, count):
    """Fixed cases stratified by family AND grid; reject any TRAIN overlap."""
    if not validation:
        raise ValueError('warm selection requires development validation examples')
    train_ids = {e['case_id'] for e in train}
    train_ops = {operator_digest(e['A']) for e in train}
    train_groups = {e['normalized_operator_digest'] for e in train
                    if e.get('normalized_operator_digest')}
    groups, seen = {}, set()
    for example in validation:
        for key in ('split', 'research_split'):
            if example.get(key, 'validation') != 'validation':
                raise ValueError('warm selection requires validation, never TRAIN/final/OOD')
        if (example['case_id'] in train_ids or operator_digest(example['A']) in train_ops or
                example.get('normalized_operator_digest') in train_groups):
            raise ValueError('TRAIN/validation operator overlap in warm selection')
        if example['case_id'] in seen:
            raise ValueError('duplicate development validation case')
        seen.add(example['case_id'])
        groups.setdefault((example['family'], tuple(example['shape'])), []).append(example)
    return [example for key in sorted(groups)
            for example in sorted(groups[key], key=lambda e: e['case_id'])[:count]]


class WarmCheckpointSelector:
    """Keep a small best-weight snapshot inside latest for crash-consistent resume.

    ``best_path`` is a derived deployment snapshot, never the source of resume
    progress. Re-exporting it from latest repairs an interrupted two-file save.
    No full iterates, hierarchy matrices or residual trajectories are persisted.
    """

    VERSION = 'bamg-warm-selection-v1'

    def __init__(self, validation, train, reference, mg_config, *, settings, best_path):
        self.settings = validation_settings(settings)
        self.examples = development_subset(validation, train, self.settings.per_family_per_size)
        self.reference = reference
        self.mg_config = mg_config
        self.best_path = Path(best_path)
        self.evaluations = []
        self.best = None
        cases = []
        for example in self.examples:
            b = np.asarray(example['b'])
            if b.shape != (example['A'].shape[0],) or not np.isfinite(b).all():
                raise ValueError('warm validation needs a finite vector RHS')
            masks = reference.masks_for(example['A'], example['shape'])
            cases.append(dict(case_id=example['case_id'], family=example['family'],
                shape=list(example['shape']), operator=operator_digest(example['A']),
                rhs_hash=_digest(b), mask_digest=mask_digest(masks)))
        self.contract = dict(version=self.VERSION, settings=asdict(self.settings), cases=cases,
            mg_config=asdict(mg_config), reference_hash=_digest(reference.state_dict()),
            split='validation', x0='zero', workload='raw warm RHS=1',
            ranking='failure count, then mean of case median penalized warm seconds',
            setup_included=False, fallback=False, gradients=False)

    def _measure(self, weight):
        rows = []
        failures = 0
        medians = []
        cfg = self.settings
        for example, identity in zip(self.examples, self.contract['cases']):
            samples = []
            try:
                # Rebuild current P, actual Galerkin matrices and factors before timing.
                bank = build_runtime(example['A'], example['shape'], weight,
                                     self.reference, self.mg_config)
            except (ValueError, RuntimeError, FloatingPointError, MemoryError,
                    np.linalg.LinAlgError) as exc:
                samples = [dict(converged=False, reason='setup_error:' + type(exc).__name__,
                                warm_seconds=None, cycles=0)] * cfg.repetitions
            else:
                if bank.mask_digest != identity['mask_digest']:
                    raise RuntimeError('development validation changed frozen masks')
                try:
                    for _ in range(cfg.warmup_cycles):
                        bank.cycle(np.zeros_like(example['b']), example['b'])
                    for _ in range(cfg.repetitions):
                        result = bank.solve(example['b'], x0=np.zeros_like(example['b']),
                                            timeout_seconds=cfg.timeout_seconds)
                        seconds = float(result['warm_seconds'])
                        valid_time = np.isfinite(seconds) and seconds >= 0
                        success = (bool(result['converged']) and valid_time and
                                   seconds <= cfg.timeout_seconds)
                        samples.append(dict(converged=success,
                            reason=result['reason'] if success or not result['converged'] else 'timeout_or_invalid_time',
                            warm_seconds=seconds if valid_time else None,
                            cycles=int(result['cycles'])))
                except (ValueError, RuntimeError, FloatingPointError, MemoryError,
                        np.linalg.LinAlgError) as exc:
                    # Preserve already measured repeats and count every missing repeat.
                    samples.extend([dict(converged=False, reason='solve_error:' + type(exc).__name__,
                        warm_seconds=None, cycles=0)] * (cfg.repetitions - len(samples)))
            failures += sum(not s['converged'] for s in samples)
            costs = [s['warm_seconds'] if s['converged'] else
                     max(cfg.failure_penalty_seconds, s['warm_seconds'] or 0.) for s in samples]
            median = float(np.median(costs))
            medians.append(median)
            rows.append(dict(case_id=example['case_id'], family=example['family'],
                shape=list(example['shape']), samples=samples, median_penalized_warm_seconds=median))
        attempts = len(self.examples) * cfg.repetitions
        return dict(failures=failures, attempts=attempts, failure_rate=failures / attempts,
            eligible=failures / attempts <= cfg.allowed_failure_rate,
            mean_case_median_warm_seconds=float(np.mean(medians)), cases=rows)

    @staticmethod
    def _rank(metrics):
        return metrics['failures'], metrics['mean_case_median_warm_seconds']

    def is_due(self, update, total_updates):
        if self.evaluations and self.evaluations[-1]['update'] == update:
            return False
        return update in (0, total_updates) or update % self.settings.every == 0

    def maybe_evaluate(self, update, total_updates, weight):
        if not self.is_due(update, total_updates):
            return None
        training = weight.training
        try:
            weight.eval()
            with torch.no_grad():
                metrics = self._measure(weight)
        finally:
            weight.train(training)
        improved = self.best is None or self._rank(metrics) < self._rank(self.best['metrics'])
        if improved:
            self.best = dict(update=update, metrics=deepcopy(metrics), weight_spec=deepcopy(weight.spec),
                weight_state={key: value.detach().cpu().clone() for key, value in weight.state_dict().items()})
        record = dict(update=update, improved=improved, **metrics)
        self.evaluations.append(record)
        print(f'Warm validation update={update}/{total_updates} failures={metrics["failures"]}/'
              f'{metrics["attempts"]} mean_case_median_ms='
              f'{1000 * metrics["mean_case_median_warm_seconds"]:.3f} '
              f'best_update={self.best["update"]}', flush=True)
        return record

    def state_dict(self):
        return dict(contract_hash=_digest(self.contract), evaluations=self.evaluations, best=self.best)

    def load_state_dict(self, state, update):
        if state.get('contract_hash') != _digest(self.contract):
            raise ValueError('warm validation provenance/configuration mismatch')
        evaluations = state['evaluations']
        steps = [record['update'] for record in evaluations]
        if not steps or steps[0] != 0 or steps != sorted(set(steps)) or steps[-1] > update:
            raise ValueError('inconsistent warm validation progress')
        winner = min(evaluations, key=self._rank)
        if state['best']['update'] != winner['update'] or self._rank(state['best']['metrics']) != self._rank(winner):
            raise ValueError('inconsistent best warm snapshot')
        self.evaluations = deepcopy(evaluations)
        self.best = deepcopy(state['best'])

    def export_best(self):
        if self.best is not None:
            _atomic_save(dict(version=self.VERSION, selection_contract=self.contract, **self.best),
                         self.best_path)

    def report(self):
        return dict(contract=self.contract, evaluations=self.evaluations,
                    selected_update=self.best['update'], selected_metrics=self.best['metrics'])
