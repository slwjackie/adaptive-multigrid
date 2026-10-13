"""Causal policies: decisions never receive future matrices or forcing schedules."""
from __future__ import annotations

from copy import copy
from dataclasses import replace
from time import perf_counter

import numpy as np

from ...provenance import stable_norm
from ..h2_fixed_p.backend import FixedPSolver

FEATURES = ("log_cells", "log_A_norm", "diagonal_cv", "previous_normalized_drift",
            "cached_normalized_drift", "cached_log_scale_ratio", "log_initial_residual",
            "log_smoother_age", "log_previous_cycles", "previous_contraction",
            "log_previous_setup", "log_previous_solve", "previous_fallback", "cache_ready")


def normalized_drift(a, b):
    if b is None:
        return 0.
    if a.shape != b.shape:
        return 1.
    an = max(float(np.linalg.norm(a.data)), 1e-300)
    bn = max(float(np.linalg.norm(b.data)), 1e-300)
    return float(np.linalg.norm((a/an - b/bn).data))


def observation(s, solver):
    core, previous, last = solver.core, solver.previous, solver.last
    reference = core.smoother_reference
    norm = max(float(np.linalg.norm(s.a.data)), 1e-300)
    diagonal = s.a.diagonal()
    refnorm = max(float(np.linalg.norm(reference.a.data)), 1e-300) if reference is not None else norm
    contraction = (last.residuals[-1]/max(last.residuals[0], 1e-300))**(1/max(last.cycles, 1)) if last else 0.
    x = np.asarray([
        np.log(s.a.shape[0]), np.log(norm),
        float(np.std(diagonal)/max(abs(np.mean(diagonal)), 1e-300)),
        normalized_drift(s.a, previous.a if previous is not None else None),
        normalized_drift(s.a, reference.a if reference is not None else None),
        np.log(norm/refnorm), np.log(max(stable_norm(s.b-s.a@s.x0), 1e-30)),
        np.log1p(core.smoother_age), np.log1p(last.cycles if last else 0), contraction,
        np.log(max(last.setup_seconds, 1e-8)) if last else -18.,
        np.log(max(last.solve_seconds, 1e-8)) if last else -18.,
        float(last.fallback) if last else 0., float(core.cached_smoother is not None),
    ], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite temporal observation")
    return x


def check_policy(name):
    if name in ("refresh", "reuse", "world", "world_horizon1", "world_stateless", "memoryless"):
        return name
    parts = name.split("_")
    if len(parts) != 2:
        raise ValueError("unknown smoother policy: "+name)
    try:
        if parts[0] == "periodic" and int(parts[1]) > 0:
            return name
        if parts[0] in ("drift", "residual") and np.isfinite(float(parts[1])) and float(parts[1]) > 0:
            return name
    except ValueError:
        pass
    raise ValueError("invalid smoother policy: "+name)


def heuristic(name, solver, s):
    if solver.core.cached_smoother is None or name == "refresh":
        return "refresh"
    if name == "reuse":
        return "reuse"
    kind, value = name.split("_")
    if kind == "periodic":
        return "refresh" if solver.core.smoother_age >= int(value) else "reuse"
    ref = solver.core.smoother_reference.a
    # Cheap rules see drift since *generation*, not just since the previous call.
    delta = normalized_drift(s.a, ref)
    scale = abs(np.log(max(np.linalg.norm(s.a.data), 1e-300)/max(np.linalg.norm(ref.data), 1e-300)))
    refresh = max(delta, scale) > float(value)
    if kind == "residual" and solver.last is not None:
        last = solver.last
        rho = (last.residuals[-1]/max(last.residuals[0], 1e-300))**(1/max(last.cycles, 1))
        refresh |= last.fallback or rho > .8
    return "refresh" if refresh else "reuse"


class ReuseSolver:
    """Public step/reset/p_digest interface shared by replay and live CFD."""
    def __init__(self, cfg, expert, policy="refresh", *, predictor=None, reference="reuse"):
        self.policy = check_policy(policy)
        self.reference = check_policy(reference)
        self.core = FixedPSolver(cfg, expert)
        self.predictor = predictor
        if policy in ("world", "world_horizon1", "world_stateless", "memoryless") and predictor is None:
            raise ValueError("a trained, provenance-checked temporal predictor is required")
        self.reset()

    @property
    def p_digest(self):
        return self.core.p_digest

    def reset(self):
        self.core.reset()
        self.previous = self.last = None
        if self.predictor is not None:
            self.predictor.reset()

    def fork(self):
        """Counterfactual branches share immutable banks, never solver state."""
        if self.predictor is not None:
            raise ValueError("fork is only for offline fixed-action measurement")
        result = copy(self)
        result.core = copy(self.core)
        return result

    def step(self, s, *, action=None):
        start = perf_counter()
        if self.previous is not None and (s.index <= self.previous.index or s.time < self.previous.time):
            raise ValueError("nonmonotone temporal stream; reset between physical cases")
        decision = {}
        if action is not None:
            if action not in ("refresh", "reuse"):
                raise ValueError("invalid fixed smoother action")
        elif self.predictor is None:
            action = heuristic(self.policy, self, s)
        else:
            raw = observation(s, self)
            reference = heuristic(self.reference, self, s)
            action, decision = self.predictor.choose(raw, reference,
                horizon=1 if self.policy in ("world_horizon1", "memoryless") else 2,
                stateless=self.policy == "world_stateless")
        if self.core.cached_smoother is None:
            action = "refresh"
        policy_seconds = perf_counter()-start
        result = self.core.step(s, smoother_action=action)
        if self.predictor is not None:
            self.predictor.accept(action)
        self.previous = s
        total = perf_counter()-start
        stats = dict(result.stats, temporal_policy=self.policy, policy_seconds=policy_seconds,
                     policy_decision=decision, smoother_action_selected=action,
                     backend_total_seconds=result.total_seconds,
                     temporal_overhead_seconds=max(0., total-result.total_seconds),
                     observation_uses_future=False)
        result = replace(result, total_seconds=total, stats=stats)
        self.last = result
        return result
