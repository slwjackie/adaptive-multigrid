"""Standalone fixed-interpolation C/H_S comparison on admitted SPD snapshots.

The first system of EACH trajectory builds the classical interpolation once.
All later systems update Galerkin operators and their numerical factors, while
retaining every P. A rejected neural trial rolls back to the last accepted x
and uses the *same current-A classical bank*, never a rebuilt interpolation.

Use ``snapshot``/``load_snapshot`` for offline SPD admission before timing.
``step`` repeats the online finite/symmetry/layout/order checks and hashes inside
its timer; it does not claim that these cheaper checks prove positive definiteness.
Instances own their banks and are not thread-safe. Call reset only between
independent trajectories. No World Model is involved in this first experiment.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import torch

from ...hierarchy import classical_cycle
from ...provenance import stable_norm
from ..banks import Stats, hybrid_cycle
from ..config import AdaptiveConfig
from ..hs_world.backend import HSSmoothingBackend, SmoothingBank
from ..models import Components
from ..spatial import SpatialState
from ..strong import StrongRules
from ..world_model.backend import StepResult, hierarchy_signature, levels
from ..world_model.data import Snapshot


def hierarchy_p_digest(root):
    """Content hash of ALL interpolation matrices, in level order."""
    return hierarchy_signature(root)


def fixed_rules(plan):
    rules = StrongRules()
    return replace(rules, strategy_by_rule=tuple((key, plan) for key in rules.rule_ids),
                   fallback_strategy_name=plan, require_coverage=False,
                   coverage_by_rule=(), provenance='fixed-P trajectory: single declared plan')


class FixedPSolver:
    """One declared classical plan, optional genuinely trained H_S, fixed P.

    ``step`` returns the existing ``StepResult`` API. Timing phases and initial/
    final P hashes are in ``result.stats``. ``bank``, ``classical_root`` and
    ``smoother_root`` expose the actual numerical hierarchy for inspection.
    A cycle-budget failure retains the bank; it cannot silently reset P on the
    next system. Incompatible trajectories raise and require an explicit reset.
    """

    def __init__(self, cfg: AdaptiveConfig, expert: Components | None = None):
        if cfg.use_transfer:
            raise ValueError('fixed-P study forbids learned transfer; set use_transfer=False')
        if expert is not None:
            if not isinstance(expert, Components):
                raise TypeError('expert must be trained Components')
            updates = expert.metadata.get('optimizer_updates', 0)
            if (expert.metadata.get('training_branch') != 'H_S'
                    or isinstance(updates, bool) or not isinstance(updates, int) or updates < 1):
                raise ValueError('genuinely trained H_S checkpoint required')
            if not all(torch.isfinite(v).all().item() for v in expert.smoother.state_dict().values()):
                raise ValueError('nonfinite H_S checkpoint')
            if (not cfg.use_smoother or not (cfg.replace_pre or cfg.replace_post)
                    or cfg.mg.smoother_gain_multiplier <= 0 or cfg.mg.nn_levels == 0
                    or cfg.smoother_levels == ()):
                raise ValueError('H_S experiment must enable actual neural smoothing')
        self.cfg = replace(cfg, mode='research' if expert is not None else 'classical',
                           branch='H_S' if expert is not None else 'C', use_transfer=False,
                           use_smoother=expert is not None, spatial=False, gate_mode='open',
                           use_learned_controller=False)
        self.backend = HSSmoothingBackend(self.cfg, fixed_rules(cfg.mg.strategy_name), expert)
        self.reset()

    @property
    def classical_root(self):
        return self.bank.classical_root if self.bank is not None else None

    @property
    def smoother_root(self):
        return self.bank.smoother_root if self.bank is not None else None

    @property
    def p_digest(self):
        return hierarchy_p_digest(self.bank.root) if self.bank is not None else None

    def reset(self):
        self.bank: SmoothingBank | None = None
        self.initial_p_digest = None
        self._last_index = None
        self._last_time = None
        self._layout = None

    def step(self, snapshot: Snapshot) -> StepResult:
        started = perf_counter()
        phase = dict(feature_seconds=0., setup_seconds=0., solve_seconds=0.,
                     verification_seconds=0., recovery_seconds=0.)
        before = perf_counter()
        if not isinstance(snapshot, Snapshot):
            raise TypeError('step requires an admitted Snapshot')
        # A fresh dataclass discards cached hashes from callers who changed a
        # mutable CSR after constructing Snapshot directly. Never trust stale A.
        s = replace(snapshot)
        if not sp.isspmatrix_csr(s.a) or not s.a.has_canonical_format:
            raise ValueError('canonical CSR snapshot required')
        s.validate(spd_check=False)
        layout = s.context.get('layout', 'structured_2d_xmajor')
        if layout != 'structured_2d_xmajor':
            raise ValueError('only structured_2d_xmajor layout is supported')
        if self._last_index is not None and (s.index <= self._last_index or s.time < self._last_time):
            raise ValueError('snapshot order must have increasing index and nondecreasing time')
        selection, cfg = self.backend.select(s)
        if self.bank is not None and (layout != self._layout or not self.backend.compatible(
                s, self.bank, selection.strategy_name)):
            raise ValueError('mesh/layout/boundary/topology changed; explicit trajectory reset required')
        # Charge fresh hashes to the online solver, including on the first step.
        _ = s.matrix_digest, s.topology_key
        phase['feature_seconds'] += perf_counter() - before

        requested = 'H_S' if self.backend.expert is not None else 'C'
        fallback = False
        reason = ''
        builds = []
        before = perf_counter()
        initial = self.bank is None
        base, st = self.backend.build(s, self.bank, 'REBUILD_C' if initial else 'REUSE_C', selection, cfg)
        builds.append(st)
        self.bank = base  # Preserve P even if a subsequent neural setup fails.
        if requested == 'H_S':
            try:
                bank, st = self.backend.build(s, base, 'REUSE_HS', selection, cfg)
                # Overlay creation is not an additional physical timestep.
                self.bank = replace(bank, p_age=base.p_age)
                builds.append(st)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                fallback = True
                reason = 'smoother_setup_rejection:' + str(exc)
                builds.append(dict(smoother_setup_rejected=True, error=str(exc)))
        phase['setup_seconds'] += perf_counter() - before

        before = perf_counter()
        p_digest = hierarchy_p_digest(self.classical_root)
        if self.initial_p_digest is None:
            self.initial_p_digest = p_digest
        if p_digest != self.initial_p_digest or hierarchy_p_digest(self.bank.root) != p_digest:
            raise RuntimeError('fixed interpolation invariant violated')
        x = np.array(s.x0, dtype=np.float64, copy=True)
        initial_residual = stable_norm(s.b - s.a @ x)
        reference = initial_residual if cfg.mg.residual_reference == 'initial' else stable_norm(s.b)
        threshold = max(cfg.mg.absolute_tolerance, cfg.mg.tolerance * reference)
        history = [initial_residual]
        phase['verification_seconds'] += perf_counter() - before

        stats = Stats()
        attempts = 0
        stagnant = 0
        neural_trials = 0
        accepted_neural = 0
        classical_trials = 0
        rejected_neural = 0
        neural = not fallback and any(l.neural_stencil is not None for l in levels(self.bank.root))
        if requested == 'H_S' and not neural and not fallback:
            fallback = True
            reason = 'no_nonterminal_neural_level'
        spatial = SpatialState(self.backend.expert, cfg) if neural else None
        trial_budget = max(0, cfg.mg.max_cycles - cfg.mg.reserve_classical_cycles)

        def recover(why):
            nonlocal neural, fallback, reason
            t0 = perf_counter()
            # Immutable bank replacement; Ac, factors, P and accepted x survive.
            self.bank = replace(self.bank, root=self.bank.classical_root)
            neural = False
            fallback = True
            reason = why
            phase['recovery_seconds'] += perf_counter() - t0

        while attempts < cfg.mg.max_cycles and history[-1] > threshold:
            if neural and (attempts >= trial_budget or stagnant >= cfg.mg.stagnation_patience):
                recover('stagnation_recovery' if stagnant >= cfg.mg.stagnation_patience else 'budget_recovery')
            was_neural = neural
            neural_trials += int(was_neural)
            classical_trials += int(not was_neural)
            stats.attempted_neural_cycles += int(was_neural)
            previous = history[-1]
            t0 = perf_counter()
            failed = ''
            try:
                with np.errstate(over='ignore', invalid='ignore'):
                    proposal = (hybrid_cycle(self.bank.root, x.copy(), s.b, cfg, stats, spatial, attempts + 1)
                                if was_neural else classical_cycle(self.classical_root, x.copy(), s.b, cfg.mg, stats))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                proposal = None
                failed = str(exc)
            phase['recovery_seconds' if fallback and not was_neural else 'solve_seconds'] += perf_counter() - t0
            attempts += 1
            t0 = perf_counter()
            valid = proposal is not None and np.shape(proposal) == s.b.shape
            norm = stable_norm(s.b - s.a @ proposal) if valid else float('inf')
            bad = not np.isfinite(norm) or norm > previous * cfg.mg.safety_growth * (1 + cfg.mg.safety_rtol_slack)
            phase['verification_seconds'] += perf_counter() - t0
            if was_neural and bad:
                rejected_neural += 1
                stats.rejected_neural_cycles += 1
                stats.rollback_count += 1
                recover('neural_cycle_error:' + failed if failed else 'residual_rejection')
                continue
            if not np.isfinite(norm):
                reason = 'classical_cycle_error:' + failed if failed else 'nonfinite_classical'
                break
            x = proposal
            history.append(float(norm))
            accepted_neural += int(was_neural)
            stats.accepted_neural_cycles += int(was_neural)
            stats.classical_cycles += int(not was_neural)
            stats.classical_recovery_cycles += int(fallback and not was_neural)
            stagnant = stagnant + 1 if norm / max(previous, 1e-300) >= cfg.mg.stagnation_rho else 0
            if norm > cfg.mg.divergence_factor * max(initial_residual, threshold):
                reason = 'divergence_limit'
                break

        before = perf_counter()
        final_residual = stable_norm(s.b - s.a @ x)
        final_digest = hierarchy_p_digest(self.bank.root)
        preserved = final_digest == self.initial_p_digest == hierarchy_p_digest(self.classical_root)
        if not preserved:
            raise RuntimeError('fixed interpolation mutated during solve')
        success = bool(np.isfinite(final_residual) and final_residual <= threshold)
        phase['verification_seconds'] += perf_counter() - before
        self._last_index, self._last_time, self._layout = s.index, s.time, layout
        actual = ('H_S_THEN_C' if accepted_neural and fallback else 'H_S' if accepted_neural else 'C')
        details = dict(stats.to_dict(), builds=builds, attempt_budget=cfg.mg.max_cycles,
                       neural_trial_cycles=neural_trials, accepted_neural_cycles=accepted_neural,
                       rejected_neural_cycles=rejected_neural, classical_trial_cycles=classical_trials,
                       actual_neural_used=bool(stats.neural_apply_calls),
                       classical_P_only=True, fixed_P=True, learned_transfer=False,
                       hierarchy_p_digest_initial=self.initial_p_digest,
                       hierarchy_p_digest_final=final_digest, hierarchy_p_preserved=preserved,
                       final_true_residual=float(final_residual), certified=False,
                       online_spd_check=False, offline_spd_admission_required=True, **phase)
        total = perf_counter() - started
        details['orchestration_seconds'] = max(0., total - sum(phase.values()))
        details['timing_scope'] = 'online validation + setup + cycles + true residuals + recovery; excludes offline SPD admission and IO'
        return StepResult(x, success, requested, actual, attempts, history, threshold,
                          phase['setup_seconds'], phase['solve_seconds'], total, self.bank,
                          fallback, reason or ('converged' if success else 'cycle_limit'), details)
