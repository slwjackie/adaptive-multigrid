"""Offline EM imitation and raw V-cycle learning of shared line operators.

EM targets belong to each *student-generated* level operator, not to a frozen
teacher hierarchy.  Target construction is detached; P -> Ac -> lower-level
student generation and the whole V-cycle retain their derivatives.  Deployment
never calls the target generator and never trains model weights online.
"""
from dataclasses import replace

import numpy as np
import torch
from torch.nn import functional as F

from ..energymin import energy_weights
from ..provenance import operator_digest
from ..strategy import get_strategy
from .joint_transfer_training import (
    InfeasibleCandidate, _binding, _check_graph, _levels, _probe_residuals,
    _train_only,
)
from .research_runtime import offline_teacher_generation
from .unroll import make_graph, cycle


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _validate(example, model, cfg, *, smoother):
    _train_only(example)
    _binding(example, model)
    strategy = get_strategy(cfg.mg.strategy_name)
    if (strategy.smoother != 'line_alt' or strategy.transfer != 'bilinear'
            or strategy.coarsening != 'full'):
        raise ValueError('EM-line learning requires line_alt + bilinear + full parent')
    if (not cfg.use_transfer or cfg.transfer_levels != 'all'
            or getattr(model.transfer, 'parameterization', None) != 'affine'
            or getattr(model.transfer, 'support', None) != 'support_preserving'
            or getattr(model.transfer, 'reference', 'actual') != 'actual'):
        raise ValueError('EM-line learning requires signed affine P on actual all-level operators')
    if cfg.application != 'replace' or cfg.spatial or cfg.gate_mode != 'open':
        raise ValueError('EM-line learning requires whole-level open replacement')
    if smoother and (not cfg.use_smoother or cfg.branch != 'H_SP'
            or cfg.smoother_levels != 'all'
            or not hasattr(model.smoother, 'line_smoother_spec')):
        raise ValueError('Line V-cycle learning requires the shared all-level line smoother')
    return cfg


def _check_line_factors(root):
    for level in _levels(root):
        if level.line_factors is None:
            raise InfeasibleCandidate('Missing learned line factors on a nonterminal level')
        for direction in ('x', 'y'):
            lower, diagonal = level.line_factors[direction]
            if (not bool(torch.isfinite(lower).all())
                    or not bool(torch.isfinite(diagonal).all())
                    or not bool((diagonal > 0).all())):
                raise InfeasibleCandidate('Nonfinite or nonpositive learned line factor')


def em_imitation_objective(example, model, cfg, *, em_iterations=5, em_rtol=1e-8):
    """Supervise every P_l with EM(A_l_student), including boundary constraints.

The mean of normalized row losses gives equal weight to each nonterminal
level.  Fixed/injection rows are omitted from the loss because they cannot be
learned.  The teacher executes a declared fixed CG budget, with the numerical
stationarity exception reported by energy_weights.
"""
    _validate(example, model, cfg, smoother=False)
    _positive_integer(em_iterations, 'em_iterations')
    if not np.isfinite(em_rtol) or not 0 < em_rtol < 1:
        raise ValueError('em_rtol must be finite and between zero and one')
    pcfg = replace(cfg, branch='H_P', use_smoother=False, replace_pre=0,
                   replace_post=0, replacement_group_pre=1,
                   replacement_group_post=1, use_learned_controller=False)
    terms, reports = [], []
    with offline_teacher_generation():
        root = make_graph(example.a, (example.n, example.n), model, pcfg)
        classical = make_graph(example.a, root.shape, model, pcfg, learned=False)
        feasibility = _check_graph(root, model, pcfg, classical)
        for level in _levels(root):
            # raw_scipy is the detached numerical value of this exact student
            # graph node.  It is not the corresponding classical parent A_l.
            with torch.no_grad():
                weights, report = energy_weights(level.raw_scipy, level.pattern,
                    maxiter=em_iterations, rtol=em_rtol, fixed_iterations=True)
                target = torch.as_tensor(weights, dtype=torch.float64).detach()
            student = level.interpolation_weights
            allowed = torch.as_tensor(np.array(level.base_weights != 0, copy=True))
            counts = allowed.sum(1)
            free = counts > 1
            if bool(free.any()):
                scale = target.abs().sum(1).clamp_min(1e-12)
                per_row = (((student-target)/scale[:, None]).square()
                           * allowed).sum(1)/counts.clamp_min(1)
                term = per_row[free].mean()
                terms.append(term)
                value = float(term.detach())
            else:
                value = 0.
            reports.append(dict(level=level.index, shape=list(level.shape),
                operator_digest=operator_digest(level.raw_scipy),
                loss=value, trainable_rows=int(free.sum()),
                teacher_target_detached=not target.requires_grad, **report))
        if not terms:
            raise ValueError('EM imitation requires trainable interpolation rows')
        loss = torch.stack(terms).mean()
        if not bool(torch.isfinite(loss)):
            raise InfeasibleCandidate('Nonfinite EM imitation loss')
    return loss, dict(loss=float(loss.detach()), em_loss=float(loss.detach()),
        levels=reports, em_iteration_budget=em_iterations,
        teacher_input='actual recursively student-generated A_l',
        teacher_targets_detached=True, feasible=feasibility['feasible'],
        fallback_used=False, online_em=False, cycles=0,
        objective='normalized P imitation on each actual student level')


def line_vcycle_objective(example, model, cfg, *, cycles=3, seed=1,
                         probes_per_kind=1, hard_smoothing=4, hard_cycles=4,
                         lambda_stability=.1, rhs=None):
    """Raw complete V-cycles, with generated factors reused in every sweep.

Use in both the frozen-P stage and joint stage.  The caller controls parameter
requires_grad; no detach of student P, actual Ac, or lower-level features is
introduced here.  Probe construction is deliberately detached.  This is a
differentiable residual surrogate, not a derivative of wall-clock time.
"""
    _validate(example, model, cfg, smoother=True)
    _positive_integer(cycles, 'cycles')
    _positive_integer(probes_per_kind, 'probes_per_kind')
    for value, name in ((hard_smoothing, 'hard_smoothing'), (hard_cycles, 'hard_cycles')):
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if not np.isfinite(lambda_stability) or lambda_stability < 0:
        raise ValueError('lambda_stability must be finite and nonnegative')
    cfg = replace(cfg, use_learned_controller=False)
    with offline_teacher_generation():
        root = make_graph(example.a, (example.n, example.n), model, cfg)
        classical = make_graph(example.a, root.shape, model, cfg, learned=False)
        feasibility = _check_graph(root, model, cfg, classical)
        _check_line_factors(root)
        if rhs is None:
            probes, labels = _probe_residuals(root, classical, model, cfg,
                seed=seed, per_kind=probes_per_kind,
                hard_smoothing=hard_smoothing, hard_cycles=hard_cycles)
        else:
            probes = torch.as_tensor(rhs, dtype=torch.float64).detach().cpu()
            if probes.ndim == 1:
                probes = probes[:, None]
            if (probes.ndim != 2 or probes.shape[0] != root.a.shape[0]
                    or probes.shape[1] < 1 or not bool(torch.isfinite(probes).all())):
                raise ValueError('rhs must contain finite nonzero probe columns')
            norms = torch.linalg.vector_norm(probes, dim=0)
            if not bool((norms > 0).all()):
                raise ValueError('rhs must contain finite nonzero probe columns')
            probes = probes/norms
            labels = ['supplied']*probes.shape[1]
        x, history = torch.zeros_like(probes), []
        for k in range(cycles):
            x = cycle(root, x, probes, model, cfg, k)
            history.append(torch.linalg.vector_norm(probes-root.a.apply(x), dim=0))
        h = torch.stack(history)
        if not bool(torch.isfinite(h).all()):
            raise InfeasibleCandidate('Raw learned line V-cycle produced nonfinite residuals')
        weights = torch.arange(1, cycles+1, dtype=h.dtype)
        weights = weights/weights.sum()
        residual = (weights[:, None]*h.clamp_min(cfg.mg.tolerance).log()).sum(0).mean()
        previous = torch.cat((torch.ones_like(h[:1]), h[:-1]), dim=0)
        # Once a probe is below the solve tolerance, tiny floating-point
        # fluctuations are not a convergence failure.  Penalize only growth
        # above the larger of its previous residual and the solve tolerance.
        stability = F.relu(h/previous.clamp_min(cfg.mg.tolerance)-1.).square().mean()
        loss = residual+lambda_stability*stability
        if not bool(torch.isfinite(loss)):
            raise InfeasibleCandidate('Nonfinite learned line V-cycle objective')
    return loss, dict(loss=float(loss.detach()), residual_loss=float(residual.detach()),
        stability_loss=float(stability.detach()), cycles=cycles,
        final_residual=float(h[-1].detach().mean()),
        worst_final_residual=float(h[-1].detach().max()), history=h.detach().tolist(),
        probe_labels=labels, feasible=feasibility['feasible'], fallback_used=False,
        line_factor_generations=len(list(_levels(root))), online_em=False,
        pre_post_coefficients_shared=True,
        objective='raw multi-cycle residual surrogate; selection by measured setup + solve time')
