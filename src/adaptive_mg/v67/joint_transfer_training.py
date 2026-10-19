"""Raw whole-cycle learning and TRAIN-only numerical operator oracles.

Offline Adam optimizes the same interpolation projection and paired 9-point
smoother representation as the small student.  Runtime measurements here never
run a recovery cycle: an infeasible or nonconvergent oracle is a failed candidate.
The production study evaluates the selected student with classical recovery.
"""
from copy import deepcopy
from dataclasses import replace
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..provenance import operator_digest
from ..hierarchy import build_fixed_hierarchy
from ..strategy import get_strategy
from .banks import Stats, hybrid_cycle, prepare_transfer_bank, prepare_smoother_bank
from .models import Components
from .paired_smoother import PairedStencilSmoother
from .research_controls import ControlledTransferCNN
from .research_runtime import offline_teacher_generation
from .research_training import transfer_feasibility
from .spatial import SpatialState
from .unroll import make_graph, cycle, smooth, _features

ORACLE_VERSION = 'joint-free-operators-v1'


class InfeasibleCandidate(ValueError):
    """The raw learned candidate cannot supply a valid training/evaluation loss."""


def _train_only(example):
    split = getattr(example, 'research_split', None)
    if split is not None and split != 'train':
        raise ValueError('Offline optimization and distillation require TRAIN operators')


def _raw_config(cfg):
    # These helpers evaluate an explicit branch, never a temporal policy.
    return replace(cfg, spatial=False, gate_mode='open', use_learned_controller=False)


def _binding(example, model):
    binding = model.metadata.get('oracle_binding')
    if binding and (binding['operator_digest'] != operator_digest(example.a)
                    or binding['shape'] != [example.n, example.n]):
        raise ValueError('Offline oracle belongs to a different operator')


def make_joint_components(*, hidden=8, seed=20261020):
    model = Components.create(hidden=hidden, seed=seed)
    model.transfer = ControlledTransferCNN(hidden=hidden, support='support_preserving')
    model.smoother = PairedStencilSmoother(hidden=hidden)
    model.metadata.update(joint_pair_version=1, initialization_seed=seed,
                          role='student', final_test_seen=False,
                          trained_against='line_alt + bilinear + full')
    return model.eval()


def curriculum_cycles(update, total_updates, *, start=3, end=6):
    if min(start, end, total_updates) < 1 or not 0 <= update < total_updates:
        raise ValueError('invalid cycle curriculum')
    return start if update < (total_updates + 1) // 2 else end


def _levels(root):
    while root.coarse is not None:
        yield root
        root = root.coarse


def _check_graph(root, model, cfg, classical):
    level = root
    while level is not None:
        values = [level.a.values]
        if level.p is not None:
            values.append(level.p.values)
        if level.dirs is not None:
            values.extend([level.dirs, level.gains])
        if not all(bool(torch.isfinite(v).all()) for v in values):
            raise InfeasibleCandidate('nonfinite learned operator')
        level = level.coarse
    feasibility, _ = transfer_feasibility(root, model, cfg, classical)
    if not feasibility['feasible']:
        raise InfeasibleCandidate(str(feasibility['violations']))
    return feasibility


def _probe_residuals(root, classical, model, cfg, *, seed, per_kind=1,
                     hard_smoothing=4, hard_cycles=4):
    if isinstance(per_kind, bool) or per_kind < 1 or min(hard_smoothing, hard_cycles) < 0:
        raise ValueError('invalid hard-probe budget')
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    labels = ['random'] * per_kind + ['classical_smoothed'] * per_kind + ['learned_hard'] * per_kind
    with torch.no_grad():
        samples = torch.randn(root.a.shape[0], 3 * per_kind,
                              generator=generator, dtype=torch.float64)
        random, classical_error, learned_error = samples.split(per_kind, dim=1)
        for k in range(hard_smoothing):
            classical_error = classical_error - smooth(classical,
                classical.a.apply(classical_error), cfg, reverse=bool(k % 2))
        zero = torch.zeros_like(learned_error)
        for k in range(hard_cycles):
            learned_error = cycle(root, learned_error, zero, model, cfg, k)
        errors = torch.cat((random, classical_error, learned_error), dim=1)
        residuals = root.a.apply(errors)
        norms = torch.linalg.vector_norm(residuals, dim=0)
        if not bool(torch.isfinite(residuals).all()):
            raise InfeasibleCandidate('hard-probe generation diverged')
        # A solved-away probe cannot give a false zero-loss reward.
        vanished = norms < 1e-100
        if bool(vanished.any()):
            residuals[:, vanished] = root.a.apply(samples[:, vanished])
            norms = torch.linalg.vector_norm(residuals, dim=0)
        residuals = residuals / norms.clamp_min(1e-100)
    return residuals.detach(), labels


def mixed_probes(example, model, cfg, *, seed=1, per_kind=1,
                 hard_smoothing=4, hard_cycles=4):
    """Detached r=Ae probes; no exact solution or held-out operator is needed."""
    _binding(example, model)
    cfg = _raw_config(cfg)
    with offline_teacher_generation():
        learned = make_graph(example.a, (example.n, example.n), model, cfg)
        classical = make_graph(example.a, learned.shape, model, cfg, learned=False)
        _check_graph(learned, model, cfg, classical)
        return _probe_residuals(learned, classical, model, cfg, seed=seed,
            per_kind=per_kind, hard_smoothing=hard_smoothing, hard_cycles=hard_cycles)


def _operator_loss(student, teacher, cfg):
    """Numerical P and B coefficients, not an arbitrary direction/gain factorization."""
    terms = []
    slevels, tlevels = list(_levels(student)), list(_levels(teacher))
    if len(slevels) != len(tlevels):
        raise ValueError('Oracle and student hierarchy depths differ')
    from ..transfer import OFFSETS_9
    for s, t in zip(slevels, tlevels):
        if not np.array_equal(s.pattern.columns, t.pattern.columns):
            raise ValueError('Oracle and student interpolation slots differ')
        if cfg.use_transfer:
            sw, tw = s.interpolation_weights, t.interpolation_weights.detach()
            allowed = torch.as_tensor(np.array(s.base_weights != 0, copy=True))
            terms.append((((sw - tw) / tw.abs().sum(1, keepdim=True).clamp_min(1e-8))[allowed]).square().mean())
        if cfg.use_smoother:
            # In B=DIRECTION * GAIN * D^{-1}, D belongs to each actual Ac.
            sb = s.dirs * s.gains[:, :, None, None, None]
            tb = t.dirs.detach() * t.gains.detach()[:, :, None, None, None]
            ds = s.a.diagonal().abs().clamp_min(1e-14).reshape(s.shape)
            dt = t.a.diagonal().detach().abs().clamp_min(1e-14).reshape(t.shape)
            nx, ny = s.shape
            valid = torch.zeros((9, nx, ny), dtype=torch.bool)
            sden = torch.ones((9, nx, ny), dtype=sb.dtype)
            tden = torch.ones_like(sden)
            for k, (di, dj) in enumerate(OFFSETS_9):
                i0, i1 = max(0, -di), min(nx, nx-di)
                j0, j1 = max(0, -dj), min(ny, ny-dj)
                valid[k, i0:i1, j0:j1] = True
                sden[k, i0:i1, j0:j1] = ds[i0+di:i1+di, j0+dj:j1+dj]
                tden[k, i0:i1, j0:j1] = dt[i0+di:i1+di, j0+dj:j1+dj]
            sb, tb = sb / sden, tb / tden
            scale = tb.square().mean().clamp_min(1e-30)
            terms.append((sb - tb)[:, :, valid].square().mean() / scale)
    if not terms:
        raise ValueError('No learned operators selected for distillation')
    return torch.stack(terms).mean()


def joint_objective(example, model, cfg, *, cycles=3, seed=1, probes_per_kind=1,
                    hard_smoothing=4, hard_cycles=4, teacher=None,
                    lambda_action=1., lambda_operator=.1, lambda_stability=.1):
    """Differentiable residual surrogate; measured total time selects models.

    All cycles are raw learned cycles. No classical tail, feasibility repair,
    line-search substitution or fallback can count as neural progress.
    """
    _train_only(example)
    _binding(example, model)
    if cycles < 1 or min(lambda_action, lambda_operator, lambda_stability) < 0:
        raise ValueError('invalid objective settings')
    cfg = _raw_config(cfg)
    with offline_teacher_generation():
        root = make_graph(example.a, (example.n, example.n), model, cfg)
        classical = make_graph(example.a, root.shape, model, cfg, learned=False)
        feasibility = _check_graph(root, model, cfg, classical)
        rhs, labels = _probe_residuals(root, classical, model, cfg, seed=seed,
            per_kind=probes_per_kind, hard_smoothing=hard_smoothing, hard_cycles=hard_cycles)
        x = torch.zeros_like(rhs)
        history = []
        for k in range(cycles):
            x = cycle(root, x, rhs, model, cfg, k)
            history.append(torch.linalg.vector_norm(rhs-root.a.apply(x), dim=0))
        h = torch.stack(history)
        if not bool(torch.isfinite(h).all()):
            raise InfeasibleCandidate('raw learned cycles produced nonfinite residuals')
        weights = torch.arange(1, cycles + 1, dtype=h.dtype)
        weights = weights / weights.sum()
        residual = (weights[:, None] * h.clamp_min(cfg.mg.tolerance).log()).sum(0).mean()
        previous = torch.cat((torch.ones_like(h[:1]), h[:-1]), dim=0)
        stability = F.relu(h/previous.clamp_min(1e-100)-1.).square().mean()
        action = operator = residual.new_zeros(())
        if teacher is not None:
            _binding(example, teacher)
            with torch.no_grad():
                target = make_graph(example.a, root.shape, teacher, cfg)
                _check_graph(target, teacher, cfg, classical)
                target_action = cycle(target, torch.zeros_like(rhs), rhs, teacher, cfg, 0).detach()
            student_action = cycle(root, torch.zeros_like(rhs), rhs, model, cfg, 0)
            difference = student_action - target_action
            denominator = (target_action * root.a.apply(target_action)).sum(0).clamp_min(1e-100)
            action = ((difference * root.a.apply(difference)).sum(0).clamp_min(0.) / denominator).mean()
            operator = _operator_loss(root, target, cfg)
        loss = residual + lambda_stability*stability + lambda_action*action + lambda_operator*operator
        if not bool(torch.isfinite(loss)):
            raise InfeasibleCandidate('nonfinite joint loss')
    return loss, dict(loss=float(loss.detach()), residual_loss=float(residual.detach()),
        stability_loss=float(stability.detach()), action_loss=float(action.detach()),
        operator_loss=float(operator.detach()), final_residual=float(h[-1].detach().mean()),
        worst_final_residual=float(h[-1].detach().max()), history=h.detach().tolist(),
        probe_labels=labels, cycles=cycles, feasible=feasibility['feasible'],
        fallback_used=False, objective='raw full-cycle residual surrogate; selection by measured total time')


class FreeTransferOperators(nn.Module):
    training_only = True
    support = 'support_preserving'

    def __init__(self, shapes, logits, caps):
        super().__init__()
        self.shapes = [tuple(shape) for shape in shapes]
        self.complexity_caps = dict(caps)
        self.logits = nn.ParameterList([nn.Parameter(v.detach().clone()) for v in logits])

    def forward_graph(self, a, pattern, baseline=None):
        try:
            index = self.shapes.index(tuple(pattern.fine_shape))
        except ValueError as error:
            raise ValueError('Oracle transfer used on a different hierarchy') from error
        return self.logits[index]


class FreePairedSmoothers(nn.Module):
    training_only = True
    basis_count = 2
    split_direction_gain = True

    def __init__(self, shapes, directions, raw_gains):
        super().__init__()
        self.shapes = [tuple(shape) for shape in shapes]
        self.directions = nn.ParameterList([nn.Parameter(v.detach().clone()) for v in directions])
        self.raw_gains = nn.ParameterList([nn.Parameter(v.detach().clone()) for v in raw_gains])

    def paired_directions_and_gains(self, features, index=0):
        if index >= len(self.shapes) or tuple(features.shape[-2:]) != self.shapes[index]:
            raise ValueError('Oracle smoother used on a different hierarchy')
        raw = self.directions[index]
        directions = raw / raw.square().sum(2, keepdim=True).clamp_min(1e-16).sqrt()
        return directions, F.softplus(self.raw_gains[index])


def make_oracle(example, model, cfg, *, initialization='network', seed=1, perturbation=0.):
    """Create a TRAIN-A-bound free pair with exactly the student's representation."""
    _train_only(example)
    _binding(example, model)
    if initialization not in ('network', 'jacobi') or perturbation < 0:
        raise ValueError('invalid oracle initialization')
    if getattr(model.transfer, 'support', None) != 'support_preserving':
        raise ValueError('Joint oracle requires the same bilinear sparse support as the student')
    if getattr(model.transfer, 'parameterization', 'softmax') != 'softmax':
        raise ValueError('Joint oracle requires the student softmax projection')
    cfg = _raw_config(cfg)
    with offline_teacher_generation(), torch.no_grad():
        source = model if initialization == 'network' else make_joint_components(hidden=8, seed=seed)
        root = make_graph(example.a, (example.n, example.n), source, cfg)
        levels = list(_levels(root))
        if not levels:
            raise ValueError('Oracle requires at least one nonterminal level')
        shapes, logits, directions, gains = [], [], [], []
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        for level in levels:
            features = _features(level.a, level.shape).to(next(source.transfer.parameters()))
            raw = (source.transfer.forward_graph(level.a, level.pattern, level.base_weights)
                   if hasattr(source.transfer, 'forward_graph') else source.transfer(features))
            dirs, gs = source.smoother.paired_directions_and_gains(features, level.index)
            raw, dirs, gs = raw.detach().cpu(), dirs.detach().cpu(), gs.detach().cpu()
            if perturbation:
                raw = raw + perturbation*torch.randn(raw.shape, generator=generator, dtype=raw.dtype)
                dirs = dirs + perturbation*torch.randn(dirs.shape, generator=generator, dtype=dirs.dtype)
            shapes.append(level.shape); logits.append(raw); directions.append(dirs)
            gains.append(gs + torch.log(-torch.expm1(-gs)))  # inverse softplus
    oracle = deepcopy(model)
    oracle.transfer = FreeTransferOperators(shapes, logits, model.transfer.complexity_caps)
    oracle.smoother = FreePairedSmoothers(shapes, directions, gains)
    oracle.metadata.update(role='offline_oracle', oracle_version=ORACLE_VERSION,
        oracle_binding=dict(operator_digest=operator_digest(example.a), shape=[example.n, example.n]),
        initialization=initialization, perturbation=float(perturbation), final_test_seen=False)
    return oracle.eval()


def export_oracle(model):
    if not isinstance(model.transfer, FreeTransferOperators) or not isinstance(model.smoother, FreePairedSmoothers):
        raise TypeError('Not a free operator oracle')
    return dict(version=ORACLE_VERSION, metadata=deepcopy(model.metadata),
        shapes=[list(s) for s in model.transfer.shapes], caps=dict(model.transfer.complexity_caps),
        logits=[v.detach().cpu().clone() for v in model.transfer.logits],
        directions=[v.detach().cpu().clone() for v in model.smoother.directions],
        raw_gains=[v.detach().cpu().clone() for v in model.smoother.raw_gains])


def import_oracle(payload):
    if payload.get('version') != ORACLE_VERSION:
        raise ValueError('Unsupported oracle checkpoint')
    model = Components.create(hidden=8, seed=1)
    model.transfer = FreeTransferOperators(payload['shapes'], payload['logits'], payload['caps'])
    model.smoother = FreePairedSmoothers(payload['shapes'], payload['directions'], payload['raw_gains'])
    model.metadata = deepcopy(payload['metadata'])
    return model.eval()


def measure_raw_candidate(example, model, cfg, *, heldout_seed=19999, target_rhs_count=1,
                          cycles=6, repeats=1, hard_smoothing=4, hard_cycles=4, probe_model=None):
    """Measure cold setup+solve for held-out probes; return failure, never fallback.

    Includes the bilinear recovery hierarchy, operator generation and all raw
    cycles. Probe generation is outside the timer. Per-A oracle search/training
    is offline; student inference must still be measured separately by the study.
    """
    _binding(example, model)
    if min(target_rhs_count, cycles, repeats) < 1:
        raise ValueError('positive measurement budgets required')
    cfg = _raw_config(cfg)
    failure = dict(successful=False, seconds=None, setup_seconds=None, solve_seconds=None,
                   cycles=0, heldout_residual=None, fallback_used=False, rhs_count=target_rhs_count)
    try:
        # All candidates must see IDENTICAL held-out errors. Using the candidate
        # itself to generate its hard probes would bias the timing comparison.
        reference = probe_model if probe_model is not None else make_joint_components(seed=heldout_seed)
        rhs, _ = mixed_probes(example, reference, cfg, seed=heldout_seed,
            per_kind=max(1, (target_rhs_count+2)//3), hard_smoothing=hard_smoothing, hard_cycles=hard_cycles)
        with offline_teacher_generation(), torch.no_grad():
            graph = make_graph(example.a, (example.n, example.n), model, cfg)
            classical = make_graph(example.a, graph.shape, model, cfg, learned=False)
            _check_graph(graph, model, cfg, classical)
            xprobe = torch.zeros_like(rhs)
            for k in range(cycles):
                xprobe = cycle(graph, xprobe, rhs, model, cfg, k)
            fixed_residual = float(torch.linalg.vector_norm(rhs-graph.a.apply(xprobe), dim=0).max())
            if not np.isfinite(fixed_residual):
                raise InfeasibleCandidate('fixed-cycle held-out probes diverged')
        # Interleave kinds so small RHS budgets do not select only random probes.
        count = rhs.shape[1]//3
        indices = [j+kind*count for j in range(count) for kind in range(3)]
        rhs = rhs[:, indices[:target_rhs_count]].numpy()
        timings, setups, solves, totals, endpoints, fixed = [], [], [], [], [], []
        for _ in range(repeats):
            stats = Stats()
            start = perf_counter()
            with offline_teacher_generation():
                baseline = build_fixed_hierarchy(example.a, (example.n, example.n),
                    get_strategy(cfg.mg.strategy_name), cfg.mg, stats)
                root = prepare_transfer_bank(baseline, model, cfg, stats) if cfg.use_transfer else baseline
                root = prepare_smoother_bank(root, model, cfg, stats) if cfg.use_smoother else root
            setup = perf_counter()-start
            began = perf_counter(); total_cycles = 0; ratios = []; at_fixed = []
            for column in range(target_rhs_count):
                b = rhs[:, column]; x = np.zeros_like(b)
                norm = max(float(np.linalg.norm(b)), 1e-100)
                spatial = SpatialState(model, cfg)
                threshold = max(cfg.mg.tolerance, cfg.mg.absolute_tolerance/norm)
                ratio = 1.; fixed_ratio = None
                for k in range(cfg.mg.max_cycles):
                    x = hybrid_cycle(root, x, b, cfg, stats, spatial, k, cfg.use_smoother)
                    total_cycles += 1
                    ratio = float(np.linalg.norm(b-example.a@x)/norm)
                    if k+1 == cycles: fixed_ratio = ratio
                    if not np.isfinite(ratio):
                        raise InfeasibleCandidate('raw candidate solve diverged')
                    if ratio <= threshold: break
                if ratio > threshold:
                    return dict(failure, reason='raw candidate failed tolerance', cycles=total_cycles,
                        heldout_residual=ratio, setup_seconds=setup, solve_seconds=perf_counter()-began,
                        seconds=perf_counter()-start)
                ratios.append(ratio); at_fixed.append(ratio if fixed_ratio is None else fixed_ratio)
            solve_seconds = perf_counter()-began
            timings.append(setup+solve_seconds); setups.append(setup); solves.append(solve_seconds)
            totals.append(total_cycles); endpoints.append(max(ratios)); fixed.append(max(at_fixed))
        return dict(successful=True, seconds=float(np.median(timings)),
            setup_seconds=float(np.median(setups)), solve_seconds=float(np.median(solves)),
            cycles=int(np.median(totals)), heldout_residual=fixed_residual,
            final_residual=float(max(endpoints)), fallback_used=False, rhs_count=target_rhs_count,
            repeats=repeats, raw_no_fallback=True, em_iterations=0,
            timing_scope='cold setup + raw solve; offline search and probe generation excluded')
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return dict(failure, reason=str(error))
