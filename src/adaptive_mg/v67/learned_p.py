"""P-only learning with fixed line_alt and a stochastic Frobenius objective.

For the linear, cached V-cycle E, E||E z||^2 = ||E||_F^2 when E[zz.T]=I.
No dense error-propagation matrix or learned smoother is needed.
"""
from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from torch import nn

from ..provenance import operator_digest
from ..strategy import get_strategy
from .models import Components, TinyNet, CONTRACT, DETECTOR_INPUTS, CONTROLLER_INPUTS
from .research_controls import ControlledTransferCNN
from .research_runtime import offline_teacher_generation
from .joint_transfer_training import FreeTransferOperators, _binding, _check_graph, _levels, _train_only
from .unroll import make_graph, cycle, _features
from .limited import vector_digest

VERSION = 'line-alt-learned-p-frobenius-v1'
ORACLE_VERSION = 'free-p-line-alt-v1'


class NoNeuralSmoother(nn.Module):
    """Zero-parameter serialization marker. All sweeps are classical line_alt."""
    def fixed_smoother_spec(self):
        return dict(strategy='line_alt', version=1)

    def forward(self, *args, **kwargs):
        raise RuntimeError('P-only models have no neural smoother')


def make_p_components(*, hidden=8, seed=20261020):
    torch.manual_seed(seed)
    model = Components(NoNeuralSmoother(),
        ControlledTransferCNN(hidden=hidden, support='support_preserving'),
        TinyNet(DETECTOR_INPUTS, 1), TinyNet(CONTROLLER_INPUTS, 4))
    model.metadata = dict(contract=CONTRACT, role='student', p_only=True,
        smoother_trained=False, transfer_trained=False, detector_trained=False,
        controller_trained=False, certificate=dict(validated=False),
        smoother='fixed classical line_alt, pre/post=2/2', final_test_seen=False)
    return model.eval()


def require_p_only(model, cfg):
    if (not isinstance(model.smoother, NoNeuralSmoother) or cfg.use_smoother
            or cfg.branch != 'H_P' or not cfg.use_transfer
            or cfg.replace_pre or cfg.replace_post
            or get_strategy(cfg.mg.strategy_name).smoother != 'line_alt'):
        raise ValueError('P-only requires no NN smoother and forced H_P with classical pre/post')


def rademacher_errors(n, probes, seed):
    if type(probes) is not int or probes < 1:
        raise ValueError('Frobenius probes must be a positive integer')
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    return (2*torch.randint(0, 2, (n, probes), generator=generator)-1).double()


def apply_error_operator(root, errors, model, cfg, *, cycles=1):
    """E z from a homogeneous V-cycle: initial error=z, right-hand side=0."""
    x = errors
    zero = torch.zeros_like(errors)
    for k in range(cycles):
        x = cycle(root, x, zero, model, cfg, k)
    return x


def frobenius_objective(example, model, cfg, *, probes=8, seed=1, errors=None,
                        teacher=None, lambda_action=1., lambda_operator=.1):
    """Unbiased estimate of ||E||_F^2/N; KD is optional and separately reported.

    Rademacher training probes have unit covariance. Explicit errors enable
    same-tensor diagnostics (or a scaled complete basis in exact small tests).
    P, Galerkin Ac, recursive line solves and terminal solves all retain AD.
    No safeguarded solve, rollback or classical repair can earn learned credit.
    """
    _train_only(example)
    require_p_only(model, cfg)
    _binding(example, model)
    z = rademacher_errors(example.a.shape[0], probes, seed) if errors is None else errors.detach().cpu().double()
    if (z.ndim != 2 or z.shape[0] != example.a.shape[0] or z.shape[1] < 1
            or not bool(torch.isfinite(z).all())):
        raise ValueError('Invalid fixed error probes')
    with offline_teacher_generation():
        root = make_graph(example.a, (example.n, example.n), model, cfg)
        classical = make_graph(example.a, root.shape, model, cfg, learned=False)
        _check_graph(root, model, cfg, classical)
        ez = apply_error_operator(root, z, model, cfg)
        if not bool(torch.isfinite(ez).all()):
            raise FloatingPointError('Nonfinite error propagation')
        raw_frob = ez.square().sum(dim=0).mean()
        task = raw_frob/root.a.shape[0]
        with torch.no_grad():
            baseline_ez = apply_error_operator(classical, z, model, cfg)
            baseline = baseline_ez.square().sum(dim=0).mean()/root.a.shape[0]
        action = operator = task.new_zeros(())
        if teacher is not None:
            require_p_only(teacher, cfg)
            _binding(example, teacher)
            with torch.no_grad():
                target = make_graph(example.a, root.shape, teacher, cfg)
                _check_graph(target, teacher, cfg, classical)
                target_ez = apply_error_operator(target, z, teacher, cfg)
                target_correction = z-target_ez
            difference = ez-target_ez
            denominator = (target_correction*root.a.apply(target_correction)).sum(0).clamp_min(1e-100)
            action = ((difference*root.a.apply(difference)).sum(0).clamp_min(0.)/denominator).mean()
            terms = []
            for current, ideal in zip(_levels(root), _levels(target)):
                w, t = current.interpolation_weights, ideal.interpolation_weights.detach()
                allowed = torch.tensor(current.base_weights != 0)
                terms.append((((w-t)/t.abs().sum(1, keepdim=True).clamp_min(1e-12))[allowed]).square().mean())
            operator = torch.stack(terms).mean()
        loss = task+lambda_action*action+lambda_operator*operator
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite Frobenius objective')
    return loss, dict(loss=float(loss.detach()), frobenius_squared_estimate=float(raw_frob.detach()),
        frobenius_per_unknown=float(task.detach()), classical_frobenius_per_unknown=float(baseline),
        action_loss=float(action.detach()), operator_loss=float(operator.detach()),
        probes=z.shape[1], cycles=1, feasible=True, fallback_used=False,
        nn_smoother=False, classical_smoother='line_alt',
        objective='stochastic full-V-cycle ||E||_F^2 / N; spectral-radius surrogate, not direct rho/time',
        theoretical_scope='rho(E)<=||E||_2<=||E||_F for exact norm; samples estimate squared norm')


def make_p_oracle(example, model, cfg, *, initialization='network', seed=1, perturbation=0.):
    _train_only(example)
    require_p_only(model, cfg)
    if initialization not in ('network', 'bilinear', 'jacobi') or perturbation < 0:
        raise ValueError('Unknown P-only initialization')
    source = model if initialization == 'network' else make_p_components(hidden=8, seed=seed)
    shapes, logits = [], []
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    with offline_teacher_generation(), torch.no_grad():
        root = make_graph(example.a, (example.n, example.n), source, cfg)
        for level in _levels(root):
            features = _features(level.a, level.shape).to(next(source.transfer.parameters()))
            value = (source.transfer.forward_graph(level.a, level.pattern, level.base_weights)
                     if hasattr(source.transfer, 'forward_graph') else source.transfer(features))
            value = value.detach().cpu()
            if perturbation:
                value = value+perturbation*torch.randn(value.shape, generator=generator, dtype=value.dtype)
            shapes.append(level.shape); logits.append(value)
    oracle = deepcopy(model)
    oracle.transfer = FreeTransferOperators(shapes, logits, model.transfer.complexity_caps)
    oracle.metadata.update(role='offline_p_oracle', p_only=True, oracle_version=ORACLE_VERSION,
        initialization='bilinear' if initialization == 'jacobi' else initialization,
        oracle_binding=dict(operator_digest=operator_digest(example.a), shape=[example.n, example.n]))
    return oracle.eval()


def export_p_oracle(model):
    if not isinstance(model.smoother, NoNeuralSmoother) or not isinstance(model.transfer, FreeTransferOperators):
        raise TypeError('Not a P-only oracle')
    return dict(version=ORACLE_VERSION, metadata=deepcopy(model.metadata),
        shapes=[list(s) for s in model.transfer.shapes], caps=dict(model.transfer.complexity_caps),
        logits=[v.detach().cpu().clone() for v in model.transfer.logits])


def import_p_oracle(payload):
    if payload.get('version') != ORACLE_VERSION:
        raise ValueError('Unsupported P oracle')
    model = make_p_components(hidden=8, seed=1)
    model.transfer = FreeTransferOperators(payload['shapes'], payload['logits'], payload['caps'])
    model.metadata = deepcopy(payload['metadata'])
    return model.eval()


def p_diagnostic(example, before, after, cfg, *, probes=16, seed=1, rhs_count=1,
                 repeats=3, warmups=1):
    """Frozen error probes for pre/post learning proof; complete time separately."""
    from .solver import PreparedAdaptiveMG
    z = rademacher_errors(example.a.shape[0], max(probes, rhs_count), seed)
    rhs = example.a@z[:, :rhs_count].numpy()
    rhs /= np.linalg.norm(rhs, axis=0, keepdims=True).clip(min=1e-100)
    rows = {}
    for name, model in (('before', before), ('after', after)):
        require_p_only(model, cfg)
        try:
            with torch.no_grad(), offline_teacher_generation():
                root = make_graph(example.a, (example.n, example.n), model, cfg)
                classical = make_graph(example.a, root.shape, model, cfg, learned=False)
                _check_graph(root, model, cfg, classical)
                ez = apply_error_operator(root, z, model, cfg)
                if not bool(torch.isfinite(ez).all()):
                    raise FloatingPointError('Nonfinite diagnostic error')
                loss = float(ez.square().sum(0).mean()/root.a.shape[0])
            row = dict(raw_valid=True, frobenius_per_unknown=loss,
                error_reduction=float((ez.norm(dim=0)/z.norm(dim=0)).mean()), raw_fallback_allowed=False)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            row = dict(raw_valid=False, reason=f'{type(error).__name__}: {error}', raw_fallback_allowed=False)
        measured = []
        for repeat in range(-warmups, repeats):
            owned = model.frozen_inference_copy()
            start = perf_counter()
            try:
                prepared = PreparedAdaptiveMG(example.a, example.n, owned, cfg)
                results = prepared.solve_many(rhs.T)
                elapsed = perf_counter()-start
                item = dict(seconds=elapsed, successful=all(r.converged and
                    r.final_true_residual <= r.stopping_threshold for r in results),
                    cycles=sum(r.executed_cycles for r in results),
                    accepted_neural_cycles=sum(r.stats.get('accepted_neural_cycles', 0) for r in results),
                    rollback_count=sum(r.stats.get('rollback_count', 0) for r in results),
                    setup_failures=sum(r.stats.get('setup_failures', 0) for r in results),
                    setup_seconds=sum(r.setup_seconds for r in results),
                    solve_seconds=sum(r.solve_seconds for r in results),
                    smoother_nn_calls=sum(r.stats.get('smoother_nn_calls', 0) for r in results))
            except (ValueError, RuntimeError, FloatingPointError) as error:
                item = dict(seconds=perf_counter()-start, successful=False, reason=str(error))
            if repeat >= 0:
                measured.append(item)
        row['measured'] = dict(seconds=float(np.median([i['seconds'] for i in measured])),
            successful=all(i['successful'] for i in measured), samples=measured,
            rhs_count=rhs_count, scope='setup + deployed solve; tolerance, classical tail and fallback included')
        rows[name] = row
    b, a = rows['before'], rows['after']
    return dict(case=example.name, n=example.n, split=getattr(example, 'research_split', None),
        operator_digest=operator_digest(example.a), error_probe_digest=vector_digest(z.numpy()),
        rhs_digest=vector_digest(rhs), probes=z.shape[1], cycles=1, models=rows,
        frobenius_improved=(a['frobenius_per_unknown'] < b['frobenius_per_unknown']
                            if a['raw_valid'] and b['raw_valid'] else None),
        time_improved=(a['measured']['seconds'] < b['measured']['seconds']
                       if a['measured']['successful'] and b['measured']['successful'] else None))
