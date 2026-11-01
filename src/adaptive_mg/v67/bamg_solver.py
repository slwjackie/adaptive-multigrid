"""Direct-P multigrid: frozen reference support, differentiable values, warm CSR solves.

The CPU FP64 sparse-value backend differentiates the complete numerical hierarchy,
including line and coarse solves.  Symbolic zeros are retained in P and Galerkin A;
no magnitude pruning or re-selection is performed during a rollout or fine-tuning.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import resource
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from ..config import MGConfig
from ..grid import next_shape, terminal, validate_root_shape
from ..hierarchy import FixedLevel, WorkStats, classical_cycle
from ..provenance import stable_norm
from ..smoothers import LineSmootherCache
from ..strategy import get_strategy
from .autograd_sparse import SparseTensor
from .unroll import TLevel, smooth, _line_batches
from .bamg_transfer import (
    CANDIDATE_VERSION, CONSTRAINT_VERSION, FEATURE_VERSION, direct_features,
    direct_pattern, fixed_bilinear_mask, model_from_spec, project_weights, select_mask,
    sparse_from_weights,
)


@dataclass(frozen=True)
class StructuralCaps:
    """Budgets count stored entries, including zeros; never change support to fit."""
    max_p_nnz: int = 5_000_000
    max_a_nnz: int = 20_000_000
    max_operator_complexity: float = 8.0
    max_sparse_bytes: int = 2_000_000_000

    def __post_init__(self):
        if min(self.max_p_nnz, self.max_a_nnz, self.max_sparse_bytes) < 1:
            raise ValueError('Sparse budgets must be positive')
        if not np.isfinite(self.max_operator_complexity) or self.max_operator_complexity < 1:
            raise ValueError('Invalid operator-complexity budget')


class HierarchyBudgetError(ValueError):
    def __init__(self, reason, report):
        self.report = report
        super().__init__('BAMG hierarchy budget exceeded: ' + reason)


def _config(mg_config):
    cfg = mg_config or MGConfig(mode='classical', strategy_name='line_alt_bilinear_full',
                                pre_steps=2, post_steps=2, stencil_backend='csr')
    if isinstance(cfg, dict):
        cfg = MGConfig(**cfg)
    if cfg.coarsest_n != 3:
        raise ValueError('BAMG CNN geometry currently requires coarsest_n=3')
    strategy = get_strategy(cfg.strategy_name)
    if strategy.smoother != 'line_alt' or strategy.coarsening != 'full':
        raise ValueError('BAMG direct P requires classical line_alt and full coarsening')
    return cfg


def _csr(a, shape, cfg):
    shape = validate_root_shape(shape, cfg.coarsest_n)
    if not sp.issparse(a):
        raise TypeError('A must be sparse; dense fine-grid operators are unsupported')
    result = a.tocsr(copy=True).astype(np.float64)
    result.sum_duplicates(); result.sort_indices()
    if result.shape != (int(np.prod(shape)),) * 2:
        raise ValueError('Operator and grid dimensions disagree')
    if not np.isfinite(result.data).all():
        raise ValueError('Nonfinite root operator')
    return result, shape


def _operator_key(a):
    """Identity includes stored zeros, dtype, and exact values, not just PDE family."""
    a = a.tocsr(copy=True); a.sum_duplicates(); a.sort_indices()
    h = hashlib.sha256(repr((a.shape, str(a.dtype))).encode())
    for array in (a.indptr.astype(np.int64), a.indices.astype(np.int64), a.data):
        h.update(array.tobytes())
    return h.hexdigest()


def _module_key(model):
    h = hashlib.sha256(json.dumps(getattr(model, 'spec', {}), sort_keys=True).encode())
    h.update(type(model).__qualname__.encode())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(repr(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def mask_digest(masks):
    h = hashlib.sha256()
    for mask in masks:
        mask = torch.as_tensor(mask).detach().cpu().bool().contiguous()
        h.update(repr(tuple(mask.shape)).encode()); h.update(mask.numpy().tobytes())
    return h.hexdigest()


def _placement(model):
    p = next(model.parameters(), None)
    return (p.device, p.dtype) if p is not None else (torch.device('cpu'), torch.float64)


def _sync(model):
    device, _ = _placement(model)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@contextmanager
def _eval(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def _weights(a, shape, pattern, mask, model, l1_cap, timings):
    device, dtype = _placement(model)
    _sync(model); start = perf_counter()
    features = direct_features(a, shape, device=device, dtype=dtype)
    _sync(model); timings['feature_seconds'] += perf_counter() - start
    start = perf_counter()
    raw = model(features, mask.to(device))
    _sync(model); timings['weight_cnn_seconds'] += perf_counter() - start
    start = perf_counter()
    weights = project_weights(raw, mask.to(device), pattern, l1_cap=l1_cap)
    _sync(model); timings['projection_seconds'] += perf_counter() - start
    start = perf_counter()
    p = sparse_from_weights(pattern, weights, mask)
    timings['assembly_transfer_seconds'] += perf_counter() - start
    if not bool(torch.isfinite(p.values).all()):
        raise FloatingPointError('Nonfinite generated P')
    return p, weights


def _timings():
    return dict(feature_seconds=0., mask_cnn_seconds=0., weight_cnn_seconds=0.,
                selection_seconds=0., projection_seconds=0., assembly_transfer_seconds=0.,
                galerkin_seconds=0., factorization_seconds=0., reference_seconds=0.)


def memory_observation():
    """Process measurements, distinct from sparse-storage budgets (Linux bytes)."""
    highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == 'darwin' else 1024
    resident = None
    try:
        with open('/proc/self/status', encoding='utf-8') as stream:
            for line in stream:
                if line.startswith('VmRSS:'):
                    resident = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    return dict(process_peak_rss_bytes=int(highwater * scale), process_resident_bytes=resident)


def _storage_bytes(a):
    return int(a.data.nbytes + a.indices.nbytes + a.indptr.nbytes)


def _feature_coverage(a, shape, index):
    """Quantify actual couplings omitted by the ten local input channels."""
    rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr)); cols = a.indices
    ny = shape[1]
    outside = (np.abs(rows // ny - cols // ny) > 1) | (np.abs(rows % ny - cols % ny) > 1)
    norm = stable_norm(a.data)
    return dict(level=int(index), shape=list(shape),
                outside_9point_stored_nnz=int(np.count_nonzero(outside)),
                outside_9point_numerical_nnz=int(np.count_nonzero(a.data[outside])),
                outside_9point_relative_frobenius=(stable_norm(a.data[outside]) / norm if norm > 0 else 0.))


def _check_budget(levels, caps):
    caps = StructuralCaps() if caps is None else StructuralCaps(**caps) if isinstance(caps, dict) else caps
    matrices = [level.a.numpy() if isinstance(level.a, SparseTensor) else level.a for level in levels]
    ps = [level.p.numpy() if isinstance(level.p, SparseTensor) else level.p
          for level in levels if level.p is not None]
    report = dict(a_stored_nnz=sum(a.nnz for a in matrices),
                  p_stored_nnz=sum(p.nnz for p in ps),
                  a_numerical_nnz=int(sum(np.count_nonzero(a.data) for a in matrices)),
                  p_numerical_nnz=int(sum(np.count_nonzero(p.data) for p in ps)),
                  sparse_storage_bytes=sum(_storage_bytes(a) for a in matrices + ps),
                  operator_complexity=sum(a.nnz for a in matrices) / max(1, matrices[0].nnz),
                  caps=asdict(caps),
                  feature_coverage=[_feature_coverage(a, level.shape, level.index)
                                    for a, level in zip(matrices, levels)])
    # Storage is a measurable lower bound, not a claim about process peak RSS.
    for metric, limit in [('a_stored_nnz', caps.max_a_nnz), ('p_stored_nnz', caps.max_p_nnz),
                          ('sparse_storage_bytes', caps.max_sparse_bytes),
                          ('operator_complexity', caps.max_operator_complexity)]:
        if report[metric] > limit:
            raise HierarchyBudgetError(metric, report)
    return report


def graph_levels(root):
    while root is not None:
        yield root
        root = root.coarse


class ReferenceModels:
    """Immutable stage-1 snapshots and problem/level masks from their own hierarchy.

    The cache stores candidate mappings as well as binary masks. Calls return
    clones so caller mutations cannot change future support. Frozen parameters,
    buffers, architecture/config and cache provenance are checked before reuse.
    """
    VERSION = 'bamg-frozen-reference-v1'

    def __init__(self, mask_model, weight_model, *, k=4, l1_cap=8., mg_config=None,
                 caps=None, support_mode='learned'):
        if support_mode not in ('learned', 'fixed_bilinear'):
            raise ValueError('support_mode must be learned or fixed_bilinear')
        if not isinstance(k, int) or not 1 <= k <= 16:
            raise ValueError('k must be an integer in [1,16]')
        self.mask_model = deepcopy(mask_model).eval().requires_grad_(False)
        self.weight_model = deepcopy(weight_model).eval().requires_grad_(False)
        self.k, self.l1_cap = k, float(l1_cap)
        self.support_mode = support_mode
        self.mg_config = _config(mg_config)
        self.caps = StructuralCaps() if caps is None else StructuralCaps(**caps) if isinstance(caps, dict) else caps
        self._cache = {}
        self._identity = self._signature()
        self.last_report = {}

    def _signature(self):
        return dict(version=self.VERSION, backend='scipy_fp64_sparse_values_v1',
                    mask=_module_key(self.mask_model),
                    weight=_module_key(self.weight_model), k=self.k, l1_cap=self.l1_cap,
                    support_mode=self.support_mode,
                    mg=asdict(self.mg_config), caps=asdict(self.caps),
                    features=FEATURE_VERSION, candidates=CANDIDATE_VERSION,
                    constraints=CONSTRAINT_VERSION)

    def check_immutable(self):
        for model in (self.mask_model, self.weight_model):
            if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
                raise RuntimeError('Reference models must remain frozen in evaluation mode')
        if self._signature() != self._identity:
            raise RuntimeError('Reference model/config changed; regenerate its mask cache explicitly')

    def clear_cache(self):
        """Explicitly charge a fresh reference setup in cold measurements."""
        self.check_immutable()
        self._cache.clear()

    def _validate_entry(self, entry, shape):
        if mask_digest(entry['masks']) != entry['digest']:
            raise RuntimeError('Corrupted reference mask cache')
        sh = tuple(shape)
        if len(entry['masks']) != len(entry['columns']):
            raise RuntimeError('Corrupted reference candidate mapping')
        for index, (mask, columns) in enumerate(zip(entry['masks'], entry['columns'])):
            if terminal(sh, self.mg_config.coarsest_n):
                raise RuntimeError('Mask cache depth mismatch')
            cs = next_shape(sh, 'full', self.mg_config.coarsest_n, level_index=index)
            pattern = direct_pattern(sh, cs)
            expected = torch.tensor(np.array(pattern.columns, copy=True), dtype=torch.int64)
            if mask.dtype != torch.bool or mask.shape != expected.shape or not torch.equal(columns, expected):
                raise RuntimeError('Corrupted reference candidate mapping')
            sh = cs
        if not terminal(sh, self.mg_config.coarsest_n):
            raise RuntimeError('Mask cache depth mismatch')

    def cache_key(self, a, shape):
        self.check_immutable()
        spec = dict(reference=self._identity, operator=_operator_key(a), shape=list(shape))
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def masks_for(self, a, shape):
        start = perf_counter()
        original_dtype = str(a.dtype)
        a, shape = _csr(a, shape, self.mg_config)
        key = self.cache_key(a, shape) + ':' + original_dtype
        if key in self._cache:
            entry = self._cache[key]
            self._validate_entry(entry, shape)
            self.last_report = dict(cache_hit=True, cache_key=key, mask_digest=entry['digest'],
                                    support_mode=self.support_mode,
                                    reference_seconds=perf_counter() - start)
            return [mask.clone() for mask in entry['masks']]
        at = SparseTensor.from_scipy(a); sh = shape; masks = []; columns = []; levels = []
        timings = _timings(); index = 0; strategy = get_strategy(self.mg_config.strategy_name)
        with torch.no_grad():
            while not terminal(sh, self.mg_config.coarsest_n):
                cs = next_shape(sh, 'full', self.mg_config.coarsest_n, level_index=index)
                pattern = direct_pattern(sh, cs)
                if self.support_mode == 'fixed_bilinear':
                    # Explicit ablation: geometry supplies edges only. Values
                    # remain direct signed Weight-CNN predictions below.
                    t = perf_counter(); mask = fixed_bilinear_mask(pattern)
                    timings['selection_seconds'] += perf_counter() - t
                else:
                    device, dtype = _placement(self.mask_model)
                    _sync(self.mask_model); t = perf_counter()
                    features = direct_features(at, sh, device=device, dtype=dtype)
                    _sync(self.mask_model); timings['feature_seconds'] += perf_counter() - t
                    t = perf_counter(); scores = self.mask_model(features)
                    _sync(self.mask_model); timings['mask_cnn_seconds'] += perf_counter() - t
                    t = perf_counter(); mask = select_mask(scores, pattern, k=self.k).detach().cpu().bool()
                    _sync(self.mask_model); timings['selection_seconds'] += perf_counter() - t
                p, _ = _weights(at, sh, pattern, mask, self.weight_model, self.l1_cap, timings)
                level = TLevel(at, sh, index, strategy, p=p)
                levels.append(level)
                t = perf_counter(); at = p.transpose().product(at.product(p))
                timings['galerkin_seconds'] += perf_counter() - t
                if not bool(torch.isfinite(at.values).all()):
                    raise FloatingPointError('Nonfinite reference Galerkin operator')
                masks.append(mask); columns.append(torch.tensor(np.array(pattern.columns, copy=True), dtype=torch.int64))
                sh = cs; index += 1
                _check_budget(levels + [TLevel(at, sh, index, strategy)], self.caps)
        entry = dict(masks=[m.clone() for m in masks], columns=columns, digest=mask_digest(masks), shape=shape)
        self._cache[key] = entry
        self.last_report = dict(cache_hit=False, cache_key=key, mask_digest=entry['digest'],
                                support_mode=self.support_mode, **timings)
        self.last_report['reference_seconds'] = perf_counter() - start
        return [m.clone() for m in masks]

    def state_dict(self, *, include_cache=False):
        self.check_immutable()
        state = dict(version=self.VERSION, identity=deepcopy(self._identity),
                     mask_spec=deepcopy(self.mask_model.spec), weight_spec=deepcopy(self.weight_model.spec),
                     mask_state={k: v.detach().cpu().clone() for k, v in self.mask_model.state_dict().items()},
                     weight_state={k: v.detach().cpu().clone() for k, v in self.weight_model.state_dict().items()},
                     k=self.k, l1_cap=self.l1_cap, mg_config=asdict(self.mg_config), caps=asdict(self.caps),
                     support_mode=self.support_mode)
        if include_cache:
            state['cache'] = deepcopy(self._cache)
        return state

    @classmethod
    def from_state_dict(cls, state, *, device='cpu'):
        if state.get('version') != cls.VERSION:
            raise ValueError('Unsupported reference checkpoint version')
        models = []
        for name in ('mask', 'weight'):
            values = state[name + '_state']
            dtype = next(v.dtype for v in values.values() if v.is_floating_point())
            model = model_from_spec(state[name + '_spec']).to(device=device, dtype=dtype)
            model.load_state_dict(values); models.append(model)
        result = cls(*models, k=state['k'], l1_cap=state['l1_cap'],
                     mg_config=state['mg_config'], caps=state['caps'], support_mode=state['support_mode'])
        if result._identity != state['identity']:
            raise ValueError('Reference checkpoint identity/specification mismatch')
        result._cache = deepcopy(state.get('cache', {}))
        for entry in result._cache.values():
            result._validate_entry(entry, entry['shape'])
        return result


def build_graph(a, shape, weight_model, masks, mg_config=None, *, l1_cap=8., caps=None):
    """Rebuild numerical P/Ac/line factors for the current theta; masks never change."""
    start = perf_counter(); cfg = _config(mg_config); a, shape = _csr(a, shape, cfg)
    strategy = get_strategy(cfg.strategy_name); timings = _timings(); levels = []
    at = SparseTensor.from_scipy(a); sh = shape; index = 0
    masks = [torch.as_tensor(m).detach().cpu().bool().clone() for m in masks]
    while True:
        level = TLevel(at, sh, index, strategy, raw_scipy=at.numpy())
        level.mg_config = cfg
        if levels:
            levels[-1].coarse = level
        levels.append(level)
        if terminal(sh, cfg.coarsest_n):
            break
        if index >= len(masks):
            raise ValueError('Missing fixed mask for a nonterminal level')
        cs = next_shape(sh, 'full', cfg.coarsest_n, level_index=index)
        pattern = direct_pattern(sh, cs); mask = masks[index]
        p, weights = _weights(at, sh, pattern, mask, weight_model, l1_cap, timings)
        level.p = p; level.pattern = pattern; level.interpolation_weights = weights
        level.fixed_mask = mask; level.r = p.transpose()
        t = perf_counter(); at = level.r.product(at.product(p))
        timings['galerkin_seconds'] += perf_counter() - t
        if not bool(torch.isfinite(at.values).all()):
            raise FloatingPointError('Nonfinite actual Galerkin operator')
        sh = cs; index += 1
        _check_budget(levels + [TLevel(at, sh, index, strategy)], caps)
    if index != len(masks):
        raise ValueError('Unused masks: hierarchy depth differs from reference')
    root = levels[0]; root.mask_digest = mask_digest(masks)
    root.hierarchy_report = _check_budget(levels, caps)
    root.setup_report = dict(**timings, setup_seconds=perf_counter() - start,
                             mask_digest=root.mask_digest, levels=len(levels))
    return root


def graph_cycle(root, x, b):
    """One classical line_alt V-cycle, differentiable in all sparse A/P values."""
    if root.coarse is None:
        return root.a.solve(b)
    cfg = root.mg_config; options = SimpleNamespace(mg=cfg)
    if not torch.is_grad_enabled() and root.a.values.requires_grad:
        # Detached hard-probe mining may visit this graph before its training
        # rollout. Its cached line slices must still retain the A-value graph.
        with torch.enable_grad():
            for direction in ('x', 'y'):
                _line_batches(root, direction)
    for _ in range(cfg.pre_steps):
        x = x + smooth(root, b - root.a.apply(x), options, reverse=False)
    rc = root.r.apply(b - root.a.apply(x))
    ec = graph_cycle(root.coarse, torch.zeros_like(rc), rc)
    x = x + root.p.apply(ec)
    for _ in range(cfg.post_steps):
        x = x + smooth(root, b - root.a.apply(x), options, reverse=True)
    return x


def residual_loss(root, b, *, x0=None, cycles=3, weighting='uniform',
                  custom_weights=None, eps=1e-12, time_weights=None, rate_objective=None):
    """Mean log relative residual of K cycles (uniform by default).

    Optional rate_objective='relative'/'log' minimizes the mean negative adjacent
    progress rate. time_weights are deployment cycle seconds, detached constants;
    autograd/backward timings must not be supplied as deployment observations.
    """
    if not isinstance(cycles, int) or cycles < 1 or not 0 < eps < 1:
        raise ValueError('Positive cycles and epsilon in (0,1) are required')
    if weighting == 'uniform':
        weights = torch.full((cycles,), 1. / cycles, dtype=torch.float64)
    elif weighting == 'terminal':
        weights = torch.zeros(cycles, dtype=torch.float64); weights[-1] = 1.
    elif weighting == 'custom':
        weights = torch.as_tensor(custom_weights, dtype=torch.float64).detach().cpu()
        if weights.shape != (cycles,) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
            raise ValueError('Custom weights must be K finite nonnegative values with positive sum')
        weights = weights / weights.sum()
    else:
        raise ValueError('weighting must be uniform, terminal or custom')
    b = torch.as_tensor(b, dtype=torch.float64, device='cpu')
    vector = b.ndim == 1
    if vector:
        b = b[:, None]
    if b.ndim != 2 or b.shape[0] != root.a.shape[0] or not bool(torch.isfinite(b).all()):
        raise ValueError('Expected finite RHS of shape N or N x probes')
    x = torch.zeros_like(b) if x0 is None else torch.as_tensor(x0, dtype=b.dtype).reshape_as(b)
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Nonfinite initial iterate')
    r0 = torch.linalg.vector_norm(b - root.a.apply(x), dim=0)
    valid = r0 > root.mg_config.absolute_tolerance
    report = dict(history=[], per_probe_history=[], weights=weights.tolist(),
                  valid_probes=int(valid.sum()), skipped_zero=int((~valid).sum()), cycles=cycles,
                  time_weights_detached=time_weights is not None, objective='log_relative_residual')
    if not bool(valid.any()):
        zero = root.a.values.sum() * 0.
        for level in graph_levels(root):
            if level.p is not None:
                zero = zero + level.p.values.sum() * 0.
        report.update(residual=zero, skipped=True)
        return zero, report, x[:, 0] if vector else x
    # Excluded zero/already-converged probes stay at their supplied solution.
    active_x = x[:, valid]; active_b = b[:, valid]; initial = r0[valid]
    norms = [initial]
    for _ in range(cycles):
        active_x = graph_cycle(root, active_x, active_b)
        norm = torch.linalg.vector_norm(active_b - root.a.apply(active_x), dim=0)
        if not bool(torch.isfinite(norm).all()) or bool((norm > root.mg_config.divergence_factor * initial).any()):
            raise FloatingPointError('Nonfinite or divergent raw BAMG residual rollout')
        norms.append(norm)
        ratios = norm / initial
        report['per_probe_history'].append(ratios)
        report['history'].append(ratios.mean())
    q = torch.stack(report['per_probe_history'])
    loss = (weights[:, None] * torch.log(q.clamp_min(eps))).sum(dim=0).mean()
    if time_weights is not None:
        seconds = torch.as_tensor(time_weights, dtype=torch.float64).detach().cpu()
        if seconds.shape != (cycles,) or not bool(torch.isfinite(seconds).all()) or bool((seconds <= 0).any()):
            raise ValueError('Deployment times must be K finite positive seconds')
        if rate_objective not in ('relative', 'log'):
            raise ValueError('Time-weighted training requires rate_objective=relative or log')
        old = torch.stack(norms[:-1]); new = torch.stack(norms[1:])
        threshold = torch.maximum(torch.full_like(initial, root.mg_config.absolute_tolerance),
                                  root.mg_config.tolerance * initial)
        active = old > threshold[None, :]
        rho = new / old.clamp_min(torch.finfo(torch.float64).tiny)
        progress = 1. - rho if rate_objective == 'relative' else -torch.log(rho.clamp_min(eps))
        rates = progress / seconds[:, None]
        aggregate = weights[:, None] * active
        loss = -(rates * aggregate).sum() / aggregate.sum().clamp_min(eps)
        report['objective'] = 'negative_' + rate_objective + '_rate_detached_seconds'
    elif rate_objective is not None:
        raise ValueError('Rate objectives require deployment times')
    x = x.clone(); x[:, valid] = active_x
    report['residual'] = loss; report['skipped'] = False
    return loss, report, x[:, 0] if vector else x


def progress_rates(residuals, cycle_seconds, *, tolerance=0.):
    """Finite adjacent rates; exact-zero terminal log rates remain undefined."""
    if len(residuals) != len(cycle_seconds) + 1:
        raise ValueError('Need initial residual and one residual per timed cycle')
    rho = []; relative = []; logarithmic = []
    for old, new, seconds in zip(residuals[:-1], residuals[1:], cycle_seconds):
        valid = np.isfinite([old, new, seconds]).all() and old > tolerance and new >= 0 and seconds > 0
        ratio = float(new / old) if valid else None
        rho.append(ratio)
        relative.append(float((1. - ratio) / seconds) if ratio is not None else None)
        logarithmic.append(float(-np.log(ratio) / seconds) if ratio is not None and ratio > 0 else None)
    total = sum(cycle_seconds)
    trajectory = (float(np.log(residuals[0] / residuals[-1]) / total)
                  if total > 0 and 0 < residuals[-1] and np.isfinite(residuals[-1])
                  and np.isfinite(residuals[0]) and residuals[0] > tolerance else None)
    return dict(rho=rho, relative_rate=relative, log_rate=logarithmic,
                trajectory_log_rate=trajectory)


class RuntimeBank:
    """Prepared CSR hierarchy. No neural execution or numerical setup in cycles."""
    def __init__(self, root, mg_config, setup_report, hierarchy_report):
        self.root = root; self.mg_config = mg_config
        self.setup_report = setup_report; self.hierarchy_report = hierarchy_report
        self.stats = WorkStats()

    def cycle(self, x, b):
        b = np.asarray(b, dtype=np.float64); x = np.asarray(x, dtype=np.float64)
        if b.shape != x.shape or b.ndim not in (1, 2) or b.shape[0] != self.root.a.shape[0]:
            raise ValueError('Cycle needs matching N or N x probes arrays')
        if b.ndim == 2:
            return np.column_stack([classical_cycle(self.root, x[:, j], b[:, j], self.mg_config, self.stats)
                                    for j in range(b.shape[1])])
        return classical_cycle(self.root, x, b, self.mg_config, self.stats)

    def solve(self, b, *, x0=None, max_cycles=None, timeout_seconds=None):
        start = perf_counter(); cfg = self.mg_config
        b = np.asarray(b, dtype=np.float64)
        if b.shape != (self.root.a.shape[0],):
            raise ValueError('Tolerance solve expects one RHS vector')
        x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
        if x.shape != b.shape:
            raise ValueError('Initial iterate shape mismatch')
        limit = cfg.max_cycles if max_cycles is None else int(max_cycles)
        if limit < 0 or (timeout_seconds is not None and timeout_seconds <= 0):
            raise ValueError('Invalid runtime budget')
        initial = stable_norm(b - self.root.a @ x)
        tolerance = max(1e-12, 1e-8 * initial)
        residuals = [initial]; times = []; reason = 'max_cycles'; converged = False
        if not np.isfinite(initial):
            reason = 'nonfinite_initial_residual'
        elif initial <= tolerance:
            reason = 'initial_converged'; converged = True
        else:
            for _ in range(limit):
                if timeout_seconds is not None and perf_counter() - start >= timeout_seconds:
                    reason = 'timeout'; break
                tick = perf_counter()
                try:
                    candidate = self.cycle(x, b)
                    norm = stable_norm(b - self.root.a @ candidate)
                except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    times.append(perf_counter() - tick); residuals.append(float('inf'))
                    reason = 'cycle_error:' + type(exc).__name__; break
                x = candidate; times.append(perf_counter() - tick); residuals.append(norm)
                if not np.isfinite(norm):
                    reason = 'nonfinite_residual'; break
                if norm <= tolerance:
                    converged = True; reason = 'converged'; break
                if norm > cfg.divergence_factor * initial:
                    reason = 'diverged'; break
        elapsed = perf_counter() - start
        return dict(x=x, converged=converged, success=converged, reason=reason,
                    cycles=len(times), residuals=residuals, cycle_seconds=times,
                    warm_seconds=elapsed, solve_seconds=elapsed,
                    setup_seconds=self.setup_report['setup_seconds'],
                    cold_seconds=self.setup_report['setup_seconds'] + elapsed,
                    tolerance=tolerance, fallback_used=False, memory=memory_observation(),
                    **progress_rates(residuals, times, tolerance=tolerance))


def _runtime_from_graph(graph, cfg, setup_report):
    start = perf_counter(); stats = WorkStats(); previous = None; root = None
    for item in graph_levels(graph):
        a = item.a.numpy().copy()
        level = FixedLevel(a, item.shape, item.index, item.strategy,
                           np.maximum(np.abs(a.diagonal()), 1e-14))
        if previous is not None:
            previous.coarse = level
        else:
            root = level
        if item.p is None:
            level.lu = spla.splu(a.tocsc()); stats.coarse_factorizations += 1
            stats.coarse_l_nnz += level.lu.L.nnz; stats.coarse_u_nnz += level.lu.U.nnz
        else:
            level.p = item.p.numpy().copy(); level.r = level.p.T.tocsr()
            level.cache = LineSmootherCache(a, item.shape)
            for direction in ('x', 'y'):
                level.cache.prepare(direction)
            stats.line_factorizations += level.cache.factorization_count
            stats.line_blocks_factorized += level.cache.line_blocks_factorized
        previous = level
    setup_report['factorization_seconds'] = perf_counter() - start
    setup_report['factorizations'] = stats.to_dict()
    hierarchy_report = graph.hierarchy_report
    return RuntimeBank(root, cfg, setup_report, hierarchy_report)


def build_runtime(a, shape, weight_model, reference, mg_config=None, *, l1_cap=None, caps=None):
    start = perf_counter(); cfg = _config(mg_config or reference.mg_config)
    if cfg.coarsest_n != reference.mg_config.coarsest_n:
        raise ValueError('Actual and reference hierarchy depths disagree')
    masks = reference.masks_for(a, shape)
    with torch.no_grad(), _eval(weight_model):
        graph = build_graph(a, shape, weight_model, masks, cfg,
                            l1_cap=reference.l1_cap if l1_cap is None else l1_cap,
                            caps=reference.caps if caps is None else caps)
    report = deepcopy(graph.setup_report)
    report['reference'] = deepcopy(reference.last_report)
    report['reference_seconds'] = reference.last_report['reference_seconds']
    report['reference_cache_hit'] = reference.last_report['cache_hit']
    bank = _runtime_from_graph(graph, cfg, report)
    bank.setup_report['setup_seconds'] = perf_counter() - start
    bank.setup_report['memory'] = memory_observation()
    bank.mask_digest = graph.mask_digest
    return bank


def runtime_from_prolongations(a, shape, prolongations, mg_config=None, *, caps=None):
    """Prepare teacher or baseline P matrices once; preserve their symbolic zeros."""
    start = perf_counter(); cfg = _config(mg_config); a, shape = _csr(a, shape, cfg)
    strategy = get_strategy(cfg.strategy_name); matrices = list(prolongations)
    levels = []; sh = shape; at = SparseTensor.from_scipy(a); index = 0
    while True:
        level = TLevel(at, sh, index, strategy)
        if levels:
            levels[-1].coarse = level
        levels.append(level)
        if terminal(sh, cfg.coarsest_n):
            break
        if index >= len(matrices):
            raise ValueError('Missing baseline prolongation')
        cs = next_shape(sh, 'full', cfg.coarsest_n, level_index=index)
        p = SparseTensor.from_scipy(matrices[index])
        if p.shape != (int(np.prod(sh)), int(np.prod(cs))) or not bool(torch.isfinite(p.values).all()):
            raise ValueError('Invalid baseline prolongation')
        level.p = p; at = p.transpose().product(at.product(p))
        sh = cs; index += 1
        _check_budget(levels + [TLevel(at, sh, index, strategy)], caps)
    if len(matrices) != index:
        raise ValueError('Unexpected extra baseline prolongations')
    graph = levels[0]; graph.hierarchy_report = _check_budget(levels, caps)
    report = _timings(); report['galerkin_seconds'] = perf_counter() - start
    bank = _runtime_from_graph(graph, cfg, report)
    bank.setup_report['setup_seconds'] = perf_counter() - start
    bank.setup_report['memory'] = memory_observation()
    return bank
