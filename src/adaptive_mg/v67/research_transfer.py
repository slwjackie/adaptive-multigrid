"""Sparse graph generators for learned interpolation; never a V-cycle solver.

Only A connectivity, geometry, the C/F split and the selected classical P are
inputs.  Graph support decisions are discrete; sparse A *values* retain their
autograd graph.  Generated deltas use the existing row-sum/injection projection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from ..grid import coarsening_ratios
from ..transfer import TransferPattern, coarse_fine_indices, weights_from_deltas_torch

NODE_FEATURES = 9
EDGE_FEATURES = 9
CANDIDATE_FEATURES = 10


def _lookup(keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Differentiable values on a sorted sparse key vector, zero for absent keys."""
    if keys.numel() == 0:
        return torch.zeros(query.shape, dtype=values.dtype, device=values.device)
    order = torch.argsort(keys)
    ordered = keys[order]
    position = torch.searchsorted(ordered, query)
    bounded = position.clamp(max=keys.numel() - 1)
    hit = (position < keys.numel()) & (ordered[bounded] == query)
    return torch.where(hit, values[order[bounded]], torch.zeros((), dtype=values.dtype, device=values.device))


def _operator_coo(a: Any, *, device: torch.device, dtype: torch.dtype):
    """No detach of SparseTensor values, no dense matrix, normalize before FP32."""
    if sp.issparse(a):
        csr = a.tocsr(copy=True)
        csr.sum_duplicates()
        coo = csr.tocoo(copy=False)
        if not np.isfinite(coo.data).all():
            raise ValueError("Graph transfer requires finite A")
        scale = float(np.max(np.abs(coo.data), initial=0.0))
        normalized = coo.data / max(scale, np.finfo(np.float64).tiny)
        values = torch.as_tensor(normalized, dtype=dtype, device=device)
        rows = torch.as_tensor(coo.row.astype(np.int64), device=device)
        cols = torch.as_tensor(coo.col.astype(np.int64), device=device)
        return rows, cols, values, csr.shape
    # Existing first-order SparseTensor backend exposes fixed symbolic indices.
    if not all(hasattr(a, key) for key in ("rows", "cols", "values", "shape")):
        raise TypeError("A must be SciPy sparse or adaptive_mg.v67 SparseTensor")
    raw = a.values
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("Graph transfer requires finite A")
    scale = raw.abs().max().clamp_min(torch.finfo(raw.dtype).tiny) if raw.numel() else raw.new_tensor(1.0)
    values = (raw / scale).to(device=device, dtype=dtype)
    rows = torch.as_tensor(np.array(a.rows, copy=True), dtype=torch.long, device=device)
    cols = torch.as_tensor(np.array(a.cols, copy=True), dtype=torch.long, device=device)
    return rows, cols, values, a.shape


@dataclass
class TransferGraph:
    nodes: torch.Tensor
    edges: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    edge_weight: torch.Tensor
    candidate_features: torch.Tensor
    candidate_nodes: torch.Tensor
    candidate_valid: torch.Tensor
    candidate_columns: torch.Tensor
    baseline: torch.Tensor
    pattern: TransferPattern


def build_transfer_graph(a: Any, pattern: TransferPattern, baseline=None, *, device="cpu", dtype=torch.float32) -> TransferGraph:
    """Build O(nnz(A) + N K) features; K is the bounded interpolation support."""
    device = torch.device(device)
    rows, cols, values, shape = _operator_coo(a, device=device, dtype=dtype)
    n = pattern.n_fine_unknowns
    if shape != (n, n):
        raise ValueError("A shape and transfer pattern disagree")
    if pattern.n_candidates > 36:
        raise ValueError("Graph transfer support must have at most 36 candidates")
    columns = torch.as_tensor(np.array(pattern.columns, copy=True), dtype=torch.long, device=device)
    if bool(((columns < -1) | (columns >= pattern.n_coarse_unknowns)).any()):
        raise ValueError("Invalid interpolation candidate column")
    valid = columns >= 0
    if isinstance(baseline, np.ndarray):
        baseline = np.array(baseline, copy=True)
    base = torch.as_tensor(np.array(pattern.bilinear_weights, copy=True), dtype=dtype, device=device) if baseline is None else torch.as_tensor(baseline, dtype=dtype, device=device)
    if base.shape != columns.shape or not bool(torch.isfinite(base).all()):
        raise ValueError("Finite baseline weights must match candidate support")
    if bool((base[~valid] != 0).any()):
        raise ValueError("Invalid support slots must have zero baseline weight")
    coarse = torch.as_tensor(coarse_fine_indices(pattern), dtype=torch.long, device=device)
    roles = torch.zeros(n, dtype=dtype, device=device).index_fill(0, coarse, 1.0)
    diagonal = torch.zeros(n, dtype=dtype, device=device).index_add(0, rows[rows == cols], values[rows == cols]).abs().clamp_min(1e-12)
    row_abs = torch.zeros_like(diagonal).index_add(0, rows, values.abs())
    row_sum = torch.zeros_like(diagonal).index_add(0, rows, values)
    degree = torch.zeros_like(diagonal).index_add(0, rows, torch.ones_like(values))
    row_max = torch.zeros_like(diagonal).scatter_reduce(0, rows, values.abs(), reduce="amax", include_self=True).clamp_min(1e-12)
    ny = pattern.fine_shape[1]
    dx = (cols // ny - rows // ny).to(dtype)
    dy = (cols % ny - rows % ny).to(dtype)
    distance = torch.sqrt(dx.square() + dy.square())
    denom = distance.clamp_min(1.0)
    coupling = values / torch.sqrt(diagonal[rows] * diagonal[cols])
    strength = values.abs() / row_max[rows]
    candidate_nodes = coarse[columns.clamp_min(0)]
    fine_rows = torch.arange(n, device=device)[:, None].expand_as(columns)
    candidate_keys = (fine_rows * n + candidate_nodes)[valid]
    edge_keys = rows * n + cols
    in_support = _lookup(candidate_keys, torch.ones(candidate_keys.numel(), dtype=dtype, device=device), edge_keys)
    edges = torch.stack((coupling, coupling.abs(), dx / denom, dy / denom,
                         torch.log1p(distance), strength, roles[rows], roles[cols], in_support), dim=-1)
    moments = []
    for factor in (dx.square(), dx * dy, dy.square()):
        moments.append(torch.zeros_like(diagonal).index_add(0, rows, -values * factor) / row_abs.clamp_min(1e-12))
    nodes = torch.stack((torch.log(diagonal / diagonal.mean().clamp_min(1e-12)),
                         row_sum / diagonal, row_abs / diagonal, degree / 9.0,
                         roles, *moments, row_max / row_abs.clamp_min(1e-12)), dim=-1)
    query = (fine_rows * n + candidate_nodes).reshape(-1)
    candidate_coupling = _lookup(edge_keys, coupling, query).reshape_as(base)
    cdx = (candidate_nodes // ny - fine_rows // ny).to(dtype)
    cdy = (candidate_nodes % ny - fine_rows % ny).to(dtype)
    cdistance = torch.sqrt(cdx.square() + cdy.square())
    cdenom = cdistance.clamp_min(1.0)
    candidate_features = torch.stack((base, base.abs(), valid.to(dtype), (base != 0).to(dtype),
                                     candidate_coupling, candidate_coupling.abs(), cdx / cdenom,
                                     cdy / cdenom, torch.log1p(cdistance), roles[:, None].expand_as(base)), dim=-1)
    candidate_features = candidate_features * valid[..., None]
    return TransferGraph(nodes, edges, rows, cols, values.abs() / row_abs[rows].clamp_min(1e-12),
                         candidate_features, candidate_nodes, valid, columns, base, pattern)


class _MessageLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * width + EDGE_FEATURES, width), nn.SiLU(), nn.Linear(width, width))
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, h, graph):
        messages = self.message(torch.cat((h[graph.rows], h[graph.cols], graph.edges), dim=-1))
        aggregate = torch.zeros_like(h).index_add(0, graph.rows, messages * graph.edge_weight[:, None])
        return self.norm(h + self.update(torch.cat((h, aggregate), dim=-1)))


class GraphTransferNet(nn.Module):
    """A small sparse Student or an explicitly training-only larger Teacher.

    Output slots are computed from candidate identities/geometry, so permutations
    of the candidate channels only permute the output.  Zero-initialized decoder
    gives exactly the selected classical P before any optimizer update.
    """
    def __init__(self, architecture="small_gnn", *, width=None, layers=None, max_delta=0.35, support="standard", complexity_caps=None):
        super().__init__()
        if architecture not in {"small_gnn", "edge_mlp", "gnn_teacher"}:
            raise ValueError(f"Unknown graph transfer architecture: {architecture}")
        teacher = architecture == "gnn_teacher"
        width = (48 if teacher else 24) if width is None else width
        layers = (4 if teacher else 0 if architecture == "edge_mlp" else 2) if layers is None else layers
        if not isinstance(width, int) or width < 4 or width > 128:
            raise ValueError("Graph width must be an integer in [4,128]")
        allowed = {0} if architecture == "edge_mlp" else range(3, 7) if teacher else {1, 2}
        if layers not in allowed:
            raise ValueError("edge_mlp needs 0 layers, Student GNN 1-2, Teacher GNN 3-6")
        if support not in {"standard", "expanded"}:
            raise ValueError("Support must be standard (16) or expanded (36)")
        if not np.isfinite(max_delta) or not 0 < max_delta <= 2:
            raise ValueError("max_delta must be finite in (0,2]")
        self.architecture = architecture
        self.training_only = teacher
        self.width, self.layers, self.max_delta, self.support = width, layers, float(max_delta), support
        caps = TransferComplexityCaps() if complexity_caps is None else TransferComplexityCaps(**complexity_caps) if isinstance(complexity_caps, dict) else complexity_caps
        self.complexity_caps = asdict(caps)
        self.node_encoder = nn.Sequential(nn.Linear(NODE_FEATURES, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.message_layers = nn.ModuleList(_MessageLayer(width) for _ in range(layers))
        self.edge_encoder = nn.Sequential(nn.Linear(2 * width + CANDIDATE_FEATURES, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.delta_head = nn.Linear(width, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def architecture_config(self):
        return dict(architecture=self.architecture, width=self.width, layers=self.layers,
                    max_delta=self.max_delta, support=self.support, complexity_caps=dict(self.complexity_caps), schema=1, training_only=self.training_only)

    def forward(self, *args, **kwargs):
        raise TypeError("GraphTransferNet requires forward_graph(A, pattern, baseline); image-only input loses sparse connectivity")

    def forward_graph(self, a, pattern, baseline=None, *, return_details=False):
        parameter = next(self.parameters())
        graph = build_transfer_graph(a, pattern, baseline, device=parameter.device, dtype=parameter.dtype)
        h = self.node_encoder(graph.nodes)
        for layer in self.message_layers:
            h = layer(h, graph)
        sources = h[:, None, :].expand(-1, pattern.n_candidates, -1)
        edge_hidden = self.edge_encoder(torch.cat((sources, h[graph.candidate_nodes], graph.candidate_features), dim=-1))
        raw = self.max_delta * torch.tanh(self.delta_head(edge_hidden)[..., 0])
        raw = raw * graph.candidate_valid
        nx, ny = pattern.fine_shape
        delta = raw.reshape(nx, ny, pattern.n_candidates).permute(2, 0, 1).unsqueeze(0)
        if not return_details:
            return delta
        # Width-independent edge summaries avoid quadratic pairwise/Gram losses.
        edge_features = torch.stack((edge_hidden.mean(-1), edge_hidden.square().mean(-1).clamp_min(1e-12).sqrt(),
                                     edge_hidden.abs().mean(-1), edge_hidden.amax(-1)), dim=-1)
        return dict(deltas=delta, weights=project_transfer_weights(self, pattern, delta, graph.baseline),
                    edge_features=edge_features, columns=graph.candidate_columns,
                    valid=graph.candidate_valid, training_only=self.training_only)


def make_graph_transfer(architecture="small_gnn", **kwargs):
    schema = kwargs.pop("schema", 1)
    training_only = kwargs.pop("training_only", architecture == "gnn_teacher")
    if schema != 1 or bool(training_only) != (architecture == "gnn_teacher"):
        raise ValueError("Incompatible graph transfer architecture metadata")
    return GraphTransferNet(architecture, **kwargs)


def graph_transfer_deltas(module, a, pattern, baseline):
    return module.forward_graph(a, pattern, baseline)


def expanded_transfer_pattern(pattern: TransferPattern, baseline=None):
    """Optional fixed coordinate support ablation: 6x6 slots, never arbitrary fill.

    The selected classical weights are embedded by column identity; row sums and
    C-node injection survive.  Baseline selection itself remains unchanged.
    """
    nx, ny = pattern.fine_shape
    cx, cy = pattern.coarse_shape
    rx, ry = coarsening_ratios(pattern.fine_shape, pattern.coarse_shape)
    offsets = np.arange(-2, 4)
    ii = np.arange(1, nx + 1)[:, None] // rx + offsets
    jj = np.arange(1, ny + 1)[:, None] // ry + offsets
    valid = ((ii >= 1) & (ii <= cx))[:, None, :, None] & ((jj >= 1) & (jj <= cy))[None, :, None, :]
    columns = np.where(valid, (ii[:, None, :, None] - 1) * cy + jj[None, :, None, :] - 1, -1).reshape(nx * ny, 36)
    geometric = np.zeros((nx * ny, 36))
    embedded = np.zeros_like(geometric)
    base = pattern.bilinear_weights if baseline is None else np.asarray(baseline)
    if base.shape != pattern.columns.shape:
        raise ValueError("baseline shape mismatch")
    # K<=36; O(N K_old K_new) remains bounded, no N x N construction.
    for old in range(pattern.n_candidates):
        matches = (columns == pattern.columns[:, old, None]) & (columns >= 0)
        geometric += matches * pattern.bilinear_weights[:, old, None]
        embedded += matches * base[:, old, None]
    return TransferPattern(pattern.fine_shape, pattern.coarse_shape, columns.astype(np.int32), geometric), embedded


def transfer_pattern_for_model(pattern, module, baseline):
    return expanded_transfer_pattern(pattern, baseline) if getattr(module, "support", "standard") == "expanded" else (pattern, baseline)


def project_transfer_weights(module, pattern, delta, baseline):
    """Existing projection plus bounded expanded-support pruning.

    The default 16-slot support uses the unchanged historical projection. A
    tighter explicit cap or expanded support keeps at most that cap, restores selected
    classical row sums and coarse injection. Training uses a straight-through
    derivative through this discrete support choice; numerical forward values
    are exactly the same pruned values as deployment.
    """
    if isinstance(baseline, np.ndarray):
        baseline = np.array(baseline, copy=True)
    weights = weights_from_deltas_torch(pattern, delta, baseline)
    limit = int(getattr(module, "complexity_caps", {"max_row_nnz": 16})["max_row_nnz"])
    if limit >= pattern.n_candidates:
        return weights
    columns = torch.as_tensor(np.array(pattern.columns, copy=True), dtype=torch.long, device=weights.device)
    valid = columns >= 0
    # Sort ties by column identity, preserving candidate-channel equivariance.
    by_column = torch.argsort(torch.where(valid, columns, torch.full_like(columns, pattern.n_coarse_unknowns)), dim=1, stable=True)
    scores = torch.where(valid, weights.detach().abs(), torch.full_like(weights, -1.0))
    ordered_scores = torch.gather(scores, 1, by_column)
    by_magnitude = torch.argsort(ordered_scores, dim=1, descending=True, stable=True)[:, :limit]
    selected = torch.gather(by_column, 1, by_magnitude)
    keep = torch.zeros_like(valid).scatter(1, selected, True) & valid
    numerical = torch.where(keep, weights, torch.zeros_like(weights))
    target = torch.as_tensor(baseline, dtype=weights.dtype, device=weights.device).sum(1, keepdim=True)
    numerical = numerical + (target - numerical.sum(1, keepdim=True)) * keep / keep.sum(1, keepdim=True).clamp_min(1)
    coarse = torch.as_tensor(coarse_fine_indices(pattern), dtype=torch.long, device=weights.device)
    inject = torch.zeros(weights.shape[0], dtype=torch.bool, device=weights.device).index_fill(0, coarse, True)
    base = torch.as_tensor(baseline, dtype=weights.dtype, device=weights.device)
    numerical = torch.where(inject[:, None], base, numerical)
    return weights + (numerical - weights).detach() if torch.is_grad_enabled() else numerical


def transfer_distillation_loss(student, teacher, *, weight_coefficient=1.0, feature_coefficient=0.05):
    """Normalized interpolation-weight KD plus optional local edge-feature KD.

    Teacher targets are always detached.  This is an auxiliary loss; full-cycle
    task training is still necessary and validation decides whether KD is used.
    """
    if not torch.equal(student["columns"], teacher["columns"].to(student["columns"].device)):
        raise ValueError("Transfer KD requires identical ordered candidate support")
    valid = student["valid"]
    if not bool(valid.any()):
        raise ValueError("Transfer KD needs valid interpolation candidates")
    sw = student["weights"]
    tw = teacher["weights"].detach().to(sw)
    norm = tw.abs().sum(-1, keepdim=True).clamp_min(1e-6)
    weight_loss = ((sw - tw) / norm)[valid].square().mean()
    se = student["edge_features"]
    te = teacher["edge_features"].detach().to(se)
    feature_loss = F.smooth_l1_loss(se[valid], te[valid])
    return weight_coefficient * weight_loss + feature_coefficient * feature_loss


@dataclass(frozen=True)
class TransferComplexityCaps:
    max_row_nnz: int = 16
    max_p_ratio: float = 4.0
    max_ac_ratio: float = 3.0
    max_operator_complexity: float = 3.0

    def __post_init__(self):
        if isinstance(self.max_row_nnz, bool) or not isinstance(self.max_row_nnz, int) or self.max_row_nnz <= 0 or not all(np.isfinite(x) and x >= 1 for x in (self.max_p_ratio, self.max_ac_ratio, self.max_operator_complexity)):
            raise ValueError("Invalid transfer complexity caps")


class TransferComplexityError(ValueError):
    """Caller should charge the failed setup and fall back to classical P."""
    def __init__(self, reasons, report):
        self.report = report
        super().__init__("Transfer complexity cap exceeded: " + ", ".join(reasons))


def _actual_csr(matrix):
    if not sp.issparse(matrix):
        raise TypeError("Complexity checks require sparse numerical matrices")
    result = matrix.tocsr(copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    if not np.isfinite(result.data).all():
        raise ValueError("Nonfinite sparse operator")
    return result


def enforce_transfer_complexity(a, p, ac, *, baseline_p=None, baseline_ac=None, caps=None):
    """Measure actual numerical nnz after zero elimination, then reject excess.

    operator_complexity here is the two-level (nnz(A)+nnz(Ac))/nnz(A)
    quantity.  A hierarchy caller must additionally check its aggregate budget.
    """
    caps = TransferComplexityCaps() if caps is None else TransferComplexityCaps(**caps) if isinstance(caps, dict) else caps
    a, p, ac = (_actual_csr(x) for x in (a, p, ac))
    if a.shape[0] != a.shape[1] or p.shape[0] != a.shape[0] or ac.shape != (p.shape[1], p.shape[1]):
        raise ValueError("Incompatible A/P/Ac dimensions")
    report = dict(a_nnz=int(a.nnz), p_nnz=int(p.nnz), ac_nnz=int(ac.nnz),
                  p_row_nnz_max=int(np.max(np.diff(p.indptr), initial=0)),
                  operator_complexity=(a.nnz + ac.nnz) / max(a.nnz, 1), caps=asdict(caps))
    reasons = []
    if report["p_row_nnz_max"] > caps.max_row_nnz:
        reasons.append("max_row_nnz")
    if report["operator_complexity"] > caps.max_operator_complexity:
        reasons.append("max_operator_complexity")
    for key, candidate, baseline, limit in (("p_ratio", p, baseline_p, caps.max_p_ratio), ("ac_ratio", ac, baseline_ac, caps.max_ac_ratio)):
        if baseline is not None:
            reference = _actual_csr(baseline)
            if candidate.shape != reference.shape:
                raise ValueError("Complexity reference shape mismatch")
            report[key] = candidate.nnz / max(reference.nnz, 1)
            if report[key] > limit:
                reasons.append("max_" + key)
    if reasons:
        raise TransferComplexityError(reasons, report)
    return report


def differentiable_transfer_cost(weights, pattern, *, baseline=None, a=None, temperature=0.02):
    """Smooth support/AP-work proxy, explicitly not measured time or exact fill.

    1-exp(-abs(w)/temperature) approximates active P slots.  Sparse A edges
    weight coarse-pair expansion by the soft row occupancies on both endpoints.
    This uses nnz(A) edge gathers, never an all-pairs coarse matrix.
    """
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    valid = torch.as_tensor(np.array(pattern.columns >= 0, copy=True), device=weights.device)
    if weights.shape != valid.shape:
        raise ValueError("weights shape mismatch")
    active = -torch.expm1(-weights.abs() / temperature) * valid
    row = active.sum(1)
    reference = torch.as_tensor(np.array(pattern.bilinear_weights, copy=True) if baseline is None else baseline, dtype=weights.dtype, device=weights.device)
    denom = (reference != 0).sum().clamp_min(1).to(weights.dtype)
    support = row.sum() / denom
    fill = support
    if a is not None:
        rr, cc, vals, shape = _operator_coo(a, device=weights.device, dtype=weights.dtype)
        if shape != (pattern.n_fine_unknowns, pattern.n_fine_unknowns):
            raise ValueError("A shape mismatch")
        base_row = (reference != 0).sum(1).to(weights.dtype)
        actual = (vals.detach() != 0).to(weights.dtype)
        denominator = (base_row[rr] * base_row[cc] * actual).sum().clamp_min(1.0)
        fill = (row[rr] * row[cc] * actual).sum() / denominator
    return dict(support=support, galerkin_work=fill, loss=support + fill)
