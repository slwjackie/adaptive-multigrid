"""Three-hop A encoder and sparse fine/coarse cross-attention for signed P.

Graph features preserve the legacy *normalization*, not its rectangular image
layout: arbitrary graphs have no fixed nine directional stencil slots. Original
uses log relative diagonal, normalized center, and signed A_ij/d_i edge values.
The opt-in row9/symmetric9 names identify analogues of the legacy feature
experiments (drop center / also change edge normalization); they are not literal
nine-channel graph node arrays or interchangeable CNN checkpoints.

Arrays follow PyG's x/edge_index/edge_attr convention, using native Torch scatter
operations so optional compiled graph extensions are not required. Attention is
only a feature aggregator; a separate unrestricted scalar head predicts P.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from adaptive_mg.v67.autograd_sparse import SparseTensor
from .structure import Structure

FEATURE_MODES = ("original", "row9", "symmetric9")
FEATURE_SCHEMA = "algebraic-graph-v1"


@dataclass
class GraphFeatures:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor


def graph_features(a: SparseTensor, structure: Structure, feature_mode="original"):
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    if a.shape != (structure.shape[0],) * 2:
        raise ValueError("Operator and structure shapes differ")
    values = a.values
    if values.dtype != torch.float64 or values.device.type != "cpu":
        raise ValueError("Neural AMG currently requires CPU float64 values")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite A values")
    rows = torch.as_tensor(a.rows, dtype=torch.long)
    cols = torch.as_tensor(a.cols, dtype=torch.long)
    diagonal = a.diagonal()
    if not bool((diagonal > 0).all()):
        raise ValueError("Graph features require positive diagonal")
    diagonal = diagonal.clamp_min(1e-14)
    logdiag = torch.log(diagonal / torch.quantile(diagonal, .5).clamp_min(1e-14) + 1e-12)
    coarse = torch.zeros(a.shape[0], dtype=values.dtype)
    coarse[torch.as_tensor(structure.coarse)] = 1.
    # The nonzero symbolic pattern is immutable during backward. Stored symbolic
    # zeros from Galerkin products are permitted and retain their derivatives.
    keep = torch.ones(len(rows), dtype=torch.bool) if feature_mode == "original" else rows != cols
    receiver, sender = rows[keep], cols[keep]
    if feature_mode == "symmetric9":
        normalized = (values[keep] / diagonal[receiver].sqrt()) / diagonal[sender].sqrt()
    else:
        normalized = values[keep] / diagonal[receiver]
    node = [logdiag]
    if feature_mode == "original":
        node.append(a.diagonal() / diagonal)
    node.append(coarse)
    x = torch.stack(node, dim=1)
    # PyG convention: edge_index[0] is source, edge_index[1] is destination.
    return GraphFeatures(x, torch.stack((sender, receiver)), normalized[:, None])


def segment_softmax(scores, rows, n):
    """Sparse row softmax, including empty rows, without an N x Nc array."""
    if scores.numel() == 0:
        return scores
    maxima = scores.new_full((n, scores.shape[1]), -torch.inf)
    indices = rows[:, None].expand_as(scores)
    maxima = maxima.scatter_reduce(0, indices, scores, reduce="amax", include_self=True)
    numerator = torch.exp(scores - maxima[rows])
    denominator = scores.new_zeros((n, scores.shape[1])).index_add(0, rows, numerator)
    return numerator / denominator[rows].clamp_min(torch.finfo(scores.dtype).tiny)


class AttentionLayer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads, self.width = heads, hidden // heads
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.edge_score = nn.Linear(1, heads, bias=False)
        self.edge_value = nn.Linear(1, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden)
        self.activation = nn.SiLU()

    def forward(self, target, source, receiver, sender, edges):
        n = len(target)
        q = self.query(target).reshape(n, self.heads, self.width)
        k = self.key(source).reshape(len(source), self.heads, self.width)
        v = self.value(source).reshape(len(source), self.heads, self.width)
        scores = (q[receiver] * k[sender]).sum(-1) / math.sqrt(self.width)
        scores = scores + self.edge_score(edges)
        alpha = segment_softmax(scores, receiver, n)
        messages = v[sender] + self.edge_value(edges).reshape(-1, self.heads, self.width)
        aggregate = target.new_zeros((n, self.heads, self.width)).index_add(
            0, receiver, alpha[..., None] * messages).reshape(n, -1)
        return self.activation(target + self.output(aggregate))


def constrain_weights(raw, structure, l1_cap=8.):
    """Enforce reference row sums, injection and an optional signed L1 cap.

    The cap uses contraction toward the minimum-L1 uniform row, a piecewise
    differentiable feasible map (not an asserted Euclidean projection).
    """
    if raw.ndim != 1 or len(raw) != len(structure.rows):
        raise ValueError("Expected one scalar per support edge")
    rows = torch.as_tensor(structure.rows, dtype=torch.long, device=raw.device)
    n = structure.shape[0]
    target = torch.as_tensor(structure.row_targets, dtype=raw.dtype, device=raw.device)
    counts = torch.bincount(rows, minlength=n).to(raw.dtype)
    center = target[rows] / counts[rows]
    mean = raw.new_zeros(n).index_add(0, rows, raw) / counts
    residual = raw - mean[rows]
    values = center + residual
    if l1_cap is not None:
        if not math.isfinite(l1_cap) or l1_cap < max(1., float(target.abs().max())):
            raise ValueError("Infeasible row L1 cap")
        norm = raw.new_zeros(n).index_add(0, rows, values.abs())
        scale = ((l1_cap - target.abs()) / (norm - target.abs()).clamp_min(1e-30)).clamp(max=1.)
        values = center + scale[rows] * residual
    coarse_mask = torch.zeros(n, dtype=torch.bool, device=raw.device)
    coarse_mask[torch.as_tensor(structure.coarse, device=raw.device)] = True
    return torch.where(coarse_mask[rows], torch.ones_like(values), values)


class GraphAttentionTransfer(nn.Module):
    def __init__(self, hidden=16, depth=3, heads=2, feature_mode="original", l1_cap=8.):
        super().__init__()
        if type(hidden) is not int or hidden < 1 or type(depth) is not int or depth < 1:
            raise ValueError("hidden and depth must be positive integers")
        if type(heads) is not int or heads < 1 or hidden % heads:
            raise ValueError("hidden must be divisible by positive heads")
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")
        self.hidden, self.depth, self.heads = hidden, depth, heads
        self.feature_mode, self.l1_cap = feature_mode, l1_cap
        self.embedding = nn.Linear(3 if feature_mode == "original" else 2, hidden)
        self.encoder = nn.ModuleList([AttentionLayer(hidden, heads) for _ in range(depth)])
        self.cross = AttentionLayer(hidden, heads)
        self.edge_head = nn.Sequential(nn.Linear(3 * hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.double()

    @property
    def spec(self):
        return dict(kind="graph_attention_transfer", hidden=self.hidden, depth=self.depth,
                    heads=self.heads, feature_mode=self.feature_mode, l1_cap=self.l1_cap,
                    feature_schema=FEATURE_SCHEMA)

    @classmethod
    def from_spec(cls, spec):
        """Reconstruct only this explicitly versioned architecture schema."""
        expected = {"kind", "hidden", "depth", "heads", "feature_mode", "l1_cap", "feature_schema"}
        if not isinstance(spec, dict) or set(spec) != expected:
            raise ValueError("Incomplete or unknown graph transfer checkpoint specification")
        if spec["kind"] != "graph_attention_transfer" or spec["feature_schema"] != FEATURE_SCHEMA:
            raise ValueError("Unsupported graph transfer kind or feature schema")
        return cls(**{key: spec[key] for key in ("hidden", "depth", "heads", "feature_mode", "l1_cap")})

    def forward(self, a, structure):
        if not isinstance(a, SparseTensor):
            a = SparseTensor.from_scipy(a)
        f = graph_features(a, structure, self.feature_mode)
        h = self.embedding(f.x)
        sender, receiver = f.edge_index
        for layer in self.encoder:
            h = layer(h, h, receiver, sender, f.edge_attr)
        rows = torch.as_tensor(structure.rows, dtype=torch.long)
        cols = torch.as_tensor(structure.cols, dtype=torch.long)
        coarse = h[torch.as_tensor(structure.coarse, dtype=torch.long)]
        # Reference interpolation weights describe even distance-two edges where
        # no A_ij exists. They are deterministic inputs, not teacher labels.
        edge = torch.as_tensor(structure.reference.data, dtype=h.dtype)[:, None]
        cross = self.cross(h, coarse, rows, cols, edge)
        raw = self.edge_head(torch.cat((h[rows], coarse[cols], cross[rows], edge), dim=1)).squeeze(-1)
        values = constrain_weights(raw, structure, self.l1_cap)
        return SparseTensor.from_coo(structure.rows, structure.cols, values, structure.shape)
