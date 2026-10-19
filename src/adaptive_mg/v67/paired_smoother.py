"""A-only generators for distinct cached pre- and post-smoothing operators.

Both heads share a residual CNN, but each smoothing slot applies only its own
operator.  These are two phases of a V-cycle, not a residual-correction cascade.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..models import make_spatial_body
from ..transfer import N_MATRIX_FEATURES


class PairedStencilSmoother(nn.Module):
    """Generate two local stencils and level-wise gains in one CNN evaluation.

Directions act on ``D^{-1} r`` with the existing 9-point stencil ordering
(``OFFSETS_9[0]`` is the center).  Width and weights are shared across levels;
the actual level operator supplies the input features.  Positive scalar gains
do not impose SPD or a convergence guarantee on the resulting sparse matrix.
"""

    training_only = False
    basis_count = 2
    split_direction_gain = True

    def __init__(self, hidden=8, body_kind='residual5', initial_gain=2 / 3,
                 direction_epsilon=1e-8, version=1):
        super().__init__()
        if version != 1:
            raise ValueError('unsupported paired smoother version')
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError('hidden must be a positive integer')
        if not math.isfinite(initial_gain) or initial_gain <= 0:
            raise ValueError('initial_gain must be finite and positive')
        if not math.isfinite(direction_epsilon) or direction_epsilon <= 0:
            raise ValueError('direction_epsilon must be finite and positive')
        self.spec = dict(hidden=hidden, body_kind=body_kind,
                         initial_gain=float(initial_gain),
                         direction_epsilon=float(direction_epsilon), version=1)
        self.body_kind = body_kind
        self.body = make_spatial_body(body_kind, N_MATRIX_FEATURES, hidden)
        self.spatial_convolutions = self.body.spatial_convolutions
        self.receptive_radius = self.body.receptive_radius
        self.stencil_heads = nn.ModuleList([nn.Conv2d(hidden, 9, 1) for _ in range(2)])
        self.gain_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(2)])
        for head in self.stencil_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.bias[0] = 1.
        for head in self.gain_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, initial_gain + math.log(-math.expm1(-initial_gain)))

    def paired_smoother_spec(self):
        return dict(self.spec)

    def paired_directions_and_gains(self, features, index=0):
        """Return ``(batch,2,9,H,W)`` directions and ``(batch,2)`` gains."""
        hidden = self.body(features)
        raw = torch.stack([head(hidden) for head in self.stencil_heads], dim=1)
        values = raw.float() if raw.dtype in (torch.float16, torch.bfloat16) else raw
        norm = values.square().sum(dim=2, keepdim=True).clamp_min(
            self.spec['direction_epsilon'] ** 2).sqrt()
        directions = (values / norm).to(raw.dtype)
        pooled = hidden.mean(dim=(-2, -1))
        gains = torch.cat([F.softplus(head(pooled)) for head in self.gain_heads], dim=1)
        return directions, gains

    def direction_and_gain(self, features):
        """Diagnostic compatibility; solver dispatch uses the paired protocol."""
        return self.paired_directions_and_gains(features)


@dataclass(frozen=True)
class PairedStencilBank:
    """Two generated banks, selected by V-cycle phase without NN inference."""

    pre: object
    post: object
    is_paired_smoother: bool = True

    def for_phase(self, reverse=False):
        return self.post if reverse else self.pre

    def apply(self, residual, stats, *, reverse=False):
        return self.for_phase(reverse).apply(residual, stats)
