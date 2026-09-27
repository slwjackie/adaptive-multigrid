"""Cached residual-correction cascades, not a reproduction of a paper's CNN.

Every stage applies B_j(A) to the UPDATED residual. No outer Krylov iteration.
The polynomial variant is inspired by small parameterized approximate inverses;
its five-parameter formula is new and is deliberately not labelled Weymouth's.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn

from ..hierarchy import matvec
from .research_smoothers import make_research_smoother


class TinyPolynomialStage(nn.Module):
    """Five dimensionless parameters; zero off-diagonals start at 0.72 Jacobi.

    Input is the existing row-diagonal-normalized A stencil. The resulting
    coefficients act on D^{-1}r, just like the current generator contract.
    This bounded formula does not assert SPD or MG convergence on arbitrary A.
    """
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(5))

    def direction_and_gain(self, features):
        q = features[:, 1:10]
        mask = torch.ones_like(q)
        mask[:, 0] = 0
        q = q * mask
        p = torch.tanh(self.theta)
        off = .25 * torch.tanh(p[1]*q + p[2]*q*q.abs() + p[3]*q**3
                               + p[4]*q*q.abs().sum(1, keepdim=True)) * mask
        center = torch.zeros_like(off)
        center[:, 0] = .72 + .5*p[0]
        coefficients = center + off
        gain = features.new_ones((features.shape[0], 1))
        return coefficients[:, None], gain


class MultiStageSmoother(nn.Module):
    training_only = False
    basis_count = 1
    split_direction_gain = True

    def __init__(self, kind='cascade_cnn', stages=2, hidden=16, level_count=1, version=1):
        super().__init__()
        if version != 1 or kind not in {'cascade_cnn', 'tiny_polynomial'}:
            raise ValueError('invalid multistage kind/version')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (stages, hidden, level_count)):
            raise ValueError('positive integer stages/hidden/level_count required')
        if stages > 3 or level_count > 8:
            raise ValueError('at most three stages and eight level experts')
        self.spec = dict(kind=kind, stages=stages, hidden=hidden, level_count=level_count, version=1)
        self.levels = nn.ModuleList([
            nn.ModuleList([TinyPolynomialStage() if kind == 'tiny_polynomial' else
                           make_research_smoother('student_cnn', hidden=hidden)
                           for _ in range(stages)]) for _ in range(level_count)])

    def multistage_spec(self):
        return dict(self.spec)

    def expert_index(self, level):
        # The last expert is shared for deeper enabled levels; not a size lookup.
        return min(int(level), len(self.levels)-1)

    def stage_directions_and_gains(self, features, level=0):
        outputs = [stage.direction_and_gain(features) for stage in self.levels[self.expert_index(level)]]
        return torch.cat([v[0] for v in outputs], 1), torch.cat([v[1] for v in outputs], 1)

    def direction_and_gain(self, features):
        # Generic diagnostics can inspect stage kernels; solver dispatch uses the
        # stage_directions_and_gains protocol and never treats them as a subspace.
        return self.stage_directions_and_gains(features, 0)

    def train_only_level(self, level):
        index = self.expert_index(level) if level is not None else None
        for i, expert in enumerate(self.levels):
            for parameter in expert.parameters():
                parameter.requires_grad_(index is None or i == index)


def make_multistage(**spec):
    return MultiStageSmoother(**spec)


@dataclass
class ResidualCascadeBank:
    stages: tuple
    a: object
    is_residual_cascade: bool = True
    native: object = None

    def apply(self, residual, stats):
        residual = np.asarray(residual, np.float64)
        d = np.zeros_like(residual)
        current = residual
        stats.multistage_applications += 1
        for index, bank in enumerate(self.stages):
            increment = bank.apply(current, stats)
            d += increment
            stats.multistage_local_applications += 1
            if index + 1 < len(self.stages):
                current = current - matvec(self.a, increment, stats)
                stats.multistage_residual_matvecs += 1
        return d
