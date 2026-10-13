"""Independent operator-conditioned CNNs for explicit multigrid levels.

Each module sees its own current Galerkin matrix. The parameters are independent;
the fixed-P training objective may still optimize their combined V-cycle jointly.
Unknown levels fail explicitly instead of borrowing a differently trained CNN.
"""
from __future__ import annotations

from torch import nn

from ..models import NeuralSmootherNet


class PerLevelSmoother(nn.Module):
    def __init__(self, levels, hidden=16, body_kind='residual5'):
        super().__init__()
        if (not isinstance(levels, (tuple, list)) or not levels
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in levels)
                or len(set(levels)) != len(levels)):
            raise ValueError('per-level CNNs require distinct nonnegative level indices')
        self.level_indices = tuple(sorted(levels))
        self.hidden = hidden
        self.body_kind = body_kind
        self.networks = nn.ModuleDict({str(i): NeuralSmootherNet(
            hidden=hidden, body_kind=body_kind, split_direction_gain=True)
            for i in self.level_indices})

    def for_level(self, level_index):
        key = str(level_index)
        if key not in self.networks:
            raise ValueError(f'per-level smoother checkpoint has no CNN for level {level_index}')
        return self.networks[key]

    def levelwise_spec(self):
        return dict(levels=list(self.level_indices), hidden=self.hidden, body_kind=self.body_kind)


def smoother_for_level(smoother, level_index):
    """Shared/legacy networks retain their exact historical inference path."""
    return smoother.for_level(level_index) if isinstance(smoother, PerLevelSmoother) else smoother
