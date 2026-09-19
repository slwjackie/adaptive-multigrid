"""Controlled global classical-strategy selection for adaptive neural MG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import scipy.sparse as sp
import torch

from .grid import CoarseningKind, GridShape, as_shape
from .smoothers import SmootherKind
from .transfer import OFFSETS_9, TransferBaseline, matrix_feature_array


@dataclass(frozen=True)
class MGStrategy:
    name: str
    smoother: SmootherKind
    transfer: TransferBaseline
    coarsening: CoarseningKind


STRATEGIES: Final[tuple[MGStrategy, ...]] = (
    MGStrategy("jacobi_bilinear_full", "jacobi", "bilinear", "full"),
    MGStrategy("line_x_bilinear_full", "line_x", "bilinear", "full"),
    MGStrategy("line_y_bilinear_full", "line_y", "bilinear", "full"),
    MGStrategy("line_alt_bilinear_full", "line_alt", "bilinear", "full"),
    MGStrategy("jacobi_operator_full", "jacobi", "operator", "full"),
    MGStrategy("line_alt_operator_full", "line_alt", "operator", "full"),
    # For strong x coupling, line-x relaxation is paired with y-first
    # semicoarsening; the converse holds for strong y coupling.
    MGStrategy("line_x_operator_semi_y", "line_x", "operator", "semi_y"),
    MGStrategy("line_y_operator_semi_x", "line_y", "operator", "semi_x"),
    MGStrategy("line_alt_operator_alternating", "line_alt", "operator", "alternating"),
)
STRATEGY_BY_NAME: Final[dict[str, MGStrategy]] = {item.name: item for item in STRATEGIES}
DEFAULT_STRATEGY_NAME: Final[str] = "line_alt_bilinear_full"
N_GLOBAL_FEATURES: Final[int] = 18


def strategy_index(name: str) -> int:
    for index, strategy in enumerate(STRATEGIES):
        if strategy.name == name:
            return index
    raise ValueError(f"unknown MG strategy: {name}")


def get_strategy(name: str) -> MGStrategy:
    try:
        return STRATEGY_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown MG strategy: {name}") from exc


def global_context_features(a: sp.csr_matrix, shape: int | GridShape) -> np.ndarray:
    """Return fixed-dimensional operator context for the global selector.

    Features are purely matrix/grid based: normalized stencil magnitudes,
    log-diagonal statistics, directional coupling shares, and grid scales.
    """

    nx, ny = as_shape(shape)
    local = matrix_feature_array(a, (nx, ny))
    stencil_means = np.mean(np.abs(local[1:]), axis=(1, 2)).astype(np.float64)
    log_diag = local[0].astype(np.float64)
    diag_stats = np.array(
        [
            float(np.mean(log_diag)),
            float(np.std(log_diag)),
            float(np.min(log_diag)),
            float(np.max(log_diag)),
        ],
        dtype=np.float64,
    )
    # Channels follow OFFSETS_9; index 0 is the center channel.
    offset_to_channel = {offset: i for i, offset in enumerate(OFFSETS_9)}
    x_strength = sum(
        stencil_means[offset_to_channel[offset]] for offset in ((-1, 0), (1, 0))
    )
    y_strength = sum(
        stencil_means[offset_to_channel[offset]] for offset in ((0, -1), (0, 1))
    )
    diagonal_strength = sum(
        stencil_means[offset_to_channel[offset]]
        for offset in ((-1, -1), (-1, 1), (1, -1), (1, 1))
    )
    total = max(x_strength + y_strength + diagonal_strength, 1.0e-12)
    directional = np.array(
        [x_strength / total, y_strength / total, diagonal_strength / total],
        dtype=np.float64,
    )
    grid = np.array(
        [np.log2(nx + 1.0) / 10.0, np.log2(ny + 1.0) / 10.0],
        dtype=np.float64,
    )
    result = np.concatenate((stencil_means, diag_stats, directional, grid)).astype(np.float32)
    if result.size != N_GLOBAL_FEATURES:
        raise RuntimeError(f"global feature count mismatch: {result.size}")
    return result


def global_context_tensor(
    a: sp.csr_matrix,
    shape: int | GridShape,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.as_tensor(global_context_features(a, shape), device=device, dtype=dtype).unsqueeze(0)
