"""Shared numerical scaling helpers."""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F


# NumPy and PyTorch must use the same numerical contract.  The dtype-aware
# helpers lift the analytical floor only when a low-precision dtype cannot
# represent it (notably IEEE FP16).
DIAG_FLOOR = 1.0e-14
FEATURE_EPS = 1.0e-12


def _numpy_floor(dtype: np.dtype, value: float) -> float:
    resolved = np.dtype(dtype)
    if not np.issubdtype(resolved, np.floating):
        return float(value)
    return max(float(value), float(np.finfo(resolved).tiny))


def _torch_floor(dtype: torch.dtype, value: float) -> float:
    if not dtype.is_floating_point:
        return float(value)
    return max(float(value), float(torch.finfo(dtype).tiny))


def clamp_abs_diagonal_numpy(diagonal: np.ndarray) -> np.ndarray:
    """Return ``abs(diagonal)`` with the shared, dtype-safe floor."""

    values = np.asarray(diagonal)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    return np.maximum(np.abs(values), _numpy_floor(values.dtype, DIAG_FLOOR))


def clamp_abs_diagonal_torch(diagonal: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of :func:`clamp_abs_diagonal_numpy`."""

    if not diagonal.dtype.is_floating_point:
        diagonal = diagonal.to(dtype=torch.get_default_dtype())
    return torch.clamp(
        torch.abs(diagonal), min=_torch_floor(diagonal.dtype, DIAG_FLOOR)
    )

