"""Single-source detector feature definitions for NumPy and PyTorch paths."""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F


def detector_features_numpy(
    matrix_features: np.ndarray,
    diagonal: np.ndarray,
    residual: np.ndarray,
    post_residual: np.ndarray,
) -> np.ndarray:
    """Build normalized residual, smoothing-failure, and roughness channels."""

    if diagonal.ndim != 2:
        raise ValueError("diagonal must be a 2-D grid")
    shape = diagonal.shape
    z = residual.reshape(shape) / np.maximum(np.abs(diagonal), 1.0e-14)
    z_post = post_residual.reshape(shape) / np.maximum(np.abs(diagonal), 1.0e-14)
    rms = np.sqrt(np.mean(z * z) + 1.0e-12)
    z_scaled = z / rms
    before = ndi.uniform_filter(z * z, size=3, mode="constant", cval=0.0)
    after = ndi.uniform_filter(z_post * z_post, size=3, mode="constant", cval=0.0)
    log_ratio = np.clip(np.log((after + 1.0e-12) / (before + 1.0e-12)), -5.0, 5.0)
    mean = ndi.uniform_filter(z, size=3, mode="constant", cval=0.0)
    roughness = np.clip(np.abs(z - mean) / np.sqrt(before + 1.0e-12), 0.0, 5.0)
    return np.concatenate(
        (
            matrix_features,
            z_scaled[None].astype(np.float32),
            log_ratio[None].astype(np.float32),
            roughness[None].astype(np.float32),
        ),
        axis=0,
    )


def detector_features_torch(
    matrix_features: torch.Tensor,
    diagonal: torch.Tensor,
    residual: torch.Tensor,
    post_residual: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Torch equivalent of :func:`detector_features_numpy`."""

    nx, ny = shape
    diag = torch.clamp(torch.abs(diagonal.reshape(1, 1, nx, ny)), min=1.0e-12)
    z = residual.reshape(1, 1, nx, ny) / diag
    z_post = post_residual.reshape(1, 1, nx, ny) / diag
    z_scaled = z / torch.sqrt(torch.mean(z.square()) + 1.0e-12)
    before = F.avg_pool2d(z.square(), kernel_size=3, stride=1, padding=1)
    after = F.avg_pool2d(z_post.square(), kernel_size=3, stride=1, padding=1)
    log_ratio = torch.log((after + 1.0e-12) / (before + 1.0e-12)).clamp(-5.0, 5.0)
    mean = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
    roughness = (torch.abs(z - mean) / torch.sqrt(before + 1.0e-12)).clamp(0.0, 5.0)
    return torch.cat((matrix_features, z_scaled, log_ratio, roughness), dim=1)
