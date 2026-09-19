"""Public diagnostics for detector and strategy decisions."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .detector_features import detector_features_numpy
from .models import NeuralComponents
from .smoothers import LineSmootherCache, SmootherKind, classical_smoothing_correction
from .strategy import STRATEGIES, global_context_tensor
from .transfer import matrix_feature_array


def detector_map(
    a: sp.csr_matrix,
    residual: np.ndarray,
    n: int,
    components: NeuralComponents,
    *,
    device: str = "cpu",
    hard: bool = False,
    threshold: float = 0.5,
    smoother: SmootherKind = "line_alt",
    jacobi_omega: float = 0.72,
    reverse: bool = False,
) -> np.ndarray:
    """Return the learned local neural-branch gate as an ``(n,n)`` array."""

    a = a.tocsr()
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if a.shape != (n * n, n * n) or residual.shape != (n * n,):
        raise ValueError("matrix/residual shape does not match n")
    matrix_features = matrix_feature_array(a, n)
    cache = None if smoother == "jacobi" else LineSmootherCache(a, (n, n))
    classical = classical_smoothing_correction(
        a,
        (n, n),
        residual,
        smoother,
        omega=jacobi_omega,
        cache=cache,
        reverse=reverse,
    )
    post = residual - np.asarray(a @ classical).reshape(-1)
    features = detector_features_numpy(
        matrix_features,
        a.diagonal().reshape(n, n),
        residual,
        post,
    )
    components.to(device).eval()
    with torch.no_grad():
        gate = components.detector(
            torch.from_numpy(features).unsqueeze(0).to(device)
        ).cpu().numpy()[0, 0]
    if hard:
        gate = (gate >= threshold).astype(np.float64)
    return gate.astype(np.float64)


def selected_strategy(
    a: sp.csr_matrix,
    n: int,
    components: NeuralComponents,
    *,
    device: str = "cpu",
) -> tuple[str, np.ndarray]:
    """Return selector choice and strategy probabilities."""

    components.to(device).eval()
    with torch.no_grad():
        logits = components.selector(global_context_tensor(a, n, device=device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return STRATEGIES[int(np.argmax(probabilities))].name, probabilities
