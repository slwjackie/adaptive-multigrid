"""Neural modules for global selection and local multigrid corrections."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import torch
from torch import nn

from .strategy import (
    DEFAULT_STRATEGY_NAME,
    N_GLOBAL_FEATURES,
    STRATEGIES,
    strategy_index,
)
from .transfer import MAX_TRANSFER_CANDIDATES, N_MATRIX_FEATURES, OFFSETS_9

CHECKPOINT_FORMAT_VERSION = 3
DETECTOR_FEATURES = N_MATRIX_FEATURES + 3


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransferNet(nn.Module):
    """Predict corrections over up to 16 candidate coarse neighbours."""

    def __init__(
        self,
        in_channels: int = N_MATRIX_FEATURES,
        hidden: int = 32,
        delta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.body = ConvBlock(in_channels, hidden)
        self.head = nn.Conv2d(hidden, MAX_TRANSFER_CANDIDATES, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.delta_scale * torch.tanh(self.head(self.body(features)))


class NeuralSmootherNet(nn.Module):
    """Zero-initialized additive correction on top of a classical smoother."""

    def __init__(
        self,
        in_channels: int = N_MATRIX_FEATURES,
        hidden: int = 32,
        coefficient_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.coefficient_scale = float(coefficient_scale)
        self.body = ConvBlock(in_channels, hidden)
        self.head = nn.Conv2d(hidden, 9, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def coefficients(self, features: torch.Tensor) -> torch.Tensor:
        return self.coefficient_scale * torch.tanh(self.head(self.body(features)))

    def forward(
        self,
        features: torch.Tensor,
        normalized_residual: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_residual.ndim != 4 or normalized_residual.shape[1] != 1:
            raise ValueError("normalized_residual must have shape (B,1,nx,ny)")
        coefficients = self.coefficients(features)
        shifted_residual = residual_stencil(normalized_residual)
        return torch.sum(coefficients * shifted_residual, dim=1, keepdim=True)


class DetectorNet(nn.Module):
    """Predict a spatial gate in [0,1] for local neural intervention."""

    def __init__(self, in_channels: int = DETECTOR_FEATURES, hidden: int = 24) -> None:
        super().__init__()
        self.body = ConvBlock(in_channels, hidden)
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.body(features)))


class GlobalSelectorNet(nn.Module):
    """Select one controlled classical MG strategy from global operator context."""

    def __init__(
        self,
        in_features: int = N_GLOBAL_FEATURES,
        hidden: int = 32,
        number_of_strategies: int = len(STRATEGIES),
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, number_of_strategies)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -1.0)
        with torch.no_grad():
            self.head.bias[strategy_index(DEFAULT_STRATEGY_NAME)] = 1.0

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError("global context must have shape (B,F)")
        return self.head(self.body(context))


def _shift_zero(x: torch.Tensor, di: int, dj: int) -> torch.Tensor:
    """Return x(i+di,j+dj) at output point (i,j), zero outside."""

    _, _, n_i, n_j = x.shape
    out = torch.zeros_like(x)
    src_i0, src_i1 = max(0, di), min(n_i, n_i + di)
    src_j0, src_j1 = max(0, dj), min(n_j, n_j + dj)
    dst_i0, dst_j0 = max(0, -di), max(0, -dj)
    dst_i1 = dst_i0 + max(0, src_i1 - src_i0)
    dst_j1 = dst_j0 + max(0, src_j1 - src_j0)
    if src_i1 > src_i0 and src_j1 > src_j0:
        out[:, :, dst_i0:dst_i1, dst_j0:dst_j1] = x[
            :, :, src_i0:src_i1, src_j0:src_j1
        ]
    return out


def residual_stencil(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([_shift_zero(x, di, dj) for di, dj in OFFSETS_9], dim=1)


@dataclass
class NeuralComponents:
    transfer: TransferNet
    smoother: NeuralSmootherNet
    detector: DetectorNet
    selector: GlobalSelectorNet

    @classmethod
    def create(
        cls,
        hidden: int = 32,
        *,
        smoother_coefficient_scale: float = 0.75,
    ) -> "NeuralComponents":
        return cls(
            transfer=TransferNet(hidden=hidden),
            smoother=NeuralSmootherNet(
                hidden=hidden, coefficient_scale=smoother_coefficient_scale
            ),
            detector=DetectorNet(hidden=max(16, hidden // 2)),
            selector=GlobalSelectorNet(hidden=max(16, hidden)),
        )

    def modules(self) -> tuple[nn.Module, ...]:
        return (self.transfer, self.smoother, self.detector, self.selector)

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "NeuralComponents":
        for module in self.modules():
            module.to(device=device, dtype=dtype)
        return self

    def eval(self) -> "NeuralComponents":
        for module in self.modules():
            module.eval()
        return self

    def train(self) -> "NeuralComponents":
        for module in self.modules():
            module.train()
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "transfer": self.transfer.state_dict(),
            "smoother": self.smoother.state_dict(),
            "detector": self.detector.state_dict(),
            "selector": self.selector.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        self.transfer.load_state_dict(state["transfer"], strict=strict)
        self.smoother.load_state_dict(state["smoother"], strict=strict)
        self.detector.load_state_dict(state["detector"], strict=strict)
        self.selector.load_state_dict(state["selector"], strict=strict)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "models": self.state_dict(),
                "metadata": metadata or {},
            },
            checkpoint,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cpu",
        hidden: int | None = None,
        allow_legacy: bool = False,
    ) -> tuple["NeuralComponents", dict[str, Any]]:
        checkpoint = Path(path)
        try:
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            warnings.warn(
                "This torch version does not support weights_only; using compatibility loading.",
                stacklevel=2,
            )
            payload = torch.load(checkpoint, map_location=device)
        if not isinstance(payload, dict) or "models" not in payload:
            raise ValueError("invalid neural-MG checkpoint payload")
        version = int(payload.get("format_version", 1))
        if version != CHECKPOINT_FORMAT_VERSION:
            message = (
                f"checkpoint format {version} is incompatible with v3 expanded-transfer/"
                f"global-selector format {CHECKPOINT_FORMAT_VERSION}"
            )
            if not allow_legacy:
                raise ValueError(message + "; retrain or pass allow_legacy=True for inspection")
            raise ValueError(message + "; legacy weights cannot be shape-mapped safely")
        state = payload["models"]
        if hidden is None:
            hidden = int(state["transfer"]["body.net.0.weight"].shape[0])
        metadata_payload = payload.get("metadata", {})
        coefficient_scale = float(
            metadata_payload.get("smoother_coefficient_scale", 0.75)
        )
        dtype_name = str(metadata_payload.get("model_dtype", "float32"))
        dtype = torch.float64 if dtype_name == "float64" else torch.float32
        components = cls.create(
            hidden=hidden, smoother_coefficient_scale=coefficient_scale
        ).to(device, dtype=dtype)
        components.load_state_dict(state)
        metadata = dict(payload.get("metadata", {}))
        metadata.setdefault("checkpoint_format_version", version)
        return components, metadata
