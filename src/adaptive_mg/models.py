"""Operator-only smoother and one-shot temporal policy (no spatial gates)."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .transfer import N_MATRIX_FEATURES, OFFSETS_9
SPATIAL_BODY_CONVOLUTIONS=5
DEFAULT_SMOOTHER_BASES=2

class ResidualConvBody(nn.Module):
    """Five-layer spatial CNN with identity-initialized residual refinement.

    ``net`` intentionally retains the two-convolution layout and state-dict
    keys used by checkpoint formats 3--7.  The three additional spatial
    convolutions live on residual branches whose final convolutions are zero
    initialized.  Consequently, migrating a shallow checkpoint preserves its
    transfer/smoother function exactly while subsequent training can exploit a
    five-cell receptive radius without BatchNorm or persistent online state.
    """

    spatial_convolutions = SPATIAL_BODY_CONVOLUTIONS
    receptive_radius = SPATIAL_BODY_CONVOLUTIONS

    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        # Keep these indices stable: old checkpoints store body.net.0/2.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.refine_activation = nn.GELU()
        self.refine = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        # Zeroing the final operator of each branch makes both additions exact
        # identities at initialization and during legacy checkpoint migration.
        nn.init.dirac_(self.residual[0].weight)
        nn.init.zeros_(self.residual[0].bias)
        nn.init.zeros_(self.residual[2].weight)
        nn.init.zeros_(self.residual[2].bias)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.net(x)
        hidden = hidden + self.residual(hidden)
        return hidden + self.refine(self.refine_activation(hidden))


class CompactConvBody(nn.Module):
    """Three spatial convolutions for distillation, with an optional DW/PW body.

    ``spatial_convolutions`` counts receptive-field-expanding convolutions; the
    depthwise variant also has two 1x1 pointwise mixing layers.
    """
    spatial_convolutions = 3
    receptive_radius = 3

    def __init__(self, in_channels: int, hidden: int, *, depthwise: bool = False):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU()]
        for _ in range(2):
            if depthwise:
                layers.extend([nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                               nn.Conv2d(hidden, hidden, 1), nn.GELU()])
            else:
                layers.extend([nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_spatial_body(kind: str, in_channels: int, hidden: int) -> nn.Module:
    if hidden < 1:
        raise ValueError("hidden width must be positive")
    if kind == "residual5":
        return ResidualConvBody(in_channels, hidden)
    if kind in {"compact3", "depthwise3"}:
        return CompactConvBody(in_channels, hidden, depthwise=kind == "depthwise3")
    raise ValueError(f"unknown spatial body: {kind}")


class NeuralSmootherNet(nn.Module):
    """Generate operator-conditioned residual-stencil correction bases.

    The network is evaluated from matrix features only.  Its output can
    therefore be cached for an unchanged multigrid level and applied to many
    residuals without re-running the convolutional body.
    """

    spatial_convolutions = ResidualConvBody.spatial_convolutions
    receptive_radius = ResidualConvBody.receptive_radius

    def __init__(
        self,
        in_channels: int = N_MATRIX_FEATURES,
        hidden: int = 32,
        coefficient_scale: float = 0.75,
        basis_count: int = DEFAULT_SMOOTHER_BASES,
        split_direction_gain: bool = False,
        direction_epsilon: float = 1.0e-8,
        body_kind: str = "residual5",
    ) -> None:
        super().__init__()
        if basis_count < 1:
            raise ValueError("basis_count must be positive")
        self.coefficient_scale = float(coefficient_scale)
        self.basis_count = int(basis_count)
        self.split_direction_gain = bool(split_direction_gain)
        self.direction_epsilon = float(direction_epsilon)
        if self.direction_epsilon <= 0.0:
            raise ValueError("direction_epsilon must be positive")
        self.body_kind = body_kind
        self.body = make_spatial_body(body_kind, in_channels, hidden)
        self.spatial_convolutions = self.body.spatial_convolutions
        self.receptive_radius = self.body.receptive_radius
        self.head = nn.Conv2d(hidden, 9 * self.basis_count, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Basis zero is the production fast path, so a newly-created model must
        # contribute exactly zero there.  This gives the runtime a truthful,
        # deterministic uncalibrated state (and lets a classical/Jacobi base be
        # the complete zero-init behavior).  A small seed is retained only for
        # the secondary subspace basis to avoid an entirely rank-zero neural
        # candidate block during early subspace training.
        if self.basis_count >= 2:
            with torch.no_grad():
                eps = 1.0e-3
                # OFFSETS_9 is ordered as a 3x3 stencil.  The secondary seed
                # resembles a y-second difference and is intentionally tiny
                # compared with the learned coefficient range.
                self.head.bias[9:18].copy_(
                    torch.tensor(
                        [-2.0 * eps, 0.0, 0.0, eps, eps, 0.0, 0.0, 0.0, 0.0],
                        dtype=self.head.bias.dtype,
                    )
                )

        if self.split_direction_gain:
            self.gain_head = nn.Linear(hidden, self.basis_count)
            nn.init.zeros_(self.gain_head.weight)
            nn.init.constant_(self.gain_head.bias, math.log(math.expm1(0.72)))
            # A nonzero direction avoids the singular derivative of an all-zero
            # normalization. OFFSETS_9[0] is the central stencil coefficient.
            with torch.no_grad():
                self.head.bias.zero_()
                self.head.bias[0] = 1.0
                if self.basis_count > 1:
                    self.head.bias[9:18].copy_(torch.tensor(
                        [1., -.25, -.25, -.25, -.25, 0., 0., 0., 0.],
                        dtype=self.head.bias.dtype))

    def _directions_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.head(hidden)
        batch, _, nx, ny = raw.shape
        raw = raw.reshape(batch, self.basis_count, 9, nx, ny)
        if not self.split_direction_gain:
            return self.coefficient_scale * torch.tanh(raw)
        # Normalize in FP32 when the neural body runs in BF16/FP16.
        values = raw.float() if raw.dtype in {torch.float16, torch.bfloat16} else raw
        norm = values.square().sum(dim=2, keepdim=True).add(
            self.direction_epsilon).sqrt()
        return (values / norm).to(raw.dtype)

    def _gains_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.split_direction_gain:
            return torch.ones((hidden.shape[0], self.basis_count),
                              device=hidden.device, dtype=hidden.dtype)
        raw = self.gain_head(hidden.mean(dim=(-2, -1)))
        # Strictly positive; alpha<=0 targets are masked during log-gain fitting.
        return F.softplus(raw) + 1.0e-8

    def direction_and_gain(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One shared encoder evaluation; gains are scalar per sample and level.

        Gains are dimensionless. The PDE adapter divides them by the level's
        mean |diag(A)| when the stencil input is the raw residual. A uniformly
        scaled operator therefore has exactly the corresponding inverse gain.
        """
        hidden = self.body(features)
        return self._directions_from_hidden(hidden), self._gains_from_hidden(hidden)

    def coefficients(self, features: torch.Tensor) -> torch.Tensor:
        """Return normalized directions, NOT gain-multiplied stencil fields."""
        return self._directions_from_hidden(self.body(features))

    def gains(self, features: torch.Tensor) -> torch.Tensor:
        return self._gains_from_hidden(self.body(features))

    @staticmethod
    def apply_coefficients(
        coefficients: torch.Tensor,
        stencil_input: torch.Tensor,
        *,
        accumulate_fp32: bool = True,
    ) -> torch.Tensor:
        """Apply cached spatially-varying stencils to one residual.

        Returns ``(B,q,nx,ny)``.  FP16/BF16 products accumulate in FP32 by
        default; FP32/FP64 inputs retain their native precision.
        """

        if coefficients.ndim != 5 or coefficients.shape[2] != 9:
            raise ValueError("coefficients must have shape (B,q,9,nx,ny)")
        if stencil_input.ndim != 4 or stencil_input.shape[1] != 1:
            raise ValueError("stencil_input must have shape (B,1,nx,ny)")
        shifted = residual_stencil(stencil_input)
        if coefficients.shape[0] != shifted.shape[0] or coefficients.shape[-2:] != shifted.shape[-2:]:
            raise ValueError("coefficient and residual shapes do not match")
        if accumulate_fp32 and coefficients.dtype in {torch.float16, torch.bfloat16}:
            # Promote both operands before multiplication.  Besides reducing
            # roundoff, this makes eager and torch.compile obey the same
            # low-precision contract instead of allowing backend fusion to
            # change where rounding occurs.
            return (
                coefficients.float() * shifted.unsqueeze(1).float()
            ).sum(dim=2)
        product = coefficients * shifted.unsqueeze(1)
        return product.sum(dim=2)

    def forward(
        self,
        features: torch.Tensor,
        stencil_input: torch.Tensor,
    ) -> torch.Tensor:
        if stencil_input.ndim != 4 or stencil_input.shape[1] != 1:
            raise ValueError("stencil_input must have shape (B,1,nx,ny)")
        directions, gains = self.direction_and_gain(features)
        return self.apply_coefficients(directions, stencil_input) * gains[:, :, None, None]


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




class SwitchNet(nn.Module):
    """One MLP invocation chooses a prefix length, not a spatial mask.

    Training minimizes expected measured time + instability cost over the
    finite action table. The softmax is used only for offline training; actual
    execution takes exactly one discrete K and never revisits the choice.
    """
    def __init__(self, in_features: int, k_values=(0, 1, 2, 3, 4, 5, 6), hidden: int = 24):
        super().__init__()
        self.k_values = tuple(int(k) for k in k_values)
        if sorted(set(self.k_values)) != list(self.k_values) or not self.k_values or self.k_values[0] != 0:
            raise ValueError('actions must be unique sorted nonnegative K, including zero')
        self.register_buffer('feature_mean', torch.zeros(in_features))
        self.register_buffer('feature_scale', torch.ones(in_features))
        self.body = nn.Sequential(nn.Linear(in_features, hidden), nn.Tanh(), nn.Linear(hidden, len(self.k_values)))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features):
        z = (features - self.feature_mean) / self.feature_scale.clamp_min(1e-4)
        return self.body(z)

    def numpy_weights(self):
        """Compile a tiny fixed CPU MLP once, avoiding per-solve torch dispatch."""
        return [v.detach().double().cpu().numpy().copy() for v in
                (self.feature_mean, self.feature_scale, self.body[0].weight, self.body[0].bias, self.body[2].weight, self.body[2].bias)]


def switch_numpy(weights, features):
    import numpy as np
    mean, scale, w1, b1, w2, b2 = weights
    z = (np.asarray(features, np.float64) - mean) / np.maximum(scale, 1e-4)
    return w2 @ np.tanh(w1 @ z + b1) + b2
