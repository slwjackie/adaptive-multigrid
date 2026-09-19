"""Explicit standalone numerical and temporal-switching contract."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class MGConfig:
    # temporal: deploy only an independently audited policy; predicted: diagnostic
    # use of an unaudited policy; fixed: explicit K ablation; classical: no NN.
    mode: str = "temporal"
    fixed_k: int = 0
    max_nn_cycles: int = 6
    max_cycles: int = 100
    tolerance: float = 1e-8
    absolute_tolerance: float = 1e-12
    residual_reference: str = "initial"
    pre_steps: int = 1
    post_steps: int = 1
    coarsest_n: int = 3
    strategy_name: str = "line_alt_bilinear_full"
    jacobi_omega: float = 0.72
    chebyshev_degree: int = 2
    chebyshev_lower_fraction: float = 0.15
    operator_sweeps: int = 3
    operator_omega: float = 0.70
    operator_blend: float = 0.75
    operator_drop_tolerance: float = 1e-4
    operator_candidate_topk: int = 8
    # Only the first nn_levels nonterminal levels use the extra smoother.
    # -1 means all nonterminal levels. This is static, not spatial selection.
    nn_levels: int = 2
    smoother_gain_multiplier: float = 1.0
    stencil_backend: str = "auto"
    native_threads: int = 1
    native_min_cells: int = 2048
    native_parallel_min: int = 32768
    # Always irreversibly stop neural calls on nonfinite/growing candidates.
    # Growth candidates are rolled back before the classical recovery cycle.
    safety_growth: float = 1.0
    safety_rtol_slack: float = 1e-12
    stagnation_rho: float = 0.98
    stagnation_patience: int = 2
    near_tolerance_factor: float = 10.0
    reserve_classical_cycles: int = 1
    divergence_factor: float = 1e12
    device: str = "cpu"
    verbose: bool = False

    def __post_init__(self):
        if self.mode not in {"classical", "fixed", "temporal", "predicted"}:
            raise ValueError("mode must be classical/fixed/temporal/predicted")
        ints = ("fixed_k", "max_nn_cycles", "max_cycles", "pre_steps", "post_steps",
                "coarsest_n", "nn_levels", "native_threads", "native_min_cells",
                "native_parallel_min", "stagnation_patience", "reserve_classical_cycles")
        if any(isinstance(getattr(self, n), bool) or not isinstance(getattr(self, n), int) for n in ints):
            raise ValueError("cycle counts, levels and thread counts must be integers")
        if self.max_cycles < 1 or self.max_nn_cycles < 0 or not 0 <= self.fixed_k <= self.max_nn_cycles:
            raise ValueError("invalid cycle budget")
        if self.pre_steps < 0 or self.post_steps < 0 or self.pre_steps + self.post_steps < 1:
            raise ValueError("at least one smoothing step required")
        if self.nn_levels < -1 or min(self.native_threads, self.native_min_cells, self.native_parallel_min, self.coarsest_n) < 1:
            raise ValueError("invalid level/thread/kernel configuration")
        if self.reserve_classical_cycles < 1 or self.stagnation_patience < 1:
            raise ValueError("reserve a classical tail and positive stagnation patience")
        if self.stencil_backend not in {"csr", "native", "auto"}:
            raise ValueError("stencil backend must be csr/native/auto")
        if self.residual_reference not in {"initial", "rhs"}:
            raise ValueError("invalid residual reference")
        for k, v in asdict(self).items():
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"{k} must be finite")
        if self.tolerance <= 0 or self.absolute_tolerance < 0:
            raise ValueError("invalid tolerance")
        if self.safety_growth < 1 or self.safety_rtol_slack < 0:
            raise ValueError("invalid growth safeguard")
        if not 0 < self.stagnation_rho <= 1 or self.near_tolerance_factor < 0 or self.divergence_factor <= 1:
            raise ValueError("invalid safety threshold")
        if self.smoother_gain_multiplier < 0:
            raise ValueError("negative gain multiplier")
        if self.device != "cpu":
            raise ValueError("runtime is CPU SciPy/FP64; only offline smoother training may use CUDA")
        from .strategy import get_strategy
        get_strategy(self.strategy_name)

    @property
    def permitted_k(self):
        return min(self.max_nn_cycles, max(0, self.max_cycles - self.reserve_classical_cycles))

    def to_dict(self):
        return asdict(self)

    def numerical_scope(self):
        """Time-policy validity excludes only the action and presentation mode."""
        d = self.to_dict()
        for k in ("mode", "fixed_k", "verbose"):
            d.pop(k)
        return d
