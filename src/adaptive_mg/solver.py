"""Hierarchical global-selector + local adaptive neural multigrid solver."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .detector_features import detector_features_numpy
from .grid import CoarseningKind, GridShape, next_shape, terminal, unknowns, validate_root_shape
from .krylov import KrylovResult, fcg, fgmres, pcg
from .models import NeuralComponents
from .smoothers import LineSmootherCache, SmootherKind, classical_smoothing_correction
from .strategy import MGStrategy, STRATEGIES, get_strategy, global_context_tensor
from .transfer import (
    TransferBaseline,
    TransferPattern,
    baseline_weights,
    build_transfer_pattern,
    galerkin_coarse_operator,
    matrix_feature_array,
    scipy_prolongation_from_weights,
    weights_from_deltas_torch,
)

Mode = Literal["classical", "nn_smoother", "learned_transfer", "combined", "adaptive"]
OuterSolver = Literal["none", "cg", "fcg", "fgmres"]
GlobalSelectionMode = Literal["fixed", "selector"]


@dataclass
class MGConfig:
    mode: Mode = "classical"
    max_cycles: int = 40
    tolerance: float = 1.0e-8
    pre_steps: int = 1
    post_steps: int = 1
    smoother: SmootherKind = "line_alt"
    jacobi_omega: float = 0.72
    transfer_baseline: TransferBaseline = "bilinear"
    coarsening: CoarseningKind = "full"
    strategy_name: str | None = None
    global_selection: GlobalSelectionMode = "fixed"
    operator_sweeps: int = 3
    operator_omega: float = 0.70
    operator_blend: float = 0.75
    operator_drop_tolerance: float = 1.0e-4
    operator_candidate_topk: int | None = 8
    coarsest_n: int = 3
    device: str = "cpu"
    detector_threshold: float = 0.5
    hard_gate: bool = False
    selective_nn: bool = True
    patch_size: int = 16
    patch_halo: int = 2
    adaptive_refresh: int = 2
    safety_fallback: bool = True
    smoother_safeguard: bool = True
    outer_solver: OuterSolver = "none"
    krylov_restart: int = 20
    preconditioner_cycles: int = 1
    verbose: bool = False


@dataclass
class WorkCounter:
    """Estimated arithmetic work in finest-grid sparse-matvec equivalents."""

    fine_nnz: int
    flops: float = 0.0
    safety_flops: float = 0.0
    matvec_count: int = 0
    transfer_apply_count: int = 0
    smoother_calls: int = 0
    transfer_nn_calls: int = 0
    smoother_nn_calls: int = 0
    detector_nn_calls: int = 0
    selector_nn_calls: int = 0
    nn_seconds: float = 0.0
    hierarchy_rebuilds: int = 0
    active_patches: int = 0
    candidate_patches: int = 0
    active_gate_cells: int = 0
    total_gate_cells: int = 0

    def add_spmv(self, matrix: sp.spmatrix, *, safety: bool = False) -> None:
        self.add_sparse_flops(2.0 * matrix.nnz, safety=safety)
        self.matvec_count += 1

    def add_sparse_flops(self, flops: float, *, safety: bool = False) -> None:
        self.flops += float(flops)
        if safety:
            self.safety_flops += float(flops)

    @property
    def work_units(self) -> float:
        return self.flops / max(2.0 * self.fine_nnz, 1.0)

    @property
    def safety_work_units(self) -> float:
        return self.safety_flops / max(2.0 * self.fine_nnz, 1.0)

    @property
    def nn_calls(self) -> int:
        return (
            self.transfer_nn_calls
            + self.smoother_nn_calls
            + self.detector_nn_calls
            + self.selector_nn_calls
        )

    @property
    def active_patch_fraction(self) -> float:
        return self.active_patches / max(self.candidate_patches, 1)

    @property
    def active_cell_fraction(self) -> float:
        return self.active_gate_cells / max(self.total_gate_cells, 1)


@dataclass
class SolveResult:
    x: np.ndarray
    converged: bool
    cycles: int
    residual_history: list[float]
    elapsed_seconds: float
    setup_seconds: float
    solve_seconds: float
    work_units: float
    safety_work_units: float
    nn_calls: int
    transfer_nn_calls: int
    smoother_nn_calls: int
    detector_nn_calls: int
    selector_nn_calls: int
    nn_seconds: float
    matvec_count: int
    gate_history: list[float] = field(default_factory=list)
    fallback_count: int = 0
    safeguard_alphas: list[float] = field(default_factory=list)
    outer_solver: OuterSolver = "none"
    estimated_residual_history: list[float] = field(default_factory=list)
    residual_iteration_indices: list[int] = field(default_factory=list)
    selected_strategy: str = "custom"
    hierarchy_rebuilds: int = 0
    active_patch_fraction: float = 0.0
    active_cell_fraction: float = 0.0

    @property
    def relative_residual(self) -> float:
        if not self.residual_history:
            return float("nan")
        return self.residual_history[-1] / max(self.residual_history[0], 1.0e-300)

    @property
    def asymptotic_contraction(self) -> float:
        if len(self.residual_history) < 2:
            return float("nan")
        history = np.asarray(self.residual_history, dtype=np.float64)
        indices = (
            np.asarray(self.residual_iteration_indices, dtype=np.int64)
            if len(self.residual_iteration_indices) == history.size
            else np.arange(history.size, dtype=np.int64)
        )
        floor = max(history[0] * 1.0e-10, 1.0e-14)
        gaps = np.diff(indices)
        valid = (history[:-1] > floor) & (gaps > 0)
        if not np.any(valid):
            return float("nan")
        # Normalize each measured reduction by its Krylov/cycle interval so
        # FGMRES restart boundaries are comparable with per-cycle histories.
        log_rates = (
            np.log(np.maximum(history[1:][valid], 1.0e-300))
            - np.log(np.maximum(history[:-1][valid], 1.0e-300))
        ) / gaps[valid]
        tail = log_rates[-min(5, log_rates.size) :]
        return float(np.exp(np.mean(tail)))

    @property
    def safeguard_alpha_mean(self) -> float:
        return float(np.mean(self.safeguard_alphas)) if self.safeguard_alphas else float("nan")

    @property
    def safeguard_active_fraction(self) -> float:
        if not self.safeguard_alphas:
            return 0.0
        return float(np.mean(np.asarray(self.safeguard_alphas) > 1.0e-12))


@dataclass
class _Level:
    a: sp.csr_matrix
    shape: GridShape
    level_index: int
    features: np.ndarray
    pattern: TransferPattern | None
    baseline_weights: np.ndarray | None
    baseline_p: sp.csr_matrix | None
    smoother_cache: LineSmootherCache | None
    learned_deltas: np.ndarray | None = None
    p: sp.csr_matrix | None = None
    coarse: "_Level | None" = None


@dataclass
class _CycleStats:
    gates: list[float] = field(default_factory=list)
    safeguard_alphas: list[float] = field(default_factory=list)
    last_pre_gate: np.ndarray | None = None


@dataclass
class _RunState:
    fallback_count: int = 0
    gate_history: list[float] = field(default_factory=list)
    safeguard_alphas: list[float] = field(default_factory=list)
    cycle_calls: int = 0


def _validate_config(config: MGConfig) -> None:
    if config.max_cycles < 0:
        raise ValueError("max_cycles must be nonnegative")
    if config.pre_steps < 0 or config.post_steps < 0:
        raise ValueError("pre_steps and post_steps must be nonnegative")
    if config.preconditioner_cycles < 1:
        raise ValueError("preconditioner_cycles must be at least one")
    if config.krylov_restart < 1:
        raise ValueError("krylov_restart must be positive")
    if config.adaptive_refresh < 1:
        raise ValueError("adaptive_refresh must be at least one")
    if config.patch_size < 1 or config.patch_halo < 0:
        raise ValueError("invalid selective patch geometry")
    if not 0.0 < config.jacobi_omega < 2.0:
        raise ValueError("jacobi_omega must lie in (0,2)")
    if config.outer_solver == "cg":
        if config.mode not in {"classical", "learned_transfer"}:
            raise ValueError(
                "CG requires a fixed symmetric preconditioner; use FCG or FGMRES "
                "for neural/adaptive smoothing"
            )
        if config.mode != "classical" and config.safety_fallback:
            raise ValueError(
                "CG cannot use residual-dependent cycle fallback; disable "
                "safety_fallback or use FCG/FGMRES"
            )
        if config.pre_steps != config.post_steps:
            raise ValueError("CG requires equal pre- and post-smoothing counts")


def _validate_problem(a: sp.csr_matrix, b: np.ndarray, n: int, config: MGConfig) -> None:
    try:
        validate_root_shape(n, config.coarsest_n)
    except ValueError as exc:
        raise ValueError(
            f"nested structured MG requires n=2**L-1; got n={n}, coarsest_n={config.coarsest_n}"
        ) from exc
    if a.shape != (n * n, n * n):
        raise ValueError("A shape does not match n")
    if b.shape != (n * n,):
        raise ValueError("b shape does not match n")
    if np.any(a.diagonal() <= 0.0):
        raise ValueError("positive diagonal required")
    if config.outer_solver in {"cg", "fcg"}:
        difference = (a - a.T).tocsr()
        error = 0.0 if difference.nnz == 0 else float(np.max(np.abs(difference.data)))
        if error > 1.0e-10:
            raise ValueError("CG/FCG require a symmetric system matrix")


def _torch_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _feature_tensor(features: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(features).unsqueeze(0).to(device=device)


def _spmv(
    a: sp.spmatrix,
    x: np.ndarray,
    work: WorkCounter,
    *,
    safety: bool = False,
) -> np.ndarray:
    work.add_spmv(a, safety=safety)
    return np.asarray(a @ x, dtype=np.float64).reshape(-1)


def _residual(
    a: sp.spmatrix,
    x: np.ndarray,
    b: np.ndarray,
    work: WorkCounter,
    *,
    safety: bool = False,
) -> np.ndarray:
    return np.asarray(b, dtype=np.float64) - _spmv(a, x, work, safety=safety)


def _baseline_kwargs(config: MGConfig) -> dict[str, object]:
    return {
        "operator_sweeps": config.operator_sweeps,
        "operator_omega": config.operator_omega,
        "operator_blend": config.operator_blend,
        "operator_drop_tolerance": config.operator_drop_tolerance,
        "operator_candidate_topk": config.operator_candidate_topk,
    }


def _make_level(a: sp.csr_matrix, shape: GridShape | int, config: MGConfig, level_index: int = 0) -> _Level:
    from .grid import as_shape
    shape = as_shape(shape)
    a = a.tocsr()
    features = matrix_feature_array(a, shape)
    if terminal(shape, config.coarsest_n):
        return _Level(a, shape, level_index, features, None, None, None, None)
    coarse_shape = next_shape(
        shape, config.coarsening, config.coarsest_n, level_index=level_index
    )
    pattern = build_transfer_pattern(shape, coarse_shape)
    base = baseline_weights(
        a,
        shape,
        config.transfer_baseline,
        coarse=coarse_shape,
        **_baseline_kwargs(config),
    )
    p_base = scipy_prolongation_from_weights(pattern, base)
    cache = None if config.smoother == "jacobi" else LineSmootherCache(a, shape)
    return _Level(a, shape, level_index, features, pattern, base, p_base, cache)


def _classical_correction(
    level: _Level,
    residual: np.ndarray,
    config: MGConfig,
    work: WorkCounter,
    *,
    reverse: bool,
    safety: bool = False,
) -> np.ndarray:
    work.smoother_calls += 1
    return classical_smoothing_correction(
        level.a,
        level.shape,
        residual,
        config.smoother,
        omega=config.jacobi_omega,
        cache=level.smoother_cache,
        reverse=reverse,
        work=work,
        safety=safety,
    )


def _tile_ranges(length: int, patch_size: int) -> list[tuple[int, int]]:
    return [(start, min(length, start + patch_size)) for start in range(0, length, patch_size)]


def _selective_forward(
    module: torch.nn.Module,
    features: torch.Tensor,
    gate: np.ndarray,
    config: MGConfig,
    work: WorkCounter,
    *,
    normalized_residual: torch.Tensor | None = None,
    kind: Literal["smoother", "transfer"],
) -> torch.Tensor:
    """Run a convolutional module only on detector-selected tiles."""

    _, _, nx, ny = features.shape
    out_channels = 1 if kind == "smoother" else int(module.head.out_channels)  # type: ignore[attr-defined]
    output = torch.zeros(
        (1, out_channels, nx, ny), device=features.device, dtype=features.dtype
    )
    mask = gate.reshape(nx, ny) >= config.detector_threshold
    x_ranges = _tile_ranges(nx, config.patch_size)
    y_ranges = _tile_ranges(ny, config.patch_size)
    work.candidate_patches += len(x_ranges) * len(y_ranges)
    for i0, i1 in x_ranges:
        for j0, j1 in y_ranges:
            if not bool(np.any(mask[i0:i1, j0:j1])):
                continue
            work.active_patches += 1
            pi0, pi1 = max(0, i0 - config.patch_halo), min(nx, i1 + config.patch_halo)
            pj0, pj1 = max(0, j0 - config.patch_halo), min(ny, j1 + config.patch_halo)
            f_patch = features[:, :, pi0:pi1, pj0:pj1]
            if normalized_residual is None:
                prediction = module(f_patch)
            else:
                prediction = module(
                    f_patch, normalized_residual[:, :, pi0:pi1, pj0:pj1]
                )
            ci0, ci1 = i0 - pi0, i1 - pi0
            cj0, cj1 = j0 - pj0, j1 - pj0
            output[:, :, i0:i1, j0:j1] = prediction[:, :, ci0:ci1, cj0:cj1]
            if kind == "smoother":
                work.smoother_nn_calls += 1
            else:
                work.transfer_nn_calls += 1
    return output


def _detector_gate(
    level: _Level,
    residual: np.ndarray,
    classical: np.ndarray,
    components: NeuralComponents,
    config: MGConfig,
    work: WorkCounter,
) -> tuple[np.ndarray, np.ndarray]:
    diagonal = level.a.diagonal().reshape(level.shape)
    post = residual - _spmv(level.a, classical, work)
    features = detector_features_numpy(
        level.features, diagonal, residual, post
    )
    _torch_sync(config.device)
    start = perf_counter()
    with torch.no_grad():
        gate = components.detector(_feature_tensor(features, config.device))
    _torch_sync(config.device)
    work.nn_seconds += perf_counter() - start
    work.detector_nn_calls += 1
    gate_np = gate.cpu().numpy().astype(np.float64)
    active = gate_np >= config.detector_threshold
    work.active_gate_cells += int(np.count_nonzero(active))
    work.total_gate_cells += int(gate_np.size)
    if config.hard_gate or config.selective_nn:
        gate_np = np.where(active, gate_np if not config.hard_gate else 1.0, 0.0)
    return gate_np, post


def _neural_delta(
    level: _Level,
    residual: np.ndarray,
    components: NeuralComponents,
    config: MGConfig,
    work: WorkCounter,
    *,
    gate: np.ndarray | None = None,
) -> np.ndarray:
    diagonal = np.maximum(np.abs(level.a.diagonal().reshape(level.shape)), 1.0e-14)
    normalized_np = (residual.reshape(level.shape) / diagonal).astype(np.float32)
    features = _feature_tensor(level.features, config.device)
    normalized = torch.from_numpy(normalized_np).reshape(1, 1, *level.shape).to(config.device)
    _torch_sync(config.device)
    start = perf_counter()
    with torch.no_grad():
        if config.selective_nn and gate is not None:
            correction = _selective_forward(
                components.smoother,
                features,
                gate,
                config,
                work,
                normalized_residual=normalized,
                kind="smoother",
            )
        else:
            correction = components.smoother(features, normalized)
            work.smoother_nn_calls += 1
    _torch_sync(config.device)
    work.nn_seconds += perf_counter() - start
    return correction.reshape(-1).cpu().numpy().astype(np.float64)


def _exact_safeguard_correction(
    level: _Level,
    residual: np.ndarray,
    classical: np.ndarray,
    direction: np.ndarray,
    work: WorkCounter,
    *,
    post_classical: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    if float(np.linalg.norm(direction)) <= 1.0e-30:
        return classical, 0.0
    s = post_classical if post_classical is not None else residual - _spmv(level.a, classical, work)
    w = _spmv(level.a, direction, work)
    denominator = float(np.dot(w, w))
    if denominator <= 1.0e-30 or not np.isfinite(denominator):
        return classical, 0.0
    alpha = float(np.clip(np.dot(s, w) / denominator, 0.0, 1.0))
    candidate = classical + alpha * direction
    if (
        not np.all(np.isfinite(candidate))
        or np.linalg.norm(s - alpha * w) > np.linalg.norm(s) * (1.0 + 1.0e-12)
    ):
        return classical, 0.0
    return candidate, alpha


def _smooth(
    level: _Level,
    x: np.ndarray,
    b: np.ndarray,
    components: NeuralComponents,
    config: MGConfig,
    stats: _CycleStats,
    work: WorkCounter,
    *,
    reverse: bool,
    safety: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    residual = _residual(level.a, x, b, work, safety=safety)
    classical = _classical_correction(
        level, residual, config, work, reverse=reverse, safety=safety
    )
    if config.mode in {"classical", "learned_transfer"}:
        return x + classical, None

    gate: np.ndarray | None = None
    post_classical: np.ndarray | None = None
    if config.mode == "adaptive":
        gate, post_classical = _detector_gate(
            level, residual, classical, components, config, work
        )
        stats.gates.append(float(gate.mean()))
        if not reverse:
            stats.last_pre_gate = gate
    elif config.mode not in {"nn_smoother", "combined"}:
        raise ValueError(f"unknown mode: {config.mode}")

    delta = _neural_delta(
        level, residual, components, config, work, gate=gate
    )
    if gate is not None:
        delta = gate.reshape(-1) * delta
    correction = classical + delta
    if config.smoother_safeguard:
        correction, alpha = _exact_safeguard_correction(
            level,
            residual,
            classical,
            delta,
            work,
            post_classical=post_classical,
        )
        stats.safeguard_alphas.append(alpha)
    return x + correction, gate


def _transfer_deltas(
    level: _Level,
    components: NeuralComponents,
    config: MGConfig,
    work: WorkCounter,
    *,
    gate: np.ndarray | None = None,
) -> np.ndarray:
    if gate is None and level.learned_deltas is not None:
        return level.learned_deltas
    features = _feature_tensor(level.features, config.device)
    _torch_sync(config.device)
    start = perf_counter()
    with torch.no_grad():
        if config.selective_nn and gate is not None:
            deltas = _selective_forward(
                components.transfer,
                features,
                gate,
                config,
                work,
                kind="transfer",
            )
        else:
            deltas = components.transfer(features)
            work.transfer_nn_calls += 1
    _torch_sync(config.device)
    work.nn_seconds += perf_counter() - start
    result = deltas.cpu().numpy().astype(np.float32)
    if gate is None:
        level.learned_deltas = result
    return result


def _learned_p(
    level: _Level,
    components: NeuralComponents,
    config: MGConfig,
    work: WorkCounter,
    gate: np.ndarray | None = None,
) -> sp.csr_matrix:
    if level.pattern is None or level.baseline_weights is None:
        raise RuntimeError("coarsest level has no prolongation")
    deltas_np = _transfer_deltas(
        level, components, config, work, gate=gate
    )
    deltas = torch.from_numpy(deltas_np).to(config.device)
    base = torch.as_tensor(level.baseline_weights, dtype=deltas.dtype, device=config.device)
    gate_t = None if gate is None else torch.from_numpy(gate.astype(np.float32)).to(config.device)
    with torch.no_grad():
        weights = weights_from_deltas_torch(level.pattern, deltas, base, gate_t)
    return scipy_prolongation_from_weights(
        level.pattern, weights.cpu().numpy().astype(np.float64)
    )


def _build_static_hierarchy(
    a: sp.csr_matrix,
    shape: GridShape,
    components: NeuralComponents,
    config: MGConfig,
    work: WorkCounter,
    level_index: int = 0,
) -> _Level:
    level = _make_level(a, shape, config, level_index)
    if terminal(shape, config.coarsest_n):
        return level
    if config.mode in {"classical", "nn_smoother"}:
        p = level.baseline_p
    elif config.mode in {"learned_transfer", "combined"}:
        p = _learned_p(level, components, config, work)
    else:
        raise ValueError("adaptive hierarchy is residual-dependent")
    if p is None or level.pattern is None:
        raise RuntimeError("missing prolongation")
    level.p = p
    ac = galerkin_coarse_operator(level.a, p)
    level.coarse = _build_static_hierarchy(
        ac,
        level.pattern.coarse_shape,
        components,
        config,
        work,
        level_index + 1,
    )
    return level


def _coarse_direct(
    a: sp.csr_matrix,
    b: np.ndarray,
    work: WorkCounter,
    *,
    safety: bool = False,
) -> np.ndarray:
    size = a.shape[0]
    work.add_sparse_flops(max(2.0 * a.nnz, (2.0 / 3.0) * size**3), safety=safety)
    return np.asarray(spla.spsolve(a, b), dtype=np.float64)


def _apply_transfer(
    p: sp.csr_matrix,
    vector: np.ndarray,
    work: WorkCounter,
    *,
    transpose: bool,
    safety: bool = False,
) -> np.ndarray:
    work.transfer_apply_count += 1
    work.add_sparse_flops(2.0 * p.nnz, safety=safety)
    return np.asarray((p.T if transpose else p) @ vector, dtype=np.float64).reshape(-1)


def _v_cycle_static(
    level: _Level,
    x: np.ndarray,
    b: np.ndarray,
    components: NeuralComponents,
    config: MGConfig,
    stats: _CycleStats,
    work: WorkCounter,
    *,
    safety: bool = False,
) -> np.ndarray:
    if terminal(level.shape, config.coarsest_n) or level.p is None or level.coarse is None:
        return _coarse_direct(level.a, b, work, safety=safety)
    for _ in range(config.pre_steps):
        x, _ = _smooth(
            level, x, b, components, config, stats, work, reverse=False, safety=safety
        )
    residual = _residual(level.a, x, b, work, safety=safety)
    rc = _apply_transfer(level.p, residual, work, transpose=True, safety=safety)
    ec = _v_cycle_static(
        level.coarse,
        np.zeros(level.coarse.a.shape[0], dtype=np.float64),
        rc,
        components,
        config,
        stats,
        work,
        safety=safety,
    )
    x = x + _apply_transfer(level.p, ec, work, transpose=False, safety=safety)
    for _ in range(config.post_steps):
        x, _ = _smooth(
            level, x, b, components, config, stats, work, reverse=True, safety=safety
        )
    return np.asarray(x, dtype=np.float64)


def _v_cycle_adaptive(
    level: _Level,
    x: np.ndarray,
    b: np.ndarray,
    components: NeuralComponents,
    config: MGConfig,
    stats: _CycleStats,
    work: WorkCounter,
    *,
    refresh: bool,
) -> np.ndarray:
    if terminal(level.shape, config.coarsest_n):
        return _coarse_direct(level.a, b, work)
    local_gate: np.ndarray | None = None
    for _ in range(config.pre_steps):
        x, local_gate = _smooth(
            level, x, b, components, config, stats, work, reverse=False
        )
    residual = _residual(level.a, x, b, work)
    transfer_gate = local_gate if local_gate is not None else stats.last_pre_gate
    if transfer_gate is None:
        diagnostic = _classical_correction(
            level, residual, config, work, reverse=False
        )
        transfer_gate, _ = _detector_gate(
            level, residual, diagnostic, components, config, work
        )
        stats.gates.append(float(transfer_gate.mean()))

    if refresh or level.p is None or level.coarse is None:
        p = _learned_p(level, components, config, work, gate=transfer_gate)
        ac = galerkin_coarse_operator(level.a, p)
        if level.pattern is None:
            raise RuntimeError("adaptive level has no transfer pattern")
        level.p = p
        level.coarse = _make_level(
            ac, level.pattern.coarse_shape, config, level.level_index + 1
        )
        work.hierarchy_rebuilds += 1
    p = level.p
    coarse = level.coarse
    if p is None or coarse is None:
        raise RuntimeError("adaptive hierarchy refresh failed")
    rc = _apply_transfer(p, residual, work, transpose=True)
    ec = _v_cycle_adaptive(
        coarse,
        np.zeros(coarse.a.shape[0], dtype=np.float64),
        rc,
        components,
        config,
        stats,
        work,
        refresh=refresh,
    )
    x = x + _apply_transfer(p, ec, work, transpose=False)
    for _ in range(config.post_steps):
        x, _ = _smooth(level, x, b, components, config, stats, work, reverse=True)
    return np.asarray(x, dtype=np.float64)


def _single_cycle(
    x: np.ndarray,
    b: np.ndarray,
    components: NeuralComponents,
    config: MGConfig,
    hierarchy: _Level | None,
    adaptive_root: _Level | None,
    work: WorkCounter,
    state: _RunState,
) -> tuple[np.ndarray, _CycleStats]:
    stats = _CycleStats()
    state.cycle_calls += 1
    if config.mode == "adaptive":
        if adaptive_root is None:
            raise RuntimeError("adaptive root was not built")
        refresh = (state.cycle_calls - 1) % config.adaptive_refresh == 0
        out = _v_cycle_adaptive(
            adaptive_root,
            x.copy(),
            b,
            components,
            config,
            stats,
            work,
            refresh=refresh,
        )
    else:
        if hierarchy is None:
            raise RuntimeError("static hierarchy was not built")
        out = _v_cycle_static(hierarchy, x.copy(), b, components, config, stats, work)
    state.safeguard_alphas.extend(stats.safeguard_alphas)
    return out, stats


def _cycle_with_fallback(
    a: sp.csr_matrix,
    x: np.ndarray,
    b: np.ndarray,
    components: NeuralComponents,
    config: MGConfig,
    hierarchy: _Level | None,
    adaptive_root: _Level | None,
    baseline_config: MGConfig | None,
    baseline_hierarchy: _Level | None,
    work: WorkCounter,
    state: _RunState,
    baseline_shadow: np.ndarray | None,
) -> tuple[np.ndarray, _CycleStats, np.ndarray | None]:
    """Apply one candidate cycle and optional exact classical-trajectory floor.

    ``baseline_shadow`` follows a pure classical trajectory independently of
    the accepted neural iterates.  Comparing against this shadow (rather than
    a classical cycle started from the current neural state) guarantees that
    the safeguarded stationary method is no worse than the matching pure
    classical method after every equal cycle count.
    """

    candidate, stats = _single_cycle(
        x, b, components, config, hierarchy, adaptive_root, work, state
    )
    if baseline_config is None or baseline_hierarchy is None:
        return candidate, stats, baseline_shadow
    shadow_input = x.copy() if baseline_shadow is None else baseline_shadow.copy()
    next_shadow = _v_cycle_static(
        baseline_hierarchy,
        shadow_input,
        b,
        components,
        baseline_config,
        _CycleStats(),
        work,
        safety=True,
    )
    candidate_residual = _residual(a, candidate, b, work, safety=True)
    baseline_residual = _residual(a, next_shadow, b, work, safety=True)
    if np.linalg.norm(baseline_residual) < np.linalg.norm(candidate_residual):
        state.fallback_count += 1
        return next_shadow, stats, next_shadow
    return candidate, stats, next_shadow


def _select_strategy(
    a: sp.csr_matrix,
    n: int,
    components: NeuralComponents,
    config: MGConfig,
    work: WorkCounter,
) -> MGStrategy:
    if config.strategy_name is not None:
        return get_strategy(config.strategy_name)
    if config.global_selection == "fixed":
        return MGStrategy(
            "custom_fixed",
            config.smoother,
            config.transfer_baseline,
            config.coarsening,
        )
    context = global_context_tensor(a, n, device=config.device)
    _torch_sync(config.device)
    start = perf_counter()
    with torch.no_grad():
        logits = components.selector(context)
    _torch_sync(config.device)
    work.nn_seconds += perf_counter() - start
    work.selector_nn_calls += 1
    return STRATEGIES[int(torch.argmax(logits, dim=1).item())]


def _make_result(
    x: np.ndarray,
    converged: bool,
    cycles: int,
    residual_history: list[float],
    setup_seconds: float,
    solve_seconds: float,
    work: WorkCounter,
    state: _RunState,
    outer_solver: OuterSolver,
    selected_strategy: str,
    estimated_residual_history: list[float] | None = None,
    residual_iteration_indices: list[int] | None = None,
) -> SolveResult:
    return SolveResult(
        x=x,
        converged=converged,
        cycles=cycles,
        residual_history=residual_history,
        elapsed_seconds=setup_seconds + solve_seconds,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
        work_units=work.work_units,
        safety_work_units=work.safety_work_units,
        nn_calls=work.nn_calls,
        transfer_nn_calls=work.transfer_nn_calls,
        smoother_nn_calls=work.smoother_nn_calls,
        detector_nn_calls=work.detector_nn_calls,
        selector_nn_calls=work.selector_nn_calls,
        nn_seconds=work.nn_seconds,
        matvec_count=work.matvec_count,
        gate_history=state.gate_history,
        fallback_count=state.fallback_count,
        safeguard_alphas=state.safeguard_alphas,
        outer_solver=outer_solver,
        estimated_residual_history=estimated_residual_history or [],
        residual_iteration_indices=(
            residual_iteration_indices
            if residual_iteration_indices is not None
            else list(range(len(residual_history)))
        ),
        selected_strategy=selected_strategy,
        hierarchy_rebuilds=work.hierarchy_rebuilds,
        active_patch_fraction=work.active_patch_fraction,
        active_cell_fraction=work.active_cell_fraction,
    )


def solve(
    a: sp.csr_matrix,
    b: np.ndarray,
    n: int,
    components: NeuralComponents | None = None,
    config: MGConfig | None = None,
    x0: np.ndarray | None = None,
) -> SolveResult:
    """Solve an SPD system by stationary MG or Krylov-accelerated MG."""

    config = config or MGConfig()
    _validate_config(config)
    a = a.tocsr()
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    _validate_problem(a, b, n, config)
    components = NeuralComponents.create() if components is None else components
    components.to(config.device).eval()
    x_initial = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    if x_initial.shape != b.shape:
        raise ValueError("x0 shape mismatch")

    work = WorkCounter(fine_nnz=max(a.nnz, 1))
    state = _RunState()
    setup_start = perf_counter()
    strategy = _select_strategy(a, n, components, config, work)
    effective = replace(
        config,
        smoother=strategy.smoother,
        transfer_baseline=strategy.transfer,
        coarsening=strategy.coarsening,
    )
    root_shape = (n, n)
    hierarchy: _Level | None = None
    adaptive_root: _Level | None = None
    if effective.mode == "adaptive":
        adaptive_root = _make_level(a, root_shape, effective, 0)
    else:
        hierarchy = _build_static_hierarchy(
            a, root_shape, components, effective, work
        )

    baseline_config: MGConfig | None = None
    baseline_hierarchy: _Level | None = None
    if effective.mode != "classical" and effective.safety_fallback:
        baseline_config = replace(
            effective,
            mode="classical",
            safety_fallback=False,
            smoother_safeguard=False,
            outer_solver="none",
            selective_nn=False,
        )
        baseline_hierarchy = _build_static_hierarchy(
            a, root_shape, components, baseline_config, work
        )
    setup_seconds = perf_counter() - setup_start

    if effective.outer_solver != "none":
        solve_start = perf_counter()

        def matvec(vector: np.ndarray) -> np.ndarray:
            return _spmv(a, vector, work)

        def precondition(vector: np.ndarray) -> np.ndarray:
            approximation = np.zeros_like(vector)
            baseline_shadow = (
                np.zeros_like(vector) if baseline_hierarchy is not None else None
            )
            local_gates: list[float] = []
            for _ in range(effective.preconditioner_cycles):
                approximation, stats, baseline_shadow = _cycle_with_fallback(
                    a,
                    approximation,
                    vector,
                    components,
                    effective,
                    hierarchy,
                    adaptive_root,
                    baseline_config,
                    baseline_hierarchy,
                    work,
                    state,
                    baseline_shadow,
                )
                local_gates.extend(stats.gates)
            state.gate_history.append(float(np.mean(local_gates)) if local_gates else 0.0)
            return approximation

        if effective.outer_solver == "cg":
            krylov = pcg(
                matvec,
                precondition,
                b,
                x0=x_initial,
                tolerance=effective.tolerance,
                max_iterations=effective.max_cycles,
            )
        elif effective.outer_solver == "fcg":
            krylov = fcg(
                matvec,
                precondition,
                b,
                x0=x_initial,
                tolerance=effective.tolerance,
                max_iterations=effective.max_cycles,
            )
        elif effective.outer_solver == "fgmres":
            krylov = fgmres(
                matvec,
                precondition,
                b,
                x0=x_initial,
                tolerance=effective.tolerance,
                max_iterations=effective.max_cycles,
                restart=effective.krylov_restart,
            )
        else:
            raise AssertionError("validated outer solver")
        solve_seconds = perf_counter() - solve_start
        return _make_result(
            krylov.x,
            krylov.converged,
            krylov.iterations,
            krylov.residual_history,
            setup_seconds,
            solve_seconds,
            work,
            state,
            effective.outer_solver,
            strategy.name,
            krylov.estimated_residual_history,
            krylov.residual_iteration_indices,
        )

    solve_start = perf_counter()
    x = x_initial
    initial_residual = float(np.linalg.norm(_residual(a, x, b, work)))
    residual_history = [initial_residual]
    b_norm = max(float(np.linalg.norm(b)), 1.0e-300)
    if initial_residual / b_norm <= effective.tolerance:
        return _make_result(
            x,
            True,
            0,
            residual_history,
            setup_seconds,
            perf_counter() - solve_start,
            work,
            state,
            "none",
            strategy.name,
        )

    baseline_shadow = x_initial.copy() if baseline_hierarchy is not None else None
    for cycle in range(1, effective.max_cycles + 1):
        x, stats, baseline_shadow = _cycle_with_fallback(
            a,
            x,
            b,
            components,
            effective,
            hierarchy,
            adaptive_root,
            baseline_config,
            baseline_hierarchy,
            work,
            state,
            baseline_shadow,
        )
        residual_norm = float(np.linalg.norm(_residual(a, x, b, work)))
        residual_history.append(residual_norm)
        state.gate_history.append(float(np.mean(stats.gates)) if stats.gates else 0.0)
        if effective.verbose:
            print(
                f"cycle={cycle:03d} mode={effective.mode:>16s} "
                f"strategy={strategy.name:>30s} relres={residual_norm / b_norm:.3e} "
                f"gate={state.gate_history[-1]:.3f} work={work.work_units:.2f} "
                f"nn={work.nn_calls} active_patches={work.active_patch_fraction:.3f}"
            )
        if residual_norm / b_norm <= effective.tolerance:
            return _make_result(
                x,
                True,
                cycle,
                residual_history,
                setup_seconds,
                perf_counter() - solve_start,
                work,
                state,
                "none",
                strategy.name,
            )

    return _make_result(
        x,
        False,
        effective.max_cycles,
        residual_history,
        setup_seconds,
        perf_counter() - solve_start,
        work,
        state,
        "none",
        strategy.name,
    )
