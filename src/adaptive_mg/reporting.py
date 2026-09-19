"""Result serialization helpers shared by research scripts."""
from __future__ import annotations

from typing import Any

import numpy as np

from .solver import SolveResult


def alpha_statistics(alphas: list[float]) -> dict[str, float | int]:
    values = np.asarray(alphas, dtype=np.float64)
    if values.size == 0:
        return {
            "alpha_count": 0,
            "alpha_mean": float("nan"),
            "alpha_median": float("nan"),
            "alpha_p10": float("nan"),
            "alpha_p90": float("nan"),
            "alpha_zero_fraction": float("nan"),
            "alpha_active_fraction": 0.0,
        }
    return {
        "alpha_count": int(values.size),
        "alpha_mean": float(np.mean(values)),
        "alpha_median": float(np.median(values)),
        "alpha_p10": float(np.quantile(values, 0.10)),
        "alpha_p90": float(np.quantile(values, 0.90)),
        "alpha_zero_fraction": float(np.mean(values <= 1.0e-12)),
        "alpha_active_fraction": float(np.mean(values > 1.0e-12)),
    }


def solve_result_record(
    result: SolveResult,
    *,
    exact: np.ndarray | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "converged": result.converged,
        "iterations": result.cycles,
        "relative_residual": result.relative_residual,
        "asymptotic_contraction": result.asymptotic_contraction,
        "elapsed_seconds": result.elapsed_seconds,
        "setup_seconds": result.setup_seconds,
        "solve_seconds": result.solve_seconds,
        "work_units": result.work_units,
        "safety_work_units": result.safety_work_units,
        "nn_calls": result.nn_calls,
        "transfer_nn_calls": result.transfer_nn_calls,
        "smoother_nn_calls": result.smoother_nn_calls,
        "detector_nn_calls": result.detector_nn_calls,
        "selector_nn_calls": result.selector_nn_calls,
        "nn_seconds": result.nn_seconds,
        "matvec_count": result.matvec_count,
        "true_residual_points": len(result.residual_history),
        "mean_gate": float(np.mean(result.gate_history)) if result.gate_history else 0.0,
        "fallback_count": result.fallback_count,
        "selected_strategy": result.selected_strategy,
        "hierarchy_rebuilds": result.hierarchy_rebuilds,
        "active_patch_fraction": result.active_patch_fraction,
        "active_cell_fraction": result.active_cell_fraction,
    }
    record.update(alpha_statistics(result.safeguard_alphas))
    if exact is not None:
        denominator = max(float(np.linalg.norm(exact)), 1.0e-300)
        record["relative_solution_error"] = float(np.linalg.norm(result.x - exact) / denominator)
    return record
