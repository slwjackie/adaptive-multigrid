#!/usr/bin/env python3
"""Randomized numerical stress validation for the v3 research code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adaptive_mg.models import NeuralComponents
from adaptive_mg.pde import DiffusionCase, assemble_problem
from adaptive_mg.solver import MGConfig, solve
from adaptive_mg.strategy import STRATEGIES
from adaptive_mg.transfer import (
    baseline_weights,
    build_transfer_pattern,
    dense_prolongation_torch,
    galerkin_coarse_operator,
    weights_from_deltas_torch,
)


def random_case(n: int, rng: np.random.Generator) -> DiffusionCase:
    pattern = str(rng.choice(["uniform", "local_patch", "checkerboard", "channel"]))
    return DiffusionCase(
        n=n,
        epsilon=float(10.0 ** rng.uniform(-4.0, 0.0)),
        angle_deg=float(rng.uniform(0.0, 90.0)),
        contrast=float(10.0 ** rng.uniform(0.0, 5.0)),
        pattern=pattern,  # type: ignore[arg-type]
        checker_cells=int(rng.choice([2, 3, 4, 5])),
        channel_width=float(rng.uniform(0.03, 0.16)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stress_v3"))
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for case_index in range(args.cases):
        n = int(rng.choice([7, 15]))
        case = random_case(n, rng)
        a, b, _ = assemble_problem(case)
        if not np.all(np.isfinite(a.data)) or np.any(a.diagonal() <= 0.0):
            failures.append(f"case {case_index}: invalid assembled matrix")
            continue

        # Every controlled classical strategy must produce a finite cycle.
        for strategy in STRATEGIES:
            result = solve(
                a,
                b,
                n,
                NeuralComponents.create(hidden=8),
                MGConfig(
                    mode="classical",
                    strategy_name=strategy.name,
                    max_cycles=2,
                    tolerance=0.0,
                    safety_fallback=False,
                    smoother_safeguard=False,
                ),
            )
            finite = bool(np.all(np.isfinite(result.x)) and np.isfinite(result.relative_residual))
            rows.append(
                {
                    "check": "classical_strategy",
                    "case": case_index,
                    "n": n,
                    "pattern": case.pattern,
                    "strategy": strategy.name,
                    "finite": finite,
                    "relative_residual": result.relative_residual,
                }
            )
            if not finite:
                failures.append(f"case {case_index}: nonfinite {strategy.name}")

        # Extreme learned transfer must preserve full rank and SPD Galerkin.
        for coarsening in ("full", "semi_x", "semi_y", "alternating"):
            pattern = build_transfer_pattern(n, coarsening=coarsening, level_index=0)
            base = baseline_weights(a, (n, n), "bilinear", coarse=pattern.coarse_shape)
            deltas = torch.as_tensor(
                rng.normal(scale=25.0, size=(1, 16, n, n)), dtype=torch.float64
            )
            weights = weights_from_deltas_torch(pattern, deltas, base)
            p_dense = dense_prolongation_torch(pattern, weights).detach().numpy()
            rank_ok = np.linalg.matrix_rank(p_dense) == p_dense.shape[1]
            import scipy.sparse as sp

            p = sp.csr_matrix(p_dense)
            ac = galerkin_coarse_operator(a, p)
            eig_min = float(np.linalg.eigvalsh(ac.toarray()).min())
            spd = bool(eig_min > 0.0)
            rows.append(
                {
                    "check": "extreme_transfer",
                    "case": case_index,
                    "n": n,
                    "pattern": case.pattern,
                    "strategy": coarsening,
                    "finite": rank_ok and spd,
                    "rank_ok": rank_ok,
                    "coarse_min_eigenvalue": eig_min,
                }
            )
            if not (rank_ok and spd):
                failures.append(f"case {case_index}: transfer {coarsening} rank/SPD failure")

        # With randomized neural weights, exact cycle fallback must never be
        # worse than the matching pure-classical trajectory.
        components = NeuralComponents.create(hidden=8)
        with torch.no_grad():
            for module in (components.transfer, components.smoother, components.detector):
                for parameter in module.parameters():
                    parameter.normal_(mean=0.0, std=0.35)
            components.detector.head.bias.fill_(2.0)
        classical = solve(
            a,
            b,
            n,
            components,
            MGConfig(
                mode="classical",
                strategy_name="jacobi_bilinear_full",
                max_cycles=2,
                tolerance=0.0,
                safety_fallback=False,
                smoother_safeguard=False,
            ),
        )
        for mode in ("nn_smoother", "learned_transfer", "combined", "adaptive"):
            safe = solve(
                a,
                b,
                n,
                components,
                MGConfig(
                    mode=mode,  # type: ignore[arg-type]
                    strategy_name="jacobi_bilinear_full",
                    max_cycles=2,
                    tolerance=0.0,
                    safety_fallback=True,
                    smoother_safeguard=True,
                    patch_size=4,
                    adaptive_refresh=2,
                ),
            )
            guaranteed = safe.relative_residual <= classical.relative_residual * (1.0 + 1.0e-10)
            rows.append(
                {
                    "check": "safe_fallback",
                    "case": case_index,
                    "n": n,
                    "pattern": case.pattern,
                    "strategy": mode,
                    "finite": bool(np.isfinite(safe.relative_residual)),
                    "guaranteed": guaranteed,
                    "classical_relative_residual": classical.relative_residual,
                    "safe_relative_residual": safe.relative_residual,
                    "fallback_count": safe.fallback_count,
                }
            )
            if not guaranteed:
                failures.append(f"case {case_index}: safe {mode} exceeded classical")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "stress_results.csv", index=False)
    summary = {
        "seed": args.seed,
        "random_cases": args.cases,
        "checks": len(rows),
        "failures": failures,
        "passed": not failures,
    }
    (args.output_dir / "stress_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
