#!/usr/bin/env python3
"""Benchmark controlled classical MG strategies, including semicoarsening."""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from adaptive_mg.models import NeuralComponents
from adaptive_mg.pde import assemble_problem, case_suite
from adaptive_mg.reporting import solve_result_record
from adaptive_mg.solver import MGConfig, solve
from adaptive_mg.strategy import MGStrategy, STRATEGIES


def parse_sizes(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one grid size is required")
    return result


def exhaustive_strategies() -> tuple[MGStrategy, ...]:
    return tuple(
        MGStrategy(f"{smoother}_{transfer}_{coarsening}", smoother, transfer, coarsening)
        for smoother, transfer, coarsening in product(
            ("jacobi", "line_x", "line_y", "line_alt"),
            ("bilinear", "operator"),
            ("full", "semi_x", "semi_y", "alternating"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=parse_sizes, default=(31, 63))
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline_v3"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    components = NeuralComponents.create(hidden=8)
    strategies = exhaustive_strategies() if args.exhaustive else STRATEGIES
    rows: list[dict[str, object]] = []

    for n in args.sizes:
        for case_name, case in case_suite(n):
            a, b, exact = assemble_problem(case)
            for strategy in strategies:
                result = solve(
                    a,
                    b,
                    case.n,
                    components,
                    MGConfig(
                        mode="classical",
                        strategy_name=strategy.name if strategy in STRATEGIES else None,
                        smoother=strategy.smoother,
                        transfer_baseline=strategy.transfer,
                        coarsening=strategy.coarsening,
                        max_cycles=args.cycles,
                        tolerance=args.tolerance,
                        safety_fallback=False,
                        smoother_safeguard=False,
                    ),
                )
                row = {
                    "n": n,
                    "case": case_name,
                    "strategy": strategy.name,
                    "smoother": strategy.smoother,
                    "transfer": strategy.transfer,
                    "coarsening": strategy.coarsening,
                    **solve_result_record(result, exact=exact),
                }
                rows.append(row)
                print(
                    f"n={n:3d} {case_name:>22s} {strategy.name:>34s} "
                    f"rho={result.asymptotic_contraction:.4f} "
                    f"relres={result.relative_residual:.3e} WU={result.work_units:.1f}"
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "all_classical_strategies.csv", index=False)
    valid = frame.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["relative_residual"]
    )
    best_indices = valid.groupby(["n", "case"])["relative_residual"].idxmin()
    best = valid.loc[best_indices].sort_values(["n", "case"])
    best.to_csv(args.output_dir / "best_classical_by_case.csv", index=False)

    # Residual/work Pareto frontier prevents a very expensive semicoarsening
    # method from being presented as the sole "best" baseline.
    pareto_rows: list[pd.Series] = []
    for _, group in valid.groupby(["n", "case"], sort=False):
        ordered = group.sort_values(["work_units", "relative_residual"])
        best_residual = float("inf")
        for _, row in ordered.iterrows():
            residual = float(row["relative_residual"])
            if residual < best_residual:
                pareto_rows.append(row)
                best_residual = residual
    pareto = pd.DataFrame(pareto_rows).sort_values(["n", "case", "work_units"])
    pareto.to_csv(args.output_dir / "pareto_classical_by_case.csv", index=False)
    print(f"all results: {args.output_dir / 'all_classical_strategies.csv'}")
    print(f"best residual summary: {args.output_dir / 'best_classical_by_case.csv'}")
    print(f"residual/work Pareto: {args.output_dir / 'pareto_classical_by_case.csv'}")


if __name__ == "__main__":
    main()
