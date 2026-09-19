#!/usr/bin/env python3
"""Targeted full-V-cycle generalization check across n=15,31,63."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adaptive_mg.models import NeuralComponents
from adaptive_mg.pde import assemble_problem, case_suite
from adaptive_mg.reporting import solve_result_record
from adaptive_mg.solver import MGConfig, solve
from adaptive_mg.strategy import STRATEGY_BY_NAME


def parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cross_grid.csv"))
    parser.add_argument("--sizes", type=parse_sizes, default=(15, 31, 63))
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument(
        "--strategy", choices=sorted(STRATEGY_BY_NAME), default="jacobi_bilinear_full"
    )
    parser.add_argument(
        "--cases",
        default="rotated_anisotropy,local_patch,checkerboard,channel",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_cases = {item.strip() for item in args.cases.split(",") if item.strip()}
    components, _ = NeuralComponents.load(args.checkpoint, device=args.device)
    methods = (
        ("classical", False, False),
        ("nn_smoother", False, False),
        ("learned_transfer", False, False),
        ("combined", False, False),
        ("adaptive", False, False),
        ("adaptive", True, True),
    )
    rows: list[dict[str, object]] = []
    for n in args.sizes:
        for case_name, case in case_suite(n):
            if case_name not in selected_cases:
                continue
            a, b, exact = assemble_problem(case)
            for mode, safeguard, fallback in methods:
                profile = "safe" if safeguard or fallback else "raw"
                result = solve(
                    a,
                    b,
                    n,
                    components,
                    MGConfig(
                        mode=mode,  # type: ignore[arg-type]
                        strategy_name=args.strategy,
                        max_cycles=args.cycles,
                        tolerance=0.0,
                        smoother_safeguard=safeguard,
                        safety_fallback=fallback,
                        selective_nn=True,
                        patch_size=8,
                        adaptive_refresh=3,
                        device=args.device,
                    ),
                )
                rows.append(
                    {
                        "n": n,
                        "case": case_name,
                        "mode": mode,
                        "safety_profile": profile,
                        "strategy": args.strategy,
                        **solve_result_record(result, exact=exact),
                    }
                )
                pd.DataFrame(rows).to_csv(args.output, index=False)
                print(
                    f"n={n:3d} {case_name:>20s} {mode:>16s} {profile:>4s} "
                    f"rho={result.asymptotic_contraction:.4f} "
                    f"relres={result.relative_residual:.3e} "
                    f"patches={result.active_patch_fraction:.3f}"
                )
    print(args.output)


if __name__ == "__main__":
    main()
