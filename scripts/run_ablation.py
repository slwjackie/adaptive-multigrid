#!/usr/bin/env python3
"""Run classical, raw-neural, and safeguarded hierarchical-MG ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_mg.models import NeuralComponents
from adaptive_mg.pde import assemble_problem, case_suite
from adaptive_mg.reporting import solve_result_record
from adaptive_mg.solver import MGConfig, SolveResult, solve
from adaptive_mg.strategy import STRATEGIES


def parse_sizes(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one grid size is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ablation_v3"))
    parser.add_argument("--sizes", type=parse_sizes, default=(15, 31, 63))
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--patch-halo", type=int, default=2)
    parser.add_argument("--detector-threshold", type=float, default=0.5)
    parser.add_argument("--adaptive-refresh", type=int, default=3)
    parser.add_argument("--skip-krylov", action="store_true")
    parser.add_argument("--skip-classical-catalog", action="store_true")
    return parser.parse_args()


def run_one(
    a,
    b,
    n: int,
    components: NeuralComponents,
    *,
    mode: str,
    safety_profile: str,
    max_cycles: int,
    tolerance: float,
    device: str,
    outer_solver: str = "none",
    global_selection: str = "selector",
    strategy_name: str | None = None,
    patch_size: int,
    patch_halo: int,
    detector_threshold: float,
    adaptive_refresh: int,
) -> SolveResult:
    safe = safety_profile == "safe"
    return solve(
        a,
        b,
        n,
        components,
        MGConfig(
            mode=mode,  # type: ignore[arg-type]
            global_selection=global_selection,  # type: ignore[arg-type]
            strategy_name=strategy_name,
            outer_solver=outer_solver,  # type: ignore[arg-type]
            max_cycles=max_cycles,
            tolerance=tolerance,
            device=device,
            smoother_safeguard=safe,
            safety_fallback=safe and mode != "classical",
            selective_nn=True,
            patch_size=patch_size,
            patch_halo=patch_halo,
            detector_threshold=detector_threshold,
            adaptive_refresh=adaptive_refresh,
        ),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is None:
        components = NeuralComponents.create()
        metadata = {
            "asset_status": "untrained v3 model; zero neural corrections and default selector"
        }
    else:
        components, metadata = NeuralComponents.load(args.checkpoint, device=args.device)
    components.eval()

    rows: list[dict[str, object]] = []
    alpha_payload: dict[str, list[float]] = {}
    histories: dict[tuple[int, str], dict[str, tuple[list[int], list[float]]]] = {}

    def record_run(
        *,
        n: int,
        case_name: str,
        exact: np.ndarray,
        method: str,
        mode: str,
        safety_profile: str,
        result: SolveResult,
        outer_solver: str,
    ) -> None:
        key = f"n{n}/{case_name}/{method}/{safety_profile}"
        alpha_payload[key] = [float(value) for value in result.safeguard_alphas]
        histories.setdefault((n, case_name), {})[f"{method}_{safety_profile}"] = (
            result.residual_iteration_indices, result.residual_history
        )
        rows.append(
            {
                "n": n,
                "case": case_name,
                "method": method,
                "mode": mode,
                "safety_profile": safety_profile,
                "outer_solver": outer_solver,
                **solve_result_record(result, exact=exact),
            }
        )
        print(
            f"n={n:3d} {case_name:>20s} {method:>30s} {safety_profile:>5s} "
            f"rho={result.asymptotic_contraction:.4f} relres={result.relative_residual:.3e} "
            f"WU={result.work_units:.1f} alpha={result.safeguard_alpha_mean:.3f} "
            f"fallback={result.fallback_count} patches={result.active_patch_fraction:.3f}"
        )

    for n in args.sizes:
        for case_name, case in case_suite(n):
            a, b, exact = assemble_problem(case)

            if not args.skip_classical_catalog:
                for strategy in STRATEGIES:
                    result = run_one(
                        a,
                        b,
                        n,
                        components,
                        mode="classical",
                        safety_profile="raw",
                        max_cycles=args.max_cycles,
                        tolerance=args.tolerance,
                        device=args.device,
                        global_selection="fixed",
                        strategy_name=strategy.name,
                        patch_size=args.patch_size,
                        patch_halo=args.patch_halo,
                        detector_threshold=args.detector_threshold,
                        adaptive_refresh=args.adaptive_refresh,
                    )
                    record_run(
                        n=n,
                        case_name=case_name,
                        exact=exact,
                        method=f"classical::{strategy.name}",
                        mode="classical",
                        safety_profile="raw",
                        result=result,
                        outer_solver="none",
                    )

            selector_classical = run_one(
                a,
                b,
                n,
                components,
                mode="classical",
                safety_profile="raw",
                max_cycles=args.max_cycles,
                tolerance=args.tolerance,
                device=args.device,
                patch_size=args.patch_size,
                patch_halo=args.patch_halo,
                detector_threshold=args.detector_threshold,
                adaptive_refresh=args.adaptive_refresh,
            )
            record_run(
                n=n,
                case_name=case_name,
                exact=exact,
                method="global_selector+classical",
                mode="classical",
                safety_profile="raw",
                result=selector_classical,
                outer_solver="none",
            )

            for mode in ("nn_smoother", "learned_transfer", "combined", "adaptive"):
                for safety_profile in ("raw", "safe"):
                    result = run_one(
                        a,
                        b,
                        n,
                        components,
                        mode=mode,
                        safety_profile=safety_profile,
                        max_cycles=args.max_cycles,
                        tolerance=args.tolerance,
                        device=args.device,
                        patch_size=args.patch_size,
                        patch_halo=args.patch_halo,
                        detector_threshold=args.detector_threshold,
                        adaptive_refresh=args.adaptive_refresh,
                    )
                    record_run(
                        n=n,
                        case_name=case_name,
                        exact=exact,
                        method=f"global_selector+{mode}",
                        mode=mode,
                        safety_profile=safety_profile,
                        result=result,
                        outer_solver="none",
                    )

            if not args.skip_krylov:
                cg = run_one(
                    a,
                    b,
                    n,
                    components,
                    mode="classical",
                    safety_profile="raw",
                    max_cycles=args.max_cycles,
                    tolerance=args.tolerance,
                    device=args.device,
                    outer_solver="cg",
                    patch_size=args.patch_size,
                    patch_halo=args.patch_halo,
                    detector_threshold=args.detector_threshold,
                    adaptive_refresh=args.adaptive_refresh,
                )
                record_run(
                    n=n,
                    case_name=case_name,
                    exact=exact,
                    method="global_selector+classical+CG",
                    mode="classical",
                    safety_profile="raw",
                    result=cg,
                    outer_solver="cg",
                )
                for outer in ("fcg", "fgmres"):
                    for profile in ("raw", "safe"):
                        result = run_one(
                            a,
                            b,
                            n,
                            components,
                            mode="adaptive",
                            safety_profile=profile,
                            max_cycles=args.max_cycles,
                            tolerance=args.tolerance,
                            device=args.device,
                            outer_solver=outer,
                            patch_size=args.patch_size,
                            patch_halo=args.patch_halo,
                            detector_threshold=args.detector_threshold,
                            adaptive_refresh=args.adaptive_refresh,
                        )
                        record_run(
                            n=n,
                            case_name=case_name,
                            exact=exact,
                            method=f"global_selector+adaptive+{outer.upper()}",
                            mode="adaptive",
                            safety_profile=profile,
                            result=result,
                            outer_solver=outer,
                        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "ablation.csv", index=False)
    (args.output_dir / "safeguard_alphas.json").write_text(
        json.dumps(alpha_payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    # Compact figures focus on the paper-relevant raw/safe distinction.
    for (n, case_name), case_histories in histories.items():
        chosen = {
            name: history
            for name, history in case_histories.items()
            if name.startswith("global_selector+adaptive_")
            or name.startswith("global_selector+combined_")
            or name == "global_selector+classical_raw"
        }
        if not chosen:
            continue
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for name, (iteration_indices, history) in chosen.items():
            relative = np.asarray(history) / max(history[0], 1.0e-300)
            ax.semilogy(iteration_indices, relative, marker="o", markersize=2.2, label=name)
        ax.set_xlabel("cycle / Krylov update")
        ax.set_ylabel("relative true residual")
        ax.set_title(f"n={n}: {case_name.replace('_', ' ').title()}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"n{n}_{case_name}_raw_vs_safe.png", dpi=180)
        plt.close(fig)

    print(f"results: {args.output_dir / 'ablation.csv'}")
    print(f"alpha distributions: {args.output_dir / 'safeguard_alphas.json'}")


if __name__ == "__main__":
    main()
