#!/usr/bin/env python3
"""Demonstrate global classical selection followed by local neural correction."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_mg.diagnostics import detector_map, selected_strategy
from adaptive_mg.models import NeuralComponents
from adaptive_mg.pde import DiffusionCase, assemble_problem
from adaptive_mg.reporting import solve_result_record
from adaptive_mg.solver import MGConfig, solve
from adaptive_mg.strategy import STRATEGIES, get_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo_v3"))
    parser.add_argument("--n", type=int, default=31)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is None:
        components = NeuralComponents.create()
        print("No checkpoint: neural deltas are zero; this is a structural smoke demo.")
    else:
        components, metadata = NeuralComponents.load(args.checkpoint, device=args.device)
        print(metadata.get("asset_status", "checkpoint loaded"))

    case = DiffusionCase(
        n=args.n,
        epsilon=1.0e-3,
        angle_deg=45.0,
        contrast=1.0e3,
        pattern="local_patch",
        patch_bounds=(0.28, 0.72, 0.28, 0.72),
    )
    a, b, exact = assemble_problem(case)
    strategy_name, probabilities = selected_strategy(a, case.n, components, device=args.device)
    strategy = get_strategy(strategy_name)
    gate = detector_map(
        a,
        b,
        case.n,
        components,
        device=args.device,
        smoother=strategy.smoother,
    )

    configurations = [
        ("selected_classical", "classical", False, False, "none"),
        ("adaptive_raw", "adaptive", False, False, "none"),
        ("adaptive_safe", "adaptive", True, True, "none"),
        ("adaptive_safe_fcg", "adaptive", True, True, "fcg"),
        ("adaptive_safe_fgmres", "adaptive", True, True, "fgmres"),
    ]
    rows = []
    results = {}
    for name, mode, safeguard, fallback, outer in configurations:
        result = solve(
            a,
            b,
            case.n,
            components,
            MGConfig(
                mode=mode,  # type: ignore[arg-type]
                global_selection="selector",
                outer_solver=outer,  # type: ignore[arg-type]
                max_cycles=args.max_cycles,
                tolerance=1.0e-8,
                device=args.device,
                smoother_safeguard=safeguard,
                safety_fallback=fallback,
                selective_nn=True,
                patch_size=args.patch_size,
                adaptive_refresh=3,
            ),
        )
        results[name] = result
        rows.append({"method": name, **solve_result_record(result, exact=exact)})

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "demo_results.csv", index=False)
    pd.DataFrame(
        {
            "strategy": [item.name for item in STRATEGIES],
            "probability": probabilities,
        }
    ).to_csv(args.output_dir / "selector_probabilities.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for name, result in results.items():
        history = np.asarray(result.residual_history)
        ax.semilogy(range(len(history)), history / history[0], marker="o", markersize=2.5, label=name)
    ax.set_xlabel("cycle / Krylov update")
    ax.set_ylabel("relative true residual")
    ax.set_title(f"Global selector ({strategy_name}) + local adaptive correction")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "residual_history.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    image = ax.imshow(gate.T, origin="lower", vmin=0.0, vmax=1.0)
    ax.set_title("Initial local detector gate")
    ax.set_xlabel("grid index")
    ax.set_ylabel("grid index")
    fig.colorbar(image, ax=ax, label="neural intervention probability")
    fig.tight_layout()
    fig.savefig(args.output_dir / "detector_gate.png", dpi=180)
    plt.close(fig)

    print(f"selected strategy: {strategy_name}")
    print(frame.to_string(index=False))
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
