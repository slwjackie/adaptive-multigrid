#!/usr/bin/env python3
"""Profile vectorized transfer setup, hierarchy refresh, and selective NN work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from adaptive_mg.models import NeuralComponents
from adaptive_mg.pde import DiffusionCase, assemble_problem, assemble_stiffness
from adaptive_mg.reporting import solve_result_record
from adaptive_mg.solver import MGConfig, solve
from adaptive_mg.transfer import operator_dependent_weights


def parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def sparse_bytes(matrix) -> int:
    matrix = matrix.tocsr()
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/profile_v3"))
    parser.add_argument("--operator-sizes", type=parse_sizes, default=(31, 63, 127))
    parser.add_argument("--adaptive-n", type=int, default=63)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    memory: list[dict[str, object]] = []
    for n in args.operator_sizes:
        a = assemble_stiffness(
            DiffusionCase(n=n, epsilon=1.0e-2, angle_deg=30.0, contrast=1.0e4, pattern="checkerboard")
        )
        durations = []
        for _ in range(args.repeats):
            start = perf_counter()
            weights = operator_dependent_weights(a, n)
            durations.append(perf_counter() - start)
        rows.append(
            {
                "profile": "operator_dependent_weights",
                "n": n,
                "seconds_min": min(durations),
                "seconds_mean": float(np.mean(durations)),
                "finite": bool(np.all(np.isfinite(weights))),
            }
        )
        memory.append(
            {
                "n": n,
                "unknowns": n * n,
                "sparse_bytes": sparse_bytes(a),
                "hypothetical_dense_bytes_float32": int((n * n) ** 2 * 4),
                "dense_to_sparse_ratio": float(((n * n) ** 2 * 4) / max(sparse_bytes(a), 1)),
            }
        )

    n = args.adaptive_n
    case = DiffusionCase(n=n, epsilon=1.0e-3, angle_deg=45.0, contrast=1.0e3, pattern="local_patch")
    a, b, exact = assemble_problem(case)
    components = NeuralComponents.create(hidden=8)
    with torch.no_grad():
        components.detector.head.bias.fill_(10.0)

    for refresh in (1, max(2, args.cycles)):
        result = solve(
            a,
            b,
            n,
            components,
            MGConfig(
                mode="adaptive",
                max_cycles=args.cycles,
                tolerance=0.0,
                safety_fallback=False,
                smoother_safeguard=False,
                selective_nn=True,
                patch_size=16,
                adaptive_refresh=refresh,
            ),
        )
        rows.append(
            {
                "profile": "adaptive_refresh",
                "n": n,
                "adaptive_refresh": refresh,
                **solve_result_record(result, exact=exact),
            }
        )

    for detector_bias, label in ((-10.0, "easy_all_skipped"), (10.0, "hard_all_active")):
        with torch.no_grad():
            components.detector.head.bias.fill_(detector_bias)
        result = solve(
            a,
            b,
            n,
            components,
            MGConfig(
                mode="adaptive",
                max_cycles=1,
                tolerance=0.0,
                safety_fallback=False,
                smoother_safeguard=False,
                selective_nn=True,
                patch_size=8,
                adaptive_refresh=1,
            ),
        )
        rows.append(
            {
                "profile": "selective_nn",
                "selection_case": label,
                "n": n,
                **solve_result_record(result, exact=exact),
            }
        )

    pd.DataFrame(rows).to_csv(args.output_dir / "profile.csv", index=False)
    pd.DataFrame(memory).to_csv(args.output_dir / "sparse_memory.csv", index=False)
    (args.output_dir / "profile.json").write_text(
        json.dumps({"rows": rows, "memory": memory}, indent=2, default=str),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"profile: {args.output_dir / 'profile.csv'}")


if __name__ == "__main__":
    main()
