# Adaptive Neural Multigrid v6.7 — three-pillar fixes

Standalone multigrid with **cached operator-generating neural networks**, a lightweight temporal break-even controller, optional block spatial selection, and a classical lock/recovery path.

The new production API is `adaptive_mg.v67.PreparedAdaptiveMG`. The v6.6 exact-K API is retained only for legacy comparison/regression tests. No external Krylov solver is used by the new production path.

## Start here: optimized strong-aware workflow

The new self-contained workflow fixes batched classical line relaxation, H_S
replacement/setup costs, and support-preserving, parent-relative learned P.
No historical checkpoints are needed. **Train P again** under the new projection;
old speed measurements and v1 selector artifacts are not evidence for this build.

See [the Korean runbook](docs/THREE_PILLARS_RUNBOOK_KR.md) for installation,
full research calibration/training, real multiple-RHS benchmarks, resume,
and the frozen single-use final/OOD evaluation.

```bash
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/build_native_stencil.py --no-openmp  # optional C++ backend
python -m pytest -ra
RUN=artifacts/three_pillars_smoke
python scripts/run_v6_7_three_pillars.py calibrate --config configs/v6_7_three_pillars_smoke.json --run-dir "$RUN"
python scripts/run_v6_7_three_pillars.py train --run-dir "$RUN" --branches H_S H_P
python scripts/run_v6_7_three_pillars.py benchmark --run-dir "$RUN" --branches H_S H_P --repeats 3 --rhs-counts 1 4
python scripts/run_v6_7_three_pillars.py demo --run-dir "$RUN" --branch H_S --family channel --n 31
```

This is a code/experimental-protocol update, **not a guarantee that Neural MG is
faster**. Inspect success counts and incremental speedup over the *same* selected
classical parent. Smoke is never an untouched final-test certificate.

## Legacy workflows

See [README_KR.md](README_KR.md) for setup, the six-stage resumable training pipeline, diagnostics, precision contracts, and the exact scope of stability/performance claims.

```bash
pip install -e '.[dev]'
python scripts/build_native_stencil.py
pytest -ra
python scripts/run_v6_7_pipeline.py --config configs/v6_7_smoke.json --output-dir artifacts/my_smoke
```

The included smoke checks do **not** establish speedup over a tuned classical MG solver. Full research training and independent performance auditing are runnable, but not represented as completed unless their generated records say so.
