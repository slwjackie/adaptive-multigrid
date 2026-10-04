# Adaptive Neural Multigrid v6.7 — warm-first expert and policy study

## New primary H_P workflow: EM + schedules + asymptotic affine transfer

[EM_AFFINE_HP_RUNBOOK_KR.md](docs/EM_AFFINE_HP_RUNBOOK_KR.md) documents the new
`run_v6_7_em_transfer_study.py` pipeline: warm-calibrated classical plans,
vectorized constrained EM, batched differentiable V-cycles, all-level
fixed-parent affine P, persistent slow probes and measured checkpoint selection,
followed by an independently fitted **C/H_P** policy. Existing H_S architectures
and old CLIs remain available. Start a NEW run after source changes.
This is an implemented experiment, not a claim that the NN beats the new baseline.


Standalone multigrid with **cached operator-generating neural networks**, a lightweight temporal break-even controller, optional block spatial selection, and a classical lock/recovery path.

The new production API is `adaptive_mg.v67.PreparedAdaptiveMG`. The v6.6 exact-K API is retained only for legacy comparison/regression tests. No external Krylov solver is used by the new production path.

## New: warm-first study (expert first, policy second)

See [WARM_STUDY_RUNBOOK_KR.md](docs/WARM_STUDY_RUNBOOK_KR.md) for the new isolated
`run_v6_7_warm_study.py` workflow: warm multi-RHS timing, tiny/multistage cached
smoothers, no-harm/coarse-aware training, grouped replacement, and a continuous-size
cost policy fitted only after expert selection. The old workflow below is retained.
Use a NEW run; changed-source timing evidence is not resumable. Research candidates
and empirical policy margins are not convergence or speedup certificates.

H_P/H_SP research evidence and proposed next experiments are separately documented
in [HP_HSP_RESEARCH_PLAN_KR.md](docs/HP_HSP_RESEARCH_PLAN_KR.md); these proposals are
not claimed as implemented improvements in the warm-first H_S study.

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
