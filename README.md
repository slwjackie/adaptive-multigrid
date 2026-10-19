# Learned multigrid operator studies

The current [P-only study](docs/learned_p_study.md) fixes classical `line_alt`
smoothing (2 pre + 2 post), full coarsening and a bilinear fallback parent. Only
the sparse P generator is learned, using a stochastic estimate of the complete
V-cycle error-propagation operator's squared Frobenius norm. No neural smoother
is instantiated, trained, serialized or executed in this path. Generated P/R/Ac
and line factors are reused across cycles and RHS for the same A. Use
`scripts/run_v6_7_learned_p_study.py`; it is also the default configuration of the
existing joint-study CLI. Fixed-probe before/after diagnostics precede optional
P-only offline refinement and distillation. Online setup+solve determines success.

The earlier [joint P/pre/post smoother study](docs/joint_transfer_study.md) uses
`line_alt + bilinear + full coarsening` as its fixed parent and fallback. Two
small residual CNNs generate sparse interpolation and separate pre/post
smoothers once per level. Its independent pipeline trains ablations and the
joint model, refines numerical operators offline with multi-start Adam, then
distills their operators and complete V-cycle actions. Six arms are compared
using measured setup-plus-solve time; EM5 is an independent reference arm only.
It remains available with its explicit legacy configuration for reproducibility.
Use a new run directory for the current P-only study. This is
an experiment implementation, not evidence of a speedup.

## Residual U-net transfer teacher study (existing independent workflow)

This branch focuses on learning sparse prolongation from synthetic 2D elliptic
systems, using a residual U-net teacher and compact CNN student. The fixed reference
is `line_alt + energy minimization + full coarsening` (EM 5, pre/post 2/2).
See [the current Korean runbook](docs/RESIDUAL_UNET_TRANSFER_KR.md) for the training
PDE, data generation, commands and evaluation protocol. Sloshing, shear mixing
layers, Navier–Stokes time integration and external CFD bridges are not included.

The direct CNN can also be studied with a learned, cached sparse approximate-inverse
smoother in place of line smoothing. See the [sparse smoother runbook](docs/direct_sparse_smoother.md)
for frozen-P and joint P/smoother controls, cycle diagnostics, timing and Mac commands.

## Legacy H_S workflow: C_tuned + H_S + solver world model

Use [`run_v6_7_hs_world_study.py`](scripts/run_v6_7_hs_world_study.py), with the
[complete Korean runbook](docs/HS_WORLD_THESIS_RUNBOOK_KR.md). One global classical
plan is tuned offline, then kept fixed for the primary C/H_S comparison. H0, H1,
H2 and H2_NH remain unchanged. The 88-plan PDE-adaptive strong_C is an optional,
separately labelled robustness audit, not the primary reference or training parent.

Temporal actions factor **classical P reuse/rebuild x classical/neural smoothing**.
World_C, H_S with matched/tuned non-neural reuse, and World_HS are all evaluated.
A changed A always gets current Galerkin coarse matrices and numeric factors;
learned smoothing is regenerated, not silently reused with stale coefficients.
The world model abstains to a tune-selected classical reuse policy.

H_P/H_SP are retained below for historical ablations, but are not trained or
called by this legacy workflow. These are synthetic elliptic solver studies.
Application interfaces (external CFD LDU import/export and supplied GAMG comparison)
have been removed from this branch; no sloshing or shear-layer flow solver is included.



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

An isolated **residual U-net P teacher → compact CNN student** experiment is
available via `scripts/run_v6_7_unet_transfer_study.py`. It compares the same
small CNN trained directly and with full-V-cycle action distillation, admits KD
only after measured teacher contraction gains, and evaluates setup-inclusive
student time against fixed `line_alt + energy minimization + full coarsening` MG
(EM budget 5, pre/post 2/2). This path uses `prepare`, has no classical selector,
and includes EM setup and CNN inference in cold/multiple timings. Attention is not used.
See [the Korean runbook](docs/RESIDUAL_UNET_TRANSFER_KR.md) for the smoke and
research commands, sparse constraints, teacher gate and independent final/OOD protocol.

See [README_KR.md](README_KR.md) for setup, the six-stage resumable training pipeline, diagnostics, precision contracts, and the exact scope of stability/performance claims.

```bash
pip install -e '.[dev]'
python scripts/build_native_stencil.py
pytest -ra
python scripts/run_v6_7_pipeline.py --config configs/v6_7_smoke.json --output-dir artifacts/my_smoke
```

The included smoke checks do **not** establish speedup over a tuned classical MG solver. Full research training and independent performance auditing are runnable, but not represented as completed unless their generated records say so.
