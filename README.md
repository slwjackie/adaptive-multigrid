# Learned prolongation with fixed line smoothing

The active branch is **`residual-u-net-teacher`**. Its current experiment trains
only a small CNN that generates sparse prolongation matrices **P**, with classical
`line_alt` smoothing fixed. The branch name comes from an earlier U-net experiment;
the current P-only path does not train or run a U-net teacher or neural smoother.

[한국어 안내](README_KR.md) · [Documentation index](docs/README.md) ·
[Current design](docs/learned_p_study.md) · [Mac runbook](docs/learned_p_runbook.md)

## Current numerical contract

| Component | Current P-only experiment |
| --- | --- |
| PDE | Synthetic 2D SPD diffusion systems; no CFD time integrator |
| Coarsening | Fixed full coarsening |
| Pre / post smoothing | Classical `line_alt`, 2 sweeps each, every nonterminal level |
| Learned component | Width-8 residual CNN, 3,208 trainable P-generator parameters |
| Transfer | Bilinear-support-preserving P, R = P transpose |
| Coarse operators | Actual Galerkin Ac = P transpose × A × P |
| Terminal solve | LU at the 3 × 3 interior grid |
| Parent and fallback | `line_alt + bilinear + full`, strategy `line_alt_bilinear_full__v22` |
| Independent reference | `line_alt + EM5 + full`; EM is not part of learned setup |
| Training objective | Stochastic estimate of the full V-cycle error operator's squared Frobenius norm, divided by the number of unknowns |
| Performance criterion | Time to the same residual tolerance, including setup for cold / multiple RHS |

The objective is a surrogate for spectral radius, not a differentiable wall clock
or an exact cycle count. Same-A P/R/Ac, line factors and LU are cached across cycles
and RHS. With the current explicit `H_P` branch, growing/nonfinite candidates or
learned setup failures trigger classical recovery; near-tolerance finishing is
also classical. There is no runtime comparison against a classical shadow solve
and no speed-based switching policy in this path. See the design for exact criteria.

## Start the current experiment

```bash
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
SCRIPT="scripts/run_v6_7_learned_p_study.py"
P_RUN="artifacts/learned_p_smoke_$(date +%Y%m%d_%H%M%S)"
python "$SCRIPT" prepare --run-dir "$P_RUN" --config configs/v6_7_learned_p_smoke.json && \
python -u "$SCRIPT" train --run-dir "$P_RUN" && \
python -u "$SCRIPT" diagnose --run-dir "$P_RUN"
```

| Budget | Train A | Validation A | Direct training updates |
| --- | ---: | ---: | ---: |
| Checked-in smoke config | 7 at n=7 | 7 at n=7 | 14 |
| Checked-in research config, preliminary budget | 84 at n=15/31/63 | 28 at n=15/31/63/127 | 336 |
| Current expanded runbook recipe | **420 at n=15/31/63** | **140 at n=15/31/63/127** | **4,200** |

The 420-operator recipe creates a new local config; it does not silently change the
checked-in default or an existing run. Follow the [Mac runbook](docs/learned_p_runbook.md)
for fresh-run setup, logs, resume and a **three-arm direct-P validation** against
bilinear classical and EM5 classical. The normal `benchmark` CLI currently also
requires `distilled_P`; the runbook provides an existing-API invocation that works
immediately after direct training. Direct validation does not produce the evidence
required by the four-arm `freeze` / `final` workflow.

Optional P-only `search` and `distill` refine P for each training A, then distill it
into the small CNN. They are later experiments, not prerequisites for direct-P
validation. Final/OOD data stay sealed during development. Smoke completion and
unit tests establish execution correctness, not a speedup; research-scale performance
of the expanded recipe has not yet been established.

## Earlier experiments

Earlier workflows remain available with their own explicit configurations. Their
baselines, losses and results do not describe the current P-only experiment.

| Earlier workflow | Documentation |
| --- | --- |
| Residual U-net teacher / student, EM5 parent | [U-net transfer](docs/RESIDUAL_UNET_TRANSFER_KR.md) |
| Direct-CNN pruning and TTT | [Pruning / TTT](docs/direct_pruning_ttt.md) |
| Cached learned sparse smoother | [Sparse smoother](docs/direct_sparse_smoother.md) |
| Joint P and pre/post smoother training | [Joint experiment](docs/joint_transfer_study.md) |
| Strong-aware, warm, EM-affine and world-model studies | [Historical document index](docs/README.md) |

Implementation locations are listed in [the architecture guide](docs/ARCHITECTURE_KR.md).
Run `python -m pytest -ra` for the test suite. The default P-only CPU/CSR path does
not require the optional native stencil backend.
