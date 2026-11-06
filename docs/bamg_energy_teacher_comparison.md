# BAMG + energy: teacher-only comparison

This entry point does **not** train or load a CNN. It compares numerical
interpolation before deciding whether to generate new CNN labels. Existing BAMG
training, checkpoints and experiment directories are not rewritten.

## Fixed normalization contract

For every actual operator/level, form the same geometric bilinear `P_ref` and
freeze the common test vectors `Z` and uniform vector weights `W`. Define

```
J_B(P) = ||(Z - P Z_c) sqrt(W)||_F^2 + mu ||P||_F^2
s_B = J_B(P_ref)
s_E = trace(P_ref.T A P_ref)
J_lambda(P) = J_B(P)/s_B + lambda * trace(P.T A P)/s_E
```

Both denominators are computed once, before optimization, and are stored with
the reference, vector and support hashes. They never follow the optimizer's P.
Zero/nonfinite scales fail explicitly; there is no hidden denominator floor.
Positive ridge and nonzero injection make `s_B` positive; SPD A makes `s_E`
positive. This is dimensionless reference normalization, not a claim that
`lambda=1` is optimal or gives equal gradient magnitudes.

The energy gradient includes **full sparse A P**, including cross-row terms and
off-stencil coarse couplings. Equality constraints use an orthonormal row-sum
nullspace. Projected CG solves the global quadratic; if the signed row L1 cap is
active, an exact row projection and accelerated projected-gradient fallback
enforce it. The report distinguishes a converged optimizer from a finite-budget
candidate. Feasible finite-budget candidates are still timed and identified as
such; they are not advertised as exact minimizers.

All methods retain the same injection and eliminated-Dirichlet boundary row
sums. The controlled support is the 1/2/4-point geometric bilinear pattern. A
selected weight equal to zero remains a stored structural entry.

## Methods and fair-comparison scope

* `fixed_C`: geometric bilinear P.
* `em5_C`: existing native five-iteration projected-CG EM. Its **declared**
  support is checked against the common pattern, independently of zero values.
* `em_constrained_C`: energy-only optimization on the common support with the
  same row-sum, injection and L1 constraints as the hybrid; separate from EM5.
* `bootstrap_C`: the existing two-pass BAMG generator's selected hierarchy.
* `hybrid_lambda_*`: fixed-bank LS + ridge + energy; zero uses the existing
  constrained local LS implementation and is a numerical replay control.

First run the existing BAMG bootstrap once per development A. Freeze the test
vectors in its selected pass at every level. All controlled interpolation
methods use these **identical arrays**, not merely identical random seeds. The
hybrid does not regenerate method-specific bootstrap vectors: this first study
isolates the interpolation objective. This favors a BAMG-derived bank and is
not a complete comparison of independently adapted algorithms.

The finest A, coarse locations, level sizes, structural supports, vectors,
line-alternating 2+2 sweeps, RHS, zero initial guesses and stopping tolerance are
shared. Every candidate nevertheless builds its own **actual** `A_next=P.T A P`.
Consequently deeper A values and their numerical smoother factors can differ.
The report stores those A/P hashes and each level's fixed normalization values;
it does not claim identical coarse A across different interpolation operators.
The `em_constrained_C` comparison remains available if a future native EM5
support changes. Native support agreement is explicitly reported.

## Development validation and lambda selection

The Q1 configuration generates a fresh, reproducible development-validation set:
7 PDE families, n=31/63, 4 operators per family/size (56 operators). It does not
read any prior study or final/OOD data. This is **development selection**, not an
unbiased final-generalization result. No operator is discarded for failure.

Default lambda grid: `[0, 0.01, 0.1, 1, 10]`. The same global lambda is used on
every level and selected across the entire development set, not per case/family.
There are three repeats and two RHS per operator: one existing manufactured
case RHS and one independently seeded random RHS. Neither RHS enters the
setup vector bank. Timing order is deterministically shuffled between methods.
All numerical factors are prepared before timing. There is no fallback and no
CNN evaluation in the warm solve. Stopping remains
`||r|| <= max(1e-12, 1e-8 ||r_initial||)`.

Aggregation: average individual RHS times in a repeat, median across repeats
for each case, then arithmetic mean across cases. `Warm ms*` is therefore the
time for one RHS averaged over the declared RHS distribution, not the sum of
two solves. Failures retain `max(actual_time, configured_penalty)`. Missing
measurements are penalized too; incomplete studies cannot select a lambda.
Within eligible candidates, minimize failure rate first, then warm time.
The default allows **zero failures**. Selecting lambda zero is valid evidence
that energy regularization did not help this experiment. Per-family and
per-grid tables are diagnostic only, not independent lambda selections.

Setup is recorded separately. For bank-dependent methods the reported charged
setup includes common-bank generation; this is an offline comparison cost,
not CNN inference cost. Optimizer convergence, complexity and sparse storage
are also retained. Warm timing on a single CPU/software environment is not a
universal performance claim.

## Run and resume

Use the same Python environment as the existing BAMG study. No new numerical
dependency is required.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

./.venv-bamg/bin/python -u scripts/run_v6_7_bamg_energy_study.py run \
  --config configs/v6_7_bamg_energy_teacher_q1.json \
  --output artifacts/bamg_energy_teacher_q1
```

To interrupt after one NEW completed case, append `--max-cases 1`. That partial
result is only a plumbing check, not a lambda selection. Resume with the same
command and `--resume`; finished measurements are not repeated. An interrupted
case reuses its compressed vector bank and reconstructs its unfinished runtimes.
Source/configuration/environment changes require a new output directory.

```bash
./.venv-bamg/bin/python scripts/run_v6_7_bamg_energy_study.py report \
  --output artifacts/bamg_energy_teacher_q1
```

Append `--group n31`, `--group n63` or a family name such as
`--group rotated_anisotropy` to inspect a subgroup without selecting its own lambda.

`teacher_comparison.json` contains compact overall, grid and family summaries,
the selected lambda and selection scope. `cases/<name>/results.json` contains
small per-case measurements, optimizer diagnostics and normalizers.
`vectors.npz` stores the common setup vectors and original BAMG P in compressed
arrays. Fine-grid matrices, per-row LS diagnostics, solutions and large residual
trajectories are not duplicated into JSON. Only one process may write an output
directory. Cached-vector checksums and resumed-P hashes are checked.

Existing studies hash source files. To preserve an old experiment's resume
environment, run this new comparison in a **separate Git checkout/worktree**
with its own output directory rather than updating the old running checkout.
The code never rewrites an old study manifest to bypass that check.

## Tests

```bash
python -m pytest tests/test_bamg_energy.py tests/test_bamg_energy_study.py -ra
```

Tests cover fixed reference scales and global A scaling, lambda-zero replay,
full cross-row gradients against finite differences and an independent dense
quadratic solve, signed L1 projection against constrained optimization, exact
injection/row sums/support, true multilevel Galerkin rebuilding, common vector
hashes, complete/incomplete selection, failed-case accounting, cache corruption
and resume without regenerating the vector bank.

## Implementation verification pilot

The new tests and existing BAMG transfer/teacher/solver/training/study tests
passed together: **78 passed**. The only warning came from an external pytest
plugin's deprecated pytest API. The report CLI, including `--group n63`, was
also exercised on the following completed comparison.

For an integration pilot, use the supplied Q1 configuration with only
`data.per_family` changed from 4 to 1. All other settings, including 32 setup
vectors, two setup passes, the lambda grid, 3 repeats and 2 RHS, are unchanged.
This gives 14 operators and 84 solve attempts per method. Environment: Linux
x86_64, one CPU thread, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
PyTorch 2.15.0+cpu. This shared execution environment was not a dedicated
benchmark host; the pilot validates the implementation and motivates the full
study, not a statistically established speed advantage or a Mac timing claim.

| Method | Overall warm ms* | Mean cycles | Fail/attempts |
| --- | ---: | ---: | ---: |
| fixed_C | 110.887 | 33.54 | 0/84 |
| em5_C | 67.283 | 21.39 | 0/84 |
| em_constrained_C | 63.080 | 20.57 | 0/84 |
| bootstrap_C | 8671.215 | — | 6/84 |
| hybrid_lambda_0 | 8678.405 | — | 6/84 |
| hybrid_lambda_0.01 | 4392.746 | — | 3/84 |
| hybrid_lambda_0.1 | 112.644 | 32.82 | 0/84 |
| hybrid_lambda_1 | 86.336 | 25.21 | 0/84 |
| hybrid_lambda_10 | 60.586 | 20.96 | 0/84 |

The declared penalty explains the very large failed-method averages; it is not
their successful-solve latency. All failures were max-cycle exits on the random
RHS of n=31 channel or n=63 local-patch cases. The failure count includes
repeated measurements of those RHS, not
six independent operator failures. Native EM5 support matched the common mask.
Hybrid and fully constrained EM optimizers converged on every level. Native
EM5 had 32 levels short of the optimizer tolerance, as expected for its fixed
five-iteration definition; all of its outer solves still converged.

The globally selected grid value was lambda=10. Its warm mean was **37.622 ms
vs EM5's 36.645 ms at n=31**, and **83.549 ms vs 97.922 ms at n=63**. Therefore
the pilot does not establish superiority at both sizes. Lambda=10 is also the
largest finite value in the declared grid, not a continuous optimum. A future
development sweep can bracket larger values while retaining the energy-only
control; use a new output directory for a different grid.

The complete pilot's JSON files occupied about 1.32 MB, and its entire output,
including compressed vector banks, about 15.97 MB. No CNN was retrained or
selected. The 56-operator default remains to be run on the intended benchmark
machine before adopting a new teacher for label generation.
