# Neural AMG: unstructured P1 study

This branch provides one discretization: **continuous P1 FEM on unstructured
triangular meshes**. Its protocol is `neural-amg-p1-study-v3`. Q1, fixed-diagonal
structured triangles, old geometric/CNN runners and arbitrary preassembled A/b
study inputs have been removed from this branch. Earlier work remains in Git
history and on `residual-u-net-teacher`.

## Numerical path

For every study operator, seeded interior points are perturbed, optionally
spatially graded, permuted and Delaunay triangulated on the unit square. The FEM
assembler uses the actual three vertex coordinates of every triangle, constant
P1 basis gradients and positive seven-point triangle quadrature. It integrates
both the diffusion tensor and the physical source f=1; homogeneous Dirichlet
unknowns are eliminated. The [P1 derivation and verification](p1_fem.md) explains
quadrature, boundary handling, mesh quality checks and convergence tests.

At each nonterminal level, Ruge–Stüben AMG chooses coarse nodes C and classical
interpolation support S from that level's actual A. S is checked to reach at most
two hops in the same A graph. Three one-hop graph attention layers encode A;
sparse fine-query/coarse-key-value attention then operates over S. A signed edge
head predicts P weights; softmax attention weights are not used directly as P.
Coarse injection and reference interpolation row sums are enforced. Hard row-L1
caps have been removed from both the network and numerical teacher.
The encoder has a three-hop receptive field; encoded coarse candidates can extend
the complete prediction's receptive field beyond three hops around a fine node.

The numerical BAMG+EM teacher fits relaxed/bootstrap test vectors, a small ridge
term and normalized trace(PᵀAP) energy on fixed C/S. The original classical P
provides fixed normalization constants and row-sum targets. Let F be the fine
rows, excluding coarse injection rows, and let sᵢ be the fixed reference row sum.
The soft cancellation penalty at a level is

```text
C(P) = mean over i in F of [ (sum_j |P_ij| - |s_i|) / (1 + |s_i|) ]²
```

Teacher optimization, NN supervision and MG fine-tuning each use a configurable
`cancellation_weight`, with **η=0.1** in both supplied configurations. This is an
initial experimental choice, not a validated optimum or a stability guarantee.
For `[100,-99]`, the normalized squared penalty is 9801 and its weighted
contribution is 980.1 before averaging across rows. For `[11.175]`, it is zero.
The penalty permits necessary signed weights while discouraging cancellation.
For a single permitted connection, the preserved row sum still fixes its weight.

The teacher first solves the quadratic part with CG and then, when mixed signs
require it, uses ADMM with an exact cancellation proximal update and sparse CG
subproblems. Iteration budgets are finite. Reports contain optimizer convergence,
CG/ADMM iterations, primal/dual residuals and cancellation before/after; a returned
label must not be described as an exact optimum merely because it is feasible.
The old `l1_cap=8` setting is rejected by v3; `None` has no cap semantics.

Supervision combines normalized weight MSE, `0.1 ×` normalized probe-action MSE
and `η C(P)`, averaged over classical hierarchy levels. Fine-tuning builds the
actual learned hierarchy and optimizes multicycle log residual reduction plus
`η C(P)`, averaged over predicted nonterminal levels. C/S and reference row sums
are frozen for each forward/backward pass. Predicted levels that are subsequently
rejected by a first-cycle guard still contribute their cancellation penalty.
The fine-tuning log prints `loss`, `mg`, `cancellation` and `optimized`; if a case has
no differentiable prediction after setup truncation, it is recorded as a direct
case without an optimizer step. The `updates` counter records processed cases.

Training and deployment share bounded weighted Jacobi pre/post smoothing and a
sparse direct terminal solve. R=Pᵀ and Ac=PᵀAP; coarse fill is retained. NN inference
runs during setup. `max_coarse=9` is the normal stopping threshold and
`max_levels` is an explicit safety limit (20 in research). Teacher bootstrap
hierarchies inherit this depth limit during both label generation and runtime
benchmarks; an explicit `teacher.bootstrap_max_levels` override takes precedence.
If bootstrap stops coarsening at the finest level, a homogeneous direct solve
would erase every test vector. The teacher retains the preceding relaxed vector
bank instead and records incomplete bootstrap cycles and the guard reason. It
does not create replacement diagnostic probes.

### Numerical checks and direct fallback

Every method (classical, teacher and learned) uses the same numerical guards.
During setup, P is checked for finite values, shape, injection and preserved row
sums. Generated coarse matrices are checked for finite entries, symmetry and
positive diagonal; the terminal LU factor is inspected. These inexpensive checks
assume SPD input and do not constitute a full SPD certificate. P1 assembly with
SPD tensors and Dirichlet elimination supplies the study's SPD matrices.

If a proposed interpolation/coarse matrix fails a numerical check, coarsening
stops at the current valid level and its A is factored directly. If terminal LU
fails, the solver returns to an earlier valid level. A failure of the finest
matrix's direct solve is reported as a failure rather than silently accepted.

At each visited level, the **first actual V-cycle** compares its incoming and
outgoing residuals. The configurable `hierarchy.first_cycle_max_growth=10.0`
allows a modest transient increase and triggers direct fallback for more than
tenfold residual growth (with rounding tolerance), nonfinite values or numerical
cycle failure. Ten is a heuristic safety threshold. No extra diagnostic vectors
or V-cycles are run. Training's existing random probes remain the input to its
MG loss. Each runtime solve, including a warm solve, performs its first-cycle
checks on the actual RHS. Truncation persists in the reused hierarchy.

On a failure, the bad cycle result is discarded, the offending level becomes a
terminal level, and the actual correction equation at that level is solved by
sparse LU. This preserves preceding learned interpolation. It is not a switch to
a separately constructed classical hierarchy. A large direct level can cost
substantial time and memory. Setup factorization is charged to setup, while
first-cycle fallback factorization is charged to solve time. `guard_events` and
`warm_guard_events` record stages, levels, sizes, reasons and residuals where
available. Benchmark repetitions also record the resulting `level_sizes`,
`terminal_size`, `warm_terminal_size` and `operator_complexity`, so direct-solve
fallbacks remain visible in performance comparisons. First-cycle tests do not
prove convergence for every later cycle.

## Input features

`model.feature_mode: original` is the default. Node features are log relative
diagonal, normalized center and coarse membership. Edge features retain signed
row-normalized coefficients Aij/di. All stored A connections, including coarse
fill, are represented. The support graph additionally uses reference interpolation
values as edge features. No directional nine-slot grid tensor or CNN checkpoint
is used.

`row9` removes the redundant center and diagonal self-edges; `symmetric9` also
changes edge normalization to Aij/sqrt(di*dj). These are optional graph feature
experiments, not validated improvements. They do not select another discretization.
Change the mode in a copied configuration and use a new run directory.

## Lenovo WSL update and smoke

For the existing environment at the Windows Desktop project folder, stop its
training process and run the following in the VS Code WSL terminal. The migration
requires the source run's NumPy/SciPy/PyTorch/PyAMG versions, so update the editable
project without upgrading its installed dependencies.

```bash
cd /mnt/c/Users/user/Desktop/adaptive-multigrid-neural-amg
source .venv/bin/activate

git status --short
git fetch origin
git switch neural-amg
git pull --ff-only origin neural-amg
python -m pip install --no-deps -e .

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
set -o pipefail
mkdir -p logs
```

Use a new smoke directory:

```bash
SMOKE="artifacts/neural_amg_p1_v3_smoke_lenovo"
python -u scripts/run_neural_amg.py train \
  --config configs/neural_amg_smoke.json --run-dir "$SMOKE" \
  2>&1 | tee logs/cancellation_smoke_train.log
python scripts/run_neural_amg.py freeze \
  --config configs/neural_amg_smoke.json --run-dir "$SMOKE"
python -u scripts/run_neural_amg.py evaluate \
  --config configs/neural_amg_smoke.json --run-dir "$SMOKE" \
  2>&1 | tee logs/cancellation_smoke_evaluate.log
```

Smoke runs on CPU FP64 and verifies execution. It does not establish research
speed improvement. For an entirely new installation, create `.venv` with
`python3 -m venv .venv`, activate it and install `python -m pip install -e '.[dev]'`.
The complete test command is `python -m pytest -ra`.

## Initialize from the interrupted Lenovo checkpoint

The helper supports source revision **d5257e2** with its v2 configuration and
original runtime versions. It checks source/configuration/checkpoint identities,
P1 meshes, operator hashes, development splits and prior recovery audits. The
source must have completed supervised training and must not be frozen/evaluated
or contain assembled holdout cases.

```bash
OLD_RUN="artifacts/neural_amg_p1_560_seed6101_lenovo_teacher_l1_fix"
RUN="artifacts/neural_amg_p1_560_seed6101_lenovo_cancellation"
CFG="configs/neural_amg_research.json"

python scripts/start_neural_amg_cancellation.py \
  --run-dir "$OLD_RUN" --output-dir "$RUN" --config "$CFG"

python -u scripts/run_neural_amg.py train \
  --config "$CFG" --run-dir "$RUN" --resume \
  2>&1 | tee logs/research_train_cancellation.log
```

This creates a **new experiment** using the latest model weights from the old
checkpoint. For the reported crash it records 11,060 source updates, then starts
8 complete fine-tuning epochs over the same 560 training operators (4,480 new
case visits). The new counter starts at zero. Optimizer/RNG state and validation
selection restart; the initial validation is remeasured under the new solver.
Old teacher labels are not copied. Old config, identity, validation history and
recovery audits are archived under `initialization/` and linked by
`cancellation_start.json`. The original folder is unchanged. This is a warm-start
experiment, distinct from training all v3 teacher labels and weights from scratch.

If interrupted after initialization, rerun only the training command with
`--resume`; the initializer requires a nonexistent output directory. After
opening a new WSL terminal, repeat environment activation/thread exports and
set `RUN` and `CFG` again, then use:

```bash
RUN="artifacts/neural_amg_p1_560_seed6101_lenovo_cancellation"
CFG="configs/neural_amg_research.json"
python -u scripts/run_neural_amg.py train \
  --config "$CFG" --run-dir "$RUN" --resume \
  2>&1 | tee -a logs/research_train_cancellation.log
```

The previous bootstrap-depth and feasible-initialization recovery procedures
belong to the archived
[v2 runbook at d5257e2](https://github.com/slwjackie/adaptive-multigrid/blob/d5257e2db441600f1afe95de4355b0be38c2443a/docs/neural_amg.md).
They are pinned to historical sources; use the v3 initializer above for the
current cancellation experiment.

## Freeze and final/OOD evaluation

After training has completed successfully, freeze its newly selected checkpoint
and run actual final/OOD measurements:

```bash
python scripts/run_neural_amg.py freeze \
  --config "$CFG" --run-dir "$RUN"
python -u scripts/run_neural_amg.py evaluate \
  --config "$CFG" --run-dir "$RUN" \
  2>&1 | tee logs/research_evaluate_cancellation.log
```

To resume interrupted evaluation:

```bash
python -u scripts/run_neural_amg.py evaluate \
  --config "$CFG" --run-dir "$RUN" --resume \
  2>&1 | tee -a logs/research_evaluate_cancellation.log
```

Freeze binds the selected weights, source/runtime/configuration, development
manifests, validation history, initialization audit and future holdout
specifications. Final/OOD data are assembled only after freeze. Frozen runs
cannot be trained. A protocol/source/runtime change requires a distinct
experiment rather than editing identity hashes to bypass resume checks.

## Fresh v3 research training

For a new experiment with v3 teacher labels and all 15 supervised plus 8
fine-tuning epochs, choose a separate output folder and omit initialization:

```bash
FRESH="artifacts/neural_amg_p1_560_seed6101_lenovo_cancellation_scratch"
CFG="configs/neural_amg_research.json"
python -u scripts/run_neural_amg.py train \
  --config "$CFG" --run-dir "$FRESH" \
  2>&1 | tee logs/research_train_cancellation_scratch.log
```

Resume this run with the same command and `--resume`, using `tee -a`. After
completion, set `RUN="$FRESH"` for the freeze/evaluate commands above.

The research configuration uses 560 TRAIN operators (four densities × seven
families × twenty cases), 280 validation, 280 final and six independently seeded
OOD groups. Model seed is 6101 and teacher lambda is 10. Final/OOD data do not
select architecture, regularization, features or checkpoints. `prepare` can
assemble development data separately, followed by `train --resume`.
`train --max-updates N` stops after N processed cases for resume checks.

Every processed case saves the model, optimizer, phase/epoch/offset, best
checkpoint and RNG state. Teacher caches and data have checksums. The initial
`validation update=0` measures the initialized model. `mean_total_ms` includes
the configured 300-second penalty per failed validation operator.

## Mesh, data and OOD meanings

Every data configuration must declare both `kind` and `discretization` as
`p1_unstructured`. Other kinds and per-split discretization overrides are rejected.
The `sizes` value n controls initial point density: default interior unknown count
is n², but positions and element connectivity vary with the mesh seed. The NN and
solver never reshape this matrix into an n×n grid.

| Split | Change |
| --- | --- |
| final | New coefficient and mesh seeds in the development ranges |
| ood_grid | Larger unstructured meshes (n=255 in research) |
| ood_rotation | Held-out diffusion tensor angles |
| ood_anisotropy | Stronger anisotropy |
| ood_contrast | Larger coefficient contrast |
| ood_topology | Different spatial shapes of the coefficient field |
| ood_mesh | Stronger point jitter and spatial grading, changing mesh geometry/connectivity |

Seven coefficient families are near-isotropic, aligned anisotropy, rotated
anisotropy, local patch, checkerboard, channel and variable coefficient.
Discontinuous coefficient interfaces can cut triangles. Seven-point quadrature
approximates those element integrals; this is not interface-fitted meshing.

Each operator stores `*.A.npz`, `*.b.npy` and `*.mesh.npz`. Mesh files contain
`points`, `triangles`, `boundary` and `interior` arrays. Manifests bind mesh file
hashes, canonical mesh digests, mesh seeds, FEM specifications and quality
statistics. Cached mesh or DOF-map changes reject resume/evaluation. Generated
final/OOD meshes and matrices are assembled only after freeze. Exact normalized
operator duplicates are rejected across splits.

## Timing and evidence

Validation chooses failure count first, then mean per-case median setup-plus-solve
seconds. Graph construction, NN inference, Galerkin products and terminal
factorization count toward setup. Same-hierarchy warm solves are measured separately.
FEM data assembly, disk loading, offline label creation and offline training are
excluded for every method; timing measures the linear-solver task given A and b.
Fine-tuning optimizes residual reduction plus cancellation, not a differentiable wall clock.

Evaluation compares classical, numerical BAMG+EM teacher and learned interpolation
under the same controls. Each method coarsens its own actual generated coarse
matrices, so deeper C/S may differ. Teacher cold timing includes numerical teacher
optimization. Failures, cycles, residual histories and repeated cold/warm timings
are recorded, with evaluation resumable by case/method.

`python -m pytest -ra` runs the complete current-branch tests, including P1 FEM
accuracy, algebraic constraints, finite-difference gradients, actual P1 multilevel
runtime parity, source/mesh/checkpoint identities and resume. These checks do not
replace a complete v3 research training and timing evaluation.
