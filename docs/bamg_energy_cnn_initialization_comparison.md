# Shared hybrid-teacher labels: scratch versus warm start

The numerical comparison in `run_v6_7_bamg_energy_study.py` does not train a CNN.
This new entry point takes its selected lambda, generates fresh TRAIN labels,
and compares two ways to initialize the same fixed-support Weight CNN. A separate
freeze and final/OOD path tests the completed models on a preregistered expanded
holdout. The original Mac study and numerical comparison remain unchanged.

## Inputs and protocol

- `--source-study`: the copied Mac package. The selected deployment comes from
  `data/research_freeze.json`, not a final-test score. The three checkpoint SHA256s,
  original data-plan/development/baseline hashes, freeze configuration, and final
  metadata claim (when present) are checked without following obsolete Mac paths.
- `--teacher-comparison`: a **completed** numerical comparison, including its
  `cases/*/results.json`, configuration, manifest and summary. The selection is
  recomputed from raw results. Its existing source files must match the numerical
  kernels in this checkout; additive new files are allowed. A completed lambda
  sweep (including 5, 30, 50) can be supplied instead of the original comparison.
- One new TRAIN split and one new validation split are generated and saved once.
  Old TRAIN, old validation, previously exposed final/OOD operator hashes, and
  numerical lambda-selection operators are excluded from both new splits.
  The first study scans the output's parent and explicit `historical_roots` for
  historical manifests. A larger study in the same development cohort explicitly
  reuses the verified exclusion snapshot with `--development-study`; it verifies
  identical validation and nested TRAIN cases. That deliberate shared development
  data is not treated as historical leakage. Only hashes/metadata from old
  final/OOD are used.
- The selected lambda, vector-bank settings and optimizer budget are inherited
  from the numerical comparison. Labels retain the original selected-pass BAMG
  vectors, but each new P rebuilds its actual `P.T @ A @ P` and mass at every level.
  Fixed bilinear reference scales, support, row sums, injection and signed L1 cap
  are preserved. One cached label set is used by every model seed and arm.
- Architecture: hidden 8, depth 5, ReLU, geometry features, early mask conditioning.
  Scratch uses random weights. Warm start imports only the validation-selected
  Mac Weight CNN tensors, with a new optimizer; every layer remains trainable.
- Three paired seeds (6101/6102/6103) use the same supervised loss, sampling seed,
  learning rates, training budgets, residual objective and selection criteria.
  Data generation has a separate seed. Hard probes use each model's current
  hierarchy, so paired random streams do not imply identical transformed probes.
- Each arm first learns the new numerical labels, then fine-tunes the actual MG
  residual with uniform cycle weighting (6 to 12 cycles). Both the supervised
  export and the validation-selected fine-tuned export are benchmarked, together
  with the unchanged old CNN, `fixed_C`, and `em5_C`.
  Fine-tune checkpoint selection uses the fixed stratified validation subset,
  including the supervised starting point (update 0). Thus a fine-tuned export
  can equal that starting model; the full 280-case validation may rank it differently.
  If no subset checkpoint is eligible, the least-failing candidate is still
  exported with `eligible=false`, and full-validation failures remain visible.
- Validation uses identical A/RHS, zero initial iterates, tolerance
  `max(1e-12, 1e-8 * initial_residual)`, three repeats, one RHS, no fallback.
  It averages per-case median warm solve times; setup/inference are excluded.
  Failed solves cost at least the declared penalty and at least observed time.
  Missing solves retain the penalty; incomplete results cannot select a winner.
- Results contain individual seeds and their aggregate mean/std. All prespecified
  seeds remain in the comparison; a favorable seed is not presented as the whole arm.

## Complete-epoch training and expanded development data

The default research configuration now uses 560 TRAIN operators. The optional
`v6_7_bamg_energy_cnn_compare_1120_q1.json` extends the same planned training pool
to 1,120 operators. Both use the same independent 280-case development validation
and 56-case checkpoint-selection subset when run in the shared development cohort.

| Setting | Default 560 | Expanded 1,120 |
| --- | ---: | ---: |
| TRAIN grids | 15, 31, 63, 127 | 15, 31, 63, 127 |
| TRAIN per grid/family | 20 | 40 |
| Full validation (four grids, seven families, ten each) | 280 | 280 |
| Checkpoint validation (four grids, seven families, two each) | 56 | 56 |
| Supervised epochs / updates | 15 / 8,400 | 15 / 16,800 |
| Residual fine-tune epochs / updates | 8 / 4,480 | 8 / 8,960 |
| Switch from 6 to 12 cycles after update | 2,240 | 4,480 |
| Warm checkpoint validation interval | 560 updates | 1,120 updates |

`sampling: "epoch_shuffle"` permutes all TRAIN indices independently each epoch.
Each problem is used exactly once per completed epoch, with one optimizer update
per problem. The same arm/seed pair uses the same problem order; the sampling
stream is independent of error-probe randomness. Interrupted training restores
optimizer/probe RNG progress and reconstructs the exact remaining epoch order
from the saved update count and immutable sampling identity. A partially completed
epoch is not restarted from its first problem. Work after the latest durable
checkpoint can be repeated, as with any checkpoint resume.

The update budgets use **15 complete supervised epochs and 8 complete residual
epochs**, rather than the approximate 16,000/8,000-update suggestion for 1,120 cases.
This gives every case the same number of visits in each completed training phase.
The latest completed weights have those visits; the deployed checkpoint can come
from an earlier validation-selected epoch, including update zero. The cycle curriculum
switches after four residual epochs, and checkpoint selection runs at update zero
and each epoch boundary. This compares equal data passes; the larger dataset has
more updates and more computation. It is not an equal-wall-time data-size ablation.

`data.sampling: "stratified_nested_v1"` introduces development-only stratification:

- TRAIN uses deterministic blocks of 20 cases per grid/family. The 20-case pool is
  a prefix of the 40-case pool; validation does not depend on the chosen TRAIN size.
- Rotation categories and checker counts are balanced to within one case where
  the category count does not divide the block size. Aligned anisotropy retains
  its 0/90-degree directions, and the other applicable families retain the five
  original TRAIN angles.
- Continuous parameters are stratified in their existing support. Epsilon and
  contrast are stratified in log space; patch position/size, channel width and
  smooth-field frequency/phase are also spread through their ranges.
- Applicable heterogeneous families deliberately include small-epsilon,
  high-contrast combinations within the declared TRAIN support. These cases are
  not removed if teacher optimization or a solver finds them difficult.
- Normalized-operator duplicate exclusion and historical TRAIN/validation/exposed
  test exclusions remain active. Only duplicates trigger deterministic resampling.

This improves coverage of existing coefficient-field families. It does not add
new channel topology or arbitrary PDE classes. The final set has the same parameter
support and family formulas, but its sampling design differs from this deliberately
stratified TRAIN design. In particular, n=255, held-out angle intervals, extreme
OOD epsilon/contrast ranges, and the twelve topology-OOD variants remain excluded
from TRAIN and validation.

Choose the dataset size from the common development validation, including
per-grid/family failures and warm solve times, **before freezing or evaluating
final/OOD**. Do not inspect the 560-case final/OOD results and then use them to
justify moving to 1,120 TRAIN cases. Three prespecified model seeds remain in
both arms; no seed is silently removed or selected for being fastest.

These are **fixed budgets, not a convergence certificate**. Inspect loss histories,
warm validation curves and best-update locations. If scratch is still improving,
use a new output directory and a longer preregistered configuration before
concluding warm start is intrinsically better. Changing data, sampling or budgets
in an existing run is not an exact resume. Preserve its checkout/configuration.
Warm start also benefited from historical training compute, so this compares
adaptation strategies rather than equal lifetime training costs. The lambda that
produced the best numerical teacher is not guaranteed to produce the best CNN.

Unlike streamed final/OOD evaluation, development operators and teacher labels
currently occupy memory together. Start with the 560-case study on Lenovo and
record elapsed time and peak resident memory before choosing the larger run.
Training convergence and laptop resource use must be measured; the counts alone
guarantee neither performance nor sufficient RAM.

## Expanded final/OOD protocol

`holdout_plan.json` is committed and hashed **before TRAIN label generation**.
The new generator is authoritative; `data/dataset_plan.json` mirrors its counts
and records that the legacy final generator is unused. Merely declaring the plan
does not assemble, inspect or solve any held-out A.

| Split | Interior grid n | Families | Independent A per grid/family | Total A |
| --- | --- | ---: | ---: | ---: |
| Final, within TRAIN parameter support | 15, 31, 63, 127 | 7 | 20 | 560 |
| OOD grid | 255 | 7 | 20 | 140 |
| OOD rotation | 31, 63, 127 | 5 | 10 | 150 |
| OOD anisotropy | 31, 63, 127 | 6 | 10 | 180 |
| OOD contrast | 31, 63, 127 | 4 | 10 | 120 |
| OOD topology | 31, 63, 127 | 4 | 10 | 120 |
| **Total** | | | | **1,270** |

The older research plan had 126 final/OOD operators. Counts here mean different
normalized operators, not different RHS, repeated timings, scalar rescalings or
different model seeds. Existing Mac TRAIN/validation/final/OOD, inherited earlier
history, numerical lambda-selection A's and new development A's are excluded.
Duplicates alone trigger deterministic resampling; hard cases or solve failures
are never discarded. Default grid OOD is also outside the old warm-start model's
development grid sizes. The configured fields can be changed only before a new
study; research final families must be represented in TRAIN.

Continuous parameters use independently jittered Latin-hypercube strata, and
discrete angles/geometries are balanced within grid/family cells. The final set
matches TRAIN's support and family formulas, but is stratified rather than IID
Monte Carlo. Each named OOD split changes one declared axis:

- Rotation uses jittered held-out angle intervals separated from TRAIN angles.
- Anisotropy uses epsilon in `[1e-5, 10^-3.1]`, outside TRAIN `[1e-3, 1e-1]`.
- Contrast uses `[10^3.2, 1e5]`, outside TRAIN `[10, 1000]`.
- Topology uses three new variants for each heterogeneous family: ellipse/annulus/L
  patches, shifted/rotated/stretched checker patterns, sinuous/crossing/branching
  channels, and oblique/Gaussian-island/radial smooth coefficient fields.

Axes apply to meaningful families only: rotation excludes near-isotropic/aligned,
anisotropy excludes near-isotropic, and contrast/topology use the four heterogeneous
families. Both final and grid OOD cover all seven families. All cases remain the
declared Q1, SPD, homogeneous-Dirichlet diffusion problem class.

Each A has three deterministic RHS: manufactured, independent random, and a smooth
localized source assembled with Q1 quadrature. Each is normalized to unit L2; every
method sees identical vectors. Three timing repetitions and all 15 prespecified
methods give **171,450 solve attempts**, in addition to training/setup. One operator
and one method hierarchy are held at a time, scalar results save after each solve,
and `--max-cases`/`--resume` permit staged execution.

`freeze` pins configuration, source/environment, development data, label identity,
all completed checkpoints, full validation evidence and lambda/model selections.
It closes training and selection. Only `evaluate-final` may materialize the heldout
operators, under the same process lock and a single-use claim. An identical
interrupted evaluation may resume; completed resume verifies sealed evidence and
does not repeat timings. Dataset records and benchmark shards are cross-checked.

Reports retain individual seeds and their all-seed arm averages. Each operator's
seed times are averaged before operator aggregation. Paired bootstrap intervals
resample A within split/grid/family cells; RHS/repeats/seeds do not increase the
independent sample count. The intervals are descriptive approximations conditional
on the prescribed seeds and synthetic sampling design, not a formal guarantee.
Failed-A counts, solve-attempt failures, missing results and penalties remain
visible; failed timings are not presented as reliable speedups. No final/OOD row
selects a winner. Inspect separate shifts and cells rather than treating the pooled
mean over these unequal distributions as the main generalization result.

This supplies stronger evidence about fresh instances from the **declared**
distributions. It does not by itself establish arbitrary PDE-family, nonlinear,
unstructured-mesh or real CFD generalization, and 1,270 successes would not prove
zero failure probability.

## Lenovo WSL commands

Keep the existing numerical-comparison checkout for its source-hash-bound resume.
Create a separate checkout for the new entry point:

```bash
cd /mnt/c/Users/user/Desktop/adaptive-multigrid-bamg
git fetch origin residual-u-net-teacher &&
git worktree add --detach ../adaptive-multigrid-cnn-expanded FETCH_HEAD

cd /mnt/c/Users/user/Desktop/adaptive-multigrid-cnn-expanded
source ../adaptive-multigrid-bamg/.venv-bamg/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
set -o pipefail
mkdir -p artifacts/logs
```

The script explicitly imports this checkout's `src`. Existing venv dependencies
can be reused; no checkpoint or Mac metadata needs to be moved again.

Start the 560-case study using the completed expanded teacher comparison
(currently selects lambda 10). `/usr/bin/time -v` records total elapsed time and
maximum resident set size in the same log:

```bash
/usr/bin/time -v python -u scripts/run_v6_7_bamg_energy_cnn_study.py run \
  --config configs/v6_7_bamg_energy_cnn_compare_q1.json \
  --source-study ../adaptive-multigrid-bamg/artifacts/initial_weights/mac_bamg \
  --teacher-comparison ../adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_lambda_sweep_q1 \
  --output artifacts/bamg_energy_cnn_560_q1 \
  2>&1 | tee artifacts/logs/bamg_energy_cnn_560_q1.log
```

If your completed sweep is stored elsewhere, use that directory consistently.
Its selected lambda is read automatically from verified raw cases; it is not
hardcoded. Use a new output for this version: changing code/holdout specifications
is not an exact resume of an older study. Keep any older running checkout intact.

After interruption, activate the same venv, set the same thread variables, then:

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_study.py run \
  --config configs/v6_7_bamg_energy_cnn_compare_q1.json \
  --source-study ../adaptive-multigrid-bamg/artifacts/initial_weights/mac_bamg \
  --teacher-comparison ../adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_lambda_sweep_q1 \
  --output artifacts/bamg_energy_cnn_560_q1 \
  --resume \
  2>&1 | tee -a artifacts/logs/bamg_energy_cnn_560_q1.log
```

Keep the same comparison path if you originally used the expanded sweep.
Completed label cases, completed arms and saved benchmark measurements are reused.
Training resumes its latest durable optimizer/RNG checkpoint; updates since its
last save may repeat. A partial label case is rebuilt. Source/configuration/input
or environment mismatches fail explicitly instead of silently mixing experiments.

```bash
python scripts/run_v6_7_bamg_energy_cnn_study.py report \
  --output artifacts/bamg_energy_cnn_560_q1
```

To compare against 1,120 TRAIN cases, first complete the 560-case development run.
Keep final/OOD unopened in both studies. Run the larger configuration in a **new**
output and explicitly link its completed development cohort:

```bash
/usr/bin/time -v python -u scripts/run_v6_7_bamg_energy_cnn_study.py run \
  --config configs/v6_7_bamg_energy_cnn_compare_1120_q1.json \
  --source-study ../adaptive-multigrid-bamg/artifacts/initial_weights/mac_bamg \
  --teacher-comparison ../adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_lambda_sweep_q1 \
  --development-study artifacts/bamg_energy_cnn_560_q1 \
  --output artifacts/bamg_energy_cnn_1120_q1 \
  2>&1 | tee artifacts/logs/bamg_energy_cnn_1120_q1.log
```

For interruption/resume, keep all arguments, including `--development-study`,
add `--resume`, and use `tee -a`. The two studies must use the same verified Mac
source and numerical teacher comparison. The larger study checks the shared
validation identities and TRAIN nesting; merely choosing equal data seeds does
not substitute for this verification. Each study has its own immutable label
cache and independent trained checkpoints; it does not resume the 560-case model
as the 1,120-case scratch arm.

```bash
python scripts/run_v6_7_bamg_energy_cnn_study.py report \
  --output artifacts/bamg_energy_cnn_1120_q1
```

Inspect the study's `runtime_metrics.json` together with the `/usr/bin/time -v`
log to assess resource use. Phase entries cover `prepare_data`, `prepare_labels`,
each arm, and validation, recording the latest invocation wall time, cumulative
invocation time, invocation count, and process peak RSS in MiB. Peak RSS is the
process high-water mark, not an additive measure of each phase. No runtime or RAM
requirement is promised for the research run.

Use the common development validation to choose the data size and training budget.
The following commands show freezing the 560-case study; if choosing 1,120, use
`artifacts/bamg_energy_cnn_1120_q1` consistently instead. Freeze only after these
choices are settled, then evaluate the selected study's final/OOD once:

```bash
python scripts/run_v6_7_bamg_energy_cnn_study.py freeze \
  --output artifacts/bamg_energy_cnn_560_q1

python -u scripts/run_v6_7_bamg_energy_cnn_study.py evaluate-final \
  --output artifacts/bamg_energy_cnn_560_q1 \
  2>&1 | tee artifacts/logs/bamg_energy_cnn_560_q1_holdout.log
```

Append `--max-cases 10` to evaluate at most ten newly completed operators per
invocation. Resume with exactly the same frozen study:

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_study.py evaluate-final \
  --output artifacts/bamg_energy_cnn_560_q1 --resume \
  2>&1 | tee -a artifacts/logs/bamg_energy_cnn_560_q1_holdout.log

python scripts/run_v6_7_bamg_energy_cnn_study.py report-final \
  --output artifacts/bamg_energy_cnn_560_q1 --group final
python scripts/run_v6_7_bamg_energy_cnn_study.py report-final \
  --output artifacts/bamg_energy_cnn_560_q1 --group ood_grid
python scripts/run_v6_7_bamg_energy_cnn_study.py report-final \
  --output artifacts/bamg_energy_cnn_560_q1 --group ood_topology/n127/channel
```

Other group keys are a split name, `n31`, a family name, `split/n31`,
`split/family`, or `split/n31/family`. Reporting reads saved scalar evidence;
it does not rerun solves. Tuning in response to these results consumes this test
set; a later developed model needs a newly declared, independent assessment.

Important outputs:

| Path under output | Content |
| --- | --- |
| `inputs/initial_weight.pt` | Verified unchanged old CNN weights |
| `data/` | Shared fresh development specifications and manifests |
| `runtime_metrics.json` | Preparation, label, arm and validation wall times and process peak RSS |
| `teacher/train/` | Per-case hybrid labels, hashes and optimizer diagnostics |
| `runs/scratch_seed6101/` etc. | Latest resumable checkpoints, compact histories, selected deployment |
| `validation/benchmark_cases/` | Compact per-case scalar timings and failures |
| `comparison.json` | Full development comparison and aggregate seed results |
| `holdout_plan.json` | Preregistered expanded distributions/counts/RHS protocol |
| `research_freeze.json` | Immutable checkpoints and validation-only choices |
| `final/data/` | Per-case deterministic parameters, operator/RHS hashes |
| `final/benchmark/benchmark_cases/` | Compact final/OOD scalar measurements |
| `final/report.json` | Split/grid/family reports, all-seed cohorts and intervals |
| `final/claim.json` | Single-use evaluation state and completed evidence hashes |

The smoke configuration uses two n=7 families, one seed and two updates per phase.
Its optional freeze/final workflow is explicitly labeled `smoke_only`; it verifies
execution and cannot establish generalization or numerical superiority.

## Implementation verification

The earlier expanded-holdout implementation passed **191 focused tests**, including fresh
process freeze/evaluate/resume/report, post-freeze training rejection, input/label
identity, lost-data provenance, all-seed aggregation and failure accounting.
An additional private-seed assembly check covers 47 cases across n=15/31/63/127/255,
all seven families and all twelve new topology variants. It checks finite/symmetric
operators, positive sampled quadratic forms, manufactured RHS consistency and RHS
normalization. It is a generator check, not a solver-performance result. The full
1,270-case research benchmark has not been run as part of this implementation.
