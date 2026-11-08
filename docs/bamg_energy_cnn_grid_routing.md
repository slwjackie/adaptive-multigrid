# Grid-based selection among three hybrid-teacher CNNs

This separate experiment trains Weight CNNs from hybrid-teacher labels with
lambda **10, 30 and 50**. It learns a grid-to-CNN selection rule from development
validation, freezes that rule with all checkpoints, and evaluates the unchanged
rule on final/OOD. A single CNN is selected from the **finest input grid** before
constructing the problem's MG hierarchy. The selected CNN supplies the learned P
throughout that hierarchy; this experiment does not switch CNNs between levels.

The entry point is `scripts/run_v6_7_bamg_energy_cnn_routing_study.py`. Existing
single-lambda studies remain separate experiments. These configurations and
execution checks do not establish that routing is faster, that any lambda is
optimal, or that the training budget reaches convergence. The full research
training and final/OOD benchmarks must be run on the target machine.

## Shared inputs and training

- `--source-study` is the verified copied Mac study. Warm start imports only its
  development-selected Weight CNN tensors; every layer remains trainable and a
  new optimizer is created. Scratch begins with random weights.
- `--teacher-comparison` is a completed, verified numerical teacher comparison.
  Its vector-bank, support, constraints, optimizer and MG settings are inherited.
  The historical numerical winner remains input provenance: it does **not**
  choose the CNN router's lambda. Each declared candidate is trained even though
  only one lambda won the numerical comparison.
- All lambdas and initialization arms use exactly the same newly prepared TRAIN
  and validation operators. Each lambda has its own teacher labels and cache.
  The selected-pass BAMG vector bank and reference support are checked to be
  identical across candidates for each TRAIN operator.
  Labels are loaded one lambda at a time rather than retaining all three label
  sets together. Development operators and the current labels still occupy
  memory together, so laptop resource use needs measurement.
- Architecture, loss, learning rates and epoch budgets are matched. Research
  configurations retain seeds 6101, 6102 and 6103 for both initialization arms:
  **3 lambdas × 2 initializations × 3 seeds = 18 training runs**. No favorable
  seed is selected to represent an arm. Both supervised and validation-selected
  fine-tuned checkpoints are evaluated as fixed-lambda controls.
- Epoch shuffle visits every TRAIN operator exactly once per completed epoch.
  Paired runs use the same problem order, and interrupted training resumes the
  saved epoch position, optimizer and random-number state. Updates after the last
  durable checkpoint may repeat. The deployed checkpoint can come from an earlier
  validation-selected epoch, including the supervised starting point.

| Setting | Initial study | Expanded study |
| --- | ---: | ---: |
| TRAIN grids | 15, 31, 63, 127 | 15, 31, 63, 127 |
| TRAIN per grid/family | 20 | 40 |
| Total TRAIN operators | 560 | 1,120 |
| Full development validation | 280 | Same 280 |
| Checkpoint-selection subset | 56 | Same 56 |
| Supervised epochs / updates per run | 15 / 8,400 | 15 / 16,800 |
| Residual fine-tune epochs / updates per run | 8 / 4,480 | 8 / 8,960 |
| Fine-tune 6-to-12-cycle switch after update | 2,240 | 4,480 |
| Research training runs | 18 | 18 |

The development generator retains balanced angles/checker counts, stratification
of log epsilon, log contrast and geometric variables, and deliberate hard
small-epsilon/high-contrast combinations within the existing TRAIN ranges.
The 560-case TRAIN pool is nested in the 1,120-case pool when the latter is linked
with `--development-study`. Larger data means more optimizer updates under equal
epoch budgets; this is not an equal-computation data-size comparison. Larger
training sets also do not guarantee better CNN or solver performance.

## Validation-only grid selection

Routing applies to the **fine-tuned** exports. The full 280-case validation has
70 operators at each of n=15,31,63,127 (seven families, ten operators per cell).
The 56-case subset selects checkpoints during each training run; it does not
replace the full validation used to select the routing rule.

A separate rule is selected for scratch and warm start. Within an initialization
arm, one shared mapping is used by all three seeds. For each grid, candidate
lambdas are ranked by failure rate first, then by the arithmetic mean of per-case
median warm solve times across all prespecified seeds. Failed/missing solves
retain their declared penalty. Selection requires complete observations; a
least-failing choice is not labeled eligible if it exceeds the allowed failure
rate. Equal ranking values have a deterministic lambda tie break.

The same full validation also selects one globally fixed lambda per
initialization arm. This is the main control for asking whether grid routing adds
anything beyond choosing a good single CNN family. A rule fitted to validation
can look better on that same validation by construction; final/OOD is the
independent assessment. Inspect problem-level and seed-level variation before
interpreting small timing differences as stable advantages.

The declared unseen-grid policy is `lower_anchor_clamp`: select the largest
validation grid no greater than the input n, clamping below the smallest and
above the largest grid. Consequently **n ≥ 127, including the unseen n=255,
uses the lambda chosen at n=127**. This is a preregistered extrapolation rule,
not evidence that this lambda is optimal on n=255. Reading a final/OOD input's n
is allowed; reading final/OOD solve results to change this mapping is not.

Routed validation is measured as a separate method after selecting the rule. It
runs the selected CNN and actual solver; its reported timing is not a per-problem
minimum stitched from the three fixed-lambda measurements. This routed validation
is a separate pass, so small fixed-versus-routed timing differences can reflect
run-to-run drift; it is not a paired same-pass speedup claim. Final/OOD measures
all fixed and routed methods together in the same paired case/repetition protocol.
At application time, only one CNN is evaluated. Three training runs are needed per initialization/seed,
but three online CNN evaluations or trial solves are not required.

## Controls, timing and frozen final/OOD

The comparison retains:

- `fixed_C`, `em5_C`, and the numerical `bootstrap_C` solver;
- the unchanged old CNN;
- each lambda's scratch/warm supervised and fine-tuned checkpoints, for every seed;
- scratch and warm routed fine-tuned CNNs, for every seed;
- paired routed-versus-validation-global-fixed comparisons within each arm.

`bootstrap_C` means the numerical BAMG hierarchy, not statistical bootstrap
resampling. Its setup/solve failures remain failures with penalties; no classical
fallback hides them. Statistical operator bootstrap intervals in held-out reports
are a separate uncertainty calculation.

Warm timings exclude model loading, CNN inference, P construction, Galerkin
products, factorization and warmup. A small grid-selection branch does not imply
that cold or total setup-plus-solve time improves. Storage, loading and offline
training costs increase with the candidate count. All methods retain the same
residual tolerance, zero initial iterate, common RHS and no-fallback protocol.
The new configs set `benchmark.cache_hierarchies=false`: validation releases a
method's hierarchy after use instead of retaining all candidate hierarchies. A
hierarchy may be rebuilt for another repetition; that setup remains outside the
warm timer and is recorded. This reduces benchmark memory at the cost of more
setup work.

Before labels are generated, the same expanded holdout plan is declared: **1,270
independent operators** across within-support final, grid OOD, rotation OOD,
anisotropy OOD, contrast OOD and topology OOD. Research evaluation uses
manufactured, random and localized RHS with three repetitions. The unseen grid
n=255, withheld angles, more extreme coefficient ranges and new topology remain
outside TRAIN and development validation. See the
[distribution specifications](bamg_energy_cnn_initialization_comparison.md#expanded-finalood-protocol).

`freeze` checks training completion and the raw development validation evidence,
recomputes the routing choice, and pins the configuration, code/environment,
prepared data, all checkpoint identities and grid mapping. Final/OOD applies
these choices unchanged and retains all prespecified seeds. It reports the fixed
models alongside the routed solver, but never chooses a winner from test results.
The single-use claim and compact per-case checkpoints allow interrupted final/OOD
evaluation to resume without retiming completed observations. A completed final
run is sealed by hashes. Final/OOD does not certify arbitrary PDE, nonlinear,
unstructured-grid or real CFD generalization.

## Lenovo WSL execution

Keep earlier numerical/CNN checkouts unchanged for their source-bound resume.
Create a new checkout for this experiment:

```bash
cd /mnt/c/Users/user/Desktop/adaptive-multigrid-bamg
git fetch origin residual-u-net-teacher &&
git worktree add --detach ../adaptive-multigrid-cnn-routing FETCH_HEAD

cd /mnt/c/Users/user/Desktop/adaptive-multigrid-cnn-routing
source ../adaptive-multigrid-bamg/.venv-bamg/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
set -o pipefail
mkdir -p artifacts/logs
```

Use the existing copied Mac package and completed numerical sweep. If the sweep
was saved under another name, replace its path consistently in every command.
No Mac files need to be copied again. Start with 560 TRAIN operators:

```bash
/usr/bin/time -v python -u scripts/run_v6_7_bamg_energy_cnn_routing_study.py run \
  --config configs/v6_7_bamg_energy_cnn_routing_q1.json \
  --source-study ../adaptive-multigrid-bamg/artifacts/initial_weights/mac_bamg \
  --teacher-comparison ../adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_lambda_sweep_q1 \
  --output artifacts/bamg_energy_cnn_routing_560_q1 \
  2>&1 | tee artifacts/logs/bamg_energy_cnn_routing_560_q1.log
```

After an interruption, restore the same checkout, environment and thread
variables, then use the same input/output arguments with `--resume`:

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_routing_study.py run \
  --config configs/v6_7_bamg_energy_cnn_routing_q1.json \
  --source-study ../adaptive-multigrid-bamg/artifacts/initial_weights/mac_bamg \
  --teacher-comparison ../adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_lambda_sweep_q1 \
  --output artifacts/bamg_energy_cnn_routing_560_q1 \
  --resume \
  2>&1 | tee -a artifacts/logs/bamg_energy_cnn_routing_560_q1.log

python scripts/run_v6_7_bamg_energy_cnn_routing_study.py report \
  --output artifacts/bamg_energy_cnn_routing_560_q1
```

Completed label cases, completed training runs and saved benchmark measurements
are reused. Changing lambda candidates, routing rules, code, data, budgets or
inputs requires a new output; it is not an exact resume. Source and environment
checks reject mismatches. No runtime or RAM requirement is promised. Inspect
`runtime_metrics.json` and the `/usr/bin/time -v` log, together with convergence
histories, before choosing a larger run. Phase RSS is the process lifetime high
water mark, not an additive estimate of individual phase allocations.

To compare 1,120 TRAIN operators, finish the 560-case development run first and
keep final/OOD unopened. Use a **new output** linked to that routing study:

```bash
/usr/bin/time -v python -u scripts/run_v6_7_bamg_energy_cnn_routing_study.py run \
  --config configs/v6_7_bamg_energy_cnn_routing_1120_q1.json \
  --source-study ../adaptive-multigrid-bamg/artifacts/initial_weights/mac_bamg \
  --teacher-comparison ../adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_lambda_sweep_q1 \
  --development-study artifacts/bamg_energy_cnn_routing_560_q1 \
  --output artifacts/bamg_energy_cnn_routing_1120_q1 \
  2>&1 | tee artifacts/logs/bamg_energy_cnn_routing_1120_q1.log
```

For resume, keep `--development-study`, add `--resume`, and use `tee -a`. The
reference and expanded study must have the same source, numerical comparison,
candidate lambdas and routing/training protocol apart from the scaled data and
update counts. Shared validation identities and TRAIN nesting are checked. The
larger scratch run remains a new randomly initialized run; it does not load the
560-case scratch checkpoint.

Choose dataset size and training budget on development validation before final.
The following example freezes the 560-case study; substitute the 1,120 output
consistently if that is the chosen study:

```bash
python scripts/run_v6_7_bamg_energy_cnn_routing_study.py freeze \
  --output artifacts/bamg_energy_cnn_routing_560_q1

python -u scripts/run_v6_7_bamg_energy_cnn_routing_study.py evaluate-final \
  --output artifacts/bamg_energy_cnn_routing_560_q1 \
  2>&1 | tee artifacts/logs/bamg_energy_cnn_routing_560_q1_holdout.log
```

Optionally add `--max-cases 10` to pause after at most ten newly completed
operators. Resume and inspect groups with:

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_routing_study.py evaluate-final \
  --output artifacts/bamg_energy_cnn_routing_560_q1 --resume \
  2>&1 | tee -a artifacts/logs/bamg_energy_cnn_routing_560_q1_holdout.log

python scripts/run_v6_7_bamg_energy_cnn_routing_study.py report-final \
  --output artifacts/bamg_energy_cnn_routing_560_q1 --group final
python scripts/run_v6_7_bamg_energy_cnn_routing_study.py report-final \
  --output artifacts/bamg_energy_cnn_routing_560_q1 --group ood_grid
python scripts/run_v6_7_bamg_energy_cnn_routing_study.py report-final \
  --output artifacts/bamg_energy_cnn_routing_560_q1 --group ood_topology/n127/channel
```

Reporting reads saved measurements. Group keys include a split, `n31`, a family,
`split/n31`, `split/family`, or `split/n31/family`. If subsequent development uses
these test results to change the routing rule or model, a new independent test
protocol is needed for the resulting model.

## Main output files

| Path under the output directory | Content |
| --- | --- |
| `inputs/initial_weight.pt` | Verified old CNN initialization |
| `data/` | Shared TRAIN/validation specifications and provenance |
| `labels/lambda_10/teacher/train/` (also 30, 50) | Lambda-specific per-case labels and manifest |
| `teacher_bank_manifest.json` | Cross-lambda equality audit of selected vectors, reference operators and support |
| `runs/lambda_10_scratch_seed6101/` etc. | Resumable training, histories and selected checkpoints |
| `runtime_metrics.json` | Phase wall times and process peak RSS |
| `validation/benchmark_cases/` | Fixed-lambda and baseline validation measurements |
| `routing_selection.json` | Validation-selected maps and globally fixed lambda controls |
| `validation_routed/benchmark_cases/` | Actual routed validation measurements |
| `comparison.json` | Full development comparison and all prescribed runs |
| `holdout_plan.json` | Distribution plan committed before label generation |
| `research_freeze.json` | Immutable choices, model identities and supporting evidence |
| `final/data/` | Deterministic held-out case records and operator/RHS hashes |
| `final/benchmark/benchmark_cases/` | Compact per-case final/OOD measurements |
| `final/report.json` | Grouped results, cohorts and uncertainty summaries |
| `final/claim.json` | Single-use evaluation and completed-evidence hashes |

The smoke configuration retains all three lambda candidates but uses tiny n=7
problems, one seed and short training budgets. It exercises workflow and resume
behavior, not numerical superiority or generalization. All routing configurations use epoch shuffle. Research configs additionally use
stratified expanded development data and three seeds.
