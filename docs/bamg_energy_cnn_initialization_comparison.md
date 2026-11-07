# Shared hybrid-teacher labels: scratch versus warm start

The numerical comparison in `run_v6_7_bamg_energy_study.py` does not train a CNN.
This new entry point takes its selected lambda, generates fresh TRAIN labels,
and compares two ways to initialize the same fixed-support Weight CNN. It never
runs final/OOD evaluation or changes the original Mac study.

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
  Files under the output's parent and explicit `historical_roots` are also scanned
  for historical manifests. Only hashes/metadata from old final/OOD are used.
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
  can equal that starting model; the full 56-case validation may rank it differently.
  If no subset checkpoint is eligible, the least-failing candidate is still
  exported with `eligible=false`, and full-validation failures remain visible.
- Validation uses identical A/RHS, zero initial iterates, tolerance
  `max(1e-12, 1e-8 * initial_residual)`, three repeats, one RHS, no fallback.
  It averages per-case median warm solve times; setup/inference are excluded.
  Failed solves cost at least the declared penalty and at least observed time.
  Missing solves retain the penalty; incomplete results cannot select a winner.
- Results contain individual seeds and their aggregate mean/std. All prespecified
  seeds remain in the comparison; a favorable seed is not presented as the whole arm.

The research configuration has 280 TRAIN operators, 56 validation operators,
4,000 supervised updates and 2,000 fine-tune updates per arm/seed. These are
**fixed budgets, not a convergence certificate**. Inspect loss histories, warm
validation curves and best-update locations. If scratch is still improving, use
a new output directory and a longer preregistered configuration before concluding
warm start is intrinsically better. Changing the budget of an existing run is not
an exact resume. Warm start also benefited from historical training compute, so
this compares adaptation strategies rather than equal lifetime training costs.
The lambda that produced the best numerical teacher is also not guaranteed to
produce the best CNN after approximation and residual fine-tuning.

## Lenovo WSL commands

Keep the existing numerical-comparison checkout for its source-hash-bound resume.
Create a separate checkout for the new entry point:

```bash
cd /mnt/c/Users/user/Desktop/adaptive-multigrid-bamg
git fetch origin codex/hybrid-cnn-initialization-comparison
git worktree add --detach ../adaptive-multigrid-cnn-compare FETCH_HEAD

cd /mnt/c/Users/user/Desktop/adaptive-multigrid-cnn-compare
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

Run using the original completed teacher comparison (currently selects lambda 10):

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_study.py run \
  --config configs/v6_7_bamg_energy_cnn_compare_q1.json \
  --source-study ../adaptive-multigrid-bamg/artifacts/initial_weights/mac_bamg \
  --teacher-comparison ../adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_q1 \
  --output artifacts/bamg_energy_cnn_compare_q1 \
  2>&1 | tee artifacts/logs/bamg_energy_cnn_compare_q1.log
```

If the expanded lambda sweep has finished, replace only the comparison argument
with `../adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_lambda_sweep_q1`
**before the first invocation**, using a fresh output if a prior run exists.
Its selected lambda is read automatically; it is never silently fixed at 10.

After interruption, activate the same venv, set the same thread variables, then:

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_study.py run \
  --config configs/v6_7_bamg_energy_cnn_compare_q1.json \
  --source-study ../adaptive-multigrid-bamg/artifacts/initial_weights/mac_bamg \
  --teacher-comparison ../adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_q1 \
  --output artifacts/bamg_energy_cnn_compare_q1 \
  --resume \
  2>&1 | tee -a artifacts/logs/bamg_energy_cnn_compare_q1.log
```

Keep the same comparison path if you originally used the expanded sweep.
Completed label cases, completed arms and saved benchmark measurements are reused.
Training resumes its latest durable optimizer/RNG checkpoint; updates since its
last save may repeat. A partial label case is rebuilt. Source/configuration/input
or environment mismatches fail explicitly instead of silently mixing experiments.

```bash
python scripts/run_v6_7_bamg_energy_cnn_study.py report \
  --output artifacts/bamg_energy_cnn_compare_q1
```

Important outputs:

| Path under output | Content |
| --- | --- |
| `inputs/initial_weight.pt` | Verified unchanged old CNN weights |
| `data/` | Shared fresh development specifications and manifests |
| `teacher/train/` | Per-case hybrid labels, hashes and optimizer diagnostics |
| `runs/scratch_seed6101/` etc. | Latest resumable checkpoints, compact histories, selected deployment |
| `validation/benchmark_cases/` | Compact per-case scalar timings and failures |
| `comparison.json` | Full development comparison and aggregate seed results |

The smoke configuration uses two n=7 families, one seed and two updates per phase;
it verifies execution only and cannot establish numerical superiority. Neither
configuration consumes a new final/OOD test. That requires a separate freeze and
evaluation after development choices are settled.
