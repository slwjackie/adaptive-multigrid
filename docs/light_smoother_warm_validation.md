# Warm-only validation of the fitted subset run

`scripts/run_v6_7_light_smoother_warm.py` evaluates a completed subset fit without
retraining or changing any original numerical/source-manifest code. It requires
all eight arms, including the verified identical-P `frozen_P_line`, and evaluates
the complete validation cohort. Training-label exclusions never filter validation.

Defaults: `warm_multiple`, RHS counts 1 and 16, one measured repeat, one warmup.
For the research cohort this is 140 × 8 × 2 × 1 = **2,240 measured workloads**,
plus warmups. Each RHS-16 workload solves 16 distinct RHS with the same A and
prepared hierarchy. The same measured RHS are used across arms. Each workload
has an independent unadapted priming RHS; adaptation is reset before timing and
adopted coefficients may be reused across its timed RHS. Adaptation is not carried
from the RHS-1 measurement into the separately prepared RHS-16 measurement.

Warm timing excludes hierarchy construction, P inference, feature extraction,
coefficient prediction and independent priming. It includes adaptation, candidate
validation, rejected attempts and fallback during the timed solves. Online time
is already part of solve time: do not add it twice. Setup/priming still consume
real execution time, even though they are excluded from the warm metric.

This is a diagnostic, not a replacement for setup-inclusive model selection. It
does not write selection/deployment/freeze evidence or access final/OOD data.
One repeat can identify large effects but cannot establish a small speedup.

## Stop the old benchmark and start the diagnostic on macOS

In the terminal running the old benchmark, press **Control-C once** and wait for
the shell prompt. A `KeyboardInterrupt` traceback is expected. Completed records
are preserved; an interrupted measurement may need to run again on later resume.

```bash
cd "$HOME/Desktop/adaptive-multigrid-light"
source .venv/bin/activate
git pull --ff-only origin residual-u-net-teacher
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Keep the same Python environment and thread settings as the existing study. No
install, new prepare, search or fit is needed.

```bash
LIGHT_RUN="$(cat artifacts/light_subset_current.txt)"
WARM_SCRIPT="scripts/run_v6_7_light_smoother_warm.py"
caffeinate -i python -u "$WARM_SCRIPT" --run-dir "$LIGHT_RUN" --rhs-counts 1 16 --repeats 1 --warmups 1
```

All eight arms, including both online variants, are enabled automatically. The
original `benchmarks/validation/` is untouched. New outputs are saved under
`benchmarks/validation_warm_rhs1_16/`, which contains `progress.json`,
`repeat_records/`, final `comparison.csv/json`, `paired_effects.json`, cycle and
failure diagnostics, training-policy reports, and a separate provenance manifest.
The startup message reports the measurement count. During evaluation, monitor
from another terminal (no competing benchmark):

```bash
cd "$HOME/Desktop/adaptive-multigrid-light"
LIGHT_RUN="$(cat artifacts/light_subset_current.txt)"
cat "$LIGHT_RUN/benchmarks/validation_warm_rhs1_16/progress.json"
```

Before the first measured workload the progress file may not yet exist because
initial preparation and the first case's warmups run first.

Resume the warm diagnostic with exactly the same protocol:

```bash
LIGHT_RUN="$(cat artifacts/light_subset_current.txt)"
caffeinate -i python -u scripts/run_v6_7_light_smoother_warm.py --run-dir "$LIGHT_RUN" --rhs-counts 1 16 --repeats 1 --warmups 1 --resume
```

Input artifacts, models, original sources/environment and the supplemental runner
are checked on resume. Existing committed measurements are reused; unfinished
cases may repeat warmups. A changed measurement protocol requires a different
`--tag validation_warm_<name>` and fresh measurement records.

The original interrupted full benchmark can still be resumed separately with its
original script and exact flags. The warm diagnostic cannot be used to freeze a
deployment model or make an end-to-end performance claim.
