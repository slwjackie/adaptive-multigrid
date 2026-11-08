# Resume grid-routing CNN studies with bounded teacher-label loading

The original label loop retains every full `TeacherResult` for a lambda, including
large per-row diagnostics for the original BAMG setup passes. With 560 operators,
memory can be exhausted even though the last failing allocation is only a source
file read in `_source_hashes`. The original checkpoint hash helper also reads an
entire checkpoint at once.

The low-memory runner stores and verifies the same label packages, then retains
only an index of file paths, identities and hashes. It reads the current case
when auditing or training, and hashes files in bounded chunks. Teacher-group
construction is lazy too; this prevents the supervised trainer from rebuilding
the full resident collection. Mathematical labels, training objective, epoch
order, random-number state and optimizer settings are unchanged. Additional disk
reads can increase elapsed time, especially on a WSL-mounted Windows directory.

This bounds residency of the teacher-label collection, not the entire process.
TRAIN/validation operators, one numerical teacher's temporary work, CNN rollout
graphs and checkpoints still require memory. It is not a promise that an
arbitrarily small WSL memory limit can run the full experiment.

## Recover an interrupted run

Keep the same virtual environment, thread settings, input paths, configuration
and output directory. Fetch this additive fix in the routing worktree. It changes
none of the existing numerical source files or original launchers, so completed
teacher comparisons and the current study identity remain valid.

```bash
cd /mnt/c/Users/user/Desktop/adaptive-multigrid-cnn-routing
git fetch origin residual-u-net-teacher && git switch --detach FETCH_HEAD

: "${BAMG_SOURCE_STUDY:?Set the existing source study path}"
: "${BAMG_TEACHER_COMPARISON:?Set the completed numerical comparison path}"
mkdir -p artifacts/logs
set -o pipefail
python -u scripts/run_v6_7_bamg_energy_cnn_routing_low_memory.py run \
  --config configs/v6_7_bamg_energy_cnn_routing_q1.json \
  --source-study "$BAMG_SOURCE_STUDY" \
  --teacher-comparison "$BAMG_TEACHER_COMPARISON" \
  --output artifacts/bamg_energy_cnn_routing_560_q1 \
  --resume \
  2>&1 | tee -a artifacts/logs/bamg_energy_cnn_routing_560_q1.log
```

Validated, indexed teacher `.pt` files are reused unchanged. The case interrupted
before an atomic save is recomputed. Existing arm checkpoints follow the original
resume contract, including replay of updates after the last durable checkpoint.
Do not delete output files or change saved hashes. For a genuinely new, empty
output directory use the same runner without `--resume`.

## Execution provenance and final evaluation

The memory adapter is kept outside `src` deliberately: changing the original
study manifest would change label and training checkpoint identities. Its exact
three source files are instead copied into `inputs/memory_execution/`, with an
immutable `memory_execution.json` binding their hashes and the original study
manifest. The old scientific manifest is never rewritten. Adapter or snapshot
changes are rejected on later low-memory invocations.

Use this runner for subsequent commands as well:

```bash
python scripts/run_v6_7_bamg_energy_cnn_routing_low_memory.py report \
  --output artifacts/bamg_energy_cnn_routing_560_q1
python scripts/run_v6_7_bamg_energy_cnn_routing_low_memory.py freeze \
  --output artifacts/bamg_energy_cnn_routing_560_q1
python scripts/run_v6_7_bamg_energy_cnn_routing_low_memory.py evaluate-final \
  --output artifacts/bamg_energy_cnn_routing_560_q1
```

The freeze binds the execution record and all three source snapshots before its
atomic publication and before any final claim. Normal frozen-file verification
then checks those snapshots too. The adapter cannot be introduced into an
already frozen or final-claimed study. Final/OOD resumption uses `--resume` only
after that final evaluation has actually begun.

For resource diagnosis without changing the experiment, run `free -h` inside
WSL and inspect the last `TRAIN teacher ...` line in the log. Those observations
identify available memory/swap and the case at which generation stopped. They
are needed before choosing a machine-specific WSL memory limit.
