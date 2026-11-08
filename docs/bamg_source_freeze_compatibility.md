# Source freeze digest compatibility

The paired and grid-routing CNN launchers now read genuine BAMG research
freezes using the same compact JSON digest as `research_data.freeze_research`.
The previous importer used the numerical comparison's spaced JSON digest and
could reject an unchanged source with `Source freeze configuration hash changed`.
The early importer encoding is also accepted, but either encoding must match the
entire stored configuration. Checkpoint, dataset, baseline, configuration and
old-final-claim checks remain mandatory.

This fix is additive: `bamg_energy_cnn_source_compat.py` supplies the corrected
source reader during CLI execution. Existing `src` files and the numerical
comparison launcher are unchanged. A completed numerical teacher comparison's
recorded source hashes therefore continue to verify. The original source freeze,
its checkpoints and the numerical comparison manifest must not be edited or
rehashed. New CNN study manifests record the compatibility module and updated
launchers. Direct Python callers can use its `source_reader()` context manager.

For a routing invocation that failed at the source configuration hash before
creating its study manifest, update the detached routing worktree and rerun the
same command. The completed numerical teacher sweep does not need to be rerun.
Keep the original BAMG worktree unchanged.

```bash
cd /mnt/c/Users/user/Desktop/adaptive-multigrid-cnn-routing
git fetch origin residual-u-net-teacher && git switch --detach FETCH_HEAD

: "${BAMG_SOURCE_STUDY:?Set the path to the copied source study}"
: "${BAMG_TEACHER_COMPARISON:?Set the path to the completed numerical comparison}"
mkdir -p artifacts/logs
set -o pipefail
python -u scripts/run_v6_7_bamg_energy_cnn_routing_study.py run \
  --config configs/v6_7_bamg_energy_cnn_routing_q1.json \
  --source-study "$BAMG_SOURCE_STUDY" \
  --teacher-comparison "$BAMG_TEACHER_COMPARISON" \
  --output artifacts/bamg_energy_cnn_routing_560_q1 \
  2>&1 | tee artifacts/logs/bamg_energy_cnn_routing_560_q1.log
```

Do not add `--resume` solely because of this startup error: it occurs before
training or output creation. An already-started CNN study still obeys its normal
source-bound resume checks; a code update does not authorize modifying its
recorded manifest or force-resuming it. Other hash mismatches still need their
own diagnosis.
