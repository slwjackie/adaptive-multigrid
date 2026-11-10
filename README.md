# Hybrid-teacher CNN multigrid

Branch: **`residual-u-net-teacher`** · [한국어 안내](README_KR.md) · [Documentation index](docs/README.md)

This branch studies learned prolongation for synthetic **2D SPD diffusion systems on structured Q1 FEM grids**. The current workflow trains Weight CNNs from BAMG + energy teacher labels at **λ = 10, 30, 50**, compares scratch and warm initialization, and selects one CNN family from the finest input grid size using development validation. The newest opt-in experiment compares **`row9` and `symmetric9` matrix features** with the original ten-channel representation.

The branch name comes from an earlier U-net experiment. The workflow described here uses a small Weight CNN with a fixed classical line smoother. Earlier EM-line, U-net, learned-smoother and P-only studies remain available for reproduction; their defaults and performance claims do not describe this workflow.

## Choose an entry point

| Purpose | Script under `scripts/` | Guide |
| --- | --- | --- |
| New 9-channel feature study: `row9` / `symmetric9` | `run_v6_7_bamg_energy_cnn_feature9.py` | [Features, warm transfer and cohort sharing](docs/bamg_feature9.md) |
| Original 10-channel, three-λ routing with bounded label loading | `run_v6_7_bamg_energy_cnn_routing_low_memory.py` | [Memory / resume](docs/bamg_cnn_low_memory_recovery.md), [routing protocol](docs/bamg_energy_cnn_grid_routing.md) |
| Original 10-channel routing launcher | `run_v6_7_bamg_energy_cnn_routing_study.py` | [Routing protocol](docs/bamg_energy_cnn_grid_routing.md) |
| Single numerical-winner λ, scratch versus warm | `run_v6_7_bamg_energy_cnn_study.py` | [Single-λ comparison](docs/bamg_energy_cnn_initialization_comparison.md) |
| Numerical teacher comparison; no CNN training | `run_v6_7_bamg_energy_study.py` | [Teacher objective and comparison](docs/bamg_energy_teacher_comparison.md) |

**The committed routing/feature9 research presets run 18 trainings per feature mode and dataset size:** 3 λ values × 2 initialization arms × seeds 6101/6102/6103. Each training includes supervised learning and residual fine-tuning. There is no committed `--scratch-only` option in these launchers; a locally modified nine-run script is a separate workflow. The two feature modes are separate invocations and outputs.

The numerical teacher configuration has a different candidate list: **`[0, 0.01, 0.1, 1, 5, 10, 30, 50]`**. Its selected λ is provenance for the routing study, not a restriction on the three CNN candidates. In the older single-λ workflow, that numerical winner does determine the labels.

## Numerical and model contract

| Component | Current routing / feature9 study |
| --- | --- |
| Discretization | Structured Q1 quadrilaterals; homogeneous Dirichlet conditions |
| Coarsening and support | Full coarsening; fixed geometric bilinear support, at most four coarse connections per row |
| Learned quantity | Signed prolongation weights; no learned smoother or learned coarsening in this workflow |
| CNN | Five 3×3 convolution layers, width 8, ReLU, followed by a 16-slot output head |
| Constraints | Injection, prescribed interior/boundary row sums, signed row L1 cap 8 |
| Restriction / coarse operators | `R = Pᵀ`; each hierarchy uses its actual `A_c = Pᵀ A P` |
| Smoother / terminal solve | Classical alternating line relaxation, pre/post 2 sweeps each; LU at 3×3 interior grid |
| Fine-tuning | Uniform residual loss; 6 then 12 cycles; random, smoothed and hard error probes |
| Evaluation | Zero initial iterate; `||r||₂ ≤ max(1e-12, 1e-8 ||r₀||₂)`; no fallback |
| Selection timing | Warm solve time; excludes loading, CNN inference, hierarchy construction and factorization |

The verified numerical comparison supplies the teacher optimizer and MG settings. The table describes the checked-in research protocol; saved `configuration.json` records the resolved settings.

| Matrix-feature mode | Matrix channels | With geometry (3) and early support mask (16) |
| --- | --- | ---: |
| Original | Log diagonal + row-normalized nine-point stencil, including center | 29 |
| `row9` | Log diagonal + eight signed neighbors `A_ij / d_i` | 28 |
| `symmetric9` | Log diagonal + eight signed neighbors `A_ij / sqrt(d_i d_j)` | 28 |

Here `d_i = max(A_ii, 1e-14)` and the log channel is `log(d_i / median(d) + 1e-12)`; feature9 rejects nonpositive diagonals. Feature normalization does not modify the linear system. Geometry and the 16-slot mask are retained; nine matrix channels do not mean nine total CNN inputs. Coarse fill-in beyond the eight neighbor offsets is not fully represented.

Scratch starts with random weights. Warm starts from the verified source CNN, resets the optimizer and trains every layer. Feature9 transfers a legacy checkpoint by dropping the first convolution's center-input column; this is **not a function-preserving conversion or an exact resume**. The `old_cnn` control continues to use its original ten-channel features.

## Development data and selection

| Setting | 560-case study | 1,120-case study |
| --- | ---: | ---: |
| TRAIN grids | 15, 31, 63, 127 | Same |
| TRAIN per grid / family (7 families) | 20 | 40 |
| Full validation | 280 | Same 280 when explicitly linked |
| Checkpoint-selection subset | 56 | Same 56 when explicitly linked |
| Supervised epochs / updates per training | 15 / 8,400 | 15 / 16,800 |
| Fine-tuning epochs / updates per training | 8 / 4,480 | 8 / 8,960 |
| Training runs per feature mode | 18 | 18 |

Epoch shuffle visits every TRAIN operator once per completed epoch. Model seeds change scratch initialization, shuffle order and residual probes, not the shared operators or each λ's teacher labels. Corresponding λ/arm runs share the sampling policy. A validation-selected deployment checkpoint may precede the last trained epoch.

Development sampling balances categorical angles/checker counts, stratifies continuous parameters within TRAIN support, and includes difficult small-epsilon/high-contrast combinations. Families are near-isotropic, aligned/rotated anisotropy, local patch, checkerboard, channel and variable coefficient.

Use **`--development-study`** to verify a shared cohort: the 1,120-case pool contains the 560-case pool, with identical validation. Equal seeds or different output folders alone do not guarantee this because historical duplicate exclusions also affect generation. Feature comparisons must share an explicit prepared development reference; see the [feature guide](docs/bamg_feature9.md). Do not use a reference whose final/OOD claim has been opened. Equal epochs give the larger dataset more updates and compute.

After full validation, `run` selects a **separate grid→λ mapping for scratch and warm**. All three seeds share each arm's mapping. Ranking uses failure rate, then mean penalized warm solve time, then lower λ as a tie-break. All seeds remain in the report. Ineligible candidates retain their status even if a least-failing route must be recorded.

Routing chooses from the **finest input grid once**, then uses that CNN throughout the learned hierarchy. It does not change models at each coarse level. Unseen grids use the largest validation grid no greater than the input, clamped at the ends: **n ≥ 127, including n=255, uses the n=127 choice**. A globally fixed λ per arm is also selected as a control. The held-out evaluation measures whether routing generalizes beyond the data used to choose it.

## Start a new study on Lenovo / Ubuntu WSL

Use Linux/WSL for these commands (the study code uses `fcntl`). Python **≥3.10** is declared; CI uses Python 3.12 with CPU PyTorch. Run the commands in Ubuntu/WSL, not Windows CMD. The example creates a fresh checkout on the Windows Desktop; adjust the Windows username/path if needed. Keep any running checkout unchanged.

```bash
cd /mnt/c/Users/user/Desktop
git clone --branch residual-u-net-teacher --single-branch \
  https://github.com/slwjackie/adaptive-multigrid.git adaptive-multigrid-feature9-current
cd adaptive-multigrid-feature9-current
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.10' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1
set -o pipefail
mkdir -p artifacts/logs
```

For an existing run, restore its original environment instead of reinstalling dependencies. These CNN launchers require **two existing inputs**, including for smoke: a compatible frozen source BAMG study and a completed numerical teacher comparison. Source architecture/discretization and recorded hashes are checked. Set their actual paths; the following paths match one example folder layout:

```bash
export BAMG_SOURCE_STUDY="/mnt/c/Users/user/Desktop/adaptive-multigrid-bamg/artifacts/initial_weights/mac_bamg"
export BAMG_TEACHER_COMPARISON="/mnt/c/Users/user/Desktop/adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_lambda_sweep_q1"
test -f "$BAMG_SOURCE_STUDY/data/research_freeze.json" && \
test -f "$BAMG_TEACHER_COMPARISON/teacher_comparison.json"
```

Stop and correct these paths if the check fails. If these inputs do not exist, follow the [BAMG source-study guide](docs/bamg_compact_weight_study.md) and [numerical comparison guide](docs/bamg_energy_teacher_comparison.md) first. A clone alone does not supply trained checkpoints or completed experiments.

### Smoke, then a fresh 560-case symmetric9 study

Smoke uses two n=7 TRAIN/validation cases, three λ values, two arms and one seed (six tiny training runs). It checks execution, not research performance. Use separate, previously unused output paths:

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_feature9.py run \
  --config configs/v6_7_bamg_energy_cnn_feature9_symmetric9_smoke_q1.json \
  --source-study "$BAMG_SOURCE_STUDY" \
  --teacher-comparison "$BAMG_TEACHER_COMPARISON" \
  --output artifacts/feature9_symmetric9_smoke \
  2>&1 | tee artifacts/logs/feature9_symmetric9_smoke.log
```

After smoke completes successfully, start research. This example creates a fresh cohort. If comparing against an existing routing/feature experiment, add its verified path with `--development-study` **on the first run and every resume**.

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_feature9.py run \
  --config configs/v6_7_bamg_energy_cnn_feature9_symmetric9_q1.json \
  --source-study "$BAMG_SOURCE_STUDY" \
  --teacher-comparison "$BAMG_TEACHER_COMPARISON" \
  --output artifacts/feature9_symmetric9_560_q1 \
  2>&1 | tee artifacts/logs/feature9_symmetric9_560_q1.log
```

To use `row9`, choose its corresponding config and a different output. Both feature9 modes already load teacher labels one case at a time. This bounds label residency, not all training memory; disk I/O can be slower on `/mnt/c`. Record `runtime_metrics.json` and measure time/RAM on the target machine.

### Resume and inspect development results

Restore the same checkout, virtual environment, thread variables and input paths. Append `--resume` to the same `run` command and use `tee -a`; retain `--development-study` if originally supplied. Do not use `--resume` for a new empty output. Saved optimizer/RNG/epoch progress is restored; work after the last durable checkpoint can repeat.

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_feature9.py run \
  --config configs/v6_7_bamg_energy_cnn_feature9_symmetric9_q1.json \
  --source-study "$BAMG_SOURCE_STUDY" \
  --teacher-comparison "$BAMG_TEACHER_COMPARISON" \
  --output artifacts/feature9_symmetric9_560_q1 \
  --resume \
  2>&1 | tee -a artifacts/logs/feature9_symmetric9_560_q1.log

python scripts/run_v6_7_bamg_energy_cnn_feature9.py report \
  --output artifacts/feature9_symmetric9_560_q1
```

The report is available after development benchmarking finishes. For 1,120 cases, use `configs/v6_7_bamg_energy_cnn_feature9_symmetric9_1120_q1.json`, a new output, and `--development-study artifacts/feature9_symmetric9_560_q1`. Complete the smaller development comparison first and keep final/OOD unopened. Choose dataset size, feature mode and budgets from development evidence before freezing.

## Freeze and final/OOD are separate commands

| Command | Work performed |
| --- | --- |
| `run` / `run --resume` | Labels, training, full validation and grid selection; no final/OOD solves |
| `report` | Read saved development results |
| `freeze` | Audit completion, recompute/verify the mapping and pin models, data, source/environment and choices |
| `evaluate-final` | Evaluate the frozen study on final and all OOD splits |
| `report-final` | Read saved held-out results, optionally with `--group` |

Use the **same runner** for every stage of a study. For the chosen 560-case feature9 study:

```bash
python scripts/run_v6_7_bamg_energy_cnn_feature9.py freeze \
  --output artifacts/feature9_symmetric9_560_q1 && \
python -u scripts/run_v6_7_bamg_energy_cnn_feature9.py evaluate-final \
  --output artifacts/feature9_symmetric9_560_q1 \
  2>&1 | tee artifacts/logs/feature9_symmetric9_560_q1_final.log
```

For an interrupted final evaluation, use `evaluate-final --resume` with the same output and `tee -a`. Do not add `--resume` to its first invocation. `--max-cases 10` can pause after at most ten newly completed operators. Freezing closes training and selection for that output.

```bash
python scripts/run_v6_7_bamg_energy_cnn_feature9.py report-final \
  --output artifacts/feature9_symmetric9_560_q1 --group final
python scripts/run_v6_7_bamg_energy_cnn_feature9.py report-final \
  --output artifacts/feature9_symmetric9_560_q1 --group ood_grid
```

| Research holdout split | Independent operators |
| --- | ---: |
| Within-support final | 560 |
| Grid OOD (n=255) | 140 |
| Rotation OOD | 150 |
| Anisotropy OOD | 180 |
| Contrast OOD | 120 |
| Topology OOD | 120 |
| **Total** | **1,270** |

Each operator has manufactured, random and localized RHS and three timing repetitions. RHS, seeds and repetitions do not increase the count of independent operators. Controls include `fixed_C`, `em5_C`, numerical `bootstrap_C`, the old CNN, fixed-λ supervised/fine-tuned CNNs and routed fine-tuned CNNs. Failed solves remain visible with penalties. Final/OOD results must not be used to revise the frozen mapping or choose a favorable seed.

## Outputs and troubleshooting

| Path relative to output | Meaning |
| --- | --- |
| `configuration.json`, `study_manifest.json`, `data/` | Resolved settings, provenance, shared TRAIN/validation |
| `labels/lambda_*/teacher/train/`, `teacher_bank_manifest.json` | Per-case labels and common-bank audit |
| `runs/lambda_10_scratch_seed6101/` (and other candidates) | Latest resumable state, histories and selected exports |
| `runtime_metrics.json` | Phase elapsed time and process peak RSS |
| `validation/`, `validation_routed/`, `comparison.json` | Fixed/routed development measurements and report |
| `routing_selection.json` | Grid mappings and globally fixed λ controls |
| `holdout_plan.json`, `research_freeze.json` | Predeclared holdout and frozen study |
| `final/report.json`, `final/claim.json` | Held-out results and evaluation/resume state |

- Preserve the source package, numerical comparison and existing run metadata. Do not repair a hash mismatch by rewriting saved hashes. See [source-freeze compatibility](docs/bamg_source_freeze_compatibility.md).
- A feature change, altered budget, changed dataset or different runner is not automatically an exact resume. Use a new output and the appropriate cohort reference. For an unfrozen original ten-channel run with label-memory exhaustion, follow the specific [low-memory recovery procedure](docs/bamg_cnn_low_memory_recovery.md).
- Keep source-bound running checkouts stable. A documentation update is not a reason to reinstall dependencies or change the scientific code of a running experiment.
- Loss reduction, fewer cycles and faster warm solves are different claims. Warm solve improvements do not establish setup-plus-solve improvements. Fixed budgets and successful smoke tests do not prove convergence, research speedup or generalization to unstructured meshes, nonlinear PDEs or real CFD.

## Verification and earlier work

```bash
python -m pytest tests/test_bamg_feature9.py tests/test_bamg_feature9_multilevel.py tests/test_bamg_feature9_study.py
python -m pytest -ra
sha256sum -c THREE_PILLARS_SOURCE.sha256
sha256sum -c P_HEADROOM_SOURCE.sha256
sha256sum -c WORLD_MODEL_SOURCE.sha256
sha256sum -c HS_WORLD_SOURCE.sha256
```

CI runs the full suite and these four source checks. The optional native stencil build is exercised by CI; the current CPU/CSR protocol does not require it. Test results validate implementation behavior, not the unexecuted research workload. Historical package manifests and historical validation records describe their own snapshots.

For earlier workflows, see [BAMG source/compact studies](docs/bamg_compact_weight_study.md), [EM-line](docs/em_line_study.md), [lightweight smoothing](docs/light_smoother_study.md), [P-only](docs/learned_p_study.md), [U-net transfer](docs/RESIDUAL_UNET_TRANSFER_KR.md), and the [full document index](docs/README.md). The separate unstructured neural-AMG workflow is not an entry point in this branch snapshot.
