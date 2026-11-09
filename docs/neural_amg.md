# Neural AMG: classical coarsening, attention interpolation

This is an isolated implementation based on commit `8892f31520062dc832d33ae47c58216185ce5f11`. The existing CNN study, checkpoints, teacher labels and source files retain their behavior. New graph checkpoints are not compatible with CNN checkpoints.

## Numerical and learning path

Each level uses classical AMG to select coarse unknowns `C` and a fixed interpolation support `S`. The support reaches at most two hops in that level's matrix graph. Hop distance does not skip MG levels. Three one-hop message-passing layers encode the full sparse matrix graph; sparse fine-query/coarse-key-value attention operates only over `S`. A separate signed edge head predicts interpolation values. Injection and reference row sums are enforced, with a signed row-L1 bound. Attention coefficients are not interpolation weights. Cross-attention adds the coarse candidates' receptive fields, so the entire model is not restricted to three hops around a fine node.

Training first constructs a numerical BAMG+EM teacher on the classical support. The objective fits frozen relaxed test vectors plus a ridge term and adds normalized `trace(P.T A P)` energy. Normalizers and row targets come from the classical algebraic reference interpolation, not geometric bilinear fractions. Labels include the numerical optimizer's convergence report; a finite-iteration teacher is not asserted to be an exact minimizer. Supervision combines normalized weight MSE and probe-action MSE on all nonterminal classical hierarchy levels. Fine-tuning builds the actual learned hierarchy and differentiates multicycle residual loss through every `P` and Galerkin product. Classical C/S selection is detached per forward pass. RL is not used.

Both training and deployment use the same bounded weighted Jacobi smoother and a sparse direct terminal solve. `R=P.T`, `Ac=P.T A P`. A learned hierarchy is constructed once per input operator; no NN runs inside solve cycles. Inputs must be real SPD matrices. Symmetry and a positive diagonal are checked, but these are not a general SPD certificate. The implementation accepts arbitrary node counts and sparse graph topology; the bundled synthetic research data remain structured Q1 FEM, not a demonstration on unstructured meshes.

Coarsening normally stops at `max_coarse=9` unknowns. `max_levels` is a safety cap, not a requested hierarchy depth: exceeding it before reaching the coarse size is reported as a failure. The research configuration allows 20 levels to accommodate larger and difficult OOD operators; the smoke configuration retains its smaller cap.

## Input features

`feature_mode: original` is the default. The original information/normalization principle is preserved: log relative diagonal, normalized diagonal, and signed row-normalized coefficients `Aij/di`. On an arbitrary AMG graph there are no nine geometric directions: coefficients live on edges rather than fixed directional CNN slots, and all stored coarse fill is represented. This is not a claim of identical 10-channel CNN tensors or transferable CNN parameters.

The opt-in modes `row9` and `symmetric9` correspond to the separate feature experiments at the base commit: `row9` removes the redundant normalized center, while `symmetric9` additionally uses signed `Aij/sqrt(di*dj)` edge normalization. On graphs these names denote the adapted feature semantics, not a nine-wide directional tensor. Neither mode is a validated improvement or enabled automatically. Change only `model.feature_mode` in a copied config and use a new output directory. Mode/schema/source hashes bind checkpoints and resume. The original exact structured-grid feature9 experiment runners also remain available unchanged.

## Install and run

From the repository root on Linux/macOS (or Lenovo Ubuntu/WSL):

```bash
git fetch origin
git switch neural-amg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' -r requirements-neural-amg.txt

python scripts/run_neural_amg.py train \
  --config configs/neural_amg_smoke.json --run-dir artifacts/neural_amg_smoke
python scripts/run_neural_amg.py freeze \
  --config configs/neural_amg_smoke.json --run-dir artifacts/neural_amg_smoke
python scripts/run_neural_amg.py evaluate \
  --config configs/neural_amg_smoke.json --run-dir artifacts/neural_amg_smoke
```

The smoke run tests plumbing only; it does not provide research performance evidence. Research configuration uses 560 train operators, 280 validation operators, 280 final operators and five separately seeded OOD groups. It uses one model seed and one teacher lambda (`10`) initially; sweep seeds/lambdas in separate runs and select on validation before accessing final/OOD. The previous CNN's optimal lambda or geometric grid-routing rule is not assumed to transfer.

```bash
python scripts/run_neural_amg.py train \
  --config configs/neural_amg_research.json --run-dir artifacts/neural_amg_560_seed6101
# Resume an interrupted run with the exact same config, source and data:
python scripts/run_neural_amg.py train \
  --config configs/neural_amg_research.json --run-dir artifacts/neural_amg_560_seed6101 --resume
python scripts/run_neural_amg.py freeze \
  --config configs/neural_amg_research.json --run-dir artifacts/neural_amg_560_seed6101
python scripts/run_neural_amg.py evaluate \
  --config configs/neural_amg_research.json --run-dir artifacts/neural_amg_560_seed6101
python scripts/run_neural_amg.py evaluate \
  --config configs/neural_amg_research.json --run-dir artifacts/neural_amg_560_seed6101 --resume
```

`prepare` can materialize development data separately; subsequent `train` then requires `--resume`. `train --max-updates N` stops cleanly at an update boundary for testing resume. Each update atomically saves model, optimizer, phase/epoch/offset, best checkpoint, and Python/NumPy/Torch RNG state. Teacher caches and data files have checksums. No final/OOD operators are assembled by the generated-data path before freeze. Exact normalized-operator duplicates are rejected across splits; metadata seeds alone are not treated as proof of disjointness.

Validation selects failures first, then mean per-case median **setup plus solve** seconds, including graph preparation, attention, Galerkin products and factorization. Warm solves reuse the same hierarchy and are reported separately. Offline label/training time and model/data disk loading are excluded. The benchmark is CPU FP64 with explicit thread limits; compare on the intended target machine. Full research runs and superiority over tuned AMG are not implied by smoke tests. Fine-tuning optimizes residuals; wall-clock time is measured for validation rather than differentiated.

`freeze` binds the selected model, configuration, source hashes, software versions, development manifests and future holdout specification. A frozen run cannot be trained further. `evaluate` compares classical, numerical BAMG+EM teacher, and learned interpolation with the same solver controls. Coarsening uses the same algorithm on each method's actual Galerkin matrices; deeper-level C/S may therefore differ. Cold cost of the numerical teacher includes its online optimization. Results include failures, cycles, residual histories, repeated cold/warm timings and are resumed by completed case/method.

## Arbitrary sparse operators

Use a JSON manifest with explicit splits; each `A` is a SciPy `save_npz` sparse matrix, and optional `b` is a NumPy `save` one-dimensional array. Paths are relative to the manifest. If `b` is omitted, a deterministic random RHS is generated. All files are hashed before training; final/OOD matrices are only loaded for evaluation after freeze. This makes existing unstructured P1/FV matrices usable without a structured-grid reshape.

```json
{
  "splits": {
    "train": [{"A": "train_A.npz", "b": "train_b.npy"}],
    "validation": [{"A": "val_A.npz", "b": "val_b.npy"}],
    "final": [{"A": "final_A.npz", "b": "final_b.npy"}],
    "ood_mesh": [{"A": "ood_A.npz", "b": "ood_b.npy"}]
  }
}
```

Replace the config's `data` object with `{"kind":"npz_manifest","manifest":"path/to/operators.json","seed":2026111001}`. Its manifest path is relative to the config file. Use genuinely disjoint operators, retaining positive/negative off-diagonal signs. Boundary-eliminated operators are allowed; no universal row-sum-one condition is imposed.
