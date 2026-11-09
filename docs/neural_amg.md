# Neural AMG: unstructured P1 study

This branch provides one discretization: **continuous P1 FEM on unstructured
triangular meshes**. Its protocol is `neural-amg-p1-study-v2`. Q1, fixed-diagonal
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
Coarse injection, reference interpolation row sums and a signed L1 cap are enforced.
The encoder has a three-hop receptive field; encoded coarse candidates can extend
the complete prediction's receptive field beyond three hops around a fine node.

The numerical BAMG+EM teacher fits relaxed/bootstrap test vectors plus a ridge
term and normalized trace(PᵀAP) energy on fixed C/S. Normalizers and constraints
come from the algebraic reference P. Finite-iteration labels carry optimizer
reports; they are not asserted to be exact minimizers. Supervision combines
normalized weight and probe-action MSE on classical hierarchy levels. Fine-tuning
builds the actual learned hierarchy and differentiates multicycle residual loss
through all Galerkin levels, freezing the discrete C/S decisions per forward pass.

Training and deployment share bounded weighted Jacobi pre/post smoothing and a
sparse direct terminal solve. R=Pᵀ and Ac=PᵀAP; no coarse fill is stencil-truncated.
The NN runs during setup, not inside V-cycles. `max_coarse=9` is the normal stopping
threshold; `max_levels` is a safety cap (20 in research), not the desired depth.
The solver assumes SPD input. P1 assembly with SPD tensors and Dirichlet elimination
provides SPD matrices; low-level solver symmetry/diagonal checks alone are not an
SPD certificate for arbitrary external matrices.

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

## Install and smoke

From the repository root on Linux/macOS or Lenovo Ubuntu/WSL:

```bash
git fetch origin
git switch neural-amg
git pull --ff-only origin neural-amg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python scripts/run_neural_amg.py train \
  --config configs/neural_amg_smoke.json --run-dir artifacts/neural_amg_p1_smoke
python scripts/run_neural_amg.py freeze \
  --config configs/neural_amg_smoke.json --run-dir artifacts/neural_amg_p1_smoke
python scripts/run_neural_amg.py evaluate \
  --config configs/neural_amg_smoke.json --run-dir artifacts/neural_amg_p1_smoke
```

The current backend is CPU FP64; GPU execution is not implemented. Smoke verifies
execution, not research-scale speed improvement. Use a new output directory;
previous Q1 study outputs, labels and checkpoints cannot be resumed as P1 runs.

## Research training and evaluation

The research config specifies 560 TRAIN operators (four densities × seven families
× twenty cases), 280 validation, 280 final and six independently seeded OOD groups.
The first run uses one model seed (6101) and teacher lambda=10. Optimal old CNN
lambdas and routing policies are not assumed transferable. Final/OOD cases are
not used to choose architecture, lambda, features or checkpoints.

```bash
python scripts/run_neural_amg.py train \
  --config configs/neural_amg_research.json --run-dir artifacts/neural_amg_p1_560_seed6101
# Resume training with exactly the same configuration, source and runtime:
python scripts/run_neural_amg.py train \
  --config configs/neural_amg_research.json --run-dir artifacts/neural_amg_p1_560_seed6101 --resume
python scripts/run_neural_amg.py freeze \
  --config configs/neural_amg_research.json --run-dir artifacts/neural_amg_p1_560_seed6101
python scripts/run_neural_amg.py evaluate \
  --config configs/neural_amg_research.json --run-dir artifacts/neural_amg_p1_560_seed6101
python scripts/run_neural_amg.py evaluate \
  --config configs/neural_amg_research.json --run-dir artifacts/neural_amg_p1_560_seed6101 --resume
```

`prepare` optionally assembles development data separately; `train` then needs
`--resume`. `train --max-updates N` stops after N updates for resume checks. Each
update saves model, optimizer, phase/epoch/offset, best checkpoint and RNG states.
Teacher caches and data have checksums. Model/configuration/source/runtime changes
reject resume. Freeze binds the selected model, development manifests, validation
history and the future holdout specification; frozen runs cannot be trained.

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
Fine-tuning optimizes residual reduction, not a differentiable wall clock.

Evaluation compares classical, numerical BAMG+EM teacher and learned interpolation
under the same controls. Each method coarsens its own actual generated coarse
matrices, so deeper C/S may differ. Teacher cold timing includes numerical teacher
optimization. Failures, cycles, residual histories and repeated cold/warm timings
are recorded, with evaluation resumable by case/method.

`python -m pytest -ra` runs the complete current-branch tests, including P1 FEM
accuracy, algebraic constraints, finite-difference gradients, actual P1 multilevel
runtime parity, source/mesh/checkpoint identities and resume. These checks do not
replace the unexecuted 560-operator research study.
