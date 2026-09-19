# Graph-aware learned interpolation

`research_transfer.py` adds three setup-only generators using the existing
selected classical interpolation as their reference: `small_gnn` (1–2 message
layers, default width 24), `edge_mlp` (no message layers, sparse candidate queries)
and `gnn_teacher` (3–6 layers, default four/width 48). The Teacher is explicitly
`training_only`; the runtime guard rejects it in deployment, including when a
saved checkpoint is restored. It can run only through the explicit offline
Teacher path.

Inputs are sparse A values/connectivity, normalized diagonal/row sums, strength,
signed geometric edge offsets, C/F roles and the selected classical candidate
weights. Sparse A values remain differentiable when an actual coarse operator
comes from `SparseTensor`. No RHS, residual, family name, exact solution or
reference solve is an input. Storage is O(nnz(A) + N K), where K is 16 ordinarily
or 36 in the optional expanded-coordinate support ablation. Output slots are
keyed by candidate column identity, so reordering candidate slots reorders the
output consistently. This is a graph model on a structured geometric candidate
support, not a claim of arbitrary unstructured AMG support learning.

The zero decoder exactly preserves the selected classical P when that baseline
already fits the configured row support cap. An explicitly tighter cap can
prune a baseline row even before training; that is an equal-budget ablation,
not unconditional zero-initialization equivalence. The shared
`project_transfer_weights` calls the existing row-sum/coarse-injection projection.
Expanded support additionally retains the largest 16 candidate weights per row
by default. An explicitly tighter cap, such as eight entries, also prunes the
standard support and the controlled CNN ablation using the same projection. It preserves the classical row sums and injection, and uses the same
pruned numerical values during training and inference. A straight-through
backward approximation is documented for the discrete support selection.
Restriction remains `P.T`; the surrounding hierarchy builds `P.T @ A @ P`.
For SPD A, retained injection makes P full column rank and the Galerkin coarse
operator SPD in exact arithmetic. Existing finite and magnitude checks remain.

`enforce_transfer_complexity` measures actual numerical nonzeros after summing
duplicates and eliminating zeros. It enforces the per-row support cap, P ratio,
coarse-A ratio and two-level operator complexity. Runtime must also check the
aggregate hierarchy budget and charge rejected setup work before falling back.
The checkpoint records the support and complexity budgets. The differentiable
cost helper is a smooth support and sparse-product work proxy; it is neither
an exact nnz count nor a wall-clock gradient.

`forward_graph(..., return_details=True)` and `transfer_distillation_loss` provide
projected-weight KD and optional width-independent local edge-feature KD.
Teacher targets are detached. These losses supplement the full multilevel
V-cycle objective; Teacher quality, KD utility and final architecture selection
must be decided using development data only. Feature KD compares local summaries
(mean, RMS, mean magnitude, maximum) of candidate embeddings, not an all-pairs
Gram matrix. Its coefficient can be zero for a weight-only KD ablation.

Tests cover sparse-only generation, far graph edges, scale invariance, A-value
and parameter gradients, candidate permutation, row sums including boundary
rows, coarse injection, Galerkin SPD, bounded expanded support, actual nnz caps,
Teacher gradient isolation, architecture metadata and independent direct models.
