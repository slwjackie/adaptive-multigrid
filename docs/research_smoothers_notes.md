# Strong-aware smoother architectures

`adaptive_mg.v67.research_smoothers.make_research_smoother` builds operator-only
models. `architecture_spec()` and `research_smoother_from_spec()` give a versioned,
JSON-compatible architecture contract; weights remain ordinary PyTorch state
dictionaries. Existing checkpoints remain ordinary `NeuralSmootherNet` models.

All new models consume the existing ten channels: log normalized diagonal and
the diagonal-normalized nine A-stencil entries. They emit one normalized
9-point direction and one bounded positive scalar gain. `direction_and_gain`
returns `(B,1,9,H,W), (B,1)` for the existing numerical runtime;
`direction_and_gain_single` omits the singleton basis axis. Gain defaults to
`[1e-4,2]`; normalization is `c/sqrt(sum(c*c)+epsilon)`. Initial direction is
the central unit stencil and initial gain is 0.72. The finite, bounded model
output does not itself prove MG stability: the existing acceptance, rollback,
and CLASSICAL LOCK mechanisms are still required.

| Kind | Default body | Role |
| --- | --- | --- |
| `ordinary_control` | width 16, existing residual5 | Current CNN architecture with the shared new gain/basis contract |
| `student_residual` | width 16, existing residual5 | Independent same-size direct/KD control; same architecture as ordinary control |
| `student_cnn` | width 16, compact3 | Smaller ordinary direct/KD student |
| `d4` | width 16, Reynolds average of residual5 | Exact 90-degree rotations and reflections |
| `o2` | 12 scalars, 6 order-1, 6 order-2 channels; three blocks | Compact steerable candidate |
| `teacher_cnn` | width 48, six residual blocks alternating dilation 1/2 | Training-only larger CNN |
| `teacher_o2` | 32 scalars, 16 order-1, 16 order-2; six blocks | Training-only larger steerable model |

All widths and depths are explicit in the spec. `cost_metadata()` records actual
parameter counts. For a close parameter-budget comparison, ordinary/D4 width
16 has 10,906 parameters and O2 width 18 is approximately 10,100. The default O2
is intentionally smaller. Parameter counts do not predict wall time, particularly
for many small tensor operations or eight D4 views. Ordinary control is the
current *architecture*, with one basis and the common bounded gain for fair new
training; historical checkpoint reproduction uses the untouched old checkpoint.

## What is exact and what is approximate

D4 transforms both grid sites and directional input/output channels. A Reynolds
average maps each transformed prediction back to the original frame before
averaging logits and normalizing. It is exact up to floating-point roundoff on
square grids and rectangular semicoarsened levels. All eight views are batched
into one encoder call for square grids and two calls for rectangles. This still
requires **eight times the encoder arithmetic** and increased activation memory;
it is not an inexpensive single-view CNN. Setup timings must include that cost.

The O2 lift is a full-rank change of the same ten input coordinates into four
scalars, two radial order-1 vector moments, and one order-2 moment. It uses the
signed stencil moments `(x,y)`, `r²(x,y)`, and `(x²-y²,2xy)`. Both radial vector
moments are retained; omitting one would discard two input channels. Scalars
transform trivially; vector pairs rotate by angle theta; order-2 pairs rotate
by twice theta. Reflection conjugates each pair, so handedness is not learned.

Spatial blocks combine radial Gaussian kernels, analytic first/second angular
harmonics, equivariant channel mixing, scalar gates, vector/tensor products,
and invariant norm denominators. They use no independent nonlinear activation
on tensor components. The stencil head evaluates scalar radial polynomials,
vector dot products, and quadratic tensor harmonics on the nine offsets.

The continuous kernel constraint and representation algebra follow the
steerable-CNN formulation in [Weiler and Cesa (2019)](https://arxiv.org/abs/1911.08251).
The [escnn kernel documentation](https://quva-lab.github.io/escnn/api/escnn.kernels.html)
describes the same distinction between continuous kernel bases and sampling them
at discrete points. This implementation is a deliberately small pure-Torch
subset, not an escnn implementation or a claim of equivalence to its full basis.

Kernel/fiber/head algebra is continuously O(2)-equivariant when vectors and
query offsets change frame together. **Arbitrary spatial rotation on a fixed
square lattice and a fixed nine-point stencil is approximate**, because rotated
sites/offsets do not stay on that lattice. Sampling, interpolation, the domain
boundary, and nine-point normalization introduce errors. Numerical tests cover
all D4 actions, arbitrary-angle/reflection fiber identities, and exact kernel
covariance. They deliberately show that rotating tensor fields while holding
the nine offsets fixed does not leave the stencil values identical. Rotation
OOD PDE contraction and end-to-end solve time must decide whether this prior is
useful; analytic equivariance alone does not establish numerical benefit.

## Runtime and teacher boundary

Each distinct selected level operator A_l generates its coefficients once and
stores them in the existing StencilBank/native cache. An unchanged A_l reuses
them across V-cycles and RHS vectors. This is one generation *per learned level
operator*, not one total network call for an entire hierarchy. New learned P can
change A_l and therefore require a corresponding new smoother bank. No smoother
in this module consumes residuals except the inherited numerical application
helper used during differentiable training.

Teacher factories set `training_only=True`; the research runtime enforces this
tag outside its explicit offline teacher context. The architecture classes alone
do not authorize teacher deployment. Teacher results, inference costs, and
checkpoints must remain separate from student deployment results. No teacher or
KD advantage is claimed by creating these architectures.
