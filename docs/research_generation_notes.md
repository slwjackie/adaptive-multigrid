# Generation-device and rotation diagnostics

`benchmark_generation(models, examples, cfg, rules, output, repeats=5)` accepts
`{label: Components}` and an Example or a list of Examples. Optional arguments
are `branches=('H_S','H_P','H_SP')`, `devices=None`, `warmups=1`, `order_seed=431`,
and `split='validation'`. It never loads train/validation/final files; the caller
must supply examples verified by the dataset manager. Final/test/OOD split
declarations are rejected because this helper supports architecture/device
selection. It writes a manifest, raw JSON repeats, and summary CSV. An existing
output is refused, not overwritten.

Each repeat constructs a fresh actual `PreparedStrongMG` and calls its real
`ensure_branch`, rather than timing an isolated synthetic network forward.
Cold preparation includes A-only selection, classical hierarchy construction,
model and input device conversion, FP32 inference, output conversion to CPU FP64,
sparse S/P assembly, and any learned hierarchy/factorization work. The JSON
separately records `classical_setup_seconds`, `bank_setup_seconds`,
`generation_seconds`, `cpu_device_copy_seconds`, and `cold_preparation_seconds`.
These fields overlap: do not add generation/copy to the already inclusive cold
time. Checkpoint/import/native-library loading is outside the timed region.
Device repetitions are deterministically shuffled and warmups are marked.

The second `ensure_branch` measures an unchanged-A warm lookup and asserts zero
heavy generation calls and object reuse. This benchmark does not solve an RHS:
it validates generated banks remain finite CPU FP64 and reports their operator
and transfer complexity. Separate solver evaluation measures native cycle and
end-to-end time-to-tolerance. Parameter counts and architecture metadata are
included; D4 explicitly retains its eight-view arithmetic cost.

CPU FP32 is measured. MPS FP32 and CUDA FP32 are measured only when the current
PyTorch process reports them available. An unavailable device is recorded as
`not_measured`; an unsupported operation is a failed repeat with its elapsed
cost, never a silent CPU fallback. A failed group has no successful timing
median. CUDA reports the actual device name. This Mac workflow makes no A100
measurement claim when CUDA is unavailable.

All devices start from the identical selected classical hierarchy. JSON records
every classical and actual learned level A digest. S-only input operators should
match exactly across devices; P inference rounding can change learned coarse
operators, so H_P/H_SP record exact matches or differences explicitly. Such
differences must not be presented as byte-identical intermediate A_l inputs.

A branch using a `training_only` expert runs inside the explicit offline teacher
generation context and is labeled `offline_teacher_generation`. Its time remains
a separate teacher result. That context closes before returning and does not
relax the production solver guard. A teacher residing in an unused component
does not cause the other, student-only branch to be labeled teacher execution.

`rotation_resampling_diagnostic(model, features, angles=(17,33,61))` adds a
CPU FP32 diagnostic with both reflection parities. It performs bilinear image
resampling plus the full-rank O2 action of the ten input features, evaluates the
model, transforms the reference stencil, and reports cropped interior relative
L2/max errors and gain differences. The model is copied before evaluation.

The continuous channel action agrees with the exact D4 permutation at quarter
turns. For other angles it is a chosen signed-moment representation action,
**not an exactly reassembled rotated PDE**. Output normalization is reapplied
because a fixed nine-point Euclidean norm is not invariant under every such
continuous channel action. Errors include interpolation, finite stencil support,
normalization, and boundaries/global pooling. They are reported, not hidden by
an asserted tiny tolerance. Rotation-OOD PDE solves remain the substantive
generalization test and are not measured by this helper.
