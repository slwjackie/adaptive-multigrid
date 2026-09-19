# Fresh strong-aware policy refit

`research_policy.py` provides `refit_research_policy`, `ResearchPolicy.save/load`
and `PreparedResearchMG`. It leaves `strong.py` unchanged. Production
certification remains false until the separate untouched-final-test process.

Each train sample must belong to a committed new development manifest, with its
operator/RHS fingerprint and stored `C*(A)` matching the unchanged selector.
Historical operators, overlapping normalized operators, final and OOD labels are
rejected. The refitter copies expert models: training policies never mutates the
direct Student or Teacher checkpoints.

WHERE uses a shared eight-input tiny detector. New numerical labels compare
local residual damping per measured smoothing time using the exact selected
classical smoother. Both H_S and H_SP experts can contribute labels from their
actual hierarchies. No genuine neural-positive blocks results in a closed
detector. These labels are local efficiency proxies, not final accuracy evidence.

WHEN uses a separately initialized 28-input/four-output tiny controller for each
H_S/H_P/H_SP branch. The existing paired C-versus-H-burst-plus-C-tail label
routine is reused with each sample's recorded classical strategy. It compares
identical residual states and fixed original stopping thresholds. Labels include
measured operator-generation overhead, instability and contraction efficiency.
The deployed numerical loop retains the existing compiled NumPy tiny inference,
break-even prediction, hysteresis, rollback and irreversible CLASSICAL LOCK.

WHICH is deliberately a conservative small table keyed only by the selector's
A-only rule ID and grid size, plus expected remaining RHS and available cached
branch. Cold and genuinely cached same-A solves are measured offline for C and
each candidate. Their medians estimate totals for 1/4/16/64 RHS; these estimates
are training labels and do not replace the separate measured multiple-RHS
benchmark. Each table entry requires both train and disjoint validation evidence,
all successful solves, actual accepted neural work and a lower log-speedup bound
above the margin. The default requirement is three unique operators in each
split/bucket. The uncertainty bound is mean minus 1.96 standard errors with a
0.02 floor; it is a conservative development heuristic, not a certified interval.
Unseen/undersampled/unfavorable buckets choose C. There are no online trial
solves or case-label lookups. In particular a small smoke run can legitimately
produce a policy that abstains everywhere.

Policies pin exact expert/controller/detector signatures, generator architecture
and complexity metadata, frozen selector digest, numerical/inference config,
and measured hardware/thread environment. Changes invalidate eligibility.
`PreparedResearchMG` chooses one eligible branch at solve/batch entry, then the
existing temporal controller can still decline or leave that branch. Operator
banks are reused across RHS. Explicit C stays exactly the current strong
classical numerical path. Cache identity includes the policy and expected-RHS
scope; no performance certificate is issued by this adapter.

A separately successful untouched-final run may export a **new** policy copy
through `policy.attach_final_certificate(certificate)`. The original frozen
policy files, tiny networks, experts and bucket table remain unchanged. The
certificate pins that policy digest, the selector, configuration, hardware,
final-freeze file and final-assessment file. Attachment and runtime check the
assessment's exact acceptance fields: at least 20 final operators and five
repeats, the same FP64 tolerance, no new failures, actual neural use, speedup
above the configured margin (at least 1.03), and CI lower bound above one. The
separate final-run certifier is responsible for raw-run/manifest provenance and
single-use final-claim validation; this adapter checks its exported evidence and
deployed identity. These local artifacts are not cryptographic authentication.

Only matching certified `auto` policies can activate the new production path.
Uncertified, stale, mismatched, Teacher-containing or altered-evidence policies
continue to choose C. Production-mode explicit H ablations do not inherit this
certificate. Evidence content hashes are rechecked when file identity, size,
modification time or change time changes, avoiding repeated reads of immutable
evidence during ordinary solves. Existing `strong.py`, legacy solver
certification and frozen `Components.metadata['certificate']` remain unchanged.
