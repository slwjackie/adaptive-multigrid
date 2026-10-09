# OpenFOAM Foundation 13 fixed-P / learned-smoother bridge

This integration targets **OpenFOAM Foundation 13, Linux, double precision,
serial execution**. OpenCFD releases such as v2312/v2406 are different APIs.
The native library is separate from all existing Python numerical kernels.
It does not contain a World Model. Classical and H_S execution retain the
initial classical interpolation hierarchy, while updating its numerical
coarse operators for the current pressure matrix.

## Status and scope

The bridge source, framing decoder, Python service interface, and case generator
are supplied. A standalone C++ decoder test can run without OpenFOAM.
**The OpenFOAM plugin has not been compiled or run in the development container,
which has no OpenFOAM installation.** Successful Python tests do not establish
that native coupling is validated. The Unix socket integration test is also
blocked by the container's socket-creation restriction; run it on the target
Linux machine before the native acceptance run. No H2 CFD speedup is claimed.

The supplied case generator creates a fixed Cartesian, one-cell-thick 2D
counterflow H2/air case with actual chemical species and energy evolution.
This is a numerical integration benchmark, not a validated prediction of a
specific experimental flame. See `case/README.md` for chemistry/transport
assumptions and generation commands. In particular, OpenFOAM transport is not
automatically equivalent to Cantera mixture-averaged/Soret transport.

Admission is deliberately limited to pressure field `p`, symmetric finalized
LDU storage, an anchored SPD operator, no coupled patches, and uniform Cartesian
x-y centres with exactly one z layer. Each x/y dimension must be `2**L-1`, at
least 3. Mesh motion, AMR, MPI, cyclic patches, transonic asymmetric operators,
and unanchored pure-Neumann systems are rejected. The geometric mapping is
constructed from actual cell centres and checked against internal face
adjacency. It does not assume native cell ordering.

## Build and decoder checks

From the repository root, after activating the Python environment:

```bash
python -m pip install -e '.[dev]'
python integrations/openfoam13/tests/check_wire.py
python integrations/openfoam13/tests/check_wire.py --ipc

# Use the installation path on your target machine.
source /opt/openfoam13/etc/bashrc
integrations/openfoam13/Allwmake
```

`check_wire.py` compiles with C++11, `-Wall -Wextra -Werror`, and checks strict
JSON handling. `--ipc` additionally checks partial reads and two requests over
one socket. These tests do not compile the OpenFOAM-facing class.

## Solver dictionary

The generated case loads `libAdaptiveFixedP.so` in `system/controlDict`. Its
`p` and `pFinal` entries in `system/fvSolution` use these controls:

```text
solver          AdaptiveFixedP;
mode            classical;       // classical, hs, or native
socketPath      "/tmp/h2-fixed-p.sock";
nx              63;
ny              63;
adaptiveAtol    1e-12;
adaptiveRtol    1e-8;
maxIter         150;
socketTimeout   120;
tolerance       1e-12;
relTol          0;
```

The plugin and Python config must agree on `adaptiveAtol`, `adaptiveRtol` and
`maxIter`. The actual stop rule is the **un-normalized Euclidean residual**:

```text
||b - A*x||_2 <= max(adaptiveAtol, adaptiveRtol * ||b - A*x0||_2)
```

`tolerance`/`relTol` are conventional dictionary entries, not an alternative
stop rule for this plugin. A missing service, rejected matrix, failed solve,
nonfinite value, or failed native residual check stops CFD. There is no hidden
fallback to an unrelated successful solver. H_S recovery inside the Python
backend is reported and remains within its common attempt budget.

## Live execution and training recordings

Start the persistent service before `foamRun`. For a native collection case
at `artifacts/h2-case-train`, use a fresh socket and output directory:

```bash
python scripts/run_v6_7_h2_fixed_p.py serve \
  --config configs/v6_7_h2_fixed_p.json \
  --case-contract artifacts/h2-case-train/contract.json \
  --mode native --socket /tmp/h2-fixed-p.sock \
  --record-dir artifacts/h2-record-train \
  --evidence artifacts/h2-record-train-service.jsonl
```

In a second terminal, set the generated case's mode/socket consistently, then:

```bash
foamRun -case artifacts/h2-case-train
```

The generator supplies case preparation/build commands; `blockMesh` and
`chemkinToFoam` must have completed before `foamRun`. Build each independent
physical case with its own `case_group` and split. Repeat native collection
for train, tune, validation and held-out test cases without splitting adjacent
timesteps of one physical case across these sets.

After preparing/training a study, the same service CLI accepts `--run-dir RUN`
instead of `--config`, and `--mode classical` or `--mode hs`. Timing arms do not
record LDU snapshots. The Python `coupled` command clones a prepared case and
runs matched Classical/H_S cases, collects native pressure timings and physical
fields, and checks both validity and physical differences before reporting
speedup. The top-level H2 fixed-P runbook describes that workflow.

## Native versus Python responsibilities

OpenFOAM's `fvScalarMatrix::solveSegregated` has added boundary diagonal and
source contributions before calling this `lduMatrix::solver`. The plugin sends
the matrix actually used there, with two independent witness vectors and
products produced by OpenFOAM's `Amul`. Python checks its CSR application
against these products and checks SPD admission. Python returns pressure in
native cell order. OpenFOAM then recomputes `b-A*x` with its own `Amul` before
acceptance. Native edge accumulation and SciPy CSR summation can have different
roundoff under cancellation. Initial-residual and threshold diagnostics are
compared with a roundoff allowance scaled by `||b||+||A*x0||`; the final native
acceptance check still uses the original native raw L2 target without this
allowance.

`mode native` runs actual OpenFOAM PCG/DIC first, then asks the service to record
the finalized system and its solution. Since native PCG uses a normalized L1
criterion, the plugin converts the common raw L2 target into a conservative
L1 bound and performs an explicit final raw L2 check. **PCG iterations and MG
cycles are different quantities.** This native mode is for recording/reference
checks, not a claim of identical algorithms or an optimized GAMG baseline.

Every successful call logs:

```text
H2FixedPPressureSeconds <seconds> index <i> time <t> mode <arm> residual <r> threshold <tol>
```

This encloses native geometry/mapping checks, witnesses, JSON conversion,
socket transport, Python admission/setup/solve, and native final verification.
The implementation favors auditable coupling over minimum IPC overhead;
bridge overhead can erase a small learned-smoother speedup. Entire CFD runtime
must be measured independently. Model loading belongs to the reported startup
scope; it must not be silently counted for only one arm.

## IPC contract

One AF_UNIX/SOCK_STREAM connection survives across pressure solves. Every
request/response is an 8-byte unsigned big-endian length followed by UTF-8 JSON,
with a 128 MiB limit. No pickle, executable payload, or network-facing listener
is used. The Python service must be bound to the case's `contract.json`.

Requests have `schema=h2-fixed-p-live-v1`, `op=solve|record`,
`mode=classical|hs|native`, `shape`, `time`, monotonically increasing `index`,
`mesh_id`, `boundary_id`, `atol`, `rtol`, `max_cycles`, and finalized native arrays:
`diag`, `lower_addr`, `upper_addr`, `lower`, `upper`, `b`, `x0`,
`structured_to_native`, `probe_vectors`, `probe_products`.
`structured_to_native[ix*ny+iy]` is the corresponding native cell index.
Two probes/products are serialized as `(n,2)` arrays. Additional contract fields
are `boundary_finalized=true`, `nullspace=none`, `coupled_interfaces=0`,
`producer`, `source_kind=external_cfd`, and context.

Record requests also contain `x_native_reference`, `reference_solver`,
`reference_seconds`, `reference_cycles`, and native initial/final residuals.
Successful responses require `success=true`, `x_native`, `cycles`, `threshold`,
`initial_residual`, and `final_residual`. Failed responses have `success=false`
and `error`. A record acknowledgement must return the unchanged native solution.

Mesh/boundary fingerprints use deterministic FNV-1a for runtime change detection;
they are not cryptographic provenance. The case contract and dataset/checkpoint
files use SHA-256. A new physical case requires a fresh process and service.

## Authoritative API references

- [OpenFOAM-13 finalized scalar solve](https://github.com/OpenFOAM/OpenFOAM-13/blob/master/src/finiteVolume/fvMatrices/fvScalarMatrix/fvScalarMatrix.C)
- [OpenFOAM-13 PCG source and residual convention](https://github.com/OpenFOAM/OpenFOAM-13/blob/master/src/OpenFOAM/matrices/lduMatrix/solvers/PCG/PCG.C)
- [Original methane counterflow tutorial](https://github.com/OpenFOAM/OpenFOAM-13/tree/master/tutorials/multicomponentFluid/counterFlowFlame2D)
- [Cantera hydrogen/oxygen submechanism](https://github.com/Cantera/cantera/blob/main/data/h2o2.yaml)
