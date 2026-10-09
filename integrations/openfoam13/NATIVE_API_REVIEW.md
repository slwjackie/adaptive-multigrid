# Native API review and remaining execution gate

Target source: OpenFOAM Foundation 13, repository `OpenFOAM/OpenFOAM-13`.
This source inspection is not a replacement for `Allwmake` and a live run.

| Integration use | Foundation 13 declaration/behavior inspected |
| --- | --- |
| Solver constructor, symmetric/asymmetric runtime registration, `solve` signature | `src/OpenFOAM/matrices/lduMatrix/solvers/PCG/PCG.{H,C}` and `lduMatrix.H` |
| Final boundary diagonal and source at plugin entry | `src/finiteVolume/fvMatrices/fvScalarMatrix/fvScalarMatrix.C` |
| `matrix_.mesh().thisDb()` and cast to `const fvMesh` | `src/OpenFOAM/meshes/lduMesh/lduMesh.H`, `fvMesh.H`, `typeInfo.H`; `thisDb()` is virtual, `fvMesh` overrides it, `refCast` wraps `dynamic_cast` |
| `mesh.dynamic()`, cell centres, points, boundary access | `src/finiteVolume/fvMesh/fvMesh.H` and `fvPatch.H` |
| Native `Amul` arguments, addressing access | `src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrix.H` |
| Symmetric `matrix_.lower()` when only upper storage exists | `lduMatrix.C` explicitly returns upper coefficients through the const lower accessor |
| `controlDict_.lookup<T>` / `lookupOrDefault<T>` | `src/OpenFOAM/db/dictionary/dictionary.H` |
| `solverPerformance` full constructor and mutable iteration count | `src/OpenFOAM/matrices/LduMatrix/LduMatrix/SolverPerformance.H` |
| Lossless DP diagnostic log precision, then restoration | `messageStream.H`: `Info()` returns `OSstream&`; `Ostream.H`: `precision(int)` returns previous precision |
| Native normalized-L1 residual and `normFactor` | `PCG.C`, `lduMatrixSolver.C` |
| `p`, `pFinal`, `$p`, `(U|h)`, `(U|h)Final`, `Yi.*` dictionary selection | Official `tutorials/multicomponentFluid/counterFlowFlame2D/system/fvSolution`; module `thermophysicalPredictor.C` calls `YiEqn.solve("Yi")` |
| Multivariate convection key `div(phi,Yi_h)` | Official tutorial `fvSchemes` and module `thermophysicalPredictor.C` |

The generated dictionary keeps a separate `pFinal` entry inheriting `p` and
defines `Yi.*` so both ordinary and final species solves resolve. Its species
solver inherits the same zero-relative-tolerance scalar configuration as `h`.
This corresponds to Foundation 13's solver selection conventions.

The bridge computes and returns raw Euclidean residuals; native PCG normalized
L1 values are not compared as if they were the same metric. The native-reference
mode converts the target to an L1 sufficient condition, then verifies raw L2.
Both timing arms always undergo a final current-matrix OpenFOAM `Amul` check.
Initial residual diagnostic comparisons account for different summation order
under cancellation; this does not relax final acceptance.

The current local verification covers standalone C++ wire compilation, strict
response parsing, and Python case generation/chemistry roundtrip. Native plugin
compilation, runtime library loading, `chemkinToFoam`, mesh generation and live
flame evolution are **unverified** until run with Foundation 13. The required
target-machine gate is:

1. Run `tests/check_wire.py --ipc` and `Allwmake`.
2. Generate/prepare an independent case; run `Allrun --native-only` as a CFD
   installation check. Inspect ignition and conservation, not only exit status.
3. In a fresh clone, run native plugin recording through the service and confirm
   finalized-matrix witnesses and matched raw residuals.
4. Run matched Classical/H_S cloned cases via the top-level coupled benchmark.
5. Claim CFD performance only after the physical/timing acceptance report passes.
