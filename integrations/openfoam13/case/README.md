# H2/air case generator for OpenFOAM Foundation 13

This generates a **real serial, transient 2D reacting-flow case** for the
`multicomponentFluid` solver. It does not generate pressure snapshots by tiling a
1D flame, and it does not claim that generating dictionaries executes CFD.
OpenFOAM itself must evolve momentum, species, energy and pressure. The provided
initial condition is an unreacted H2/air mixing layer plus a localized 2D hot
ignition kernel near the stoichiometric mixture fraction. Sustained ignition,
mesh/time convergence and physical accuracy must be checked from an actual run.

The fixed Cartesian domain has opposed fuel and air inlets, pressure outlets at
both transverse edges, and empty front/back patches. There are no coupled or
cyclic patches. It uses one cell through its thickness and each in-plane count
must be `2**L - 1`, at least 7. Start with the default 63 x 63; 7 x 15 is only a
generation/interface test, not a resolved flame.

## Dependencies and generation

Use an environment containing this repository's dependencies plus optional
`cantera>=3.0,<4`. OpenFOAM Foundation 13 is a separate native dependency.

```bash
python -m pip install 'cantera>=3.0,<4'
python integrations/openfoam13/case/generate.py \
  --output artifacts/h2_train_a \
  --case-id h2_train_a --case-group inlet_a --split train \
  --nx 63 --ny 63 --mode native --socket-path /tmp/h2.sock
```

The generator refuses an existing output path. `--split` is explicit and is one
of `train`, `tune`, `validation`, `test`; related trajectories must have the same
`--case-group` and remain in one split. A counterflow diffusion flame does not
have one global premixed equivalence ratio, so no misleading `--phi` option is
provided. Vary `--fuel-h2-mole-fraction` (balance N2), inlet speeds, temperature,
pressure, geometry or ignition initialization to produce distinct cases.
`--end-time` must be an integer multiple of `--delta-t`. The generator chooses
a timestep write interval that divides that step count, so final-time fields
are always scheduled for paired physical comparison.

After sourcing Foundation 13's environment:

```bash
cd artifacts/h2_train_a
./Allrun --prepare-only
```

This runs `chemkinToFoam`, `blockMesh` and `checkMesh`, records the converted
chemistry and mesh hashes, and stops before running CFD. Start the repository's
persistent Python service with this case's `contract.json`, then use `./Allrun`.
The `native` plugin mode uses PCG and records the matrices; `classical` and `hs`
request the corresponding Python solver. The agreed plugin dictionary is:

```text
solver AdaptiveFixedP;
mode native;               // native, classical, hs
socketPath "/tmp/h2.sock";
nx 63;
ny 63;
adaptiveAtol 1e-12;
adaptiveRtol 1e-8;
maxIter 150;
```

For a native-only installation preflight, `./Allrun --native-only` temporarily
uses PCG/DIC without the plugin or service. Its ordinary OpenFOAM residual
normalization differs from the bridge's raw L2 contract, so **this preflight is
not an equal-tolerance performance comparison**. The run restores the original
solver dictionaries even on failure. A normal bridge run must have a running
service; it must not silently bypass that service.

`./Allclean --yes` explicitly removes positive-time results, generated mesh,
post-processing and logs only inside a case carrying the expected contract. It
retains the initial fields, chemistry and input dictionaries. Prefer separate
cloned cases for native, Classical MG and H_S comparisons.

## Chemistry and transport provenance

Cantera's bundled `h2o2.yaml` is the H2/O2 submechanism extracted from GRI-Mech 3.0
with N2: 10 species and 29 reactions, including third-body/falloff chemistry.
Cantera `yaml2ck` exports CHEMKIN chemistry and NASA thermochemistry. The native
OpenFOAM `chemkinToFoam` converter creates `constant/reactions` and
`constant/thermo.compressibleGas`. The full source YAML and exported input files
are retained and hashed, along with the installed Cantera version. No methane
mechanism is renamed to claim H2 support.

**OpenFOAM-13's `chemkinToFoam` third argument is an OpenFOAM species transport
dictionary**, despite its command-line label saying "CHEMKIN transport file".
This generator passes `chemkin/transport.foam`, not `transport.dat`. The latter
is retained for CHEMKIN roundtrip checks. Each species' Sutherland coefficients
are fitted from Cantera pure-species viscosities over 300--2500 K; the fit method
and maximum relative errors are saved in `transport-fit.json`.

The CFD uses Wilke/Sutherland thermophysical properties and
`unityLewisFourier` species transport. **It omits preferential diffusion and
Soret effects and is not an equivalent transport model to a Cantera H2 flame.**
It is an explicit initial numerical-solver benchmark, not a validated prediction
of hydrogen flame speed, extinction or high-pressure kinetics. Those claims
require a more appropriate validated mechanism/transport model and additional
physical verification. The default case runs for only 2 ms; reaching that end
time is not evidence of a steady or sustained flame.

## Outputs and evaluation boundary

`contract.json` contains the case/split identifiers, chemistry, initialization
and boundary hashes, shape, geometry, end time and explicit execution status.
`cfd-run.json` is written after actual execution, including failure details.
Its timer covers the `foamRun` process, including startup and normal field
output; it excludes mesh/chemistry preparation. It must not be labelled pure
pressure-solver time. Pressure timing comes from the plugin/service records.

Write settings retain T, species, velocity and pressure; a `Qdot` function object
in `libcombustionModels.so` also writes heat-release density. Inspect integrated
heat release/fuel consumption, continuity and fields, then compare identical
physical times in independent full reruns. A successful process exit alone does
not verify these physical quantities. Use clones of the same generated input
for solver comparisons, with identical initial/boundary hashes and timestep.

Cell ordering is single-block `blockMesh` **x-fastest then y**. Do not run
`renumberMesh`; bridge mapping must account for its own array order. No mesh
refinement, topology changes or parallel decomposition are supported here.

## Source references and local checks

Dictionary conventions were checked against these official Foundation 13
sources; the H2 fields and generator are new code:

- <https://github.com/OpenFOAM/OpenFOAM-13/tree/master/tutorials/multicomponentFluid/counterFlowFlame2D>
- <https://github.com/OpenFOAM/OpenFOAM-13/blob/master/applications/utilities/thermophysical/chemkinToFoam/chemkinToFoam.C>
- <https://github.com/OpenFOAM/OpenFOAM-13/blob/master/applications/utilities/thermophysical/chemkinToFoam/chemkinReader/chemkinReader.C>
- <https://github.com/OpenFOAM/OpenFOAM-13/blob/master/src/ThermophysicalTransportModels/fluid/laminar/unityLewisFourier/unityLewisFourier.H>
- <https://github.com/OpenFOAM/OpenFOAM-13/blob/master/src/combustionModels/Make/files>
- <https://github.com/Cantera/cantera/blob/main/data/h2o2.yaml>
- <https://github.com/Cantera/cantera/blob/main/interfaces/cython/cantera/yaml2ck.py>

Run `python -m pytest integrations/openfoam13/case/test_generation.py -q` for
CHEMKIN roundtrip reaction-rate/thermochemistry checks, species mass closure,
two-dimensional initialization, deterministic hashes and overwrite rejection.
These Python checks do not compile the plugin, validate `chemkinToFoam` parsing
or execute a flame. Those separate native gates remain mandatory.
