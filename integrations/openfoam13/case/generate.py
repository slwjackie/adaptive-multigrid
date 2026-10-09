#!/usr/bin/env python3
"""Generate an actual, serial OpenFOAM Foundation 13 H2/air CFD case.

Cantera supplies chemistry, thermochemistry, and viscosity fitting samples. It
does not produce the CFD trajectory; foamRun evolves the 2D reacting flow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


def foam_header(name: str, cls: str = "dictionary") -> str:
    return f"FoamFile\n{{\n    format ascii;\n    class {cls};\n    object {name};\n}}\n\n"


def digest_files(root: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        content = (root / relative).read_bytes()
        digest.update(relative.encode() + b"\0" + content + b"\0")
    return digest.hexdigest()


def validate(args: argparse.Namespace) -> None:
    for key in ("nx", "ny"):
        n = getattr(args, key)
        if n < 7 or (n + 1) & n:
            raise ValueError(f"{key} must equal 2**L - 1 and be at least 7")
    for key in ("width", "height", "depth", "pressure", "inlet_temperature",
                "hotspot_temperature", "fuel_speed", "air_speed", "end_time", "delta_t"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if args.hotspot_temperature <= args.inlet_temperature:
        raise ValueError("hotspot_temperature must exceed inlet_temperature")
    if not 0 < args.fuel_h2_mole_fraction <= 1:
        raise ValueError("fuel_h2_mole_fraction must be in (0,1]")
    if args.delta_t > args.end_time:
        raise ValueError("delta_t must not exceed end_time")
    steps = args.end_time / args.delta_t
    if not math.isfinite(steps) or not math.isclose(steps, round(steps), rel_tol=0., abs_tol=1e-8):
        raise ValueError("end_time must be an integer multiple of delta_t")
    if not Path(args.socket_path).is_absolute() or any(c in args.socket_path for c in '\n\r"'):
        raise ValueError("socket_path must be an absolute path without quotes/newlines")
    if len(args.socket_path.encode()) > 100:
        raise ValueError("Unix socket path must be at most 100 bytes")
    for key in ("case_id", "case_group"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", getattr(args, key)):
            raise ValueError(f"{key} must match [A-Za-z0-9_-]+")


def write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def scalar_field(name: str, dimensions: str, values: list[float], fuel: str,
                 air: str, outlet: str) -> str:
    internal = "\n".join(f"{v:.16g}" for v in values)
    return foam_header(name, "volScalarField") + f"""dimensions {dimensions};
internalField nonuniform List<scalar>
{len(values)}
(
{internal}
);
boundaryField
{{
    fuel {{ {fuel} }}
    air {{ {air} }}
    outlet {{ {outlet} }}
    frontAndBack {{ type empty; }}
}}
"""


def sutherland_transport(gas, temperatures) -> tuple[str, dict]:
    """Fit T**1.5/mu=(T+Ts)/As using pure-species Cantera viscosities.

    OpenFOAM chemkinToFoam expects an OpenFOAM transport dictionary as its
    third positional argument, despite the CHEMKIN label in its help text.
    """
    import numpy as np

    samples = []
    for temperature in temperatures:
        gas.TP = float(temperature), 101325.0
        samples.append(gas.species_viscosities.copy())
    samples = np.asarray(samples)
    fit_report = {}
    pieces = []
    for i, name in enumerate(gas.species_names):
        viscosity = samples[:, i]
        slope, intercept = np.polyfit(temperatures, temperatures ** 1.5 / viscosity, 1)
        As, Ts = 1.0 / slope, intercept / slope
        if As <= 0 or np.any(temperatures + Ts <= 0):
            raise ValueError(f"nonphysical Sutherland fit for {name}")
        approximation = As * temperatures ** 1.5 / (temperatures + Ts)
        fit_report[name] = {"As": float(As), "Ts": float(Ts),
                            "max_relative_error": float(np.max(abs(approximation / viscosity - 1)))}
        pieces.append(f"{name}\n{{\n    transport {{ As {As:.16g}; Ts {Ts:.16g}; }}\n}}\n")
    return "\n".join(pieces), fit_report


def generate(args: argparse.Namespace) -> Path:
    validate(args)
    steps = int(round(args.end_time / args.delta_t))
    # The final physical time must be written for paired field comparison.
    write_interval = math.gcd(steps, max(1, int(round(steps / 10))))
    try:
        import cantera as ct
        from cantera import yaml2ck
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Case generation needs optional Cantera >=3.0: python -m pip install 'cantera>=3.0,<4'") from error

    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing case: {root}")
    gas = ct.Solution("h2o2.yaml")
    if "H2" not in gas.species_names or "O2" not in gas.species_names or "N2" not in gas.species_names:
        raise ValueError("h2o2.yaml must contain H2, O2, N2")
    # Prepare all chemistry in a new directory. No existing case is ever mutated.
    root.mkdir(parents=True)
    (root / "chemkin").mkdir()
    gas.write_yaml(root / "chemkin/source.yaml")
    yaml2ck.convert(gas, mechanism_path=root / "chemkin/h2.inp",
                    thermo_path=root / "chemkin/thermo.dat",
                    transport_path=root / "chemkin/transport.dat")
    # Remove only volatile writer timestamps so identical case definitions have
    # identical provenance hashes when generated at different wall-clock times.
    for path in (root / "chemkin").iterdir():
        path.write_text("\n".join(line for line in path.read_text().splitlines()
                                  if not line.startswith(("date:", "! date:"))) + "\n")
    temperatures = np.linspace(300., 2500., 40)
    transport, fit_report = sutherland_transport(gas, temperatures)
    write(root, "chemkin/transport.foam", transport)
    write(root, "chemkin/transport-fit.json", json.dumps({
        "method": "linear least squares of T**1.5/mu(T)", "temperature_range_K": [300, 2500],
        "sample_count": 40, "species": fit_report,
        "limitation": "Sutherland fit and unity Lewis transport differ from Cantera mixture transport; no Soret"
    }, indent=2) + "\n")
    gas.TPX = args.inlet_temperature, args.pressure, {"O2": 1, "N2": 3.76}
    air_y = gas.Y.copy()
    gas.TPX = args.inlet_temperature, args.pressure, {
        "H2": args.fuel_h2_mole_fraction, "N2": 1 - args.fuel_h2_mole_fraction}
    fuel_y = gas.Y.copy()

    # blockMesh's single Cartesian block uses x-fastest ordering. A localized
    # y-dependent ignition kernel intentionally makes initial fields truly 2D.
    xs = (np.arange(args.nx) + .5) * args.width / args.nx
    ys = ((np.arange(args.ny) + .5) / args.ny - .5) * args.height
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    mixture_fraction = .5 * (1 - np.tanh((xx - .5 * args.width) / (.08 * args.width)))
    oxygen_per_hydrogen = .5 * gas.molecular_weights[gas.species_index("O2")] / gas.molecular_weights[gas.species_index("H2")]
    oxygen_air = air_y[gas.species_index("O2")]
    hydrogen_fuel = fuel_y[gas.species_index("H2")]
    z_st = oxygen_air / (oxygen_per_hydrogen * hydrogen_fuel + oxygen_air)
    hotspot_x = float(.5 * args.width + .08 * args.width * np.arctanh(1 - 2 * z_st))
    hotspot = np.exp(-((xx - hotspot_x) / (.10 * args.width)) ** 2
                     - (yy / (.20 * args.height)) ** 2)
    temperature = args.inlet_temperature + (args.hotspot_temperature - args.inlet_temperature) * hotspot
    n = args.nx * args.ny
    fixed = lambda value: f"type fixedValue; value uniform {value:.16g};"
    inlet_outlet = lambda value: f"type inletOutlet; inletValue uniform {value:.16g}; value uniform {value:.16g};"
    for i, name in enumerate(gas.species_names):
        values = (mixture_fraction * fuel_y[i] + (1 - mixture_fraction) * air_y[i]).ravel().tolist()
        write(root, f"0/{name}", scalar_field(name, "[0 0 0 0 0 0 0]", values,
              fixed(fuel_y[i]), fixed(air_y[i]), inlet_outlet(air_y[i])))
    write(root, "0/T", scalar_field("T", "[0 0 0 1 0 0 0]", temperature.ravel().tolist(),
          fixed(args.inlet_temperature), fixed(args.inlet_temperature), inlet_outlet(args.inlet_temperature)))
    write(root, "0/p", scalar_field("p", "[1 -1 -2 0 0 0 0]", [args.pressure] * n,
          "type zeroGradient;", "type zeroGradient;", fixed(args.pressure)))
    write(root, "0/U", foam_header("U", "volVectorField") + f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
    fuel {{ type fixedValue; value uniform ({args.fuel_speed:.16g} 0 0); }}
    air {{ type fixedValue; value uniform ({-args.air_speed:.16g} 0 0); }}
    outlet {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
    frontAndBack {{ type empty; }}
}}
""")
    write(root, "constant/physicalProperties", foam_header("physicalProperties") + """thermoType
{
    type hePsiThermo;
    mixture coefficientWilkeMulticomponentMixture;
    transport sutherland;
    thermo janaf;
    energy sensibleEnthalpy;
    equationOfState perfectGas;
    specie specie;
}
defaultSpecie N2;
#include "thermo.compressibleGas"
""")
    write(root, "constant/chemistryProperties", foam_header("chemistryProperties") + """chemistryType { solver ode; }
chemistry on;
initialChemicalTimeStep 1e-8;
odeCoeffs { solver Rosenbrock43; absTol 1e-10; relTol 1e-6; }
#include "reactions"
""")
    write(root, "constant/combustionProperties", foam_header("combustionProperties") + "combustionModel laminar;\n")
    write(root, "constant/momentumTransport", foam_header("momentumTransport") + "simulationType laminar;\n")
    write(root, "constant/thermophysicalTransport", foam_header("thermophysicalTransport") + "laminar { model unityLewisFourier; }\n")

    w, h, d = args.width, args.height / 2, args.depth / 2
    vertices = [(0,-h,-d),(w,-h,-d),(w,h,-d),(0,h,-d),(0,-h,d),(w,-h,d),(w,h,d),(0,h,d)]
    vertex_text = "\n".join("    (" + " ".join(f"{v:.16g}" for v in vertex) + ")" for vertex in vertices)
    write(root, "system/blockMeshDict", foam_header("blockMeshDict") + f"""convertToMeters 1;
vertices
(
{vertex_text}
);
blocks (hex (0 1 2 3 4 5 6 7) ({args.nx} {args.ny} 1) simpleGrading (1 1 1));
boundary
(
    fuel {{ type patch; faces ((0 4 7 3)); }}
    air {{ type patch; faces ((1 2 6 5)); }}
    outlet {{ type patch; faces ((0 1 5 4) (7 6 2 3)); }}
    frontAndBack {{ type empty; faces ((4 5 6 7) (0 3 2 1)); }}
);
""")
    write(root, "system/controlDict", foam_header("controlDict") + f"""solver multicomponentFluid;
libs ("libAdaptiveFixedP.so");
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {args.end_time:.16g};
deltaT {args.delta_t:.16g};
adjustTimeStep no;
writeControl timeStep;
writeInterval {write_interval};
purgeWrite 0;
writeFormat ascii;
writePrecision 16;
writeCompression off;
timeFormat general;
timePrecision 12;
runTimeModifiable false;
functions
{{
    heatRelease
    {{
        type Qdot;
        libs ("libcombustionModels.so");
        executeControl timeStep;
        executeInterval 1;
        writeControl writeTime;
    }}
}}
""")
    write(root, "system/fvSchemes", foam_header("fvSchemes") + """ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) Gauss limitedLinearV 1;
    div(phi,Yi_h) Gauss limitedLinear 1;
    div(phi,K) Gauss limitedLinear 1;
    div(phid,p) Gauss limitedLinear 1;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
""")
    write(root, "system/fvSolution", foam_header("fvSolution") + f"""solvers
{{
    "rho.*" {{ solver diagonal; }}
    p
    {{
        solver AdaptiveFixedP;
        mode {args.mode};
        socketPath "{args.socket_path}";
        nx {args.nx};
        ny {args.ny};
        adaptiveAtol 1e-12;
        adaptiveRtol 1e-8;
        tolerance 1e-10;
        relTol 0;
        maxIter 150;
        socketTimeout 120;
    }}
    pFinal {{ $p; }}
    "(U|h)" {{ solver PBiCGStab; preconditioner DILU; tolerance 1e-9; relTol 0; }}
    "(U|h)Final" {{ $U; }}
    "Yi.*" {{ $h; }}
}}
PIMPLE
{{
    transonic no;
    momentumPredictor no;
    nOuterCorrectors 1;
    nCorrectors 2;
    nNonOrthogonalCorrectors 0;
}}
""")
    # Native-only reference can be used before the bridge is available. The
    # bridge's native mode additionally records finalized matrices for training.
    native_solution = (root / "system/fvSolution").read_text().replace("solver AdaptiveFixedP;", "solver PCG;\n        preconditioner DIC;")
    write(root, "system/fvSolution.native", native_solution)
    initial_paths = [f"0/{name}" for name in [*gas.species_names, "T", "p", "U"]]
    boundary_text = "\n".join((root / p).read_text().split("boundaryField", 1)[1] for p in sorted(initial_paths))
    chemistry_files = ["chemkin/source.yaml", "chemkin/h2.inp", "chemkin/thermo.dat", "chemkin/transport.foam"]
    contract = {
        "schema": "h2-fixed-p-case-v1", "case_id": args.case_id, "case_group": args.case_group,
        "split": args.split,
        "physics": {"fuel": "H2", "oxidizer": "air", "dimension": 2, "fixed_grid": True,
                    "solver": "OpenFOAM-13 multicomponentFluid", "chemistry": "Cantera h2o2.yaml",
                    "transport": "Sutherland + unityLewisFourier; no Soret", "laminar": True},
        "chemistry_sha256": digest_files(root, chemistry_files), "chemistry_files": chemistry_files,
        "initial_conditions_sha256": digest_files(root, initial_paths),
        "boundary_conditions_sha256": hashlib.sha256(boundary_text.encode()).hexdigest(),
        "shape": [args.nx, args.ny], "end_time": args.end_time, "delta_t": args.delta_t,
        "n_steps": steps, "write_interval": write_interval,
        "geometry_m": [args.width, args.height, args.depth], "pressure_Pa": args.pressure,
        "inlet_temperature_K": args.inlet_temperature, "hotspot_temperature_K": args.hotspot_temperature,
        "initial_hotspot_x_m": hotspot_x, "stoichiometric_mixture_fraction": float(z_st),
        "fuel_h2_mole_fraction": args.fuel_h2_mole_fraction,
        "inlet_speeds_m_s": {"fuel": args.fuel_speed, "air": args.air_speed},
        "cantera_version": ct.__version__, "species": gas.species_names, "reaction_count": gas.n_reactions,
        "initial_cell_order": "single Cartesian block: x-fastest, then y; no renumberMesh",
        "status": "generated_not_executed", "foam_conversion_verified": False,
        "provenance": "OpenFOAM-13 counterFlowFlame2D dictionary conventions; new H2 initial/boundary fields",
    }
    write(root, "contract.json", json.dumps(contract, indent=2) + "\n")
    for name in ("run_case.py", "Allrun", "Allclean"):
        shutil.copyfile(Path(__file__).with_name(name), root / name)
        (root / name).chmod(0o755)
    return root


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--nx", type=int, default=63)
    p.add_argument("--ny", type=int, default=63)
    p.add_argument("--case-id", required=True)
    p.add_argument("--case-group", required=True, help="All closely related trajectories must share a split")
    p.add_argument("--split", choices=("train", "tune", "validation", "test"), required=True)
    p.add_argument("--mode", choices=("native", "classical", "hs"), default="native")
    p.add_argument("--socket-path", default="/tmp/h2.sock")
    p.add_argument("--width", type=float, default=.02)
    p.add_argument("--height", type=float, default=.02)
    p.add_argument("--depth", type=float, default=.001)
    p.add_argument("--pressure", type=float, default=101325.)
    p.add_argument("--inlet-temperature", type=float, default=300.)
    p.add_argument("--hotspot-temperature", type=float, default=1800.)
    p.add_argument("--fuel-h2-mole-fraction", type=float, default=1.)
    p.add_argument("--fuel-speed", type=float, default=.1)
    p.add_argument("--air-speed", type=float, default=.1)
    p.add_argument("--end-time", type=float, default=.002)
    p.add_argument("--delta-t", type=float, default=1e-6)
    return p


if __name__ == "__main__":
    print(generate(parser().parse_args()))
