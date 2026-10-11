#!/usr/bin/env python3
"""Generate an inert He/N2 counterflow case for OpenFOAM Foundation 13.

This is a compressible, pressure-based calculation operated at low Mach number,
not a custom incompressible projection solver. Temperature starts uniform but
the energy equation remains active. No Cantera or chemical ODE is required.

Foundation 13 source contracts checked for this generator:
https://github.com/OpenFOAM/OpenFOAM-13/blob/master/src/thermophysicalModels/multicomponentThermo/psiMulticomponentThermo/psiMulticomponentThermos.C
https://github.com/OpenFOAM/OpenFOAM-13/blob/master/src/thermophysicalModels/specie/include/forGases.H
https://cpp.openfoam.org/v13/noCombustion_8C_source.html
https://github.com/OpenFOAM/OpenFOAM-13/blob/master/applications/modules/isothermalFluid/isothermalFluid.C
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil


_spec = importlib.util.spec_from_file_location("_case_dictionary_helpers", Path(__file__).with_name("generate.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
foam_header, write, scalar_field, digest_files = (
    _helpers.foam_header, _helpers.write, _helpers.scalar_field, _helpers.digest_files)

# Ideal-gas benchmark properties near 300 K, held constant by design. These
# define a controlled numerical benchmark, not a high-fidelity transport model.
SPECIES = {
    "He": {"molWeight": 4.002602, "Cp": 5193.0, "mu": 1.96e-5, "Pr": .67},
    "N2": {"molWeight": 28.0134, "Cp": 1040.0, "mu": 1.76e-5, "Pr": .72},
}
R_UNIVERSAL = 8314.46261815324  # J/(kmol K); molecular weights in kg/kmol


def validate(args: argparse.Namespace) -> None:
    for name in ("nx", "ny"):
        n = getattr(args, name)
        if n < 7 or (n + 1) & n:
            raise ValueError(f"{name} must equal 2**L - 1 and be at least 7")
    for name in ("width", "height", "depth", "pressure", "inlet_temperature",
                 "fuel_speed", "air_speed", "end_time", "delta_t"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 < args.light_mass_fraction <= 1:
        raise ValueError("light_mass_fraction must be in (0,1]")
    if args.delta_t > args.end_time:
        raise ValueError("delta_t must not exceed end_time")
    steps = args.end_time / args.delta_t
    if not math.isfinite(steps) or not math.isclose(steps, round(steps), rel_tol=0., abs_tol=1e-8):
        raise ValueError("end_time must be an integer multiple of delta_t")
    if not Path(args.socket_path).is_absolute() or any(c in args.socket_path for c in '\n\r"'):
        raise ValueError("socket_path must be an absolute path without quotes/newlines")
    if len(args.socket_path.encode()) > 100:
        raise ValueError("Unix socket path must be at most 100 bytes")
    for name in ("case_id", "case_group"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", getattr(args, name)):
            raise ValueError(f"{name} must match [A-Za-z0-9_-]+")
    # Conservative reference speed includes the counterflow's transverse exit.
    max_speed = math.hypot(max(args.fuel_speed, args.air_speed),
                           .75 * (args.fuel_speed + args.air_speed) * args.height / args.width)
    r_n2 = R_UNIVERSAL / SPECIES["N2"]["molWeight"]
    cp_n2 = SPECIES["N2"]["Cp"]
    sound_speed = math.sqrt(cp_n2 / (cp_n2 - r_n2) * r_n2 * args.inlet_temperature)
    if max_speed / sound_speed >= .1:
        raise ValueError("reference Mach number must be below 0.1 for this low-Mach benchmark")


def initial_fields(args: argparse.Namespace) -> dict[str, list]:
    """x-fastest cell ordering; smooth wavy composition and inlet-matched flow."""
    values = {name: [] for name in ("He", "N2", "T", "p", "rho", "U")}
    for j in range(args.ny):
        y = ((j + .5) / args.ny - .5) * args.height
        interface = args.width * (.5 + .08 * math.sin(2 * math.pi * y / args.height))
        for i in range(args.nx):
            s = (i + .5) / args.nx
            x = s * args.width
            he = args.light_mass_fraction * .5 * (1 - math.tanh((x - interface) / (.08 * args.width)))
            r_mix = R_UNIVERSAL * (he / SPECIES["He"]["molWeight"] + (1 - he) / SPECIES["N2"]["molWeight"])
            # Streamfunction f(x)*y, with f'(0)=f'(width)=0. This is
            # divergence-free initially and matches both uniform inlet vectors.
            u = args.fuel_speed - (args.fuel_speed + args.air_speed) * s * s * (3 - 2 * s)
            v = 6 * (args.fuel_speed + args.air_speed) * s * (1 - s) * y / args.width
            for name, value in (("He", he), ("N2", 1 - he), ("T", args.inlet_temperature),
                                ("p", args.pressure), ("rho", args.pressure / (r_mix * args.inlet_temperature)),
                                ("U", (u, v, 0.))):
                values[name].append(value)
    return values


def generate(args: argparse.Namespace) -> Path:
    validate(args)
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing case: {root}")
    fields = initial_fields(args)
    n = args.nx * args.ny
    steps = int(round(args.end_time / args.delta_t))
    write_interval = math.gcd(steps, max(1, int(round(steps / 10))))
    fixed = lambda v: f"type fixedValue; value uniform {v:.16g};"
    inlet_outlet = lambda v: f"type inletOutlet; inletValue uniform {v:.16g}; value uniform {v:.16g};"
    for name, left, right in (("He", args.light_mass_fraction, 0.), ("N2", 1 - args.light_mass_fraction, 1.)):
        write(root, f"0/{name}", scalar_field(name, "[0 0 0 0 0 0 0]", fields[name],
                                             fixed(left), fixed(right), inlet_outlet(right)))
    write(root, "0/T", scalar_field("T", "[0 0 0 1 0 0 0]", fields["T"],
          fixed(args.inlet_temperature), fixed(args.inlet_temperature), inlet_outlet(args.inlet_temperature)))
    write(root, "0/p", scalar_field("p", "[1 -1 -2 0 0 0 0]", fields["p"],
          "type zeroGradient;", "type zeroGradient;", fixed(args.pressure)))
    # rho is deliberately not prescribed: multicomponentFluid computes it from
    # its thermodynamics and isothermalFluid declares it AUTO_WRITE in OF13.
    vectors = "\n".join("(" + " ".join(f"{v:.16g}" for v in row) + ")" for row in fields["U"])
    write(root, "0/U", foam_header("U", "volVectorField") + f"""dimensions [0 1 -1 0 0 0 0];
internalField nonuniform List<vector>
{n}
(
{vectors}
);
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
    transport const;
    thermo hConst;
    energy sensibleEnthalpy;
    equationOfState perfectGas;
    specie specie;
}
defaultSpecie N2;
#include "thermo.compressibleGas"
""")
    thermo = foam_header("thermo.compressibleGas") + "species (He N2);\n\n"
    for name, prop in SPECIES.items():
        thermo += f"""{name}
{{
    specie {{ molWeight {prop['molWeight']:.16g}; }}
    thermodynamics {{ Cp {prop['Cp']:.16g}; hf 0; }}
    transport {{ mu {prop['mu']:.16g}; Pr {prop['Pr']:.16g}; }}
}}
"""
    write(root, "constant/thermo.compressibleGas", thermo)
    # none never constructs a chemistry model; chemistry off is explicit
    # provenance, not an additional chemical integration pathway.
    write(root, "constant/chemistryProperties", foam_header("chemistryProperties") + "chemistry off;\n")
    write(root, "constant/combustionProperties", foam_header("combustionProperties") + "combustionModel none;\n")
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
    solution = foam_header("fvSolution") + f"""solvers
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
"""
    write(root, "system/fvSolution", solution)
    write(root, "system/fvSolution.native", solution.replace("solver AdaptiveFixedP;", "solver PCG;\n        preconditioner DIC;"))
    initial_paths = ["0/He", "0/N2", "0/T", "0/p", "0/U"]
    boundary_text = "\n".join((root / p).read_text().split("boundaryField", 1)[1] for p in sorted(initial_paths))
    thermo_paths = ["constant/physicalProperties", "constant/thermo.compressibleGas",
                    "constant/combustionProperties", "constant/chemistryProperties",
                    "constant/thermophysicalTransport", "constant/momentumTransport"]
    max_speed = max(math.sqrt(sum(v*v for v in row)) for row in fields["U"])
    r_n2 = R_UNIVERSAL / SPECIES["N2"]["molWeight"]
    sound_speed = math.sqrt(SPECIES["N2"]["Cp"] / (SPECIES["N2"]["Cp"] - r_n2) * r_n2 * args.inlet_temperature)
    contract = {
        "schema": "variable-density-case-v1", "case_id": args.case_id, "case_group": args.case_group,
        "split": args.split, "benchmark_role": "primary",
        "physics": {"kind": "variable_density_nonreacting", "reacting": False,
                    "dimension": 2, "fixed_grid": True, "solver": "OpenFOAM-13 multicomponentFluid",
                    "flow": "inert He/N2 counterflow mixing", "chemistry": "none",
                    "equation_of_state": "perfectGas", "laminar": True,
                    "formulation": "compressible pressure-based, operated at low Mach number; energy equation active",
                    "transport": "constant species viscosity and Cp, Wilke mixture, unityLewisFourier; no Soret"},
        "preparation_kind": "native_thermophysical",
        "thermophysical_files": thermo_paths, "thermophysical_sha256": digest_files(root, thermo_paths),
        "initial_conditions_sha256": digest_files(root, initial_paths),
        "boundary_conditions_sha256": hashlib.sha256(boundary_text.encode()).hexdigest(),
        "shape": [args.nx, args.ny], "end_time": args.end_time, "delta_t": args.delta_t,
        "n_steps": steps, "write_interval": write_interval,
        "geometry_m": [args.width, args.height, args.depth], "pressure_Pa": args.pressure,
        "inlet_temperature_K": args.inlet_temperature, "light_mass_fraction": args.light_mass_fraction,
        "inlet_speeds_m_s": {"fuel": args.fuel_speed, "air": args.air_speed},
        "patch_meanings": {"fuel": "light He/N2 inlet", "air": "heavy N2 inlet"},
        "species": ["He", "N2"], "reaction_count": 0,
        "comparison_fields": ["T", "He", "N2", "rho", "U", "p"],
        "species_properties": SPECIES,
        "initial_density_range_kg_m3": [min(fields["rho"]), max(fields["rho"])],
        "initial_reference_mach": max_speed / sound_speed,
        "initial_advective_courant_estimate": args.delta_t * max(
            abs(u) * args.nx / args.width + abs(v) * args.ny / args.height for u, v, _ in fields["U"]),
        "initial_cell_order": "single Cartesian block: x-fastest, then y; no renumberMesh",
        "status": "generated_not_executed", "foam_conversion_verified": False,
        "provenance": "native Foundation 13 dictionaries; idealized inert binary-gas benchmark",
        "limitations": ["Pressure dominance and useful matrix drift must be measured, not assumed.",
                        "Constant-property, unity-Lewis transport is a numerical benchmark approximation.",
                        "Initial and inlet T are uniform; temperature is not externally fixed during evolution."],
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
    p.add_argument("--case-group", required=True, help="Closely related trajectories must share a split")
    p.add_argument("--split", choices=("train", "tune", "validation", "test"), required=True)
    p.add_argument("--mode", choices=("native", "classical", "hs"), default="native")
    p.add_argument("--socket-path", default="/tmp/vd.sock")
    p.add_argument("--width", type=float, default=.02)
    p.add_argument("--height", type=float, default=.02)
    p.add_argument("--depth", type=float, default=.001)
    p.add_argument("--pressure", type=float, default=101325.)
    p.add_argument("--inlet-temperature", type=float, default=300.)
    p.add_argument("--light-mass-fraction", type=float, default=.8)
    p.add_argument("--fuel-speed", type=float, default=1., help="Light-gas inlet speed; inherited patch name")
    p.add_argument("--air-speed", type=float, default=1., help="N2 inlet speed; inherited patch name")
    p.add_argument("--end-time", type=float, default=.01)
    p.add_argument("--delta-t", type=float, default=1e-4)
    return p


if __name__ == "__main__":
    print(generate(parser().parse_args()))
