#!/usr/bin/env python3
"""Isothermal, fixed-property LOX/pure-He tank-frame sloshing (Foundation 13).

This is a mechanical VOF application: sealed 2D rectangular tank, constant
gravity and intermittent horizontal translational acceleration. It omits GOX,
pressurant injection, thermal stratification, wall heat transfer and phase
change. The gas is incompressible in this initial mechanical approximation.

Property constants were evaluated with CoolProp 7.2.0 HEOS at 90 K, 300000 Pa.
Surface tension uses pure oxygen saturation at 90 K; the helium effect on
interfacial tension is neglected. No CoolProp runtime dependency is required.
Sources and reproducible queries are saved in constant/property_basis.json.

Foundation 13 API sources:
https://doc.cfd.direct/openfoam/user-guide-v13/dambreak
https://cpp.openfoam.org/v13/acceleration_8C_source.html
https://cpp.openfoam.org/v13/Table_8H_source.html
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_slosh_common", Path(__file__).with_name("flow_generator_common.py"))
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)

PROPERTIES = {
    "lox": {"rho": 1142.5502246693018, "mu": .0001960422548871036},
    "he": {"rho": 1.597374888579797, "mu": 9.17889001360198e-6},
}
SURFACE_TENSION = .013192663477889676
PROPERTY_BASIS = {
    "package": "CoolProp", "version": "7.2.0", "backend": "HEOS",
    "temperature_K": 90., "pressure_Pa": 300000.,
    "queries": ["PropsSI('D','T',90,'P',300000,'Oxygen')",
                "PropsSI('V','T',90,'P',300000,'Oxygen')",
                "PropsSI('D','T',90,'P',300000,'Helium')",
                "PropsSI('V','T',90,'P',300000,'Helium')",
                "PropsSI('I','T',90,'Q',0,'Oxygen')"],
    "phase_properties_SI": PROPERTIES, "surface_tension_N_m": SURFACE_TENSION,
    "sources": ["https://coolprop.org/fluid_properties/fluids/Oxygen.html",
                "https://coolprop.org/fluid_properties/fluids/Helium.html",
                "https://doi.org/10.1063/1.555897",
                "https://doi.org/10.1023/B:IJOT.0000022327.04529.f3",
                "https://doi.org/10.1063/1.4768782"],
    "approximation": "pure-component fixed properties; pure oxygen saturated surface tension; pure He ullage, no GOX",
}


def validate(args):
    c.validate_common(args)
    for name in ("fill_fraction", "acceleration_amplitude", "frequency", "phase", "pulse_shift", "gravity"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError("finite " + name + " required")
    if not .15 <= args.fill_fraction <= .85:
        raise ValueError("fill_fraction must be in [.15,.85] for the initial mechanical benchmark")
    if args.gravity <= 0 or not 0 <= args.acceleration_amplitude <= .3 * args.gravity:
        raise ValueError("positive gravity and acceleration_amplitude <= .3g required")
    if args.frequency <= 0 or args.frequency * args.delta_t > .05 or abs(args.pulse_shift) > .08:
        raise ValueError("resolve positive frequency by >=20 steps; |pulse_shift| <= .08")
    if not math.isclose(args.inlet_temperature, 90., abs_tol=1e-10) or not math.isclose(args.pressure, 300000., abs_tol=1e-6):
        raise ValueError("fixed property table is bound to inlet-temperature 90 K and pressure 300000 Pa")
    # Conservative capillary timescale estimate. Actual Co and alpha Co still
    # need validation from native CFD; this does not promise time convergence.
    cell = min(args.width / args.nx, args.height / args.ny)
    capillary_dt = math.sqrt((PROPERTIES['lox']['rho'] + PROPERTIES['he']['rho']) * cell**3 / (4*math.pi*SURFACE_TENSION))
    if args.delta_t > .5 * capillary_dt:
        raise ValueError("delta_t exceeds half the estimated capillary timestep; refine time")
    if int(round(args.end_time / args.delta_t)) < 2:
        raise ValueError("at least two positive-time samples are required")


def tank_acceleration(args, time):
    return (args.acceleration_amplitude * c.pulse_envelope(time, args.end_time, args.pulse_shift)
            * math.sin(2 * math.pi * args.frequency * time + args.phase))


def motion_table(args):
    """Integrate midpoint acceleration exactly at CFD timestep endpoints.

    fv::acceleration differentiates this velocity table backward over deltaT.
    Therefore it applies the midpoint-sampled acceleration, with the native
    model's negative inertial-force sign, even during constant-velocity gaps.
    The table stores *tank velocity*, never a mislabeled acceleration.
    """
    n = int(round(args.end_time / args.delta_t))
    values = [(0., 0.)]
    for k in range(1, n + 1):
        velocity = values[-1][1] + args.delta_t * tank_acceleration(args, (k-.5) * args.delta_t)
        values.append((k * args.delta_t, velocity))
    return values


def initial_fields(args):
    height = args.fill_fraction * args.height
    dy = args.height / args.ny
    yref = args.height - .5 * dy
    rho_l, rho_g = PROPERTIES['lox']['rho'], PROPERTIES['he']['rho']
    values = {name: [] for name in ("alpha.lox", "U", "p", "p_rgh")}
    for j in range(args.ny):
        y = (j+.5) * dy
        alpha = min(1., max(0., (height - j*dy)/dy))
        rho = alpha*rho_l + (1-alpha)*rho_g
        pressure = args.pressure + rho_g*args.gravity*(yref-y) + (rho_l-rho_g)*args.gravity*max(height-y, 0.)
        for _ in range(args.nx):
            values["alpha.lox"].append(alpha)
            values["U"].append((0., 0., 0.))
            values["p"].append(pressure)
            values["p_rgh"].append(pressure + rho * args.gravity * y)
    return values


def generate(args):
    validate(args)
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing case: {root}")
    fields = initial_fields(args)
    n = args.nx * args.ny
    steps = int(round(args.end_time / args.delta_t))
    write_interval = math.gcd(steps, max(1, int(round(steps / 20))))
    c.write(root, "0/alpha.lox", c.field("alpha.lox", "[0 0 0 0 0 0 0]", fields['alpha.lox'],
        {"walls": "type zeroGradient;"}))
    c.write(root, "0/U", c.field("U", "[0 1 -1 0 0 0 0]", fields['U'],
        {"walls": "type fixedValue; value uniform (0 0 0);"}))
    c.write(root, "0/p_rgh", c.field("p_rgh", "[1 -1 -2 0 0 0 0]", fields['p_rgh'],
        {"walls": f"type fixedFluxPressure; value uniform {args.pressure:.16g};"}))
    # The module reconstructs p from p_rgh and rho*gh (NO_READ). This analytic
    # copy binds the diagnostic initial state; it is not a second pressure BC.
    c.write(root, "0/p", c.field("p", "[1 -1 -2 0 0 0 0]", fields['p'],
        {"walls": f"type calculated; value uniform {args.pressure:.16g};"}))
    c.write(root, "system/blockMeshDict", c.block_mesh(args, [
        ("walls", "wall", "(0 4 7 3) (1 2 6 5) (0 1 5 4) (7 6 2 3)")]))
    c.write(root, "constant/phaseProperties", c.foam_header("phaseProperties") + f"phases (lox he);\nsigma {SURFACE_TENSION:.17g};\n")
    for phase, prop in PROPERTIES.items():
        c.write(root, "constant/physicalProperties."+phase, c.foam_header("physicalProperties."+phase)
            + f"viscosityModel constant;\nnu {prop['mu']/prop['rho']:.17g};\nrho {prop['rho']:.17g};\n")
    c.write(root, "constant/momentumTransport", c.foam_header("momentumTransport") + "simulationType laminar;\n")
    c.write(root, "constant/g", c.foam_header("g", "uniformDimensionedVectorField")
        + f"dimensions [0 1 -2 0 0 0 0];\nvalue (0 {-args.gravity:.17g} 0);\n")
    table = motion_table(args)
    table_text = "\n".join(f"        ({time:.17g} ({velocity:.17g} 0 0))" for time, velocity in table)
    c.write(root, "constant/fvModels", c.foam_header("fvModels") + f"""tankFrame
{{
    type acceleration;
    cellZone all;
    U U;
    velocity
    {{
        type table;
        values
        (
{table_text}
        );
        outOfBounds clamp;
        interpolationScheme linear;
    }}
}}
""")
    c.write(root, "constant/property_basis.json", json.dumps(PROPERTY_BASIS, indent=2)+"\n")
    c.write(root, "system/controlDict", c.foam_header("controlDict") + f"""solver incompressibleVoF;
libs ("libAdaptiveFixedP.so");
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {args.end_time:.16g};
deltaT {args.delta_t:.16g};
adjustTimeStep no;
maxCo 0.5;
maxAlphaCo 0.25;
writeControl timeStep;
writeInterval {write_interval};
purgeWrite 0;
writeFormat ascii;
writePrecision 16;
writeCompression off;
timeFormat general;
timePrecision 12;
runTimeModifiable false;
""")
    c.write(root, "system/fvSchemes", c.foam_header("fvSchemes") + """ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,alpha) Gauss interfaceCompression vanLeer 1;
    div(rhoPhi,U) Gauss limitedLinearV 1;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
""")
    # v13 source now uses nCorrectors/nSubCycles in alpha solver controls;
    # historical nAlphaCorr/nAlphaSubCycles are not used here.
    solution = c.foam_header("fvSolution") + f"""solvers
{{
    "alpha.lox.*"
    {{
        nCorrectors 2;
        nSubCycles 2;
        MULESCorr yes;
        MULES {{ nIter 10; tolerance 1e-3; }}
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-10;
        relTol 0;
    }}
    "pcorr.*" {{ solver PCG; preconditioner DIC; tolerance 1e-10; relTol 0; maxIter 600; }}
    p_rgh
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
        maxIter 600;
        socketTimeout 120;
    }}
    p_rghFinal {{ $p_rgh; }}
    "U.*" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0; minIter 1; }}
}}
PIMPLE
{{
    momentumPredictor no;
    nOuterCorrectors 1;
    nCorrectors 2;
    nNonOrthogonalCorrectors 0;
    pRefCell {n-1};
    pRefValue {args.pressure:.16g};
}}
relaxationFactors {{ equations {{ ".*" 1; }} }}
"""
    c.write(root, "system/fvSolution", solution)
    c.write(root, "system/fvSolution.native", solution.replace("solver AdaptiveFixedP;", "solver PCG; preconditioner DIC;"))
    thermo_files = ["constant/phaseProperties", "constant/physicalProperties.lox", "constant/physicalProperties.he",
                    "constant/momentumTransport", "constant/g", "constant/fvModels", "constant/property_basis.json"]
    k = math.pi / args.width
    omega = math.sqrt(args.gravity * k * math.tanh(k * args.fill_fraction * args.height))
    contract = dict(schema="lox-he-sloshing-case-v1", experiment_kind="lox_he_sloshing",
        case_id=args.case_id, case_group=args.case_group, split=args.split, benchmark_role="primary",
        physics=dict(kind="lox_he_sloshing", reacting=False, dimension=2, fixed_grid=True, isothermal=True,
                     phase_change=False, gas_phase_approximation="pure_helium", solver="OpenFOAM-13 incompressibleVoF",
                     flow="sealed LOX-He tank with intermittent horizontal tank-frame acceleration",
                     laminar=True, formulation="incompressible immiscible two-phase VOF; no energy equation",
                     gravity=True, surface_tension=True, wall_adhesion="zeroGradient alpha; no calibrated wetting model"),
        flow_family="intermittently_forced_lox_he_sloshing", preparation_kind="native_two_phase",
        phases=["lox", "he"], species=[], reaction_count=0,
        comparison_fields=["alpha.lox", "U", "p", "p_rgh"], pressure_field="p_rgh", pressure_gauge="reference_fixed",
        pressure_reference=dict(cell=n-1, value=args.pressure, field="p"),
        phase_densities_kg_m3={phase: prop['rho'] for phase, prop in PROPERTIES.items()},
        phase_dynamic_viscosities_Pa_s={phase: prop['mu'] for phase, prop in PROPERTIES.items()},
        surface_tension_N_m=SURFACE_TENSION,
        thermophysical_files=thermo_files, thermophysical_sha256=c.digest_files(root, thermo_files),
        shape=[args.nx, args.ny], end_time=args.end_time, delta_t=args.delta_t, n_steps=steps,
        write_interval=write_interval, geometry_m=[args.width, args.height, args.depth],
        pressure_Pa=args.pressure, property_temperature_K=args.inlet_temperature,
        fill_fraction=args.fill_fraction, initial_liquid_fraction=sum(fields['alpha.lox'])/n,
        initial_liquid_volume_m3=sum(fields['alpha.lox'])/n*args.width*args.height*args.depth,
        physics_checks=dict(alpha_bounds_tolerance=1e-8, liquid_volume_relative_tolerance=1e-6, interface_motion_threshold=1e-8),
        forcing=dict(kind="horizontal translational acceleration in two smooth pulse windows",
                     acceleration_amplitude_m_s2=args.acceleration_amplitude, frequency_Hz=args.frequency,
                     phase=args.phase, pulse_shift=args.pulse_shift,
                     native_model="fv::acceleration differentiates prescribed tank velocity; negative inertial force",
                     velocity_table_samples=len(table), schedule_known_to_controller=False),
        linear_inviscid_slosh_frequency_Hz=omega/(2*math.pi),
        natural_slosh_periods=args.end_time*omega/(2*math.pi),
        initial_cell_order="single Cartesian block: x-fastest, then y; no renumberMesh",
        status="generated_not_executed", foam_conversion_verified=False,
        provenance="native Foundation 13 dictionaries; CoolProp 7.2.0 HEOS fixed mechanical property basis",
        limitations=["Pure He approximates the real He plus oxygen-vapor ullage; no GOX is represented.",
                     "No heat transfer, phase change, pressurant injection or gas compressibility.",
                     "Reference absolute pressure sets property basis/gauge; pressure collapse is not modeled.",
                     "Two-dimensional laminar rectangular tank; no baffles or flight-geometry validation.",
                     "Surface tension uses pure oxygen saturation; helium and wall wetting effects are not calibrated.",
                     "Initial p is derived from p_rgh, density and gravity; the native solver does not read 0/p.",
                     "Actual CFL, conservation, residuals and mesh/time refinement must pass before physical claims.",
                     "Pressure timing fraction and World Model gain must be measured."])
    contract.update(c.condition_hashes(root, ["alpha.lox", "U", "p", "p_rgh"]))
    c.write(root, "contract.json", json.dumps(contract, indent=2)+"\n")
    c.copy_runners(root)
    return root


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--nx", type=int, default=31)
    p.add_argument("--ny", type=int, default=31)
    p.add_argument("--case-id", required=True)
    p.add_argument("--case-group", required=True)
    p.add_argument("--split", choices=("train", "tune", "validation", "test"), required=True)
    p.add_argument("--mode", choices=("native", "classical", "hs"), default="native")
    p.add_argument("--socket-path", default="/tmp/lox-he.sock")
    p.add_argument("--width", type=float, default=.2)
    p.add_argument("--height", type=float, default=.1)
    p.add_argument("--depth", type=float, default=.001)
    p.add_argument("--pressure", type=float, default=300000.)
    p.add_argument("--inlet-temperature", type=float, default=90., help="Fixed property temperature metadata; sealed tank has no inlet")
    p.add_argument("--fill-fraction", type=float, default=.5)
    p.add_argument("--acceleration-amplitude", type=float, default=.5)
    p.add_argument("--frequency", type=float, default=2.)
    p.add_argument("--phase", type=float, default=0.)
    p.add_argument("--pulse-shift", type=float, default=0.)
    p.add_argument("--gravity", type=float, default=9.81)
    p.add_argument("--end-time", type=float, default=4.)
    p.add_argument("--delta-t", type=float, default=.002)
    return p


if __name__ == "__main__":
    print(generate(parser().parse_args()))
