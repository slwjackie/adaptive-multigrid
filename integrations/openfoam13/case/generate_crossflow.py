#!/usr/bin/env python3
"""Pulsed, nonreacting H2/N2 slot jet entering an N2 crossflow (Foundation 13).

The one-block bottom patch uses codedMixed for species/T: fixed injection values
on the slot, zero normal gradients on the impermeable wall. U is exactly zero
outside the slot. This retains the static Cartesian pressure adapter contract.
This 2D laminar slot model is not a round-jet DNS or a hydrogen combustor.

API: https://cpp.openfoam.org/v13/codedMixedFvPatchField_8H_source.html
Property basis: https://coolprop.org/fluid_properties/fluids/Hydrogen.html
Species Cp/mu are rounded constant 300 K benchmark values, not cryogenic fits.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_crossflow_common", Path(__file__).with_name("flow_generator_common.py"))
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)
base = c.base
SPECIES = {"H2": {"molWeight": 2.01588, "Cp": 14300., "mu": 8.9e-6, "Pr": .69},
           "N2": dict(base.SPECIES["N2"])}


def slot_profile(args, x):
    z = (x - args.jet_center) / (.5 * args.jet_width)
    return math.cos(.5 * math.pi * z) ** 2 if abs(z) < 1 else 0.


def jet_state(args, time):
    q = c.pulse_envelope(time, args.end_time, args.pulse_shift) * math.sin(2 * math.pi * args.frequency * time + args.phase)
    return args.jet_speed * (1 + args.forcing_amplitude * q), args.light_mass_fraction + args.composition_amplitude * q


def validate(args):
    c.validate_common(args)
    for key in ("crossflow_speed", "jet_speed", "jet_width", "jet_center", "boundary_layer_thickness",
                "forcing_amplitude", "composition_amplitude", "frequency", "phase", "pulse_shift", "light_mass_fraction"):
        if not math.isfinite(getattr(args, key)):
            raise ValueError("finite " + key + " required")
    if args.crossflow_speed <= 0 or args.jet_speed <= 0 or not 0 <= args.forcing_amplitude < 1:
        raise ValueError("positive speeds and forcing_amplitude in [0,1) required")
    if not 0 <= args.composition_amplitude < args.light_mass_fraction <= 1 - args.composition_amplitude:
        raise ValueError("H2 mass fractions must remain in (0,1]")
    if not 3 * args.width / args.nx <= args.jet_width < .5 * args.width:
        raise ValueError("jet_width must span at least 3 x cells and less than half the domain")
    if not .5 * args.jet_width < args.jet_center < args.width - .5 * args.jet_width:
        raise ValueError("slot must lie strictly inside the bottom boundary")
    if not 2 * args.height / args.ny <= args.boundary_layer_thickness <= .3 * args.height:
        raise ValueError("boundary layer thickness must span >= 2 y cells and <= .3 height")
    if args.frequency <= 0 or args.frequency * args.delta_t > .05 or abs(args.pulse_shift) > .08:
        raise ValueError("resolve positive frequency by >= 20 steps and require |pulse_shift| <= .08")
    speed = max(args.crossflow_speed, args.jet_speed * (1 + args.forcing_amplitude))
    sound = math.sqrt(1.4 * base.R_UNIVERSAL / SPECIES["N2"]["molWeight"] * args.inlet_temperature)
    if speed / sound >= .1:
        raise ValueError("maximum forced reference Mach must remain below .1")
    if args.delta_t * (args.crossflow_speed * args.nx / args.width + speed * args.ny / args.height) > .8:
        raise ValueError("conservative jet Courant estimate exceeds .8; reduce delta_t")


def injector(args, name):
    parameters = c.cpp_pulse(args)
    zcode = f"const scalar z = (centres[i].x()-{args.jet_center:.17g})/{.5 * args.jet_width:.17g};"
    if name == "U":
        body = f"""
        vectorField values(centres.size(), vector::zero);
        forAll(values, i)
        {{
            {zcode}
            if (Foam::mag(z)<1)
                values[i] = vector(0,{args.jet_speed:.17g}*(1+{args.forcing_amplitude:.17g}*q)
                    *Foam::sqr(Foam::cos(1.570796326794897*z)),0);
        }}
        this->operator==(values);
"""
        kind, entries = "codedFixedValue", "value uniform (0 0 0);"
    else:
        h2 = f"({args.light_mass_fraction:.17g}+{args.composition_amplitude:.17g}*q)"
        value = h2 if name == "H2" else f"1-{h2}" if name == "N2" else f"{args.inlet_temperature:.17g}"
        body = f"""
        this->refGrad() = 0;
        this->refValue() = {value};
        forAll(centres, i)
        {{
            {zcode}
            this->valueFraction()[i] = Foam::mag(z)<1 ? 1 : 0;
        }}
"""
        initial = args.light_mass_fraction if name == "H2" else 1 - args.light_mass_fraction if name == "N2" else args.inlet_temperature
        kind = "codedMixed"
        entries = f"refValue uniform {initial:.16g}; refGradient uniform 0; valueFraction uniform 0; value uniform {initial:.16g};"
    identity = hashlib.sha256((parameters + body).encode()).hexdigest()[:16]
    return f"type {kind}; {entries}\nname crossflow{name}_{identity};\ncode\n#{{\n{parameters}{body}\n#}};"


def generate(args):
    validate(args)
    # Reuse the tested multicomponentFluid numerical dictionaries, then replace
    # all gas properties, fields, geometry and their provenance together.
    inherited = type(args)(**vars(args))
    inherited.fuel_speed, inherited.air_speed = args.crossflow_speed, args.crossflow_speed
    root = base.generate(inherited)
    n = args.nx * args.ny
    velocities = [(args.crossflow_speed * (1 - math.exp(-((j+.5)*args.height/args.ny)/args.boundary_layer_thickness)), 0., 0.)
                  for j in range(args.ny) for _ in range(args.nx)]
    inlet_values = "\n".join(f"({velocities[j*args.nx][0]:.16g} 0 0)" for j in range(args.ny))
    inlet_u = f"type fixedValue; value nonuniform List<vector>\n{args.ny}\n(\n{inlet_values}\n);"
    for name, initial in (("H2", 0.), ("N2", 1.), ("T", args.inlet_temperature)):
        dims = "[0 0 0 1 0 0 0]" if name == "T" else "[0 0 0 0 0 0 0]"
        bcs = dict(inlet=f"type fixedValue; value uniform {initial:.16g};",
                   outlet=f"type inletOutlet; inletValue uniform {initial:.16g}; value uniform {initial:.16g};",
                   upper="type zeroGradient;", injectorWall=injector(args, name))
        c.write(root, "0/"+name, c.field(name, dims, [initial]*n, bcs))
    (root/"0/He").unlink()
    c.write(root, "0/U", c.field("U", "[0 1 -1 0 0 0 0]", velocities,
        dict(inlet=inlet_u, outlet="type pressureInletOutletVelocity; value uniform (0 0 0);",
             upper="type slip;", injectorWall=injector(args, "U"))))
    c.write(root, "0/p", c.field("p", "[1 -1 -2 0 0 0 0]", [args.pressure]*n,
        dict(inlet="type zeroGradient;", outlet=f"type fixedValue; value uniform {args.pressure:.16g};",
             upper="type zeroGradient;", injectorWall="type zeroGradient;")))
    c.write(root, "system/blockMeshDict", c.block_mesh(args, [
        ("inlet", "patch", "(0 4 7 3)"), ("outlet", "patch", "(1 2 6 5)"),
        ("upper", "patch", "(7 6 2 3)"), ("injectorWall", "wall", "(0 1 5 4)")]))
    thermo = c.foam_header("thermo.compressibleGas") + "species (H2 N2);\n"
    for name, prop in SPECIES.items():
        thermo += f"""{name}
{{
    specie {{ molWeight {prop['molWeight']:.16g}; }}
    thermodynamics {{ Cp {prop['Cp']:.16g}; hf 0; }}
    transport {{ mu {prop['mu']:.16g}; Pr {prop['Pr']:.16g}; }}
}}
"""
    c.write(root, "constant/thermo.compressibleGas", thermo)
    # Raise native pressure guard to match the budget-comparison experiment.
    for path in ("system/fvSolution", "system/fvSolution.native"):
        c.write(root, path, (root/path).read_text().replace("maxIter 150;", "maxIter 600;"))
    contract = json.loads((root/"contract.json").read_text())
    contract.update(schema="h2-crossflow-case-v1", experiment_kind="h2_crossflow_nonreacting",
        species=["H2", "N2"], comparison_fields=["T", "H2", "N2", "rho", "U", "p"],
        flow_family="pulsed_h2_slot_jet_in_n2_crossflow", pressure_field="p", pressure_gauge="absolute",
        species_properties=SPECIES, coded_inlet=True,
        initial_density_range_kg_m3=[args.pressure/(base.R_UNIVERSAL/SPECIES["N2"]["molWeight"]*args.inlet_temperature)]*2,
        inlet_speeds_m_s={"crossflow": args.crossflow_speed, "jet_centerline": args.jet_speed},
        jet_width_m=args.jet_width, jet_center_m=args.jet_center,
        boundary_layer_thickness_m=args.boundary_layer_thickness,
        patch_meanings={"inlet": "N2 crossflow", "outlet": "fixed-pressure downstream outlet",
                        "upper": "slip truncation", "injectorWall": "no-slip wall outside a smooth H2/N2 slot"},
        forcing=dict(kind="two smooth envelopes of harmonic jet-speed and composition modulation",
                     amplitude=args.forcing_amplitude, composition_amplitude=args.composition_amplitude,
                     frequency_Hz=args.frequency, phase=args.phase, pulse_shift=args.pulse_shift,
                     schedule_known_to_controller=False),
        property_sources=["https://coolprop.org/fluid_properties/fluids/Hydrogen.html",
                          "https://coolprop.org/fluid_properties/fluids/Nitrogen.html"])
    contract["physics"].update(kind="h2_crossflow_nonreacting", flow="nonreacting H2/N2 slot jet in N2 crossflow",
        formulation="compressible pressure-based, low Mach; energy equation active", fuel="H2", oxidizer="none")
    contract["limitations"] += ["No oxygen or reaction: this case validates hydrogen mixing, not combustion.",
        "Two-dimensional laminar slot; three-dimensional round-jet structure is not represented.",
        "Unity-Lewis transport omits hydrogen differential diffusion and Soret transport."]
    contract.update(c.condition_hashes(root, ["H2", "N2", "T", "p", "U"]))
    contract["thermophysical_sha256"] = c.digest_files(root, contract["thermophysical_files"])
    c.write(root, "contract.json", json.dumps(contract, indent=2)+"\n")
    return root


def parser():
    p = base.parser()
    p.description = __doc__
    p.set_defaults(nx=31, ny=31, width=.04, height=.02, end_time=.04, delta_t=1e-5)
    p.add_argument("--crossflow-speed", type=float, default=1.)
    p.add_argument("--jet-speed", type=float, default=8.)
    p.add_argument("--jet-width", type=float, default=.006)
    p.add_argument("--jet-center", type=float, default=.012)
    p.add_argument("--boundary-layer-thickness", type=float, default=.002)
    p.add_argument("--forcing-amplitude", type=float, default=.7)
    p.add_argument("--composition-amplitude", type=float, default=.08)
    p.add_argument("--frequency", type=float, default=100.)
    p.add_argument("--phase", type=float, default=0.)
    p.add_argument("--pulse-shift", type=float, default=0.)
    return p


if __name__ == "__main__":
    print(generate(parser().parse_args()))
