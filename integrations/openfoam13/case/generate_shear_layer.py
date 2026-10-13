#!/usr/bin/env python3
"""Spatial He/N2 coflow shear layer with smooth, deterministic inlet modulation.

Static Cartesian 2D, no periodic/coupled patches, no chemistry. An unforced
inlet control is obtained with --forcing-amplitude 0. A codedFixedValue inlet
uses Foundation 13's documented 'name' and 'code' interface; its source is
embedded in the hashed initial fields. This is a laminar/2D mechanism study,
not a resolved three-dimensional turbulent DNS or a hydrogen surrogate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_inert_case", Path(__file__).with_name("generate_variable_density.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def envelope(time, end, shift=0.):
    """Two C1 pulses separated by genuinely constant *inlet* intervals."""
    s = time / end
    value = 0.
    for lo, hi in ((.12 + shift, .42 + shift), (.60 - shift, .90 - shift)):
        if lo < s < hi:
            value += math.sin(math.pi * (s - lo) / (hi - lo)) ** 2
    return value


def forcing(args, time):
    e = envelope(time, args.end_time, args.pulse_shift)
    wave = (math.sin(2 * math.pi * args.frequency * time + args.phase)
            + .35 * math.sin(2 * math.pi * args.frequency * args.frequency_ratio * time - args.phase)) / 1.35
    return e * wave


def inlet(args, time, y):
    h = .5 * (1 - math.tanh(y / args.shear_thickness))
    q = forcing(args, time)
    delta = (args.fuel_speed - args.air_speed) * (1 + args.forcing_amplitude * q)
    mean = .5 * (args.fuel_speed + args.air_speed)
    u = mean + delta * (h - .5)
    he = (args.light_mass_fraction + args.composition_amplitude * q) * h
    return u, he


def validate(args):
    base.validate(args)
    for key in ("forcing_amplitude", "composition_amplitude", "frequency", "frequency_ratio",
                "phase", "pulse_shift", "shear_thickness", "perturbation"):
        if not math.isfinite(getattr(args, key)):
            raise ValueError("finite " + key + " required")
    if not 0 <= args.forcing_amplitude < 1 or args.composition_amplitude < 0:
        raise ValueError("forcing amplitude in [0,1), composition amplitude >= 0 required")
    if not 0 < args.light_mass_fraction - args.composition_amplitude:
        raise ValueError("minimum light-stream He fraction must be positive")
    if args.light_mass_fraction + args.composition_amplitude > 1:
        raise ValueError("maximum He fraction must not exceed one")
    if args.fuel_speed <= args.air_speed:
        raise ValueError("light lower stream must be faster than the upper stream")
    if args.air_speed - .5 * (args.fuel_speed - args.air_speed) * args.forcing_amplitude <= 0:
        raise ValueError("forcing must preserve positive coflow velocities")
    if args.frequency <= 0 or args.frequency_ratio <= 0 or abs(args.pulse_shift) > .08:
        raise ValueError("positive frequencies and |pulse_shift| <= .08 required")
    if args.frequency * max(1., args.frequency_ratio) * args.delta_t > .05:
        raise ValueError("need at least 20 timesteps per fastest forcing period")
    if not 2 * args.height / args.ny <= args.shear_thickness <= .2 * args.height:
        raise ValueError("shear thickness must span >= 2 y cells and <= 20% domain height")
    if not 0 <= args.perturbation <= .05:
        raise ValueError("perturbation must be a velocity fraction in [0,.05]")
    # The scalar inlet stays exactly complementary even with composition forcing.
    max_u = args.fuel_speed + .5 * (args.fuel_speed - args.air_speed) * args.forcing_amplitude
    sound = math.sqrt(1.4 * base.R_UNIVERSAL / base.SPECIES["N2"]["molWeight"] * args.inlet_temperature)
    if max_u / sound >= .1:
        raise ValueError("maximum forced reference Mach must remain below 0.1")
    if max_u * args.delta_t * args.nx / args.width > .8:
        raise ValueError("initial/forced advective Courant estimate exceeds .8; reduce delta-t")


def code_parameters(args):
    """One scalar C++ expression shared by velocity and both species fields."""
    return f"""
        const scalar t = this->db().time().value();
        const scalar s = t/{args.end_time:.17g};
        scalar e = 0;
        const scalar lo[2] = {{{.12 + args.pulse_shift:.17g}, {.60 - args.pulse_shift:.17g}}};
        const scalar hi[2] = {{{.42 + args.pulse_shift:.17g}, {.90 - args.pulse_shift:.17g}}};
        for (int j=0; j<2; ++j)
            if (s>lo[j] && s<hi[j])
                e += Foam::sqr(Foam::sin(3.141592653589793*(s-lo[j])/(hi[j]-lo[j])));
        const scalar q = e*(Foam::sin(6.283185307179586*{args.frequency:.17g}*t+({args.phase:.17g}))
            + 0.35*Foam::sin(6.283185307179586*{args.frequency * args.frequency_ratio:.17g}*t-({args.phase:.17g})))/1.35;
        const vectorField& c = this->patch().Cf();
"""


def coded_inlet(args, name):
    parameters = code_parameters(args)
    initial = []
    for j in range(args.ny):
        u, he = inlet(args, 0., ((j+.5)/args.ny-.5)*args.height)
        initial.append(f"({u:.16g} 0 0)" if name == "U" else f"{he if name == 'He' else 1-he:.16g}")
    kind = "vector" if name == "U" else "scalar"
    value = f"nonuniform List<{kind}>\n{args.ny}\n(\n"+"\n".join(initial)+"\n)"
    amplitude = args.forcing_amplitude if name == "U" else args.composition_amplitude
    if amplitude == 0:
        return f"type fixedValue; value {value};"
    if name == "U":
        body = f"""
        vectorField values(c.size(), vector::zero);
        forAll(values, i)
        {{
            const scalar h = 0.5*(1-Foam::tanh(c[i].y()/{args.shear_thickness:.17g}));
            const scalar du = {args.fuel_speed - args.air_speed:.17g}*(1+{args.forcing_amplitude:.17g}*q);
            values[i] = vector({.5 * (args.fuel_speed + args.air_speed):.17g}+du*(h-0.5),0,0);
        }}
"""
    else:
        expression = "he" if name == "He" else "1-he"
        body = f"""
        scalarField values(c.size(), 0);
        forAll(values, i)
        {{
            const scalar h = 0.5*(1-Foam::tanh(c[i].y()/{args.shear_thickness:.17g}));
            const scalar he = ({args.light_mass_fraction:.17g}+{args.composition_amplitude:.17g}*q)*h;
            values[i] = {expression};
        }}
"""
    identity = hashlib.sha256((parameters + body).encode()).hexdigest()[:16]
    return f"""type codedFixedValue;
        value {value};
        name shear{name}_{identity};
        code
        #{{
{parameters}{body}
        this->operator==(values);
        #}};
"""


def field(name, dimensions, values, inlet_bc, outlet_bc, lateral_bc):
    vector = name == "U"
    kind = "vector" if vector else "scalar"
    text = "\n".join("(" + " ".join(f"{v:.16g}" for v in row) + ")" if vector else f"{row:.16g}" for row in values)
    return base.foam_header(name, "volVectorField" if vector else "volScalarField") + f"""dimensions {dimensions};
internalField nonuniform List<{kind}>
{len(values)}
(
{text}
);
boundaryField
{{
    inlet {{ {inlet_bc} }}
    outlet {{ {outlet_bc} }}
    lateral {{ {lateral_bc} }}
    frontAndBack {{ type empty; }}
}}
"""


def generate(args):
    validate(args)
    root = base.generate(args)
    values = {name: [] for name in ("He", "N2", "T", "p", "U", "rho")}
    for j in range(args.ny):
        y = ((j + .5) / args.ny - .5) * args.height
        for i in range(args.nx):
            x = (i + .5) / args.nx * args.width
            u, he = inlet(args, 0., y)
            # Analytic streamfunction perturbation. Both velocity components
            # vanish at inlet/outlet and lateral boundaries; divergence = 0.
            k = 2 * math.pi / args.width
            window = math.sin(math.pi * x / args.width) ** 2
            wx = math.pi / args.width * math.sin(2 * math.pi * x / args.width)
            gy = math.cos(math.pi * y / args.height) ** 2
            dgy = -math.pi / args.height * math.sin(2 * math.pi * y / args.height)
            a = args.perturbation * (args.fuel_speed - args.air_speed) * args.shear_thickness
            du = a * window * math.sin(k*x) * dgy
            v = -a * (wx * math.sin(k*x) + window * k * math.cos(k*x)) * gy
            rmix = base.R_UNIVERSAL * (he/base.SPECIES["He"]["molWeight"] + (1-he)/base.SPECIES["N2"]["molWeight"])
            for name, value in (("U", (u+du, v, 0.)), ("He", he), ("N2", 1-he),
                                ("p", args.pressure), ("T", args.inlet_temperature),
                                ("rho", args.pressure/(rmix*args.inlet_temperature))):
                values[name].append(value)
    for name in ("He", "N2"):
        fallback = 0 if name == "He" else 1
        base.write(root, "0/"+name, field(name, "[0 0 0 0 0 0 0]", values[name], coded_inlet(args, name),
                   f"type inletOutlet; inletValue uniform {fallback}; value uniform {fallback};", "type zeroGradient;"))
    base.write(root, "0/U", field("U", "[0 1 -1 0 0 0 0]", values["U"], coded_inlet(args, "U"),
               "type pressureInletOutletVelocity; value uniform (0 0 0);", "type slip;"))
    base.write(root, "0/T", field("T", "[0 0 0 1 0 0 0]", values["T"],
               f"type fixedValue; value uniform {args.inlet_temperature};",
               f"type inletOutlet; inletValue uniform {args.inlet_temperature}; value uniform {args.inlet_temperature};",
               "type zeroGradient;"))
    base.write(root, "0/p", field("p", "[1 -1 -2 0 0 0 0]", values["p"],
               "type zeroGradient;", f"type fixedValue; value uniform {args.pressure};", "type zeroGradient;"))
    mesh = (root/"system/blockMeshDict").read_text()
    mesh = mesh.replace("fuel {", "inlet {").replace("air {", "outlet {").replace(
        "outlet { type patch; faces ((0 1 5 4) (7 6 2 3)); }",
        "lateral { type patch; faces ((0 1 5 4) (7 6 2 3)); }")
    base.write(root, "system/blockMeshDict", mesh)
    contract = json.loads((root/"contract.json").read_text())
    initial = [f"0/{name}" for name in ("He", "N2", "T", "p", "U")]
    boundary = "\n".join((root/p).read_text().split("boundaryField", 1)[1] for p in sorted(initial))
    contract["physics"]["flow"] = "spatially developing He/N2 planar coflow shear layer"
    contract.update(flow_family="planar_shear_mixing_layer",
        coded_inlet=bool(args.forcing_amplitude or args.composition_amplitude),
        benchmark_role="control" if args.forcing_amplitude == args.composition_amplitude == 0 else "primary",
        initial_conditions_sha256=base.digest_files(root, initial),
        boundary_conditions_sha256=hashlib.sha256(boundary.encode()).hexdigest(),
        patch_meanings={"inlet": "lower light fast stream and upper heavy slow stream",
                        "outlet": "downstream fixed pressure", "lateral": "slip far-field truncation"},
        initial_density_range_kg_m3=[min(values["rho"]), max(values["rho"])],
        inlet_speeds_m_s={"lower": args.fuel_speed, "upper": args.air_speed},
        shear_thickness_m=args.shear_thickness,
        convective_transits=args.end_time*.5*(args.fuel_speed+args.air_speed)/args.width,
        forcing=dict(kind="two smooth pulse envelopes with two temporal frequencies",
                     amplitude=args.forcing_amplitude, composition_amplitude=args.composition_amplitude,
                     frequency_Hz=args.frequency, frequency_ratio=args.frequency_ratio,
                     phase=args.phase, pulse_shift=args.pulse_shift,
                     characteristic_strouhal=args.frequency*args.shear_thickness/(args.fuel_speed-args.air_speed),
                     schedule_known_to_controller=False),
        initial_perturbation_velocity_fraction=args.perturbation,
        initial_reference_mach=max(math.hypot(u,v) for u,v,_ in values["U"])/math.sqrt(
            1.4*base.R_UNIVERSAL/base.SPECIES["N2"]["molWeight"]*args.inlet_temperature),
        initial_advective_courant_estimate=args.delta_t*max(abs(u)*args.nx/args.width+abs(v)*args.ny/args.height
                                                         for u,v,_ in values["U"]))
    contract["limitations"] += [
        "2D laminar mechanism study; grid/time convergence and 3D turbulence remain unverified.",
        "A constant inlet interval does not imply a steady flow or pressure operator.",
        "He/N2 is an inert density-contrast benchmark, not hydrogen flame physics.",
        "Longer runs must cover convective residence and forcing times before physical claims."]
    base.write(root, "contract.json", json.dumps(contract, indent=2)+"\n")
    return root


def parser():
    p = base.parser()
    p.description = __doc__
    p.set_defaults(width=.04, height=.02, fuel_speed=4., air_speed=1.,
                   end_time=.04, delta_t=1e-4, nx=31, ny=31)
    p.add_argument("--forcing-amplitude", type=float, default=.30)
    p.add_argument("--composition-amplitude", type=float, default=0.)
    p.add_argument("--frequency", type=float, default=100.)
    p.add_argument("--frequency-ratio", type=float, default=.37)
    p.add_argument("--phase", type=float, default=0.)
    p.add_argument("--pulse-shift", type=float, default=0.)
    p.add_argument("--shear-thickness", type=float, default=.002)
    p.add_argument("--perturbation", type=float, default=.02)
    return p


if __name__ == "__main__":
    print(generate(parser().parse_args()))
