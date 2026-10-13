"""Small dictionary helpers for the new static Cartesian flow generators."""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import re
import shutil

_spec = importlib.util.spec_from_file_location("_flow_base", Path(__file__).with_name("generate_variable_density.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
foam_header, write, digest_files = base.foam_header, base.write, base.digest_files


def validate_common(args):
    for name in ("nx", "ny"):
        n = getattr(args, name)
        if n < 7 or (n + 1) & n:
            raise ValueError(f"{name} must equal 2**L - 1 and be at least 7")
    if args.nx != args.ny:
        raise ValueError("current pressure adapter requires nx == ny")
    for name in ("width", "height", "depth", "pressure", "inlet_temperature", "end_time", "delta_t"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    steps = args.end_time / args.delta_t
    if args.delta_t > args.end_time or not math.isclose(steps, round(steps), rel_tol=0., abs_tol=1e-8):
        raise ValueError("end_time must be an integer number of positive delta_t steps")
    if not Path(args.socket_path).is_absolute() or any(c in args.socket_path for c in '\n\r"'):
        raise ValueError("socket_path must be absolute without quotes/newlines")
    if len(args.socket_path.encode()) > 100:
        raise ValueError("Unix socket path must be at most 100 bytes")
    for name in ("case_id", "case_group"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", getattr(args, name)):
            raise ValueError(f"invalid {name}")


def field(name, dimensions, values, boundaries):
    vector = name == "U"
    kind = "vector" if vector else "scalar"
    text = "\n".join("(" + " ".join(f"{v:.16g}" for v in row) + ")" if vector else f"{row:.16g}" for row in values)
    patches = "\n".join(f"    {patch} {{ {condition} }}" for patch, condition in boundaries.items())
    return foam_header(name, "volVectorField" if vector else "volScalarField") + f"""dimensions {dimensions};
internalField nonuniform List<{kind}>
{len(values)}
(
{text}
);
boundaryField
{{
{patches}
    frontAndBack {{ type empty; }}
}}
"""


def block_mesh(args, boundaries):
    """y=0 is the bottom; single-block x-fastest ordering is preserved."""
    w, h, d = args.width, args.height, args.depth / 2
    vertices = [(0,0,-d),(w,0,-d),(w,h,-d),(0,h,-d),(0,0,d),(w,0,d),(w,h,d),(0,h,d)]
    text = "\n".join("    (" + " ".join(f"{v:.16g}" for v in row) + ")" for row in vertices)
    patches = "\n".join(f"    {name} {{ type {kind}; faces ({faces}); }}" for name, kind, faces in boundaries)
    return foam_header("blockMeshDict") + f"""convertToMeters 1;
vertices
(
{text}
);
blocks (hex (0 1 2 3 4 5 6 7) ({args.nx} {args.ny} 1) simpleGrading (1 1 1));
boundary
(
{patches}
    frontAndBack {{ type empty; faces ((4 5 6 7) (0 3 2 1)); }}
);
"""


def condition_hashes(root, names):
    files = ["0/" + name for name in names]
    boundary = "\n".join((root / path).read_text().split("boundaryField", 1)[1] for path in sorted(files))
    return dict(initial_fields=list(names), initial_conditions_sha256=digest_files(root, files),
                boundary_conditions_sha256=hashlib.sha256(boundary.encode()).hexdigest())


def copy_runners(root):
    for name in ("run_case.py", "Allrun", "Allclean"):
        shutil.copyfile(Path(__file__).with_name(name), root / name)
        (root / name).chmod(0o755)


def pulse_envelope(time, end_time, shift=0.):
    s = time / end_time
    return sum(math.sin(math.pi * (s - lo) / (hi - lo)) ** 2
               for lo, hi in ((.12 + shift, .42 + shift), (.60 - shift, .90 - shift)) if lo < s < hi)


def cpp_pulse(args):
    return f"""
        const scalar t = this->db().time().value();
        const scalar s = t/{args.end_time:.17g};
        scalar envelope = 0;
        const scalar lo[2] = {{{.12 + args.pulse_shift:.17g}, {.60 - args.pulse_shift:.17g}}};
        const scalar hi[2] = {{{.42 + args.pulse_shift:.17g}, {.90 - args.pulse_shift:.17g}}};
        for (int j=0; j<2; ++j)
            if (s>lo[j] && s<hi[j])
                envelope += Foam::sqr(Foam::sin(3.141592653589793*(s-lo[j])/(hi[j]-lo[j])));
        const scalar q = envelope*Foam::sin(6.283185307179586*{args.frequency:.17g}*t+({args.phase:.17g}));
        const vectorField& centres = this->patch().Cf();
"""
