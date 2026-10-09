#!/usr/bin/env python3
"""Prepare and execute a generated OpenFOAM-13 case, recording real status."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def digest_files(root, paths):
    digest = hashlib.sha256()
    for relative in sorted(paths):
        digest.update(relative.encode() + b"\0" + (root / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def times(root):
    answer = []
    for path in root.iterdir():
        if path.is_dir():
            try:
                if float(path.name) > 0:
                    answer.append(path)
            except ValueError:
                pass
    return answer


def command(root, argv, log_name):
    with (root / log_name).open("w") as log:
        result = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}); inspect {root / log_name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default=str(Path(__file__).resolve().parent))
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--native-only", action="store_true", help="PCG preflight without plugin; not an equal-tolerance benchmark")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()
    root = Path(args.case).resolve()
    contract_path = root / "contract.json"
    contract = json.loads(contract_path.read_text())
    if contract.get("schema") != "h2-fixed-p-case-v1":
        raise ValueError("Not a generated H2 case; refusing mutation")
    if args.clean:
        if not args.yes:
            raise ValueError("Cleaning generated outputs requires Allclean --yes")
        targets = [*times(root), root / "constant/polyMesh", root / "postProcessing"]
        for path in targets:
            if path.is_symlink():
                raise ValueError(f"Refusing to clean symlink: {path}")
            if path.exists():
                shutil.rmtree(path)
        for path in [*root.glob("log.*"), root / "cfd-run.json"]:
            if path.is_file():
                path.unlink()
        contract["status"] = "generated_not_executed"
        contract.pop("mesh_sha256", None)
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
        return
    if os.environ.get("WM_PROJECT_VERSION") != "13":
        raise RuntimeError("Source the OpenFOAM Foundation 13 environment first (WM_PROJECT_VERSION=13)")
    required = ["blockMesh", "checkMesh", "chemkinToFoam", "foamRun"]
    for executable in required:
        if not shutil.which(executable):
            raise RuntimeError(f"Missing OpenFOAM command: {executable}")
    if times(root):
        raise RuntimeError("Positive-time output already exists. Copy a fresh case or explicitly use Allclean --yes")
    if digest_files(root, contract["chemistry_files"]) != contract["chemistry_sha256"]:
        raise RuntimeError("Chemistry input content changed after generation")
    fields = [f"0/{name}" for name in [*contract["species"], "T", "p", "U"]]
    if digest_files(root, fields) != contract["initial_conditions_sha256"]:
        raise RuntimeError("Initial fields changed after generation; regenerate the case")
    command(root, ["chemkinToFoam", "-precision", "16", "chemkin/h2.inp", "chemkin/thermo.dat",
                   "chemkin/transport.foam", "constant/reactions", "constant/thermo.compressibleGas"], "log.chemkinToFoam")
    native_chemistry = ["constant/reactions", "constant/thermo.compressibleGas"]
    if any(not (root / path).stat().st_size for path in native_chemistry):
        raise RuntimeError("chemkinToFoam produced empty files")
    contract["foam_conversion_verified"] = True
    contract["converted_chemistry_sha256"] = digest_files(root, native_chemistry)
    command(root, ["blockMesh"], "log.blockMesh")
    command(root, ["checkMesh", "-allGeometry", "-allTopology"], "log.checkMesh")
    check = (root / "log.checkMesh").read_text()
    if "Mesh OK." not in check:
        raise RuntimeError("checkMesh did not report Mesh OK; inspect log.checkMesh")
    mesh_files = [str(p.relative_to(root)) for p in (root / "constant/polyMesh").iterdir() if p.is_file()]
    contract["mesh_sha256"] = digest_files(root, mesh_files)
    contract["status"] = "prepared_not_executed"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    if args.prepare_only:
        print("OpenFOAM chemistry conversion and mesh checks passed; no CFD trajectory executed.")
        return

    solution = root / "system/fvSolution"
    control = root / "system/controlDict"
    original_solution, original_control = solution.read_text(), control.read_text()
    if args.native_only:
        solution.write_text((root / "system/fvSolution.native").read_text())
        control.write_text(original_control.replace('libs ("libAdaptiveFixedP.so");', ""))
    status = {"schema": "h2-cfd-execution-v1", "case_id": contract["case_id"],
              "native_only_preflight": args.native_only,
              "timing_scope": "foamRun process wall time, including CFD setup/output; case preparation excluded",
              "success": False, "physical_validation_performed": False}
    started = time.perf_counter()
    try:
        command(root, ["foamRun"], "log.foamRun")
        log = (root / "log.foamRun").read_text()
        observed_times = [float(v) for v in re.findall(r"^Time = ([0-9.eE+\-]+)\s*$", log, re.M)]
        final_time = max(observed_times, default=0.)
        status["last_observed_time"] = final_time
        if final_time < contract["end_time"] - max(1e-12, 1e-8 * contract["end_time"]):
            raise RuntimeError("foamRun exited before the requested physical end time")
        if "FOAM FATAL" in log:
            raise RuntimeError("OpenFOAM reported a fatal error")
        status["success"] = True
        contract["status"] = "executed_physics_not_yet_validated"
    except BaseException as error:
        status["error"] = str(error)
        contract["status"] = "execution_failed"
        raise
    finally:
        status["cfd_process_wall_seconds"] = time.perf_counter() - started
        if args.native_only:
            solution.write_text(original_solution)
            control.write_text(original_control)
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
        (root / "cfd-run.json").write_text(json.dumps(status, indent=2) + "\n")
    print("CFD reached end time. Inspect cfd-run.json, heat release, continuity and fields before drawing physical conclusions.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
