"""Real in-process OpenFOAM PCG/GAMG timing, independent of Python IPC costs."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import numpy as np

from ..h2_fixed_p.fullrun import _verify_prepared, _rewrite_bridge, _stop, parse_solver_log, collect_fields
from ..world_model.data import file_hash, write_json


def profile(case, output, cfg, settings, *, repeats=3, timeout=7200):
    if os.environ.get("WM_PROJECT_VERSION") != "13" or not shutil.which("foamRun"):
        raise RuntimeError("native profiling requires OpenFOAM Foundation 13; no stand-in is used")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("positive repeats required")
    prepared, out = Path(case).resolve(), Path(output).resolve()
    if out == prepared or out.is_relative_to(prepared):
        raise ValueError("native output must be outside the prepared case")
    contract = _verify_prepared(prepared)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for rep in range(repeats):
        modes = ("native_pcg", "native_gamg") if rep % 2 == 0 else ("native_gamg", "native_pcg")
        for mode in modes:
            dest = out/f"{mode}_{rep}"
            shutil.copytree(prepared, dest)
            _rewrite_bridge(dest, mode, "/tmp/unused-native-profile.sock", cfg)
            process = None
            started = perf_counter()
            try:
                with (dest/"foamRun.log").open("w") as log:
                    process = subprocess.Popen(["foamRun"], cwd=dest, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    process.wait(timeout=timeout)
                    wall = perf_counter()-started
                if process.returncode:
                    raise RuntimeError("native foamRun failed; inspect "+str(dest/"foamRun.log"))
                parsed = parse_solver_log((dest/"foamRun.log").read_text(), contract, settings["continuity_max_cumulative"])
                fields = collect_fields(dest, contract)
                physics = fields.get("physics_evidence") or fields["reaction_evidence"]
                passed = physics.get("passed", physics.get("positive_heat_release_observed", False))
                if len(parsed["pressure_rows"]) != 2*contract["n_steps"]:
                    raise ValueError("native pressure count differs from the declared two correctors per step")
                row = dict(mode=mode, repeat=rep, success=bool(passed and parsed["continuity"]["passed"]),
                    cfd_wall_seconds=wall, pressure_seconds=parsed["pressure_seconds"],
                    pressure_fraction=parsed["pressure_seconds"]/wall,
                    pressure_calls=len(parsed["pressure_rows"]), continuity=parsed["continuity"],
                    physics_evidence=physics, solver_log_sha256=file_hash(dest/"foamRun.log"),
                    fvSolution_sha256=file_hash(dest/"system/fvSolution"), field_files_sha256=fields["hashes"])
            except Exception as exc:
                row = dict(mode=mode, repeat=rep, success=False, error=f"{type(exc).__name__}: {exc}")
            finally:
                _stop(process)
            rows.append(row)
            write_json(dest/"native_profile.json", row)
    report = dict(schema="fixed-p-native-pressure-profile-v1", rows=rows,
        success=all(r["success"] for r in rows), ipc_used=False, case_id=contract["case_id"],
        median_pressure_fraction={mode:float(np.median([r["pressure_fraction"] for r in rows if r["mode"] == mode]))
                                  if all(r["success"] for r in rows if r["mode"] == mode) else None
                                  for mode in ("native_pcg", "native_gamg")},
        speed_claim_eligible=False,
        scope="in-process native solve plus raw-residual checks; no Python service or serialization",
        limitation="Declared native configurations, not globally tuned best solvers. Conservative normalized-L1 stopping may oversolve the common raw-L2 criterion. Field sanity is not paired equivalence.")
    write_json(out/"report.json", report)
    return report
