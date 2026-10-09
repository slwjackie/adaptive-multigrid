"""Execute paired, online OpenFOAM-13 CFD runs on isolated prepared-case copies.

This module never substitutes synthetic fields or replay for CFD. Native tools
and the built bridge must exist. A failed process/solve/physical comparison is
retained in report.json and makes the aggregate speed claim unavailable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from ...provenance import hardware_environment
from ..world_model.data import digest, file_hash, write_json
from .data import read, validate_case_contract
from .evaluation import compare_coupled_runs


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _clean(text):
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", text, flags=re.S))


def read_internal_field(path, n_cells, components=1):
    """Read strict ASCII OpenFOAM scalar/vector internalField, uniform or not."""
    text = _clean(Path(path).read_text())
    if not re.search(r"\bformat\s+ascii\s*;", text):
        raise ValueError("ASCII OpenFOAM fields required: " + str(path))
    if components not in (1, 3) or n_cells < 1:
        raise ValueError("positive cell count and scalar/3-vector required")
    match = re.search(r"\binternalField\s+(uniform|nonuniform)\s+", text)
    if not match:
        raise ValueError("missing internalField: " + str(path))
    tail = text[match.end():]
    if match.group(1) == "uniform":
        value = tail.split(";", 1)[0].strip()
        if components == 3 and not (value.startswith("(") and value.endswith(")")):
            raise ValueError("uniform vector must have three components")
        tokens = value.replace("(", " ").replace(")", " ").split()
        if len(tokens) != components:
            raise ValueError("uniform field component count mismatch")
        values = np.array([float(v) for v in tokens], dtype=float)
        result = np.full(n_cells, values[0]) if components == 1 else np.tile(values, (n_cells, 1))
    else:
        expected = "scalar" if components == 1 else "vector"
        header = re.match(r"List<" + expected + r">\s+(\d+)\s*\(", tail)
        if not header or int(header.group(1)) != n_cells:
            raise ValueError("nonuniform field type/cell count mismatch")
        begin, depth, end = header.end(), 1, None
        for i, char in enumerate(tail[begin:], begin):
            depth += (char == "(") - (char == ")")
            if depth == 0:
                end = i
                break
        if end is None or not re.match(r"\s*;", tail[end + 1:]):
            raise ValueError("unterminated internalField list")
        body = tail[begin:end]
        if components == 3:
            vectors = re.findall(r"\(([^()]*)\)", body)
            if len(vectors) != n_cells or re.sub(r"\([^()]*\)", "", body).strip():
                raise ValueError("malformed vector field list")
            result = np.array([[float(v) for v in row.split()] for row in vectors], dtype=float)
            if result.shape != (n_cells, 3):
                raise ValueError("vector field component count mismatch")
        else:
            result = np.array([float(v) for v in body.split()], dtype=float)
            if result.shape != (n_cells,):
                raise ValueError("scalar field cell count mismatch")
    if not np.isfinite(result).all():
        raise ValueError("nonfinite field: " + str(path))
    return result


def positive_times(root):
    result = []
    for path in Path(root).iterdir():
        if not path.is_dir():
            continue
        try:
            t = float(path.name)
        except ValueError:
            continue
        if np.isfinite(t) and t > 0:
            result.append((t, path))
    result.sort(key=lambda pair: pair[0])
    if len({t for t, _ in result}) != len(result):
        raise ValueError("duplicate numeric output time directories")
    return result


def parse_solver_log(text, contract, continuity_limit):
    """Require the completed fixed-dt trajectory and every pressure witness."""
    if "FOAM FATAL" in text or not re.search(r"^End\s*$", text, re.M):
        raise ValueError("CFD log does not show normal completion")
    times = np.array([float(v) for v in re.findall(r"^Time\s*=\s*(" + NUMBER + r")\s*$", text, re.M)])
    end, dt = float(contract["end_time"]), float(contract["delta_t"])
    count = int(round(end / dt))
    expected = np.arange(1, count + 1) * dt
    if count < 1 or not np.isclose(count * dt, end, rtol=1.e-9, atol=1.e-14):
        raise ValueError("case end_time must be an integer fixed-dt trajectory")
    if len(times) != count or not np.allclose(times, expected, rtol=1.e-8, atol=1.e-13):
        raise ValueError("CFD skipped/changed timesteps or did not reach declared end time")
    pattern = (r"H2FixedPPressureSeconds\s+(" + NUMBER + r")\s+index\s+(\d+)\s+time\s+(" + NUMBER +
               r")\s+mode\s+(classical|hs)\s+residual\s+(" + NUMBER + r")\s+threshold\s+(" + NUMBER + r")")
    rows = [dict(seconds=float(s), index=int(i), time=float(t), mode=m, residual=float(r), threshold=float(th))
            for s, i, t, m, r, th in re.findall(pattern, text)]
    if not rows or len(rows) != text.count("H2FixedPPressureSeconds"):
        raise ValueError("missing or malformed native pressure timing/residual witnesses")
    if [row["index"] for row in rows] != list(range(len(rows))):
        raise ValueError("native pressure indices skipped or restarted")
    if any(not np.isfinite([r["seconds"], r["time"], r["residual"], r["threshold"]]).all()
           or r["seconds"] <= 0 or r["residual"] < 0 or r["residual"] > r["threshold"] for r in rows):
        raise ValueError("native pressure timing or residual check failed")
    pressure_times = np.array([r["time"] for r in rows])
    grid_indices = np.rint(pressure_times / dt).astype(np.int64)
    if (np.any(grid_indices < 1) or np.any(grid_indices > count) or np.any(np.diff(pressure_times) < 0)
            or not np.allclose(pressure_times, grid_indices * dt, rtol=1.e-8, atol=1.e-13)):
        raise ValueError("pressure solve occurred outside declared timesteps")
    if not np.array_equal(np.unique(grid_indices), np.arange(1, count + 1)):
        raise ValueError("a completed CFD timestep has no online pressure evidence")
    continuity = [float(v) for v in re.findall(r"continuity errors\s*:[^\n]*?cumulative\s*=\s*(" + NUMBER + r")", text)]
    if (not np.isfinite(continuity_limit) or continuity_limit < 0 or not continuity
            or not np.isfinite(continuity).all()):
        raise ValueError("finite continuity evidence and predeclared limit required")
    maximum = float(np.max(np.abs(continuity)))
    return dict(times=times.tolist(), dt_schedule=np.diff(np.r_[0., times]).tolist(),
                end_time=float(times[-1]), pressure_rows=rows, pressure_seconds=sum(r["seconds"] for r in rows),
                continuity=dict(passed=maximum <= continuity_limit, maximum_absolute_cumulative=maximum,
                                limit=float(continuity_limit), observations=len(continuity)))


def collect_fields(root, contract):
    samples = positive_times(root)
    if not samples or not np.isclose(samples[-1][0], contract["end_time"], rtol=1.e-8, atol=1.e-13):
        raise ValueError("saved CFD fields do not reach declared end time")
    n = int(np.prod(contract["shape"]))
    values, hashes = {name: [] for name in ("T", "H2", "U", "p")}, {}
    heat_release = []
    for t, folder in samples:
        for name in values:
            path = folder / name
            values[name].append(read_internal_field(path, n, 3 if name == "U" else 1))
            hashes[str(path.relative_to(root))] = file_hash(path)
        qpath = folder / "Qdot"
        qdot = read_internal_field(qpath, n)
        heat_release.append(float(np.max(qdot)))
        hashes[str(qpath.relative_to(root))] = file_hash(qpath)
    return dict(sample_times=[t for t, _ in samples],
                fields={key: np.stack(value) for key, value in values.items()}, hashes=hashes,
                reaction_evidence=dict(finite_saved_Qdot=True, maximum_Qdot_by_time=heat_release,
                                       positive_heat_release_observed=any(v > 0 for v in heat_release),
                                       experimentally_validated=False))


def _digest_files(root, paths):
    h = hashlib.sha256()
    for relative in sorted(paths):
        h.update(relative.encode() + b"\0" + (root / relative).read_bytes() + b"\0")
    return h.hexdigest()


def _verify_prepared(root):
    contract = validate_case_contract(read(root / "contract.json"))
    if contract.get("status") != "prepared_not_executed" or contract.get("foam_conversion_verified") is not True:
        raise ValueError("run generated case Allrun --prepare-only first; prepared_not_executed case required")
    if positive_times(root):
        raise ValueError("prepared case already contains positive-time outputs; use a fresh case")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("prepared case must contain local regular files, not symlinks")
    initial = [f"0/{name}" for name in [*contract["species"], "T", "p", "U"]]
    mesh = [str(p.relative_to(root)) for p in (root / "constant/polyMesh").iterdir() if p.is_file()]
    checks = {"chemistry_sha256": contract["chemistry_files"], "initial_conditions_sha256": initial,
              "converted_chemistry_sha256": ["constant/reactions", "constant/thermo.compressibleGas"],
              "mesh_sha256": mesh}
    for key, files in checks.items():
        if not files or _digest_files(root, files) != contract.get(key):
            raise ValueError("prepared case provenance changed: " + key)
    boundary = "\n".join((root / p).read_text().split("boundaryField", 1)[1] for p in sorted(initial))
    if hashlib.sha256(boundary.encode()).hexdigest() != contract.get("boundary_conditions_sha256"):
        raise ValueError("prepared boundary fields changed")
    control = _clean((root / "system/controlDict").read_text())
    for expression in (r"startFrom\s+startTime\s*;", r"startTime\s+0(?:\.0*)?\s*;",
                       r"adjustTimeStep\s+no\s*;", r"writeFormat\s+ascii\s*;", r"writeCompression\s+off\s*;"):
        if not re.search(expression, control):
            raise ValueError("prepared case must start at zero with fixed dt and ASCII output")
    for key, wanted in (("endTime", contract["end_time"]), ("deltaT", contract["delta_t"])):
        match = re.search(r"\b" + key + r"\s+(" + NUMBER + r")\s*;", control)
        if not match or float(match.group(1)) != wanted:
            raise ValueError("controlDict and contract disagree: " + key)
    steps = int(round(contract["end_time"] / contract["delta_t"]))
    if steps < 1 or not np.isclose(steps * contract["delta_t"], contract["end_time"], rtol=1.e-9, atol=1.e-14):
        raise ValueError("prepared case end time must be an integer number of fixed timesteps")
    interval = re.search(r"\bwriteInterval\s+(\d+)\s*;", control)
    if (not re.search(r"\bwriteControl\s+timeStep\s*;", control) or not interval
            or int(interval.group(1)) < 1 or steps % int(interval.group(1))):
        raise ValueError("prepared case must save the final timestep: writeInterval must divide timestep count")
    return contract


def _rewrite_bridge(case, mode, socket_path, cfg):
    path = case / "system/fvSolution"
    text = path.read_text()
    if "solver AdaptiveFixedP;" not in text:
        raise ValueError("case pressure solver must be AdaptiveFixedP")
    substitutions = {"mode": mode, "socketPath": '"' + str(socket_path) + '"',
                     "adaptiveAtol": repr(cfg.mg.absolute_tolerance), "adaptiveRtol": repr(cfg.mg.tolerance),
                     "maxIter": str(cfg.mg.max_cycles)}
    for key, value in substitutions.items():
        text, count = re.subn(r"\b" + key + r"\s+[^;]+;", key + " " + value + ";", text)
        if count != 1:
            raise ValueError("exactly one bridge control required: " + key)
    path.write_text(text)


def _stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5.)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.)


def _one_run(out, prepared, contract, run_dir, cfg, settings, arm, timeout, hardware):
    case = out / "case"
    shutil.copytree(prepared, case)
    mode = "classical" if arm == "classical" else "hs"
    log_path, service_log, evidence = out / "foamRun.log", out / "service.log", out / "service.jsonl"
    status = dict(schema="h2-fixed-p-coupled-run-v1", arm=arm, source_kind="external_cfd",
                  execution_mode="online_coupled", success=False, case_id=contract["case_id"],
                  mesh_sha256=contract["mesh_sha256"], chemistry_sha256=contract["chemistry_sha256"],
                  initial_conditions_sha256=contract["initial_conditions_sha256"],
                  boundary_conditions_sha256=contract["boundary_conditions_sha256"],
                  fixed_p_sha256=None, dt_schedule=[], end_time=0., hardware=hardware,
                  execution_threads=settings.get("torch_threads", 1), pressure_gauge="absolute",
                  stopping_criteria=dict(rtol=cfg.mg.tolerance, atol=cfg.mg.absolute_tolerance,
                                         max_cycles=cfg.mg.max_cycles, residual_reference="initial_raw_l2"),
                  continuity={"passed": False}, fields={}, evidence={}, timing={})
    process = bridge = None
    foam_wall = startup = 0.
    execution_wall = None
    started = None
    with tempfile.TemporaryDirectory(prefix="h2p-", dir="/tmp") as short:
        socket_path = Path(short) / "bridge.sock"
        _rewrite_bridge(case, mode, socket_path, cfg)
        argv = [sys.executable, "-c", "from adaptive_mg.v67.h2_fixed_p.study import main; main()",
                "serve", "--run-dir", str(run_dir), "--case-contract", str(case / "contract.json"),
                "--socket", str(socket_path), "--mode", mode, "--evidence", str(evidence), "--timeout", str(timeout)]
        env = dict(os.environ)
        source = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
        env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(settings.get("torch_threads", 1))
        try:
            with service_log.open("w") as bridge_output, log_path.open("w") as foam_output:
                started = time.perf_counter()
                bridge = subprocess.Popen(argv, stdout=bridge_output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                # Loading/verifying a large held-out dataset can take longer
                # than a minute. Startup and CFD share the caller's explicit
                # total budget rather than an unrelated fixed startup cap.
                ready_deadline = started + timeout
                while "READY " not in service_log.read_text():
                    if bridge.poll() is not None:
                        raise RuntimeError("pressure service exited before READY; inspect service.log")
                    if time.perf_counter() >= ready_deadline:
                        raise TimeoutError("pressure bridge did not become ready")
                    time.sleep(.05)
                startup = time.perf_counter() - started
                foam_start = time.perf_counter()
                process = subprocess.Popen(["foamRun"], cwd=case, stdout=foam_output, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
                process.wait(timeout=max(.01, timeout - startup))
                foam_wall = time.perf_counter() - foam_start
                if process.returncode:
                    raise RuntimeError(f"foamRun failed ({process.returncode}); inspect foamRun.log")
                bridge.wait(timeout=10.)
                if bridge.returncode:
                    raise RuntimeError("pressure service failed; inspect service.log")
                execution_wall = time.perf_counter() - started
            parsed = parse_solver_log(log_path.read_text(), contract, settings["continuity_max_cumulative"])
            fields = collect_fields(case, contract)
            if not fields["reaction_evidence"]["positive_heat_release_observed"]:
                raise ValueError("no positive saved heat release; cannot validate this as a reacting-flame benchmark")
            if any(not any(np.isclose(t, x, rtol=1.e-8, atol=1.e-13) for x in parsed["times"]) for t in fields["sample_times"]):
                raise ValueError("output field times not present in actual CFD log")
            stream = [json.loads(line) for line in evidence.read_text().splitlines() if line.strip()]
            native = parsed["pressure_rows"]
            if len(stream) != len(native) or not stream:
                raise ValueError("native and service pressure event counts disagree")
            p_hashes = {row.get("p_digest") for row in stream}
            if None in p_hashes or len(p_hashes) != 1:
                raise ValueError("fixed P changed or was absent in coupled run")
            for row, witness in zip(stream, native):
                if (row.get("mode") != mode or witness["mode"] != mode or not row.get("success")
                        or row.get("index") != witness["index"]
                        or not np.isclose(row["time"], witness["time"], rtol=1.e-8, atol=1.e-13)
                        or not np.isfinite(row.get("final_residual", np.nan))
                        or row["final_residual"] > row["threshold"]):
                    raise ValueError("paired native/service pressure witness mismatch or failure")
            uses = sum(row.get("step", {}).get("stats", {}).get("neural_apply_calls", 0) for row in stream)
            status.update(success=True, fixed_p_sha256=next(iter(p_hashes)), dt_schedule=parsed["dt_schedule"],
                          end_time=parsed["end_time"], sample_times=fields["sample_times"], fields=fields["fields"],
                          continuity=parsed["continuity"], reaction_evidence=fields["reaction_evidence"])
            status["evidence"] = dict(coupled_run_attested=True, solver_log_sha256=file_hash(log_path),
                                      field_files_sha256=fields["hashes"], service_log_sha256=file_hash(evidence),
                                      expert_checkpoint_sha256=file_hash(run_dir / "expert/candidate.pt") if arm == "H_S" else None,
                                      neural_apply_calls=int(uses), pressure_calls=len(stream),
                                      producer="this runner launched real foamRun and persistent pressure service")
            status["timing"]["pressure_seconds"] = parsed["pressure_seconds"]
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            _stop(process)
            _stop(bridge)
            status["timing"].update(wall_seconds=execution_wall if execution_wall is not None else time.perf_counter() - started if started else 0.,
                foam_process_wall_seconds=foam_wall, bridge_startup_seconds=startup,
                scope="bridge startup + foamRun + bridge completion; case cloning/preparation and report parsing excluded",
                pressure_scope="sum of native pressure call timers including socket transfers, setup and residual verification")
            if log_path.exists():
                status["evidence"]["solver_log_sha256"] = file_hash(log_path)
            if evidence.exists():
                status["evidence"]["service_log_sha256"] = file_hash(evidence)
    saved = {key: value for key, value in status.items() if key != "fields"}
    write_json(out / "run.json", saved)
    return status


def run_paired(run_dir, case, output, *, repeats=3, timeout=3600.):
    """Run identical prepared CFD cases in independent C/H_S process pairs."""
    from .study import load_run, load_expert, check_freeze

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("positive integer repeats required")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("positive finite timeout required")
    if os.environ.get("WM_PROJECT_VERSION") != "13" or not shutil.which("foamRun"):
        raise RuntimeError("OpenFOAM Foundation 13 is required: source its environment and build integrations/openfoam13/Allwmake; no CFD stand-in is used")
    library_paths = [Path(p) / "libAdaptiveFixedP.so" for p in
                     [os.environ.get("FOAM_USER_LIBBIN", ""), *os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)] if p]
    library = next((p for p in library_paths if p.is_file()), None)
    if library is None:
        raise RuntimeError("libAdaptiveFixedP.so not found; build integrations/openfoam13/Allwmake in OpenFOAM 13")
    run_dir, settings, cfg, manifest = load_run(run_dir)
    load_expert(run_dir, cfg)
    prepared, out = Path(case).resolve(), Path(output).resolve()
    if out == prepared or out.is_relative_to(prepared):
        raise ValueError("paired output must be outside the preserved prepared case")
    if out.exists():
        raise FileExistsError("new coupled output directory required")
    contract = _verify_prepared(prepared)
    if contract["split"] not in ("validation", "test"):
        raise ValueError("coupled evaluation requires an independent validation/test case")
    if contract["split"] == "test":
        check_freeze(run_dir)
    elif (run_dir / "freeze.json").exists():
        raise ValueError("validation development is closed after freeze")
    dataset = read(run_dir / "data/sequence_manifest.json")
    for producer in dataset["producers"]:
        other = read(run_dir / "data" / producer["contract_path"])
        if (other["case_group"] == contract["case_group"] or other["case_id"] == contract["case_id"]):
            if other["split"] != contract["split"]:
                raise ValueError("coupled physical case leaks across data splits")
        same = all(other.get(k) == contract.get(k) for k in
                   ("initial_conditions_sha256", "chemistry_sha256", "mesh_sha256", "end_time", "delta_t"))
        if same and other["split"] != contract["split"]:
            raise ValueError("renamed identical physical case leaks across data splits")
    tolerances = settings.get("physical_tolerances", {})
    if tolerances.get("declared_before_evaluation") is not True or not np.isfinite(settings.get("continuity_max_cumulative", np.nan)):
        raise ValueError("declare physical tolerances and continuity limit before CFD evaluation")
    out.mkdir(parents=True)
    hardware = hardware_environment(refresh=True)
    hardware.update(openfoam_version="13", native_bridge_sha256=file_hash(library))
    write_json(out / "protocol.json", dict(case_contract=contract, tolerances=tolerances,
        continuity_limit=settings["continuity_max_cumulative"], hardware=hardware, repeats=repeats,
        run_manifest_sha256=file_hash(run_dir / "run_manifest.json"), configuration_digest=manifest["configuration_digest"],
        source_digest=manifest["source_digest"], fixed_P=True, world_model_enabled=False))
    rng = np.random.default_rng(settings.get("seed", 20261010))
    first = int(rng.integers(0, 2))
    pairs = []
    for repeat in range(repeats):
        order = ["classical", "H_S"] if (first + repeat) % 2 == 0 else ["H_S", "classical"]
        runs = {}
        for arm in order:
            destination = out / f"repeat_{repeat:03d}" / arm
            destination.mkdir(parents=True)
            try:
                runs[arm] = _one_run(destination, prepared, contract, run_dir, cfg, settings, arm, timeout, hardware)
            except Exception as exc:
                # Preparation/copy/bridge-dictionary errors also remain visible.
                # This row explicitly does not attest that CFD was launched.
                runs[arm] = dict(schema="h2-fixed-p-coupled-run-v1", arm=arm, success=False,
                                 source_kind="external_cfd", execution_mode="not_started",
                                 case_id=contract["case_id"], error=f"{type(exc).__name__}: {exc}",
                                 timing={}, evidence={}, fields={}, continuity={"passed": False})
                write_json(destination / "run.json", {k: v for k, v in runs[arm].items() if k != "fields"})
        comparison = compare_coupled_runs(runs["classical"], runs["H_S"], tolerances)
        comparison.update(repeat=repeat, execution_order=order)
        pairs.append(comparison)
        write_json(out / f"repeat_{repeat:03d}" / "comparison.json", comparison)
    success = all(pair["success"] for pair in pairs)
    wall = {arm: [pair["timing"][arm]["wall_seconds"] for pair in pairs] for arm in ("classical", "H_S")}
    pressure = {arm: [pair["timing"][arm]["pressure_seconds"] for pair in pairs] for arm in ("classical", "H_S")}
    report = dict(schema="h2-fixed-p-paired-cfd-v1", split=contract["split"], case_id=contract["case_id"],
                  repeats=repeats, success=success, speed_claim_eligible=success, pairs=pairs,
                  full_cfd_speedup=float(np.median(wall["classical"]) / np.median(wall["H_S"])) if success else None,
                  pressure_speedup=float(np.median(pressure["classical"]) / np.median(pressure["H_S"])) if success else None,
                  world_model_enabled=False, original_case_modified=False,
                  physical_claim="paired field agreement at saved times, not independent experimental flame validation",
                  provenance="locally launched online CFD runs; native compilation/runtime success must be established on target machine")
    write_json(out / "report.json", report)
    return report
