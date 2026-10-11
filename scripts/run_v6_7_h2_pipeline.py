#!/usr/bin/env python3
"""Continue a successful OpenFOAM preflight through collection, training and test.

Run from the repository root in the OpenFOAM-13 Python environment. Each output
directory is a new experiment; existing recordings and prepared cases survive
failures. This runner never substitutes synthetic data for an OpenFOAM run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
DRIVER = REPO / "scripts/run_v6_7_h2_fixed_p.py"
CASES = (("train_300", "train", 300), ("train_400", "train", 400),
         ("validation_325", "validation", 325), ("test_350", "test", 350))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stop(process):
    """Terminate only a child process group created by this runner."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    except ProcessLookupError:
        process.wait(timeout=5)


def run_logged(argv, log, timeout):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        with log.open("w") as output:
            process = subprocess.Popen([str(v) for v in argv], cwd=REPO,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"command exited {code}; inspect {log}")
    finally:
        stop(process)


def require_success(report, *, eligible=False):
    value = read(report)
    if value.get("success") is not True or (eligible and value.get("speed_claim_eligible") is not True):
        raise RuntimeError(f"evaluation did not pass; inspect {report}")
    # A slowdown is still a valid measured result. Never tune to or discard it.
    return value


def check_native_recording(recording, evidence, case):
    """Check every pressure event, not just the final few success lines.

The supplied generator uses nCorrectors=2 and one outer corrector, so a complete
recording has two ordered pressure systems per fixed timestep. Import below
separately checks the actual arrays, file hashes, native Amul witnesses and SPD.
"""
    contract = read(Path(recording) / "case_contract.json")
    raw = read(Path(recording) / "ldu_sequence.json")
    trajectories = raw.get("trajectories", [])
    if len(trajectories) != 1:
        raise ValueError("one physical trajectory per recording required")
    entries = trajectories[0]["snapshots"]
    dt, end = contract["delta_t"], contract["end_time"]
    if not all(math.isfinite(v) and v > 0 for v in (dt, end)):
        raise ValueError("invalid recorded timestep or end time")
    steps = round(end / dt)
    if steps < 1 or not math.isclose(steps * dt, end, rel_tol=1e-9, abs_tol=1e-14):
        raise ValueError("recording does not declare integer fixed timesteps")
    rows = [json.loads(line) for line in Path(evidence).read_text().splitlines() if line.strip()]
    if len(rows) != 2 * steps or len(entries) != len(rows):
        raise ValueError(f"incomplete pressure recording: events={len(rows)}, snapshots={len(entries)}, expected={2 * steps}")
    for index, (row, entry) in enumerate(zip(rows, entries)):
        expected_time = (index // 2 + 1) * dt
        if (row.get("success") is not True or row.get("error") != "" or row.get("mode") != "native"
                or row.get("index") != index or entry.get("index") != index
                or not re.fullmatch(r"[0-9a-f]{64}", row.get("matrix_digest", ""))):
            raise ValueError(f"failed/missing native pressure event {index}")
        residual, threshold = row["final_residual"], row["threshold"]
        if not (math.isfinite(residual) and math.isfinite(threshold) and 0 <= residual <= threshold and threshold > 0):
            raise ValueError(f"native pressure event {index} failed residual tolerance")
        if not all(math.isclose(value, expected_time, rel_tol=1e-8, abs_tol=1e-13)
                   for value in (row["time"], entry["time"])):
            raise ValueError(f"missing or changed pressure timestep {index}")
    # Reuse the fixed Foundation-13 time parser (including the seconds suffix).
    spec = importlib.util.spec_from_file_location("h2_case_runner", REPO / "integrations/openfoam13/case/run_case.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = module.completed_time((Path(case) / "log.foamRun").read_text())
    if not math.isclose(observed, end, rel_tol=1e-8, abs_tol=1e-13):
        raise ValueError("CFD did not reach the recorded contract's end time")
    return dict(case_id=contract["case_id"], snapshots=len(entries), shape=contract["shape"], end_time=end)


def verify_preflight(args, root):
    from adaptive_mg.v67.h2_fixed_p.data import import_recordings, verify_dataset
    from adaptive_mg.v67.world_model.data import load_snapshot
    result = check_native_recording(args.preflight_recording, args.preflight_evidence, args.preflight_case)
    target = root / "preflight_verified"
    manifest = import_recordings([args.preflight_recording], target)
    verify_dataset(target)
    entry = manifest["trajectories"][0]["snapshots"][0]
    first = load_snapshot(target / entry["path"])
    result.update(matrix_shape=list(first.a.shape), b_shape=list(first.b.shape),
                  x0_shape=list(first.x0.shape), success=True)
    write(root / "preflight_report.json", result)
    print("PREFLIGHT PASS " + json.dumps(result), flush=True)


def collect(args, root, name, split, temperature, *, generator=None,
            case_group=None, generator_options=()):
    """Collect a generated real CFD trajectory through the shared native bridge.

    The defaults preserve the supplementary H2 workflow. Other case generators
    may share this process lifecycle while declaring their own physical inputs.
    """
    prepared = root / "cases" / name
    executed = root / "record_cases" / name
    recording = root / "recordings" / name
    evidence = root / "logs" / (name + "_service.jsonl")
    service_log = root / "logs" / (name + "_service.log")
    bridge = process = None
    with tempfile.TemporaryDirectory(prefix="h2c-", dir="/tmp") as short:
        socket_path = Path(short) / "bridge.sock"
        run_logged([sys.executable, generator or REPO / "integrations/openfoam13/case/generate.py",
                    "--output", prepared, "--case-id", name, "--case-group", case_group or f"inlet_{temperature}",
                    "--split", split, "--inlet-temperature", str(temperature),
                    "--nx", str(args.n), "--ny", str(args.n), "--end-time", str(args.end_time),
                    "--delta-t", str(args.delta_t), "--mode", "native", "--socket-path", socket_path,
                    *generator_options],
                   root / "logs" / (name + "_generate.log"), args.timeout)
        run_logged([sys.executable, REPO / "integrations/openfoam13/case/run_case.py",
                    "--case", prepared, "--prepare-only"],
                   root / "logs" / (name + "_prepare.log"), args.timeout)
        shutil.copytree(prepared, executed)
        try:
            with service_log.open("w") as service_output, (executed / "log.foamRun").open("w") as foam_output:
                started = time.monotonic()
                bridge = subprocess.Popen([sys.executable, str(DRIVER), "serve", "--config", str(args.config),
                    "--case-contract", str(prepared / "contract.json"), "--mode", "native",
                    "--socket", str(socket_path), "--record-dir", str(recording),
                    "--evidence", str(evidence), "--timeout", str(args.timeout)],
                    cwd=REPO, stdout=service_output, stderr=subprocess.STDOUT, start_new_session=True)
                while "READY " not in service_log.read_text():
                    if bridge.poll() is not None:
                        raise RuntimeError(f"service exited before READY; inspect {service_log}")
                    if time.monotonic() - started >= args.timeout:
                        raise TimeoutError(f"service startup timed out; inspect {service_log}")
                    # Do not probe-connect: the service accepts exactly one producer.
                    time.sleep(.05)
                foam_started = time.monotonic()
                process = subprocess.Popen(["foamRun", "-case", str(executed)], cwd=REPO,
                                           stdout=foam_output, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                process.wait(timeout=max(.01, args.timeout - (time.monotonic() - started)))
                foam_seconds = time.monotonic() - foam_started
                if process.returncode:
                    raise RuntimeError(f"foamRun exited {process.returncode}; inspect {executed / 'log.foamRun'} and {service_log}")
                bridge.wait(timeout=10)
                if bridge.returncode:
                    raise RuntimeError(f"recording service failed; inspect {service_log}")
            result = check_native_recording(recording, evidence, executed)
            result.update(success=True, cfd_process_wall_seconds=foam_seconds,
                          scope="native recording run, including recording overhead; not MG benchmark timing")
            write(executed / "collection_status.json", result)
            print("COLLECT PASS " + json.dumps(result), flush=True)
        finally:
            stop(process)
            stop(bridge)
    return recording


def pipeline(args):
    from adaptive_mg.v67.h2_fixed_p.study import source_digest
    from adaptive_mg.v67.world_model.data import file_hash
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    stage = "preflight"

    def announce(value):
        nonlocal stage
        stage = value
        write(root / "pipeline_status.json", dict(status="running", stage=stage, output=str(root)))
        print(f"\n[{stage}] {root}", flush=True)

    def study(name, *options):
        announce(name + (" " + " ".join(str(v) for v in options) if options else ""))
        run_logged([sys.executable, DRIVER, name, "--run-dir", root / "study", *options],
                   root / "logs" / (name + "_" + str(len(list((root / "logs").glob(name + "_*.log")))) + ".log"),
                   args.timeout * (2 * args.repeats + 1) if name == "coupled" else args.timeout)

    try:
        write(root / "pipeline_plan.json", dict(schema="h2-pipeline-v1", source_digest=source_digest(),
              runner_sha256=file_hash(Path(__file__)), config_sha256=file_hash(args.config),
              config=str(args.config), grid=[args.n, args.n], end_time=args.end_time, delta_t=args.delta_t,
              cases=[dict(name=n, split=s, inlet_temperature=t) for n, s, t in CASES],
              repeats=args.repeats, stop_after=args.stop_after,
              purpose="initial complete solver study; mesh/time convergence and flame validation remain separate"))
        announce("preflight")
        verify_preflight(args, root)
        recordings = []
        for name, split, temperature in CASES:
            announce("collect " + name)
            recordings.append(collect(args, root, name, split, temperature))
        study("prepare", "--config", args.config, "--input-ldu", *recordings)
        study("train")
        trained = read(root / "study/expert/status.json")
        if trained.get("status") != "complete" or trained.get("optimizer_updates", 0) < 1:
            raise RuntimeError("training is incomplete; inspect study/expert/status.json")
        study("evaluate", "--split", "validation", "--repeats", str(args.repeats))
        validation = require_success(root / "study/validation/replay.json", eligible=True)
        study("coupled", "--case", root / "cases/validation_325", "--output", root / "coupled_validation",
              "--repeats", str(args.repeats), "--timeout", str(args.timeout))
        validation_cfd = require_success(root / "coupled_validation/report.json", eligible=True)
        summary = dict(status="complete", completed_through="validation", output=str(root),
                       optimizer_updates=trained["optimizer_updates"],
                       validation_replay_speedup=validation["speedup"],
                       validation_pressure_speedup=validation_cfd["pressure_speedup"],
                       validation_full_cfd_speedup=validation_cfd["full_cfd_speedup"])
        if args.stop_after == "test":
            study("freeze")
            study("evaluate", "--split", "test", "--repeats", str(args.repeats))
            test = require_success(root / "study/test/replay.json", eligible=True)
            study("coupled", "--case", root / "cases/test_350", "--output", root / "coupled_test",
                  "--repeats", str(args.repeats), "--timeout", str(args.timeout))
            test_cfd = require_success(root / "coupled_test/report.json", eligible=True)
            summary.update(completed_through="test", test_replay_speedup=test["speedup"],
                           test_pressure_speedup=test_cfd["pressure_speedup"],
                           test_full_cfd_speedup=test_cfd["full_cfd_speedup"])
        write(root / "summary.json", summary)
        write(root / "pipeline_status.json", summary)
        print("\nCOMPLETED\n" + json.dumps(summary, indent=2), flush=True)
        return summary
    except BaseException as exc:
        write(root / "pipeline_status.json", dict(status="failed", stage=stage,
              error=f"{type(exc).__name__}: {exc}", output=str(root), results_preserved=True))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new directory; never overwritten")
    parser.add_argument("--config", type=Path, default=Path("configs/v6_7_h2_fixed_p.json"))
    parser.add_argument("--preflight-recording", type=Path, default=Path("artifacts/h2_record_preflight"))
    parser.add_argument("--preflight-evidence", type=Path, default=Path("artifacts/h2_record_preflight_service.jsonl"))
    parser.add_argument("--preflight-case", type=Path, default=Path("artifacts/h2_bridge_preflight"))
    parser.add_argument("--n", type=int, default=31)
    parser.add_argument("--end-time", type=float, default=2e-4)
    parser.add_argument("--delta-t", type=float, default=1e-6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=7200, help="seconds per stage, or per arm in coupled CFD")
    parser.add_argument("--stop-after", choices=("validation", "test"), default="test")
    args = parser.parse_args(argv)
    if args.n < 7 or args.n & (args.n + 1) or args.repeats < 1:
        parser.error("n must be 2**L-1 >= 7; repeats must be positive")
    if not all(math.isfinite(v) and v > 0 for v in (args.end_time, args.delta_t, args.timeout)):
        parser.error("time values must be finite and positive")
    steps = args.end_time / args.delta_t
    if steps < 1 or not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-8):
        parser.error("end-time must be an integer multiple of delta-t")
    for key in ("output", "config", "preflight_recording", "preflight_evidence", "preflight_case"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.exists():
        parser.error("output already exists; preserve it and use a new output path")
    if os.environ.get("WM_PROJECT_VERSION") != "13":
        parser.error("source /opt/openfoam13/etc/bashrc first")
    for command in ("foamRun", "blockMesh", "checkMesh", "chemkinToFoam"):
        if not shutil.which(command):
            parser.error("missing executable: " + command)
    for path in (args.config, args.preflight_recording / "ldu_sequence.json",
                 args.preflight_recording / "case_contract.json", args.preflight_evidence,
                 args.preflight_case / "log.foamRun"):
        if not path.is_file():
            parser.error("missing input: " + str(path))
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(read(args.config).get("torch_threads", 1))
    try:
        return pipeline(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"STOPPED: {type(exc).__name__}: {exc}\nPreserved output: {args.output}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
