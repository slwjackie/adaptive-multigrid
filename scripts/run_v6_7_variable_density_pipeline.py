#!/usr/bin/env python3
"""Run the primary variable-density, nonreacting OpenFOAM benchmark.

Collect independent He/N2 mixing trajectories, verify their pressure matrices
and physical fields, then train and compare fixed-P classical and H_S solvers.
H2 combustion remains a separate supplementary workflow. No synthetic fallback
is used, and a valid measured slowdown is retained in the final report.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
_spec = importlib.util.spec_from_file_location("fixed_p_pipeline_shared", REPO / "scripts/run_v6_7_h2_pipeline.py")
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)
DRIVER = shared.DRIVER
GENERATOR = REPO / "integrations/openfoam13/case/generate_variable_density.py"
read, write, run_logged, require_success = shared.read, shared.write, shared.run_logged, shared.require_success

# Entire physical trajectories belong to a single split. The held-out cases
# differ in inlet composition; all temperatures stay at 300 K, with no chemistry.
CASES = tuple(dict(name=name, split=split, light_mass_fraction=fraction,
                   inlet_temperature=300., fuel_speed=1., air_speed=.5,
                   case_group=f"he{round(100 * fraction):03d}_u100_050")
              for name, split, fraction in (
                  ("train_he065", "train", .65), ("train_he095", "train", .95),
                  ("validation_he080", "validation", .80), ("test_he090", "test", .90)))


def collect(args, root, case):
    """Record native systems and attest actual nonreacting CFD field outputs."""
    from adaptive_mg.v67.h2_fixed_p.fullrun import collect_fields
    recording = shared.collect(
        args, root, case["name"], case["split"], case["inlet_temperature"],
        generator=GENERATOR, case_group=case["case_group"],
        generator_options=("--light-mass-fraction", str(case["light_mass_fraction"]),
                           "--fuel-speed", str(case["fuel_speed"]),
                           "--air-speed", str(case["air_speed"])))
    contract = read(root / "cases" / case["name"] / "contract.json")
    result = collect_fields(root / "record_cases" / case["name"], contract)
    write(recording / "physics_profile.json", dict(
        experiment_kind="variable_density_nonreacting", case_id=case["name"],
        sample_times=result["sample_times"], field_files_sha256=result["hashes"],
        physics_evidence=result["physics_evidence"],
        scope="native collection field sanity checks; not mesh/time-convergence validation"))
    if result["physics_evidence"].get("passed") is not True:
        raise RuntimeError(f"nonreacting field checks failed; inspect {recording / 'physics_profile.json'}")
    return recording


def pipeline(args):
    from adaptive_mg.v67.benchmark_profile import profile_coupled, profile_dataset
    from adaptive_mg.v67.h2_fixed_p.study import source_digest
    from adaptive_mg.v67.world_model.data import file_hash
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    stage = "plan"
    settings = read(args.config)
    threshold = settings["benchmark"]["pressure_fraction_threshold"]

    def announce(value):
        nonlocal stage
        stage = value
        write(root / "pipeline_status.json", dict(status="running", stage=stage, output=str(root),
              experiment_kind="variable_density_nonreacting", benchmark_role="primary"))
        print(f"\n[{stage}] {root}", flush=True)

    def study(name, *options):
        announce(name + (" " + " ".join(str(v) for v in options) if options else ""))
        run_logged([sys.executable, DRIVER, name, "--run-dir", root / "study", *options],
                   root / "logs" / (name + "_" + str(len(list((root / "logs").glob(name + "_*.log")))) + ".log"),
                   args.timeout * (2 * args.repeats + 1) if name == "coupled" else args.timeout)

    def evaluate_split(split):
        study("evaluate", "--split", split, "--repeats", str(args.repeats))
        replay = require_success(root / "study" / split / "replay.json", eligible=True)
        case = next(case for case in CASES if case["split"] == split)
        output = root / ("coupled_" + split)
        study("coupled", "--case", root / "cases" / case["name"], "--output", output,
              "--repeats", str(args.repeats), "--timeout", str(args.timeout))
        # Keep profiling available even when the physical equivalence gate fails.
        report_path = output / "report.json"
        profile = profile_coupled(report_path, pressure_fraction_threshold=threshold)
        write(output / "benchmark_profile.json", profile)
        coupled = require_success(report_path, eligible=True)
        return {split + "_replay_speedup": replay["speedup"],
                split + "_pressure_speedup": coupled["pressure_speedup"],
                split + "_full_cfd_speedup": coupled["full_cfd_speedup"],
                split + "_benchmark_profile": profile}

    try:
        write(root / "pipeline_plan.json", dict(
            schema="variable-density-pipeline-v1", benchmark_role="primary",
            experiment_kind="variable_density_nonreacting", source_digest=source_digest(),
            runner_sha256=file_hash(Path(__file__)), shared_runner_sha256=file_hash(Path(shared.__file__)),
            generator_sha256=file_hash(GENERATOR), config_sha256=file_hash(args.config),
            config=str(args.config), grid=[args.n, args.n], end_time=args.end_time,
            delta_t=args.delta_t, cases=list(CASES), repeats=args.repeats, stop_after=args.stop_after,
            world_model_enabled=False, pressure_fraction_threshold=threshold,
            purpose="primary fixed-P pressure study; profile the bottleneck before a World Model study",
            physical_validation="field sanity checks and paired equivalence; mesh/time convergence remains separate"))
        recordings = []
        for case in CASES:
            announce("collect " + case["name"])
            recordings.append(collect(args, root, case))
        study("prepare", "--config", args.config, "--input-ldu", *recordings)
        announce("profile train and validation")
        profile = profile_dataset(root / "study/data", splits=("train", "validation"))
        write(root / "dataset_profile.json", profile)
        summary = dict(status="complete", completed_through="collect", output=str(root),
                       benchmark_role="primary", experiment_kind="variable_density_nonreacting",
                       world_model_enabled=False, dataset_profile=profile,
                       physics_profiles=[str(path / "physics_profile.json") for path in recordings])
        if args.stop_after != "collect":
            study("train")
            trained = read(root / "study/expert/status.json")
            if trained.get("status") != "complete" or trained.get("optimizer_updates", 0) < 1:
                raise RuntimeError("training is incomplete; inspect study/expert/status.json")
            summary.update(optimizer_updates=trained["optimizer_updates"],
                           completed_through="validation", **evaluate_split("validation"))
            if args.stop_after == "test":
                study("freeze")
                summary.update(completed_through="test", **evaluate_split("test"))
        write(root / "summary.json", summary)
        write(root / "pipeline_status.json", summary)
        print("\nCOMPLETED\n" + json.dumps(summary, indent=2), flush=True)
        return summary
    except BaseException as exc:
        write(root / "pipeline_status.json", dict(status="failed", stage=stage,
              experiment_kind="variable_density_nonreacting", benchmark_role="primary",
              error=f"{type(exc).__name__}: {exc}", output=str(root), results_preserved=True))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new directory; never overwritten")
    parser.add_argument("--config", type=Path, default=REPO / "configs/v6_7_variable_density_fixed_p.json")
    parser.add_argument("--n", type=int, default=31, help="31 is an initial smoke grid; increase for timing studies")
    parser.add_argument("--end-time", type=float, default=.002)
    parser.add_argument("--delta-t", type=float, default=1e-5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=7200, help="seconds per stage, or per arm in paired CFD")
    parser.add_argument("--stop-after", choices=("collect", "validation", "test"), default="test")
    args = parser.parse_args(argv)
    if args.n < 7 or args.n & (args.n + 1) or args.repeats < 1:
        parser.error("n must be 2**L-1 >= 7; repeats must be positive")
    if not all(math.isfinite(v) and v > 0 for v in (args.end_time, args.delta_t, args.timeout)):
        parser.error("time values must be finite and positive")
    steps = args.end_time / args.delta_t
    if not math.isfinite(steps) or steps < 2 or not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-8):
        parser.error("end-time must be an integer multiple of delta-t with at least two timesteps for density evolution checks")
    args.output, args.config = args.output.resolve(), args.config.resolve()
    if args.output.exists():
        parser.error("output already exists; preserve it and use a new output path")
    if not args.config.is_file():
        parser.error("missing input: " + str(args.config))
    settings = read(args.config)
    benchmark = settings.get("benchmark", {})
    if benchmark.get("experiment_kind") != "variable_density_nonreacting" or benchmark.get("role") != "primary":
        parser.error("config must declare the primary variable_density_nonreacting benchmark")
    threshold = benchmark.get("pressure_fraction_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        parser.error("pressure_fraction_threshold must lie strictly between zero and one")
    if os.environ.get("WM_PROJECT_VERSION") != "13":
        parser.error("source /opt/openfoam13/etc/bashrc first")
    for command in ("foamRun", "blockMesh", "checkMesh"):
        if not shutil.which(command):
            parser.error("missing executable: " + command)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(settings.get("torch_threads", 1))
    try:
        return pipeline(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"STOPPED: {type(exc).__name__}: {exc}\nPreserved output: {args.output}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
