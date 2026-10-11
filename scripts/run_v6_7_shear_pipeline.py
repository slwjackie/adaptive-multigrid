#!/usr/bin/env python3
"""Modulated inert shear layer -> fixed-P reuse study -> optional H2 validation.

No manufactured CFD fallback. New output directories only. A missing measured
refresh/reuse tradeoff leaves World Model fitting disabled, while retaining all
baseline measurements. The independent test split is evaluated only after
both the base and temporal studies have frozen.
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
sys.path.insert(0, str(REPO/"src"))
_spec = importlib.util.spec_from_file_location("_shear_shared", REPO/"scripts/run_v6_7_h2_pipeline.py")
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)
read, write, run_logged = shared.read, shared.write, shared.run_logged
DRIVER = shared.DRIVER
REUSE_DRIVER = REPO/"scripts/run_v6_7_smoother_reuse.py"
GENERATOR = REPO/"integrations/openfoam13/case/generate_shear_layer.py"

# Distinct inlet conditions at t=0 prevent shared training/test pressure
# operators. Control/forced validation are one physical family in one split.
CASES = (
    dict(name="train_slow", split="train", he=.65, lower=4., upper=1., frequency=90., ratio=.37, phase=.2, shift=0., amplitude=.30, composition=0.),
    dict(name="train_fast", split="train", he=.90, lower=3.8, upper=.9, frequency=140., ratio=.53, phase=1.1, shift=.02, amplitude=.28, composition=0.),
    dict(name="tune_shear", split="tune", he=.75, lower=4.1, upper=1., frequency=110., ratio=.41, phase=.7, shift=-.015, amplitude=.30, composition=0.),
    dict(name="validation_forced", split="validation", he=.82, lower=4., upper=1., frequency=120., ratio=.47, phase=1.8, shift=.03, amplitude=.30, composition=0.),
    dict(name="validation_control", split="validation", he=.82, lower=4., upper=1., frequency=120., ratio=.47, phase=1.8, shift=.03, amplitude=0., composition=0.),
    dict(name="test_frequency", split="test", he=.87, lower=3.9, upper=.9, frequency=175., ratio=.61, phase=2.3, shift=-.04, amplitude=.32, composition=0.),
    dict(name="test_composition", split="test", he=.70, lower=4.15, upper=1.05, frequency=135., ratio=.73, phase=2.8, shift=.045, amplitude=.25, composition=.08),
)
H2_CASES = tuple(dict(name="h2_"+str(t), split=split, temperature=t) for split, t in
                 (("train",300), ("train",400), ("tune",375), ("validation",325), ("test",350)))


def collect(args, root, case, *, hydrogen=False):
    from adaptive_mg.v67.h2_fixed_p.fullrun import collect_fields
    from adaptive_mg.v67.smoother_reuse.physics import shear_metrics
    group = "shear_validation" if case["name"].startswith("validation_") else case["name"]
    options = ()
    if not hydrogen:
        options = ("--light-mass-fraction", str(case["he"]), "--fuel-speed", str(case["lower"]),
                   "--air-speed", str(case["upper"]), "--frequency", str(case["frequency"]),
                   "--frequency-ratio", str(case["ratio"]), "--phase", str(case["phase"]),
                   "--pulse-shift", str(case["shift"]), "--forcing-amplitude", str(case["amplitude"]),
                   "--composition-amplitude", str(case["composition"]))
    recording = shared.collect(args, root, case["name"], case["split"], case.get("temperature",300.),
        generator=None if hydrogen else GENERATOR, case_group=group, generator_options=options)
    prepared, executed = root/"cases"/case["name"], root/"record_cases"/case["name"]
    contract = read(prepared/"contract.json")
    fields = collect_fields(executed, contract)
    evidence = fields["reaction_evidence"] if hydrogen else fields["physics_evidence"]
    profile = dict(case_id=case["name"], sample_times=fields["sample_times"],
                   physics_evidence=evidence, field_files_sha256=fields["hashes"])
    if not hydrogen:
        profile["observables"] = shear_metrics(fields["fields"], contract, fields["sample_times"])
    write(recording/"physics_profile.json", profile)
    if not evidence.get("positive_heat_release_observed" if hydrogen else "passed", False):
        raise RuntimeError("actual CFD field checks failed: "+str(recording/"physics_profile.json"))
    # Compilation is setup, not a different cost for each solver arm. Preserve
    # the generated inlet libraries produced by the successful native run.
    if contract.get("coded_inlet"):
        if not (executed/"dynamicCode").is_dir():
            raise RuntimeError("coded inlet compilation output missing from actual run")
        shutil.copytree(executed/"dynamicCode", prepared/"dynamicCode")
    return recording


def run_study(args, *, hydrogen=False):
    from adaptive_mg.v67.benchmark_profile import profile_dataset
    from adaptive_mg.v67.h2_fixed_p.study import source_digest, configuration
    from adaptive_mg.v67.smoother_reuse.native_profile import profile as native_profile
    from adaptive_mg.v67.world_model.data import file_hash
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    (root/"logs").mkdir()
    cases = H2_CASES if hydrogen else CASES
    stage = "plan"
    def announce(value):
        nonlocal stage
        stage = value
        write(root/"pipeline_status.json", dict(status="running", stage=stage, output=str(root)))
        print("["+stage+"] "+str(root), flush=True)
    def command(temporal, name, *options):
        announce(("reuse " if temporal else "")+name+" "+" ".join(map(str,options)))
        prefix = ("reuse_" if temporal else "")+name
        index = len(list((root/"logs").glob(prefix+"_*.log")))
        run_logged([sys.executable, REUSE_DRIVER if temporal else DRIVER, name,
                    "--run-dir", root/"study", *options], root/"logs"/f"{prefix}_{index}.log",
                   args.timeout*(2*args.repeats+1) if name == "coupled" else args.timeout)
    try:
        settings, cfg = configuration(args.config)
        write(root/"pipeline_plan.json", dict(schema="shear-temporal-pipeline-v1",
            source_digest=source_digest(), runner_sha256=file_hash(Path(__file__)),
            config_sha256=file_hash(args.config), cases=list(cases), n=args.n, end_time=args.end_time,
            delta_t=args.delta_t, stop_after=args.stop_after, repeats=args.repeats,
            experiment_kind="h2_combustion" if hydrogen else "variable_density_nonreacting",
            physics_status="not_executed", world_model_activation="only after measured train/tune tradeoff"))
        recordings = []
        for case in cases:
            announce("collect "+case["name"])
            recordings.append(collect(args, root, case, hydrogen=hydrogen))
        command(False, "prepare", "--config", args.config, "--input-ldu", *recordings)
        write(root/"dataset_profile.json", profile_dataset(root/"study/data", splits=("train","tune","validation")))
        summary = dict(status="complete", completed_through="collect", output=str(root),
            hydrogen_validation_completed=False, world_model_benefit_established=False)
        if args.stop_after == "collect":
            return summary
        command(False, "train")
        command(True, "collect")
        probe = read(root/"study/reuse/collection.json")["probe"]
        summary.update(completed_through="probe", probe=probe)
        if args.stop_after == "probe":
            return summary
        command(True, "train")
        training = read(root/"study/reuse/training.json")
        summary["temporal_training"] = training
        def evaluate(split):
            command(False, "evaluate", "--split", split, "--repeats", str(args.repeats))
            command(True, "evaluate", "--split", split, "--repeats", str(args.repeats))
            reports = {}
            policies = ["refresh", "selected"]
            if training["models"]:
                policies += ["world", "world_horizon1", "world_stateless", "memoryless"]
            for case in cases:
                if case["split"] != split:
                    continue
                announce("native pressure profile "+case["name"])
                native = native_profile(root/"cases"/case["name"], root/("native_"+case["name"]),
                                        cfg, settings, repeats=args.repeats, timeout=args.timeout)
                reports[case["name"]] = {"native":native}
                for policy in policies:
                    output = root/"coupled"/case["name"]/policy
                    command(False, "coupled", "--case", root/"cases"/case["name"], "--output", output,
                            "--repeats", str(args.repeats), "--timeout", str(args.timeout), "--smoother-policy", policy)
                    reports[case["name"]][policy] = read(output/"report.json")
            valid = (read(root/"study"/split/"replay.json")["success"]
                     and read(root/"study/reuse"/split/"report.json")["success"]
                     and all(r["success"] for case in reports.values() for r in case.values()))
            summary[split] = reports
            summary[split+"_passed"] = valid
            if not valid:
                raise RuntimeError(split+" numerical/physical checks failed; evidence retained, no test advance")
        evaluate("validation")
        summary["completed_through"] = "validation"
        summary["hydrogen_validation_completed"] = hydrogen
        if args.stop_after == "test":
            command(False, "freeze")
            command(True, "freeze")
            evaluate("test")
            summary["completed_through"] = "test"
        return summary
    except BaseException as exc:
        if "summary" in locals():
            summary.update(status="failed", failed_stage=stage, error=f"{type(exc).__name__}: {exc}")
        write(root/"pipeline_status.json", dict(status="failed", stage=stage, output=str(root),
              error=f"{type(exc).__name__}: {exc}", results_preserved=True))
        raise
    finally:
        if "summary" in locals():
            write(root/"summary.json", summary)
            if sys.exc_info()[0] is None:
                write(root/"pipeline_status.json", summary)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=REPO/"configs/v6_7_shear_temporal.json")
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--end-time", type=float, default=.04)
    p.add_argument("--delta-t", type=float, default=1e-4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=14400)
    p.add_argument("--stop-after", choices=("collect","probe","validation","test"), default="validation")
    p.add_argument("--with-h2", action="store_true", help="fresh, separate H2 study after the primary stage")
    p.add_argument("--h2-config", type=Path, default=REPO/"configs/v6_7_h2_smoother_reuse.json")
    p.add_argument("--h2-end-time", type=float, default=2e-4)
    p.add_argument("--h2-delta-t", type=float, default=1e-6)
    a = p.parse_args(argv)
    a.output, a.config, a.h2_config = a.output.resolve(), a.config.resolve(), a.h2_config.resolve()
    if a.output.exists():
        p.error("output already exists; use a new directory")
    if a.n < 31 or a.n & (a.n+1) or a.repeats < 1:
        p.error("n=2**L-1 >= 31 and positive repeats required")
    for end, dt in ((a.end_time,a.delta_t),(a.h2_end_time,a.h2_delta_t)):
        if not all(math.isfinite(v) and v>0 for v in (end,dt,a.timeout)):
            p.error("finite positive times required")
        if end/dt < 2 or not math.isclose(end/dt,round(end/dt),rel_tol=0,abs_tol=1e-8):
            p.error("end-time must be an integer multiple of delta-t with >= 2 steps")
    settings = read(a.config)
    if settings.get("benchmark",{}).get("experiment_kind") != "variable_density_nonreacting":
        p.error("primary config must declare the inert experiment kind")
    from adaptive_mg.v67.smoother_reuse.study import settings_from
    settings_from(settings)
    if a.with_h2:
        h2_settings = read(a.h2_config)
        if h2_settings.get("benchmark",{}).get("experiment_kind") != "h2_combustion":
            p.error("H2 validation config must declare h2_combustion")
        settings_from(h2_settings)
        if importlib.util.find_spec("cantera") is None:
            p.error("H2 validation needs Cantera: python -m pip install cantera")
    if os.environ.get("WM_PROJECT_VERSION") != "13":
        p.error("source /opt/openfoam13/etc/bashrc first")
    for executable in ("foamRun","blockMesh","checkMesh") + (("chemkinToFoam",) if a.with_h2 else ()):
        if not shutil.which(executable):
            p.error("missing executable: "+executable)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[key] = str(settings.get("torch_threads",1))
    try:
        summary = run_study(a)
        if a.with_h2:
            import copy
            h2 = copy.copy(a)
            h2.output, h2.config = a.output/"hydrogen_validation", a.h2_config
            h2.end_time, h2.delta_t = a.h2_end_time,a.h2_delta_t
            result = run_study(h2, hydrogen=True)
            summary["hydrogen_validation"] = result
            summary["hydrogen_validation_completed"] = result["hydrogen_validation_completed"]
        write(a.output/"summary.json", summary)
        write(a.output/"pipeline_status.json", summary)
        print(json.dumps({key:summary[key] for key in ("status","completed_through","output",
            "hydrogen_validation_completed","world_model_benefit_established")}, indent=2, allow_nan=False))
        print("Full results: "+str(a.output/"summary.json"), flush=True)
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        if a.output.exists():
            failure = dict(status="failed",
                stage="hydrogen_validation" if (a.output/"hydrogen_validation").exists() else "primary",
                error=f"{type(exc).__name__}: {exc}", results_preserved=True)
            write(a.output/"pipeline_status.json", failure)
            if "summary" in locals():
                summary.update(failure, hydrogen_validation_completed=False)
                write(a.output/"summary.json", summary)
        print(f"STOPPED: {type(exc).__name__}: {exc}\nPreserved output: {a.output}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
