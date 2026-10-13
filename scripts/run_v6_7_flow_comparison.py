#!/usr/bin/env python3
"""Real-CFD comparison of composition forcing, LOX/He slosh and H2 crossflow.

Plan/generate work without OpenFOAM. Later stages require Foundation 13 and the
native bridge. --resume continues immutable completed stages; it never erases
failed evidence. A short probe is a screen, not a physics or speed claim.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


shared = module("_flow_shared", REPO / "scripts/run_v6_7_h2_pipeline.py")
shear = module("_flow_shear", REPO / "scripts/run_v6_7_shear_pipeline.py")
DRIVER = shared.DRIVER
REUSE_DRIVER = REPO / "scripts/run_v6_7_smoother_reuse.py"
STAGES = ("plan", "generate", "collect", "probe", "validation", "test")
FAMILIES = ("shear_composition", "lox_he_sloshing", "h2_crossflow", "shear_control")
WORLD_KEYS = ("hidden", "world_memory", "world_head", "world_head_hidden")
H2_FAMILY = "hydrogen_reacting"
read, run_logged = shared.read, shared.run_logged


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shared.write(path, value)


def cases_for(family):
    """Distinct initial conditions separate physical train/tune/test groups."""
    if family.startswith("shear_"):
        cases = deepcopy(list(shear.CASES))
        if family == "shear_composition":
            for case, composition in zip(cases, (.10, .07, .09, .11, 0., .06, .13)):
                case["composition"] = composition
            velocity_only = deepcopy(cases[3])
            velocity_only.update(name="validation_velocity_only", composition=0.)
            cases.append(velocity_only)
        for case in cases:
            case["group"] = "shear_validation" if case["split"] == "validation" else case["name"]
        return cases
    if family == "h2_crossflow":
        values = (("train_slow", "train", .55, 6., 70., .2),
                  ("train_fast", "train", .85, 9., 130., 1.1),
                  ("tune_jet", "tune", .70, 7.5, 100., .7),
                  ("validation_jet", "validation", .78, 8., 115., 1.8),
                  ("test_frequency", "test", .65, 8.5, 160., 2.3),
                  ("test_composition", "test", .90, 7., 125., 2.8))
        return [dict(name=n, group=n, split=s, fraction=h, jet_speed=u, frequency=f, phase=p,
                     crossflow_speed=cross, composition=.08 if s != "test" else .05)
                for (n, s, h, u, f, p), cross in zip(values, (.8, 1.2, 1., 1.1, .9, 1.3))]
    if family == "lox_he_sloshing":
        values = (("train_low", "train", .40, 1.7, .2, .35),
                  ("train_high", "train", .58, 2.1, 1.1, .55),
                  ("tune_slosh", "tune", .49, 1.9, .7, .45),
                  ("validation_slosh", "validation", .54, 2.0, 1.8, .50),
                  ("test_frequency", "test", .45, 2.3, 2.3, .60),
                  ("test_fill", "test", .64, 1.8, 2.8, .40))
        return [dict(name=n, group=n, split=s, fill=h, frequency=f, phase=p, acceleration=a)
                for n, s, h, f, p, a in values]
    raise ValueError("unknown flow family: " + family)


def generator_for(family):
    if family == H2_FAMILY:
        return REPO / "integrations/openfoam13/case/generate.py"
    name = "generate_shear_layer.py" if family.startswith("shear_") else (
        "generate_crossflow.py" if family == "h2_crossflow" else "generate_sloshing.py")
    return REPO / "integrations/openfoam13/case" / name


def generator_options(family, case):
    if family == H2_FAMILY:
        return ["--inlet-temperature", str(case["temperature"])]
    if family.startswith("shear_"):
        mapping = dict(light_mass_fraction="he", fuel_speed="lower", air_speed="upper",
                       frequency="frequency", frequency_ratio="ratio", phase="phase",
                       pulse_shift="shift", forcing_amplitude="amplitude", composition_amplitude="composition")
    elif family == "h2_crossflow":
        mapping = dict(light_mass_fraction="fraction", jet_speed="jet_speed", crossflow_speed="crossflow_speed", frequency="frequency",
                       phase="phase", composition_amplitude="composition")
    else:
        mapping = dict(fill_fraction="fill", frequency="frequency", phase="phase", acceleration_amplitude="acceleration")
    return [v for key, name in mapping.items() for v in ("--"+key.replace("_", "-"), str(case[name]))]


def family_settings(base, family, arm, profile):
    result = deepcopy(base)
    result["solver"]["mg"].update(max_cycles=600, nn_levels=len(arm["levels"]))
    result["solver"]["smoother_levels"] = arm["levels"]
    result["training"].update(smoother_architecture=arm["architecture"], updates=profile["training_updates"])
    result["smoother_reuse"]["epochs"] = profile["world_epochs"]
    kind = {"h2_crossflow": "h2_crossflow_nonreacting", "lox_he_sloshing": "lox_he_sloshing"}.get(
        family, "variable_density_nonreacting")
    result["benchmark"].update(experiment_kind=kind, flow_family=family)
    fields = result["physical_tolerances"]["fields"]
    if family == "h2_crossflow":
        fields["H2"] = fields.pop("He")
    if family == "lox_he_sloshing":
        result["physical_tolerances"]["pressure_alignment"] = "none"
        result["physical_tolerances"]["fields"] = {
            "alpha.lox": dict(atol=1e-5, rtol=1e-3),
            "U": dict(atol=1e-4, rtol=1e-3), "p_rgh": dict(atol=.1, rtol=1e-5),
            "p": dict(atol=.1, rtol=1e-5)}
    return result


def build_plan(args, config):
    from adaptive_mg.v67.h2_fixed_p.study import source_digest
    from adaptive_mg.v67.world_model.data import file_hash
    if config.get("schema") != "three-flow-comparison-v1":
        raise ValueError("three-flow-comparison-v1 configuration required")
    profile_name = args.profile or "quick"
    profile = deepcopy(config["profiles"][profile_name])
    n = profile["n"] if args.n is None else args.n
    if n < 31 or n & (n+1):
        raise ValueError("n=2**L-1 >=31 required")
    families = list(args.families or config["default_families"])
    if args.include_legacy_control and "shear_control" not in families:
        families.append("shear_control")
    if len(set(families)) != len(families):
        raise ValueError("duplicate families")
    arms = [dict(id="shared_l0", architecture="shared", levels=[0]),
            dict(id="shared_l01", architecture="shared", levels=[0, 1]),
            dict(id="per_level_l01", architecture="per_level", levels=[0, 1])]
    variants = [dict(id=f"shared_h{hidden}_{head}", hidden=hidden, world_memory="shared",
                     world_head=head, world_head_hidden=config["world_head_hidden"])
                for hidden in config["world_hidden_sizes"] for head in config["world_heads"]]
    if args.independent_grus:
        variants += [dict(id=f"independent_h{hidden}_role_mlp", hidden=hidden,
                         world_memory="independent_roles", world_head="role_mlp",
                         world_head_hidden=config["world_head_hidden"])
                     for hidden in config["world_hidden_sizes"]]
    prepared = {}
    for family in families:
        prepared[family] = dict(cases=cases_for(family), timing=profile["families"][family],
            configs={arm["id"]: family_settings(config["base"], family, arm, profile) for arm in arms})
        for settings in prepared[family]["configs"].values():
            settings["smoother_reuse"]["baseline_repeats"] = args.repeats or 3
            settings["smoother_reuse"]["measurement_repeats"] = args.repeats or 3
    hydrogen = dict(enabled=bool(args.with_h2))
    if args.with_h2:
        source_family = args.h2_source_family or "h2_crossflow"
        if source_family not in families:
            raise ValueError("H2 source family must be included in --families")
        end = 2e-4 if args.h2_end_time is None else args.h2_end_time
        dt = 1e-6 if args.h2_delta_t is None else args.h2_delta_t
        if (not all(math.isfinite(v) and v > 0 for v in (end, dt)) or end/dt < 2
                or not math.isclose(end/dt, round(end/dt), rel_tol=0, abs_tol=1e-8)):
            raise ValueError("H2 times must be positive, with an integer number of >=2 steps")
        path = REPO/"configs/v6_7_h2_smoother_reuse.json"
        hydrogen.update(source_family=source_family, end_time=end, delta_t=dt,
            base_config=read(path), base_config_path=str(path), base_config_sha256=file_hash(path),
            cases=[dict(case, group=case["name"]) for case in shear.H2_CASES],
            transfer_mode="architecture_retrained_on_h2", weights_transferred=False,
            selection="predeclared source family; its tune-selected structure, validation gated",
            test_selection_forbidden=True)
    elif any(getattr(args, key) is not None for key in ("h2_source_family", "h2_end_time", "h2_delta_t")):
        raise ValueError("H2 options require --with-h2 in the initial plan")
    return dict(schema=config["schema"], source_digest=source_digest(), runner_sha256=file_hash(Path(__file__)),
        input_config_sha256=file_hash(args.config), input_config=str(args.config),
        profile=profile_name, n=n, repeats=args.repeats or 3, timeout=args.timeout or 14400,
        families=prepared, smoother_arms=arms, world_variants=variants, hydrogen=hydrogen,
        full_factorial=bool(args.full_factorial), independent_grus=bool(args.independent_grus),
        include_legacy_control=bool(args.include_legacy_control), selection=config["selection"], notes=config["notes"],
        test_selection_forbidden=True, synthetic_fallback=False,
        default_world_sweep="tune-selected two-level CNN; all CNNs only with --full-factorial")


def check_plan(root, args):
    from adaptive_mg.v67.h2_fixed_p.study import source_digest
    from adaptive_mg.v67.world_model.data import file_hash
    plan = read(root/"plan.json")
    if plan["source_digest"] != source_digest() or plan["runner_sha256"] != file_hash(Path(__file__)):
        raise ValueError("source changed since plan; use a new output directory")
    if plan["input_config_sha256"] != file_hash(Path(plan["input_config"])):
        raise ValueError("comparison configuration changed since plan")
    for key in ("n", "profile", "repeats", "timeout"):
        value = getattr(args, key)
        if value is not None and value != plan[key]:
            raise ValueError("--resume cannot change " + key)
    if args.families and list(args.families) != list(plan["families"]):
        raise ValueError("--resume cannot change families")
    if args.full_factorial and not plan["full_factorial"]:
        raise ValueError("--resume cannot expand architecture selection")
    if args.independent_grus and not plan["independent_grus"]:
        raise ValueError("--resume cannot add independent GRUs")
    if args.include_legacy_control and not plan["include_legacy_control"]:
        raise ValueError("--resume cannot add a legacy control")
    hydrogen = plan.get("hydrogen", {"enabled": False})
    if args.with_h2 and not hydrogen["enabled"]:
        raise ValueError("--resume cannot add H2 validation; declare --with-h2 in a new plan")
    for key in ("source_family", "end_time", "delta_t"):
        value = getattr(args, "h2_"+key)
        if value is not None and value != hydrogen.get(key):
            raise ValueError("--resume cannot change H2 " + key)
    if hydrogen["enabled"] and file_hash(Path(hydrogen["base_config_path"])) != hydrogen["base_config_sha256"]:
        raise ValueError("H2 base configuration changed")
    for family, values in plan["families"].items():
        for arm, settings in values["configs"].items():
            if read(root/"configurations"/(family+"_"+arm+".json")) != settings:
                raise ValueError("saved experiment configuration changed")
    return plan


def preflight(plan):
    if os.environ.get("WM_PROJECT_VERSION") != "13":
        raise RuntimeError("source /opt/openfoam13/etc/bashrc first; no synthetic CFD fallback")
    for executable in ("foamRun", "blockMesh", "checkMesh"):
        if not shutil.which(executable):
            raise RuntimeError("missing OpenFOAM executable: " + executable)
    folders = [os.environ.get("FOAM_USER_LIBBIN", ""), *os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)]
    if not any((Path(folder)/"libAdaptiveFixedP.so").is_file() for folder in folders if folder):
        raise RuntimeError("build integrations/openfoam13/Allwmake first; libAdaptiveFixedP.so is missing")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"


def generate_case(root, family, case, plan):
    from adaptive_mg.v67.h2_fixed_p.data import validate_case_contract
    target = root/"families"/family/"cases"/case["name"]
    if target.exists():
        validate_case_contract(read(target/"contract.json"))
        return target
    timing = plan["families"][family]["timing"]
    run_logged([sys.executable, generator_for(family), "--output", target, "--case-id", case["name"],
        "--case-group", case["group"], "--split", case["split"], "--nx", str(plan["n"]), "--ny", str(plan["n"]),
        "--end-time", str(timing["end_time"]), "--delta-t", str(timing["delta_t"]),
        "--mode", "native", "--socket-path", "/tmp/flow-comparison-placeholder.sock",
        *generator_options(family, case)], target.parent.parent/"logs"/(case["name"]+"_generate.log"), plan["timeout"])
    validate_case_contract(read(target/"contract.json"))
    return target


def collection_physics(fields, family):
    if family == H2_FAMILY:
        evidence = deepcopy(fields.get("reaction_evidence", {}))
        evidence["passed"] = (evidence.get("positive_heat_release_observed") is True
                              and evidence.get("finite_saved_Qdot") is True)
        return evidence
    return fields.get("physics_evidence", {})


def collect_case(root, family, case, plan):
    """Record once, retaining real field evidence and compiled boundary code."""
    from adaptive_mg.v67.h2_fixed_p.study import configuration
    from adaptive_mg.v67.h2_fixed_p.fullrun import _rewrite_bridge, _verify_prepared, collect_fields
    from adaptive_mg.v67.smoother_reuse.physics import flow_metrics
    base = root/"families"/family
    prepared = generate_case(root, family, case, plan)
    executed, recording = base/"record_cases"/case["name"], base/"recordings"/case["name"]
    evidence = base/"logs"/(case["name"]+"_service.jsonl")
    profile = recording/"physics_profile.json"
    if profile.exists():
        shared.check_native_recording(recording, evidence, executed)
        if read(profile).get("physics_evidence", {}).get("passed") is not True:
            raise ValueError("saved collection physics evidence failed")
        _verify_prepared(prepared)
        fields = collect_fields(executed, read(prepared/"contract.json"))
        saved = read(profile)
        if (fields["hashes"] != saved["field_files_sha256"] or
                fields["sample_times"] != saved["sample_times"] or
                collection_physics(fields, family).get("passed") is not True):
            raise ValueError("saved native CFD field evidence changed")
        return recording
    if executed.exists() or recording.exists():
        raise FileExistsError("incomplete native collection preserved: " + str(executed) + "; use a new output")
    if read(prepared/"contract.json").get("status") != "prepared_not_executed":
        run_logged([sys.executable, REPO/"integrations/openfoam13/case/run_case.py", "--case", prepared,
                    "--prepare-only"], base/"logs"/(case["name"]+"_prepare.log"), plan["timeout"])
    contract = _verify_prepared(prepared)
    recording_arm = plan["families"][family].get("recording_arm", "shared_l01")
    cfg_path = root/"configurations"/(family+"_"+recording_arm+".json")
    _, cfg = configuration(cfg_path)
    shutil.copytree(prepared, executed)
    bridge = process = None
    service_log = base/"logs"/(case["name"]+"_service.log")
    with tempfile.TemporaryDirectory(prefix="amg-fc-", dir="/tmp") as short:
        socket = Path(short)/"bridge.sock"
        _rewrite_bridge(executed, "native", socket, cfg)
        try:
            with service_log.open("w") as service_output, (executed/"log.foamRun").open("w") as foam_output:
                started = time.monotonic()
                bridge = subprocess.Popen([sys.executable, str(DRIVER), "serve", "--config", str(cfg_path),
                    "--case-contract", str(prepared/"contract.json"), "--mode", "native", "--socket", str(socket),
                    "--record-dir", str(recording), "--evidence", str(evidence), "--timeout", str(plan["timeout"])],
                    cwd=REPO, stdout=service_output, stderr=subprocess.STDOUT, start_new_session=True)
                while "READY " not in service_log.read_text():
                    if bridge.poll() is not None:
                        raise RuntimeError("recording bridge failed: " + str(service_log))
                    if time.monotonic()-started > plan["timeout"]:
                        raise TimeoutError("recording bridge startup timed out")
                    time.sleep(.05)
                foam_started = time.monotonic()
                process = subprocess.Popen(["foamRun", "-case", str(executed)], cwd=REPO, stdout=foam_output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                process.wait(timeout=max(.01, plan["timeout"]-(time.monotonic()-started)))
                if process.returncode:
                    raise RuntimeError("foamRun failed: " + str(executed/"log.foamRun"))
                seconds = time.monotonic()-foam_started
                bridge.wait(timeout=10)
                if bridge.returncode:
                    raise RuntimeError("recording service failed: " + str(service_log))
            result = shared.check_native_recording(recording, evidence, executed)
            fields = collect_fields(executed, contract)
            physics = collection_physics(fields, family)
            if physics.get("passed") is not True:
                raise RuntimeError("native field checks failed: " + json.dumps(physics))
            if (executed/"dynamicCode").is_dir() and not (prepared/"dynamicCode").exists():
                shutil.copytree(executed/"dynamicCode", prepared/"dynamicCode")
            write(profile, dict(case_id=case["name"], physics_evidence=physics,
                sample_times=fields["sample_times"], field_files_sha256=fields["hashes"],
                observables={} if family == H2_FAMILY else
                    flow_metrics(fields["fields"], contract, fields["sample_times"])))
            write(executed/"collection_status.json", dict(result, success=True, cfd_process_wall_seconds=seconds,
                scope="native recording with I/O overhead; not solver benchmark timing"))
        finally:
            shared.stop(process)
            shared.stop(bridge)
    return recording


def command(plan, out, temporal, name, *options):
    log_dir = out.parent/"logs"
    prefix = out.name+("_reuse_" if temporal else "_")+name
    count = len(list(log_dir.glob(prefix+"_*.log"))) if log_dir.exists() else 0
    run_logged([sys.executable, REUSE_DRIVER if temporal else DRIVER, name, "--run-dir", out, *options],
               log_dir/f"{prefix}_{count}.log", plan["timeout"]*(2*plan["repeats"]+1) if name == "coupled" else plan["timeout"])


def prepare_arm(root, family, arm, plan, recordings):
    out = root/"families"/family/"studies"/arm["id"]
    if not (out/"run_manifest.json").exists():
        command(plan, out, False, "prepare", "--config", root/"configurations"/(family+"_"+arm["id"]+".json"),
                "--input-ldu", *recordings)
    if not (out/"expert/status.json").exists():
        command(plan, out, False, "train")
    if not (out/"reuse/collection.json").exists():
        command(plan, out, True, "collect")
    verify_probe(out)
    return out


def verify_probe(out):
    from adaptive_mg.v67.smoother_reuse import study as reuse
    from adaptive_mg.v67.world_model.data import file_hash
    _, _, _, _, contract = reuse.load(out)
    collection = read(out/"reuse/collection.json")
    if collection["contract"] != contract:
        raise ValueError("saved probe collection contract changed")
    for path, sha in collection["episodes"].items():
        if file_hash(out/"reuse"/path) != sha:
            raise ValueError("saved probe episode changed")
    pins = {str(path): file_hash(out/path) for path in (
        "configuration.json", "run_manifest.json", "data/sequence_manifest.json", "expert/candidate.pt", "reuse/collection.json")}
    target = out/"comparison_probe_pin.json"
    if target.exists() and read(target) != pins:
        raise ValueError("completed probe evidence changed")
    write(target, pins)


def _link_or_copy(source, target):
    try:
        os.link(source, target)
        return target
    except OSError:
        return shutil.copy2(source, target)


def clone_world_study(parent, target, variant):
    """Branch only the WM architecture; retain and hash actual parent evidence.

    Expert weights, solver configuration, recorded systems, sampled behaviors
    and labels are byte-identical. Changed contracts describe this derivative,
    never claim that its labels were independently measured again.
    """
    from adaptive_mg.v67.h2_fixed_p import study as base
    from adaptive_mg.v67.smoother_reuse import study as reuse
    from adaptive_mg.v67.world_model.data import digest, file_hash
    _, settings, _, manifest = base.load_run(parent)
    _, _, _, _, old_contract = reuse.load(parent)
    collection = read(parent/"reuse/collection.json")
    if collection["contract"] != old_contract:
        raise ValueError("parent collection contract changed")
    candidate = deepcopy(settings)
    for key in WORLD_KEYS:
        candidate["smoother_reuse"][key] = variant[key]
    before, after = deepcopy(settings), deepcopy(candidate)
    for key in WORLD_KEYS:
        before["smoother_reuse"].pop(key, None)
        after["smoother_reuse"].pop(key, None)
    if before != after:
        raise ValueError("only the four declared world architecture settings may change")
    parent_pins = dict(parent=str(parent.resolve()), configuration_sha256=file_hash(parent/"configuration.json"),
        run_manifest_sha256=file_hash(parent/"run_manifest.json"), collection_sha256=file_hash(parent/"reuse/collection.json"),
        checkpoint_sha256=file_hash(parent/"expert/candidate.pt"),
        parent_training_sha256=file_hash(parent/"reuse/training.json"), episodes=collection["episodes"],
        changed_settings={key: dict(before=settings["smoother_reuse"].get(key), after=variant[key]) for key in WORLD_KEYS})
    if target.exists():
        if read(target/"world_parent.json") != parent_pins or read(target/"configuration.json") != candidate:
            raise ValueError("world branch parent/configuration changed")
        reuse.load(target)
        return target
    target.mkdir(parents=True)
    shutil.copytree(parent/"data", target/"data", copy_function=_link_or_copy)
    for name in ("expert", "reuse/episodes"):
        shutil.copytree(parent/name, target/name)
    write(target/"configuration.json", candidate)
    manifest = deepcopy(manifest)
    manifest["configuration_digest"] = digest(candidate)
    write(target/"run_manifest.json", manifest)
    new_contract = dict(old_contract, configuration_digest=digest(candidate),
                        settings_digest=digest(candidate["smoother_reuse"]))
    collection["contract"] = new_contract
    collection["inherited_measurements"] = dict(parent=str(parent.resolve()),
        collection_sha256=parent_pins["collection_sha256"], byte_identical_episodes=True,
        immutable_data_reuse="hardlinks where supported, otherwise byte copies")
    for name, sha in collection["episodes"].items():
        if file_hash(target/"reuse"/name) != sha:
            raise ValueError("copied counterfactual episode differs")
    write(target/"reuse/collection.json", collection)
    write(target/"world_parent.json", parent_pins)
    reuse.load(target)
    return target


def fit_and_score(out, plan):
    """Tune only: report total online wall time, including predictor construction."""
    from adaptive_mg.v67.smoother_reuse import study as reuse
    from adaptive_mg.v67.world_model.data import load_trajectories, file_hash
    import numpy as np
    if not (out/"reuse/training.json").exists():
        command(plan, out, True, "train")
    _, w, cfg, expert, contract = reuse.load(out)
    training = reuse.trained(out, contract)
    target = out/"comparison_tune.json"
    if target.exists():
        value = read(target)
        if value["training_sha256"] != file_hash(out/"reuse/training.json"):
            raise ValueError("tune evidence has changed training")
        for name, sha in value["trace_sha256"].items():
            if file_hash(out/name) != sha:
                raise ValueError("tune timing trace changed")
        return value
    scores = deepcopy(training["baseline_scores"])
    names = ["classical"] + (["world", "world_horizon1", "world_stateless", "memoryless"] if training["models"] else [])
    artifacts = {name: reuse.load_artifact(out, name, training) for name in names if name != "classical"}
    rng = np.random.default_rng(w["seed"]+751)
    for name in names:
        scores[name] = []
    for meta, states in load_trajectories(out/"data", "tune"):
        trials = {name: [] for name in names}
        for rep in range(plan["repeats"]):
            for name in rng.permutation(names):
                result = reuse.run_episode(states, cfg, expert, str(name), artifact=artifacts.get(name),
                    reference=training["selected"], minimum_gain=w["minimum_gain"])
                trials[name].append(result)
                write(out/"tune_traces"/f'{meta["id"]}_{name}_{rep}.json', result)
        for name, runs in trials.items():
            scores[name].append(dict(trajectory=meta["id"], success=all(r["success"] for r in runs),
                seconds=float(np.median([r["total_seconds"] for r in runs]))))
    metrics = {name: dict(success=bool(rows) and all(row["success"] for row in rows),
                         seconds=sum(row["seconds"] for row in rows)) for name, rows in scores.items()}
    result = dict(split="tune", metrics=metrics, scores=scores, selected_reference=training["selected"],
        training_sha256=file_hash(out/"reuse/training.json"), models_available=bool(training["models"]),
        model_details=training.get("model_details", {}), measured_policy_traces="tune_traces",
        trace_sha256={str(path.relative_to(out)):file_hash(path) for path in sorted((out/"tune_traces").glob("*.json"))},
        time_scope="sum of per-trajectory median solver construction + setup + solve + policy + verification + recovery",
        test_read=False, validation_read=False, offline_training_included=False)
    write(target, result)
    return result


def choose_successful(candidates, policy):
    eligible = [(key, value["metrics"][policy]["seconds"]) for key, value in candidates.items()
                if policy in value["metrics"] and value["metrics"][policy]["success"]]
    if not eligible:
        raise RuntimeError("no successful tune candidate for policy " + policy)
    return min(eligible, key=lambda item: item[1])[0]


def evaluate_study(root, family, out, split, plan, *, coupled=True):
    from adaptive_mg.v67.h2_fixed_p.study import configuration
    from adaptive_mg.v67.h2_fixed_p import study as base_study
    from adaptive_mg.v67.smoother_reuse import study as reuse_study
    from adaptive_mg.v67.smoother_reuse.native_profile import profile as native_profile
    from adaptive_mg.v67.world_model.data import file_hash
    _, _, _, _, contract = reuse_study.load(out)
    training = reuse_study.trained(out, contract)
    if split == "test":
        base_study.check_freeze(out)
        reuse_study.check_freeze(out)
    if not (out/split/"replay.json").exists():
        command(plan, out, False, "evaluate", "--split", split, "--repeats", str(plan["repeats"]))
    if not (out/"reuse"/split/"report.json").exists():
        command(plan, out, True, "evaluate", "--split", split, "--repeats", str(plan["repeats"]))
    reports = dict(base=read(out/split/"replay.json"), temporal=read(out/"reuse"/split/"report.json"), coupled={})
    if (reports["temporal"]["contract"] != contract or
            reports["temporal"]["training_sha256"] != file_hash(out/"reuse/training.json")):
        raise ValueError("cached temporal evaluation provenance changed")
    evidence = reports["base"].get("evidence", {})
    if (evidence.get("checkpoint_sha256") != file_hash(out/"expert/candidate.pt") or
            any(evidence.get(key) != contract[key] for key in
                ("configuration_digest", "source_digest", "dataset_sha256"))):
        raise ValueError("cached smoother evaluation provenance changed")
    policies = ["refresh", training["selected"]] + (["world"] if training["models"] else [])
    if plan["full_factorial"]:
        policies = list(read(out/"configuration.json")["smoother_reuse"]["policies"])
        if training["models"]:
            policies += ["world", "world_horizon1", "world_stateless", "memoryless"]
    if coupled:
        settings, cfg = configuration(out/"configuration.json")
        for case in plan["families"][family]["cases"]:
            if case["split"] != split:
                continue
            prepared = root/"families"/family/"cases"/case["name"]
            native = root/"families"/family/"native_profiles"/case["name"]
            if not (native/"report.json").exists():
                result = native_profile(prepared, native, cfg, settings, repeats=plan["repeats"], timeout=plan["timeout"])
                write(native/"report.json", result)
            native_report = read(native/"report.json")
            reports["coupled"][case["name"]] = dict(native=native_report)
            for policy in dict.fromkeys(policies):
                output = out/"coupled"/case["name"]/policy
                if not (output/"report.json").exists():
                    command(plan, out, False, "coupled", "--case", prepared, "--output", output,
                            "--repeats", str(plan["repeats"]), "--timeout", str(plan["timeout"]), "--smoother-policy", policy)
                reports["coupled"][case["name"]][policy] = read(output/"report.json")
    passed = reports["base"]["success"] and reports["temporal"]["success"]
    passed = passed and all(r["success"] for case in reports["coupled"].values() for r in case.values())
    reports.update(success=bool(passed), split=split, no_selection_on_test=True,
                   full_cfd_wall_clock_measured=coupled)
    comparison_path = out/("comparison_"+split+".json")
    if comparison_path.exists() and read(comparison_path) != reports:
        raise ValueError("completed comparative evaluation changed")
    write(comparison_path, reports)
    if not passed:
        raise RuntimeError("numerical/physical " + split + " failure; evidence retained: " + str(out))
    return reports


def comparative_metrics(results, selection):
    """Keep level/architecture effects distinct from added temporal control."""
    metrics = {name: value["temporal"]["metrics"] for name, value in results.items()}
    def ratio(numerator, denominator):
        return numerator/denominator if denominator > 0 else None
    smooth = {name: dict(classical_seconds=row["classical"]["median_seconds"],
        refresh_seconds=row["refresh"]["median_seconds"],
        speedup_vs_classical=row["refresh"].get("speedup_vs_classical"))
        for name, row in metrics.items() if name in ("shared_l0", "shared_l01", "per_level_l01")}
    levels = ratio(smooth["shared_l0"]["refresh_seconds"], smooth["shared_l01"]["refresh_seconds"])
    independent = ratio(smooth["shared_l01"]["refresh_seconds"], smooth["per_level_l01"]["refresh_seconds"])
    chosen = selection.get("world_study")
    world = None
    if chosen and "world" in metrics[chosen]:
        ref = results[chosen]["temporal"]["selected_reference"]
        world = dict(study=chosen, frozen_reference=ref,
            world_seconds=metrics[chosen]["world"]["median_seconds"],
            reference_seconds=metrics[chosen][ref]["median_seconds"],
            speedup_vs_tuned_heuristic=metrics[chosen]["world"]["speedup_vs_tuned_reuse"])
    return dict(smoother_only=smooth, multilevel_refresh_speedup=levels,
        independent_vs_shared_two_level_refresh_speedup=independent, world_additional=world,
        world_model_benefit_established=False, test_ranking_performed=False,
        note="Ratios are measured outputs, not proof of significance; coupled CFD reports are separate.")


def advance_family(root, family, plan, stop):
    base = root/"families"/family
    cases = plan["families"][family]["cases"]
    for case in cases:
        generate_case(root, family, case, plan)
    summary = dict(completed_through="generate", world_model_benefit_established=False)
    if stop == "generate":
        return summary
    recordings = [collect_case(root, family, case, plan) for case in cases]
    summary["completed_through"] = "collect"
    if stop == "collect":
        return summary
    studies = {arm["id"]: prepare_arm(root, family, arm, plan, recordings) for arm in plan["smoother_arms"]}
    summary.update(completed_through="probe", probes={name: read(out/"reuse/collection.json")["probe"] for name, out in studies.items()})
    write(base/"probe_summary.json", summary)
    if stop == "probe":
        return summary
    tune = {name: fit_and_score(out, plan) for name, out in studies.items()}
    two_level = {name: tune[name] for name in ("shared_l01", "per_level_l01")}
    cnn = choose_successful(two_level, "refresh")
    world_studies, world_scores = {}, {}
    for arm_name in (list(two_level) if plan["full_factorial"] else [cnn]):
        if not tune[arm_name]["models_available"]:
            continue
        for variant in plan["world_variants"]:
            key = arm_name+"__"+variant["id"]
            out = clone_world_study(studies[arm_name], base/"world_studies"/key, variant)
            world_studies[key] = out
            world_scores[key] = fit_and_score(out, plan)
    world = choose_successful(world_scores, "world") if world_scores else None
    selection = dict(selected_two_level_cnn=cnn, smoother_selection_policy="refresh",
        world_study=world, world_reference=world_scores[world]["selected_reference"] if world else tune[cnn]["selected_reference"],
        selection_split="tune", validation_role="required numerical/physical gate",
        smoother_scores=tune, world_scores=world_scores, test_read=False)
    selection_path = base/"selection.json"
    if selection_path.exists() and read(selection_path) != selection:
        raise ValueError("frozen tune selection changed")
    write(selection_path, selection)
    finalists = dict(studies)
    if plan["full_factorial"]:
        finalists.update(world_studies)
    elif world:
        finalists[world] = world_studies[world]
    validation = {name: evaluate_study(root, family, out, "validation", plan) for name, out in finalists.items()}
    summary.update(completed_through="validation", selection=selection,
        validation=comparative_metrics(validation, selection))
    if stop == "validation":
        return summary
    from adaptive_mg.v67.world_model.data import file_hash
    pins = {"selection_sha256": file_hash(selection_path), "finalists": list(finalists),
            "plan_sha256": file_hash(root/"plan.json"), "selection_split": "tune"}
    freeze = base/"comparison_freeze.json"
    if freeze.exists() and read(freeze) != pins:
        raise ValueError("comparison selection changed after freeze")
    for out in finalists.values():
        if not (out/"freeze.json").exists():
            command(plan, out, False, "freeze")
        if not (out/"reuse/freeze.json").exists():
            command(plan, out, True, "freeze")
    write(freeze, pins)
    test = {name: evaluate_study(root, family, out, "test", plan) for name, out in finalists.items()}
    summary.update(completed_through="test", test=comparative_metrics(test, selection),
                   test_selection_performed=False)
    return summary


def hydrogen_settings(source, plan, *, world_selected):
    """Transfer a method, not inert weights or inert physical tolerances."""
    settings = deepcopy(plan["hydrogen"]["base_config"])
    settings["solver"] = deepcopy(source["solver"])
    settings["solver"]["mg"]["max_cycles"] = 600
    settings["training"] = deepcopy(source["training"])
    settings["smoother_reuse"] = deepcopy(source["smoother_reuse"])
    settings["smoother_reuse"]["fit_world"] = bool(world_selected)
    settings["benchmark"].update(role="supplementary", experiment_kind="h2_combustion",
                                 flow_family="counterflow_hydrogen_validation")
    return settings


def pin_hydrogen_transfer(root, plan):
    from adaptive_mg.v67.smoother_reuse import study as reuse
    from adaptive_mg.v67.world_model.data import file_hash
    family = plan["hydrogen"]["source_family"]
    base = root/"families"/family
    selection = read(base/"selection.json")
    if selection.get("selection_split") != "tune" or selection.get("test_read") is not False:
        raise ValueError("H2 transfer requires test-independent tune selection")
    chosen = selection["world_study"] or selection["selected_two_level_cnn"]
    source = base/("world_studies" if selection["world_study"] else "studies")/chosen
    _, _, _, _, contract = reuse.load(source)
    reuse.trained(source, contract)
    report = read(source/"comparison_validation.json")
    if (report.get("success") is not True or report.get("split") != "validation"
            or report.get("full_cfd_wall_clock_measured") is not True or not report.get("coupled")
            or not all(r.get("success") is True for c in report["coupled"].values() for r in c.values())):
        raise ValueError("source method has not passed coupled physical validation")
    settings = hydrogen_settings(read(source/"configuration.json"), plan,
                                 world_selected=bool(selection["world_study"]))
    files = [base/"selection.json", source/"configuration.json", source/"expert/candidate.pt",
             source/"reuse/training.json", source/"comparison_validation.json"]
    pin = dict(schema="reacting-method-transfer-v1", source_family=family, source_study=chosen,
        source_files_sha256={str(p.relative_to(root)): file_hash(p) for p in files},
        plan_sha256=file_hash(root/"plan.json"), configuration=settings,
        transfer_mode="architecture_retrained_on_h2", weights_transferred=False,
        source_selection_split="tune", test_read_for_selection=False,
        source_world_selected=bool(selection["world_study"]),
        world_model_benefit_established=False,
        target_world_gate="H2 train/tune probe must independently show refresh/reuse tradeoff")
    target = root/"hydrogen_transfer.json"
    if target.exists() and read(target) != pin:
        raise ValueError("frozen hydrogen transfer changed")
    write(target, pin)
    return settings, pin


def advance_hydrogen(root, plan, stop):
    """Fresh reactive learning, then paired real CFD; no test-driven redesign."""
    if importlib.util.find_spec("cantera") is None or not shutil.which("chemkinToFoam"):
        raise RuntimeError("H2 validation requires Cantera and OpenFOAM chemkinToFoam")
    settings, pin = pin_hydrogen_transfer(root, plan)
    family = H2_FAMILY
    local = deepcopy(plan)
    h2 = plan["hydrogen"]
    local["families"][family] = dict(cases=h2["cases"], timing=h2, recording_arm="selected_method")
    config = root/"configurations"/(family+"_selected_method.json")
    if config.exists() and read(config) != settings:
        raise ValueError("saved H2 configuration changed")
    write(config, settings)
    recordings = [collect_case(root, family, case, local) for case in h2["cases"]]
    out = prepare_arm(root, family, dict(id="selected_method"), local, recordings)
    tune = fit_and_score(out, local)
    validation = evaluate_study(root, family, out, "validation", local)
    result = dict(completed_through="validation", transfer=pin,
        study=str(out.relative_to(root)), probe=read(out/"reuse/collection.json")["probe"],
        models_available=tune["models_available"], validation=validation,
        hydrogen_combustion_validated=True, held_out_test_completed=False,
        world_model_benefit_established=False,
        timing_scope="measured full CFD process wall time including chemistry, transport, pressure and I/O",
        chemistry_time_separately_measured=False,
        accuracy_scope="T, H2, U, p and continuity against reference CFD; positive finite Qdot required; not experimental validation")
    if stop == "test":
        from adaptive_mg.v67.world_model.data import file_hash
        freeze_path = root/"families"/family/"transfer_freeze.json"
        frozen = dict(transfer_sha256=file_hash(root/"hydrogen_transfer.json"),
                      validation_sha256=file_hash(out/"comparison_validation.json"))
        if freeze_path.exists() and read(freeze_path) != frozen:
            raise ValueError("H2 transfer changed after test freeze")
        for temporal, path in ((False, out/"freeze.json"), (True, out/"reuse/freeze.json")):
            if not path.exists():
                command(local, out, temporal, "freeze")
        write(freeze_path, frozen)
        result.update(test=evaluate_study(root, family, out, "test", local),
                      completed_through="test", held_out_test_completed=True)
    write(root/"families"/family/"summary.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=REPO/"configs/v6_7_flow_comparison.json")
    p.add_argument("--families", choices=FAMILIES, nargs="+")
    p.add_argument("--profile", choices=("quick", "full"))
    p.add_argument("--n", type=int)
    p.add_argument("--repeats", type=int)
    p.add_argument("--timeout", type=float)
    p.add_argument("--stop-after", choices=STAGES, default="probe")
    p.add_argument("--dry-run", action="store_true", help="write the pinned plan only; no native tools required")
    p.add_argument("--resume", action="store_true", help="continue the same pinned plan; keep completed stages")
    p.add_argument("--include-legacy-control", action="store_true", help="also probe the original zero-composition training cohort")
    p.add_argument("--independent-grus", action="store_true", help="add independent role memories of widths 16/32/64")
    p.add_argument("--full-factorial", action="store_true", help="sweep WM on both two-level CNNs and couple every declared policy")
    p.add_argument("--with-h2", action="store_true", help="after nonreacting validation, retrain the selected structure on reacting H2")
    p.add_argument("--h2-source-family", choices=FAMILIES,
                   help="predeclare structure source, default h2_crossflow; never select a source from test results")
    p.add_argument("--h2-end-time", type=float, help="reacting duration, default 2e-4 s (smoke validation, not developed flame)")
    p.add_argument("--h2-delta-t", type=float, help="reacting time step, default 1e-6 s")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.output, args.config = args.output.resolve(), args.config.resolve()
    if args.repeats is not None and args.repeats < 1:
        p.error("positive repeats required")
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        p.error("finite positive timeout required")
    root = args.output
    if args.resume:
        plan = check_plan(root, args)
    else:
        if root.exists():
            p.error("output exists; use --resume or a new directory")
        plan = build_plan(args, read(args.config))
        root.mkdir(parents=True)
        write(root/"plan.json", plan)
        for family, value in plan["families"].items():
            for arm, settings in value["configs"].items():
                write(root/"configurations"/(family+"_"+arm+".json"), settings)
    stop = "plan" if args.dry_run else args.stop_after
    summary = dict(status="running", completed_through=stop, output=str(root), families={},
                   world_model_benefit_established=False, hydrogen_combustion_validated=False)
    try:
        if STAGES.index(stop) >= STAGES.index("collect"):
            preflight(plan)
        with_h2 = plan.get("hydrogen", {}).get("enabled", False)
        primary_stop = "validation" if with_h2 and stop == "test" else stop
        if stop != "plan":
            for family in plan["families"]:
                print("[flow comparison]", family, "through", stop, flush=True)
                write(root/"status.json", dict(status="running", family=family, target_stage=stop))
                summary["families"][family] = advance_family(root, family, plan, primary_stop)
                write(root/"summary.json", summary)
        if with_h2 and stop in ("validation", "test"):
            print("[reacting H2] transfer and validation", flush=True)
            summary["hydrogen"] = advance_hydrogen(root, plan, "validation")
            summary["hydrogen_combustion_validated"] = summary["hydrogen"]["hydrogen_combustion_validated"]
            write(root/"summary.json", summary)
            if stop == "test":
                for family in plan["families"]:
                    summary["families"][family] = advance_family(root, family, plan, "test")
                    write(root/"summary.json", summary)
                summary["hydrogen"] = advance_hydrogen(root, plan, "test")
        elif with_h2:
            summary["hydrogen"] = dict(status="pending", reason="requires all primary validation gates first")
        summary["status"] = "complete"
        write(root/"summary.json", summary)
        write(root/"status.json", summary)
        print(json.dumps(dict(status="complete", completed_through=stop, output=str(root),
                             world_model_benefit_established=False), indent=2))
        return summary
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}", results_preserved=True)
        write(root/"summary.json", summary)
        write(root/"status.json", summary)
        raise


if __name__ == "__main__":
    main()
