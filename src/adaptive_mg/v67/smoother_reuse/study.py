"""Measure stencil reuse, fit only on train, select on tune, freeze before test."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ...provenance import hardware_environment, operator_digest
from ..h2_fixed_p import study as base
from ..h2_fixed_p.backend import FixedPSolver
from ..h2_fixed_p.data import read
from ..h2_fixed_p.evaluation import _vector_digest, _json_finite
from ..world_model.data import digest, file_hash, load_trajectories, write_json
from . import VERSION, ACTIONS
from .policy import ReuseSolver, check_policy, observation
from .world import fit, calibrate, Predictor, architecture


WORLD_ARCHITECTURE_KEYS = ("hidden", "world_memory", "world_head", "world_head_hidden")


def world_architecture(w):
    """One predeclared architecture per run; missing keys reproduce v1."""
    return architecture(w["hidden"], memory=w.get("world_memory", "shared"),
                        head=w.get("world_head", "linear"), head_hidden=w.get("world_head_hidden", 16))


def settings_from(settings):
    w = settings.get("smoother_reuse")
    if not isinstance(w, dict) or w.get("schema") != VERSION:
        raise ValueError("declare a fixed-p-smoother-reuse-v1 configuration before collection")
    names = w.get("policies", [])
    if len(names) != len(set(names)) or not {"refresh", "reuse"}.issubset(names):
        raise ValueError("unique refresh/reuse and predeclared heuristic baselines required")
    for name in names:
        check_policy(name)
        if name.startswith("world") or name == "memoryless":
            raise ValueError("learned policies are fitted separately, not heuristic candidates")
    for key in ("measurement_repeats", "baseline_repeats", "hidden", "epochs", "behavior_max_age", "probe_horizon"):
        if type(w.get(key)) is not int or w[key] < 1:
            raise ValueError("positive integer " + key + " required")
    if not 0 < w.get("minimum_gain", 0) < 1:
        raise ValueError("minimum_gain must be in (0,1)")
    if type(w.get("seed")) is not int or w["seed"] < 0 or type(w.get("fit_world")) is not bool:
        raise ValueError("nonnegative integer seed and boolean fit_world required")
    world_architecture(w)
    return w


def load(run_dir):
    out, settings, cfg, manifest = base.load_run(run_dir)
    w = settings_from(settings)
    expert = base.load_expert(out, cfg)
    contract = dict(schema=VERSION, source_digest=manifest["source_digest"],
        configuration_digest=manifest["configuration_digest"], dataset_sha256=manifest["dataset_sha256"],
        expert_sha256=file_hash(out/"expert/candidate.pt"), settings_digest=digest(w),
        experiment_kind=manifest["experiment_kind"])
    return out, w, cfg, expert, contract


def assert_open(out):
    if (out/"freeze.json").exists() or (out/"reuse/freeze.json").exists():
        raise ValueError("development closed after freeze; use a new run")


def compact(result, s):
    st = result.stats
    return _json_finite(dict(index=s.index, time=s.time, matrix_digest=s.matrix_digest,
        rhs_digest=_vector_digest(s.b), x0_digest=_vector_digest(s.x0),
        success=result.success, threshold=result.threshold, final_true_residual=st["final_true_residual"],
        cycles=result.cycles, fallback=result.fallback, reason=result.reason,
        total_seconds=result.total_seconds, setup_seconds=result.setup_seconds, solve_seconds=result.solve_seconds,
        policy_seconds=st.get("policy_seconds", 0.), recovery_seconds=st["recovery_seconds"],
        nn_setup_seconds=sum(v.get("nn_setup_seconds", 0.) for v in st["builds"]),
        nn_generation_seconds=sum(v.get("operator_generation_seconds", 0.) for v in st["builds"]),
        operator_feature_seconds=sum(v.get("operator_feature_seconds", 0.) for v in st["builds"]),
        neural_apply_seconds=st.get("neural_apply_seconds", 0.),
        backend_total_seconds=st.get("backend_total_seconds", result.total_seconds),
        temporal_overhead_seconds=st.get("temporal_overhead_seconds", 0.),
        feature_seconds=st.get("feature_seconds", 0.),
        verification_seconds=st["verification_seconds"], fixed_p_sha256=st["hierarchy_p_digest_final"],
        smoother_age=st["smoother_age"], smoother_reused=st["smoother_reused"],
        smoother_refreshed=st["smoother_refreshed"], neural_apply_calls=st["neural_apply_calls"],
        smoother_nn_calls=sum(v.get("smoother_nn_calls", 0) for v in st["builds"]),
        smoother_action=st.get("smoother_action_selected"), policy_decision=st.get("policy_decision", {})))


def collect_episode(meta, states, cfg, expert, w):
    if meta["split"] not in ("train", "tune"):
        raise ValueError("counterfactual label collection cannot read validation/test")
    rng = np.random.default_rng(w["seed"] + int(digest(meta["id"])[:8], 16))
    solver = ReuseSolver(cfg, expert)
    ep = dict(trajectory=meta["id"], case_group=meta["case_group"], split=meta["split"],
              obs=[], available=[], log_cost=[], success=[], next_obs=[], behavior=[], measurements=[],
              horizon_measurements=[], probe_horizon=w["probe_horizon"])
    for i, s in enumerate(states):
        raw = observation(s, solver)
        available = [True, solver.core.cached_smoother is not None]
        outcomes, rows, horizon_rows = {}, {}, {}
        for action in rng.permutation(np.flatnonzero(available)):
            trials = []
            horizon_rows[str(int(action))] = []
            for _ in range(w["measurement_repeats"]):
                branch = solver.fork()
                result = branch.step(s, action=ACTIONS[action])
                trials.append((branch, result))
                # An upfront refresh can repay its cost later. Actual train/tune
                # lookahead supplies labels, never online decision inputs.
                # Both branches then reuse, with the same residual safeguards.
                future = branch.fork()
                cost, okay = result.total_seconds, result.success
                calls = [compact(result, s)]
                following = states[i+1:i+w["probe_horizon"]]
                for next_state in following:
                    next_result = future.step(next_state, action="reuse")
                    cost += next_result.total_seconds
                    okay = okay and next_result.success
                    calls.append(compact(next_result, next_state))
                horizon_rows[str(int(action))].append(dict(total_seconds=cost, success=bool(okay),
                    pressure_calls=1+len(following), calls=calls))
            trials.sort(key=lambda pair: pair[1].total_seconds)
            outcomes[int(action)] = trials[len(trials)//2]
            rows[str(int(action))] = [compact(result, s) for _, result in trials]
        costs, success = [0., 0.], [False, False]
        nxt = np.zeros((2, len(raw)), np.float32)
        for a, (branch, result) in outcomes.items():
            costs[a] = np.log(max(result.total_seconds, 1e-8))
            success[a] = all(row["success"] for row in rows[str(a)])
            if i+1 < len(states):
                nxt[a] = observation(states[i+1], branch)
        # Randomized ages cover different cached operators, not just always-new.
        a = 0 if not available[1] or solver.core.smoother_age >= w["behavior_max_age"] or rng.random() < .1 else 1
        if not success[a] and success[0]:
            a = 0
        solver = outcomes[a][0]
        for key, value in dict(obs=raw.tolist(), available=available, log_cost=costs, success=success,
                               next_obs=nxt.tolist(), behavior=a, measurements=rows,
                               horizon_measurements=horizon_rows).items():
            ep[key].append(value)
    return ep


def tradeoff_report(episodes, margin):
    """Retain raw windows, but gate fits using complete disjoint winning windows."""
    counts = dict(refresh_wins=0, reuse_wins=0, inconclusive=0, failed=0)
    wins = dict(refresh_wins=0, reuse_wins=0)
    selected, failed_measurements = [], 0
    for episode_index, ep in enumerate(episodes):
        end = 0
        horizon = ep["probe_horizon"]
        for index, (available, measurements) in enumerate(zip(ep["available"], ep["horizon_measurements"])):
            # Failures when only refresh is available still invalidate the gate.
            failed_measurements += sum(not row["success"] for rows in measurements.values() for row in rows)
            if not all(available):
                continue
            refresh, reuse = measurements["0"], measurements["1"]
            if not all(row["success"] for row in refresh+reuse):
                counts["failed"] += 1
                continue
            a, b = [r["total_seconds"] for r in refresh], [r["total_seconds"] for r in reuse]
            if not a or not b or not all(v is not None and np.isfinite(v) and v > 0 for v in a+b):
                raise ValueError("invalid probe timing")
            if min(a) > max(b)*(1+margin):
                action = "reuse"
            elif min(b) > max(a)*(1+margin):
                action = "refresh"
            else:
                action = None
                counts["inconclusive"] += 1
            if action is not None:
                counts[action+"_wins"] += 1
                complete = index+horizon <= len(ep["available"]) and all(
                    row["pressure_calls"] == horizon for row in refresh+reuse)
                if index >= end and complete:
                    wins[action+"_wins"] += 1
                    selected.append(dict(trajectory=ep.get("trajectory", str(episode_index)),
                        start_position=index, end_position_exclusive=index+horizon, action=action))
                    end = index+horizon
    candidate = all(n >= 3 for n in wins.values()) and failed_measurements == 0
    return dict(counts, world_model_candidate=candidate, world_model_benefit_established=False,
        nonoverlapping_wins=wins, selected_windows=selected, failed_rollout_measurements=failed_measurements,
        margin=margin, horizons=sorted({e["probe_horizon"] for e in episodes}),
        rule="at least three complete nonoverlapping repeated-cost wins per action, no failed rollouts",
        count_scope="top-level wins/inconclusive/failed retain all-window counts; candidacy uses "
                    "nonoverlapping_wins; disjoint pressure calls can still be temporally correlated",
        scope="train/tune short rollouts: refresh or reuse now, guarded reuse afterward; overlapping windows are not independent evidence, not an oracle bound or speed claim")


def collect(run_dir):
    out, w, cfg, expert, contract = load(run_dir)
    assert_open(out)
    target = out/"reuse"
    target.mkdir(exist_ok=True)
    if (target/"collection.json").exists() or (target/"episodes").exists():
        raise FileExistsError("new temporal collection required; existing evidence is preserved")
    episodes, files = [], {}
    for split in ("train", "tune"):
        cohort = load_trajectories(out/"data", split)
        if not cohort:
            raise ValueError("independent train AND tune cases required for temporal fitting")
        for meta, states in cohort:
            print("[reuse labels]", split, meta["id"], flush=True)
            ep = collect_episode(meta, states, cfg, expert, w)
            path = target/"episodes"/split/(meta["id"]+".json")
            write_json(path, ep)
            files[str(path.relative_to(target))] = file_hash(path)
            episodes.append(ep)
    report = dict(contract=contract, episodes=files, probe=tradeoff_report(episodes, w["minimum_gain"]),
                  test_read=False, validation_read=False, actions=list(ACTIONS))
    write_json(target/"collection.json", report)
    return report


def run_episode(states, cfg, expert, policy, *, artifact=None, reference="reuse", minimum_gain=.05):
    started = perf_counter()
    if policy == "classical":
        solver = FixedPSolver(cfg)
    else:
        pred = Predictor(artifact, minimum_gain=minimum_gain) if artifact is not None else None
        solver = ReuseSolver(cfg, expert, policy, predictor=pred, reference=reference)
    construction = perf_counter()-started
    rows = []
    expected_p = None
    for s in states:
        identity = (s.matrix_digest, _vector_digest(s.b), _vector_digest(s.x0))
        r = solver.step(s)
        row = compact(r, s)
        current = (operator_digest(s.a), _vector_digest(s.b), _vector_digest(s.x0))
        if identity != current:
            raise ValueError("solver mutated the paired input")
        expected_p = expected_p or solver.p_digest
        if expected_p != solver.p_digest:
            raise ValueError("P changed during temporal sequence")
        rows.append(row)
    return dict(rows=rows, success=all(r["success"] for r in rows),
                construction_seconds=construction, total_seconds=construction+sum(r["total_seconds"] for r in rows),
                fixed_p_sha256=expected_p, policy=policy,
                timing_subset_note="nn_generation_seconds and operator_feature_seconds are within nn_setup_seconds, "
                                   "which is within setup_seconds; neural_apply_seconds is "
                                   "within solve/recovery; policy_seconds is within temporal_overhead_seconds; "
                                   "do not add these subset timings to the enclosing phases")


def inherited_baselines(out, contract, collection):
    """Pin one tune-selected cheap baseline across architecture-only branches.

    This only accepts explicit branches made by the comparison runner. No
    solver, dataset, CNN, behavior-policy labels or non-architecture setting
    may change. Parent evidence is retained, rather than relabeled as fresh
    measurements for every architecture.
    """
    path = out/"world_parent.json"
    if not path.exists():
        return None
    pins = read(path)
    parent = Path(pins["parent"]).resolve()
    if parent == out.resolve():
        raise ValueError("world baseline parent must be a distinct study")
    files = dict(configuration_sha256="configuration.json", run_manifest_sha256="run_manifest.json",
                 collection_sha256="reuse/collection.json", checkpoint_sha256="expert/candidate.pt",
                 parent_training_sha256="reuse/training.json")
    if any(pins.get(key) != file_hash(parent/name) for key, name in files.items()):
        raise ValueError("world baseline parent evidence changed")
    _, _, _, _, parent_contract = load(parent)
    changed_contract = {"configuration_digest", "settings_digest"}
    if ({k: v for k, v in parent_contract.items() if k not in changed_contract}
            != {k: v for k, v in contract.items() if k not in changed_contract}):
        raise ValueError("world baseline parent solver/data/CNN/source differs")
    before, after = read(parent/"configuration.json"), read(out/"configuration.json")
    for settings in (before, after):
        for key in WORLD_ARCHITECTURE_KEYS:
            settings["smoother_reuse"].pop(key, None)
    if before != after:
        raise ValueError("world baseline inheritance permits architecture changes only")
    parent_collection = read(parent/"reuse/collection.json")
    if parent_collection["contract"] != parent_contract:
        raise ValueError("world baseline parent collection contract differs")
    inherited = collection.get("inherited_measurements", {})
    if (collection["episodes"] != parent_collection["episodes"] or collection["episodes"] != pins["episodes"]
            or inherited.get("parent") != str(parent)
            or inherited.get("collection_sha256") != pins["collection_sha256"]
            or inherited.get("byte_identical_episodes") is not True):
        raise ValueError("world baseline parent counterfactual labels differ")
    for name, sha in collection["episodes"].items():
        if file_hash(parent/"reuse"/name) != sha or file_hash(out/"reuse"/name) != sha:
            raise ValueError("world baseline parent counterfactual episode changed")
    training = trained(parent, parent_contract)
    return dict(scores=deepcopy(training["baseline_scores"]), selected=training["selected"],
                provenance=dict(parent=str(parent), training_sha256=pins["parent_training_sha256"],
                    parent_pin_sha256=file_hash(path), measured_again=False))


def train(run_dir):
    out, w, cfg, expert, contract = load(run_dir)
    assert_open(out)
    target = out/"reuse"
    if (target/"training.json").exists():
        raise FileExistsError("temporal fitting already completed")
    collection = read(target/"collection.json")
    if collection["contract"] != contract:
        raise ValueError("temporal collection provenance changed")
    eps = dict(train=[], tune=[])
    for path, sha in collection["episodes"].items():
        if file_hash(target/path) != sha:
            raise ValueError("counterfactual labels changed")
        ep = read(target/path)
        eps[ep["split"]].append(ep)
    if not eps["train"] or not eps["tune"]:
        raise ValueError("missing independent train/tune trajectories")
    if {e["case_group"] for e in eps["train"]} & {e["case_group"] for e in eps["tune"]}:
        raise ValueError("physical case leaks between fit and tuning")
    inherited = inherited_baselines(out, contract, collection)
    if inherited is not None:
        scores, selected = inherited["scores"], inherited["selected"]
    else:
        scores = {name: [] for name in w["policies"]}
        rng = np.random.default_rng(w["seed"])
        for meta, states in load_trajectories(out/"data", "tune"):
            per = {name: [] for name in scores}
            for _ in range(w["baseline_repeats"]):
                for name in rng.permutation(list(scores)):
                    per[name].append(run_episode(states, cfg, expert, str(name)))
            for name, runs in per.items():
                scores[name].append(dict(trajectory=meta["id"], success=all(r["success"] for r in runs),
                                        seconds=float(np.median([r["total_seconds"] for r in runs]))))
        eligible = [name for name in scores if all(r["success"] for r in scores[name])]
        if not eligible:
            raise RuntimeError("no heuristic passed all tune trajectories")
        selected = min(eligible, key=lambda name: sum(r["seconds"] for r in scores[name]))
    models = {}
    model_details = {}
    fit_seconds = 0.
    if collection["probe"]["world_model_candidate"] and w.get("fit_world", True):
        start = perf_counter()
        for name, recurrent in (("world", True), ("memoryless", False)):
            print("[reuse fit]", name, flush=True)
            artifact = fit(eps["train"], hidden=w["hidden"], epochs=w["epochs"],
                           seed=w["seed"], recurrent=recurrent, **world_architecture(w))
            artifact["calibration_margin"] = calibrate(artifact, eps["tune"])
            artifact["contract"] = contract
            path = target/(name+".pt")
            torch.save(artifact, path)
            models[name] = file_hash(path)
            model_details[name] = dict(hidden=artifact["hidden"], recurrent=recurrent,
                architecture=artifact["architecture"], parameter_count=artifact["parameter_count"])
        fit_seconds = perf_counter()-start
    report = dict(schema=VERSION, contract=contract, selected=selected, baseline_scores=scores,
        collection_sha256=file_hash(target/"collection.json"), models=models, training_seconds=fit_seconds,
        model_details=model_details,
        inherited_baseline=inherited["provenance"] if inherited is not None else None,
        world_status="fitted_requires_validation" if models else "skipped_no_measured_tradeoff_or_disabled",
        probe=collection["probe"], world_model_benefit_established=False)
    write_json(target/"training.json", report)
    return report


def trained(out, contract=None):
    target = Path(out)/"reuse"
    report = read(target/"training.json")
    if report.get("schema") != VERSION or (contract is not None and report["contract"] != contract):
        raise ValueError("temporal training contract changed")
    if report["collection_sha256"] != file_hash(target/"collection.json"):
        raise ValueError("temporal collection changed after fitting")
    for name, sha in report["models"].items():
        if file_hash(target/(name+".pt")) != sha:
            raise ValueError("temporal checkpoint changed")
    inherited = report.get("inherited_baseline")
    if inherited:
        pins = Path(out)/"world_parent.json"
        if (file_hash(pins) != inherited["parent_pin_sha256"]
                or file_hash(Path(inherited["parent"])/"reuse/training.json") != inherited["training_sha256"]):
            raise ValueError("inherited temporal baseline evidence changed")
    return report


def load_artifact(out, name, training):
    key = "memoryless" if name == "memoryless" else "world"
    if key not in training["models"]:
        raise ValueError("world fitting was not enabled by measured train/tune evidence")
    artifact = torch.load(Path(out)/"reuse"/(key+".pt"), map_location="cpu", weights_only=True)
    if artifact["contract"] != training["contract"]:
        raise ValueError("world artifact belongs to a different solver/data/physics")
    return artifact


def policy_solver(run_dir, cfg, expert, policy, *, split):
    """Load a checked policy for the existing persistent CFD service."""
    out, w, checked_cfg, _, contract = load(run_dir)
    if cfg.to_dict() != checked_cfg.to_dict():
        raise ValueError("live solver config and temporal training differ")
    if split == "test":
        check_freeze(out)
    elif (out/"reuse/freeze.json").exists():
        raise ValueError("temporal validation is closed after freeze")
    training = trained(out, contract)
    policy = training["selected"] if policy == "selected" else check_policy(policy)
    learned = policy.startswith("world") or policy == "memoryless"
    if not learned and policy not in w["policies"]:
        raise ValueError("live policy was not declared before evaluation")
    predictor = Predictor(load_artifact(out, policy, training), minimum_gain=w["minimum_gain"]) if learned else None
    return ReuseSolver(cfg, expert, policy, predictor=predictor, reference=training["selected"])


def evaluate(run_dir, split="validation", repeats=3):
    out, w, cfg, expert, contract = load(run_dir)
    if split not in ("validation", "test") or type(repeats) is not int or repeats < 1:
        raise ValueError("validation/test and positive repeat count required")
    if split == "test":
        check_freeze(out)
    else:
        assert_open(out)
    target = out/"reuse"/split
    if target.exists():
        raise FileExistsError("temporal evaluation is immutable; use a new run")
    training = trained(out, contract)
    names = ["classical", *w["policies"]]
    if training["models"]:
        names += ["world", "world_horizon1", "world_stateless", "memoryless"]
    artifacts = {n: load_artifact(out, n, training) for n in names if n.startswith("world") or n == "memoryless"}
    rng = np.random.default_rng(w["seed"]+391)
    totals = {n: np.zeros(repeats) for n in names}
    successes = {n: True for n in names}
    cases = []
    for meta, states in load_trajectories(out/"data", split):
        cases.append(meta["id"])
        expected_p = None
        for rep in range(repeats):
            for name in rng.permutation(names):
                run = run_episode(states, cfg, expert, str(name), artifact=artifacts.get(name),
                                  reference=training["selected"], minimum_gain=w["minimum_gain"])
                expected_p = expected_p or run["fixed_p_sha256"]
                if run["fixed_p_sha256"] != expected_p:
                    raise ValueError("P differs across paired policies")
                totals[name][rep] += run["total_seconds"]
                successes[name] &= run["success"]
                write_json(target/f'{meta["id"]}_{name}_{rep}.json', run)
        print("[reuse evaluate]", split, meta["id"], flush=True)
    if not cases:
        raise ValueError("empty temporal evaluation cohort")
    median = {n: float(np.median(v)) for n, v in totals.items()}
    ref = training["selected"]
    metrics = {n: dict(success=successes[n], median_seconds=median[n],
        speedup_vs_classical=median["classical"]/median[n] if successes[n] and successes["classical"] else None,
        speedup_vs_tuned_reuse=median[ref]/median[n] if successes[n] and successes[ref] else None) for n in names}
    report = dict(schema=VERSION, contract=contract, split=split, repeats=repeats, cases=cases,
        selected_reference=ref, metrics=metrics, success=all(successes.values()),
        training_sha256=file_hash(out/"reuse/training.json"), hardware=hardware_environment(refresh=True),
        world_model_enabled=bool(training["models"]), world_model_benefit_established=False,
        model_details=training.get("model_details", {}),
        full_cfd_wall_clock_measured=False, physical_observables_validated=False,
        time_scope="solver construction + current-A setup + learned refresh/reuse + policy + true residual + recovery",
        excluded_costs=["offline NN/world training (separately reported)", "input/report I/O"],
        note="Replay speed is not coupled CFD speed; valid slowdowns are retained.")
    write_json(target/"report.json", report)
    return report


def freeze(run_dir):
    out, _, _, _, contract = load(run_dir)
    base.check_freeze(out)
    target = out/"reuse"
    if (target/"freeze.json").exists():
        raise FileExistsError("temporal study already frozen")
    training = trained(out, contract)
    report = read(target/"validation/report.json")
    if report["contract"] != contract or report["training_sha256"] != file_hash(target/"training.json"):
        raise ValueError("validation evidence does not bind the temporal model")
    names = ["freeze.json", "reuse/collection.json", "reuse/training.json", "reuse/validation/report.json"]
    names += ["reuse/"+n+".pt" for n in training["models"]]
    names += ["reuse/"+p for p in read(target/"collection.json")["episodes"]]
    if training.get("inherited_baseline"):
        names.append("world_parent.json")
    value = dict(schema=VERSION, contract=contract, artifacts={n:file_hash(out/n) for n in names})
    write_json(target/"freeze.json", value)
    return value


def check_freeze(run_dir):
    out = Path(run_dir)
    base.check_freeze(out)
    value = read(out/"reuse/freeze.json")
    if value.get("schema") != VERSION or value["contract"]["source_digest"] != base.source_digest():
        raise ValueError("temporal freeze contract changed")
    required = {"freeze.json", "reuse/collection.json", "reuse/training.json", "reuse/validation/report.json"}
    training = trained(out)
    required |= {"reuse/"+n+".pt" for n in training["models"]}
    required |= {"reuse/"+p for p in read(out/"reuse/collection.json")["episodes"]}
    if training.get("inherited_baseline"):
        required.add("world_parent.json")
    if set(value["artifacts"]) != required:
        raise ValueError("incomplete temporal freeze")
    for name, sha in value["artifacts"].items():
        if file_hash(out/name) != sha:
            raise ValueError("frozen temporal artifact changed: "+name)
    if training["contract"] != value["contract"]:
        raise ValueError("temporal training changed after freeze")
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("collect", "train", "evaluate", "freeze"))
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args(argv)
    result = evaluate(a.run_dir, a.split, a.repeats) if a.command == "evaluate" else globals()[a.command](a.run_dir)
    import json
    print(json.dumps(result, indent=2, allow_nan=False))
    return result
