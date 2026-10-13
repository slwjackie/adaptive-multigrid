#!/usr/bin/env python3
"""Re-evaluate a verified, trained study with a larger solve budget only.

This standalone runner deliberately leaves the numerical package and original
study untouched. Its own hash and the effective configuration are recorded in a
separate diagnostic contract; the original source/checkpoint checks still run.
No CFD collection, optimizer update, policy fit, validation or test solve occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from adaptive_mg.provenance import hardware_environment
from adaptive_mg.v67.smoother_reuse import study as reuse
from adaptive_mg.v67.world_model.data import digest, file_hash, load_trajectories, write_json


SCHEMA = "fixed-p-budget-replay-v1"
POLICIES = ("classical", "refresh", "reuse")
PHASES = ("setup_seconds", "solve_seconds", "recovery_seconds",
          "verification_seconds", "policy_seconds")
IDENTITY = ("index", "time", "matrix_digest", "rhs_digest", "x0_digest",
            "threshold", "fixed_p_sha256")


class ProgressStates(list):
    def __init__(self, states, label):
        super().__init__(states)
        self.label = label

    def __iter__(self):
        started = perf_counter()
        for i, state in enumerate(super().__iter__()):
            if i % 25 == 0:
                print(f"[{self.label}] {i}/{len(self)} "
                      f"elapsed={perf_counter()-started:.1f}s", flush=True)
            yield state


def run_totals(run):
    rows = run["rows"]
    totals = {key: sum(row[key] for row in rows) for key in PHASES}
    # Feature extraction and unclassified orchestration are retained, not
    # dropped from speed comparisons. Recovery is disjoint from solve time.
    totals["other_seconds"] = sum(row["total_seconds"] for row in rows) - sum(totals.values())
    totals["construction_seconds"] = run["construction_seconds"]
    totals["total_seconds"] = run["total_seconds"]
    return totals


def summarize_runs(runs):
    totals = [run_totals(run) for run in runs]
    rows = [row for run in runs for row in run["rows"]]
    return dict(
        success=all(run["success"] for run in runs),
        pressure_calls_per_repeat=len(runs[0]["rows"]), repeats=len(runs),
        failed_indices=sorted({row["index"] for row in rows if not row["success"]}),
        failed_measurements=sum(not row["success"] for row in rows),
        maximum_cycles=max(row["cycles"] for row in rows),
        median_cycles=float(np.median([row["cycles"] for row in rows])),
        fallback_measurements=sum(row["fallback"] for row in rows),
        neural_apply_calls=sum(row["neural_apply_calls"] for row in rows),
        total_seconds_by_repeat=[run["total_seconds"] for run in runs],
        phase_totals_by_repeat=totals,
        median_total_seconds=float(np.median([run["total_seconds"] for run in runs])),
        median_phase_seconds={key: float(np.median([t[key] for t in totals])) for key in totals[0]},
    )


def compare(source, target, cfg, expert, settings, repeats):
    rng = np.random.default_rng(settings["seed"])
    cases, files, orders = [], {}, []
    warmup_seconds = 0.
    for split in ("train", "tune"):
        for meta, states in load_trajectories(source / "data", split):
            # Warm lazy Python/Torch kernels equally, then create fresh solvers
            # for every measured repeat. Cold hierarchy setup remains charged.
            start = perf_counter()
            for policy in POLICIES:
                reuse.run_episode(states[:2], cfg, expert, policy)
            warmup_seconds += perf_counter() - start
            runs = {policy: [] for policy in POLICIES}
            expected = None
            for repeat in range(repeats):
                order = rng.permutation(POLICIES).tolist()
                orders.append(dict(trajectory=meta["id"], repeat=repeat, policies=order))
                for policy in order:
                    label = f"compare {meta['id']} {policy} repeat={repeat+1}/{repeats}"
                    run = reuse.run_episode(ProgressStates(states, label), cfg, expert, policy)
                    identities = [tuple(row[key] for key in IDENTITY) for row in run["rows"]]
                    if expected is not None and identities != expected:
                        raise ValueError("paired A/b/x0, stopping threshold or fixed P changed")
                    expected = identities
                    runs[policy].append(run)
                    path = target / "comparison" / split / f"{meta['id']}_{policy}_{repeat}.json"
                    write_json(path, run)
                    files[str(path.relative_to(target))] = file_hash(path)
                    print(f"[{label}] done failed={sum(not r['success'] for r in run['rows'])} "
                          f"total={run['total_seconds']:.3f}s", flush=True)
            cases.append(dict(trajectory=meta["id"], split=split,
                              policies={name: summarize_runs(value) for name, value in runs.items()}))
    overall = {}
    for policy in POLICIES:
        arms = [case["policies"][policy] for case in cases]
        overall[policy] = dict(
            success=all(arm["success"] for arm in arms),
            failed_systems=sum(len(arm["failed_indices"]) for arm in arms),
            failed_measurements=sum(arm["failed_measurements"] for arm in arms),
            maximum_cycles=max(arm["maximum_cycles"] for arm in arms),
            summed_trajectory_median_seconds=sum(arm["median_total_seconds"] for arm in arms),
        )
    success = all(arm["success"] for arm in overall.values())
    for policy, arm in overall.items():
        eligible = arm["success"] and overall["classical"]["success"]
        arm["replay_speedup_vs_classical"] = (
            overall["classical"]["summed_trajectory_median_seconds"] /
            arm["summed_trajectory_median_seconds"] if eligible else None)
    return dict(schema=SCHEMA, success=success, cases=cases, overall=overall,
                files=files, measurement_order=orders, warmup_seconds=warmup_seconds,
                timing_scope="sum of per-trajectory median solver totals, including construction, "
                             "setup, cycles, residual checks, recovery and policy overhead; "
                             "excludes offline data admission, IO, warmup and reporting; no CFD/IPC timing",
                phase_note="phase medians need not sum to median total; other includes features/orchestration",
                full_cfd_wall_clock_measured=False, held_out_speedup_established=False)


def probe_report(episodes, margin):
    # Share the full-horizon, nonoverlapping gate with the main study runner.
    return reuse.tradeoff_report(episodes, margin)


def collect_probe(source, target, cfg, expert, settings):
    episodes, files = [], {}
    for split in ("train", "tune"):
        for meta, states in load_trajectories(source / "data", split):
            ep = reuse.collect_episode(meta, ProgressStates(states, f"probe {meta['id']}"), cfg, expert, settings)
            path = target / "probe" / "episodes" / split / f"{meta['id']}.json"
            write_json(path, ep)
            files[str(path.relative_to(target))] = file_hash(path)
            episodes.append(ep)
    report = dict(schema=SCHEMA, episodes=files, probe=probe_report(episodes, settings["minimum_gain"]),
                  numerical_splits=["train", "tune"], model_fitted=False)
    write_json(target / "probe" / "collection.json", report)
    return report


def diagnostic_zip(target):
    path = target / "budget_diagnostics.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(target.rglob("*.json")):
            archive.write(file, file.relative_to(target))
    return path


def run(args):
    source, target = args.run_dir.resolve(), args.output.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("output must be separate from the original study")
    if target.exists():
        raise FileExistsError("output already exists; choose a new directory to preserve evidence")
    print("[verify] original source, configuration, recordings and trained checkpoint", flush=True)
    source, settings, original, expert, contract = reuse.load(source)
    reuse.assert_open(source)
    if args.max_cycles <= original.mg.max_cycles:
        raise ValueError("new max_cycles must exceed the original budget")
    repeats = args.repeats if args.repeats is not None else settings["baseline_repeats"]
    if type(repeats) is not int or repeats < 1:
        raise ValueError("positive repeat count required")
    cfg = replace(original, mg=replace(original.mg, max_cycles=args.max_cycles))
    before, after = original.to_dict(), cfg.to_dict()
    after["mg"]["max_cycles"] = before["mg"]["max_cycles"]
    if before != after:
        raise ValueError("only max_cycles may change")
    target.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, source_run=str(source), original_contract=contract,
                    runner_sha256=file_hash(__file__), original_solver=before, effective_solver=cfg.to_dict(),
                    changed_setting={"mg.max_cycles": [original.mg.max_cycles, args.max_cycles]},
                    effective_solver_digest=digest(cfg.to_dict()), checkpoint_retrained=False,
                    settings=settings, repeats=repeats, policies=list(POLICIES),
                    numerical_splits=["train", "tune"],
                    source_integrity_scope="original loader also hashes recording bytes from all splits; "
                                           "no validation/test arrays are solved or used for tuning",
                    hardware=hardware_environment())
    write_json(target / "manifest.json", manifest)
    summary = dict(schema=SCHEMA, status="running", output=str(target),
                   max_cycles=args.max_cycles, checkpoint_retrained=False,
                   world_model_benefit_established=False, hydrogen_validation_completed=False,
                   full_cfd_wall_clock_measured=False)
    write_json(target / "summary.json", summary)
    try:
        comparison = compare(source, target, cfg, expert, settings, repeats)
        write_json(target / "comparison.json", comparison)
        summary.update(comparison=comparison["overall"], comparison_passed=comparison["success"])
        if not comparison["success"]:
            summary.update(status="comparison_failed", completed_through="comparison",
                           probe_status="skipped_due_to_solver_failures")
        elif args.stop_after == "comparison":
            summary.update(status="complete", completed_through="comparison", probe_status="not_requested")
        else:
            write_json(target / "summary.json", dict(summary, status="probing"))
            collection = collect_probe(source, target, cfg, expert, settings)
            probe = collection["probe"]
            summary.update(status="probe_failed" if probe["failed_rollout_measurements"] else "complete",
                           completed_through="probe", probe_status="evaluated", probe=probe)
        if file_hash(source / "expert/candidate.pt") != contract["expert_sha256"]:
            raise ValueError("original checkpoint changed during replay")
    except Exception as exc:
        summary.update(status="error", error=str(exc))
        raise
    finally:
        write_json(target / "summary.json", summary)
        archive = diagnostic_zip(target)
        print(f"Results: {target / 'summary.json'}\nDiagnostics: {archive}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="original study directory")
    parser.add_argument("--output", type=Path, required=True, help="new, separate diagnostic directory")
    parser.add_argument("--max-cycles", type=int, default=600)
    parser.add_argument("--repeats", type=int, help="defaults to original baseline_repeats")
    parser.add_argument("--stop-after", choices=("comparison", "probe"), default="probe")
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:
        parser.exit(1, f"budget replay failed: {exc}\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
