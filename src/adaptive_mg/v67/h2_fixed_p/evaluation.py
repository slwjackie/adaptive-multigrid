"""Paired fixed-P replay and externally supplied, actually coupled CFD evidence.

The replay benchmark is a pressure linear-system experiment, not a flame
simulation.  The coupled comparator checks the supplied evidence contract and
field agreement; it cannot authenticate who ran an external CFD executable.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from time import perf_counter
from collections.abc import Mapping

import numpy as np

from ..world_model.data import digest
from ...provenance import operator_digest


ARMS = ("classical", "H_S")
FIELDS = ("T", "H2", "U", "p")


def _vector_digest(value):
    a = np.ascontiguousarray(value, dtype=np.float64)
    return sha256(a.tobytes()).hexdigest()


def _finite_number(value, *, positive=False):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.number))
            and np.isfinite(value) and (value > 0 if positive else value >= 0))


def _json_finite(value):
    """Retain a failed observation without writing nonstandard JSON NaN."""
    if isinstance(value, Mapping):
        return {str(k): _json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _heterogeneity(rows, margin=.05):
    """Descriptive temporal/residual-state evidence, never a controller gate."""
    paired = []
    for item in rows:
        runs = item["runs"]
        if not all(r["success"] for arm in ARMS for r in runs[arm]):
            continue
        for index in range(len(runs["classical"][0]["records"])):
            c = [r["records"][index] for r in runs["classical"]]
            h = [r["records"][index] for r in runs["H_S"]]
            ct = float(np.median([r["total_seconds"] for r in c]))
            ht = float(np.median([r["total_seconds"] for r in h]))
            if ct <= 0 or ht <= 0:
                continue
            paired.append(dict(trajectory=item["trajectory"], index=c[0]["index"],
                               time=c[0]["time"], initial_true_residual=c[0].get("initial_true_residual"),
                               speedup=ct / ht,
                               neural_cycles=int(sum(r.get("neural_cycles", 0) for r in h))))
    ratios = [p["speedup"] for p in paired]
    all_success = all(r["success"] for item in rows for arm in ARMS for r in item["runs"][arm])
    candidate = (all_success and len(paired) >= 2 and any(p["neural_cycles"] for p in paired)
                 and any(v > 1 + margin for v in ratios) and any(v < 1 / (1 + margin) for v in ratios))
    return dict(world_model_candidate=bool(candidate), world_model_enabled=False,
                observational_only=True, margin=margin, paired_snapshots=paired,
                explanation="Mixed measured winners can motivate a separate selector study; "
                "this is not causal residual-state evidence or proof that a world model helps. "
                "No automatic world-model training or activation occurs.")


def evaluate_replay(trajectories, cfg, expert, *, repeats=3, seed=20261010,
                    split="validation", source_metadata=None):
    """Run C and trained H_S with identical snapshots and independently built P.

    ``trajectories`` has the form returned by ``load_trajectories``.  Each arm
    and repeat gets a fresh solver; arm order alternates from a seeded random
    first arm.  All ``step`` wall time, including setup/validation/fallback, is
    charged.  Model/solver construction is reported and charged once per run.
    Input file I/O and report serialization are outside the measured scope.
    """
    from .backend import FixedPSolver

    if split not in ("validation", "test"):
        raise ValueError("paired evaluation requires validation or test")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("positive integer repeat count required")
    if expert is None or expert.metadata.get("training_branch") != "H_S" or expert.metadata.get("optimizer_updates", 0) < 1:
        raise ValueError("paired evaluation requires a genuinely trained H_S expert")
    cfg = replace(cfg, use_transfer=False, spatial=False, gate_mode="open")
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("empty evaluation cohort")
    if len({tr["id"] for tr, _ in trajectories}) != len(trajectories):
        raise ValueError("duplicate trajectory identity")
    rng = np.random.default_rng(seed)
    rows, kinds = [], set()
    for meta, snapshots in trajectories:
        snapshots = list(snapshots)
        if not snapshots or meta.get("split", split) != split:
            raise ValueError("empty trajectory or split mismatch")
        if any(b.index <= a.index or b.time < a.time for a, b in zip(snapshots, snapshots[1:])):
            raise ValueError("snapshot sequence order must be preserved")
        kinds.update(s.source_kind for s in snapshots)
        inputs = [dict(index=s.index, time=s.time, matrix_digest=s.matrix_digest,
                       rhs_digest=_vector_digest(s.b), x0_digest=_vector_digest(s.x0)) for s in snapshots]
        item = dict(trajectory=meta["id"], case_group=meta.get("case_group"), runs={a: [] for a in ARMS}, order=[])
        first = int(rng.integers(0, 2))
        expected_p = None
        for repeat in range(repeats):
            order = ARMS if (first + repeat) % 2 == 0 else ARMS[::-1]
            item["order"].append(list(order))
            for arm in order:
                start = perf_counter()
                acfg = replace(cfg, mode="classical" if arm == "classical" else "research",
                               branch="C" if arm == "classical" else "H_S", use_smoother=arm == "H_S")
                solver = FixedPSolver(acfg, expert=None if arm == "classical" else expert)
                construction = perf_counter() - start
                records = []
                for given, snapshot in zip(inputs, snapshots):
                    start = perf_counter()
                    try:
                        result = solver.step(snapshot)
                        elapsed = perf_counter() - start
                        record = result.record()
                        record["backend_total_seconds"] = record.get("total_seconds")
                        record["total_seconds"] = max(elapsed, float(record.get("total_seconds", 0.)))
                        record["step_wall_seconds"] = elapsed
                        record["initial_true_residual"] = float(result.residuals[0])
                        stats = record.get("stats", {})
                        record["neural_cycles"] = int(stats.get("neural_trial_cycles", record.get("neural_cycles", 0)))
                        record["actual_neural_used"] = bool(stats.get("actual_neural_used", record["neural_cycles"] > 0))
                        finite_residual = _finite_number(record.get("final_true_residual"))
                        finite_threshold = _finite_number(record.get("threshold"))
                        record["success"] = bool(record.get("success") and finite_residual and finite_threshold
                                                 and record["final_true_residual"] <= record["threshold"])
                        for key in ("total_seconds", "setup_seconds", "solve_seconds"):
                            if not _finite_number(record.get(key)):
                                record["success"] = False
                                record["reason"] = "nonfinite or missing timing: " + key
                        pd = solver.p_digest
                        if not pd:
                            record["success"] = False
                            record["reason"] = "missing fixed-P digest"
                        elif expected_p is None:
                            expected_p = pd
                        elif pd != expected_p:
                            record["success"] = False
                            record["reason"] = "fixed-P changed across snapshots/arms/repeats"
                        record["fixed_p_sha256"] = pd
                    except Exception as exc:
                        elapsed = perf_counter() - start
                        record = dict(success=False, reason=f"{type(exc).__name__}: {exc}",
                                      total_seconds=elapsed, setup_seconds=None, solve_seconds=None,
                                      cycles=None, neural_cycles=0, residuals=[], final_true_residual=None,
                                      initial_true_residual=None, threshold=None, fallback=False,
                                      fixed_p_sha256=getattr(solver, "p_digest", None))
                    # Input identity is verified after execution as well as supplied to each arm.
                    if (operator_digest(snapshot.a) != given["matrix_digest"] or _vector_digest(snapshot.b) != given["rhs_digest"]
                            or _vector_digest(snapshot.x0) != given["x0_digest"]):
                        record["success"] = False
                        record["reason"] = "solver mutated paired input"
                    record.update(given)
                    records.append(_json_finite(record))
                success = all(r["success"] for r in records)
                run = dict(repeat=repeat, arm=arm, records=records, success=success,
                           construction_seconds=construction, step_seconds=sum(r["total_seconds"] for r in records),
                           total_seconds=construction + sum(r["total_seconds"] for r in records),
                           cycles=sum(r["cycles"] or 0 for r in records),
                           neural_cycles=sum(r["neural_cycles"] for r in records),
                           fallback_count=sum(bool(r["fallback"]) for r in records))
                item["runs"][arm].append(run)
        valid = all(r["success"] for arm in ARMS for r in item["runs"][arm])
        neural_used = any(r.get("actual_neural_used", False) for run in item["runs"]["H_S"] for r in run["records"])
        item["success"] = valid
        item["trained_neural_smoothing_used"] = neural_used
        item["fixed_p_sha256"] = expected_p
        item["speedup"] = (float(np.median([r["total_seconds"] for r in item["runs"]["classical"]])) /
                           float(np.median([r["total_seconds"] for r in item["runs"]["H_S"]]))) if valid and neural_used else None
        rows.append(item)
    success = all(row["success"] for row in rows)
    eligible = success and all(row["trained_neural_smoothing_used"] for row in rows)
    # Aggregate every trajectory; never drop failures to construct a speed claim.
    totals = {arm: [sum(row["runs"][arm][i]["total_seconds"] for row in rows)
                    for i in range(repeats)] for arm in ARMS}
    speed = float(np.median(totals["classical"]) / np.median(totals["H_S"])) if eligible else None
    return dict(schema="h2-fixed-p-replay-evaluation-v1", split=split, repeats=repeats, seed=seed,
                source_kinds=sorted(kinds), source_metadata=source_metadata or {}, rows=rows,
                success=success, speed_claim_eligible=eligible, speedup=speed, sequence_totals_seconds=totals,
                time_scope="solver construction + all snapshot setup, solve, residual checks, fallback and bridge-free step overhead",
                excludes=["input file I/O", "report serialization", "offline training"],
                full_cfd_wall_clock_measured=False, physical_observables_validated=False,
                combustion_authenticity_verified=False,
                diagnostic=_heterogeneity(rows))


def compare_coupled_runs(classical, neural, tolerances):
    """Compare supplied online CFD runs and arrays without inventing a result.

    Required run metadata: schema='h2-fixed-p-coupled-run-v1', arm,
    source_kind='external_cfd', execution_mode='online_coupled', success,
    case_id, mesh_sha256, chemistry_sha256, initial_conditions_sha256,
    boundary_conditions_sha256, dt_schedule, stopping_criteria, end_time,
    fixed_p_sha256, hardware, execution_threads, pressure_gauge,
    continuity={'passed': bool}, evidence={coupled_run_attested: True,
    solver_log_sha256, field_files_sha256}; the H_S evidence also needs
    expert_checkpoint_sha256 and neural_apply_calls > 0. timing={wall_seconds,
    pressure_seconds}, and fields={T,H2,U,p}.  H2 may be supplied as Y_H2.

    Arrays can describe a final field or a series.  A series must declare
    sample_times, with time on axis zero.  Tolerances must be predeclared:
    {declared_before_evaluation: True, fields: {name: {atol,rtol}},
    pressure_alignment: 'none'|'subtract_spatial_mean'}.  Only explicitly
    declared free-pressure-gauge runs may request mean alignment.

    Invalid or unsuccessful runs remain in the report.  Any invalid evidence,
    failed field comparison, or failed run makes all speedup fields None.
    """
    blockers = []
    field_reports, timing = {}, {}
    runs = {"classical": classical, "H_S": neural}
    tolerances = dict(tolerances or {})
    fields_tol = tolerances.get("fields", {})
    if tolerances.get("declared_before_evaluation") is not True:
        blockers.append("physical tolerances were not declared before evaluation")
    alignment = tolerances.get("pressure_alignment", "none")
    if alignment not in ("none", "subtract_spatial_mean"):
        blockers.append("unknown pressure alignment")
    match_keys = ("case_id", "mesh_sha256", "chemistry_sha256", "initial_conditions_sha256",
                  "boundary_conditions_sha256", "dt_schedule", "stopping_criteria", "end_time",
                  "fixed_p_sha256", "hardware", "execution_threads", "pressure_gauge")
    for key in match_keys:
        a, b = classical.get(key), neural.get(key)
        def missing(value):
            return value is None or isinstance(value, (str, dict, list, tuple)) and len(value) == 0
        if missing(a) or missing(b):
            blockers.append("missing paired metadata: " + key)
        else:
            try:
                if digest(a) != digest(b):
                    blockers.append("paired metadata mismatch: " + key)
            except (TypeError, ValueError):
                blockers.append("invalid/nonfinite paired metadata: " + key)
    try:
        dt = np.asarray(classical.get("dt_schedule"), dtype=float)
        if dt.ndim != 1 or not len(dt) or not np.isfinite(dt).all() or not np.all(dt > 0):
            blockers.append("positive finite dt schedule required")
    except (TypeError, ValueError):
        blockers.append("invalid dt schedule")
    if not _finite_number(classical.get("end_time"), positive=True):
        blockers.append("positive finite end_time required")
    sample_times = classical.get("sample_times")
    try:
        if digest(sample_times) != digest(neural.get("sample_times")):
            blockers.append("field sample times mismatch")
        if sample_times is not None:
            st = np.asarray(sample_times, dtype=float)
            if st.ndim != 1 or not len(st) or not np.isfinite(st).all() or np.any(np.diff(st) <= 0):
                blockers.append("invalid field sample times")
            elif not np.isclose(st[-1], classical.get("end_time", np.nan), rtol=1.e-12, atol=1.e-14):
                blockers.append("field samples do not include declared CFD end time")
    except (TypeError, ValueError):
        blockers.append("invalid field sample times")
    if alignment == "subtract_spatial_mean" and any(r.get("pressure_gauge") != "free_constant" for r in runs.values()):
        blockers.append("pressure mean alignment requires both gauges declared free_constant")
    for arm, run in runs.items():
        if (run.get("schema") != "h2-fixed-p-coupled-run-v1" or run.get("arm") != arm
                or run.get("source_kind") != "external_cfd" or run.get("execution_mode") != "online_coupled"):
            blockers.append(arm + ": actual online coupled CFD evidence required")
        if run.get("success") is not True:
            blockers.append(arm + ": coupled run failed")
        if run.get("pressure_gauge") not in ("absolute", "free_constant"):
            blockers.append(arm + ": pressure gauge must be declared")
        if run.get("continuity", {}).get("passed") is not True:
            blockers.append(arm + ": declared continuity check did not pass")
        evidence = run.get("evidence", {})
        if evidence.get("coupled_run_attested") is not True:
            blockers.append(arm + ": no coupled-run producer attestation")
        if not evidence.get("solver_log_sha256") or not evidence.get("field_files_sha256"):
            blockers.append(arm + ": missing log/field provenance hashes")
        if arm == "H_S" and (not evidence.get("expert_checkpoint_sha256")
                             or not _finite_number(evidence.get("neural_apply_calls"), positive=True)):
            blockers.append("H_S: checkpoint provenance and actual neural application required")
        times = run.get("timing", {})
        wall, pressure = times.get("wall_seconds"), times.get("pressure_seconds")
        if not _finite_number(wall, positive=True) or not _finite_number(pressure, positive=True):
            blockers.append(arm + ": positive finite full-wall and pressure timings required")
        elif pressure > wall:
            blockers.append(arm + ": pressure time exceeds CFD wall time")
        timing[arm] = _json_finite(dict(wall_seconds=wall, pressure_seconds=pressure, success=run.get("success", False)))
    for name in FIELDS:
        try:
            ca, na = classical.get("fields", {}), neural.get("fields", {})
            cv = ca.get(name, ca.get("Y_H2") if name == "H2" else None)
            nv = na.get(name, na.get("Y_H2") if name == "H2" else None)
            a, b = np.asarray(cv, dtype=float), np.asarray(nv, dtype=float)
            if a.shape != b.shape or not a.size or a.ndim == 0:
                raise ValueError("missing or mismatched field shapes")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("nonfinite field values")
            if sample_times is not None and a.shape[0] != len(sample_times):
                raise ValueError("field/time axis mismatch")
            tol = fields_tol.get(name, fields_tol.get("Y_H2", {}) if name == "H2" else {})
            atol, rtol = tol.get("atol"), tol.get("rtol")
            if not _finite_number(atol) or not _finite_number(rtol):
                raise ValueError("explicit finite nonnegative atol and rtol required")
            if name == "p" and alignment == "subtract_spatial_mean":
                axes = tuple(range(1, a.ndim)) if sample_times is not None else tuple(range(a.ndim))
                if not axes:
                    raise ValueError("pressure series needs a spatial axis")
                a, b = a - np.mean(a, axis=axes, keepdims=True), b - np.mean(b, axis=axes, keepdims=True)
            error, limit = np.abs(b - a), atol + rtol * np.abs(a)
            passed = bool(np.all(error <= limit))
            field_reports[name] = dict(passed=passed, shape=list(a.shape), atol=atol, rtol=rtol,
                                       max_absolute_error=float(np.max(error)),
                                       rms_error=float(np.sqrt(np.mean(error ** 2))),
                                       classical_mean=float(np.mean(a)), neural_mean=float(np.mean(b)),
                                       classical_max=float(np.max(a)), neural_max=float(np.max(b)),
                                       pressure_aligned=name == "p" and alignment != "none")
            if not passed:
                blockers.append("physical field outside declared tolerance: " + name)
        except (TypeError, ValueError) as exc:
            field_reports[name] = dict(passed=False, reason=str(exc))
            blockers.append(name + ": " + str(exc))
    blockers = list(dict.fromkeys(blockers))
    valid = not blockers
    run_summaries = {arm: _json_finite({key: run.get(key) for key in
                     ("schema", "arm", "source_kind", "execution_mode", "success", *match_keys,
                      "sample_times", "evidence", "continuity")}) for arm, run in runs.items()}
    return dict(schema="h2-fixed-p-coupled-comparison-v1", success=valid,
                evidence_contract_valid=valid, speed_claim_eligible=valid, blockers=blockers,
                full_cfd_speedup=classical["timing"]["wall_seconds"] / neural["timing"]["wall_seconds"] if valid else None,
                pressure_speedup=classical["timing"]["pressure_seconds"] / neural["timing"]["pressure_seconds"] if valid else None,
                fields=field_reports, timing=timing, runs=run_summaries, tolerances=_json_finite(tolerances),
                pressure_alignment=alignment, supplied_coupled_run_evidence=True,
                independent_authenticity_verified=False,
                provenance_status="producer-attested actual coupled runs; metadata, timing and supplied field arrays checked",
                full_cfd_wall_clock_measured=valid, physical_observables_validated=all(v["passed"] for v in field_reports.values()),
                claim_scope="Agreement of supplied coupled runs within predeclared tolerances; "
                "not independent verification of physical accuracy or provenance.")
