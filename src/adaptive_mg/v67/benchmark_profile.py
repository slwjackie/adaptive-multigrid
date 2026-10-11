"""Measure benchmark suitability without claiming an unmeasured CFD speedup.

Matrix drift is diagnostic, not proof that a World Model can select useful
actions. Pressure-call timing includes the bridge; it is not pure MG time or
the pressure fraction of an unmodified OpenFOAM solver.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..provenance import operator_digest
from .world_model.data import SPLITS, file_hash, inside, load_manifest


def _summary(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), minimum=float(a.min()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                maximum=float(a.max()) if len(a) else None)


def _drift(previous, current):
    if previous.shape != current.shape:
        raise ValueError("matrix shape changed within a fixed-grid trajectory")
    old, new = np.linalg.norm(previous.data), np.linalg.norm(current.data)
    if not (old > 0 and new > 0 and math.isfinite(old) and math.isfinite(new)):
        raise ValueError("positive finite matrix norms required")
    return (float(np.linalg.norm((current - previous).data) / old),
            float(np.linalg.norm((current / new - previous / old).data)))


def profile_dataset(root, *, splits=("train", "validation")):
    """Stream admitted matrices; never read test arrays unless explicitly asked.

Hashes bind this diagnostic to the already admitted CFD dataset. It does not
repeat the importer's SPD test for every matrix. Timestep drift compares the
last pressure correction of consecutive physical times, so inner correctors
are not mislabeled as time evolution.
"""
    root = Path(root)
    if not splits or any(split not in SPLITS for split in splits):
        raise ValueError("explicit valid nonempty splits required")
    manifest = load_manifest(root)
    if manifest.get("source_kind") != "external_cfd":
        raise ValueError("benchmark profiling requires recorded CFD data")
    rows = []
    for trajectory in manifest["trajectories"]:
        if trajectory["split"] not in splits:
            continue
        raw_drift, normalized_drift, time_raw, time_normalized = [], [], [], []
        previous = last_step = None
        previous_time = None
        previous_index = -1
        unique = set()
        physical_steps = 0

        def finish_step(a):
            nonlocal last_step, physical_steps
            if last_step is not None:
                raw, normalized = _drift(last_step, a)
                time_raw.append(raw)
                time_normalized.append(normalized)
            last_step = a
            physical_steps += 1

        for entry in trajectory["snapshots"]:
            path = inside(root, entry["path"])
            if file_hash(path) != entry["sha256"]:
                raise ValueError("snapshot hash changed")
            with np.load(path, allow_pickle=False) as arrays:
                meta = json.loads(str(arrays["metadata"]))
                n = len(arrays["b"])
                a = sp.csr_matrix((arrays["data"], arrays["indices"], arrays["indptr"]), shape=(n, n))
            current_time, index = float(meta["time"]), meta["index"]
            if (not math.isfinite(current_time) or index <= previous_index
                    or previous_time is not None and current_time < previous_time):
                raise ValueError("pressure solve ordering changed")
            matrix_hash = operator_digest(a)
            if matrix_hash != entry["matrix_digest"]:
                raise ValueError("matrix digest changed")
            unique.add(matrix_hash)
            if previous is not None:
                raw, normalized = _drift(previous, a)
                raw_drift.append(raw)
                normalized_drift.append(normalized)
                if current_time != previous_time:
                    finish_step(previous)
            previous, previous_time, previous_index = a, current_time, index
        if previous is None or len(raw_drift) < 1:
            raise ValueError("at least two recorded pressure systems required")
        finish_step(previous)
        rows.append(dict(case_id=trajectory["id"], split=trajectory["split"],
            pressure_systems=len(raw_drift) + 1, physical_steps=physical_steps,
            unique_matrices=len(unique), successive_relative_frobenius=_summary(raw_drift),
            successive_normalized_frobenius=_summary(normalized_drift),
            timestep_relative_frobenius=_summary(time_raw),
            timestep_normalized_frobenius=_summary(time_normalized),
            normalized_time_variation_observed=any(v > 1e-10 for v in time_normalized)))
    if not rows:
        raise ValueError("no trajectories in requested splits")
    return dict(schema="cfd-benchmark-matrix-profile-v1", splits=list(splits),
        experiment_kind=manifest.get("experiment_kind"), trajectories=rows,
        test_arrays_read="test" in splits, world_model_benefit_established=False,
        interpretation="Normalized drift removes overall scalar scaling of A. Density variation alone does not establish operator drift or an advantage for learned rebuild/reuse decisions.")


def profile_coupled(report_or_path, *, pressure_fraction_threshold=0.5):
    """Summarize complete pairs, preserving failures and every repeat."""
    threshold = pressure_fraction_threshold
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("pressure_fraction_threshold must be between zero and one")
    report = (json.loads(Path(report_or_path).read_text())
              if isinstance(report_or_path, (str, Path)) else report_or_path)
    pairs = report.get("pairs", [])
    valid = bool(pairs) and report.get("success") is True and report.get("speed_claim_eligible") is True
    arms = {}
    for arm in ("classical", "H_S"):
        fractions, foam_fractions, walls, pressures = [], [], [], []
        failures = []
        for index, pair in enumerate(pairs):
            timing = pair.get("timing", {}).get(arm, {})
            wall, pressure = timing.get("wall_seconds"), timing.get("pressure_seconds")
            if (pair.get("success") is not True or timing.get("success") is not True
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and math.isfinite(v) and v > 0 for v in (wall, pressure))
                    or pressure > wall):
                failures.append(index)
                valid = False
                continue
            fractions.append(pressure / wall)
            walls.append(wall)
            pressures.append(pressure)
            foam = timing.get("foam_process_wall_seconds")
            if foam is not None:
                if (not isinstance(foam, (int, float)) or isinstance(foam, bool)
                        or not math.isfinite(foam) or not pressure <= foam <= wall):
                    failures.append(index)
                    valid = False
                else:
                    foam_fractions.append(pressure / foam)
        arms[arm] = dict(failed_repeats=failures, wall_seconds=_summary(walls),
            pressure_seconds=_summary(pressures), pressure_fraction_of_coupled_wall=_summary(fractions),
            pressure_fraction_of_foam_process=_summary(foam_fractions))
    classical = arms["classical"]
    foam = classical["pressure_fraction_of_foam_process"]
    # A missing foam-process denominator is not silently substituted by total
    # coupled wall time (which also includes Python service startup).
    observed = valid and foam["count"] == len(pairs)
    fraction = foam["median"] if observed else None
    dominant = fraction >= threshold if observed else None
    return dict(schema="cfd-benchmark-timing-profile-v1", valid_pairs=valid,
        pressure_fraction_threshold=threshold, arms=arms,
        pressure_dominant_in_instrumented_run=dominant,
        ideal_pressure_elimination_speedup_bound=1 / (1 - fraction) if observed and fraction < 1 else None,
        native_openfoam_pressure_bottleneck_established=False,
        world_model_benefit_established=False,
        scope="Pressure timer includes native extraction, socket/JSON transport, matrix admission, hierarchy setup, solve, and residual verification. CFD process wall includes output. This is an instrumented application profile, not the unmodified native solver's pressure share.",
        next_step=("Compare rebuild/reuse action costs on training/validation trajectories before enabling a World Model."
                   if dominant else "Inspect the full timing profile and operator drift before expanding this benchmark; do not infer pressure dominance from the physics label."))
