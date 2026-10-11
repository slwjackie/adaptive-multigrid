"""Diagnostic math fixtures; no CFD run or measured performance is fabricated."""
from copy import deepcopy

import numpy as np
import pytest
import scipy.sparse as sp

from adaptive_mg.v67.benchmark_profile import profile_coupled, profile_dataset
from adaptive_mg.v67.world_model.data import save_snapshot, snapshot, write_json


def recorded_fixture(root, *, change_shape=False):
    entries = []
    for i in range(4):
        diagonal = np.ones(49) * (1 + i)
        if change_shape and i >= 2:
            diagonal[0] *= 2
        s = snapshot(sp.diags(diagonal, format="csr"), np.ones(49), shape=(7, 7),
                     time=(i // 2 + 1) * .01, index=i, mesh_id="fixture",
                     boundary_id="fixture", source_kind="external_cfd",
                     context={"test_fixture_only": True, "actual_CFD": False})
        entries.append(save_snapshot(root / f"{i}.npz", s))
    write_json(root / "sequence_manifest.json", dict(schema="world-sequence-v1",
        source_kind="external_cfd", experiment_kind="variable_density_nonreacting",
        trajectories=[dict(id="fixture", case_group="fixture", split="train", snapshots=entries),
                      dict(id="unseen", case_group="unseen", split="test", snapshots=[
                          dict(path="test_must_not_be_read.npz", matrix_digest="unseen", sha256="unseen")])]))


def test_scalar_rescaling_is_not_reported_as_operator_shape_drift(tmp_path):
    recorded_fixture(tmp_path)
    report = profile_dataset(tmp_path)
    row = report["trajectories"][0]
    assert row["pressure_systems"] == 4 and row["physical_steps"] == 2
    assert row["successive_relative_frobenius"]["maximum"] > 0
    assert row["timestep_relative_frobenius"]["median"] == pytest.approx(1.)
    assert row["timestep_normalized_frobenius"]["maximum"] < 1e-14
    assert not row["normalized_time_variation_observed"]
    assert not report["test_arrays_read"]
    assert not report["world_model_benefit_established"]


def test_normalized_timestep_drift_and_corruption_detection(tmp_path):
    recorded_fixture(tmp_path, change_shape=True)
    report = profile_dataset(tmp_path)
    assert report["trajectories"][0]["normalized_time_variation_observed"]
    (tmp_path / "2.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        profile_dataset(tmp_path)


def timing_fixture():
    return dict(success=True, speed_claim_eligible=True, pairs=[dict(success=True, timing={
        "classical": dict(success=True, wall_seconds=12., foam_process_wall_seconds=10., pressure_seconds=6.),
        "H_S": dict(success=True, wall_seconds=11., foam_process_wall_seconds=9., pressure_seconds=5.)})])


def test_pressure_fraction_uses_cfd_process_and_labels_bridge_scope():
    report = profile_coupled(timing_fixture())
    assert report["pressure_dominant_in_instrumented_run"] is True
    assert report["ideal_pressure_elimination_speedup_bound"] == pytest.approx(2.5)
    assert report["arms"]["classical"]["pressure_fraction_of_coupled_wall"]["median"] == .5
    assert report["arms"]["classical"]["pressure_fraction_of_foam_process"]["median"] == .6
    assert not report["native_openfoam_pressure_bottleneck_established"]


def test_no_foam_denominator_is_not_silently_replaced():
    given = timing_fixture()
    del given["pairs"][0]["timing"]["classical"]["foam_process_wall_seconds"]
    assert profile_coupled(given)["pressure_dominant_in_instrumented_run"] is None


@pytest.mark.parametrize("problem", ["failed", "missing", "nan", "overlap", "bad_foam"])
def test_failed_or_invalid_repeat_blocks_qualification(problem):
    given = timing_fixture()
    extra = deepcopy(given["pairs"][0])
    classical = extra["timing"]["classical"]
    if problem == "failed":
        extra["success"] = False
    elif problem == "missing":
        classical.pop("pressure_seconds")
    elif problem == "nan":
        classical["pressure_seconds"] = float("nan")
    elif problem == "overlap":
        classical["pressure_seconds"] = 13.
    else:
        classical["foam_process_wall_seconds"] = 5.
    given["pairs"].append(extra)
    report = profile_coupled(given)
    assert not report["valid_pairs"]
    assert report["pressure_dominant_in_instrumented_run"] is None
    assert report["ideal_pressure_elimination_speedup_bound"] is None
