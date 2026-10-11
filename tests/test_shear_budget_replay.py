"""Budget-only replay integrity; numerical fixtures are not real CFD evidence."""
import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import zipfile

import pytest

from test_h2_fixed_p_study import make_fixture_recording, settings
from test_smoother_reuse import reuse_settings
from adaptive_mg.v67.h2_fixed_p import study as base
from adaptive_mg.v67.h2_fixed_p.data import read
from adaptive_mg.v67.world_model.data import file_hash, write_json

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("budget_replay", REPO / "scripts/run_v6_7_shear_budget_replay.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def trained(tmp_path):
    config = settings()
    config["smoother_reuse"] = reuse_settings()
    path = tmp_path / "config.json"
    write_json(path, config)
    inputs = [make_fixture_recording(tmp_path / split, split, offset) for split, offset in
              (("train", 0), ("tune", 7), ("validation", 15), ("test", 30))]
    source = tmp_path / "source"
    # Real source digest, data validation and actual optimizer evidence.
    base.prepare(path, inputs, source)
    base.train(source)
    return source


def args(source, output, stop="probe"):
    return argparse.Namespace(run_dir=source, output=output, max_cycles=600,
                              repeats=2, stop_after=stop)


def hashes(root):
    return {str(path.relative_to(root)): file_hash(path)
            for path in root.rglob("*") if path.is_file()}


def test_real_replay_preserves_checkpoint_inputs_tolerances_and_runs_probe(trained, tmp_path, monkeypatch):
    original = hashes(trained)
    monkeypatch.setattr(base, "train", lambda *a: pytest.fail("retraining forbidden"))
    real_load = runner.load_trajectories
    splits = []

    def load(root, split):
        assert split in ("train", "tune")
        splits.append(split)
        return real_load(root, split)

    monkeypatch.setattr(runner, "load_trajectories", load)
    target = tmp_path / "budget600"
    summary = runner.run(args(trained, target))
    assert summary["status"] == "complete" and summary["completed_through"] == "probe"
    assert summary["comparison_passed"] and summary["probe"]["failed_rollout_measurements"] == 0
    assert not summary["world_model_benefit_established"]
    assert splits == ["train", "tune", "train", "tune"]
    assert hashes(trained) == original
    manifest = read(target / "manifest.json")
    assert manifest["original_contract"]["expert_sha256"] == file_hash(trained / "expert/candidate.pt")
    before, after = manifest["original_solver"], manifest["effective_solver"]
    assert after["mg"]["max_cycles"] == 600
    after["mg"]["max_cycles"] = before["mg"]["max_cycles"]
    assert after == before
    comparison = read(target / "comparison.json")
    for case in comparison["cases"]:
        for arm in case["policies"].values():
            assert arm["success"] and arm["repeats"] == 2
            for phases in arm["phase_totals_by_repeat"]:
                assert phases["total_seconds"] == pytest.approx(
                    sum(value for key, value in phases.items() if key != "total_seconds"))
    with zipfile.ZipFile(target / "budget_diagnostics.zip") as archive:
        assert {"summary.json", "manifest.json", "comparison.json", "probe/collection.json"} <= set(archive.namelist())
        assert all(name.endswith(".json") for name in archive.namelist())
    with pytest.raises(FileExistsError):
        runner.run(args(trained, target))


def test_failed_numerical_comparison_blocks_probe_and_retains_diagnostics(trained, tmp_path, monkeypatch):
    real_run = runner.reuse.run_episode

    def fail(states, cfg, expert, policy):
        result = real_run(states, cfg, expert, policy)
        if policy == "reuse":
            result["success"] = False
            result["rows"][0]["success"] = False
        return result

    monkeypatch.setattr(runner.reuse, "run_episode", fail)
    monkeypatch.setattr(runner, "collect_probe", lambda *a: pytest.fail("probe after failure"))
    target = tmp_path / "failed"
    summary = runner.run(args(trained, target))
    assert summary["status"] == "comparison_failed"
    assert summary["comparison"]["reuse"]["replay_speedup_vs_classical"] is None
    assert not (target / "probe").exists()
    assert (target / "comparison.json").is_file() and (target / "budget_diagnostics.zip").is_file()


def test_replay_rejects_changed_checkpoint_and_nested_output(trained, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        runner.run(args(trained, trained / "budget600"))
    assert not (trained / "budget600").exists()
    path = trained / "expert/candidate.pt"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checkpoint"):
        runner.run(args(trained, tmp_path / "bad_checkpoint"))
    assert not (tmp_path / "bad_checkpoint").exists()


def episode(refresh, reuse):
    def rows(cost):
        return [dict(total_seconds=cost, success=True, pressure_calls=4) for _ in range(3)]
    measurements = []
    for i in range(24):
        measurements.append({"0": rows(1. if i in refresh else 2.),
                             "1": rows(1. if i in reuse else 2.)})
    return dict(trajectory="fixture", probe_horizon=4, available=[[True, True]] * 24,
                horizon_measurements=measurements)


def test_overlapping_wins_cannot_establish_candidate_and_failures_are_retained():
    ep = episode(range(3), range(4, 7))
    report = runner.probe_report([ep], .05)
    assert report["refresh_wins"] == report["reuse_wins"] == 3
    assert report["nonoverlapping_wins"] == dict(refresh_wins=1, reuse_wins=1)
    assert not report["world_model_candidate"]
    ep = episode(range(12), range(12, 24))
    report = runner.probe_report([ep], .05)
    assert report["world_model_candidate"]
    assert report["nonoverlapping_wins"] == dict(refresh_wins=3, reuse_wins=3)
    assert not report["world_model_benefit_established"]
    broken = deepcopy(ep)
    broken["available"][0] = [True, False]
    broken["horizon_measurements"][0] = {"0": [dict(total_seconds=1., success=False, pressure_calls=4)]}
    assert runner.probe_report([broken], .05)["failed_rollout_measurements"] == 1
    assert not runner.probe_report([broken], .05)["world_model_candidate"]
