"""Orchestration checks only: mocked commands do not simulate OpenFOAM physics."""
import argparse
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_v6_7_variable_density_pipeline.py"
spec = importlib.util.spec_from_file_location("variable_density_pipeline", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fake_pipeline(tmp_path, monkeypatch, *, failed=None, stop_after="test"):
    from adaptive_mg.v67 import benchmark_profile
    args = argparse.Namespace(output=tmp_path / "primary", config=runner.REPO / "configs/v6_7_variable_density_fixed_p.json",
                              n=31, end_time=.002, delta_t=1e-5, repeats=3, timeout=60, stop_after=stop_after)
    calls, collections, profiles = [], [], []

    def collect(args, root, case):
        collections.append(case)
        return root / "recordings" / case["name"]

    def profile_dataset(root, *, splits):
        profiles.append(tuple(splits))
        return dict(test_arrays_read="test" in splits, trajectories=[], world_model_benefit_established=False)

    def profile_coupled(report_path, *, pressure_fraction_threshold):
        value = runner.read(report_path)
        return dict(valid_pairs=value["success"], pressure_fraction_threshold=pressure_fraction_threshold,
                    pressure_dominant_in_instrumented_run=False)

    def command(argv, log, timeout):
        command = argv[2]
        split = argv[argv.index("--split") + 1] if "--split" in argv else None
        calls.append((command, split))
        if command == "train":
            target = args.output / "study/expert/status.json"
            value = dict(status="incomplete" if failed == "training" else "complete", optimizer_updates=200)
        elif command == "evaluate":
            target = args.output / "study" / split / "replay.json"
            value = dict(success=failed != "replay", speed_claim_eligible=True, speedup=.8)
        elif command == "coupled":
            target = Path(argv[argv.index("--output") + 1]) / "report.json"
            value = dict(success=failed != "coupled", speed_claim_eligible=True,
                         pressure_speedup=.9, full_cfd_speedup=.95)
        else:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        runner.write(target, value)

    monkeypatch.setattr(runner, "collect", collect)
    monkeypatch.setattr(runner, "run_logged", command)
    monkeypatch.setattr(benchmark_profile, "profile_dataset", profile_dataset)
    monkeypatch.setattr(benchmark_profile, "profile_coupled", profile_coupled)
    return args, calls, collections, profiles


def test_primary_collect_needs_no_h2_preflight_and_stops_before_training(tmp_path, monkeypatch):
    args, calls, cases, profiles = fake_pipeline(tmp_path, monkeypatch, stop_after="collect")
    summary = runner.pipeline(args)
    assert calls == [("prepare", None)]
    assert profiles == [("train", "validation")]
    assert summary["completed_through"] == "collect" and summary["benchmark_role"] == "primary"
    assert not summary["dataset_profile"]["test_arrays_read"] and not summary["world_model_enabled"]
    assert len(cases) == 4 and len({case["case_group"] for case in cases}) == 4
    assert {case["inlet_temperature"] for case in cases} == {300}
    assert [case["split"] for case in cases] == ["train", "train", "validation", "test"]
    plan = runner.read(args.output / "pipeline_plan.json")
    assert plan["experiment_kind"] == "variable_density_nonreacting"
    assert plan["cases"] == list(runner.CASES)
    assert plan["generator_sha256"] and plan["shared_runner_sha256"]


def test_primary_freezes_before_test_and_preserves_real_slowdowns(tmp_path, monkeypatch):
    args, calls, _, profiles = fake_pipeline(tmp_path, monkeypatch)
    summary = runner.pipeline(args)
    assert calls == [("prepare", None), ("train", None), ("evaluate", "validation"),
                     ("coupled", None), ("freeze", None), ("evaluate", "test"), ("coupled", None)]
    assert profiles == [("train", "validation")]
    assert summary["test_replay_speedup"] == .8 and summary["test_full_cfd_speedup"] == .95
    assert summary["validation_benchmark_profile"]["pressure_dominant_in_instrumented_run"] is False
    assert runner.read(args.output / "coupled_test/benchmark_profile.json")["valid_pairs"]
    assert runner.read(args.output / "pipeline_status.json")["status"] == "complete"
    with pytest.raises(FileExistsError):
        runner.pipeline(args)


@pytest.mark.parametrize("failed", ["training", "replay", "coupled"])
def test_failed_primary_validation_preserves_evidence_and_seals_test(tmp_path, monkeypatch, failed):
    args, calls, _, _ = fake_pipeline(tmp_path, monkeypatch, failed=failed)
    with pytest.raises(RuntimeError):
        runner.pipeline(args)
    assert ("freeze", None) not in calls and ("evaluate", "test") not in calls
    status = runner.read(args.output / "pipeline_status.json")
    assert status["status"] == "failed" and status["results_preserved"]
    assert status["experiment_kind"] == "variable_density_nonreacting"
    if failed == "coupled":
        assert not runner.read(args.output / "coupled_validation/benchmark_profile.json")["valid_pairs"]


def test_primary_validation_stop_does_not_freeze_or_evaluate_test(tmp_path, monkeypatch):
    args, calls, _, _ = fake_pipeline(tmp_path, monkeypatch, stop_after="validation")
    assert runner.pipeline(args)["completed_through"] == "validation"
    assert ("freeze", None) not in calls and ("evaluate", "test") not in calls


@pytest.mark.parametrize("physical_pass", [True, False])
def test_native_primary_collection_checks_and_preserves_actual_field_evidence(tmp_path, monkeypatch, physical_pass):
    from adaptive_mg.v67.h2_fixed_p import fullrun
    calls = []
    case = runner.CASES[0]
    prepared = tmp_path / "cases" / case["name"]
    prepared.mkdir(parents=True)
    runner.write(prepared / "contract.json", dict(experiment_kind="variable_density_nonreacting"))
    recording = tmp_path / "recordings" / case["name"]
    recording.mkdir(parents=True)

    def collect(*args, **kwargs):
        calls.append(kwargs)
        return recording

    def fields(root, contract):
        assert root == tmp_path / "record_cases" / case["name"]
        assert contract["experiment_kind"] == "variable_density_nonreacting"
        return dict(sample_times=[.001, .002], hashes={"0.002/rho": "a" * 64},
                    fields={"rho": "not serialized"}, physics_evidence={"passed": physical_pass})

    monkeypatch.setattr(runner.shared, "collect", collect)
    monkeypatch.setattr(fullrun, "collect_fields", fields)
    if physical_pass:
        assert runner.collect(object(), tmp_path, case) == recording
    else:
        with pytest.raises(RuntimeError, match="field checks failed"):
            runner.collect(object(), tmp_path, case)
    profile = runner.read(recording / "physics_profile.json")
    assert profile["physics_evidence"]["passed"] is physical_pass
    assert "fields" not in profile
    assert calls[0]["generator"] == runner.GENERATOR
    assert calls[0]["case_group"] == case["case_group"]
    options = calls[0]["generator_options"]
    assert float(options[options.index("--light-mass-fraction") + 1]) == .65


def test_primary_cli_has_no_chemkin_or_previous_recording_dependency(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setenv("WM_PROJECT_VERSION", "13")
    monkeypatch.setattr(runner.shutil, "which", lambda command: commands.append(command) or "/mock/" + command)
    monkeypatch.setattr(runner, "pipeline", lambda args: args)
    args = runner.main(["--output", str(tmp_path / "new"), "--stop-after", "collect"])
    assert commands == ["foamRun", "blockMesh", "checkMesh"]
    assert not hasattr(args, "preflight_recording")
    assert args.n == 31 and args.end_time == .002 and args.delta_t == 1e-5


def test_primary_cli_rejects_h2_configuration(tmp_path):
    with pytest.raises(SystemExit) as raised:
        runner.main(["--output", str(tmp_path / "new"), "--config", str(runner.REPO / "configs/v6_7_h2_fixed_p.json")])
    assert raised.value.code == 2


@pytest.mark.parametrize("case", runner.CASES, ids=lambda case: case["name"])
def test_every_pipeline_case_passes_real_generator_contract(tmp_path, monkeypatch, case):
    """Exercise the actual generated command and files, stopping before OpenFOAM."""
    from adaptive_mg.v67.h2_fixed_p.data import experiment_kind, validate_case_contract
    spec = importlib.util.spec_from_file_location("real_primary_case_generator", runner.GENERATOR)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    args = argparse.Namespace(config=runner.REPO / "configs/v6_7_variable_density_fixed_p.json",
                              n=31, end_time=.002, delta_t=1e-5, timeout=60)
    (tmp_path / "logs").mkdir()
    validated = []

    class BeforeOpenFoam(Exception):
        pass

    def generate_or_stop(argv, log, timeout):
        if Path(argv[1]) == runner.GENERATOR:
            parsed = generator.parser().parse_args([str(value) for value in argv[2:]])
            generator.validate(parsed)
            output = generator.generate(parsed)
            validated.append(runner.read(output / "contract.json"))
        else:
            assert "--prepare-only" in argv
            raise BeforeOpenFoam

    monkeypatch.setattr(runner.shared, "run_logged", generate_or_stop)
    with pytest.raises(BeforeOpenFoam):
        runner.collect(args, tmp_path, case)
    assert len(validated) == 1
    contract = validate_case_contract(validated[0])
    assert contract["case_id"] == case["name"] and contract["case_group"] == case["case_group"]
    assert contract["split"] == case["split"]
    assert experiment_kind(contract) == "variable_density_nonreacting"
    assert contract["n_steps"] == 200 and contract["n_steps"] // contract["write_interval"] >= 2


def test_primary_cli_rejects_single_saved_timestep(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        runner.main(["--output", str(tmp_path / "new"), "--end-time", "1e-5", "--delta-t", "1e-5"])
    assert raised.value.code == 2
    assert "at least two timesteps" in capsys.readouterr().err
