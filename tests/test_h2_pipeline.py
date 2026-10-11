"""Runner orchestration and rejection checks; these are NOT CFD simulations."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_v6_7_h2_pipeline.py"
spec = importlib.util.spec_from_file_location("h2_pipeline", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def native_trace(tmp_path):
    record, case = tmp_path / "record", tmp_path / "case"
    record.mkdir(); case.mkdir()
    contract = dict(case_id="preflight", shape=[7, 7], delta_t=1e-6, end_time=2e-6)
    entries = [dict(index=i, time=(i // 2 + 1) * 1e-6) for i in range(4)]
    rows = [dict(**entry, mode="native", success=True, error="", matrix_digest="a" * 64,
                 final_residual=1e-14, threshold=1e-12) for entry in entries]
    runner.write(record / "case_contract.json", contract)
    runner.write(record / "ldu_sequence.json", dict(trajectories=[dict(snapshots=entries)]))
    evidence = tmp_path / "service.jsonl"
    evidence.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (case / "log.foamRun").write_text("Time = 1e-06s\nTime = 2e-06s\nEnd\n")
    return record, evidence, case


def test_all_native_events_checked_with_foundation_seconds_suffix(native_trace):
    result = runner.check_native_recording(*native_trace)
    assert result["snapshots"] == 4
    assert result["end_time"] == 2e-6


@pytest.mark.parametrize("corruption", ["missing", "failed", "residual", "time", "index", "nan"])
def test_rejects_incomplete_or_failed_native_recording(native_trace, corruption):
    record, evidence, case = native_trace
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    if corruption == "missing":
        rows.pop(0)
    elif corruption == "failed":
        rows[0]["success"] = False
    elif corruption == "residual":
        rows[0]["final_residual"] = 1e-6
    elif corruption == "time":
        rows[0]["time"] = 0
    elif corruption == "index":
        rows[0]["index"] = 9
    else:
        rows[0]["final_residual"] = float("nan")
    evidence.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        runner.check_native_recording(record, evidence, case)


def test_rejects_foam_failure_even_with_successful_service_records(native_trace):
    record, evidence, case = native_trace
    (case / "log.foamRun").write_text("Time = 2e-06s\nFOAM FATAL\n")
    with pytest.raises(RuntimeError, match="normal completion"):
        runner.check_native_recording(record, evidence, case)


def test_nonzero_child_retains_diagnostic_log(tmp_path):
    log = tmp_path / "failed.log"
    with pytest.raises(RuntimeError, match="exited 7"):
        runner.run_logged([sys.executable, "-c", "print('diagnostic'); raise SystemExit(7)"], log, 10)
    assert "diagnostic" in log.read_text()


def test_timed_out_child_is_reaped(tmp_path, monkeypatch):
    created = []
    original = runner.subprocess.Popen

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_logged([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "timeout.log", .1)
    assert created[0].poll() is not None


def test_native_foam_failure_stops_its_service_and_preserves_logs(tmp_path, monkeypatch):
    args = argparse.Namespace(config=tmp_path / "config.json", n=31, end_time=2e-4,
                              delta_t=1e-6, timeout=10)
    (tmp_path / "logs").mkdir()
    stopped = []

    def prepare(argv, log, timeout):
        if "--output" in argv:
            target = Path(argv[argv.index("--output") + 1])
            target.mkdir(parents=True)
            runner.write(target / "contract.json", {})

    class Process:
        def __init__(self, argv, **kwargs):
            self.is_foam = argv[0] == "foamRun"
            self.returncode = None
            kwargs["stdout"].write("CFD FAILURE\n" if self.is_foam else "READY socket\n")
            kwargs["stdout"].flush()

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 9 if self.is_foam else 0
            return self.returncode

    monkeypatch.setattr(runner, "run_logged", prepare)
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "stop", lambda process: stopped.append(process))
    with pytest.raises(RuntimeError, match="foamRun exited 9"):
        runner.collect(args, tmp_path, "train_300", "train", 300)
    assert len(stopped) == 2 and stopped[0].is_foam and not stopped[1].is_foam
    assert "CFD FAILURE" in (tmp_path / "record_cases/train_300/log.foamRun").read_text()
    assert (tmp_path / "cases/train_300/contract.json").exists()


def test_preflight_really_imports_arrays_and_rejects_corrupt_payload(tmp_path):
    import numpy as np
    import scipy.sparse as sp
    from adaptive_mg.v67.world_model.adapter import FinalizedLduRecorder
    from adaptive_mg.v67.world_model.data import digest
    physics = dict(fuel="H2", oxidizer="air", dimension=2, fixed_grid=True)
    contract = dict(schema="h2-fixed-p-case-v1", case_id="fixture", case_group="fixture_only",
                    split="train", shape=[7, 7], chemistry_sha256="0" * 64, physics=physics,
                    delta_t=1e-6, end_time=1e-6, test_fixture_only=True, actual_CFD=False)
    record = tmp_path / "record"
    recorder = FinalizedLduRecorder(record, producer={"test_fixture_only": True, "actual_CFD": False}, physics=physics)
    runner.write(record / "case_contract.json", contract)
    a = sp.diags([-np.ones(48), 4 * np.ones(49), -np.ones(48)], [-1, 0, 1], format="csr")
    upper = sp.triu(a, k=1).tocoo()
    probes = np.column_stack((np.ones(49), np.linspace(-1, 1, 49)))
    rows = []
    for index in range(2):
        recorder.record(trajectory="fixture", case_group="fixture_only", split="train", index=index,
                        time=1e-6, shape=(7, 7), mesh_id="fixture", boundary_id="anchored",
                        diag=a.diagonal(), lower_addr=upper.row, upper_addr=upper.col,
                        lower=upper.data, upper=upper.data, b=np.ones(49), x0=np.zeros(49),
                        probe_vectors=probes, probe_products=a @ probes, structured_to_native=np.arange(49),
                        boundary_finalized=True, nullspace="none", coupled_interfaces=0,
                        context=dict(case_contract_digest=digest(contract), physics=physics, test_fixture_only=True))
        rows.append(dict(index=index, time=1e-6, mode="native", success=True, error="",
                         matrix_digest="a" * 64, final_residual=1e-14, threshold=1e-12))
    evidence = tmp_path / "service.jsonl"
    evidence.write_text("".join(json.dumps(row) + "\n" for row in rows))
    case = tmp_path / "case"
    case.mkdir()
    (case / "log.foamRun").write_text("Time = 1e-06s\nEnd\n")
    args = argparse.Namespace(preflight_recording=record, preflight_evidence=evidence, preflight_case=case)
    root = tmp_path / "good"
    root.mkdir()
    runner.verify_preflight(args, root)
    report = runner.read(root / "preflight_report.json")
    assert report["matrix_shape"] == [49, 49] and report["b_shape"] == [49]
    assert (root / "preflight_verified/sequence_manifest.json").exists()
    entry = runner.read(record / "ldu_sequence.json")["trajectories"][0]["snapshots"][0]
    (record / entry["path"]).write_bytes(b"corrupt recording")
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(ValueError, match="recording changed"):
        runner.verify_preflight(args, bad)


def fake_pipeline(tmp_path, monkeypatch, failed=None, stop_after="test"):
    args = argparse.Namespace(output=tmp_path / "experiment", config=runner.REPO / "configs/v6_7_h2_fixed_p.json",
                              n=31, end_time=2e-4, delta_t=1e-6, repeats=3, timeout=60, stop_after=stop_after)
    calls = []
    monkeypatch.setattr(runner, "verify_preflight", lambda *args: None)
    monkeypatch.setattr(runner, "collect", lambda args, root, name, split, temperature: root / "recordings" / name)

    def fake_command(argv, log, timeout):
        command = argv[2]
        split = argv[argv.index("--split") + 1] if "--split" in argv else None
        calls.append((command, split))
        if command == "train":
            target = args.output / "study/expert/status.json"
            value = dict(status="complete", optimizer_updates=200)
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

    monkeypatch.setattr(runner, "run_logged", fake_command)
    return args, calls


def test_freeze_precedes_test_and_slowdowns_are_preserved(tmp_path, monkeypatch):
    args, calls = fake_pipeline(tmp_path, monkeypatch)
    summary = runner.pipeline(args)
    assert calls == [("prepare", None), ("train", None), ("evaluate", "validation"),
                     ("coupled", None), ("freeze", None), ("evaluate", "test"), ("coupled", None)]
    assert summary["test_full_cfd_speedup"] == .95
    assert summary["test_replay_speedup"] == .8
    assert runner.read(args.output / "pipeline_status.json")["status"] == "complete"
    with pytest.raises(FileExistsError):
        runner.pipeline(args)


@pytest.mark.parametrize("failed", ["replay", "coupled"])
def test_failed_validation_stops_before_freeze_and_test(tmp_path, monkeypatch, failed):
    args, calls = fake_pipeline(tmp_path, monkeypatch, failed=failed)
    with pytest.raises(RuntimeError, match="did not pass"):
        runner.pipeline(args)
    assert ("freeze", None) not in calls
    assert ("evaluate", "test") not in calls
    status = runner.read(args.output / "pipeline_status.json")
    assert status["status"] == "failed" and status["results_preserved"]


def test_optional_validation_stop_keeps_test_sealed(tmp_path, monkeypatch):
    args, calls = fake_pipeline(tmp_path, monkeypatch, stop_after="validation")
    summary = runner.pipeline(args)
    assert summary["completed_through"] == "validation"
    assert ("freeze", None) not in calls and ("evaluate", "test") not in calls
