"""Runner log regressions from target Foundation 13 output; no native CFD run."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("h2_case_runner", Path(__file__).with_name("run_case.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("suffix", ["", "s", " s"])
def test_completed_physical_time_ignores_execution_time(suffix):
    log = (f"Time = 1.9e-05{suffix}\n"
           "ExecutionTime = 0.338762 s  ClockTime = 0 s\n"
           f"Time = 2e-05{suffix}\n"
           "ExecutionTime = 0.359528 s  ClockTime = 0 s\n\nEnd\n")
    assert runner.completed_time(log) == 2e-5


@pytest.mark.parametrize("log", ["Time = 2e-05s\n", "Time = 2e-05s\nFOAM FATAL ERROR\nEnd\n"])
def test_incomplete_or_fatal_log_cannot_pass(log):
    with pytest.raises(RuntimeError, match="normal completion"):
        runner.completed_time(log)


def test_unknown_time_units_do_not_count_as_seconds():
    assert runner.completed_time("Time = 2e-05ms\nEnd\n") == 0.
