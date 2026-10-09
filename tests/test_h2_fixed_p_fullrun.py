"""Real OpenFOAM evidence parsers; these fixtures are not a CFD benchmark."""
from pathlib import Path
import importlib.util
import json
import shutil

import numpy as np
import pytest

from adaptive_mg.v67.h2_fixed_p.fullrun import (
    read_internal_field, parse_solver_log, collect_fields, run_paired, positive_times,
    _verify_prepared, _rewrite_bridge, _digest_files,
)
from adaptive_mg.v67.config import AdaptiveConfig


def field(path, expression):
    path.write_text("FoamFile { version 2.0; format ascii; class volScalarField; }\n"
                    "// fixture only\ninternalField " + expression + ";\nboundaryField {}\n")
    return path


def test_uniform_scalar_and_vector_fields(tmp_path):
    a = read_internal_field(field(tmp_path / "T", "uniform 300"), 4)
    b = read_internal_field(field(tmp_path / "U", "uniform (1e-2 0 -2)"), 4, 3)
    np.testing.assert_array_equal(a, [300] * 4)
    assert b.shape == (4, 3)
    np.testing.assert_array_equal(b[2], [.01, 0, -2])


def test_nonuniform_fields_and_comments(tmp_path):
    a = read_internal_field(field(tmp_path / "T", "nonuniform List<scalar> 3 (300 /*hot*/ 900 1800)"), 3)
    b = read_internal_field(field(tmp_path / "U", "nonuniform List<vector> 2 ((1 0 2) (-3 4 0))"), 2, 3)
    assert a.tolist() == [300, 900, 1800]
    assert b.tolist() == [[1, 0, 2], [-3, 4, 0]]


@pytest.mark.parametrize("expression,n,components", [
    ("uniform nan", 3, 1), ("uniform (1 2)", 3, 3),
    ("nonuniform List<scalar> 3 (1 2)", 3, 1),
    ("nonuniform List<scalar> 2 (1 2)", 3, 1),
    ("nonuniform List<vector> 2 ((1 2 3) (4 nan 6))", 2, 3),
    ("nonuniform List<vector> 2 (1 2 3 4 5 6)", 2, 3),
])
def test_field_parser_refuses_malformed_nonfinite_or_wrong_count(tmp_path, expression, n, components):
    with pytest.raises(ValueError):
        read_internal_field(field(tmp_path / "bad", expression), n, components)


def log_text():
    return """Time = 0.01
H2FixedPPressureSeconds 0.002 index 0 time 0.01 mode classical residual 1e-10 threshold 1e-8
time step continuity errors : sum local = 1e-8, global = 2e-9, cumulative = -2e-9
Time = 0.02
H2FixedPPressureSeconds 0.003 index 1 time 0.02 mode classical residual 2e-10 threshold 1e-8
time step continuity errors : sum local = 1e-8, global = 2e-9, cumulative = -4e-9
End
"""


def test_actual_log_requires_complete_time_pressure_and_continuity_evidence():
    result = parse_solver_log(log_text(), {"end_time": .02, "delta_t": .01}, 1.e-6)
    assert result["pressure_seconds"] == .005
    assert result["dt_schedule"] == [.01, .01]
    assert result["continuity"]["passed"]
    result = parse_solver_log(log_text(), {"end_time": .02, "delta_t": .01}, 1.e-10)
    assert not result["continuity"]["passed"]


@pytest.mark.parametrize("old,new", [
    ("End", ""), ("Time = 0.02", "Time = 0.03"),
    ("index 1", "index 2"), ("residual 2e-10", "residual 2e-3"),
    ("H2FixedPPressureSeconds 0.003", "H2FixedPPressureSeconds nan"),
    ("continuity errors", "other errors"),
])
def test_incomplete_or_inaccurate_log_cannot_support_claim(old, new):
    with pytest.raises(ValueError):
        parse_solver_log(log_text().replace(old, new), {"end_time": .02, "delta_t": .01}, 1.e-6)


def test_saved_fields_all_times_and_reaction_evidence(tmp_path):
    for time in ("0.01", "0.02"):
        folder = tmp_path / time
        folder.mkdir()
        for name, value in {"T": "300", "H2": ".01", "U": "(0 0 0)", "p": "101325", "Qdot": "10"}.items():
            field(folder / name, "uniform " + value)
    result = collect_fields(tmp_path, {"shape": [3, 3], "end_time": .02})
    assert result["sample_times"] == [.01, .02]
    assert result["fields"]["T"].shape == (2, 9)
    assert result["fields"]["U"].shape == (2, 9, 3)
    assert len(result["hashes"]) == 10
    assert result["reaction_evidence"]["positive_heat_release_observed"]
    with pytest.raises(ValueError, match="end time"):
        collect_fields(tmp_path, {"shape": [3, 3], "end_time": .03})


def test_duplicate_numeric_time_directories_refused(tmp_path):
    (tmp_path / "0.01").mkdir()
    (tmp_path / "1e-2").mkdir()
    with pytest.raises(ValueError, match="duplicate"):
        positive_times(tmp_path)


def test_native_absence_explicit_without_output_mutation(tmp_path, monkeypatch):
    monkeypatch.delenv("WM_PROJECT_VERSION", raising=False)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="OpenFOAM Foundation 13"):
        run_paired(tmp_path / "run", tmp_path / "case", output)
    assert not output.exists()


def test_real_generated_dictionary_rewrite_preserves_original_and_checks_hashes(tmp_path):
    """Native preparation outputs are hash fixtures, not executed CFD results."""
    pytest.importorskip("cantera", minversion="3.0")
    generator_path = Path(__file__).resolve().parents[1] / "integrations/openfoam13/case/generate.py"
    spec = importlib.util.spec_from_file_location("h2_case_for_runner_test", generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    args = generator.parser().parse_args(["--output", str(tmp_path / "original"), "--case-id", "case-test",
        "--case-group", "group-test", "--split", "validation", "--nx", "7", "--ny", "7"])
    original = generator.generate(args)
    text = (original / "system/fvSolution").read_text()
    cloned = tmp_path / "clone"
    shutil.copytree(original, cloned)
    cfg = AdaptiveConfig()
    _rewrite_bridge(cloned, "hs", Path("/tmp/test-bridge.sock"), cfg)
    changed = (cloned / "system/fvSolution").read_text()
    assert "mode hs;" in changed and 'socketPath "/tmp/test-bridge.sock";' in changed
    assert "adaptiveRtol " + repr(cfg.mg.tolerance) + ";" in changed
    assert (original / "system/fvSolution").read_text() == text
    contract = json.loads((cloned / "contract.json").read_text())
    for relative in ("constant/reactions", "constant/thermo.compressibleGas", "constant/polyMesh/faces"):
        target = cloned / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("native-preparation hash fixture; never executed")
    contract.update(status="prepared_not_executed", foam_conversion_verified=True,
        converted_chemistry_sha256=_digest_files(cloned, ["constant/reactions", "constant/thermo.compressibleGas"]),
        mesh_sha256=_digest_files(cloned, ["constant/polyMesh/faces"]))
    (cloned / "contract.json").write_text(json.dumps(contract))
    assert _verify_prepared(cloned)["case_id"] == "case-test"
    (cloned / "constant/polyMesh/faces").write_text("tampered fixture")
    with pytest.raises(ValueError, match="mesh_sha256"):
        _verify_prepared(cloned)
