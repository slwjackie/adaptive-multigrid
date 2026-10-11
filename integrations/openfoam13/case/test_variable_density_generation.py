"""Inert case contracts and preparation routing; no native CFD claim."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load("inert_case_generator", "generate_variable_density.py")
runner = load("inert_case_runner", "run_case.py")


def args(path, *extra):
    return generator.parser().parse_args(["--output", str(path), "--case-id", "inert-train",
        "--case-group", "inert-family", "--split", "train", "--nx", "7", "--ny", "15", *extra])


def scalar(root, name):
    source = (root / "0" / name).read_text()
    return np.fromstring(source.split("nonuniform List<scalar>")[1].split("(", 1)[1].split(")", 1)[0], sep=" ")


def test_inert_generation_needs_no_cantera_and_preserves_mixture_eos(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cantera", None)
    case_args = args(tmp_path / "case")
    root = generator.generate(case_args)
    contract = json.loads((root / "contract.json").read_text())
    assert contract["schema"] == "variable-density-case-v1"
    assert contract["physics"]["reacting"] is False
    assert contract["reaction_count"] == 0
    assert contract["species"] == ["He", "N2"]
    assert contract["comparison_fields"] == ["T", "He", "N2", "rho", "U", "p"]
    assert not (root / "chemkin").exists()
    assert "combustionModel none;" in (root / "constant/combustionProperties").read_text()
    assert "chemistry off;" in (root / "constant/chemistryProperties").read_text()
    assert "hConst" in (root / "constant/physicalProperties").read_text()
    assert "Qdot" in (root / "system/controlDict").read_text()
    assert contract["foam_conversion_verified"] is False
    he, n2 = scalar(root, "He"), scalar(root, "N2")
    np.testing.assert_allclose(he + n2, 1., atol=1e-14)
    assert np.all((0 <= he) & (he <= .8))
    assert np.max(np.ptp(he.reshape(15, 7), axis=0)) > .1
    np.testing.assert_allclose(scalar(root, "T"), 300.)
    rho = np.array(generator.initial_fields(case_args)["rho"])
    rmix = generator.R_UNIVERSAL * (he / 4.002602 + n2 / 28.0134)
    np.testing.assert_allclose(rho * rmix * 300, 101325.)
    assert rho.max() / rho.min() > 4
    assert 0 < contract["initial_reference_mach"] < .1
    assert contract["n_steps"] % contract["write_interval"] == 0


def test_inert_provenance_is_deterministic_and_no_overwrite(tmp_path):
    first = generator.generate(args(tmp_path / "one"))
    second = generator.generate(args(tmp_path / "two"))
    c1, c2 = (json.loads((root / "contract.json").read_text()) for root in (first, second))
    for key in ("thermophysical_sha256", "initial_conditions_sha256", "boundary_conditions_sha256"):
        assert c1[key] == c2[key]
    assert generator.digest_files(first, c1["thermophysical_files"]) == c1["thermophysical_sha256"]
    with pytest.raises(FileExistsError):
        generator.generate(args(first))


@pytest.mark.parametrize("extra,match", [
    (("--light-mass-fraction", "0"), "light_mass_fraction"),
    (("--light-mass-fraction", "1.01"), "light_mass_fraction"),
    (("--light-mass-fraction", "nan"), "light_mass_fraction"),
    (("--fuel-speed", "100"), "Mach"),
    (("--nx", "8"), "2\\*\\*L"),
    (("--end-time", "0.00015", "--delta-t", "0.0001"), "integer multiple"),
])
def test_invalid_inert_inputs_are_rejected(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        generator.generate(args(tmp_path / "bad", *extra))


def test_prepare_inert_case_never_requires_or_calls_chemkin(tmp_path, monkeypatch):
    root = generator.generate(args(tmp_path / "case"))
    called, required = [], []

    def which(name):
        required.append(name)
        return None if name == "chemkinToFoam" else "/fake/" + name

    def fake_command(case, argv, log):
        called.append(argv[0])
        if argv[0] == "blockMesh":
            mesh = case / "constant/polyMesh"
            mesh.mkdir()
            (mesh / "points").write_text("fixture mesh")
        (case / log).write_text("Mesh OK.\n" if argv[0] == "checkMesh" else "fixture\n")

    monkeypatch.setenv("WM_PROJECT_VERSION", "13")
    monkeypatch.setattr(runner.shutil, "which", which)
    monkeypatch.setattr(runner, "command", fake_command)
    monkeypatch.setattr(sys, "argv", ["run_case.py", "--case", str(root), "--prepare-only"])
    runner.main()
    assert "chemkinToFoam" not in required
    assert called == ["blockMesh", "checkMesh"]
    contract = json.loads((root / "contract.json").read_text())
    assert contract["status"] == "prepared_not_executed"
    assert contract["native_thermophysical_verified"] is True
    assert contract["foam_conversion_verified"] is True
    assert "converted_chemistry_sha256" not in contract
    from adaptive_mg.v67.h2_fixed_p.fullrun import _verify_prepared
    assert _verify_prepared(root)["case_id"] == contract["case_id"]


def test_prepare_inert_case_rejects_modified_physics_before_commands(tmp_path, monkeypatch):
    root = generator.generate(args(tmp_path / "case"))
    (root / "constant/combustionProperties").write_text("combustionModel laminar;\n")
    monkeypatch.setenv("WM_PROJECT_VERSION", "13")
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/fake/" + name)
    monkeypatch.setattr(sys, "argv", ["run_case.py", "--case", str(root), "--prepare-only"])
    with pytest.raises(RuntimeError, match="Thermophysical input content changed"):
        runner.main()
