"""Chemistry/physical initialization tests; no test claims to execute OpenFOAM."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("h2_case_generator", Path(__file__).with_name("generate.py"))
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def args(path, *extra):
    return generator.parser().parse_args(["--output", str(path), "--case-id", "test-case",
                                         "--case-group", "test-group", "--split", "train",
                                         "--nx", "7", "--ny", "15", *extra])


def read_scalar(root, name):
    text = (root / "0" / name).read_text()
    return np.fromstring(text.split("nonuniform List<scalar>")[1].split("(", 1)[1].split(")", 1)[0], sep=" ")


def test_chemkin_roundtrip_preserves_hydrogen_rates(tmp_path):
    ct = pytest.importorskip("cantera", minversion="3.0")
    from cantera import ck2yaml
    root = generator.generate(args(tmp_path / "case"))
    ck2yaml.convert(str(root / "chemkin/h2.inp"), str(root / "chemkin/thermo.dat"),
                    str(root / "chemkin/transport.dat"), out_name=str(tmp_path / "roundtrip.yaml"), quiet=True)
    original, restored = ct.Solution("h2o2.yaml"), ct.Solution(str(tmp_path / "roundtrip.yaml"))
    assert original.species_names == restored.species_names
    assert original.n_reactions == restored.n_reactions == 29
    for temperature in (300, 1000, 1800, 2500):
        for pressure in (101325, 506625):
            for gas in (original, restored):
                gas.TPX = temperature, pressure, "H2:2,O2:1,N2:3.76,H2O:0.2"
            np.testing.assert_allclose(restored.forward_rate_constants, original.forward_rate_constants, rtol=1e-6)
            np.testing.assert_allclose(restored.standard_enthalpies_RT, original.standard_enthalpies_RT, rtol=1e-6)


def test_initialization_is_mass_conserving_and_actually_two_dimensional(tmp_path):
    pytest.importorskip("cantera", minversion="3.0")
    root = generator.generate(args(tmp_path / "case", "--fuel-h2-mole-fraction", "0.7"))
    contract = json.loads((root / "contract.json").read_text())
    fractions = np.stack([read_scalar(root, name) for name in contract["species"]])
    assert fractions.shape == (10, 7 * 15)
    assert np.all(fractions >= 0)
    np.testing.assert_allclose(fractions.sum(axis=0), 1, atol=1e-14)
    temperature = read_scalar(root, "T").reshape(15, 7)
    assert np.max(abs(temperature[7] - temperature[0])) > 100
    assert np.max(abs(temperature[:, 3] - temperature[:, 0])) > 100
    assert 0 < contract["initial_hotspot_x_m"] < contract["geometry_m"][0]
    assert contract["status"] == "generated_not_executed"
    assert contract["foam_conversion_verified"] is False
    assert "CH4" not in contract["species"]


def test_generation_has_stable_inputs_and_rejects_overwrite(tmp_path):
    pytest.importorskip("cantera", minversion="3.0")
    first = generator.generate(args(tmp_path / "first"))
    second = generator.generate(args(tmp_path / "second"))
    c1 = json.loads((first / "contract.json").read_text())
    c2 = json.loads((second / "contract.json").read_text())
    for key in ("chemistry_sha256", "initial_conditions_sha256", "boundary_conditions_sha256"):
        assert c1[key] == c2[key]
    with pytest.raises(FileExistsError):
        generator.generate(args(first))
    with pytest.raises(ValueError, match="2\\*\\*L"):
        generator.validate(args(tmp_path / "bad", "--nx", "16"))
    with pytest.raises(ValueError, match="case_id"):
        generator.validate(args(tmp_path / "bad", "--case-id", "bad/path"))


def test_final_time_is_always_a_field_write_and_fractional_steps_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="integer multiple"):
        generator.validate(args(tmp_path / "fractional", "--delta-t", "0.000001", "--end-time", "0.0000215"))
    pytest.importorskip("cantera", minversion="3.0")
    root = generator.generate(args(tmp_path / "21-steps", "--delta-t", "0.000001", "--end-time", "0.000021"))
    control = (root / "system/controlDict").read_text()
    interval = int(re.search(r"writeInterval\s+(\d+)\s*;", control).group(1))
    contract = json.loads((root / "contract.json").read_text())
    assert contract["n_steps"] == 21
    assert interval == contract["write_interval"]
    assert interval > 0 and 21 % interval == 0
