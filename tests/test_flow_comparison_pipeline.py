"""Study design/provenance gates; native CFD execution is never fabricated."""
import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from adaptive_mg.v67.world_model.data import digest, file_hash, write_json

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("flow_comparison", REPO/"scripts/run_v6_7_flow_comparison.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def arguments(tmp_path, *extras):
    return runner.parser().parse_args(["--output", str(tmp_path/"run"), *extras])


def plan(tmp_path, *extras):
    args = arguments(tmp_path, *extras)
    return runner.build_plan(args, runner.read(args.config))


def test_composition_train_tune_and_legacy_control_are_explicitly_separate():
    forced = runner.cases_for("shear_composition")
    control = runner.cases_for("shear_control")
    assert all(c["composition"] > 0 for c in forced if c["split"] in ("train", "tune"))
    assert all(c["composition"] == 0 for c in control if c["split"] in ("train", "tune"))
    assert {c["group"] for c in forced if c["split"] == "validation"} == {"shear_validation"}
    assert any(c["composition"] == 0 and c["amplitude"] > 0 for c in forced if c["split"] == "validation")
    for cases in (forced, control, runner.cases_for("h2_crossflow"), runner.cases_for("lox_he_sloshing")):
        groups = {split: {c["group"] for c in cases if c["split"] == split}
                  for split in ("train", "tune", "validation", "test")}
        assert all(groups[a].isdisjoint(groups[b]) for a in groups for b in groups if a != b)


def test_design_controls_levels_budget_tolerance_and_world_factorial(tmp_path):
    value = plan(tmp_path)
    assert value["smoother_arms"] == [
        dict(id="shared_l0", architecture="shared", levels=[0]),
        dict(id="shared_l01", architecture="shared", levels=[0, 1]),
        dict(id="per_level_l01", architecture="per_level", levels=[0, 1])]
    assert len(value["world_variants"]) == 6
    assert len(plan(tmp_path, "--independent-grus")["world_variants"]) == 9
    assert all(v["world_memory"] == "shared" for v in value["world_variants"])
    from adaptive_mg.v67.h2_fixed_p.study import configuration
    for family, data in value["families"].items():
        for arm, settings in data["configs"].items():
            path = tmp_path/(family+arm+".json")
            write_json(path, settings)
            _, cfg = configuration(path)
            assert cfg.mg.max_cycles == 600
            assert cfg.mg.tolerance == 1e-8 and cfg.mg.absolute_tolerance == 1e-12
            assert cfg.smoother_levels == ((0,) if arm == "shared_l0" else (0, 1))
    assert value["families"]["h2_crossflow"]["timing"]["delta_t"] == 2e-5
    with pytest.raises(ValueError, match="n="):
        plan(tmp_path, "--n", "0")


@pytest.mark.parametrize("family", runner.FAMILIES)
def test_all_declared_native_cases_generate_and_validate(tmp_path, family):
    generator = runner.module("_generator_"+family, runner.generator_for(family))
    value = plan(tmp_path, "--families", family)
    timing = value["families"][family]["timing"]
    from adaptive_mg.v67.h2_fixed_p.data import validate_case_contract
    for case in value["families"][family]["cases"]:
        args = generator.parser().parse_args([
            "--output", str(tmp_path/family/case["name"]), "--case-id", case["name"],
            "--case-group", case["group"], "--split", case["split"], "--nx", "31", "--ny", "31",
            "--end-time", str(timing["end_time"]), "--delta-t", str(timing["delta_t"]),
            "--mode", "native", *runner.generator_options(family, case)])
        output = generator.generate(args)
        contract = validate_case_contract(runner.read(output/"contract.json"))
        assert contract["split"] == case["split"]
        assert contract["physics"]["reacting"] is False


def test_dry_run_requires_no_openfoam_and_resume_rejects_mutation(tmp_path, monkeypatch):
    monkeypatch.delenv("WM_PROJECT_VERSION", raising=False)
    result = runner.main(["--output", str(tmp_path/"run"), "--dry-run"])
    assert result["completed_through"] == "plan"
    assert not (tmp_path/"run/families").exists()
    args = arguments(tmp_path, "--resume")
    assert runner.check_plan(args.output, args)["test_selection_forbidden"]
    with pytest.raises(ValueError, match="independent GRUs"):
        runner.check_plan(args.output, arguments(tmp_path, "--resume", "--independent-grus"))
    with pytest.raises(ValueError, match="legacy control"):
        runner.check_plan(args.output, arguments(tmp_path, "--resume", "--include-legacy-control"))
    first = next((args.output/"configurations").glob("*.json"))
    settings = runner.read(first)
    settings["solver"]["mg"]["tolerance"] = 1e-4
    runner.write(first, settings)
    with pytest.raises(ValueError, match="configuration changed"):
        runner.check_plan(args.output, args)


def test_missing_openfoam_never_generates_a_surrogate(tmp_path, monkeypatch):
    monkeypatch.delenv("WM_PROJECT_VERSION", raising=False)
    with pytest.raises(RuntimeError, match="no synthetic"):
        runner.main(["--output", str(tmp_path/"run"), "--stop-after", "collect"])
    assert runner.read(tmp_path/"run/status.json")["status"] == "failed"
    assert not (tmp_path/"run/families").exists()


def clone_fixture(tmp_path, monkeypatch):
    from adaptive_mg.v67.h2_fixed_p import study as base
    from adaptive_mg.v67.smoother_reuse import study as reuse
    parent, target = tmp_path/"parent", tmp_path/"variant"
    settings = plan(tmp_path)["families"]["shear_composition"]["configs"]["shared_l01"]
    manifest = dict(configuration_digest=digest(settings), dataset_sha256="same-real-recording", source_digest="source")
    contract = dict(configuration_digest=digest(settings), settings_digest=digest(settings["smoother_reuse"]),
                    dataset_sha256="same-real-recording", expert_sha256="expert", source_digest="source")
    write_json(parent/"configuration.json", settings)
    write_json(parent/"run_manifest.json", manifest)
    write_json(parent/"expert/candidate.pt", {"actual_parent_weights": True})
    write_json(parent/"data/sequence_manifest.json", {"trajectories": []})
    write_json(parent/"reuse/episodes/train/a.json", {"split": "train", "measured": True})
    write_json(parent/"reuse/training.json", {"selected": "reuse", "models": {}})
    episode = "episodes/train/a.json"
    write_json(parent/"reuse/collection.json", dict(contract=contract,
        episodes={episode:file_hash(parent/"reuse"/episode)}, probe={"world_model_candidate": True}))
    monkeypatch.setattr(base, "load_run", lambda out: (out, runner.read(out/"configuration.json"), None,
                                                        runner.read(out/"run_manifest.json")))
    monkeypatch.setattr(reuse, "load", lambda out: (out, None, None, None,
                                                    runner.read(out/"reuse/collection.json")["contract"]))
    return parent, target


def test_world_clone_reuses_exact_labels_weights_and_pins_parent(tmp_path, monkeypatch):
    parent, target = clone_fixture(tmp_path, monkeypatch)
    variant = dict(hidden=32, world_memory="shared", world_head="role_mlp", world_head_hidden=16)
    runner.clone_world_study(parent, target, variant)
    assert (parent/"expert/candidate.pt").read_bytes() == (target/"expert/candidate.pt").read_bytes()
    assert (parent/"reuse/episodes/train/a.json").read_bytes() == (target/"reuse/episodes/train/a.json").read_bytes()
    a, b = runner.read(parent/"configuration.json"), runner.read(target/"configuration.json")
    assert a["solver"] == b["solver"] and a["training"] == b["training"]
    assert b["smoother_reuse"]["hidden"] == 32
    assert runner.read(target/"reuse/collection.json")["inherited_measurements"]["byte_identical_episodes"]
    assert runner.read(target/"world_parent.json")["parent_training_sha256"] == file_hash(parent/"reuse/training.json")
    runner.clone_world_study(parent, target, variant)
    (parent/"expert/candidate.pt").write_text("changed weights")
    with pytest.raises(ValueError, match="parent/configuration changed"):
        runner.clone_world_study(parent, target, variant)


def test_world_selection_uses_successful_tune_cost_only():
    candidates = {"fast_failed": {"metrics": {"world": {"success": False, "seconds": .1}}},
                  "good": {"metrics": {"world": {"success": True, "seconds": 2.}}},
                  "slow": {"metrics": {"world": {"success": True, "seconds": 4.}}}}
    assert runner.choose_successful(candidates, "world") == "good"
    with pytest.raises(RuntimeError, match="no successful tune"):
        runner.choose_successful({"bad": candidates["fast_failed"]}, "world")


def fake_family(tmp_path, monkeypatch, stop, *, fail_validation=False):
    value = plan(tmp_path, "--families", "shear_composition")
    write_json(tmp_path/"plan.json", value)
    calls = []
    monkeypatch.setattr(runner, "generate_case", lambda *a: tmp_path/"prepared")
    monkeypatch.setattr(runner, "collect_case", lambda *a: tmp_path/"recording")
    def prepare(root, family, arm, plan, recordings):
        out = root/"families"/family/"studies"/arm["id"]
        write_json(out/"reuse/collection.json", {"probe": {"world_model_candidate": False}})
        return out
    monkeypatch.setattr(runner, "prepare_arm", prepare)
    monkeypatch.setattr(runner, "fit_and_score", lambda out, plan: dict(
        metrics={"refresh":dict(success=True, seconds=2.)}, models_available=False, selected_reference="reuse"))
    def evaluate(root, family, out, split, plan):
        calls.append((out.name, "evaluate", split))
        if split == "validation" and fail_validation:
            raise RuntimeError("numerical/physical validation failure")
        metrics = {name:dict(median_seconds=3. if name == "classical" else 2., speedup_vs_classical=1.5)
                   for name in ("classical", "refresh", "reuse")}
        return dict(success=True, temporal=dict(metrics=metrics, selected_reference="reuse"))
    monkeypatch.setattr(runner, "evaluate_study", evaluate)
    def command(plan, out, temporal, name, *args):
        calls.append((out.name, "temporal_freeze" if temporal else "base_freeze", None))
        write_json(out/("reuse/freeze.json" if temporal else "freeze.json"), {"frozen": True})
    monkeypatch.setattr(runner, "command", command)
    return value, calls


def test_probe_does_not_fit_world_or_touch_validation_test(tmp_path, monkeypatch):
    value, calls = fake_family(tmp_path, monkeypatch, "probe")
    monkeypatch.setattr(runner, "fit_and_score", lambda *a: pytest.fail("probe fitted a temporal model"))
    result = runner.advance_family(tmp_path, "shear_composition", value, "probe")
    assert result["completed_through"] == "probe" and not calls


def test_freeze_precedes_test_and_smoother_vs_world_are_separate(tmp_path, monkeypatch):
    value, calls = fake_family(tmp_path, monkeypatch, "test")
    result = runner.advance_family(tmp_path, "shear_composition", value, "test")
    first_test = next(i for i, row in enumerate(calls) if row[2] == "test")
    assert sum(row[1].endswith("freeze") for row in calls[:first_test]) == 6
    assert result["test_selection_performed"] is False
    assert result["test"]["world_additional"] is None
    assert result["test"]["multilevel_refresh_speedup"] == 1.
    frozen = runner.read(tmp_path/"families/shear_composition/comparison_freeze.json")
    assert frozen["selection_split"] == "tune"


def test_validation_failure_blocks_test_and_freeze(tmp_path, monkeypatch):
    value, calls = fake_family(tmp_path, monkeypatch, "test", fail_validation=True)
    with pytest.raises(RuntimeError, match="validation failure"):
        runner.advance_family(tmp_path, "shear_composition", value, "test")
    assert not any(row[1].endswith("freeze") or row[2] == "test" for row in calls)
