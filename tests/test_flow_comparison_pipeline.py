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


def test_collect_one_keeps_full_plan_for_resume_without_claiming_full_collection(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "preflight", lambda plan: None)
    def collect(root, family, case, plan):
        calls.append((family, case["name"]))
        return root/"recording_fixture"
    monkeypatch.setattr(runner, "collect_case", collect)
    result = runner.main(["--output", str(tmp_path/"run"), "--with-h2",
        "--stop-after", "collect", "--collect-one", "lox_he_sloshing", "tune_slosh"])
    assert calls == [("lox_he_sloshing", "tune_slosh")]
    assert result["status"] == "partial" and result["completed_through"] == "collect_one"
    assert not result["hydrogen_combustion_validated"] and not result["world_model_benefit_established"]
    pinned = runner.check_plan(tmp_path/"run", arguments(tmp_path, "--resume"))
    assert len(pinned["families"]["lox_he_sloshing"]["cases"]) == 6
    assert pinned["hydrogen"]["enabled"]
    resumed = []
    def advance(root, family, plan, stop):
        resumed.append((family, stop))
        return {"completed_through": stop}
    monkeypatch.setattr(runner, "advance_family", advance)
    runner.main(["--output", str(tmp_path/"run"), "--resume", "--stop-after", "collect"])
    assert resumed == [(family, "collect") for family in pinned["families"]]


@pytest.mark.parametrize("options", [
    ["--collect-one", "lox_he_sloshing", "tune_slosh"],
    ["--stop-after", "collect", "--collect-one", "lox_he_sloshing", "test_fill"],
    ["--stop-after", "collect", "--collect-one", "lox_he_sloshing", "missing"],
    ["--stop-after", "collect", "--families", "h2_crossflow", "--collect-one", "lox_he_sloshing", "tune_slosh"],
])
def test_collect_one_invalid_scope_rejected_before_creating_output(tmp_path, options):
    with pytest.raises(SystemExit):
        runner.main(["--output", str(tmp_path/"run"), *options])
    assert not (tmp_path/"run").exists()


def test_collect_one_failure_is_preserved_and_does_not_advance_other_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda plan: None)
    def fail(*args):
        raise RuntimeError("native field checks failed")
    monkeypatch.setattr(runner, "collect_case", fail)
    with pytest.raises(RuntimeError, match="native field checks"):
        runner.main(["--output", str(tmp_path/"run"), "--stop-after", "collect",
                     "--collect-one", "lox_he_sloshing", "tune_slosh"])
    result = runner.read(tmp_path/"run/status.json")
    assert result["status"] == "failed" and result["results_preserved"]
    assert not result["families"]


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


def test_reacting_plan_is_opt_in_predeclared_and_pinned(tmp_path):
    assert plan(tmp_path)["hydrogen"] == {"enabled": False}
    value = plan(tmp_path, "--with-h2")
    h2 = value["hydrogen"]
    assert h2["source_family"] == "h2_crossflow"
    assert h2["weights_transferred"] is False
    assert {c["split"] for c in h2["cases"]} == {"train", "tune", "validation", "test"}
    for options in (("--with-h2", "--families", "shear_composition"),
                    ("--h2-source-family", "shear_composition"),
                    ("--with-h2", "--h2-end-time", "0"),
                    ("--with-h2", "--h2-delta-t", "nan")):
        with pytest.raises(ValueError):
            plan(tmp_path, *options)
    runner.main(["--output", str(tmp_path/"run"), "--dry-run", "--with-h2"])
    assert runner.check_plan(tmp_path/"run", arguments(tmp_path, "--resume"))["hydrogen"] == h2
    with pytest.raises(ValueError, match="cannot change H2"):
        runner.check_plan(tmp_path/"run", arguments(tmp_path, "--resume", "--h2-source-family", "shear_composition"))


def test_resume_cannot_append_reacting_experiment_after_inspecting_results(tmp_path):
    runner.main(["--output", str(tmp_path/"run"), "--dry-run"])
    with pytest.raises(ValueError, match="cannot add H2"):
        runner.check_plan(tmp_path/"run", arguments(tmp_path, "--resume", "--with-h2"))


def test_hydrogen_inherits_actual_cnn_and_world_not_inert_physics(tmp_path):
    value = plan(tmp_path, "--with-h2")
    source = value["families"]["lox_he_sloshing"]["configs"]["per_level_l01"]
    source["smoother_reuse"].update(hidden=64, world_memory="independent_roles", world_head="role_mlp")
    target = runner.hydrogen_settings(source, value, world_selected=True)
    assert target["training"]["smoother_architecture"] == "per_level"
    assert target["solver"]["smoother_levels"] == [0, 1]
    assert target["solver"]["mg"]["max_cycles"] == 600
    assert target["smoother_reuse"]["hidden"] == 64
    assert target["smoother_reuse"]["world_memory"] == "independent_roles"
    assert target["physical_tolerances"] == value["hydrogen"]["base_config"]["physical_tolerances"]
    assert set(target["physical_tolerances"]["fields"]) == {"T", "H2", "U", "p"}
    assert target["benchmark"]["experiment_kind"] == "h2_combustion"
    assert not runner.hydrogen_settings(source, value, world_selected=False)["smoother_reuse"]["fit_world"]
    target["training"]["hidden"] = 99
    assert source["training"]["hidden"] != 99


def test_reacting_collection_requires_real_finite_positive_heat_release():
    for evidence in ({}, dict(finite_saved_Qdot=True, positive_heat_release_observed=False),
                     dict(finite_saved_Qdot=False, positive_heat_release_observed=True)):
        assert not runner.collection_physics({"reaction_evidence": evidence}, runner.H2_FAMILY)["passed"]
    assert runner.collection_physics({"reaction_evidence": dict(
        finite_saved_Qdot=True, positive_heat_release_observed=True)}, runner.H2_FAMILY)["passed"]
    assert runner.generator_for(runner.H2_FAMILY).name == "generate.py"
    assert runner.generator_options(runner.H2_FAMILY, {"temperature": 325}) == ["--inlet-temperature", "325"]


def transfer_fixture(tmp_path, monkeypatch):
    from adaptive_mg.v67.smoother_reuse import study as reuse
    value = plan(tmp_path, "--with-h2")
    write_json(tmp_path/"plan.json", value)
    base = tmp_path/"families/h2_crossflow"
    source = base/"studies/per_level_l01"
    write_json(base/"selection.json", dict(selection_split="tune", test_read=False,
        world_study=None, selected_two_level_cnn="per_level_l01"))
    write_json(source/"configuration.json", value["families"]["h2_crossflow"]["configs"]["per_level_l01"])
    for name in ("expert/candidate.pt", "reuse/training.json"):
        write_json(source/name, {"fixture": True})
    write_json(source/"comparison_validation.json", dict(success=True, split="validation",
        full_cfd_wall_clock_measured=True, coupled={"case": {"native": {"success": True}}}))
    monkeypatch.setattr(reuse, "load", lambda out: (None, None, None, None, {}))
    monkeypatch.setattr(reuse, "trained", lambda *a: {})
    return value, source


def test_transfer_pin_rejects_changed_source_and_never_reads_test(tmp_path, monkeypatch):
    value, source = transfer_fixture(tmp_path, monkeypatch)
    settings, pin = runner.pin_hydrogen_transfer(tmp_path, value)
    assert pin["weights_transferred"] is False and pin["test_read_for_selection"] is False
    assert settings["training"]["smoother_architecture"] == "per_level"
    write_json(source/"comparison_test.json", {"do_not_read": True})
    assert runner.pin_hydrogen_transfer(tmp_path, value)[1] == pin
    write_json(source/"expert/candidate.pt", {"fixture": "changed"})
    with pytest.raises(ValueError, match="frozen hydrogen transfer changed"):
        runner.pin_hydrogen_transfer(tmp_path, value)


def test_failed_source_physics_cannot_transfer(tmp_path, monkeypatch):
    value, source = transfer_fixture(tmp_path, monkeypatch)
    report = runner.read(source/"comparison_validation.json")
    report["coupled"]["case"]["native"]["success"] = False
    write_json(source/"comparison_validation.json", report)
    with pytest.raises(ValueError, match="coupled physical validation"):
        runner.pin_hydrogen_transfer(tmp_path, value)
    assert not (tmp_path/"hydrogen_transfer.json").exists()


@pytest.mark.parametrize("fail_h2", [False, True])
def test_integrated_workflow_gates_all_tests_on_reacting_validation(tmp_path, monkeypatch, fail_h2):
    calls = []
    monkeypatch.setattr(runner, "preflight", lambda *a: None)
    def primary(root, family, plan, stop):
        calls.append((family, stop))
        return {"completed_through": stop}
    def hydrogen(root, plan, stop):
        calls.append(("h2", stop))
        if fail_h2:
            raise RuntimeError("reacting validation failed")
        return {"hydrogen_combustion_validated": True}
    monkeypatch.setattr(runner, "advance_family", primary)
    monkeypatch.setattr(runner, "advance_hydrogen", hydrogen)
    args = ["--output", str(tmp_path/"run"), "--with-h2", "--stop-after", "test"]
    if fail_h2:
        with pytest.raises(RuntimeError, match="reacting validation failed"):
            runner.main(args)
        assert not any(stage == "test" for _, stage in calls)
        assert not runner.read(tmp_path/"run/summary.json")["hydrogen_combustion_validated"]
    else:
        result = runner.main(args)
        assert calls[:4] == [(f, "validation") for f in runner.FAMILIES[:3]] + [("h2", "validation")]
        assert calls[-1] == ("h2", "test")
        assert result["hydrogen_combustion_validated"]


def test_hydrogen_freezes_both_studies_before_test(tmp_path, monkeypatch):
    value, source = transfer_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda *a: object())
    monkeypatch.setattr(runner.shutil, "which", lambda *a: "/mock/chemkinToFoam")
    monkeypatch.setattr(runner, "collect_case", lambda *a: tmp_path/"recordings")
    out = tmp_path/"families/hydrogen_reacting/studies/selected_method"
    write_json(out/"reuse/collection.json", {"probe": {"world_model_candidate": False}})
    monkeypatch.setattr(runner, "prepare_arm", lambda *a: out)
    monkeypatch.setattr(runner, "fit_and_score", lambda *a: {"models_available": False})
    def evaluate(root, family, study, split, plan):
        calls.append(split)
        if split == "test":
            assert (out/"freeze.json").exists() and (out/"reuse/freeze.json").exists()
        report = dict(success=True, full_cfd_wall_clock_measured=True)
        write_json(out/("comparison_"+split+".json"), report)
        return report
    monkeypatch.setattr(runner, "evaluate_study", evaluate)
    def freeze(plan, out, temporal, name, *args):
        calls.append("freeze")
        write_json(out/("reuse/freeze.json" if temporal else "freeze.json"), {})
    monkeypatch.setattr(runner, "command", freeze)
    result = runner.advance_hydrogen(tmp_path, value, "test")
    assert calls == ["validation", "freeze", "freeze", "test"]
    assert result["held_out_test_completed"] and result["hydrogen_combustion_validated"]
    assert not result["world_model_benefit_established"] and not result["chemistry_time_separately_measured"]
