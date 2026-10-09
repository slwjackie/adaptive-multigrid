"""Benchmark fairness and full-CFD claim gates, including unsuccessful runs."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.world_model.data import snapshot
from adaptive_mg.v67.h2_fixed_p.evaluation import evaluate_replay, compare_coupled_runs


@pytest.fixture
def paired_solver(monkeypatch):
    from adaptive_mg.v67.h2_fixed_p import backend
    created = []

    class Solver:
        fail_at = None
        change_p = False
        use_neural = True

        def __init__(self, cfg, expert=None):
            self.expert = expert
            self.p_digest = None
            self.seen = []
            created.append(self)

        def step(self, s):
            self.seen.append((s.matrix_digest, s.b.copy(), s.x0.copy()))
            neural = self.expert is not None
            self.p_digest = "changed" if self.change_p and neural and s.index == 1 else "same-P"
            success = not (neural and self.fail_at == s.index)
            d = dict(success=success, cycles=2, setup_seconds=.2, solve_seconds=.7,
                     total_seconds=1. if neural else 2., residuals=[1., 1.e-9 if success else 1.],
                     threshold=1.e-8, final_true_residual=1.e-9 if success else 1., fallback=not success,
                     stats={"neural_trial_cycles": 2 if neural and self.use_neural else 0,
                            "actual_neural_used": neural and self.use_neural,
                            "verification_seconds": .1, "recovery_seconds": .2 if not success else 0.})
            return SimpleNamespace(residuals=d["residuals"], record=lambda: deepcopy(d))

    monkeypatch.setattr(backend, "FixedPSolver", Solver)
    return Solver, created


def cohort():
    states = [snapshot(sp.eye(9, format="csr") * (i + 1), np.arange(9, dtype=float),
                       x0=np.ones(9) * .1, shape=(3, 3), index=i, time=i * .1,
                       mesh_id="mesh", boundary_id="dirichlet") for i in range(2)]
    return [(dict(id="validation-0", case_group="case-0", split="validation"), states)]


def trained():
    return SimpleNamespace(metadata={"training_branch": "H_S", "optimizer_updates": 2})


def test_replay_has_independent_streams_counterbalanced_order_and_identical_inputs(paired_solver):
    _, created = paired_solver
    result = evaluate_replay(cohort(), AdaptiveConfig(), trained(), repeats=4)
    assert result["success"] and result["speed_claim_eligible"]
    assert len(created) == 8 and all(len(s.seen) == 2 for s in created)
    assert result["rows"][0]["order"][0] == result["rows"][0]["order"][2]
    assert result["rows"][0]["order"][0] == result["rows"][0]["order"][1][::-1]
    for solver in created[1:]:
        for lhs, rhs in zip(solver.seen, created[0].seen):
            assert lhs[0] == rhs[0]
            np.testing.assert_array_equal(lhs[1], rhs[1])
            np.testing.assert_array_equal(lhs[2], rhs[2])
    assert result["speedup"] == pytest.approx(2., rel=.001)
    assert result["full_cfd_wall_clock_measured"] is False
    assert result["diagnostic"]["world_model_enabled"] is False


def test_replay_keeps_failure_and_forbids_aggregate_speed_claim(paired_solver):
    solver, _ = paired_solver
    solver.fail_at = 1
    result = evaluate_replay(cohort(), AdaptiveConfig(), trained(), repeats=2)
    assert not result["success"] and result["speedup"] is None
    runs = result["rows"][0]["runs"]["H_S"]
    assert len(runs) == 2 and all(len(r["records"]) == 2 for r in runs)
    assert all(r["fallback_count"] == 1 for r in runs)
    assert all(r["total_seconds"] >= 2. for r in runs)
    assert not result["diagnostic"]["world_model_candidate"]


def test_replay_rejects_changed_p_and_unused_hs_claim(paired_solver):
    solver, _ = paired_solver
    solver.change_p = True
    result = evaluate_replay(cohort(), AdaptiveConfig(), trained(), repeats=1)
    assert not result["success"] and result["speedup"] is None
    solver.change_p = False
    solver.use_neural = False
    result = evaluate_replay(cohort(), AdaptiveConfig(), trained(), repeats=1)
    assert result["success"] and not result["speed_claim_eligible"] and result["speedup"] is None


def test_replay_refuses_untrained_or_wrong_branch(paired_solver):
    for metadata in ({"training_branch": "H_S", "optimizer_updates": 0},
                     {"training_branch": "H_P", "optimizer_updates": 3}):
        with pytest.raises(ValueError, match="genuinely trained"):
            evaluate_replay(cohort(), AdaptiveConfig(), SimpleNamespace(metadata=metadata))


@pytest.fixture
def coupled():
    c = dict(schema="h2-fixed-p-coupled-run-v1", arm="classical", source_kind="external_cfd",
             execution_mode="online_coupled", success=True, case_id="flame-1",
             mesh_sha256="mesh", chemistry_sha256="chem", initial_conditions_sha256="init",
             boundary_conditions_sha256="bc", dt_schedule=[.01, .01],
             stopping_criteria={"rtol": 1.e-8, "atol": 1.e-12}, end_time=.02,
             fixed_p_sha256="P", hardware={"cpu": "fixture"}, execution_threads=1,
             pressure_gauge="absolute", continuity={"passed": True},
             evidence={"coupled_run_attested": True, "solver_log_sha256": "log",
                       "field_files_sha256": {"T": "T", "H2": "h", "U": "u", "p": "p"}},
             timing={"wall_seconds": 10., "pressure_seconds": 4.},
             fields={"T": np.array([300., 1500.]), "H2": np.array([.02, .001]),
                     "U": np.array([[.2, 0.], [.4, 0.]]), "p": np.array([100., 101.])})
    h = deepcopy(c)
    h["arm"] = "H_S"
    h["timing"] = dict(wall_seconds=8., pressure_seconds=2.)
    h["evidence"].update(expert_checkpoint_sha256="trained-checkpoint", neural_apply_calls=4)
    tolerance = dict(declared_before_evaluation=True, pressure_alignment="none",
                     fields={key: dict(atol=1.e-5, rtol=1.e-6) for key in ("T", "H2", "U", "p")})
    return c, h, tolerance


def test_coupled_reports_full_and_pressure_speedup_separately(coupled):
    report = compare_coupled_runs(*coupled)
    assert report["success"]
    assert report["pressure_speedup"] == 2.
    assert report["full_cfd_speedup"] == 1.25
    assert report["physical_observables_validated"]
    assert report["independent_authenticity_verified"] is False


@pytest.mark.parametrize("change", ["case", "dt", "chemistry", "P", "failure", "nonfinite", "field", "synthetic", "offline", "tolerance", "continuity", "unused_hs"])
def test_coupled_blocks_invalid_or_inaccurate_claims(coupled, change):
    c, h, tol = coupled
    if change == "case": h["case_id"] = "different"
    elif change == "dt": h["dt_schedule"] = [.02]
    elif change == "chemistry": h["chemistry_sha256"] = "different"
    elif change == "P": h["fixed_p_sha256"] = "different"
    elif change == "failure": h["success"] = False
    elif change == "nonfinite": h["fields"]["T"][0] = np.nan
    elif change == "field": h["fields"]["H2"][1] += .01
    elif change == "synthetic": h["source_kind"] = "synthetic_elliptic"
    elif change == "offline": h["execution_mode"] = "replay"
    elif change == "tolerance": tol["declared_before_evaluation"] = False
    elif change == "continuity": h["continuity"]["passed"] = False
    elif change == "unused_hs": h["evidence"]["neural_apply_calls"] = 0
    report = compare_coupled_runs(c, h, tol)
    assert not report["success"] and report["blockers"]
    assert report["pressure_speedup"] is None and report["full_cfd_speedup"] is None


def test_pressure_offset_only_removed_with_explicit_free_gauge(coupled):
    c, h, tol = coupled
    h["fields"]["p"] += 1000.
    assert not compare_coupled_runs(c, h, tol)["success"]
    tol["pressure_alignment"] = "subtract_spatial_mean"
    assert not compare_coupled_runs(c, h, tol)["success"]
    c["pressure_gauge"] = h["pressure_gauge"] = "free_constant"
    assert compare_coupled_runs(c, h, tol)["success"]


def test_series_pressure_alignment_is_per_timestamp(coupled):
    c, h, tol = coupled
    for run in (c, h):
        run["sample_times"] = [.01, .02]
        run["fields"] = {key: np.stack([arr, arr]) for key, arr in run["fields"].items()}
        run["pressure_gauge"] = "free_constant"
    h["fields"]["p"] += np.array([[100.], [200.]])
    tol["pressure_alignment"] = "subtract_spatial_mean"
    report = compare_coupled_runs(c, h, tol)
    assert report["success"] and report["fields"]["p"]["max_absolute_error"] == 0.


def test_missing_field_or_timing_is_retained_as_failed_evidence(coupled):
    c, h, tol = coupled
    del h["fields"]["U"]
    h["timing"]["pressure_seconds"] = float("nan")
    report = compare_coupled_runs(c, h, tol)
    assert not report["success"] and not report["fields"]["U"]["passed"]
    assert report["timing"]["H_S"]["pressure_seconds"] is None
