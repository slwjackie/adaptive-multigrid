"""Independent numerical invariants for the A-only classical selector.

Expected directions and eigenvalue ratios come from the analytic constant
diffusion tensor, not from benchmark outcomes or family labels.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from adaptive_mg.pde import DiffusionCase, assemble_stiffness
from adaptive_mg.v67.strong import operator_features, select_strong_strategy


def constant_operator(n=7, epsilon=0.001, angle=30.0):
    return assemble_stiffness(DiffusionCase(
        n=n, epsilon=epsilon, angle_deg=angle, contrast=1.0, pattern="uniform"))


def axis_distance(first, second):
    """Eigenvectors define an unoriented axis, so angles are modulo 180."""
    return abs((first - second + 90.0) % 180.0 - 90.0)


@pytest.mark.parametrize("n", [3, 7, 15])
@pytest.mark.parametrize("epsilon,angle", [
    (0.001, 0.0), (0.001, 90.0), (0.001, 30.0),
    (0.001, 45.0), (0.001, -45.0), (0.2, -17.0),
])
def test_interior_moments_recover_analytic_p1_anisotropy(n, epsilon, angle):
    features = operator_features(constant_operator(n, epsilon, angle), n)
    assert features["moment_rows"] == (n - 2) ** 2
    assert features["valid_tensor_fraction"] == 1.0
    assert features["tensor_anisotropy_ratio"] == pytest.approx(1.0 / epsilon, rel=2e-10)
    assert axis_distance(features["principal_angle_deg"], angle) < 2e-10
    assert features["local_anisotropy_log_std"] < 2e-10
    assert features["orientation_variation"] < 2e-10


def test_positive_and_negative_diagonal_anisotropy_are_not_confused():
    positive = select_strong_strategy(constant_operator(angle=45.0), 7)
    negative = select_strong_strategy(constant_operator(angle=-45.0), 7)
    assert positive.rule_id == "diagonal_positive"
    assert negative.rule_id == "rotated"
    assert positive.strategy_name.startswith("line_diag45_")
    assert not negative.strategy_name.startswith("line_diag45_")
    # The fixed triangulation can produce positive off-diagonals while A is SPD.
    assert negative.features["positive_offdiagonal_fraction"] > 0.0
    assert negative.features["tensor_anisotropy_ratio"] == pytest.approx(1000.0, rel=2e-10)


@pytest.mark.parametrize("factor", [1e-100, 1e-9, 7.0, 1e9, 1e100])
def test_global_operator_rescaling_cannot_change_classical_choice(factor):
    a = constant_operator(angle=30.0)
    reference = select_strong_strategy(a, 7)
    scaled = select_strong_strategy(a * factor, 7)
    assert (reference.rule_id, reference.strategy_name) == (scaled.rule_id, scaled.strategy_name)
    assert reference.features.keys() == scaled.features.keys()
    for key in reference.features:
        assert scaled.features[key] == pytest.approx(reference.features[key], rel=2e-9, abs=2e-10), key


def test_removed_dirichlet_edges_do_not_bias_tensor_estimate():
    n = 7
    a = constant_operator(n=n, angle=30.0)
    # A positive semidefinite rank-one perturbation wholly on the excluded
    # outer ring changes boundary coupling without touching any interior row.
    edge = sp.csr_matrix(([1.0, -1.0], ([0, 1], [0, 0])), shape=(n*n, 1))
    changed = a + 1000.0 * (edge @ edge.T)
    reference, boundary = operator_features(a, n), operator_features(changed, n)
    for key in ("tensor_anisotropy_ratio", "principal_angle_deg", "valid_tensor_fraction",
                "orientation_variation", "local_anisotropy_log_std", "local_anisotropic_fraction"):
        assert boundary[key] == pytest.approx(reference[key], rel=2e-9, abs=2e-10), key
    assert boundary["diagonal_contrast_proxy"] > reference["diagonal_contrast_proxy"]


def test_grid_axis_exchange_exchanges_aligned_line_and_semicoarsening():
    n = 7
    a = constant_operator(n=n, angle=0.0)
    exchange = np.arange(n*n).reshape(n, n).T.ravel()
    transposed = a[exchange][:, exchange]
    first, second = select_strong_strategy(a, n), select_strong_strategy(transposed, n)
    assert first.rule_id == "aligned_x" and second.rule_id == "aligned_y"
    assert first.strategy_name == "line_x_operator_semi_y"
    assert second.strategy_name == "line_y_operator_semi_x"
    assert first.features["tensor_anisotropy_ratio"] == pytest.approx(second.features["tensor_anisotropy_ratio"])


@pytest.mark.parametrize("storage", ["csr", "csc", "coo", "lil", "dok"])
def test_sparse_storage_format_does_not_change_selection(storage):
    a = constant_operator()
    reference = select_strong_strategy(a, 7)
    candidate = select_strong_strategy(a.asformat(storage), 7)
    assert candidate.rule_id == reference.rule_id
    assert candidate.strategy_name == reference.strategy_name
    assert candidate.features == pytest.approx(reference.features)


def test_duplicate_unsorted_csr_is_not_mutated_by_feature_extraction():
    a = constant_operator()
    data, indices, indptr = [], [], [0]
    for row in range(a.shape[0]):
        start, end = a.indptr[row:row+2]
        values, columns = a.data[start:end][::-1], a.indices[start:end][::-1]
        data.extend(np.repeat(values * 0.5, 2))
        indices.extend(np.repeat(columns, 2))
        indptr.append(len(data))
    duplicate = sp.csr_matrix((data, indices, indptr), shape=a.shape)
    before = (duplicate.data.copy(), duplicate.indices.copy(), duplicate.indptr.copy(),
              duplicate.has_sorted_indices, duplicate.has_canonical_format)
    selected = select_strong_strategy(duplicate, 7)
    reference = select_strong_strategy(a, 7)
    assert selected.features == pytest.approx(reference.features)
    for actual, expected in zip((duplicate.data, duplicate.indices, duplicate.indptr), before[:3]):
        np.testing.assert_array_equal(actual, expected)
    assert duplicate.has_sorted_indices == before[3]
    assert duplicate.has_canonical_format == before[4]


def test_finite_extreme_dynamic_range_remains_finite_and_unresolved():
    a = sp.diags(np.geomspace(1e-300, 1e300, 49), format="csr")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        selection = select_strong_strategy(a, 7)
    assert selection.rule_id == "unresolved_tensor"
    assert all(np.isfinite(value) for value in selection.features.values())
    assert selection.features["diagonal_contrast_proxy"] > 1e100


def test_selector_uses_no_dense_operator_trial_solve_or_neural_forward(monkeypatch):
    a = constant_operator(n=63, angle=-17.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("A-only feature extraction must stay sparse and require no solve/NN")

    for name in ("spsolve", "splu", "eigsh", "eigs"):
        monkeypatch.setattr(spla, name, forbidden)
    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(sp.csc_matrix, "toarray", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    selection = select_strong_strategy(a, 63)
    assert selection.rule_id == "rotated"
    assert selection.features["tensor_anisotropy_ratio"] == pytest.approx(1000.0, rel=2e-10)


def test_calibration_cannot_swap_away_an_incumbent_selector_rescue():
    from dataclasses import replace
    from adaptive_mg.v67.config import AdaptiveConfig
    from adaptive_mg.v67.strong import StrongRules
    from adaptive_mg.v67.strong_calibration import calibrate_leaf_mapping

    rules = StrongRules()
    incumbent = "jacobi_bilinear_full"
    challenger = "chebyshev_bilinear_full"

    def run(strategy, success, seconds):
        cfg = AdaptiveConfig(mode="classical", branch="C")
        cfg = replace(cfg, mg=replace(cfg.mg, strategy_name=strategy))
        residual = cfg.mg.tolerance / 2 if success else 1.0
        return dict(measurement_config=cfg.to_dict(), converged=success,
                    wall_seconds=seconds, stopping_threshold=cfg.mg.tolerance,
                    final_true_residual=residual, recomputed_true_residual=residual,
                    residual_history=[1.0, residual],
                    relative_residual_history=[1.0, residual], executed_cycles=4)

    rows = []
    # Fixed C succeeds only on A; incumbent additionally solves B, challenger C.
    # Equal success counts and faster A must not hide the loss of solved case B.
    for case_id, fixed_ok, old_ok, new_ok in (
        ("A", True, True, True), ("B", False, True, False), ("C", False, False, True)
    ):
        old_run = run(incumbent, old_ok, 3.0)
        rows.append(dict(example={"name": case_id}, runs={
            "fixed_C": [run("line_alt_bilinear_full", fixed_ok, 2.0)],
            "strong_C": [dict(old_run, selection=dict(rule_id="near_isotropic", rules_digest=rules.digest()))],
            "oracle::" + incumbent: [old_run],
            "oracle::" + challenger: [run(challenger, new_ok, 1.0)],
        }))
    mapping, _ = calibrate_leaf_mapping(rows, rules, repeats=1)
    assert mapping["near_isotropic"] == incumbent
