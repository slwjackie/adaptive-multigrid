"""Direct-P geometry, fixed structural support, and differentiable CNN values."""
from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg.transfer import build_transfer_pattern, coarse_fine_indices
from adaptive_mg.v67.autograd_sparse import SparseTensor
from adaptive_mg.v67.bamg_transfer import (
    MaskCNN, WeightCNN, direct_features, direct_pattern, fixed_bilinear_mask, geometry_constraints,
    model_from_spec, project_weights, row_slots, scipy_from_weights,
    select_mask, sparse_from_weights,
)


def fixed_support(n=7, k=4):
    pattern = direct_pattern(n)
    mask = select_mask(torch.zeros(n * n, 16, dtype=torch.float64), pattern, k)
    return pattern, mask


def free_interior_row(pattern, mask):
    target, injection, _ = geometry_constraints(pattern)
    return int(np.flatnonzero((target == 1) & (injection < 0)
                             & (mask.sum(1).numpy() == 4))[0])


@pytest.mark.parametrize("n", [7, 15])
def test_constraints_use_geometry_and_include_all_local_candidates(n):
    pattern = direct_pattern(n)
    baseline = build_transfer_pattern(n)
    target, injection, valid = geometry_constraints(pattern)
    np.testing.assert_array_equal(pattern.columns, baseline.columns)
    np.testing.assert_allclose(target, baseline.bilinear_weights.sum(1), atol=0, rtol=0)
    np.testing.assert_array_equal(valid, pattern.columns >= 0)
    coarse_rows = coarse_fine_indices(baseline)
    np.testing.assert_array_equal(np.flatnonzero(injection >= 0), coarse_rows)
    np.testing.assert_array_equal(pattern.columns[coarse_rows, injection[coarse_rows]],
                                  np.arange(pattern.n_coarse_unknowns))
    poisoned = replace(pattern, bilinear_weights=np.full(pattern.columns.shape, np.nan))
    for actual, expected in zip(geometry_constraints(poisoned), (target, injection, valid)):
        np.testing.assert_array_equal(actual, expected)


def test_top_k_ties_follow_coarse_columns_with_exact_injection():
    pattern = direct_pattern(7)
    # Scrambling channel order tests the explicit coarse-column secondary key.
    order = np.random.default_rng(5).permutation(16)
    pattern = replace(pattern, columns=pattern.columns[:, order],
                      bilinear_weights=pattern.bilinear_weights[:, order])
    scores = torch.zeros(49, 16, dtype=torch.float64, requires_grad=True)
    mask = select_mask(scores, pattern, k=4)
    _, injection, valid = geometry_constraints(pattern)
    assert mask.dtype == torch.bool and not mask.requires_grad
    assert torch.equal(mask, select_mask(scores, pattern, k=4))
    assert not mask[torch.tensor(~valid)].any()
    for row in range(49):
        chosen = pattern.columns[row, mask[row].numpy()]
        if injection[row] >= 0:
            np.testing.assert_array_equal(chosen, [pattern.columns[row, injection[row]]])
        else:
            np.testing.assert_array_equal(np.sort(chosen),
                                          np.sort(pattern.columns[row, valid[row]])[:4])


def test_decoder_can_activate_a_connection_absent_from_bilinear():
    pattern = direct_pattern(15)
    baseline = build_transfer_pattern(15)
    _, injection, valid = geometry_constraints(pattern)
    row, slot = np.argwhere(valid & (baseline.bilinear_weights == 0)
                            & (injection[:, None] < 0))[0]
    scores = torch.zeros(225, 16, dtype=torch.float64)
    scores[row, slot] = 100
    mask = select_mask(scores, pattern, k=4)
    weights = project_weights(torch.zeros_like(scores), mask, pattern)
    assert mask[row, slot]
    assert weights[row, slot] > 0
    assert baseline.bilinear_weights[row, slot] == 0


@pytest.mark.parametrize("shape,coarse", [((7, 7), (3, 3)), ((3, 7), (3, 3)),
                                         ((3, 3), (1, 1))])
def test_fixed_bilinear_support_control_uses_only_geometry(shape, coarse):
    pattern = direct_pattern(shape, coarse)
    expected = build_transfer_pattern(shape, coarse).bilinear_weights > 0
    poisoned = replace(pattern, bilinear_weights=np.full(pattern.columns.shape, np.nan))
    mask = fixed_bilinear_mask(poisoned)
    np.testing.assert_array_equal(mask.numpy(), expected)
    raw = torch.randn(*mask.shape, dtype=torch.float64,
                      generator=torch.Generator().manual_seed(40))
    weights = project_weights(raw, mask, poisoned, l1_cap=3.)
    assert (weights[~mask] == 0).all()
    np.testing.assert_allclose(weights.sum(1).numpy(), geometry_constraints(pattern)[0],
                               atol=1e-15, rtol=0)


def test_signed_projection_preserves_constraints_and_l1_cap():
    pattern, mask = fixed_support()
    raw = torch.randn(49, 16, dtype=torch.float64,
                      generator=torch.Generator().manual_seed(12)) * 10
    weights = project_weights(raw, mask, pattern, l1_cap=2.)
    target, injection, _ = geometry_constraints(pattern)
    torch.testing.assert_close(weights.sum(1), torch.tensor(target), atol=2e-15, rtol=0)
    assert (weights.abs().sum(1) <= 2 + 2e-14).all()
    assert (weights[~mask] == 0).all()
    assert (weights[mask] < 0).any()
    rows = np.flatnonzero(injection >= 0)
    assert (weights[rows, injection[rows]] == 1).all()
    assert (weights[rows].count_nonzero(dim=1) == 1).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_candidate_boundary_and_runtime_cast_constraints(dtype):
    pattern = direct_pattern((3, 3), (1, 1))
    raw = torch.randn(9, 16, dtype=dtype,
                      generator=torch.Generator().manual_seed(21)) * 100
    mask = select_mask(raw, pattern, k=8)
    weights = project_weights(raw, mask, pattern, l1_cap=1.)
    target, injection, _ = geometry_constraints(pattern)
    assert (mask.sum(1) == 1).all()
    # A single retained candidate is uniquely fixed by its geometry row sum.
    torch.testing.assert_close(weights.sum(1), torch.tensor(target, dtype=dtype),
                               atol=0, rtol=0)
    runtime = scipy_from_weights(pattern, weights, mask)
    np.testing.assert_allclose(np.asarray(runtime.sum(axis=1)).ravel(), target,
                               atol=0, rtol=0)
    assert runtime.nnz == 9
    assert weights[4, injection[4]] == 1


def test_injection_guarantees_full_column_rank_and_galerkin_spd():
    pattern, mask = fixed_support()
    raw = torch.randn(49, 16, dtype=torch.float64,
                      generator=torch.Generator().manual_seed(27))
    weights = project_weights(raw, mask, pattern, l1_cap=3.)
    p = scipy_from_weights(pattern, weights, mask)
    injection_rows = coarse_fine_indices(build_transfer_pattern(7))
    np.testing.assert_array_equal(p[injection_rows].toarray(), np.eye(9))
    a = sp.diags([-.2 * np.ones(48), np.linspace(2., 5., 49), -.2 * np.ones(48)],
                 [-1, 0, 1], format="csr")
    pt = sparse_from_weights(pattern, weights, mask)
    actual = pt.transpose().product(SparseTensor.from_scipy(a).product(pt)).numpy()
    expected = p.T @ a @ p
    # Only the small coarsest 9x9 operator is made dense for this oracle.
    np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=2e-14, rtol=1e-13)
    np.testing.assert_allclose(actual.toarray(), actual.T.toarray(), atol=2e-14, rtol=1e-13)
    assert np.linalg.eigvalsh(actual.toarray()).min() > 0


@pytest.mark.parametrize("values,cap", [([.12, -.04, .31, .2], 8.),
                                         ([8., -6., 4., -3.], 2.)])
def test_projection_gradient_matches_finite_difference_with_or_without_active_cap(values, cap):
    pattern, mask = fixed_support()
    row = free_interior_row(pattern, mask)
    slots = mask[row].nonzero().flatten()
    rows = torch.full_like(slots, row)
    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)

    def decode(variable):
        raw = torch.zeros(49, 16, dtype=torch.float64).index_put((rows, slots), variable)
        return project_weights(raw, mask, pattern, l1_cap=cap)[row, slots]

    if cap == 2:
        torch.testing.assert_close(decode(x).abs().sum(), torch.tensor(cap),
                                   check_dtype=False, atol=1e-13, rtol=0)
    assert torch.autograd.gradcheck(decode, (x,), eps=1e-6, atol=1e-6, rtol=2e-5)


def test_selected_zero_keeps_structure_and_receives_gradient():
    pattern, mask = fixed_support()
    weights = project_weights(torch.zeros(49, 16, dtype=torch.float64), mask, pattern)
    row = free_interior_row(pattern, mask)
    slots = mask[row].nonzero().flatten()
    weights[row, slots] = torch.tensor([0., .25, .25, .5], dtype=torch.float64)
    variants = []
    for perturbation in (-1e-5, 0., 1e-5):
        candidate = weights.clone()
        candidate[row, slots[0]] += perturbation
        candidate[row, slots[1]] -= perturbation
        matrix = scipy_from_weights(pattern, candidate.numpy(), mask.numpy())
        assert matrix.nnz == int(mask.sum())
        variants.append(matrix)
    for matrix in variants[1:]:
        np.testing.assert_array_equal(matrix.indptr, variants[0].indptr)
        np.testing.assert_array_equal(matrix.indices, variants[0].indices)
    variable = weights.detach().requires_grad_()
    differentiable = sparse_from_weights(pattern, variable, mask)
    assert differentiable.pattern.nnz == int(mask.sum())
    probe = torch.arange(1, pattern.n_coarse_unknowns + 1, dtype=torch.float64)
    torch.testing.assert_close(differentiable.apply(probe),
                               torch.tensor(variants[1] @ probe.numpy()))
    differentiable.apply(probe).sum().backward()
    assert variable.grad[row, slots[0]] == probe[pattern.columns[row, slots[0]]]
    assert (variable.grad[~mask] == 0).all()


def test_actual_operator_features_match_runtime_and_preserve_sparse_gradient(monkeypatch):
    a = sp.diags([-.2 * np.ones(48), np.linspace(2., 5., 49), -.2 * np.ones(48)],
                 [-1, 0, 1], format="csr")
    differentiable = SparseTensor.from_scipy(a, requires_grad=True)

    def prohibit_dense(*args, **kwargs):
        raise AssertionError("feature generation must remain sparse")

    monkeypatch.setattr(sp.csr_matrix, "toarray", prohibit_dense)
    monkeypatch.setattr(torch.Tensor, "to_dense", prohibit_dense)
    runtime = direct_features(a, (7, 7))
    actual = direct_features(differentiable, (7, 7))
    assert actual.shape == (1, 10, 7, 7) and actual.dtype == torch.float64
    torch.testing.assert_close(actual, runtime, atol=0, rtol=0)
    actual.square().sum().backward()
    assert torch.isfinite(differentiable.values.grad).all()
    assert differentiable.values.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["early", "late", "none"])
def test_independent_cnns_mask_formats_and_state_reconstruction(mode):
    torch.manual_seed(7)
    pattern, mask = fixed_support()
    feature = torch.randn(1, 10, 7, 7, dtype=torch.float64)
    selection = MaskCNN(hidden=4, depth=2, activation="tanh").double().eval()
    weight = WeightCNN(hidden=5, depth=1, activation="silu", mask_mode=mode).double().eval()
    assert {p.data_ptr() for p in selection.parameters()}.isdisjoint(
        {p.data_ptr() for p in weight.parameters()})
    for model, args in ((selection, (feature,)), (weight, (feature, mask))):
        output = model(*args)
        assert output.shape == (1, 16, 7, 7)
        torch.testing.assert_close(row_slots(output, pattern),
                                   output[0].permute(1, 2, 0).reshape(49, 16))
        clone = model_from_spec(model.spec).double().eval()
        clone.load_state_dict(model.state_dict())
        assert clone.spec == model.spec
        torch.testing.assert_close(clone(*args), output, atol=0, rtol=0)
    spatial = mask.reshape(7, 7, 16).permute(2, 0, 1).unsqueeze(0)
    torch.testing.assert_close(weight(feature, mask), weight(feature, spatial))
    torch.testing.assert_close(weight(feature, mask), weight(feature, mask.unsqueeze(0)))
    if mode == "late":
        torch.testing.assert_close(weight.decode(weight.encode(feature), mask),
                                   weight(feature, mask))
    logits = row_slots(selection(feature), pattern)
    _, injection, valid = geometry_constraints(pattern)
    learned = torch.tensor(valid & (injection[:, None] < 0))
    torch.nn.functional.binary_cross_entropy_with_logits(logits[learned], mask.double()[learned]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in selection.parameters())
    selection_gradients = [p.grad.clone() for p in selection.parameters()]
    values = project_weights(weight(feature, mask), mask, pattern)
    (values * torch.arange(16, dtype=values.dtype)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in weight.parameters())
    for parameter, before in zip(selection.parameters(), selection_gradients):
        torch.testing.assert_close(parameter.grad, before, atol=0, rtol=0)


def test_invalid_or_empty_masks_are_rejected():
    pattern, mask = fixed_support()
    raw = torch.zeros(49, 16, dtype=torch.float64)
    target, injection, valid = geometry_constraints(pattern)
    bad = mask.clone()
    row, slot = np.argwhere(~valid)[0]
    bad[row, slot] = True
    with pytest.raises(ValueError, match="invalid"):
        project_weights(raw, bad, pattern)
    empty = mask.clone()
    row = np.flatnonzero((target > 0) & (injection < 0))[0]
    empty[row] = False
    with pytest.raises(ValueError, match="empty"):
        project_weights(raw, empty, pattern)
