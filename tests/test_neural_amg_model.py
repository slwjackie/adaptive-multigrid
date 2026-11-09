import numpy as np
import pytest
import scipy.sparse as sp
import torch
from neural_amg.structure import Structure, build_structure, validate_support_hops
from neural_amg.model import GraphAttentionTransfer, graph_features, constrain_weights
from neural_amg.autograd_sparse import SparseTensor


def problem():
    import pyamg
    return pyamg.gallery.poisson((5, 5), format="csr")


def test_classical_structure_injection_hop_bounds():
    a = problem()
    s = build_structure(a)
    np.testing.assert_allclose(s.reference[s.coarse].toarray(), np.eye(len(s.coarse)))
    validate_support_hops(a, s, 2)
    with pytest.raises(ValueError, match="at most"):
        build_structure(a, max_hops=3)


@pytest.mark.parametrize("mode", ["original", "row9", "symmetric9"])
def test_model_constraints_gradients_and_full_coarse_fill(mode):
    torch.manual_seed(4)
    a = problem()
    s = build_structure(a)
    at = SparseTensor.from_scipy(a, requires_grad=True)
    m = GraphAttentionTransfer(feature_mode=mode)
    p = m(at, s)
    np.testing.assert_array_equal(p.rows, s.rows)
    np.testing.assert_array_equal(p.cols, s.cols)
    np.testing.assert_allclose(np.asarray(p.numpy().sum(1)).ravel(), s.row_targets, atol=1e-12)
    np.testing.assert_allclose(p.numpy()[s.coarse].toarray(), np.eye(len(s.coarse)))
    loss = (p.values ** 2).sum()
    loss.backward()
    assert at.values.grad is not None and torch.isfinite(at.values.grad).all()
    assert at.values.grad.abs().sum() > 0
    assert any(t.grad is not None and t.grad.abs().sum() > 0 for t in m.parameters())
    ac = p.transpose().product(at).product(p).symmetrize()
    sc = build_structure(ac.numpy())
    f = graph_features(ac, sc, mode)
    expected = ac.pattern.nnz if mode == "original" else np.sum(ac.rows != ac.cols)
    assert f.edge_index.shape == (2, expected)
    assert m(ac, sc).shape[0] == len(s.coarse)


def test_signed_head_constraints_and_l1_cap():
    reference = sp.csr_matrix([[1., 0.], [.5, .5], [0., 1.]])
    s = Structure(np.array([0, 2]), reference)
    raw = torch.tensor([0., -10., 10., 0.], dtype=torch.float64, requires_grad=True)
    values = constrain_weights(raw, s, 3.)
    assert values[1] < 0
    assert torch.allclose(values[1:3].sum(), torch.tensor(1., dtype=torch.float64))
    assert values[1:3].abs().sum() <= 3. + 1e-12
    values.square().sum().backward()
    assert torch.isfinite(raw.grad).all()
    with pytest.raises(ValueError, match="Infeasible"):
        constrain_weights(raw, s, .5)


def test_permutation_equivariance_with_same_classical_decisions():
    # Greedy RS tie breaking need not be permutation invariant. Freeze decisions
    # and permute them together with A to test the *learned predictor* correctly.
    a = problem()
    s = build_structure(a)
    permutation = np.random.default_rng(19).permutation(a.shape[0])
    inverse = np.argsort(permutation)
    permuted = Structure(inverse[s.coarse], s.reference[permutation])
    torch.manual_seed(7)
    model = GraphAttentionTransfer()
    original = model(SparseTensor.from_scipy(a), s).numpy()
    actual = model(SparseTensor.from_scipy(a[permutation][:, permutation]), permuted).numpy()
    np.testing.assert_allclose(actual.toarray(), original[permutation].toarray(), atol=1e-12)


def test_feature_normalization_modes_are_explicit():
    a = sp.csr_matrix([[4., -1., 0.], [-1., 3., -2.], [0., -2., 5.]])
    s = Structure(np.array([0, 2]), sp.csr_matrix([[1., 0.], [.3, .7], [0., 1.]]))
    at = SparseTensor.from_scipy(a)
    original = graph_features(at, s)
    row = graph_features(at, s, "row9")
    sym = graph_features(at, s, "symmetric9")
    assert original.x.shape == (3, 3)
    assert row.x.shape == sym.x.shape == (3, 2)
    assert original.edge_index.shape[1] == 7
    assert row.edge_index.shape[1] == 4
    assert row.edge_attr[0, 0].item() == pytest.approx(-1 / 4)
    assert sym.edge_attr[0, 0].item() == pytest.approx(-1 / np.sqrt(12))
    assert torch.allclose(row.x[:, 0], original.x[:, 0])
    assert GraphAttentionTransfer().spec["feature_mode"] == "original"


def test_invalid_matrix_and_invalid_structures_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        build_structure(sp.csr_matrix([[1., 1.], [0., 1.]]))
    with pytest.raises(ValueError, match="positive diagonal"):
        build_structure(sp.diags([1., -1.]))
    with pytest.raises(ValueError, match="orphan"):
        Structure(np.array([0]), sp.csr_matrix([[1.], [0.]]))


def test_three_encoder_hops_do_not_add_output_edges():
    a = problem()
    s = build_structure(a)
    model = GraphAttentionTransfer(depth=3)
    assert len(model.encoder) == 3
    p = model(a, s)
    assert p.pattern.nnz == s.reference.nnz


def test_checkpoint_spec_round_trip_and_schema_validation():
    m = GraphAttentionTransfer(feature_mode="symmetric9", hidden=8, heads=2)
    restored = GraphAttentionTransfer.from_spec(m.spec)
    restored.load_state_dict(m.state_dict(), strict=True)
    assert restored.spec == m.spec
    a = problem()
    s = build_structure(a)
    torch.testing.assert_close(restored(a, s).values, m(a, s).values)
    with pytest.raises(ValueError, match="schema"):
        GraphAttentionTransfer.from_spec({**m.spec, "feature_schema": "wrong"})
    with pytest.raises(ValueError, match="specification"):
        GraphAttentionTransfer.from_spec({**m.spec, "unexpected": 1})
