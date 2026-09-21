"""Sparse graph-P invariants, numerical structure and differentiable A values."""
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg.pde import DiffusionCase, assemble_stiffness
from adaptive_mg.transfer import (TransferPattern, build_transfer_pattern, coarse_fine_indices,
    weights_from_deltas_torch, scipy_prolongation_from_weights, galerkin_coarse_operator)
from adaptive_mg.v67.autograd_sparse import SparseTensor
from adaptive_mg.v67.research_transfer import (make_graph_transfer, build_transfer_graph,
    transfer_pattern_for_model, transfer_distillation_loss, enforce_transfer_complexity,
    TransferComplexityCaps, TransferComplexityError, differentiable_transfer_cost, project_transfer_weights)


def operator(n=7, angle=30):
    return assemble_stiffness(DiffusionCase(n=n, epsilon=.05, angle_deg=angle, contrast=1., pattern="uniform"))


def nonzero_decoder(model):
    with torch.no_grad():
        model.delta_head.weight.normal_(std=.3)
        model.delta_head.bias.fill_(.05)
    return model


@pytest.mark.parametrize("architecture", ["small_gnn", "edge_mlp", "gnn_teacher"])
def test_zero_init_shapes_metadata_and_classical_equivalence(architecture):
    a=operator(); pattern=build_transfer_pattern(7)
    model=make_graph_transfer(architecture).double()
    delta=model.forward_graph(a,pattern)
    assert delta.shape == (1,16,7,7)
    assert torch.count_nonzero(delta)==0
    weights=weights_from_deltas_torch(pattern,delta)
    assert np.array_equal(weights.detach().numpy(),pattern.bilinear_weights)
    clone=make_graph_transfer(**model.architecture_config())
    clone.load_state_dict(model.state_dict())
    assert clone.architecture_config()==model.architecture_config()
    assert model.training_only == (architecture=="gnn_teacher")
    with pytest.raises(TypeError,match="forward_graph"):
        model(torch.zeros(1,10,7,7))


@pytest.mark.parametrize("coarsening", ["full", "semi_x", "semi_y"])
def test_sparse_graph_preserves_galerkin_spd_injection_and_boundary_rows(coarsening):
    a=operator(); pattern=build_transfer_pattern(7,coarsening=coarsening)
    net=nonzero_decoder(make_graph_transfer()).double()
    weights=weights_from_deltas_torch(pattern,net.forward_graph(a,pattern)).detach().numpy()
    # Summation order differs between NumPy/Torch/BLAS. Use a K-term FP64
    # roundoff bound rather than a sub-two-epsilon absolute constant.
    bound=pattern.n_candidates*np.finfo(weights.dtype).eps*max(1.,float(np.max(np.abs(weights).sum(1))))
    np.testing.assert_allclose(weights.sum(1),pattern.bilinear_weights.sum(1),rtol=0,atol=bound)
    assert np.all(weights[pattern.columns<0]==0)
    coarse=coarse_fine_indices(pattern)
    assert np.array_equal(weights[coarse],pattern.bilinear_weights[coarse])
    p=scipy_prolongation_from_weights(pattern,weights)
    r=p.T.tocsr(); ac=galerkin_coarse_operator(a,p)
    assert np.max(np.abs((ac-r@a@p).data),initial=0) < 1e-14
    assert np.max(np.abs((ac-ac.T).data),initial=0) < 1e-14
    # Dense terminal 9x9/21x21 is used only by this independent SPD check.
    assert np.linalg.eigvalsh(ac.toarray()).min()>0


@pytest.mark.parametrize("architecture", ["small_gnn", "edge_mlp"])
def test_support_channel_permutation_is_equivariant(architecture):
    torch.manual_seed(102)
    a=operator(); pattern=build_transfer_pattern(7)
    net=nonzero_decoder(make_graph_transfer(architecture)).double()
    permutation=np.random.default_rng(1).permutation(16)
    other=TransferPattern(pattern.fine_shape,pattern.coarse_shape,pattern.columns[:,permutation],pattern.bilinear_weights[:,permutation])
    original=net.forward_graph(a,pattern)
    permuted=net.forward_graph(a,other)
    torch.testing.assert_close(permuted,original[:,permutation],rtol=1e-12,atol=1e-13)


def test_sparse_values_receive_gradients_through_coarse_graph_and_galerkin():
    torch.manual_seed(17)
    at=SparseTensor.from_scipy(operator(),requires_grad=True)
    pattern=build_transfer_pattern(7)
    model=nonzero_decoder(make_graph_transfer(width=8,layers=1)).double()
    result=model.forward_graph(at,pattern,return_details=True)
    weights=result["weights"]
    cols=pattern.columns.ravel(); rows=np.repeat(np.arange(49),16); valid=cols>=0
    p=SparseTensor.from_coo(rows[valid],cols[valid],weights.ravel()[torch.tensor(valid)],(49,9))
    ac=p.transpose().product(at.product(p)).symmetrize()
    probe=torch.linspace(-1,1,9,dtype=torch.float64)
    # Both first-level operator values and graph parameters influence coarse energy.
    objective=ac.apply(probe).square().mean()+weights.square().mean()
    objective.backward()
    assert at.values.grad is not None and torch.isfinite(at.values.grad).all()
    assert at.values.grad.abs().sum()>0
    assert model.message_layers[0].message[0].weight.grad.abs().sum()>0
    assert model.delta_head.weight.grad.abs().sum()>0
    # Actual coarse connectivity includes farther-than-one local graph edges.
    coarse_pattern=build_transfer_pattern((3,3),(1,1),coarsest_n=1)
    next_delta=model.forward_graph(ac,coarse_pattern)
    assert next_delta.shape==(1,16,3,3)


def test_generator_never_allocates_dense_operator(monkeypatch):
    a=operator(15); pattern=build_transfer_pattern(15)
    def forbidden(*args,**kwargs):
        raise AssertionError("dense matrix conversion forbidden")
    monkeypatch.setattr(sp.csr_matrix,"toarray",forbidden)
    monkeypatch.setattr(sp.coo_matrix,"toarray",forbidden)
    monkeypatch.setattr(torch.Tensor,"to_dense",forbidden)
    original_zeros=torch.zeros
    def bounded_zeros(*args,**kwargs):
        dimensions=args[0] if args and isinstance(args[0],(tuple,list)) else args
        assert tuple(dimensions)!=(225,225)
        return original_zeros(*args,**kwargs)
    monkeypatch.setattr(torch,"zeros",bounded_zeros)
    graph=build_transfer_graph(a,pattern)
    assert graph.nodes.shape==(225,9)
    assert graph.edges.shape==(a.nnz,9)
    assert graph.candidate_features.shape==(225,16,10)
    model=make_graph_transfer(width=8,layers=1)
    model.forward_graph(a,pattern).sum().backward()


def test_far_sparse_connectivity_and_operator_scale_are_observed():
    a=operator(); pattern=build_transfer_pattern(7)
    # Symmetric PSD far-edge update outside the nine-point stencil.
    v=sp.csr_matrix(([1.,-1.],([0,48],[0,0])),shape=(49,1))
    b=a+3*(v@v.T)
    graph=build_transfer_graph(b,pattern,dtype=torch.float64)
    assert ((graph.rows==0)&(graph.cols==48)).any()
    torch.testing.assert_close(graph.nodes,build_transfer_graph(b*1e100,pattern,dtype=torch.float64).nodes)
    torch.testing.assert_close(graph.edges,build_transfer_graph(b*1e-100,pattern,dtype=torch.float64).edges)


def test_bounded_expanded_support_preserves_selected_baseline():
    a=operator(15); original=build_transfer_pattern(15)
    model=make_graph_transfer(support="expanded")
    expanded,base=transfer_pattern_for_model(original,model,original.bilinear_weights)
    assert expanded.columns.shape==(225,36)
    p=scipy_prolongation_from_weights(original,original.bilinear_weights)
    expanded_p=scipy_prolongation_from_weights(expanded,base)
    assert (p-expanded_p).nnz==0
    assert model.forward_graph(a,expanded,base).shape==(1,36,15,15)
    np.testing.assert_array_equal(expanded.bilinear_weights.sum(1),original.bilinear_weights.sum(1))


def test_teacher_targets_are_detached_and_direct_student_independent():
    torch.manual_seed(11)
    a=operator();pattern=build_transfer_pattern(7)
    teacher=nonzero_decoder(make_graph_transfer("gnn_teacher",width=12,layers=3)).double()
    student=make_graph_transfer(width=8,layers=1).double()
    direct=make_graph_transfer(width=8,layers=1).double()
    direct_before={k:v.clone() for k,v in direct.state_dict().items()}
    s=student.forward_graph(a,pattern,return_details=True)
    t=teacher.forward_graph(a,pattern,return_details=True)
    loss=transfer_distillation_loss(s,t)
    assert loss>0
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert student.delta_head.weight.grad.abs().sum()>0
    assert student.node_encoder[0].weight.grad.abs().sum()>0
    assert all(torch.equal(direct_before[k],v) for k,v in direct.state_dict().items())


def test_actual_nnz_caps_reject_fill_and_do_not_count_explicit_zeros():
    a=operator();pattern=build_transfer_pattern(7)
    base=scipy_prolongation_from_weights(pattern,pattern.bilinear_weights)
    ac=galerkin_coarse_operator(a,base)
    report=enforce_transfer_complexity(a,base,ac,baseline_p=base,baseline_ac=ac)
    assert report["p_ratio"]==1 and report["ac_ratio"]==1
    model=nonzero_decoder(make_graph_transfer()).double()
    p=scipy_prolongation_from_weights(pattern,weights_from_deltas_torch(pattern,model.forward_graph(a,pattern)).detach().numpy())
    learned_ac=galerkin_coarse_operator(a,p)
    with pytest.raises(TransferComplexityError,match="max_p_ratio") as failure:
        enforce_transfer_complexity(a,p,learned_ac,baseline_p=base,baseline_ac=ac,caps=TransferComplexityCaps(max_p_ratio=1))
    assert failure.value.report["p_nnz"]>base.nnz
    with pytest.raises(TransferComplexityError,match="max_row_nnz"):
        enforce_transfer_complexity(a,p,learned_ac,caps=TransferComplexityCaps(max_row_nnz=1))
    symbolic=sp.coo_matrix((np.r_[base.data,0.],(np.r_[base.tocoo().row,0],np.r_[base.tocoo().col,8])),shape=base.shape).tocsr()
    assert enforce_transfer_complexity(a,symbolic,ac)["p_nnz"]==base.nnz


def test_differentiable_complexity_proxy_has_nonzero_weight_gradient():
    a=operator();pattern=build_transfer_pattern(7)
    values=torch.tensor(pattern.bilinear_weights,dtype=torch.float64,requires_grad=True)
    perturbed=values+torch.as_tensor(pattern.columns>=0)*.02
    proxies=differentiable_transfer_cost(perturbed,pattern,a=SparseTensor.from_scipy(a))
    assert proxies["support"]>1 and proxies["galerkin_work"]>1
    proxies["loss"].backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum()>0


@pytest.mark.parametrize("kwargs", [dict(layers=3),dict(architecture="gnn_teacher",layers=2),dict(architecture="edge_mlp",layers=1),dict(width=0),dict(max_delta=float("nan")),dict(schema=7),dict(training_only=True)])
def test_invalid_architecture_contract_is_rejected(kwargs):
    with pytest.raises(ValueError):
        make_graph_transfer(**kwargs)


def test_expanded_projection_prunes_and_matches_inference_with_gradients():
    torch.manual_seed(3)
    a=operator(15); original=build_transfer_pattern(15)
    model=nonzero_decoder(make_graph_transfer(support="expanded",width=8,layers=1)).double()
    pattern,base=transfer_pattern_for_model(original,model,original.bilinear_weights)
    delta=model.forward_graph(a,pattern,base)
    weights=project_transfer_weights(model,pattern,delta,base)
    assert torch.max(torch.count_nonzero(weights,dim=1))<=16
    np.testing.assert_allclose(weights.detach().sum(1).numpy(),base.sum(1),rtol=0,atol=1e-15)
    np.testing.assert_array_equal(weights.detach().numpy()[coarse_fine_indices(pattern)],base[coarse_fine_indices(pattern)])
    with torch.inference_mode():
        inference=project_transfer_weights(model,pattern,model.forward_graph(a,pattern,base),base)
    torch.testing.assert_close(weights,inference,rtol=0,atol=1e-16)
    weights.square().sum().backward()
    assert model.delta_head.weight.grad.abs().sum()>0
    p=scipy_prolongation_from_weights(pattern,weights.detach().numpy())
    ac=galerkin_coarse_operator(a,p)
    enforce_transfer_complexity(a,p,ac,caps=model.complexity_caps)


def test_complexity_budget_is_checkpoint_identity_and_validated():
    one=make_graph_transfer(complexity_caps={"max_row_nnz":8})
    two=make_graph_transfer(complexity_caps={"max_row_nnz":16})
    assert one.architecture_config()!=two.architecture_config()
    assert make_graph_transfer(**one.architecture_config()).complexity_caps["max_row_nnz"]==8
    with pytest.raises(ValueError,match="complexity caps"):
        make_graph_transfer(complexity_caps={"max_row_nnz":1.5})


def test_explicit_tight_standard_support_cap_is_enforced_without_changing_default():
    a=operator(15);pattern=build_transfer_pattern(15)
    net=nonzero_decoder(make_graph_transfer(width=8,layers=1,complexity_caps={"max_row_nnz":8})).double()
    delta=net.forward_graph(a,pattern)
    weights=project_transfer_weights(net,pattern,delta,pattern.bilinear_weights)
    assert torch.max(torch.count_nonzero(weights,dim=1))<=8
    np.testing.assert_allclose(weights.detach().sum(1).numpy(),pattern.bilinear_weights.sum(1),atol=1e-15,rtol=0)
    default=make_graph_transfer(width=8,layers=1).double()
    torch.testing.assert_close(project_transfer_weights(default,pattern,delta,pattern.bilinear_weights),weights_from_deltas_torch(pattern,delta),rtol=0,atol=0)
