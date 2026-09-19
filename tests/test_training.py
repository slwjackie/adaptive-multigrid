import math

import pytest
import torch

import adaptive_mg.torch_ops as torch_ops
import adaptive_mg.train as train_module
from adaptive_mg.models import CHECKPOINT_FORMAT_VERSION, NeuralComponents
from adaptive_mg.strategy import get_strategy
from adaptive_mg.torch_ops import full_v_cycle, sample_error
from adaptive_mg.train import TrainingConfig, build_training_pools, train_staged


def test_training_pool_keeps_roots_and_full_v_cycle_builds_galerkin_levels(monkeypatch) -> None:
    config = TrainingConfig(
        n=15,
        pool_size=1,
        validation_pool_size=1,
        samples_per_epoch=1,
        selector_epochs=0,
        smoother_epochs=0,
        transfer_epochs=0,
        combined_epochs=0,
        detector_epochs=0,
        adaptive_epochs=0,
        seed=2,
    )
    train_pool, validation_pool = build_training_pools(config)
    assert [sample.n for sample in train_pool] == [15]
    assert [sample.n for sample in validation_pool] == [15]

    child_shapes: list[tuple[int, int]] = []
    original = torch_ops.make_child_problem

    def record_child(a: torch.Tensor, shape: tuple[int, int]):
        child_shapes.append(shape)
        return original(a, shape)

    monkeypatch.setattr(torch_ops, "make_child_problem", record_child)
    sample = train_pool[0]
    components = NeuralComponents.create(hidden=8)
    error = sample_error(
        sample.problem.shape,
        device=sample.problem.device,
        dtype=sample.problem.dtype,
        problem=sample.problem,
        smooth_steps=1,
        smoother="line_alt",
    )
    full_v_cycle(
        sample.problem,
        error,
        torch.zeros_like(error),
        components,
        mode="classical",
        strategy=get_strategy("line_alt_bilinear_full"),
        coarsest_n=3,
    )
    assert child_shapes == [(7, 7), (3, 3)]


def test_staged_training_checkpoint_contains_common_cross_mode_validation(tmp_path) -> None:
    components = NeuralComponents.create(hidden=8)
    config = TrainingConfig(
        n=7,
        pool_size=1,
        validation_pool_size=1,
        samples_per_epoch=1,
        validation_samples_per_problem=1,
        cycles_per_sample=1,
        selector_epochs=0,
        smoother_epochs=1,
        transfer_epochs=1,
        combined_epochs=1,
        detector_epochs=1,
        adaptive_epochs=0,
        seed=3,
    )
    checkpoint = tmp_path / "smoke.pt"
    records = train_staged(components, config, checkpoint=checkpoint, verbose=False)
    assert checkpoint.exists()
    assert len(records) == 4
    loaded, metadata = NeuralComponents.load(checkpoint)
    assert loaded is not None
    assert metadata["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    # Selector summary is retained even when selector training is disabled.
    assert len(metadata["stage_summaries"]) == 5
    assert all("accepted" in summary for summary in metadata["stage_summaries"])
    common = metadata["common_validation"]
    assert set(common) == {"protocol", "aggregate", "by_grid"}
    assert set(common["aggregate"]) == {
        "classical", "nn_smoother", "learned_transfer", "combined", "adaptive"
    }
    assert "7" in common["by_grid"]
    assert metadata["full_v_cycle_training"] is True


def test_cross_grid_validation_is_required_for_research_configuration() -> None:
    with pytest.raises(ValueError, match="cross-grid validation"):
        TrainingConfig(
            train_sizes=(15, 31),
            validation_sizes=(15, 31),
            ood_validation_sizes=(),
        ).validate()
    TrainingConfig(
        train_sizes=(15, 31),
        validation_sizes=(15, 31, 63),
        ood_validation_sizes=(),
    ).validate()


def test_validation_reports_geometric_mean_consistent_with_log_objective(monkeypatch) -> None:
    config = TrainingConfig(
        n=7,
        pool_size=2,
        validation_pool_size=2,
        validation_samples_per_problem=1,
        selector_epochs=0,
        smoother_epochs=0,
        transfer_epochs=0,
        combined_epochs=0,
        detector_epochs=0,
        adaptive_epochs=0,
    )
    _, pool = build_training_pools(config)
    contractions = {id(pool[0]): 0.1, id(pool[1]): 0.9}

    def fake_objective(sample, components, mode, cfg, **kwargs):
        value = contractions[id(sample)]
        tensor = torch.tensor(value, dtype=sample.problem.dtype)
        return torch.log(tensor), tensor, torch.zeros_like(tensor)

    monkeypatch.setattr(train_module, "_cycle_objective", fake_objective)
    objective, rho, _ = train_module._evaluate_cycle_mode(
        NeuralComponents.create(hidden=8),
        pool,
        "classical",
        config,
        smooth_error_steps=1,
    )
    assert rho == pytest.approx(math.sqrt(0.1 * 0.9), rel=1e-7)
    assert objective == pytest.approx(math.log(rho), rel=1e-7)


def test_float64_training_path_matches_model_dtype() -> None:
    config = TrainingConfig(
        n=7,
        dtype="float64",
        pool_size=1,
        validation_pool_size=1,
        samples_per_epoch=1,
        selector_epochs=0,
        smoother_epochs=0,
        transfer_epochs=0,
        combined_epochs=0,
        detector_epochs=0,
        adaptive_epochs=0,
    )
    components = NeuralComponents.create(hidden=8).to("cpu", dtype=torch.float64)
    train_pool, _ = build_training_pools(config)
    sample = train_pool[0]
    error = sample_error(
        sample.problem.shape,
        device=sample.problem.device,
        dtype=sample.problem.dtype,
    )
    output, _ = full_v_cycle(
        sample.problem,
        error,
        torch.zeros_like(error),
        components,
        mode="learned_transfer",
        strategy=get_strategy("line_alt_bilinear_full"),
    )
    assert output.dtype == torch.float64


def test_legacy_checkpoint_is_rejected_by_default(tmp_path) -> None:
    components = NeuralComponents.create(hidden=8)
    legacy = tmp_path / "legacy.pt"
    torch.save({"models": components.state_dict(), "metadata": {}}, legacy)
    with pytest.raises(ValueError, match="checkpoint format"):
        NeuralComponents.load(legacy)


def test_research_training_rejects_two_grid_root_samples() -> None:
    with pytest.raises(ValueError, match="three-grid full V-cycle"):
        TrainingConfig(
            train_sizes=(7, 15),
            validation_sizes=(15, 31),
            ood_validation_sizes=(),
        ).validate()
    TrainingConfig(
        train_sizes=(7, 15),
        validation_sizes=(15, 31),
        ood_validation_sizes=(),
        allow_two_grid_training=True,
    ).validate()


def test_differentiable_classical_v_cycle_matches_production_hierarchy() -> None:
    import numpy as np
    from adaptive_mg.pde import DiffusionCase, assemble_stiffness
    from adaptive_mg.solver import MGConfig, solve
    from adaptive_mg.torch_ops import make_torch_problem

    case = DiffusionCase(n=15, epsilon=1.0e-3, angle_deg=45.0)
    a = assemble_stiffness(case)
    rng = np.random.default_rng(19)
    error = rng.normal(size=a.shape[0])
    error /= np.linalg.norm(error)
    components = NeuralComponents.create(hidden=8).to("cpu", dtype=torch.float64)
    production = solve(
        a,
        np.zeros_like(error),
        case.n,
        components,
        MGConfig(
            mode="classical",
            strategy_name="jacobi_bilinear_full",
            max_cycles=1,
            tolerance=0.0,
            safety_fallback=False,
            smoother_safeguard=False,
        ),
        x0=error,
    ).x
    problem = make_torch_problem(a, case.n, dtype=torch.float64)
    differentiable, _ = full_v_cycle(
        problem,
        torch.as_tensor(error, dtype=torch.float64),
        torch.zeros(error.size, dtype=torch.float64),
        components,
        mode="classical",
        strategy=get_strategy("jacobi_bilinear_full"),
    )
    assert np.allclose(differentiable.detach().numpy(), production, atol=1.0e-13)


def test_torch_sparse_operator_baseline_matches_scipy_without_dense_matrix() -> None:
    import numpy as np
    from adaptive_mg.pde import DiffusionCase, assemble_stiffness
    from adaptive_mg.torch_ops import make_torch_problem, operator_dependent_weights_torch
    from adaptive_mg.transfer import build_transfer_pattern, operator_dependent_weights

    case = DiffusionCase(
        n=15, epsilon=1.0e-3, angle_deg=33.0,
        contrast=1.0e3, pattern="local_patch"
    )
    a = assemble_stiffness(case)
    problem = make_torch_problem(a, case.n)
    for coarsening in ("full", "semi_x", "semi_y", "alternating"):
        pattern = build_transfer_pattern(case.n, coarsening=coarsening)
        torch_weights = operator_dependent_weights_torch(problem, pattern).cpu().numpy()
        scipy_weights = operator_dependent_weights(
            a, case.n, coarse=pattern.coarse_shape
        )
        assert np.allclose(torch_weights, scipy_weights, atol=3.0e-6, rtol=3.0e-6)
