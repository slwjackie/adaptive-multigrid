import pytest
from adaptive_mg.models import NeuralComponents
from adaptive_mg.pde import DiffusionCase, assemble_problem
from adaptive_mg.solver import MGConfig, solve


def test_krylov_acceleration_and_flexible_methods() -> None:
    case = DiffusionCase(n=15, epsilon=1.0e-3, angle_deg=45.0, pattern="uniform")
    a, b, _ = assemble_problem(case)
    components = NeuralComponents.create(hidden=8)
    stationary = solve(
        a,
        b,
        case.n,
        components,
        MGConfig(
            mode="classical",
            smoother="jacobi",
            max_cycles=8,
            tolerance=0.0,
            outer_solver="none",
        ),
    )
    for method in ("cg", "fcg", "fgmres"):
        accelerated = solve(
            a,
            b,
            case.n,
            components,
            MGConfig(
                mode="classical",
                smoother="jacobi",
                max_cycles=8,
                tolerance=0.0,
                outer_solver=method,  # type: ignore[arg-type]
                krylov_restart=4,
            ),
        )
        assert accelerated.relative_residual < stationary.relative_residual
        assert accelerated.outer_solver == method


def test_adaptive_fcg_and_fgmres_run_with_variable_preconditioner() -> None:
    case = DiffusionCase(
        n=7, epsilon=1.0e-3, angle_deg=45.0, contrast=1.0e3, pattern="local_patch"
    )
    a, b, _ = assemble_problem(case)
    components = NeuralComponents.create(hidden=8)
    for method in ("fcg", "fgmres"):
        result = solve(
            a,
            b,
            case.n,
            components,
            MGConfig(
                mode="adaptive",
                smoother="line_alt",
                max_cycles=3,
                tolerance=0.0,
                outer_solver=method,  # type: ignore[arg-type]
                safety_fallback=True,
                krylov_restart=2,
            ),
        )
        assert result.relative_residual < 1.0
        assert result.nn_calls > 0
        assert result.safety_work_units > 0.0


def test_fgmres_keeps_true_and_estimated_residual_histories_separate() -> None:
    import numpy as np
    from adaptive_mg.krylov import fgmres

    diagonal = np.array([1.0, 2.0, 5.0, 11.0, 23.0])
    b = np.arange(1.0, 6.0)
    result = fgmres(
        lambda x: diagonal * x,
        lambda r: r / np.sqrt(diagonal),
        b,
        tolerance=1.0e-12,
        max_iterations=12,
        restart=2,
    )
    true_final = np.linalg.norm(b - diagonal * result.x)
    assert result.residual_history[-1] == pytest.approx(true_final, rel=1e-12, abs=1e-14)
    assert len(result.estimated_residual_history) == result.iterations + 1
    assert len(result.residual_history) <= result.iterations + 1
    assert result.estimated_residual_history is not result.residual_history
    assert len(result.residual_iteration_indices) == len(result.residual_history)
    assert result.residual_iteration_indices[0] == 0
    assert result.residual_iteration_indices[-1] == result.iterations
    assert all(
        right > left
        for left, right in zip(
            result.residual_iteration_indices, result.residual_iteration_indices[1:]
        )
    )


def test_fcg_converges_with_iteration_dependent_positive_preconditioner() -> None:
    import numpy as np
    from adaptive_mg.krylov import fcg

    diagonal = np.array([1.0, 3.0, 7.0, 13.0, 29.0, 41.0])
    b = np.linspace(1.0, 2.0, diagonal.size)
    calls = {"count": 0}

    def variable_preconditioner(r: np.ndarray) -> np.ndarray:
        calls["count"] += 1
        exponent = 0.35 if calls["count"] % 2 else 0.65
        return r / diagonal**exponent

    result = fcg(
        lambda x: diagonal * x,
        variable_preconditioner,
        b,
        tolerance=1.0e-11,
        max_iterations=20,
    )
    assert np.linalg.norm(b - diagonal * result.x) < 1.0e-9
    assert result.residual_history[-1] < result.residual_history[0]
