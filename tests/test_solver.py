import numpy as np
import pytest
import torch

from adaptive_mg.models import NeuralComponents
from adaptive_mg.pde import DiffusionCase, assemble_problem
from adaptive_mg.solver import (
    MGConfig,
    WorkCounter,
    _classical_correction,
    _exact_safeguard_correction,
    _make_level,
    solve,
)


@pytest.mark.parametrize("smoother", ["jacobi", "line_x", "line_alt"])
@pytest.mark.parametrize("transfer", ["bilinear", "operator"])
def test_untrained_neural_ablations_reproduce_selected_classical_baseline(
    smoother: str, transfer: str
) -> None:
    case = DiffusionCase(n=7, epsilon=1.0e-2, angle_deg=45.0, pattern="uniform")
    a, b, _ = assemble_problem(case)
    components = NeuralComponents.create(hidden=12)
    results = {}
    for mode in ("classical", "nn_smoother", "learned_transfer", "combined"):
        results[mode] = solve(
            a,
            b,
            case.n,
            components,
            MGConfig(
                mode=mode,  # type: ignore[arg-type]
                smoother=smoother,  # type: ignore[arg-type]
                transfer_baseline=transfer,  # type: ignore[arg-type]
                max_cycles=2,
                tolerance=0.0,
            ),
        )
    baseline = results["classical"].residual_history
    for mode in ("nn_smoother", "learned_transfer", "combined"):
        assert np.allclose(results[mode].residual_history, baseline, rtol=2.0e-5, atol=1.0e-10)


def test_line_relaxation_is_a_strong_aligned_anisotropy_baseline() -> None:
    case = DiffusionCase(n=15, epsilon=1.0e-3, angle_deg=0.0, pattern="uniform")
    a, b, _ = assemble_problem(case)
    components = NeuralComponents.create(hidden=8)
    jacobi = solve(
        a,
        b,
        case.n,
        components,
        MGConfig(mode="classical", smoother="jacobi", max_cycles=3, tolerance=0.0),
    )
    line = solve(
        a,
        b,
        case.n,
        components,
        MGConfig(mode="classical", smoother="line_x", max_cycles=3, tolerance=0.0),
    )
    assert line.relative_residual < jacobi.relative_residual * 1.0e-4


def test_exact_safeguard_is_never_worse_than_classical_correction() -> None:
    rng = np.random.default_rng(3)
    case = DiffusionCase(n=7, epsilon=1.0e-3, angle_deg=45.0, pattern="uniform")
    a, b, _ = assemble_problem(case)
    config = MGConfig(smoother="line_alt", transfer_baseline="bilinear")
    level = _make_level(a, case.n, config)
    work = WorkCounter(a.nnz)
    classical = _classical_correction(level, b, config, work, reverse=False)
    direction = rng.normal(size=b.size)
    safeguarded, alpha = _exact_safeguard_correction(
        level, b, classical, direction, work
    )
    assert 0.0 <= alpha <= 1.0
    assert np.linalg.norm(b - a @ safeguarded) <= np.linalg.norm(b - a @ classical) * (1.0 + 1.0e-12)


def test_work_accounting_and_invalid_n() -> None:
    case = DiffusionCase(n=7)
    a, b, _ = assemble_problem(case)
    result = solve(a, b, case.n, NeuralComponents.create(hidden=8), MGConfig(max_cycles=1))
    assert result.work_units > 0.0
    assert result.matvec_count > 0
    bad = DiffusionCase(n=9)
    a_bad, b_bad, _ = assemble_problem(bad)
    with pytest.raises(ValueError, match=r"n=2\*\*L-1"):
        solve(a_bad, b_bad, bad.n, config=MGConfig())


def test_adaptive_cycle_fallback_is_no_worse_than_same_state_classical_cycle() -> None:
    case = DiffusionCase(
        n=7, epsilon=1.0e-3, angle_deg=41.0, contrast=1.0e3, pattern="local_patch"
    )
    a, b, _ = assemble_problem(case)
    components = NeuralComponents.create(hidden=8)
    with torch.no_grad():
        torch.nn.init.normal_(components.transfer.head.weight, std=0.02)
        torch.nn.init.normal_(components.transfer.head.bias, std=0.02)
        torch.nn.init.normal_(components.smoother.head.weight, std=0.02)
        torch.nn.init.normal_(components.smoother.head.bias, std=0.02)
        torch.nn.init.normal_(components.detector.head.weight, std=0.02)
        components.detector.head.bias.zero_()
    classical = solve(
        a, b, case.n, components,
        MGConfig(mode="classical", smoother="line_alt", max_cycles=1, tolerance=0.0),
    )
    adaptive = solve(
        a, b, case.n, components,
        MGConfig(
            mode="adaptive", smoother="line_alt", max_cycles=1, tolerance=0.0,
            safety_fallback=True,
        ),
    )
    assert adaptive.residual_history[-1] <= classical.residual_history[-1] * (1.0 + 1.0e-10)
