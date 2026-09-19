import numpy as np

from adaptive_mg.diagnostics import detector_map
from adaptive_mg.models import NeuralComponents
from adaptive_mg.pde import DiffusionCase, assemble_problem
from adaptive_mg.solver import MGConfig, solve


def test_adaptive_gate_and_safety_fallback() -> None:
    case = DiffusionCase(n=7, epsilon=1.0e-3, angle_deg=45.0, contrast=1.0e3, pattern="local_patch")
    a, b, _ = assemble_problem(case)
    components = NeuralComponents.create(hidden=16)
    gate = detector_map(a, b, case.n, components)
    assert gate.shape == (case.n, case.n)
    assert np.all((0.0 <= gate) & (gate <= 1.0))

    classical = solve(a, b, case.n, components, MGConfig(mode="classical", max_cycles=2, tolerance=0.0))
    adaptive = solve(
        a,
        b,
        case.n,
        components,
        MGConfig(mode="adaptive", max_cycles=2, tolerance=0.0, safety_fallback=True),
    )
    assert adaptive.residual_history[-1] <= classical.residual_history[-1] * (1.0 + 1.0e-10)
