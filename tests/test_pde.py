import numpy as np

from adaptive_mg.pde import DiffusionCase, assemble_problem


def test_anisotropic_operator_is_spd() -> None:
    case = DiffusionCase(n=7, epsilon=1.0e-3, angle_deg=37.0, contrast=1.0e3, pattern="local_patch")
    a, b, exact = assemble_problem(case)
    dense = a.toarray()
    assert np.allclose(dense, dense.T, atol=1.0e-12)
    assert np.linalg.eigvalsh(dense).min() > 0.0
    assert np.allclose(b, a @ exact)
