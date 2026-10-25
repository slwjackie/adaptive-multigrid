"""Q1 assembly checked against analytical elements, PDE convergence and P1."""
from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from adaptive_mg.pde import (
    DiffusionCase, assemble_problem, assemble_q1_stiffness,
    assemble_q1_stiffness_reference, assemble_stiffness,
    assemble_stiffness_reference,
)


def _constant_element_reference(n, tensor):
    """Closed-form integrated element; deliberately no quadrature calls."""
    xx = np.array([[2, -2, -1, 1], [-2, 2, 1, -1],
                   [-1, 1, 2, -2], [1, -1, -2, 2]]) / 6.0
    yy = np.array([[2, 1, -1, -2], [1, 2, -2, -1],
                   [-1, -2, 2, 1], [-2, -1, 1, 2]]) / 6.0
    xy = np.array([[1, 0, -1, 0], [0, -1, 0, 1],
                   [-1, 0, 1, 0], [0, 1, 0, -1]]) / 2.0
    element = tensor[0, 0] * xx + tensor[1, 1] * yy + tensor[0, 1] * xy
    assembled = np.zeros((n * n, n * n))
    for i in range(n + 1):
        for j in range(n + 1):
            nodes = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            for a, (ia, ja) in enumerate(nodes):
                for b, (ib, jb) in enumerate(nodes):
                    if 1 <= ia <= n and 1 <= ja <= n and 1 <= ib <= n and 1 <= jb <= n:
                        assembled[(ia - 1) * n + ja - 1, (ib - 1) * n + jb - 1] += element[a, b]
    return assembled


def test_q1_isotropic_nine_point_stencil_and_dirichlet_elimination():
    n = 7
    a = assemble_stiffness(DiffusionCase(n=n, epsilon=1., discretization="q1"))
    expected = np.full((3, 3), -1.0 / 3.0)
    expected[1, 1] = 8.0 / 3.0
    row = a.getrow(3 * n + 3).toarray().reshape(n, n)
    np.testing.assert_allclose(row[2:5, 2:5], expected, atol=1e-14)
    assert np.count_nonzero(row) == 9
    assert a.shape == (n * n, n * n)
    # Boundary nodes were removed, not treated as additional unknowns.
    assert a.getrow(0).nnz == 4
    np.testing.assert_allclose(a[0, 0], 8.0 / 3.0)
    np.testing.assert_allclose(a.getrow(0).sum(), 5.0 / 3.0)


@pytest.mark.parametrize("tensor", [np.eye(2), np.array([[2., .65], [.65, .7]]),
                                   np.array([[2., -.65], [-.65, .7]])])
def test_q1_constant_tensor_matches_closed_form_element(tensor):
    n = 4
    actual = assemble_q1_stiffness(n, lambda x, y: tensor)
    np.testing.assert_allclose(actual.toarray(), _constant_element_reference(n, tensor), atol=2e-14)


@pytest.mark.parametrize("pattern", ["uniform", "local_patch", "checkerboard", "channel"])
@pytest.mark.parametrize("n", [3, 7])
def test_q1_reference_parity_symmetry_and_positive_definiteness(pattern, n):
    case = DiffusionCase(n=n, epsilon=.003, angle_deg=-27., contrast=43.,
                         pattern=pattern, channel_width=.113,
                         patch_bounds=(.23, .64, .17, .79), discretization="q1")
    a = assemble_stiffness(case)
    reference = assemble_stiffness_reference(case)
    np.testing.assert_allclose(a.toarray(), reference.toarray(), rtol=3e-14, atol=3e-14)
    assert sp.isspmatrix_csr(a)
    assert (a - a.T).nnz == 0
    assert np.linalg.eigvalsh(a.toarray()).min() > 0.
    _, b, exact = assemble_problem(case)
    np.testing.assert_allclose(spsolve(a, b), exact, rtol=1e-11, atol=1e-11)


def test_q1_vectorized_callback_samples_four_points_per_cell():
    calls = []
    def field(x, y):
        x, y = np.asarray(x), np.asarray(y)
        calls.append((x.copy(), y.copy()))
        return (1.0 + x + 2 * y)[..., None, None] * np.array([[2., .4], [.4, 1.]])
    vectorized = assemble_q1_stiffness(7, field)
    assert len(calls) == 4
    assert all(x.shape == y.shape == (64,) for x, y in calls)
    points = np.concatenate([np.stack((x, y), axis=-1) for x, y in calls])
    assert np.unique(points, axis=0).shape == (256, 2)
    scalar = assemble_q1_stiffness_reference(7, field)
    np.testing.assert_allclose(vectorized.toarray(), scalar.toarray(), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("pattern", ["uniform", "local_patch", "checkerboard", "channel"])
def test_default_p1_matches_explicit_p1_and_original_reference(pattern):
    case = DiffusionCase(n=7, epsilon=.04, angle_deg=31., contrast=53., pattern=pattern)
    default = assemble_stiffness(case)
    explicit = assemble_stiffness(replace(case, discretization="p1"))
    assert np.array_equal(default.indptr, explicit.indptr)
    assert np.array_equal(default.indices, explicit.indices)
    assert np.array_equal(default.data, explicit.data)
    np.testing.assert_allclose(default.toarray(), assemble_stiffness_reference(case).toarray(),
                               rtol=1e-13, atol=1e-13)


def test_q1_has_expected_second_order_error_for_continuum_poisson():
    # Unlike b=A*u, an independently integrated physical load tests consistency
    # with -Delta u = 2*pi**2*sin(pi*x)*sin(pi*y), u|boundary=0.
    errors = []
    points, weights = np.polynomial.legendre.leggauss(8)
    for n in (7, 15, 31):
        h = 1.0 / (n + 1)
        coords = np.arange(1, n + 1) * h
        integrals = np.zeros(n)
        for side in (-1, 1):
            t = 0.5 * (points + 1.0)
            integrals += 0.5 * h * np.sum(weights * (1.0 - t) *
                np.sin(np.pi * (coords[:, None] + side * h * t)), axis=1)
        b = (2.0 * np.pi**2 * np.outer(integrals, integrals)).ravel()
        a = assemble_stiffness(DiffusionCase(n=n, epsilon=1., discretization="q1"))
        exact = np.outer(np.sin(np.pi * coords), np.sin(np.pi * coords)).ravel()
        errors.append(np.linalg.norm(spsolve(a, b) - exact) / np.linalg.norm(exact))
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((ratios > 3.8) & (ratios < 4.2)), (errors, ratios)


@pytest.mark.parametrize("method", ["Q1", "triangle", "", None, [], {}])
def test_unknown_discretization_rejected(method):
    case = DiffusionCase(n=3, discretization=method)
    for assemble in (assemble_stiffness, assemble_stiffness_reference, assemble_problem):
        with pytest.raises(ValueError, match="discretization"):
            assemble(case)


@pytest.mark.parametrize("n", [0, -1, True, 3.5])
def test_generic_q1_assembler_rejects_invalid_grid_size(n):
    for assemble in (assemble_q1_stiffness, assemble_q1_stiffness_reference):
        with pytest.raises(ValueError, match="positive integer"):
            assemble(n, lambda x, y: np.eye(2))
