import copy

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from neural_amg.fem import (TriangleMesh, generate_mesh, element_matrices,
    assemble_p1, error_norms, DISCRETIZATION)
from neural_amg.data import FAMILIES, materialize
from neural_amg import study


def test_triangle_element_exact_stiffness_load_and_orientation():
    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    expected = np.array([[1., -.5, -.5], [-.5, .5, 0], [-.5, 0, .5]])
    for order in ([0, 1, 2], [0, 2, 1]):
        mesh = TriangleMesh(points, np.array([order]), np.ones(3, dtype=bool))
        stiffness, load = element_matrices(mesh, np.eye(2))
        np.testing.assert_allclose(stiffness[0], expected[np.ix_(order, order)], atol=2e-15)
        np.testing.assert_allclose(load[0], np.full(3, 1/6), atol=2e-15)


def test_mesh_is_reproducible_nonrectangular_and_has_valid_boundary():
    mesh = generate_mesh(9, 40)
    repeated, different = generate_mesh(9, 40), generate_mesh(9, 41)
    assert mesh.digest() == repeated.digest() != different.digest()
    assert not np.array_equal(mesh.triangles, different.triangles)
    assert mesh.triangles.shape[1] == 3
    assert len(mesh.interior) == 81
    # Interior coordinates are not on the original Cartesian point lattice.
    coords = mesh.points[mesh.interior]*10
    assert np.all(np.any(np.abs(coords-np.round(coords)) > 1e-8, axis=1))
    stats = mesh.statistics()
    assert stats['domain_area'] == pytest.approx(1.)
    assert stats['min_angle_degrees'] > 5
    boundary_points = mesh.points[mesh.boundary]
    assert np.all(np.any(np.isclose(boundary_points, 0)|np.isclose(boundary_points, 1), axis=1))
    with pytest.raises(ValueError, match='jitter'):
        generate_mesh(9, 40, jitter=0)


def test_affine_patch_and_nonzero_dirichlet_elimination():
    mesh = generate_mesh(7, 123)
    exact = lambda x, y: 1+2*x-3*y
    system = assemble_p1(mesh, np.array([[2., .3], [.3, 1.]]), forcing=0., dirichlet=exact)
    solution = spsolve(system.a, system.b)
    np.testing.assert_allclose(system.lift(solution), exact(*mesh.points.T), atol=2e-13)
    np.linalg.cholesky(system.a.toarray())


def test_physical_load_is_integrated_and_differs_from_discrete_manufacture():
    mesh = generate_mesh(7, 4)
    system = assemble_p1(mesh, np.eye(2), forcing=1.)
    assert system.full_load.sum() == pytest.approx(1.)
    assert np.all(system.b > 0)
    assert system.b.sum() < 1.  # Boundary basis loads are eliminated, not reassigned.
    exact_nodes = np.sin(np.pi*mesh.points[mesh.interior, 0])*np.sin(np.pi*mesh.points[mesh.interior, 1])
    assert not np.allclose(system.b, system.a @ exact_nodes)


def test_smooth_poisson_p1_l2_and_h1_convergence():
    exact = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)
    gradient = lambda x, y: np.stack((np.pi*np.cos(np.pi*x)*np.sin(np.pi*y),
                                    np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)), axis=-1)
    errors = []
    for n in (7, 15, 31):
        system = assemble_p1(generate_mesh(n, 5), np.eye(2), lambda x, y: 2*np.pi**2*exact(x, y))
        errors.append(error_norms(system, spsolve(system.a, system.b), exact, gradient))
    for previous, current in zip(errors[:-1], errors[1:]):
        assert previous['l2']/current['l2'] > 3.3
        assert previous['h1_seminorm']/current['h1_seminorm'] > 1.8


@pytest.mark.parametrize('family', FAMILIES)
def test_all_seven_families_assemble_unstructured_p1_spd(family):
    config = study.load_config('configs/neural_amg_smoke.json')
    config['data']['families'] = [family]
    spec = study._specifications(config)['train'][0]
    assert spec['discretization'] == DISCRETIZATION
    system = materialize(spec)
    assert system.mesh.triangles.shape == (128, 3)
    np.testing.assert_allclose(system.a.toarray(), system.a.T.toarray(), atol=1e-12)
    np.linalg.cholesky(system.a.toarray())
    assert np.isfinite(spsolve(system.a, system.b)).all()


def test_mesh_ood_changes_geometry_controls():
    config = study.load_config('configs/neural_amg_research.json')
    plan = study._specifications(config)
    assert len(plan['train']) == 560 and len(plan['validation']) == 280
    assert plan['ood_mesh'][0]['mesh']['grading'] == 1.5
    assert plan['train'][0]['mesh']['grading'] == 1.
    seeds = [s['mesh']['seed'] for rows in plan.values() for s in rows]
    assert len(set(seeds)) == len(seeds)


def test_non_spd_tensors_and_bad_triangles_rejected():
    mesh = generate_mesh(3, 12)
    with pytest.raises(ValueError, match='positive definite'):
        assemble_p1(mesh, np.diag([1., -1.]))
    with pytest.raises(ValueError, match='symmetric'):
        assemble_p1(mesh, np.array([[1., .2], [0., 1.]]))
    triangles = mesh.triangles.copy(); triangles[0, 1] = triangles[0, 0]
    with pytest.raises(ValueError, match='Degenerate'):
        TriangleMesh(mesh.points, triangles, mesh.boundary)
