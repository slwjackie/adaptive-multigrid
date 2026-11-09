"""P1 FEM on seeded, unstructured Delaunay triangles of the unit square.

No structured element connectivity, quadrilateral element or finite difference
assembly is used here. Seven-point positive triangle quadrature integrates both
the diffusion tensor and the physical load. Discontinuous coefficient interfaces
need not follow element edges; their integrals are quadrature approximations.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay

DISCRETIZATION = 'p1_unstructured'
MESH_VERSION = 'jittered-delaunay-square-v1'

# Degree-five Dunavant rule; weights sum to one (multiply by physical area).
BARYCENTRIC = np.array([
    [1/3, 1/3, 1/3],
    [.059715871789770, .470142064105115, .470142064105115],
    [.470142064105115, .059715871789770, .470142064105115],
    [.470142064105115, .470142064105115, .059715871789770],
    [.797426985353087, .101286507323456, .101286507323456],
    [.101286507323456, .797426985353087, .101286507323456],
    [.101286507323456, .101286507323456, .797426985353087],
])
QUADRATURE_WEIGHTS = np.array([.225] + [.132394152788506] * 3 + [.125939180544827] * 3)


def element_geometry(points, triangles):
    vertices = np.asarray(points, dtype=np.float64)[triangles]
    x, y = vertices[..., 0], vertices[..., 1]
    determinant = (x[:, 1]-x[:, 0])*(y[:, 2]-y[:, 0]) - (x[:, 2]-x[:, 0])*(y[:, 1]-y[:, 0])
    if not np.isfinite(determinant).all() or np.any(np.abs(determinant) <= 1e-15):
        raise ValueError('Degenerate or nonfinite triangle')
    gradients = np.stack((y[:, [1, 2, 0]]-y[:, [2, 0, 1]],
                          x[:, [2, 0, 1]]-x[:, [1, 2, 0]]), axis=-1) / determinant[:, None, None]
    return np.abs(determinant) / 2, gradients


@dataclass(frozen=True)
class TriangleMesh:
    points: np.ndarray
    triangles: np.ndarray
    boundary: np.ndarray

    def __post_init__(self):
        points = np.asarray(self.points, dtype=np.float64).copy()
        triangles = np.asarray(self.triangles)
        boundary = np.asarray(self.boundary)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError('Expected finite two-dimensional mesh vertices')
        if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles) or triangles.dtype.kind not in 'iu':
            raise ValueError('Expected integer triangular connectivity')
        triangles = triangles.astype(np.int64, copy=True)
        if np.any(triangles < 0) or np.any(triangles >= len(points)):
            raise ValueError('Triangle index outside vertex array')
        if boundary.shape != (len(points),) or boundary.dtype != bool:
            raise ValueError('Expected one boolean boundary flag per vertex')
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError('Duplicate mesh vertices')
        if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
            raise ValueError('Duplicate mesh triangles')
        if len(np.unique(triangles)) != len(points):
            raise ValueError('Mesh contains unused vertices')
        element_geometry(points, triangles)
        edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        if np.any(counts > 2):
            raise ValueError('Nonmanifold triangular mesh')
        expected_boundary = np.zeros(len(points), dtype=bool)
        expected_boundary[np.unique(unique[counts == 1])] = True
        if not np.array_equal(expected_boundary, boundary):
            raise ValueError('Boundary flags differ from mesh boundary edges')
        adjacency = sp.coo_matrix((np.ones(len(unique)), (unique[:, 0], unique[:, 1])),
                                  shape=(len(points), len(points))).tocsr()
        if connected_components(adjacency, directed=False, return_labels=False) != 1:
            raise ValueError('Disconnected mesh')
        for name, value in (('points', points), ('triangles', triangles), ('boundary', boundary.copy())):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def interior(self):
        return np.flatnonzero(~self.boundary)

    def digest(self):
        h = hashlib.sha256(MESH_VERSION.encode())
        for array in (self.points.astype('<f8'), self.triangles.astype('<i8'), self.boundary.astype('u1')):
            h.update(np.asarray(array.shape, dtype='<i8').tobytes())
            h.update(array.tobytes())
        return h.hexdigest()

    def statistics(self):
        vertices = self.points[self.triangles]
        lengths = np.linalg.norm(vertices[:, [1, 2, 0]]-vertices[:, [2, 0, 1]], axis=2)
        a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
        angles = np.arccos(np.clip(np.stack(((b*b+c*c-a*a)/(2*b*c),
            (a*a+c*c-b*b)/(2*a*c), (a*a+b*b-c*c)/(2*a*b)), axis=1), -1, 1))
        area, _ = element_geometry(self.points, self.triangles)
        return dict(vertices=len(self.points), triangles=len(self.triangles),
                    interior_unknowns=len(self.interior), boundary_vertices=int(self.boundary.sum()),
                    min_angle_degrees=float(np.rad2deg(angles.min())),
                    max_edge_length=float(lengths.max()), min_area=float(area.min()),
                    domain_area=float(area.sum()))


def generate_mesh(n, seed, *, jitter=.3, grading=1.):
    """Randomize interior vertices, triangulate them, and permute vertex IDs.

    n controls target point density (n² interior vertices), not matrix reshaping
    or fixed triangle connectivity. Positive jitter is mandatory. Grading > 1
    concentrates points toward the left/bottom boundaries for mesh OOD tests.
    """
    if type(n) is not int or n < 2:
        raise ValueError('Mesh resolution n must be an integer >= 2')
    if not np.isfinite(jitter) or not 0 < jitter <= .45:
        raise ValueError('Unstructured mesh requires 0 < jitter <= .45')
    if not np.isfinite(grading) or not .7 <= grading <= 2.:
        raise ValueError('Mesh grading must lie in [.7, 2]')
    rng = np.random.default_rng(seed)
    h = 1. / (n + 1)
    x, y = np.meshgrid(np.arange(1, n+1)*h, np.arange(1, n+1)*h, indexing='ij')
    interior = np.column_stack((x.ravel(), y.ravel()))
    interior += rng.uniform(-jitter*h, jitter*h, interior.shape)
    t = np.arange(n+1)*h
    boundary_points = np.concatenate((np.column_stack((t, np.zeros_like(t))),
        np.column_stack((np.ones_like(t), t)), np.column_stack((1-t, np.ones_like(t))),
        np.column_stack((np.zeros_like(t), 1-t))))
    points = np.concatenate((interior, boundary_points)) ** grading
    boundary = np.arange(len(points)) >= len(interior)
    permutation = rng.permutation(len(points))
    points, boundary = points[permutation], boundary[permutation]
    delaunay = Delaunay(points)
    if len(delaunay.coplanar):
        raise ValueError('Delaunay omitted vertices; choose a different mesh seed')
    triangles = delaunay.simplices.copy()
    v = points[triangles]
    signed = (v[:, 1, 0]-v[:, 0, 0])*(v[:, 2, 1]-v[:, 0, 1])-(v[:, 2, 0]-v[:, 0, 0])*(v[:, 1, 1]-v[:, 0, 1])
    triangles[signed < 0] = triangles[signed < 0][:, [0, 2, 1]]
    # Canonical storage order for reproducible mesh digests.
    triangles = np.take_along_axis(triangles, (np.arange(3)[None, :]+triangles.argmin(1)[:, None]) % 3, axis=1)
    triangles = triangles[np.lexsort(triangles[:, ::-1].T)]
    mesh = TriangleMesh(points, triangles, boundary)
    if not np.isclose(mesh.statistics()['domain_area'], 1., rtol=0, atol=1e-12):
        raise ValueError('Delaunay triangles do not cover the unit square')
    return mesh


def _evaluate(field, coordinates, tail=()):
    raw = field(coordinates[..., 0], coordinates[..., 1]) if callable(field) else field
    result = np.broadcast_to(np.asarray(raw, dtype=np.float64), coordinates.shape[:-1] + tail)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite FEM coefficient or source')
    return result


def element_matrices(mesh, tensor_field, forcing=1.):
    area, gradients = element_geometry(mesh.points, mesh.triangles)
    coordinates = np.einsum('qi,tid->tqd', BARYCENTRIC, mesh.points[mesh.triangles])
    tensors = _evaluate(tensor_field, coordinates, (2, 2))
    if not np.allclose(tensors, tensors.swapaxes(-1, -2), rtol=1e-12, atol=1e-14):
        raise ValueError('Diffusion tensor must be symmetric')
    if np.any(tensors[..., 0, 0] <= 0) or np.any(np.linalg.det(tensors) <= 0):
        raise ValueError('Diffusion tensor must be positive definite')
    average = np.einsum('q,tqde->tde', QUADRATURE_WEIGHTS, tensors)
    stiffness = area[:, None, None] * np.einsum('tid,tde,tje->tij', gradients, average, gradients)
    source = _evaluate(forcing, coordinates)
    load = area[:, None] * np.einsum('q,tq,qi->ti', QUADRATURE_WEIGHTS, source, BARYCENTRIC)
    return stiffness, load


@dataclass
class FEMSystem:
    mesh: TriangleMesh
    a: sp.csr_matrix
    b: np.ndarray
    full_load: np.ndarray
    boundary_values: np.ndarray

    def lift(self, interior_solution):
        values = self.boundary_values.copy()
        values[self.mesh.interior] = interior_solution
        return values


def assemble_p1(mesh, tensor_field, forcing=1., dirichlet=0.):
    """Assemble weak-form stiffness/load and eliminate Dirichlet unknowns.

    K^T_ab = integral_T grad(phi_a)^T K(x) grad(phi_b) dx;
    f^T_a = integral_T f(x) phi_a dx; b_I = f_I - A_IB g_B.
    """
    if not len(mesh.interior):
        raise ValueError('P1 solve requires interior unknowns')
    stiffness, load = element_matrices(mesh, tensor_field, forcing)
    triangles = mesh.triangles
    rows = np.broadcast_to(triangles[:, :, None], stiffness.shape).ravel()
    cols = np.broadcast_to(triangles[:, None, :], stiffness.shape).ravel()
    full = sp.coo_matrix((stiffness.ravel(), (rows, cols)), shape=(len(mesh.points),)*2).tocsr()
    full = (.5*(full+full.T)).tocsr()
    full_load = np.bincount(triangles.ravel(), weights=load.ravel(), minlength=len(mesh.points))
    boundary_values = np.zeros(len(mesh.points))
    boundary_values[mesh.boundary] = _evaluate(dirichlet, mesh.points[mesh.boundary])
    interior = mesh.interior
    a = full[interior][:, interior].tocsr()
    a.eliminate_zeros(); a.sort_indices()
    b = np.asarray(full_load[interior] - (full @ boundary_values)[interior])
    return FEMSystem(mesh, a, b, full_load, boundary_values)


def error_norms(system, interior_solution, exact, exact_gradient):
    """Quadrature L2 and H1-seminorm errors against a smooth exact solution."""
    mesh = system.mesh
    values = system.lift(interior_solution)[mesh.triangles]
    area, gradients = element_geometry(mesh.points, mesh.triangles)
    coordinates = np.einsum('qi,tid->tqd', BARYCENTRIC, mesh.points[mesh.triangles])
    discrete = values @ BARYCENTRIC.T
    difference = discrete - _evaluate(exact, coordinates)
    discrete_gradient = np.einsum('ti,tid->td', values, gradients)
    grad_difference = discrete_gradient[:, None, :] - _evaluate(exact_gradient, coordinates, (2,))
    return dict(l2=float(np.sqrt(np.einsum('t,q,tq->', area, QUADRATURE_WEIGHTS, difference**2))),
                h1_seminorm=float(np.sqrt(np.einsum('t,q,tqd->', area, QUADRATURE_WEIGHTS, grad_difference**2))))
