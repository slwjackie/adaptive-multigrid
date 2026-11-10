"""Sparse arbitrary-graph Galerkin V-cycles with matching runtime/autograd paths.

Classical discrete C/S are detached once per hierarchy construction. Gradients
pass through learned P, every Galerkin product, feature extraction and smoothing.
No fine-grid dense matrix, geometry reshape or CNN stencil truncation is used.
"""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch

from neural_amg.autograd_sparse import SparseTensor
from .structure import build_structure


@dataclass
class Level:
    a: SparseTensor
    p: SparseTensor | None = None
    structure: object = None
    omega: torch.Tensor | None = None
    diagonal: torch.Tensor | None = None
    teacher_report: dict | None = None


@dataclass
class Hierarchy:
    levels: list
    setup_seconds: float
    pre_steps: int = 2
    post_steps: int = 2
    method: str = 'learned'
    runtime: list = field(default_factory=list)
    terminal_lu: object = None

    @property
    def operator_complexity(self):
        return sum(level.a.pattern.nnz for level in self.levels) / self.levels[0].a.pattern.nnz


def _smooth_data(a):
    diagonal = a.diagonal()
    if not bool(torch.isfinite(diagonal).all()) or not bool((diagonal > 0).all()):
        raise ValueError('AMG requires a finite positive diagonal')
    rows = torch.as_tensor(a.rows, dtype=torch.long)
    sums = torch.zeros_like(diagonal).index_add(0, rows, a.values.abs())
    # Gershgorin upper bound for rho(D^-1 A); differentiable in the values.
    omega = .8 / (sums / diagonal).max()
    return diagonal, omega


def build_hierarchy(A, model=None, method='learned', *, max_levels=10,
                    max_coarse=9, strength=10., seed=0, max_hops=2,
                    pre_steps=2, post_steps=2, structures=None,
                    max_operator_complexity=20., teacher_options=None):
    """Build hierarchy, including feature/NN/Galerkin/coarse-factor setup cost.

    Optional structures freeze an existing sequence for controlled comparisons
    or gradient checks. Otherwise the same classical rule is rerun on each
    actual generated coarse A (not on a cached geometric/reference hierarchy).
    """
    start = perf_counter()
    if method not in ('learned', 'classical', 'teacher'):
        raise ValueError('Unknown interpolation method')
    if method == 'learned' and model is None:
        raise ValueError('learned hierarchy requires a model')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (max_levels, max_coarse, pre_steps, post_steps)):
        raise ValueError('Hierarchy limits and smoothing counts must be positive integers')
    if not np.isfinite(max_operator_complexity) or max_operator_complexity < 1:
        raise ValueError('Invalid operator complexity budget')
    if not sp.issparse(A) or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError('Expected a nonempty square sparse SPD operator')
    matrix = sp.csr_matrix(A, dtype=np.float64, copy=True)
    matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
    delta = matrix - matrix.T
    if not np.isfinite(matrix.data).all() or (delta.nnz and np.max(abs(delta.data)) > 1e-11 * max(1., np.max(abs(matrix.data)))):
        raise ValueError('Expected finite symmetric A')
    # Positive diagonal and symmetry do not certify SPD; SPD is an input contract.
    a = SparseTensor.from_scipy(matrix)
    levels, runtime = [], []
    for index in range(max_levels):
        diagonal, omega = _smooth_data(a)
        level = Level(a=a, diagonal=diagonal, omega=omega)
        levels.append(level)
        if sum(x.a.pattern.nnz for x in levels) / matrix.nnz > max_operator_complexity:
            raise ValueError('Hierarchy operator complexity budget exceeded')
        if a.shape[0] <= max_coarse:
            break
        if index == max_levels - 1:
            raise ValueError('max_levels reached before max_coarse; increase limit explicitly')
        structure = (structures[index] if structures is not None else
                     build_structure(a.numpy(), max_hops=max_hops))
        if structure.reference.shape[0] != a.shape[0]:
            raise ValueError('Frozen structure dimension differs from generated A')
        if not 0 < structure.reference.shape[1] < a.shape[0]:
            raise ValueError('Classical coarsening failed to reduce unknown count')
        level.structure = structure
        if method == 'classical':
            p = SparseTensor.from_scipy(structure.reference)
        elif method == 'teacher':
            from .teacher import fit_teacher
            options = dict(teacher_options or {})
            options.setdefault('bootstrap_max_levels', max_levels)
            result = fit_teacher(a.numpy(), structure, strength=strength,
                                 seed=seed + index, **options)
            p = SparseTensor.from_scipy(result.p)
            level.teacher_report = result.report
        else:
            p = model(a, structure)
        if not bool(torch.isfinite(p.values).all()):
            raise FloatingPointError('Nonfinite interpolation')
        level.p = p
        a = p.transpose().product(a.product(p)).symmetrize()
    # Runtime path is ordinary sparse arithmetic, with all factors cached here.
    for level in levels:
        runtime.append(dict(a=level.a.numpy(),
                            p=None if level.p is None else level.p.numpy(),
                            diagonal=level.diagonal.detach().numpy(),
                            omega=float(level.omega.detach())))
    terminal_lu = sla.splu(runtime[-1]['a'].tocsc())
    return Hierarchy(levels, perf_counter() - start, pre_steps, post_steps,
                     method, runtime, terminal_lu)


def vcycle(hierarchy, b, x=None, level=0):
    """Differentiable V-cycle; b may contain multiple RHS/error probes."""
    entry = hierarchy.levels[level]
    if entry.p is None:
        return entry.a.solve(b)
    x = torch.zeros_like(b) if x is None else x
    inverse = entry.omega / entry.diagonal
    if b.ndim == 2:
        inverse = inverse[:, None]
    for _ in range(hierarchy.pre_steps):
        x = x + inverse * (b - entry.a.apply(x))
    residual = b - entry.a.apply(x)
    coarse_rhs = entry.p.transpose().apply(residual)
    x = x + entry.p.apply(vcycle(hierarchy, coarse_rhs, level=level + 1))
    for _ in range(hierarchy.post_steps):
        x = x + inverse * (b - entry.a.apply(x))
    return x


def _runtime_cycle(hierarchy, b, x=None, level=0):
    entry = hierarchy.runtime[level]
    if entry['p'] is None:
        return hierarchy.terminal_lu.solve(b)
    x = np.zeros_like(b) if x is None else x
    inverse = entry['omega'] / entry['diagonal']
    if b.ndim == 2:
        inverse = inverse[:, None]
    for _ in range(hierarchy.pre_steps):
        x = x + inverse * (b - entry['a'] @ x)
    r = b - entry['a'] @ x
    x = x + entry['p'] @ _runtime_cycle(hierarchy, entry['p'].T @ r, level=level + 1)
    for _ in range(hierarchy.post_steps):
        x = x + inverse * (b - entry['a'] @ x)
    return x


def solve(hierarchy, b, x0=None, *, rtol=1e-8, atol=1e-12, max_cycles=100):
    """Deployment solve timing excludes setup, includes residual checks."""
    start = perf_counter()
    b = np.asarray(b, dtype=np.float64)
    if b.shape != (hierarchy.levels[0].a.shape[0],) or not np.isfinite(b).all():
        raise ValueError('Expected one finite RHS of matching size')
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0 or max(rtol, atol) == 0:
        raise ValueError('Invalid tolerance')
    if isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or max_cycles < 0:
        raise ValueError('Invalid cycle budget')
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    if x.shape != b.shape or not np.isfinite(x).all():
        raise ValueError('Invalid initial solution')
    a = hierarchy.runtime[0]['a']
    residuals = [float(np.linalg.norm(b - a @ x))]
    tolerance = max(atol, rtol * residuals[0])
    times = [perf_counter() - start]
    for _ in range(max_cycles):
        if residuals[-1] <= tolerance or not np.isfinite(residuals[-1]):
            break
        x = _runtime_cycle(hierarchy, b, x)
        residuals.append(float(np.linalg.norm(b - a @ x)))
        times.append(perf_counter() - start)
    return dict(x=x, converged=bool(np.isfinite(residuals[-1]) and residuals[-1] <= tolerance),
                cycles=len(residuals) - 1, residuals=residuals,
                solve_seconds=perf_counter() - start, residual_seconds=times,
                tolerance=tolerance)


def multicycle_loss(hierarchy, probes, cycles=3):
    """Mean log relative residual for homogeneous equations, nonzero errors.

    Setup/runtime seconds are measured for model selection, not misrepresented
    as differentiable wall-clock losses. All levels remain in the autograd graph.
    """
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError('cycles must be a positive integer')
    x = torch.as_tensor(probes, dtype=torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != hierarchy.levels[0].a.shape[0] or not bool(torch.isfinite(x).all()):
        raise ValueError('Invalid error probes')
    a = hierarchy.levels[0].a
    initial = torch.linalg.vector_norm(a.apply(x), dim=0)
    if not bool((initial > 0).all()):
        raise ValueError('Error probes must have nonzero residual')
    b = torch.zeros_like(x)
    terms = []
    for _ in range(cycles):
        x = vcycle(hierarchy, b, x)
        residual = torch.linalg.vector_norm(a.apply(x), dim=0)
        terms.append(torch.log(residual / initial + 1e-12).mean())
    return torch.stack(terms).mean()
