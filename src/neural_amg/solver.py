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
from .structure import CoarseningError, build_structure


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
    guard_events: list = field(default_factory=list)
    autograd_checked: set = field(default_factory=set)
    runtime_checked: set = field(default_factory=set)
    first_cycle_max_growth: float = 10.

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
    if not bool(torch.isfinite(sums).all()) or not bool(torch.isfinite(omega)) or float(omega.detach()) <= 0:
        raise NumericalHierarchyError('Nonfinite or degenerate smoother scaling')
    return diagonal, omega



class NumericalHierarchyError(FloatingPointError):
    """A generated operator or its direct solve failed numerical checks."""


def _check_operator(matrix):
    if not np.isfinite(matrix.data).all():
        raise NumericalHierarchyError('Nonfinite Galerkin operator')
    diagonal = matrix.diagonal()
    if not np.isfinite(diagonal).all() or not (diagonal > 0).all():
        raise NumericalHierarchyError('Galerkin operator requires a positive diagonal')
    difference = matrix - matrix.T
    scale = max(1., float(np.max(np.abs(matrix.data))))
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-11 * scale:
        raise NumericalHierarchyError('Nonsymmetric Galerkin operator')


def _check_interpolation(p, structure):
    if not isinstance(p, SparseTensor) or p.shape != structure.shape:
        raise NumericalHierarchyError('Interpolation dimensions differ from its support')
    matrix = p.numpy()
    if not np.isfinite(matrix.data).all():
        raise NumericalHierarchyError('Nonfinite interpolation')
    reference = structure.reference
    if (not np.array_equal(matrix.indptr, reference.indptr)
            or not np.array_equal(matrix.indices, reference.indices)):
        raise NumericalHierarchyError('Interpolation changed the fixed support')
    target = structure.row_targets
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.allclose(sums, target, rtol=1e-10, atol=1e-12):
        raise NumericalHierarchyError('Interpolation does not preserve reference row sums')
    injection = matrix[structure.coarse] - sp.eye(p.shape[1], format='csr')
    if injection.nnz and np.max(np.abs(injection.data)) > 1e-12:
        raise NumericalHierarchyError('Interpolation does not preserve coarse injection')


def _checked_factor(a):
    # Share the factor with SparseTensor.solve so setup does not factor twice.
    values = a.values
    key = (id(values), values._version, values.data_ptr(), str(values.dtype), str(values.device))
    if a.cache.get('lu_key') == key:
        return a.cache['lu']
    try:
        lu = sla.splu(a.numpy().tocsc())
    except (RuntimeError, ValueError) as error:
        raise NumericalHierarchyError(f'Coarse LU factorization failed: {error}') from error
    if (not np.isfinite(lu.L.data).all() or not np.isfinite(lu.U.data).all()
            or np.any(lu.U.diagonal() == 0)):
        raise NumericalHierarchyError('Coarse LU has nonfinite factors or zero pivots')
    a.cache.update(lu=lu, lu_key=key,
                   factorizations=a.cache.get('factorizations', 0) + 1)
    return lu


def _truncate(hierarchy, index, stage, reason, **details):
    """Replace this level and all descendants by a cached direct terminal solve."""
    level = hierarchy.levels[index]
    lu = _checked_factor(level.a)
    level.p = None
    level.structure = None
    del hierarchy.levels[index + 1:]
    if hierarchy.runtime:
        hierarchy.runtime[index]['p'] = None
        del hierarchy.runtime[index + 1:]
    hierarchy.terminal_lu = lu
    hierarchy.guard_events.append(dict(stage=stage, level=index, n=level.a.shape[0],
                                       reason=str(reason), **details))


def _direct_result_check(a, b, x):
    if not np.isfinite(x).all():
        raise NumericalHierarchyError('Coarse LU solve returned nonfinite values')
    residual = b - a @ x
    if not np.isfinite(residual).all():
        raise NumericalHierarchyError('Coarse LU solve has nonfinite residual')
    norm_a = float(np.max(np.asarray(abs(a).sum(axis=1)).ravel()))
    norm_x = np.max(np.abs(x), axis=0)
    norm_b = np.max(np.abs(b), axis=0)
    bound = 1e-10 * (norm_a * norm_x + norm_b) + np.finfo(float).tiny
    if np.any(np.max(np.abs(residual), axis=0) > bound):
        raise NumericalHierarchyError('Coarse LU solve failed its backward residual check')



def _torch_direct(entry, b):
    try:
        result = entry.a.solve(b)
    except (RuntimeError, ValueError, FloatingPointError) as error:
        raise NumericalHierarchyError(f'Coarse LU solve failed: {error}') from error
    _direct_result_check(entry.a.numpy(), b.detach().numpy(), result.detach().numpy())
    return result


def _numpy_direct(hierarchy, entry, b):
    try:
        result = hierarchy.terminal_lu.solve(b)
    except (RuntimeError, ValueError, FloatingPointError) as error:
        raise NumericalHierarchyError(f'Coarse LU solve failed: {error}') from error
    _direct_result_check(entry['a'], b, result)
    return result

def _first_cycle_bad(a, b, initial, result, max_growth=10.):
    """Check actual cycle operands against a numerical growth budget.

    Residual norms may initially grow even for convergent AMG. The configurable
    budget is a numerical safeguard, not a certificate of convergence.
    """
    before = np.linalg.norm(b - a @ initial, axis=0)
    after = np.linalg.norm(b - a @ result, axis=0)
    rounding = 1e-12 * (1 + np.linalg.norm(b, axis=0) + np.linalg.norm(a @ initial, axis=0))
    bad = (not np.isfinite(result).all() or not np.isfinite(after).all()
           or np.any(after > before * max_growth * (1 + 1e-8) + rounding))
    return bad, dict(first_cycle_max_growth=float(max_growth),
        residual_before=[float(v) if np.isfinite(v) else None for v in np.atleast_1d(before)],
        residual_after=[float(v) if np.isfinite(v) else None for v in np.atleast_1d(after)])

def build_hierarchy(A, model=None, method='learned', *, max_levels=10,
                    max_coarse=9, strength=10., seed=0, max_hops=2,
                    pre_steps=2, post_steps=2, structures=None,
                    max_operator_complexity=20., teacher_options=None,
                    first_cycle_max_growth=10.):
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
    if (isinstance(first_cycle_max_growth, bool) or not np.isfinite(first_cycle_max_growth)
            or first_cycle_max_growth < 1):
        raise ValueError('first_cycle_max_growth must be finite and at least 1')
    if not sp.issparse(A) or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError('Expected a nonempty square sparse SPD operator')
    matrix = sp.csr_matrix(A, dtype=np.float64, copy=True)
    matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
    delta = matrix - matrix.T
    if not np.isfinite(matrix.data).all() or (delta.nnz and np.max(abs(delta.data)) > 1e-11 * max(1., np.max(abs(matrix.data)))):
        raise ValueError('Expected finite symmetric A')
    # Positive diagonal and symmetry do not certify SPD; SPD is an input contract.
    a = SparseTensor.from_scipy(matrix)
    levels, runtime, events = [], [], []
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
        try:
            structure = (structures[index] if structures is not None else
                         build_structure(a.numpy(), max_hops=max_hops))
        except CoarseningError as error:
            events.append(dict(stage='setup', level=index, n=a.shape[0], reason=str(error)))
            break
        if structure.reference.shape[0] != a.shape[0]:
            raise ValueError('Frozen structure dimension differs from generated A')
        if not 0 < structure.reference.shape[1] < a.shape[0]:
            raise ValueError('Classical coarsening failed to reduce unknown count')
        level.structure = structure
        try:
            if method == 'classical':
                p = SparseTensor.from_scipy(structure.reference)
            elif method == 'teacher':
                from .teacher import fit_teacher
                options = dict(teacher_options or {})
                options.setdefault('bootstrap_max_levels', max_levels)
                options.setdefault('bootstrap_first_cycle_max_growth', first_cycle_max_growth)
                result = fit_teacher(a.numpy(), structure, strength=strength,
                                     seed=seed + index, **options)
                p = SparseTensor.from_scipy(result.p)
                level.teacher_report = result.report
            else:
                p = model(a, structure)
            _check_interpolation(p, structure)
            coarse_a = p.transpose().product(a.product(p)).symmetrize()
            _check_operator(coarse_a.numpy())
            _smooth_data(coarse_a)
        except (NumericalHierarchyError, FloatingPointError) as error:
            level.structure = None
            events.append(dict(stage='setup', level=index, n=a.shape[0], reason=str(error)))
            break
        level.p = p
        a = coarse_a
    # Runtime path is ordinary sparse arithmetic, with all factors cached here.
    for level in levels:
        runtime.append(dict(a=level.a.numpy(),
                            p=None if level.p is None else level.p.numpy(),
                            diagonal=level.diagonal.detach().numpy(),
                            omega=float(level.omega.detach())))
    hierarchy = Hierarchy(levels, 0., pre_steps, post_steps, method, runtime,
                          guard_events=events, first_cycle_max_growth=float(first_cycle_max_growth))
    while True:
        try:
            hierarchy.terminal_lu = _checked_factor(hierarchy.levels[-1].a)
            break
        except NumericalHierarchyError as error:
            if len(hierarchy.levels) == 1:
                raise
            rejected = len(hierarchy.levels) - 1
            hierarchy.guard_events.append(dict(stage='setup', level=rejected,
                n=hierarchy.levels[-1].a.shape[0], reason=str(error), action='reject_terminal'))
            hierarchy.levels.pop()
            hierarchy.runtime.pop()
            hierarchy.levels[-1].p = None
            hierarchy.levels[-1].structure = None
            hierarchy.runtime[-1]['p'] = None
    hierarchy.setup_seconds = perf_counter() - start
    return hierarchy


def vcycle(hierarchy, b, x=None, level=0):
    """Differentiable V-cycle with checks on the first actual cycle at each level."""
    entry = hierarchy.levels[level]
    if entry.p is None:
        return _torch_direct(entry, b)
    x = torch.zeros_like(b) if x is None else x
    initial = x
    first = level not in hierarchy.autograd_checked
    inverse = entry.omega / entry.diagonal
    if b.ndim == 2:
        inverse = inverse[:, None]
    try:
        for _ in range(hierarchy.pre_steps):
            x = x + inverse * (b - entry.a.apply(x))
        residual = b - entry.a.apply(x)
        coarse_rhs = entry.p.transpose().apply(residual)
        x = x + entry.p.apply(vcycle(hierarchy, coarse_rhs, level=level + 1))
        for _ in range(hierarchy.post_steps):
            x = x + inverse * (b - entry.a.apply(x))
        if first:
            bad, details = _first_cycle_bad(entry.a.numpy(), b.detach().numpy(),
                                           initial.detach().numpy(), x.detach().numpy(),
                                           hierarchy.first_cycle_max_growth)
            if bad:
                _truncate(hierarchy, level, 'first_cycle', 'Actual first V-cycle residual exceeded growth budget or became nonfinite', **details)
                x = _torch_direct(entry, b)
        if not bool(torch.isfinite(x).all()):
            raise NumericalHierarchyError('V-cycle returned nonfinite values')
        hierarchy.autograd_checked.add(level)
        return x
    except NumericalHierarchyError as error:
        _truncate(hierarchy, level, 'cycle_solve', str(error))
        return _torch_direct(entry, b)


def _runtime_cycle(hierarchy, b, x=None, level=0):
    entry = hierarchy.runtime[level]
    if entry['p'] is None:
        return _numpy_direct(hierarchy, entry, b)
    x = np.zeros_like(b) if x is None else x
    initial = x
    first = level not in hierarchy.runtime_checked
    inverse = entry['omega'] / entry['diagonal']
    if b.ndim == 2:
        inverse = inverse[:, None]
    try:
        for _ in range(hierarchy.pre_steps):
            x = x + inverse * (b - entry['a'] @ x)
        r = b - entry['a'] @ x
        x = x + entry['p'] @ _runtime_cycle(hierarchy, entry['p'].T @ r, level=level + 1)
        for _ in range(hierarchy.post_steps):
            x = x + inverse * (b - entry['a'] @ x)
        if first:
            bad, details = _first_cycle_bad(entry['a'], b, initial, x,
                                           hierarchy.first_cycle_max_growth)
            if bad:
                _truncate(hierarchy, level, 'first_cycle', 'Actual first V-cycle residual exceeded growth budget or became nonfinite', **details)
                x = _numpy_direct(hierarchy, entry, b)
        if not np.isfinite(x).all():
            raise NumericalHierarchyError('V-cycle returned nonfinite values')
        hierarchy.runtime_checked.add(level)
        return x
    except NumericalHierarchyError as error:
        _truncate(hierarchy, level, 'cycle_solve', str(error))
        return _numpy_direct(hierarchy, entry, b)


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
    hierarchy.runtime_checked.clear()
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
                tolerance=tolerance, guard_events=list(hierarchy.guard_events),
                level_sizes=[level.a.shape[0] for level in hierarchy.levels],
                terminal_size=hierarchy.levels[-1].a.shape[0],
                operator_complexity=hierarchy.operator_complexity)


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
    hierarchy.autograd_checked.clear()
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
