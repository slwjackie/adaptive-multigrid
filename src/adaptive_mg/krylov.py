"""CG, flexible CG, and restarted flexible GMRES implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

KrylovMethod = Literal["cg", "fcg", "fgmres"]
VectorMap = Callable[[np.ndarray], np.ndarray]


@dataclass
class KrylovResult:
    x: np.ndarray
    converged: bool
    iterations: int
    # True residual norms only.  For FGMRES these are measured at restart/update
    # points rather than mixing Arnoldi estimates with true norms.
    residual_history: list[float]
    estimated_residual_history: list[float] = field(default_factory=list)
    # Krylov iteration associated with each true residual measurement.
    residual_iteration_indices: list[int] = field(default_factory=list)


def _initial_state(
    matvec: VectorMap,
    b: np.ndarray,
    x0: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    residual = np.asarray(b - matvec(x), dtype=np.float64)
    return x, residual, float(np.linalg.norm(residual))


def _target_norm(b: np.ndarray, tolerance: float) -> float:
    b_norm = float(np.linalg.norm(b))
    requested = max(float(tolerance), 0.0) * max(b_norm, 1.0e-300)
    floor = 50.0 * np.finfo(np.float64).eps * max(b_norm, 1.0)
    return max(requested, floor)


def pcg(
    matvec: VectorMap,
    precondition: VectorMap,
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
) -> KrylovResult:
    """Preconditioned conjugate gradients for fixed SPD preconditioners."""

    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    b = np.asarray(b, dtype=np.float64)
    x, residual, initial_norm = _initial_state(matvec, b, x0)
    history = [initial_norm]
    target = _target_norm(b, tolerance)
    if initial_norm <= target or max_iterations == 0:
        return KrylovResult(x, initial_norm <= target, 0, history, history.copy(), [0])

    z = np.asarray(precondition(residual), dtype=np.float64)
    rho = float(np.dot(residual, z))
    if not np.isfinite(rho) or rho <= 0.0:
        raise RuntimeError("CG preconditioner is not positive definite")
    direction = z.copy()

    for iteration in range(1, max_iterations + 1):
        a_direction = np.asarray(matvec(direction), dtype=np.float64)
        denominator = float(np.dot(direction, a_direction))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("CG encountered non-positive p^T A p")
        alpha = rho / denominator
        x = x + alpha * direction
        residual = residual - alpha * a_direction
        residual_norm = float(np.linalg.norm(residual))
        history.append(residual_norm)
        if residual_norm <= target:
            return KrylovResult(x, True, iteration, history, history.copy(), list(range(len(history))))

        z_new = np.asarray(precondition(residual), dtype=np.float64)
        rho_new = float(np.dot(residual, z_new))
        if not np.isfinite(rho_new) or rho_new <= 0.0:
            raise RuntimeError("CG preconditioner lost positive definiteness")
        direction = z_new + (rho_new / rho) * direction
        rho = rho_new

    return KrylovResult(x, False, max_iterations, history, history.copy(), list(range(len(history))))


def fcg(
    matvec: VectorMap,
    precondition: VectorMap,
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
) -> KrylovResult:
    """Flexible CG with full A-orthogonalization.

    Search directions are restarted with the current preconditioned residual,
    then steepest descent if needed.  The numerator ``rho`` is recomputed after
    either restart, avoiding the stale-numerator bug present in v2.
    """

    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    b = np.asarray(b, dtype=np.float64)
    x, residual, initial_norm = _initial_state(matvec, b, x0)
    history = [initial_norm]
    target = _target_norm(b, tolerance)
    if initial_norm <= target or max_iterations == 0:
        return KrylovResult(x, initial_norm <= target, 0, history, history.copy(), [0])

    directions: list[np.ndarray] = []
    a_directions: list[np.ndarray] = []
    denominators: list[float] = []

    for iteration in range(1, max_iterations + 1):
        z = np.asarray(precondition(residual), dtype=np.float64)
        rho = float(np.dot(residual, z))
        if not np.isfinite(rho) or rho <= 0.0:
            z = residual.copy()
            rho = float(np.dot(residual, z))

        direction = z.copy()
        for _ in range(2):
            for old_direction, old_a_direction, old_denominator in zip(
                directions, a_directions, denominators
            ):
                coefficient = float(np.dot(direction, old_a_direction)) / old_denominator
                direction -= coefficient * old_direction

        a_direction = np.asarray(matvec(direction), dtype=np.float64)
        denominator = float(np.dot(direction, a_direction))
        if not np.isfinite(denominator) or denominator <= 1.0e-30:
            if history[-1] <= target:
                return KrylovResult(x, True, iteration - 1, history, history.copy(), list(range(len(history))))
            direction = z.copy()
            rho = float(np.dot(residual, direction))
            a_direction = np.asarray(matvec(direction), dtype=np.float64)
            denominator = float(np.dot(direction, a_direction))
            if not np.isfinite(denominator) or denominator <= 1.0e-30 or rho <= 0.0:
                direction = residual.copy()
                rho = float(np.dot(residual, direction))
                a_direction = np.asarray(matvec(direction), dtype=np.float64)
                denominator = float(np.dot(direction, a_direction))
            if (
                not np.isfinite(denominator)
                or denominator <= 1.0e-30
                or not np.isfinite(rho)
                or rho <= 0.0
            ):
                raise RuntimeError("FCG encountered a non-positive search direction")
            directions.clear()
            a_directions.clear()
            denominators.clear()

        alpha = rho / denominator
        x = x + alpha * direction
        residual = residual - alpha * a_direction
        residual_norm = float(np.linalg.norm(residual))
        history.append(residual_norm)
        directions.append(direction)
        a_directions.append(a_direction)
        denominators.append(denominator)
        if residual_norm <= target:
            return KrylovResult(x, True, iteration, history, history.copy(), list(range(len(history))))

    return KrylovResult(x, False, max_iterations, history, history.copy(), list(range(len(history))))


def fgmres(
    matvec: VectorMap,
    precondition: VectorMap,
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
    restart: int = 20,
) -> KrylovResult:
    """Restarted right-preconditioned flexible GMRES.

    ``residual_history`` contains measured true residuals at solution updates.
    Arnoldi/Givens estimates for every inner iteration are returned separately
    in ``estimated_residual_history``.
    """

    if restart < 1:
        raise ValueError("restart must be positive")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    b = np.asarray(b, dtype=np.float64)
    x, residual, initial_norm = _initial_state(matvec, b, x0)
    true_history = [initial_norm]
    true_iterations = [0]
    estimated_history = [initial_norm]
    target = _target_norm(b, tolerance)
    if initial_norm <= target or max_iterations == 0:
        return KrylovResult(
            x, initial_norm <= target, 0, true_history, estimated_history, true_iterations
        )

    total_iterations = 0
    while total_iterations < max_iterations:
        residual = np.asarray(b - matvec(x), dtype=np.float64)
        beta = float(np.linalg.norm(residual))
        if beta <= target:
            if true_history[-1] != beta:
                true_history.append(beta)
                true_iterations.append(total_iterations)
            return KrylovResult(
                x, True, total_iterations, true_history, estimated_history, true_iterations
            )

        subspace_size = min(restart, max_iterations - total_iterations)
        v = np.zeros((b.size, subspace_size + 1), dtype=np.float64)
        z_basis = np.zeros((b.size, subspace_size), dtype=np.float64)
        h = np.zeros((subspace_size + 1, subspace_size), dtype=np.float64)
        cosines = np.zeros(subspace_size, dtype=np.float64)
        sines = np.zeros(subspace_size, dtype=np.float64)
        rhs = np.zeros(subspace_size + 1, dtype=np.float64)
        v[:, 0] = residual / beta
        rhs[0] = beta
        used = 0

        for column in range(subspace_size):
            z_basis[:, column] = np.asarray(precondition(v[:, column]), dtype=np.float64)
            work = np.asarray(matvec(z_basis[:, column]), dtype=np.float64)

            for _ in range(2):
                for row in range(column + 1):
                    coefficient = float(np.dot(v[:, row], work))
                    h[row, column] += coefficient
                    work -= coefficient * v[:, row]

            h[column + 1, column] = np.linalg.norm(work)
            happy_breakdown = h[column + 1, column] <= 1.0e-14
            if not happy_breakdown:
                v[:, column + 1] = work / h[column + 1, column]

            for row in range(column):
                temporary = (
                    cosines[row] * h[row, column]
                    + sines[row] * h[row + 1, column]
                )
                h[row + 1, column] = (
                    -sines[row] * h[row, column]
                    + cosines[row] * h[row + 1, column]
                )
                h[row, column] = temporary

            denominator = float(np.hypot(h[column, column], h[column + 1, column]))
            if denominator <= 1.0e-30:
                cosines[column], sines[column] = 1.0, 0.0
            else:
                cosines[column] = h[column, column] / denominator
                sines[column] = h[column + 1, column] / denominator
            h[column, column] = (
                cosines[column] * h[column, column]
                + sines[column] * h[column + 1, column]
            )
            h[column + 1, column] = 0.0
            rhs[column + 1] = -sines[column] * rhs[column]
            rhs[column] = cosines[column] * rhs[column]

            used = column + 1
            total_iterations += 1
            estimated_history.append(abs(rhs[column + 1]))
            if (
                abs(rhs[column + 1]) <= target
                or total_iterations >= max_iterations
                or happy_breakdown
            ):
                break

        if used == 0:
            raise RuntimeError("FGMRES failed to build a Krylov basis")
        try:
            coefficients = np.linalg.solve(h[:used, :used], rhs[:used])
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(h[:used, :used], rhs[:used], rcond=None)[0]
        x = x + z_basis[:, :used] @ coefficients
        true_residual = float(np.linalg.norm(b - matvec(x)))
        true_history.append(true_residual)
        true_iterations.append(total_iterations)
        if true_residual <= target:
            return KrylovResult(
                x, True, total_iterations, true_history, estimated_history, true_iterations
            )

    return KrylovResult(
        x, False, total_iterations, true_history, estimated_history, true_iterations
    )
