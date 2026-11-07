"""TRAIN labels for the same frozen-bank objective as the teacher comparison.

The original BAMG teacher selects the vector bank once. Its coarse operators
are provenance only: every energy teacher level uses its own actual Galerkin
operator and mass. No validation/final examples or benchmark right-hand sides
are accepted by this interface.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from ..config import MGConfig
from ..grid import next_shape, terminal, validate_root_shape
from ..transfer import weights_from_sparse_matrix
from .bamg_energy import FixedObjective, array_digest, minimize
from .bamg_teacher import (
    TeacherConfig, TeacherLevel, TeacherResult, _checked_matrix,
    build_teacher, matrix_digest,
)
from .bamg_transfer import direct_pattern, fixed_bilinear_mask, scipy_from_weights

VERSION = "bamg-energy-train-labels-v1"


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


def _galerkin(operator, p):
    # Match the teacher-only experiment, including its sparse arithmetic order.
    result = (p.T @ operator @ p).tocsr()
    result.sum_duplicates()
    result.sort_indices()
    if not np.isfinite(result.data).all():
        raise FloatingPointError("Nonfinite energy-teacher Galerkin matrix")
    return result


def build_energy_teacher(a, shape, *, config: TeacherConfig,
                         mg_config: MGConfig, strength: float, optimizer: dict,
                         energy_only: bool = False) -> TeacherResult:
    """Return consistent teacher-forcing labels, with failures propagated.

    ``strength`` is the selected lambda, not a search on the TRAIN labels.
    A finite-budget feasible minimizer is retained and explicitly reported as
    unconverged, exactly as in the numerical teacher comparison. An infeasible
    matrix, bad vector bank, or invalid hierarchy raises instead of dropping a
    case or replacing its label with a different interpolation rule.
    """
    started = perf_counter()
    if config.support_mode != "fixed_bilinear":
        raise ValueError("Energy teacher labels require fixed_bilinear support")
    if isinstance(strength, bool) or not math.isfinite(strength) or strength < 0:
        raise ValueError("lambda must be finite and nonnegative")
    if type(energy_only) is not bool:
        raise ValueError("energy_only must be boolean")
    if not isinstance(optimizer, dict) or set(optimizer) - {"maxiter", "rtol"}:
        raise ValueError("optimizer may contain only maxiter and rtol")
    options = dict(maxiter=500, rtol=1e-8)
    options.update(optimizer)
    if (type(options["maxiter"]) is not int or options["maxiter"] < 1
            or not math.isfinite(options["rtol"]) or not 0 < options["rtol"] < 1):
        raise ValueError("Invalid optimizer budget/tolerance")
    shape = validate_root_shape(shape, mg_config.coarsest_n)
    root = _checked_matrix(a, shape)
    original = build_teacher(root, shape, config=config, mg_config=mg_config)
    vectors = [level.vectors.copy() for level in original.levels[:-1]]
    levels = []
    current, mass, current_shape = root, sp.eye(root.shape[0], format="csr"), shape
    index = 0
    while True:
        current = _checked_matrix(current, current_shape)
        mass = _checked_matrix(mass, current_shape)
        level = TeacherLevel(current, current_shape, mass)
        level.diagnostics = dict(
            level=index, split="train", A_digest=matrix_digest(current),
            H_digest=matrix_digest(mass), A_nnz=current.nnz, H_nnz=mass.nnz,
        )
        levels.append(level)
        if terminal(current_shape, mg_config.coarsest_n):
            if index != len(vectors):
                raise ValueError("BAMG vector bank and energy hierarchy depths differ")
            # These dense matrices are only the configured tiny terminal grid.
            la.cholesky(current.toarray(), lower=True)
            la.cholesky(mass.toarray(), lower=True)
            break
        if index >= len(vectors):
            raise ValueError("BAMG vector bank has too few levels")
        z = vectors[index]
        if z.ndim != 2 or z.shape[0] != current.shape[0]:
            raise ValueError("BAMG vector bank has incompatible level dimensions")
        coarse_shape = next_shape(current_shape, "full", mg_config.coarsest_n,
                                  level_index=index)
        pattern = direct_pattern(current_shape, coarse_shape)
        mask = fixed_bilinear_mask(pattern).cpu().numpy().copy()
        objective = FixedObjective(current, pattern, z, config)
        p, report = minimize(objective, strength, energy_only=energy_only, **options)
        if p.shape != objective.base.shape:
            raise ValueError("Energy label has incompatible interpolation dimensions")
        weights = weights_from_sparse_matrix(pattern, p)
        if np.any(weights[~mask] != 0):
            raise ValueError("Energy label has coefficients outside fixed support")
        reconstructed = scipy_from_weights(pattern, weights, mask)
        difference = p - reconstructed
        difference.eliminate_zeros()
        if difference.nnz:
            raise ValueError("Energy label has entries outside the candidate slots")
        constraints = objective.validate(objective.gather(p))
        level.pattern, level.mask, level.weights = pattern, mask, weights
        level.P, level.vectors = p, z
        coarse_a, coarse_mass = _galerkin(current, p), _galerkin(mass, p)
        level.diagnostics.update(
            report, **constraints,
            bank_vectors_digest=array_digest(z), P_digest=matrix_digest(p),
            coarse_A_digest=matrix_digest(coarse_a), coarse_H_digest=matrix_digest(coarse_mass),
            coarse_nnz=coarse_a.nnz, source_A_digest=matrix_digest(original.levels[index].A),
            source_P_digest=matrix_digest(original.levels[index].P),
        )
        complexity = (sum(item.A.nnz for item in levels) + coarse_a.nnz) / root.nnz
        if complexity > config.max_operator_complexity or coarse_mass.nnz > config.max_mass_nnz:
            raise MemoryError("Energy teacher exceeded configured sparse complexity budget")
        current, mass, current_shape = coarse_a, coarse_mass, coarse_shape
        index += 1
    elapsed = perf_counter() - started
    diagnostics = dict(
        version=VERSION, method="constrained_energy" if energy_only else "hybrid_bamg_energy",
        split="train", A_digest=matrix_digest(root), lambda_value=float(strength),
        objective="E/s_E" if energy_only else "B/s_B + lambda*E/s_E",
        reference_policy="bilinear", normalizers_fixed=True,
        config=config.to_dict(), mg_config=mg_config.to_dict(), optimizer=options,
        selected_pass=original.diagnostics["selected_pass"],
        vector_policy="frozen selected-pass BAMG bank; no refit on hybrid coarse A",
        vector_digests=[array_digest(z) for z in vectors],
        support_digests=[item.diagnostics["support_digest"] for item in levels[:-1]],
        reference_A_digests=[matrix_digest(item.A) for item in original.levels],
        reference_P_digests=[matrix_digest(item.P) for item in original.levels[:-1]],
        source_hashes=_source_hashes(), source_teacher=original.diagnostics,
        operator_complexity=sum(item.A.nnz for item in levels) / root.nnz,
        optimizer_converged=all(item.diagnostics["optimizer_converged"] for item in levels[:-1]),
        unconverged_levels=[i for i, item in enumerate(levels[:-1])
                            if not item.diagnostics["optimizer_converged"]],
        failure_policy="raise invalid/failing labels; retain and report feasible optimizer-budget limits",
        setup_seconds=elapsed,
    )
    return TeacherResult(levels, diagnostics, elapsed, config, mg_config)
