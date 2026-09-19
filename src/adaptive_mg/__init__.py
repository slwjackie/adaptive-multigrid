"""Adaptive neural multigrid research package."""

from .models import (
    DetectorNet,
    GlobalSelectorNet,
    NeuralComponents,
    NeuralSmootherNet,
    TransferNet,
)
from .pde import DiffusionCase, assemble_problem, case_suite
from .solver import MGConfig, SolveResult, solve
from .strategy import MGStrategy, STRATEGIES
from .train import TrainingConfig, train_staged
from .transfer import (
    MAX_TRANSFER_CANDIDATES,
    N_MATRIX_FEATURES,
    validate_multigrid_size,
)

__all__ = [
    "DiffusionCase",
    "assemble_problem",
    "case_suite",
    "NeuralComponents",
    "TransferNet",
    "NeuralSmootherNet",
    "DetectorNet",
    "GlobalSelectorNet",
    "MGStrategy",
    "STRATEGIES",
    "MGConfig",
    "SolveResult",
    "solve",
    "TrainingConfig",
    "train_staged",
    "N_MATRIX_FEATURES",
    "MAX_TRANSFER_CANDIDATES",
    "validate_multigrid_size",
]
