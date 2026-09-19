"""Cross-grid, full-V-cycle staged training for hierarchical adaptive neural MG."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .grid import full_coarsening_depth
from .models import NeuralComponents
from .solver import MGConfig, solve
from .pde import DiffusionCase, assemble_stiffness
from .strategy import (
    DEFAULT_STRATEGY_NAME,
    MGStrategy,
    STRATEGIES,
    get_strategy,
    global_context_tensor,
)
from .torch_ops import (
    Mode,
    TorchProblem,
    apply_a,
    classical_correction,
    detector_feature_tensor,
    full_v_cycle,
    local_energy,
    make_torch_problem,
    multi_cycle_energy_contraction,
    sample_error,
    vector_to_grid,
)
from .transfer import validate_multigrid_size

StrategySource = Literal["fixed", "selector", "oracle"]
DEFAULT_TRAIN_SIZES = (15, 31, 63)
DEFAULT_VALIDATION_SIZES = (15, 31, 63, 127)


@dataclass
class TrainingConfig:
    # ``n`` remains as a compatibility shortcut.  When set, it overrides
    # ``train_sizes`` and (unless explicitly supplied) validation sizes.
    n: int | None = None
    train_sizes: tuple[int, ...] = DEFAULT_TRAIN_SIZES
    validation_sizes: tuple[int, ...] = DEFAULT_VALIDATION_SIZES
    ood_validation_sizes: tuple[int, ...] = ()
    coarsest_n: int = 3
    pool_size: int = 3
    validation_pool_size: int = 2
    samples_per_epoch: int = 4
    validation_samples_per_problem: int = 1
    cycles_per_sample: int = 2
    pre_steps: int = 1
    post_steps: int = 1
    selector_epochs: int = 15
    smoother_epochs: int = 30
    transfer_epochs: int = 30
    combined_epochs: int = 20
    detector_epochs: int = 15
    adaptive_epochs: int = 10
    learning_rate: float = 2.0e-3
    detector_learning_rate: float = 1.0e-3
    selector_learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    gradient_clip: float = 2.0
    validation_min_delta: float = 1.0e-4
    detector_failure_ratio: float = 0.35
    detector_benefit_margin: float = 0.02
    detector_temperature: float = 0.15
    adaptive_gate_cost: float = 0.03
    raw_instability_penalty: float = 1.0
    training_strategy_source: StrategySource = "selector"
    fixed_strategy_name: str = DEFAULT_STRATEGY_NAME
    selector_label_max_n: int = 31
    selector_work_penalty: float = 0.01
    common_validation_smooth_steps: int = 1
    max_training_levels: int | None = None
    require_cross_grid_validation: bool = True
    # Research training requires at least two transfers (three grid levels) for
    # every root sample.  Tiny two-grid samples are allowed only for explicit
    # smoke/compatibility runs.
    allow_two_grid_training: bool = False
    jacobi_omega: float = 0.72
    operator_sweeps: int = 3
    operator_omega: float = 0.70
    operator_blend: float = 0.75
    operator_drop_tolerance: float = 1.0e-4
    operator_candidate_topk: int | None = 8
    seed: int = 7
    device: str = "cpu"
    dtype: str = "float32"

    def resolved_train_sizes(self) -> tuple[int, ...]:
        return (self.n,) if self.n is not None else tuple(self.train_sizes)

    def resolved_validation_sizes(self) -> tuple[int, ...]:
        if self.n is not None and self.validation_sizes == DEFAULT_VALIDATION_SIZES:
            return (self.n,)
        return tuple(self.validation_sizes)

    def all_validation_sizes(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(self.resolved_validation_sizes() + self.ood_validation_sizes))

    def validate(self) -> None:
        for n in set(self.resolved_train_sizes() + self.all_validation_sizes()):
            validate_multigrid_size(n, self.coarsest_n)
            if n <= self.coarsest_n:
                raise ValueError("training/validation grids must exceed coarsest_n")
        if self.pool_size < 1 or self.validation_pool_size < 1:
            raise ValueError("training and validation pools must be nonempty")
        if self.samples_per_epoch < 1 or self.validation_samples_per_problem < 1:
            raise ValueError("sample counts must be positive")
        if self.cycles_per_sample < 1:
            raise ValueError("cycles_per_sample must be positive")
        if self.max_training_levels is not None and self.max_training_levels < 2:
            raise ValueError(
                "training must use at least a three-grid hierarchy; "
                "max_training_levels must be None or >=2"
            )
        if self.n is None and self.require_cross_grid_validation:
            unseen = set(self.all_validation_sizes()) - set(self.resolved_train_sizes())
            if not unseen:
                raise ValueError(
                    "cross-grid validation requires at least one validation size "
                    "that is absent from train_sizes"
                )
        if self.n is None and not self.allow_two_grid_training:
            shallow = [
                n for n in self.resolved_train_sizes()
                if full_coarsening_depth(n, self.coarsest_n) < 2
            ]
            if shallow:
                raise ValueError(
                    "research training requires at least a three-grid full V-cycle "
                    f"for every root sample; shallow train sizes: {shallow}"
                )
        get_strategy(self.fixed_strategy_name)
        for name in (
            "selector_epochs",
            "smoother_epochs",
            "transfer_epochs",
            "combined_epochs",
            "detector_epochs",
            "adaptive_epochs",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass
class TrainingSample:
    case: DiffusionCase
    problem: TorchProblem
    context: torch.Tensor
    oracle_strategy_index: int | None = None

    @property
    def n(self) -> int:
        return self.case.n


@dataclass
class EpochRecord:
    stage: str
    epoch: int
    loss: float
    energy_contraction: float | None = None
    validation_contraction: float | None = None
    validation_objective: float | None = None
    gate_mean: float | None = None
    accepted: bool | None = None


@dataclass
class StageSummary:
    stage: str
    evaluation_mode: str
    smooth_error_steps: int
    entry_objective: float
    best_objective: float
    entry_validation: float
    best_validation: float
    best_epoch: int
    accepted: bool


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def random_case(n: int, rng: np.random.Generator) -> DiffusionCase:
    pattern = str(rng.choice(["uniform", "local_patch", "checkerboard", "channel"]))
    center_x = float(rng.uniform(0.35, 0.65))
    center_y = float(rng.uniform(0.35, 0.65))
    half = float(rng.uniform(0.12, 0.28))
    return DiffusionCase(
        n=n,
        epsilon=float(10.0 ** rng.uniform(-3.2, 0.0)),
        angle_deg=float(rng.uniform(0.0, 90.0)),
        contrast=float(10.0 ** rng.uniform(0.0, 4.0)),
        pattern=pattern,  # type: ignore[arg-type]
        patch_bounds=(
            max(0.05, center_x - half),
            min(0.95, center_x + half),
            max(0.05, center_y - half),
            min(0.95, center_y + half),
        ),
        checker_cells=int(rng.choice([2, 3, 4, 5])),
        channel_width=float(rng.uniform(0.04, 0.15)),
    )


def _build_pool(
    config: TrainingConfig,
    *,
    sizes: tuple[int, ...],
    cases_per_grid: int,
    seed: int,
) -> list[TrainingSample]:
    rng = np.random.default_rng(seed)
    samples: list[TrainingSample] = []
    for n in sizes:
        cases = [
            DiffusionCase(
                n=n,
                epsilon=1.0,
                angle_deg=0.0,
                contrast=1.0,
                pattern="uniform",
            )
        ]
        cases.extend(random_case(n, rng) for _ in range(max(0, cases_per_grid - 1)))
        for case in cases:
            a = assemble_stiffness(case)
            problem = make_torch_problem(
                a, n, device=config.device, dtype=_dtype(config.dtype)
            )
            samples.append(
                TrainingSample(
                    case=case,
                    problem=problem,
                    context=global_context_tensor(
                        a, n, device=config.device, dtype=_dtype(config.dtype)
                    ),
                )
            )
    return samples


def build_training_pools(
    config: TrainingConfig,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    config.validate()
    return (
        _build_pool(
            config,
            sizes=config.resolved_train_sizes(),
            cases_per_grid=config.pool_size,
            seed=config.seed,
        ),
        _build_pool(
            config,
            sizes=config.all_validation_sizes(),
            cases_per_grid=config.validation_pool_size,
            seed=config.seed + 100_003,
        ),
    )


def build_training_pool(config: TrainingConfig) -> list[TorchProblem]:
    """Compatibility helper returning root TorchProblem objects."""

    return [sample.problem for sample in build_training_pools(config)[0]]


def _requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def _optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: TrainingConfig,
    *,
    detector: bool = False,
    selector: bool = False,
) -> torch.optim.Optimizer:
    if selector:
        learning_rate = config.selector_learning_rate
    elif detector:
        learning_rate = config.detector_learning_rate
    else:
        learning_rate = config.learning_rate
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=config.weight_decay)


def _generator(problem: TorchProblem, seed: int) -> torch.Generator:
    generator = torch.Generator(device=problem.device)
    generator.manual_seed(seed)
    return generator


def _oracle_strategy_index(
    sample: TrainingSample,
    components: NeuralComponents,
    config: TrainingConfig,
) -> int:
    """Label the controlled selector with the production classical solver.

    The v2/v3 prototype originally evaluated every oracle candidate through the
    differentiable training graph, which was needlessly slow and distorted the
    selector cost.  Labels now use one actual SciPy MG cycle and combine A-norm
    contraction with a small work-unit penalty.
    """

    if sample.oracle_strategy_index is not None:
        return sample.oracle_strategy_index
    if sample.case.n > config.selector_label_max_n:
        sample.oracle_strategy_index = -1
        return -1
    a = sample.problem.scipy_a
    if a is None:
        raise RuntimeError("selector oracle labels require a root SciPy operator")
    rng = np.random.default_rng(config.seed + 600_001 + 7919 * sample.case.n)
    exact = rng.normal(size=a.shape[0]).astype(np.float64)
    exact /= max(float(np.linalg.norm(exact)), 1.0e-14)
    b = np.asarray(a @ exact, dtype=np.float64)
    initial_energy = max(float(np.sqrt(np.dot(exact, a @ exact))), 1.0e-14)
    scores: list[float] = []
    for strategy in STRATEGIES:
        result = solve(
            a,
            b,
            sample.case.n,
            components,
            MGConfig(
                mode="classical",
                strategy_name=strategy.name,
                max_cycles=1,
                tolerance=0.0,
                pre_steps=config.pre_steps,
                post_steps=config.post_steps,
                jacobi_omega=config.jacobi_omega,
                coarsest_n=config.coarsest_n,
                operator_sweeps=config.operator_sweeps,
                operator_omega=config.operator_omega,
                operator_blend=config.operator_blend,
                operator_drop_tolerance=config.operator_drop_tolerance,
                operator_candidate_topk=config.operator_candidate_topk,
                safety_fallback=False,
                smoother_safeguard=False,
                outer_solver="none",
            ),
        )
        remaining = exact - result.x
        energy = float(np.sqrt(max(np.dot(remaining, a @ remaining), 0.0)))
        contraction = max(energy / initial_energy, 1.0e-12)
        score = np.log(contraction) + config.selector_work_penalty * np.log1p(
            result.work_units
        )
        scores.append(float(score))
    sample.oracle_strategy_index = int(np.argmin(scores))
    return sample.oracle_strategy_index


def _selected_strategy(
    sample: TrainingSample,
    components: NeuralComponents,
    config: TrainingConfig,
) -> MGStrategy:
    if config.training_strategy_source == "fixed":
        return get_strategy(config.fixed_strategy_name)
    if config.training_strategy_source == "oracle":
        index = _oracle_strategy_index(sample, components, config)
        return STRATEGIES[index] if index >= 0 else get_strategy(config.fixed_strategy_name)
    with torch.no_grad():
        index = int(torch.argmax(components.selector(sample.context), dim=1).item())
    return STRATEGIES[index]


def _cycle_objective(
    sample: TrainingSample,
    components: NeuralComponents,
    mode: Mode,
    config: TrainingConfig,
    *,
    smooth_error_steps: int,
    gate_cost: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    problem = sample.problem
    strategy = _selected_strategy(sample, components, config)
    error = sample_error(
        problem.shape,
        device=problem.device,
        dtype=problem.dtype,
        smooth_steps=smooth_error_steps,
        problem=problem,
        smoother=strategy.smoother,
        omega=config.jacobi_omega,
        generator=generator,
    )
    contraction, gate_mean, _ = multi_cycle_energy_contraction(
        problem,
        error,
        components,
        mode=mode,
        strategy=strategy,
        cycles=config.cycles_per_sample,
        coarsest_n=config.coarsest_n,
        omega=config.jacobi_omega,
        pre_steps=config.pre_steps,
        post_steps=config.post_steps,
        max_levels=config.max_training_levels,
        operator_sweeps=config.operator_sweeps,
        operator_omega=config.operator_omega,
        operator_blend=config.operator_blend,
        operator_drop_tolerance=config.operator_drop_tolerance,
        operator_candidate_topk=config.operator_candidate_topk,
    )
    instability = config.raw_instability_penalty * torch.relu(contraction - 1.0).square()
    loss = torch.log(contraction + 1.0e-8) + gate_cost * gate_mean + instability
    return loss, contraction, gate_mean


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.exp(np.mean(np.log(np.maximum(values, 1.0e-300)))))


def _evaluate_cycle_mode(
    components: NeuralComponents,
    pool: list[TrainingSample],
    mode: Mode,
    config: TrainingConfig,
    *,
    smooth_error_steps: int,
    gate_cost: float = 0.0,
    seed_offset: int = 0,
) -> tuple[float, float, float]:
    components.eval()
    objectives: list[float] = []
    contractions: list[float] = []
    gates: list[float] = []
    with torch.no_grad():
        for problem_index, sample in enumerate(pool):
            for sample_index in range(config.validation_samples_per_problem):
                seed = config.seed + 700_001 + seed_offset + 997 * problem_index + sample_index
                loss, contraction, gate = _cycle_objective(
                    sample,
                    components,
                    mode,
                    config,
                    smooth_error_steps=smooth_error_steps,
                    gate_cost=gate_cost,
                    generator=_generator(sample.problem, seed),
                )
                objective_value = float(loss.cpu())
                contraction_value = float(contraction.cpu())
                gate_value = float(gate.cpu())
                objectives.append(
                    objective_value if np.isfinite(objective_value) else float("inf")
                )
                contractions.append(
                    contraction_value if np.isfinite(contraction_value) else float("inf")
                )
                gates.append(gate_value if np.isfinite(gate_value) else 1.0)
    components.train()
    # Objective is a mean log-contraction; report the matching geometric mean.
    return float(np.mean(objectives)), _geometric_mean(contractions), float(np.mean(gates))


def _run_cycle_stage(
    stage: str,
    mode: Literal["nn_smoother", "learned_transfer", "combined", "adaptive"],
    components: NeuralComponents,
    train_pool: list[TrainingSample],
    validation_pool: list[TrainingSample],
    config: TrainingConfig,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    *,
    smooth_error_steps: int,
    gate_cost: float = 0.0,
    verbose: bool = True,
) -> tuple[list[EpochRecord], StageSummary]:
    offsets = {
        "nn_smoother": 1001,
        "learned_transfer": 2001,
        "combined": 3001,
        "adaptive": 5001,
    }
    rng = random.Random(config.seed + offsets[stage])
    records: list[EpochRecord] = []
    entry_state = deepcopy(components.state_dict())
    entry_objective, entry_contraction, _ = _evaluate_cycle_mode(
        components,
        validation_pool,
        mode,
        config,
        smooth_error_steps=smooth_error_steps,
        gate_cost=gate_cost,
        seed_offset=offsets[stage],
    )
    best_state = deepcopy(entry_state)
    best_objective = entry_objective
    best_contraction = entry_contraction
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        components.train()
        losses: list[float] = []
        contractions: list[float] = []
        gates: list[float] = []
        for sample_index in range(config.samples_per_epoch):
            sample = train_pool[rng.randrange(len(train_pool))]
            optimizer.zero_grad(set_to_none=True)
            loss, contraction, gate = _cycle_objective(
                sample,
                components,
                mode,
                config,
                smooth_error_steps=smooth_error_steps,
                gate_cost=gate_cost,
                generator=_generator(
                    sample.problem,
                    config.seed + offsets[stage] * 10_000 + epoch * 101 + sample_index,
                ),
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            trainable = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            contractions.append(float(contraction.detach().cpu()))
            gates.append(float(gate.detach().cpu()))

        val_objective, val_contraction, val_gate = _evaluate_cycle_mode(
            components,
            validation_pool,
            mode,
            config,
            smooth_error_steps=smooth_error_steps,
            gate_cost=gate_cost,
            seed_offset=offsets[stage],
        )
        if val_objective < best_objective - config.validation_min_delta:
            best_objective = val_objective
            best_contraction = val_contraction
            best_state = deepcopy(components.state_dict())
            best_epoch = epoch
        record = EpochRecord(
            stage=stage,
            epoch=epoch,
            loss=float(np.mean(losses)) if losses else float("inf"),
            energy_contraction=_geometric_mean(contractions) if contractions else None,
            validation_contraction=val_contraction,
            validation_objective=val_objective,
            gate_mean=float(np.mean(gates)) if gates else val_gate,
        )
        records.append(record)
        if verbose and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0):
            train_rho = float("nan") if record.energy_contraction is None else record.energy_contraction
            print(
                f"[{stage:>16s}] epoch={epoch:04d}/{epochs:04d} "
                f"train_rho={train_rho:.4f} val_rho={val_contraction:.4f} "
                f"val_obj={val_objective:.4f} best_rho={best_contraction:.4f} "
                f"best_obj={best_objective:.4f} gate={record.gate_mean:.3f}"
            )

    accepted = best_epoch > 0
    components.load_state_dict(best_state)
    for record in records:
        record.accepted = accepted
    return records, StageSummary(
        stage=stage,
        evaluation_mode=mode,
        smooth_error_steps=smooth_error_steps,
        entry_objective=entry_objective,
        best_objective=best_objective,
        entry_validation=entry_contraction,
        best_validation=best_contraction,
        best_epoch=best_epoch,
        accepted=accepted,
    )


def _selector_dataset(
    pool: list[TrainingSample],
    components: NeuralComponents,
    config: TrainingConfig,
) -> list[tuple[torch.Tensor, int]]:
    result: list[tuple[torch.Tensor, int]] = []
    for sample in pool:
        label = _oracle_strategy_index(sample, components, config)
        if label >= 0:
            result.append((sample.context, label))
    return result


def _selector_metrics(
    components: NeuralComponents,
    dataset: list[tuple[torch.Tensor, int]],
) -> tuple[float, float]:
    if not dataset:
        return float("nan"), float("nan")
    losses: list[float] = []
    correct = 0
    components.selector.eval()
    with torch.no_grad():
        for context, label in dataset:
            logits = components.selector(context)
            target = torch.tensor([label], device=logits.device, dtype=torch.long)
            losses.append(float(F.cross_entropy(logits, target).cpu()))
            correct += int(torch.argmax(logits, dim=1).item() == label)
    components.selector.train()
    return float(np.mean(losses)), correct / len(dataset)


def _train_selector(
    components: NeuralComponents,
    train_pool: list[TrainingSample],
    validation_pool: list[TrainingSample],
    config: TrainingConfig,
    *,
    verbose: bool,
) -> tuple[list[EpochRecord], StageSummary]:
    _requires_grad(components.selector, True)
    train_data = _selector_dataset(train_pool, components, config)
    validation_data = _selector_dataset(validation_pool, components, config)
    if not train_data or not validation_data or config.selector_epochs == 0:
        loss, _ = _selector_metrics(components, validation_data)
        return [], StageSummary(
            stage="global_selector",
            evaluation_mode="cross_entropy",
            smooth_error_steps=0,
            entry_objective=loss,
            best_objective=loss,
            entry_validation=loss,
            best_validation=loss,
            best_epoch=0,
            accepted=False,
        )
    optimizer = _optimizer(components.selector.parameters(), config, selector=True)
    rng = random.Random(config.seed + 2718)
    entry_state = deepcopy(components.state_dict())
    entry_loss, _ = _selector_metrics(components, validation_data)
    best_state = deepcopy(entry_state)
    best_loss, best_epoch = entry_loss, 0
    records: list[EpochRecord] = []
    for epoch in range(1, config.selector_epochs + 1):
        losses: list[float] = []
        for _ in range(config.samples_per_epoch):
            context, label = train_data[rng.randrange(len(train_data))]
            logits = components.selector(context)
            target = torch.tensor([label], device=logits.device, dtype=torch.long)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(components.selector.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_loss, accuracy = _selector_metrics(components, validation_data)
        if val_loss < best_loss - config.validation_min_delta:
            best_loss, best_epoch = val_loss, epoch
            best_state = deepcopy(components.state_dict())
        records.append(
            EpochRecord(
                stage="global_selector",
                epoch=epoch,
                loss=float(np.mean(losses)),
                validation_objective=val_loss,
                validation_contraction=1.0 - accuracy,
            )
        )
        if verbose and (epoch == 1 or epoch == config.selector_epochs):
            print(
                f"[ global_selector] epoch={epoch:04d}/{config.selector_epochs:04d} "
                f"val_ce={val_loss:.4e} accuracy={accuracy:.3f}"
            )
    components.load_state_dict(best_state)
    accepted = best_epoch > 0
    for record in records:
        record.accepted = accepted
    return records, StageSummary(
        stage="global_selector",
        evaluation_mode="cross_entropy",
        smooth_error_steps=0,
        entry_objective=entry_loss,
        best_objective=best_loss,
        entry_validation=entry_loss,
        best_validation=best_loss,
        best_epoch=best_epoch,
        accepted=accepted,
    )


def _detector_loss(
    components: NeuralComponents,
    sample: TrainingSample,
    config: TrainingConfig,
    *,
    generator: torch.Generator,
    smooth_error_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    problem = sample.problem
    strategy = _selected_strategy(sample, components, config)
    error = sample_error(
        problem.shape,
        device=problem.device,
        dtype=problem.dtype,
        smooth_steps=smooth_error_steps,
        problem=problem,
        smoother=strategy.smoother,
        omega=config.jacobi_omega,
        generator=generator,
    )
    b = torch.zeros_like(error)
    residual = -apply_a(problem, error)
    base = classical_correction(
        problem,
        residual,
        smoother=strategy.smoother,
        omega=config.jacobi_omega,
        reverse=False,
    )
    features = detector_feature_tensor(
        problem,
        residual,
        smoother=strategy.smoother,
        omega=config.jacobi_omega,
        reverse=False,
        classical=base,
    )
    with torch.no_grad():
        x_classical, _ = full_v_cycle(
            problem,
            error,
            b,
            components,
            mode="classical",
            strategy=strategy,
            coarsest_n=config.coarsest_n,
            omega=config.jacobi_omega,
            pre_steps=config.pre_steps,
            post_steps=config.post_steps,
            max_levels=config.max_training_levels,
            operator_sweeps=config.operator_sweeps,
            operator_omega=config.operator_omega,
            operator_blend=config.operator_blend,
            operator_drop_tolerance=config.operator_drop_tolerance,
            operator_candidate_topk=config.operator_candidate_topk,
        )
        x_neural, _ = full_v_cycle(
            problem,
            error,
            b,
            components,
            mode="combined",
            strategy=strategy,
            coarsest_n=config.coarsest_n,
            omega=config.jacobi_omega,
            pre_steps=config.pre_steps,
            post_steps=config.post_steps,
            max_levels=config.max_training_levels,
            operator_sweeps=config.operator_sweeps,
            operator_omega=config.operator_omega,
            operator_blend=config.operator_blend,
            operator_drop_tolerance=config.operator_drop_tolerance,
            operator_candidate_topk=config.operator_candidate_topk,
        )
        r_classical = -apply_a(problem, x_classical)
        r_neural = -apply_a(problem, x_neural)
        diagonal = torch.clamp(
            torch.abs(problem.diag.reshape(1, 1, *problem.shape)), min=1.0e-12
        )
        z_before = vector_to_grid(residual, problem.shape) / diagonal
        z_classical = vector_to_grid(r_classical, problem.shape) / diagonal
        z_neural = vector_to_grid(r_neural, problem.shape) / diagonal
        e_before = torch.nan_to_num(
            local_energy(z_before), nan=1.0e12, posinf=1.0e12, neginf=1.0e12
        ).clamp(1.0e-12, 1.0e12)
        e_classical = torch.nan_to_num(
            local_energy(z_classical), nan=1.0e12, posinf=1.0e12, neginf=1.0e12
        ).clamp(1.0e-12, 1.0e12)
        e_neural = torch.nan_to_num(
            local_energy(z_neural), nan=1.0e12, posinf=1.0e12, neginf=1.0e12
        ).clamp(1.0e-12, 1.0e12)
        log_classical_ratio = 0.5 * torch.log(e_classical / e_before)
        log_neural_advantage = torch.log(e_classical / e_neural)
        failure = torch.sigmoid(
            (log_classical_ratio - float(np.log(config.detector_failure_ratio)))
            / config.detector_temperature
        )
        benefit = torch.sigmoid(
            (log_neural_advantage - config.detector_benefit_margin)
            / config.detector_temperature
        )
        target = torch.nan_to_num(
            (failure * benefit).clamp(0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0
        )
    prediction = components.detector(features)
    return F.binary_cross_entropy(prediction, target), prediction.mean(), target.mean()


def _evaluate_detector(
    components: NeuralComponents,
    pool: list[TrainingSample],
    config: TrainingConfig,
) -> tuple[float, float, float]:
    components.eval()
    losses: list[float] = []
    gates: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for index, sample in enumerate(pool):
            loss, gate, target = _detector_loss(
                components,
                sample,
                config,
                generator=_generator(sample.problem, config.seed + 900_001 + index),
                smooth_error_steps=index % 4,
            )
            losses.append(float(loss.cpu()))
            gates.append(float(gate.cpu()))
            targets.append(float(target.cpu()))
    components.train()
    return float(np.mean(losses)), float(np.mean(gates)), float(np.mean(targets))


def _train_detector(
    components: NeuralComponents,
    train_pool: list[TrainingSample],
    validation_pool: list[TrainingSample],
    config: TrainingConfig,
    *,
    verbose: bool,
) -> tuple[list[EpochRecord], StageSummary]:
    _requires_grad(components.transfer, False)
    _requires_grad(components.smoother, False)
    _requires_grad(components.detector, True)
    optimizer = _optimizer(components.detector.parameters(), config, detector=True)
    rng = random.Random(config.seed + 4001)
    records: list[EpochRecord] = []
    entry_state = deepcopy(components.state_dict())
    entry_loss, _, _ = _evaluate_detector(components, validation_pool, config)
    best_state, best_loss, best_epoch = deepcopy(entry_state), entry_loss, 0

    for epoch in range(1, config.detector_epochs + 1):
        losses: list[float] = []
        gates: list[float] = []
        targets: list[float] = []
        for sample_index in range(config.samples_per_epoch):
            sample = train_pool[rng.randrange(len(train_pool))]
            loss, gate, target = _detector_loss(
                components,
                sample,
                config,
                generator=_generator(
                    sample.problem, config.seed + 4_001_000 + epoch * 101 + sample_index
                ),
                smooth_error_steps=rng.choice([0, 1, 2, 3]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(components.detector.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gates.append(float(gate.detach().cpu()))
            targets.append(float(target.detach().cpu()))
        val_loss, val_gate, val_target = _evaluate_detector(
            components, validation_pool, config
        )
        if val_loss < best_loss - config.validation_min_delta:
            best_loss, best_epoch = val_loss, epoch
            best_state = deepcopy(components.state_dict())
        records.append(
            EpochRecord(
                "detector",
                epoch,
                float(np.mean(losses)),
                float(np.mean(targets)),
                val_target,
                val_loss,
                float(np.mean(gates)) if gates else val_gate,
            )
        )
        if verbose and (epoch == 1 or epoch == config.detector_epochs):
            print(
                f"[        detector] epoch={epoch:04d}/{config.detector_epochs:04d} "
                f"val_bce={val_loss:.4e} gate={val_gate:.3f} target={val_target:.3f}"
            )
    accepted = best_epoch > 0
    components.load_state_dict(best_state)
    for record in records:
        record.accepted = accepted
    return records, StageSummary(
        stage="detector",
        evaluation_mode="binary_cross_entropy",
        smooth_error_steps=0,
        entry_objective=entry_loss,
        best_objective=best_loss,
        entry_validation=entry_loss,
        best_validation=best_loss,
        best_epoch=best_epoch,
        accepted=accepted,
    )


def _common_mode_metrics(
    components: NeuralComponents,
    pool: list[TrainingSample],
    config: TrainingConfig,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for mode in ("classical", "nn_smoother", "learned_transfer", "combined", "adaptive"):
        objective, contraction, gate = _evaluate_cycle_mode(
            components,
            pool,
            mode,  # type: ignore[arg-type]
            config,
            smooth_error_steps=config.common_validation_smooth_steps,
            gate_cost=config.adaptive_gate_cost if mode == "adaptive" else 0.0,
            seed_offset=42_000,
        )
        metrics[mode] = {
            "objective": objective,
            "geometric_energy_contraction": contraction,
            "gate_mean": gate,
        }
    return metrics


def evaluate_all_modes_common(
    components: NeuralComponents,
    validation_pool: list[TrainingSample],
    config: TrainingConfig,
) -> dict[str, object]:
    """Evaluate all modes under one protocol, globally and per grid size.

    The aggregate and every per-grid table use identical random seeds, error
    smoothing, cycle count, and A-norm metric.  This is the comparison table
    intended for ablation/checkpoint metadata; stage-specific validation
    numbers are deliberately kept separate.
    """

    by_grid: dict[str, dict[str, dict[str, float]]] = {}
    for n in sorted({sample.case.n for sample in validation_pool}):
        subpool = [sample for sample in validation_pool if sample.case.n == n]
        by_grid[str(n)] = _common_mode_metrics(components, subpool, config)
    return {
        "protocol": {
            "metric": "geometric mean full-V-cycle A-norm contraction",
            "smooth_error_steps": config.common_validation_smooth_steps,
            "cycles_per_sample": config.cycles_per_sample,
            "validation_samples_per_problem": config.validation_samples_per_problem,
            "strategy_source": config.training_strategy_source,
            "max_training_levels": config.max_training_levels,
        },
        "aggregate": _common_mode_metrics(components, validation_pool, config),
        "by_grid": by_grid,
    }


def train_staged(
    components: NeuralComponents,
    config: TrainingConfig,
    *,
    checkpoint: str | Path | None = None,
    verbose: bool = True,
) -> list[EpochRecord]:
    """Train selector and local modules using validation-guarded full V-cycles."""

    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    components.to(config.device, dtype=_dtype(config.dtype)).train()
    train_pool, validation_pool = build_training_pools(config)
    records: list[EpochRecord] = []
    summaries: list[StageSummary] = []

    selector_records, selector_summary = _train_selector(
        components, train_pool, validation_pool, config, verbose=verbose
    )
    records.extend(selector_records)
    summaries.append(selector_summary)
    _requires_grad(components.selector, False)

    _requires_grad(components.transfer, False)
    _requires_grad(components.smoother, True)
    _requires_grad(components.detector, False)
    stage_records, summary = _run_cycle_stage(
        "nn_smoother",
        "nn_smoother",
        components,
        train_pool,
        validation_pool,
        config,
        config.smoother_epochs,
        _optimizer(components.smoother.parameters(), config),
        smooth_error_steps=0,
        verbose=verbose,
    )
    records.extend(stage_records)
    summaries.append(summary)

    _requires_grad(components.transfer, True)
    _requires_grad(components.smoother, False)
    stage_records, summary = _run_cycle_stage(
        "learned_transfer",
        "learned_transfer",
        components,
        train_pool,
        validation_pool,
        config,
        config.transfer_epochs,
        _optimizer(components.transfer.parameters(), config),
        smooth_error_steps=3,
        verbose=verbose,
    )
    records.extend(stage_records)
    summaries.append(summary)

    _requires_grad(components.transfer, True)
    _requires_grad(components.smoother, True)
    stage_records, summary = _run_cycle_stage(
        "combined",
        "combined",
        components,
        train_pool,
        validation_pool,
        config,
        config.combined_epochs,
        _optimizer(
            list(components.transfer.parameters()) + list(components.smoother.parameters()),
            config,
        ),
        smooth_error_steps=1,
        verbose=verbose,
    )
    records.extend(stage_records)
    summaries.append(summary)

    detector_records, detector_summary = _train_detector(
        components, train_pool, validation_pool, config, verbose=verbose
    )
    records.extend(detector_records)
    summaries.append(detector_summary)

    if config.adaptive_epochs > 0:
        _requires_grad(components.transfer, True)
        _requires_grad(components.smoother, True)
        _requires_grad(components.detector, True)
        stage_records, summary = _run_cycle_stage(
            "adaptive",
            "adaptive",
            components,
            train_pool,
            validation_pool,
            config,
            config.adaptive_epochs,
            _optimizer(
                list(components.transfer.parameters())
                + list(components.smoother.parameters())
                + list(components.detector.parameters()),
                config,
            ),
            smooth_error_steps=1,
            gate_cost=config.adaptive_gate_cost,
            verbose=verbose,
        )
        records.extend(stage_records)
        summaries.append(summary)

    components.eval()
    common_validation = evaluate_all_modes_common(components, validation_pool, config)
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        components.save(
            checkpoint,
            metadata={
                "asset_status": (
                    "trained verification asset; raw and safeguarded ablations are "
                    "required before any paper-performance claim"
                ),
                "training_config": asdict(config),
                "records": [asdict(record) for record in records],
                "stage_summaries": [asdict(summary) for summary in summaries],
                "common_validation": common_validation,
                "training_root_sizes": sorted({sample.case.n for sample in train_pool}),
                "validation_root_sizes": sorted({sample.case.n for sample in validation_pool}),
                "in_distribution_validation_sizes": list(config.resolved_validation_sizes()),
                "ood_validation_sizes": list(config.ood_validation_sizes),
                "validation_aggregation": "geometric mean of full-V-cycle A-norm contractions",
                "full_v_cycle_training": True,
                "training_transfer_levels": {
                    str(n): full_coarsening_depth(n, config.coarsest_n)
                    for n in sorted({sample.case.n for sample in train_pool})
                },
                "minimum_training_transfer_levels": min(
                    full_coarsening_depth(n, config.coarsest_n)
                    for n in {sample.case.n for sample in train_pool}
                ),
                "smoother_coefficient_scale": components.smoother.coefficient_scale,
                "model_dtype": config.dtype,
            },
        )
    return records
