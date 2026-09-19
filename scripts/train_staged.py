#!/usr/bin/env python3
"""Cross-grid, validation-guarded full-V-cycle neural-MG training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adaptive_mg.models import NeuralComponents
from adaptive_mg.strategy import STRATEGY_BY_NAME
from adaptive_mg.train import TrainingConfig, train_staged


def parse_sizes(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("invalid comma-separated grid sizes")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoint_v3.pt"))
    parser.add_argument("--train-sizes", type=parse_sizes, default=(15, 31, 63))
    parser.add_argument("--validation-sizes", type=parse_sizes, default=(15, 31, 63))
    parser.add_argument("--ood-validation-sizes", type=parse_sizes, default=(127,))
    parser.add_argument("--pool-size", type=int, default=3)
    parser.add_argument("--validation-pool-size", type=int, default=2)
    parser.add_argument("--samples-per-epoch", type=int, default=6)
    parser.add_argument("--validation-samples-per-problem", type=int, default=1)
    parser.add_argument("--cycles-per-sample", type=int, default=2)
    parser.add_argument("--selector-epochs", type=int, default=15)
    parser.add_argument("--smoother-epochs", type=int, default=30)
    parser.add_argument("--transfer-epochs", type=int, default=30)
    parser.add_argument("--combined-epochs", type=int, default=20)
    parser.add_argument("--detector-epochs", type=int, default=15)
    parser.add_argument("--adaptive-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--detector-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--selector-learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--strategy-source", choices=["fixed", "selector", "oracle"], default="selector"
    )
    parser.add_argument(
        "--fixed-strategy", choices=sorted(STRATEGY_BY_NAME), default="line_alt_bilinear_full"
    )
    parser.add_argument("--selector-label-max-n", type=int, default=31)
    parser.add_argument("--max-training-levels", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--allow-no-cross-grid-validation", action="store_true")
    parser.add_argument(
        "--verification-profile",
        action="store_true",
        help="Use a short multi-grid run for code verification, not paper claims.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verification_profile:
        args.train_sizes = (7, 15)
        args.validation_sizes = (15, 31)
        args.ood_validation_sizes = ()
        args.pool_size = 2
        args.validation_pool_size = 2
        args.samples_per_epoch = 2
        args.selector_epochs = 1
        args.smoother_epochs = 2
        args.transfer_epochs = 2
        args.combined_epochs = 1
        args.detector_epochs = 1
        args.adaptive_epochs = 1

    config = TrainingConfig(
        train_sizes=args.train_sizes,
        validation_sizes=args.validation_sizes,
        ood_validation_sizes=args.ood_validation_sizes,
        pool_size=args.pool_size,
        validation_pool_size=args.validation_pool_size,
        samples_per_epoch=args.samples_per_epoch,
        validation_samples_per_problem=args.validation_samples_per_problem,
        cycles_per_sample=args.cycles_per_sample,
        selector_epochs=args.selector_epochs,
        smoother_epochs=args.smoother_epochs,
        transfer_epochs=args.transfer_epochs,
        combined_epochs=args.combined_epochs,
        detector_epochs=args.detector_epochs,
        adaptive_epochs=args.adaptive_epochs,
        learning_rate=args.learning_rate,
        detector_learning_rate=args.detector_learning_rate,
        selector_learning_rate=args.selector_learning_rate,
        training_strategy_source=args.strategy_source,
        fixed_strategy_name=args.fixed_strategy,
        selector_label_max_n=args.selector_label_max_n,
        max_training_levels=args.max_training_levels,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        require_cross_grid_validation=not args.allow_no_cross_grid_validation,
        allow_two_grid_training=args.verification_profile,
    )
    components = NeuralComponents.create(hidden=args.hidden)
    records = train_staged(components, config, checkpoint=args.output, verbose=True)
    csv_path = args.output.with_suffix(".training.csv")
    pd.DataFrame([record.__dict__ for record in records]).to_csv(csv_path, index=False)
    _, metadata = NeuralComponents.load(args.output, device=args.device)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    validation_path = args.output.with_suffix(".common_validation.json")
    validation_path.write_text(
        json.dumps(metadata["common_validation"], indent=2, default=str), encoding="utf-8"
    )
    print(f"checkpoint: {args.output}")
    print(f"training log: {csv_path}")
    print(f"common validation: {validation_path}")


if __name__ == "__main__":
    main()
