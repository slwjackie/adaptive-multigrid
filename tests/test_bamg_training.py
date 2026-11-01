"""Two-stage training, exact resume, and immutable-reference contracts."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from adaptive_mg.config import MGConfig
from adaptive_mg.pde import DiffusionCase, assemble_stiffness
from adaptive_mg.transfer import TransferPattern
from adaptive_mg.v67.bamg_transfer import (
    MaskCNN, WeightCNN, direct_pattern, geometry_constraints, project_weights,
    select_mask,
)
from adaptive_mg.v67.bamg_solver import ReferenceModels
from adaptive_mg.v67.bamg_training import (
    FinetuneSettings, SupervisedSettings, curriculum_cycles,
    load_training_checkpoint, supervised_objective, train_finetune,
    train_supervised, refit_fixed_mask_labels, warmup_fixed_masks,
)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def operator():
    return assemble_stiffness(DiffusionCase(n=7, epsilon=.2, angle_deg=30.,
                                            contrast=1., pattern="uniform"))


def models(seed=101):
    torch.manual_seed(seed)
    return MaskCNN(hidden=3, depth=1).double(), WeightCNN(hidden=3, depth=1).double()


def labels():
    pattern = direct_pattern((7, 7))
    generator = torch.Generator().manual_seed(91)
    mask = select_mask(torch.randn((49, 16), generator=generator, dtype=torch.float64), pattern, 4)
    target = project_weights(torch.randn((49, 16), generator=generator,
                                        dtype=torch.float64) * .1, mask, pattern)
    return dict(A=operator(), shape=(7, 7), pattern=pattern, mask=mask.numpy(),
                weights=target.detach().numpy(), vectors=np.random.default_rng(7).normal(size=(49, 5)),
                split="train")


def equal_models(left, right):
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)


def test_supervised_both_independent_cnns_learn_without_topk_gradient():
    mask, weight = models()
    settings = SupervisedSettings(updates=8, learning_rate=.01, alpha_action=.05)
    teacher = labels()
    before_mask, before_weight = deepcopy(mask), deepcopy(weight)
    start, _ = supervised_objective(mask, weight, [teacher], settings=settings)
    history = train_supervised(mask, weight, [teacher], settings=settings)
    finish, report = supervised_objective(mask, weight, [teacher], settings=settings)
    assert finish < start
    assert any(not torch.equal(a, b) for a, b in zip(mask.parameters(), before_mask.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(weight.parameters(), before_weight.parameters()))
    assert all(row["hard_topk_gradient"] is False for row in history)
    assert report["normalization"] == "equal levels, equal rows"


def test_teacher_labels_join_by_coarse_column_after_slot_permutation():
    mask, weight = models()
    level = labels()
    permutation = np.random.default_rng(3).permutation(16)
    old = level["pattern"]
    other = dict(level, pattern=TransferPattern(old.fine_shape, old.coarse_shape,
                 old.columns[:, permutation], old.bilinear_weights[:, permutation]),
                 mask=level["mask"][:, permutation], weights=level["weights"][:, permutation])
    settings = SupervisedSettings(alpha_action=.2)
    a, _ = supervised_objective(mask, weight, [level], settings=settings)
    b, _ = supervised_objective(mask, weight, [other], settings=settings)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_injection_rows_are_excluded_from_learnable_losses():
    mask, weight = models()
    level = labels()
    altered = dict(level, weights=level["weights"].copy())
    _, injection, _ = geometry_constraints(level["pattern"])
    rows = np.flatnonzero(injection >= 0)
    altered["weights"][rows, injection[rows]] += 10.
    first, _ = supervised_objective(mask, weight, [level])
    second, _ = supervised_objective(mask, weight, [altered])
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_stage_one_resume_is_exact_and_rejects_stale_labels(tmp_path):
    base_mask, base_weight = models()
    full_mask, full_weight = deepcopy(base_mask), deepcopy(base_weight)
    resumed_mask, resumed_weight = deepcopy(base_mask), deepcopy(base_weight)
    settings = SupervisedSettings(updates=4, learning_rate=.003)
    teacher = labels()
    full = train_supervised(full_mask, full_weight, [teacher], settings=settings)
    path = tmp_path / "stage1.pt"
    train_supervised(resumed_mask, resumed_weight, [teacher], settings=settings,
                     checkpoint_path=path, stop_after=2, manifest={"dataset": "train-a"})
    assert load_training_checkpoint(path)["status"] == "paused"
    # Simulate a different process with different model/RNG initialization.
    resumed_mask, resumed_weight = models(seed=999)
    resumed = train_supervised(resumed_mask, resumed_weight, [teacher], settings=settings,
                     checkpoint_path=path, resume=True, manifest={"dataset": "train-a"})
    equal_models(full_mask, resumed_mask); equal_models(full_weight, resumed_weight)
    assert full == resumed
    state = load_training_checkpoint(path)
    assert state["status"] == "complete" and state["update"] == 4
    stale = dict(teacher, weights=teacher["weights"].copy())
    stale["weights"][teacher["mask"]] += .001
    with pytest.raises(ValueError, match="provenance"):
        train_supervised(resumed_mask, resumed_weight, [stale], settings=settings,
                         checkpoint_path=path, resume=True, manifest={"dataset": "train-a"})
    with pytest.raises(ValueError, match="provenance"):
        train_supervised(resumed_mask, resumed_weight, [teacher], settings=replace(settings, seed=4),
                         checkpoint_path=path, resume=True, manifest={"dataset": "train-a"})
    with pytest.raises(ValueError, match="placement_hash"):
        train_supervised(resumed_mask.float(), resumed_weight.float(), [teacher], settings=settings,
                         checkpoint_path=path, resume=True, manifest={"dataset": "train-a"})


def test_stage_two_resume_freezes_reference_masks_and_updates_only_weights(tmp_path, monkeypatch):
    mask, base = models()
    cfg = MGConfig(mode="classical", strategy_name="line_alt_bilinear_full",
                   pre_steps=1, post_steps=1, stencil_backend="csr")
    reference = ReferenceModels(mask, base, mg_config=cfg)
    before_mask, before_weight = deepcopy(reference.mask_model), deepcopy(reference.weight_model)
    example = dict(A=operator(), shape=(7, 7), case_id="test-train-a", split="train")
    before_support = reference.masks_for(example["A"], example["shape"])
    settings = FinetuneSettings(updates=3, cycles=1, final_cycles=2, curriculum_step=1,
        smoothing_sweeps=1, hard_cycles=1, learning_rate=1e-3)
    full_weight = deepcopy(base)
    full = train_finetune(full_weight, reference, [example], settings=settings, mg_config=cfg)
    path = tmp_path / "stage2.pt"
    partial = deepcopy(base)
    train_finetune(partial, reference, [example], settings=settings, mg_config=cfg,
                   checkpoint_path=path, stop_after=1)
    fresh_reference = ReferenceModels(mask, base, mg_config=cfg)
    _, resumed_weight = models(seed=22)
    # Resuming must reuse serialized masks; selecting new connections is forbidden.
    def forbidden(*args, **kwargs):
        raise AssertionError("top-k must not run while restoring fixed masks")
    monkeypatch.setattr("adaptive_mg.v67.bamg_solver.select_mask", forbidden)
    resumed = train_finetune(resumed_weight, fresh_reference, [example], settings=settings,
                            mg_config=cfg, checkpoint_path=path, resume=True)
    equal_models(full_weight, resumed_weight)
    assert full == resumed
    assert [row["cycles"] for row in full] == [1, 2, 2]
    assert all(row["probe_kinds"] == ["random", "smoothed", "hard"] for row in full)
    assert all(row["valid_probes"] == 3 for row in full)
    equal_models(before_mask, reference.mask_model)
    equal_models(before_weight, reference.weight_model)
    assert all(p.grad is None for p in reference.mask_model.parameters())
    assert all(p.grad is None for p in reference.weight_model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(base.parameters(), full_weight.parameters()))
    for old, new in zip(before_support, reference.masks_for(example["A"], example["shape"])):
        assert torch.equal(old, new)
    state = load_training_checkpoint(path)
    assert state["extra"]["mask_hash"]
    assert state["status"] == "complete"


def test_failure_is_recorded_instead_of_skipping_example(tmp_path, monkeypatch):
    mask, weight = models()
    path = tmp_path / "failed.pt"
    def bad(*args, **kwargs):
        return torch.tensor(float("nan"), requires_grad=True), {}
    monkeypatch.setattr("adaptive_mg.v67.bamg_training.supervised_objective", bad)
    with pytest.raises(RuntimeError, match="nonfinite"):
        train_supervised(mask, weight, [labels()], settings=SupervisedSettings(updates=2),
                         checkpoint_path=path)
    state = load_training_checkpoint(path)
    assert state["status"] == "failed"
    assert state["update"] == 0 and state["history"] == []
    assert state["failure"]["attempted_update"] == 1


def test_optional_fixed_mask_refit_and_weight_warmup_preserve_reference(tmp_path):
    mask, weight = models()
    reference = ReferenceModels(mask, weight)
    level = labels()
    fixed = reference.masks_for(level["A"], level["shape"])
    refitted = refit_fixed_mask_labels([level], fixed)
    assert np.array_equal(refitted[0]["mask"], fixed[0].numpy())
    before_mask, before_weight = deepcopy(reference.mask_model), deepcopy(reference.weight_model)
    path = tmp_path / "warmup.pt"
    history = warmup_fixed_masks(weight, reference, refitted,
        settings=SupervisedSettings(updates=2, alpha_mask=0.), checkpoint_path=path)
    assert len(history) == 2
    assert load_training_checkpoint(path)["phase"] == "fixed_mask_warmup"
    equal_models(reference.mask_model, before_mask)
    equal_models(reference.weight_model, before_weight)
    for old, new in zip(fixed, reference.masks_for(level["A"], level["shape"])):
        assert torch.equal(old, new)


def test_split_and_curriculum_validation():
    mask, weight = models()
    with pytest.raises(ValueError, match="TRAIN"):
        train_supervised(mask, weight, [dict(labels(), split="validation")])
    settings = FinetuneSettings(updates=4)
    assert [curriculum_cycles(i, settings) for i in range(4)] == [3, 3, 6, 6]
    with pytest.raises(ValueError, match="required together"):
        curriculum_cycles(0, replace(settings, time_weights=(1.,) * 6))
