"""Physical channel pruning for the direct transfer CNN.

All residual additions use one shared retained-channel set.  The resulting
network is an ordinary, narrower ``CompactTransferStudent``: no pruning masks,
hooks or sparse neural kernels are needed during inference.  Numerical P
support, projection and Galerkin construction remain the existing contracts.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .models import Components
from .research_runtime import expert_signature
from .unet_transfer import CompactTransferStudent


_CONVOLUTIONS = ('body.net.0', 'body.net.2', 'body.residual.0',
                 'body.residual.2', 'body.refine', 'head')


def prune_direct(model: Components, hidden: int) -> tuple[Components, dict]:
    """Return a pruned copy and an auditable deterministic channel-selection report.

    The score of index ``i`` is the sum of absolute incident weights and output
    biases over all occurrences of that hidden channel, including the head's
    input weights.  A hidden-to-hidden weight contributes to both its incoming
    and outgoing channel scores.  Ties prefer the smaller original index.  The
    same selection is used for the stem, residual stream and intermediate
    residual branch; this deliberately conservative grouping respects every
    identity addition without inserting channel projections.

    Only the direct/distilled study's standard residual5 student is supported.
    The caller is responsible for choosing a *direct* trained checkpoint and
    for any subsequent task fine-tuning; no teacher or training is used here.
    ``hidden == original hidden`` is an exact function-preserving copy.
    """
    if not isinstance(model, Components):
        raise TypeError('prune_direct requires Components')
    if type(model.transfer) is not CompactTransferStudent:
        raise ValueError('Pruning requires a CompactTransferStudent, not a teacher')
    if model.transfer.body_kind != 'residual5':
        raise ValueError('Direct channel pruning currently requires residual5')
    original_hidden = model.transfer.hidden
    if isinstance(hidden, bool) or not isinstance(hidden, int) or not 1 <= hidden <= original_hidden:
        raise ValueError(f'hidden must be an integer in [1,{original_hidden}]')

    result = deepcopy(model)
    source = result.transfer
    layers = {name: source.get_submodule(name) for name in _CONVOLUTIONS}
    scores = torch.zeros(original_hidden, dtype=torch.float64)
    for name, layer in layers.items():
        if not isinstance(layer, nn.Conv2d) or layer.groups != 1:
            raise ValueError(f'Unsupported convolution at {name}')
        weight = layer.weight.detach().to(device='cpu', dtype=torch.float64)
        bias = layer.bias.detach().to(device='cpu', dtype=torch.float64)
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError('Cannot prune nonfinite transfer parameters')
        if name != 'head':
            scores += weight.abs().sum((1, 2, 3)) + bias.abs()
        if name != 'body.net.0':
            scores += weight.abs().sum((0, 2, 3))
    ranking = sorted(range(original_hidden), key=lambda i: (-float(scores[i]), i))
    kept = sorted(ranking[:hidden])
    original_count = sum(p.numel() for p in source.parameters())
    source_signature = expert_signature(source)

    # Construction initializes weights, but must not change the caller's RNG
    # trajectory. All random initial values are subsequently overwritten.
    with torch.random.fork_rng(devices=[]):
        narrower = CompactTransferStudent(hidden=hidden, body_kind=source.body_kind,
            support=source.support, reference=source.reference,
            parameterization=source.parameterization,
            complexity_caps=deepcopy(source.complexity_caps))
    template = next(source.parameters())
    narrower.to(device=template.device, dtype=template.dtype)
    index = torch.tensor(kept, dtype=torch.long, device=template.device)
    with torch.no_grad():
        for name, old in layers.items():
            new = narrower.get_submodule(name)
            weight, bias = old.weight, old.bias
            if name != 'head':
                weight = weight.index_select(0, index)
                bias = bias.index_select(0, index)
            if name != 'body.net.0':
                weight = weight.index_select(1, index)
            new.weight.copy_(weight)
            new.bias.copy_(bias)
            new.weight.requires_grad_(old.weight.requires_grad)
            new.bias.requires_grad_(old.bias.requires_grad)
    narrower.train(source.training)
    result.transfer = narrower
    new_count = sum(p.numel() for p in narrower.parameters())
    report = dict(schema=1, method='physical_shared_channel_l1',
        source_transfer_signature=source_signature,
        pruned_transfer_signature=expert_signature(narrower),
        signature_scope='immediately_after_pruning_before_fine_tuning',
        original_hidden=original_hidden, hidden=hidden,
        original_parameter_count=original_count, pruned_parameter_count=new_count,
        parameter_reduction_fraction=1. - new_count / original_count,
        kept_channels=kept, ranking=ranking, scores=scores.tolist(),
        scoring='sum_absolute_incident_weights_and_output_biases',
        dependency_group='shared_indices_across_all_hidden_convolutions_and_residual_adds',
        convolution_names=list(_CONVOLUTIONS), physically_compacted=True,
        fine_tuned=False, teacher_used=False,
        training_baseline_digest=model.metadata.get('training_baseline_digest'))
    result.metadata['direct_pruning'] = deepcopy(report)
    if hidden != original_hidden:
        result.mark_policy_stale('direct_channels_pruned_requires_revalidation')
    return result, report
