"""Residual U-net P teacher and a matching compact CNN student.

Both generators map the ten local A-feature channels to sixteen interpolation
delta channels.  Spatial scales *inside* the teacher are neural feature maps;
they do not replace the numerical MG hierarchy.  The existing transfer
projection and banks assemble P, R=P.T and Ac=P.T@A@P for both architectures.
"""
from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from ..transfer import N_MATRIX_FEATURES
from .models import TransferNet
from .research_transfer import TransferComplexityCaps


def _positive_integer(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [1,{maximum}]')
    return value


def _transfer_contract(module, *, support, reference, parameterization, complexity_caps):
    if support != 'support_preserving':
        raise ValueError('U-net transfer study requires support_preserving')
    if reference != 'frozen_parent':
        raise ValueError('U-net transfer study requires a frozen_parent reference')
    if parameterization not in ('softmax', 'affine'):
        raise ValueError('Transfer parameterization must be softmax or affine')
    if complexity_caps is None:
        caps = TransferComplexityCaps(max_p_ratio=1., max_ac_ratio=1.15,
            complexity_reference='parent', max_operator_complexity_ratio=1.15)
    elif isinstance(complexity_caps, TransferComplexityCaps):
        caps = complexity_caps
    else:
        caps = TransferComplexityCaps(**complexity_caps)
    module.support = support
    module.reference = reference
    module.parameterization = parameterization
    module.projection_version = 2
    module.complexity_caps = asdict(caps)


def _contract_spec(module):
    return dict(support=module.support, reference=module.reference,
                parameterization=module.parameterization,
                complexity_caps=dict(module.complexity_caps))


class ResidualFeatureBlock(nn.Module):
    """Learn a local feature update, with a projection when channels change."""
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.shortcut = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)
        self.update = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1),
                                    nn.GELU(), nn.Conv2d(outgoing, outgoing, 3, padding=1))

    def forward(self, x):
        return self.shortcut(x) + self.update(x)


def _stage(incoming, outgoing, count):
    return nn.Sequential(ResidualFeatureBlock(incoming, outgoing),
                         *(ResidualFeatureBlock(outgoing, outgoing) for _ in range(count - 1)))


class ResidualUNetTransferTeacher(nn.Module):
    """Training-only multi-resolution generator with concatenating U-net skips.

    Strided padded convolutions and explicit skip-sized interpolation support
    odd, rectangular and even one-cell spatial axes.  No normalization uses
    batch statistics, so the coarsest feature maps also support batch size one.
    """
    training_only = True

    def __init__(self, hidden=16, depth=2, blocks_per_stage=1, *,
                 support='support_preserving', reference='frozen_parent',
                 parameterization='softmax', complexity_caps=None):
        super().__init__()
        self.hidden = _positive_integer(hidden, 'hidden', 128)
        self.depth = _positive_integer(depth, 'depth', 4)
        self.blocks_per_stage = _positive_integer(blocks_per_stage, 'blocks_per_stage', 4)
        self.candidates = 16
        _transfer_contract(self, support=support, reference=reference,
                           parameterization=parameterization, complexity_caps=complexity_caps)
        widths = [hidden * 2 ** level for level in range(depth + 1)]
        self.stem = nn.Conv2d(N_MATRIX_FEATURES, widths[0], 3, padding=1)
        self.encoders = nn.ModuleList(_stage(widths[i], widths[i], blocks_per_stage)
                                      for i in range(depth))
        self.downsamples = nn.ModuleList(nn.Conv2d(widths[i], widths[i + 1], 3, stride=2, padding=1)
                                         for i in range(depth))
        self.bottleneck = _stage(widths[-1], widths[-1], blocks_per_stage)
        self.decoders = nn.ModuleList(_stage(widths[i + 1] + widths[i], widths[i], blocks_per_stage)
                                      for i in reversed(range(depth)))
        self.head = nn.Conv2d(hidden, self.candidates, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != N_MATRIX_FEATURES or min(x.shape[-2:]) < 1:
            raise ValueError('Transfer features must have shape (batch,10,height,width)')
        h = self.stem(x)
        skips = []
        for encoder, downsample in zip(self.encoders, self.downsamples):
            h = encoder(h)
            skips.append(h)
            h = downsample(h)
        h = self.bottleneck(h)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            h = decoder(torch.cat((h, skip), dim=1))
        return self.head(h)

    def unet_transfer_spec(self):
        return dict(architecture='teacher', schema=1, training_only=True,
                    hidden=self.hidden, depth=self.depth, blocks_per_stage=self.blocks_per_stage,
                    **_contract_spec(self))


class CompactTransferStudent(TransferNet):
    """The same small CNN architecture for direct and distilled training."""
    training_only = False

    def __init__(self, hidden=8, body_kind='residual5', *, support='support_preserving',
                 reference='frozen_parent', parameterization='softmax', complexity_caps=None):
        _positive_integer(hidden, 'hidden', 128)
        super().__init__(hidden=hidden, candidates=16, body_kind=body_kind)
        _transfer_contract(self, support=support, reference=reference,
                           parameterization=parameterization, complexity_caps=complexity_caps)

    def unet_transfer_spec(self):
        return dict(architecture='student', schema=1, training_only=False,
                    hidden=self.hidden, body_kind=self.body_kind, **_contract_spec(self))


def make_unet_transfer(architecture='teacher', **kwargs):
    """Restore only known, schema-checked roles; a teacher cannot load as deployable."""
    if architecture not in ('teacher', 'student'):
        raise ValueError('Unknown U-net transfer architecture')
    schema = kwargs.pop('schema', 1)
    training_only = kwargs.pop('training_only', architecture == 'teacher')
    if schema != 1 or training_only is not (architecture == 'teacher'):
        raise ValueError('Incompatible U-net transfer checkpoint metadata')
    cls = ResidualUNetTransferTeacher if architecture == 'teacher' else CompactTransferStudent
    return cls(**kwargs)
