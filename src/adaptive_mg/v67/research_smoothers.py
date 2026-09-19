"""Operator-only research smoothers with explicit geometric contracts.

The deployed protocol is identical to NeuralSmootherNet: one normalized
9-point basis and one scalar gain per sample.  D4 is exact on square/rectangular
lattices (rotations may exchange their dimensions).  The O(2) model uses exact
continuous irreducible-representation algebra and analytic steerable kernels;
sampling them on a square lattice is NOT exact arbitrary-angle image rotation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from ..models import NeuralSmootherNet, make_spatial_body
from ..transfer import N_MATRIX_FEATURES, OFFSETS_9

SMOOTHER_SPEC_VERSION = 1


def _spatial_d4(x, quarter_turns=0, reflect=False, inverse=False):
    k = int(quarter_turns) % 4
    if inverse:
        x = torch.rot90(x, -k, (-2, -1))
        return x.flip(-1) if reflect else x
    x = x.flip(-1) if reflect else x
    return torch.rot90(x, k, (-2, -1))


def d4_stencil_transform(x, quarter_turns=0, reflect=False, *, inverse=False):
    """Rotate/reflect sites AND their 9 directional channels (channel axis -3)."""
    if x.shape[-3] != 9:
        raise ValueError('nine stencil channels required')
    mapped = []
    for di, dj in OFFSETS_9:
        if reflect:
            dj = -dj
        for _ in range(int(quarter_turns) % 4):
            di, dj = -dj, di
        mapped.append(OFFSETS_9.index((di, dj)))
    permutation = mapped if inverse else [mapped.index(j) for j in range(9)]
    return _spatial_d4(x, quarter_turns, reflect, inverse).index_select(
        -3, torch.as_tensor(permutation, device=x.device))


def d4_feature_transform(x, quarter_turns=0, reflect=False, *, inverse=False):
    """Correct group action on log diagonal plus normalized A stencil features."""
    if x.ndim != 4 or x.shape[1] != N_MATRIX_FEATURES:
        raise ValueError('features must have shape (B,10,H,W)')
    scalar = _spatial_d4(x[:, :1], quarter_turns, reflect, inverse)
    stencil = d4_stencil_transform(x[:, 1:], quarter_turns, reflect, inverse=inverse)
    return torch.cat((scalar, stencil), 1)


class _ResearchSmoother(nn.Module):
    basis_count = 1
    split_direction_gain = True
    coefficient_scale = 1.0
    training_only = False

    def _init_contract(self, kind, hidden, gain_min, gain_max, direction_epsilon, **extra):
        if hidden < 1 or not math.isfinite(gain_min) or not math.isfinite(gain_max):
            raise ValueError('invalid width or gain limits')
        if not 0 < gain_min < gain_max or not math.isfinite(direction_epsilon) or direction_epsilon <= 0:
            raise ValueError('require 0 < gain_min < gain_max and positive epsilon')
        self.kind, self.hidden = kind, int(hidden)
        self.gain_min, self.gain_max = float(gain_min), float(gain_max)
        self.direction_epsilon = float(direction_epsilon)
        self.body_kind = 'research_' + kind
        self._spec = dict(version=SMOOTHER_SPEC_VERSION, kind=kind, hidden=int(hidden),
                          gain_min=float(gain_min), gain_max=float(gain_max),
                          direction_epsilon=float(direction_epsilon), **extra)

    def architecture_spec(self):
        return dict(self._spec)

    def cost_metadata(self):
        return {'parameters': sum(p.numel() for p in self.parameters()),
                'kind': self.kind, 'training_only': self.training_only,
                'equivalent_encoder_evaluations': 1,
                'generation_only': True, 'basis_count': self.basis_count}

    def _gain(self, raw):
        return self.gain_min + (self.gain_max - self.gain_min) * torch.sigmoid(raw)

    def _initial_gain_bias(self):
        target = min(self.gain_max - .01 * (self.gain_max - self.gain_min),
                     max(self.gain_min + .01 * (self.gain_max - self.gain_min), .72))
        p = (target - self.gain_min) / (self.gain_max - self.gain_min)
        return math.log(p / (1. - p))

    def _normalize(self, raw):
        work = raw.float() if raw.dtype in (torch.float16, torch.bfloat16) else raw
        norm = work.square().sum(1, keepdim=True).add(self.direction_epsilon).sqrt()
        return (work / norm).to(raw.dtype)

    def direction_and_gain_single(self, features):
        if features.ndim != 4 or features.shape[1] != N_MATRIX_FEATURES:
            raise ValueError('features must have shape (B,10,H,W)')
        raw, gain_logits = self._raw_and_gain(features)
        return self._normalize(raw), self._gain(gain_logits)

    def direction_and_gain(self, features):
        directions, gains = self.direction_and_gain_single(features)
        return directions[:, None], gains

    def coefficients(self, features):
        return self.direction_and_gain(features)[0]

    def gains(self, features):
        return self.direction_and_gain(features)[1]

    def forward(self, features, stencil_input):
        directions, gains = self.direction_and_gain(features)
        return NeuralSmootherNet.apply_coefficients(directions, stencil_input) * gains[:, :, None, None]


class _DilatedResidualBody(nn.Module):
    def __init__(self, hidden, depth):
        super().__init__()
        self.lift = nn.Conv2d(N_MATRIX_FEATURES, hidden, 3, padding=1)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 if i % 2 else 1
            self.blocks.append(nn.Sequential(nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=dilation, dilation=dilation),
                nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1)))
            nn.init.zeros_(self.blocks[-1][-1].weight)
            nn.init.zeros_(self.blocks[-1][-1].bias)
        self.spatial_convolutions = 1 + 2 * depth
        self.receptive_radius = 1 + sum(3 if i % 2 else 2 for i in range(depth))

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = x + block(x)
        return F.gelu(x)


class OrdinaryResearchSmoother(_ResearchSmoother):
    """Current residual5 control or compact/direct and larger offline variants."""
    def __init__(self, kind='ordinary_control', hidden=16, gain_min=1e-4,
                 gain_max=2., direction_epsilon=1e-8, depth=3):
        super().__init__()
        if kind not in {'ordinary_control', 'student_cnn', 'student_residual', 'teacher_cnn'}:
            raise ValueError('invalid ordinary smoother kind')
        if depth < 1:
            raise ValueError('depth must be positive')
        self._init_contract(kind, hidden, gain_min, gain_max, direction_epsilon, depth=int(depth))
        self.training_only = kind == 'teacher_cnn'
        if self.training_only:
            self.body = _DilatedResidualBody(hidden, depth)
        else:
            body_kind = 'compact3' if kind == 'student_cnn' else 'residual5'
            self.body = make_spatial_body(body_kind, N_MATRIX_FEATURES, hidden)
        self.spatial_convolutions = self.body.spatial_convolutions
        self.receptive_radius = self.body.receptive_radius
        self.head = nn.Conv2d(hidden, 9, 1)
        self.gain_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = 1.
        nn.init.zeros_(self.gain_head.weight)
        nn.init.constant_(self.gain_head.bias, self._initial_gain_bias())

    def _raw_and_gain(self, x):
        hidden = self.body(x)
        return self.head(hidden), self.gain_head(hidden.mean((-2, -1)))


class D4ResearchSmoother(_ResearchSmoother):
    """Exact Reynolds symmetrization, with eight equivalent encoder evaluations.

    Square inputs use one batch-concatenated encoder call.  Rectangles use two
    calls because odd rotations exchange H and W.  FLOPs still increase eightfold.
    """
    def __init__(self, hidden=16, gain_min=1e-4, gain_max=2., direction_epsilon=1e-8, depth=3):
        super().__init__()
        self._init_contract('d4', hidden, gain_min, gain_max, direction_epsilon, depth=int(depth))
        self.encoder = OrdinaryResearchSmoother('ordinary_control', hidden, gain_min,
                                                gain_max, direction_epsilon, depth)
        self.spatial_convolutions = self.encoder.spatial_convolutions
        self.receptive_radius = self.encoder.receptive_radius

    @property
    def head(self):
        return self.encoder.head

    def cost_metadata(self):
        return dict(super().cost_metadata(), equivalent_encoder_evaluations=8,
                    encoder_batch_calls_square=1, encoder_batch_calls_rectangle=2,
                    symmetry='exact_D4_up_to_roundoff', generation_flop_multiplier=8)

    def _raw_and_gain(self, x):
        groups = defaultdict(list)
        for reflection in (False, True):
            for k in range(4):
                transformed = d4_feature_transform(x, k, reflection)
                groups[tuple(transformed.shape[-2:])].append((k, reflection, transformed))
        directions, gain_logits = [], []
        for entries in groups.values():
            raw, gain = self.encoder._raw_and_gain(torch.cat([z[2] for z in entries], 0))
            for (k, reflection, _), raw_part, gain_part in zip(
                    entries, raw.split(x.shape[0]), gain.split(x.shape[0])):
                directions.append(d4_stencil_transform(raw_part, k, reflection, inverse=True))
                gain_logits.append(gain_part)
        return torch.stack(directions).mean(0), torch.stack(gain_logits).mean(0)


class O2Fields(NamedTuple):
    scalar: torch.Tensor  # B,S,H,W; reflection-even order 0
    vector: torch.Tensor  # B,V,2,H,W; polar order 1
    tensor: torch.Tensor  # B,T,2,H,W; (Kxx-Kyy,2Kxy), order 2


def transform_o2_fields(fields, angle, reflect=False):
    """Continuous fiber action R(angle) F; does not resample spatial positions."""
    def irrep(x, order):
        c, s = math.cos(order * angle), math.sin(order * angle)
        a, b = x[:, :, 0], x[:, :, 1]
        b = -b if reflect else b
        return torch.stack((c * a - s * b, s * a + c * b), 2)
    return O2Fields(fields.scalar, irrep(fields.vector, 1), irrep(fields.tensor, 2))


def analytic_harmonics(points, order):
    """Analytic angular kernels with a radial Gaussian envelope, including r=0."""
    if order not in (0, 1, 2):
        raise ValueError('only scalar/order-1/order-2 kernels implemented')
    x, y = points.unbind(-1)
    radial = torch.exp(-.5 * (x.square() + y.square()))
    if order == 0:
        return radial[..., None]
    if order == 1:
        return torch.stack((x * radial, y * radial), -1)
    return torch.stack(((x.square() - y.square()) * radial, 2 * x * y * radial), -1)


def stencil_to_o2_fields(features, offsets=None):
    """Full-rank signed moment lifting of the SAME ten operator-only channels.

    Rotating offsets together with the operator stencil gives exact tensor
    covariance. Resampling an arbitrary rotated PDE onto nine fixed grid offsets
    does not generally commute with this lift.
    """
    if features.ndim != 4 or features.shape[1] != N_MATRIX_FEATURES:
        raise ValueError('features must have shape (B,10,H,W)')
    points = torch.as_tensor(OFFSETS_9, device=features.device, dtype=features.dtype) if offsets is None else offsets.to(features)
    if points.shape != (9, 2):
        raise ValueError('offsets must be (9,2)')
    stencil = features[:, 1:]
    x, y = points.unbind(-1)
    radius2 = x.square() + y.square()
    scalar = torch.cat((features[:, :2], stencil.sum(1, keepdim=True),
                       (stencil * radius2[None, :, None, None]).sum(1, keepdim=True)), 1)
    # Two radial order-1 moments retain the two independent axial/diagonal
    # vector fields. Dropping the radius-weighted moment would lose two inputs.
    vector = torch.stack((torch.einsum('bkhw,kd->bdhw', stencil, points),
        torch.einsum('bkhw,kd->bdhw', stencil, points * radius2[:, None])), 1)
    quadratic = torch.stack((x.square() - y.square(), 2 * x * y), -1)
    tensor = torch.einsum('bkhw,kd->bdhw', stencil, quadratic)[:, None]
    return O2Fields(scalar, vector, tensor)


class _FiberMix(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return torch.einsum('oi,bidhw->bodhw', self.weight, x)


class _RadialMix(nn.Module):
    """Continuously radial kernel sampled at 3x3 grid positions.

    Independent learned mixing of three Gaussian radial basis functions.
    No angular dependence or tensor-component bias is introduced.
    """
    def __init__(self, channels):
        super().__init__()
        coords = torch.stack(torch.meshgrid(torch.arange(-1., 2.), torch.arange(-1., 2.), indexing='ij'), -1)
        radius2 = coords.square().sum(-1)
        basis = torch.stack([torch.exp(-a * radius2) for a in (.5, 1.5, 4.)])
        basis = basis / basis.sum((-2, -1), keepdim=True)
        self.register_buffer('basis', basis)
        self.weight = nn.Parameter(torch.empty(channels, channels, 3))
        nn.init.normal_(self.weight, std=.1 / math.sqrt(channels))

    def forward(self, x):
        kernel = torch.einsum('oir,rhw->oihw', self.weight, self.basis)
        if x.ndim == 4:
            return F.conv2d(x, kernel, padding=1)
        b, c, d, h, w = x.shape
        result = F.conv2d(x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w), kernel, padding=1)
        return result.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)


class _HarmonicLift(nn.Module):
    def __init__(self, inputs, outputs, order):
        super().__init__()
        coords = torch.stack(torch.meshgrid(torch.arange(-1., 2.), torch.arange(-1., 2.), indexing='ij'), -1)
        harmonic = analytic_harmonics(coords, order).permute(2, 0, 1)
        # One common scalar normalization preserves the irrep transformation.
        harmonic = harmonic / harmonic.square().sum().sqrt()
        self.register_buffer('harmonic', harmonic)
        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        nn.init.normal_(self.weight, std=.1 / math.sqrt(inputs))

    def forward(self, x):
        kernel = torch.einsum('oi,dhw->odihw', self.weight, self.harmonic)
        out = F.conv2d(x, kernel.flatten(0, 1), padding=1)
        return out.reshape(x.shape[0], self.weight.shape[0], 2, *x.shape[-2:])


class O2PointwiseBlock(nn.Module):
    """Exactly O(2)-equivariant nonlinear fiber algebra, including reflections."""
    def __init__(self, scalars, vectors):
        super().__init__()
        self.scalar_mix = nn.Conv2d(scalars + 2 * vectors, scalars, 1)
        self.gates = nn.Conv2d(scalars, 4 * vectors, 1)
        self.vector_mix = _FiberMix(vectors, vectors)
        self.tensor_mix = _FiberMix(vectors, vectors)

    def forward(self, fields):
        s, v, t = fields
        # Norms are O(2) invariants; no componentwise activation on non-scalars.
        inv = torch.cat((s, v.square().sum(2), t.square().sum(2)), 1)
        snew = s + .1 * F.gelu(self.scalar_mix(inv))
        gates = torch.tanh(self.gates(snew)).chunk(4, 1)
        vm, tm = self.vector_mix(v), self.tensor_mix(t)
        vx, vy = vm.unbind(2)
        tx, ty = tm.unbind(2)
        tv = torch.stack((tx * vx + ty * vy, ty * vx - tx * vy), 2)
        vv = torch.stack((vx.square() - vy.square(), 2 * vx * vy), 2)
        # Invariant denominators bound polynomial interactions without clipping
        # tensor components independently (which would break equivariance).
        tv = tv / (1 + tm.square().sum(2, keepdim=True))
        vv = vv / (1 + vm.square().sum(2, keepdim=True))
        vnew = v + .1 * (gates[0][:, :, None] * vm + gates[1][:, :, None] * tv)
        tnew = t + .1 * (gates[2][:, :, None] * tm + gates[3][:, :, None] * vv)
        return O2Fields(snew, vnew, tnew)


class _O2SpatialBlock(nn.Module):
    def __init__(self, scalars, vectors):
        super().__init__()
        self.scalar_radial = _RadialMix(scalars)
        self.vector_radial = _RadialMix(vectors)
        self.tensor_radial = _RadialMix(vectors)
        self.lift_vector = _HarmonicLift(scalars, vectors, 1)
        self.lift_tensor = _HarmonicLift(scalars, vectors, 2)
        self.pointwise = O2PointwiseBlock(scalars, vectors)

    def forward(self, fields):
        s, v, t = fields
        return self.pointwise(O2Fields(s + self.scalar_radial(s),
            v + self.vector_radial(v) + self.lift_vector(s),
            t + self.tensor_radial(t) + self.lift_tensor(s)))


class O2ResearchSmoother(_ResearchSmoother):
    """Small pure-Torch scalar/order1/order2 steerable residual CNN.

    O(2) kernel/fiber/head algebra is analytic; lattice D4 equivariance is exact
    up to rounding. Arbitrary spatial rotation requires interpolation and has
    discretization, boundary, and finite-stencil errors, which must be measured.
    """
    def __init__(self, hidden=12, gain_min=1e-4, gain_max=2., direction_epsilon=1e-8,
                 depth=3, kind='o2'):
        super().__init__()
        if kind not in {'o2', 'teacher_o2'} or depth < 1:
            raise ValueError('invalid O2 kind/depth')
        self._init_contract(kind, hidden, gain_min, gain_max, direction_epsilon, depth=int(depth))
        self.training_only = kind == 'teacher_o2'
        self.vector_channels = max(1, hidden // 2)
        self.scalar_lift = nn.Conv2d(4, hidden, 1)
        self.vector_lift = _FiberMix(2, self.vector_channels)
        self.tensor_lift = _FiberMix(1, self.vector_channels)
        self.blocks = nn.ModuleList([_O2SpatialBlock(hidden, self.vector_channels) for _ in range(depth)])
        self.head = nn.Conv2d(hidden, 3, 1)
        self.vector_head = _FiberMix(self.vector_channels, 1)
        self.tensor_head = _FiberMix(self.vector_channels, 1)
        self.gain_head = nn.Linear(hidden, 1)
        self.spatial_convolutions = depth
        self.receptive_radius = depth
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([1., -1.5, .5]))
        nn.init.zeros_(self.vector_head.weight)
        nn.init.zeros_(self.tensor_head.weight)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.constant_(self.gain_head.bias, self._initial_gain_bias())

    def cost_metadata(self):
        return dict(super().cost_metadata(), scalar_channels=self.hidden,
                    order1_channels=self.vector_channels, order2_channels=self.vector_channels,
                    symmetry='analytic_O2_kernels_and_fibers; sampled_lattice_D4',
                    arbitrary_angle_lattice_equivariance='approximate_requires_measurement',
                    input_moment_lifting='full_rank_signed_moments_of_same_10_features')

    def encode_fields(self, x):
        s, v, t = stencil_to_o2_fields(x)
        fields = O2Fields(F.gelu(self.scalar_lift(s)), self.vector_lift(v), self.tensor_lift(t))
        for block in self.blocks:
            fields = block(fields)
        return fields

    def stencil_logits_from_fields(self, fields, offsets=None):
        s, v, t = fields
        points = torch.as_tensor(OFFSETS_9, device=s.device, dtype=s.dtype) if offsets is None else offsets.to(s)
        x, y = points.unbind(-1)
        radius2 = x.square() + y.square()
        radial_basis = torch.stack((torch.ones_like(radius2), radius2, radius2.square()), -1)
        raw = torch.einsum('bchw,kc->bkhw', self.head(s), radial_basis)
        raw = raw + torch.einsum('bdhw,kd->bkhw', self.vector_head(v)[:, 0], points)
        harmonic2 = torch.stack((x.square() - y.square(), 2 * x * y), -1)
        return raw + torch.einsum('bdhw,kd->bkhw', self.tensor_head(t)[:, 0], harmonic2)

    def _raw_and_gain(self, x):
        fields = self.encode_fields(x)
        return self.stencil_logits_from_fields(fields), self.gain_head(fields.scalar.mean((-2, -1)))


def make_research_smoother(kind='ordinary_control', *, hidden=None, gain_min=1e-4,
                           gain_max=2., direction_epsilon=1e-8, depth=None):
    """Factory; teacher kinds are explicitly tagged training_only for runtime guards."""
    if hidden is None:
        hidden = 48 if kind == 'teacher_cnn' else 32 if kind == 'teacher_o2' else 12 if kind == 'o2' else 16
    if depth is None:
        depth = 6 if kind.startswith('teacher_') else 3
    kwargs = dict(hidden=hidden, gain_min=gain_min, gain_max=gain_max,
                  direction_epsilon=direction_epsilon, depth=depth)
    if kind == 'd4':
        return D4ResearchSmoother(**kwargs)
    if kind in {'o2', 'teacher_o2'}:
        return O2ResearchSmoother(kind=kind, **kwargs)
    return OrdinaryResearchSmoother(kind=kind, **kwargs)


def research_smoother_from_spec(spec):
    spec = dict(spec)
    if spec.pop('version', None) != SMOOTHER_SPEC_VERSION:
        raise ValueError('unsupported research smoother spec version')
    return make_research_smoother(**spec)
