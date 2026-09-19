"""New temporal checkpoints; legacy policies/certificates are NEVER inherited."""
from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
import torch
from .models import NeuralSmootherNet, SwitchNet
from .provenance import module_signature

FORMAT = 12
CONTRACT = 'v66_standalone_shared_hierarchy_one_way_prefix'


@dataclass
class TemporalComponents:
    smoother: NeuralSmootherNet
    switch: SwitchNet | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(cls, *, hidden=16, dtype=torch.float64, body_kind='residual5'):
        net = NeuralSmootherNet(hidden=hidden, split_direction_gain=True, body_kind=body_kind).to(dtype=dtype)
        return cls(net, metadata={'contract': CONTRACT, 'policy_validated': False, 'fast_path_quality_certified': False})

    def eval(self):
        self.smoother.eval()
        if self.switch is not None: self.switch.eval()
        return self

    def save(self, path):
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        m = self.smoother
        s = self.switch
        payload = {'format_version': FORMAT, 'contract': CONTRACT,
            'smoother_spec': dict(hidden=m.head.in_channels, coefficient_scale=m.coefficient_scale,
                basis_count=m.basis_count, split_direction_gain=m.split_direction_gain,
                direction_epsilon=m.direction_epsilon, body_kind=m.body_kind),
            'dtype': str(next(m.parameters()).dtype).split('.')[-1],
            'smoother': m.state_dict(),
            'switch_spec': None if s is None else dict(in_features=s.feature_mean.numel(), k_values=list(s.k_values), hidden=s.body[0].out_features),
            'switch': None if s is None else s.state_dict(), 'metadata': deepcopy(self.metadata)}
        tmp = p.with_suffix(p.suffix + '.tmp'); torch.save(payload, tmp); tmp.replace(p)

    @classmethod
    def load(cls, path):
        # Own new checkpoints contain only safe primitives and tensors.
        d = torch.load(path, map_location='cpu', weights_only=True)
        if d.get('format_version') != FORMAT or d.get('contract') != CONTRACT:
            raise ValueError('not a v6.6 temporal checkpoint; migrate old smoother explicitly')
        if d['dtype'] not in {'float32','float64'}: raise ValueError('unsupported model dtype')
        m = NeuralSmootherNet(**d['smoother_spec']).to(dtype=getattr(torch, d['dtype']))
        m.load_state_dict(d['smoother'], strict=True)
        s = None
        if d['switch_spec'] is not None:
            s = SwitchNet(**d['switch_spec']).to(dtype=d['switch']['feature_mean'].dtype)
            s.load_state_dict(d['switch'], strict=True)
        return cls(m, s, deepcopy(d['metadata'])).eval()

    @classmethod
    def migrate_v65(cls, path, *, trusted=False):
        # Legacy files may carry Python metadata. Explicit acknowledgement is
        # required before the pickle-capable loader is used.
        if not trusted:
            raise ValueError('legacy pickle requires trusted=True / --trust-legacy-checkpoint')
        d = torch.load(path, map_location='cpu', weights_only=False)
        if d.get('format_version', d.get('checkpoint_format_version')) not in {10,11}:
            raise ValueError('only v6.4/v6.5 format 10/11 smoother migration supported')
        meta=d.get('metadata',{}); state=d['models']['smoother']
        h=state['body.net.0.weight'].shape[0]
        model=NeuralSmootherNet(hidden=h,
                basis_count=int(meta.get('smoother_basis_count',state['head.weight'].shape[0]//9)),
                coefficient_scale=float(meta.get('smoother_coefficient_scale',.75)),
                split_direction_gain=bool(meta.get('split_direction_gain','gain_head.weight' in state)),
                body_kind=meta.get('smoother_body_kind','residual5'),
                direction_epsilon=float(meta.get('direction_epsilon',1e-8))).to(dtype=state['head.weight'].dtype)
        model.load_state_dict(state, strict=True)
        metadata={'contract':CONTRACT, 'migration_source':str(Path(path).name),
                  'migration_retained':['smoother_direction_and_gain'],
                  'migration_removed':['spatial_detector','global_selector','router','learned_transfer_auxiliary_cycle','old_quality_certificates'],
                  'policy_validated':False, 'fast_path_quality_certified':False,
                  'smoother_signature':module_signature(model)}
        return cls(model,metadata=metadata).eval()
