"""Heavy A-conditioned generators and NumPy-compiled online micro-models."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
import numpy as np
import torch
from torch import nn
from ..models import NeuralSmootherNet, make_spatial_body
from ..transfer import N_MATRIX_FEATURES
from ..checkpoint import TemporalComponents
from ..provenance import module_signature

CONTRACT='v67_standalone_hierarchical_replacement'

class TransferNet(nn.Module):
    def __init__(self, hidden=16, candidates=16, body_kind='residual5'):
        super().__init__()
        self.hidden=hidden; self.candidates=candidates; self.body_kind=body_kind
        self.body=make_spatial_body(body_kind,N_MATRIX_FEATURES,hidden)
        self.head=nn.Conv2d(hidden,candidates,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        return self.head(self.body(x))

class TinyNet(nn.Module):
    def __init__(self, inputs, outputs, hidden=12):
        super().__init__()
        self.inputs=inputs; self.outputs=outputs; self.hidden=hidden
        self.register_buffer('mean',torch.zeros(inputs))
        self.register_buffer('scale',torch.ones(inputs))
        self.fc1=nn.Linear(inputs,hidden); self.fc2=nn.Linear(hidden,outputs)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
    def forward(self,x):
        return self.fc2(torch.tanh(self.fc1((x-self.mean)/self.scale.clamp_min(1e-4))))
    def normalize_on(self,x):
        with torch.no_grad():
            self.mean.copy_(x.mean(0)); self.scale.copy_(x.std(0,unbiased=False).clamp_min(1e-3))
    def compile(self):
        return CompiledTiny(*[v.detach().cpu().float().numpy().copy() for v in
            (self.mean,self.scale,self.fc1.weight,self.fc1.bias,self.fc2.weight,self.fc2.bias)])

@dataclass
class CompiledTiny:
    mean: np.ndarray
    scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    def __call__(self,x):
        z=(np.asarray(x,np.float32)-self.mean)/np.maximum(self.scale,1e-4)
        return np.tanh(z@self.w1.T+self.b1)@self.w2.T+self.b2

# Operator context is 18 in the inherited strategy implementation.
CONTROLLER_INPUTS=28
DETECTOR_INPUTS=8

@dataclass
class Components:
    smoother: NeuralSmootherNet
    transfer: TransferNet
    detector: TinyNet
    controller: TinyNet
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(cls, hidden=16, seed=7):
        torch.manual_seed(seed)
        c=cls(NeuralSmootherNet(hidden=hidden,split_direction_gain=True),
              TransferNet(hidden=hidden),TinyNet(DETECTOR_INPUTS,1),TinyNet(CONTROLLER_INPUTS,4))
        c.metadata={'contract':CONTRACT,'smoother_trained':False,'transfer_trained':False,
                    'detector_trained':False,'controller_trained':False,'certificate':{'validated':False}}
        # An untrained spatial detector defaults off; no invented positive labels.
        with torch.no_grad(): c.detector.fc2.bias.fill_(-2.)
        return c.eval()

    @classmethod
    def migrate_v66(cls,path,hidden=16):
        old=TemporalComponents.load(path)
        c=cls.create(hidden=old.smoother.head.in_channels)
        c.smoother=old.smoother.cpu().float()
        c.metadata.update(migration_source=Path(path).name,
            migration_retained=['operator_only_smoother'],
            migration_discarded=['exact_K_policy','old_certificates'],
            transfer_initialization='new_zero_delta_classical_P')
        return c

    def modules(self): return (self.smoother,self.transfer,self.detector,self.controller)
    def eval(self):
        for m in self.modules(): m.eval()
        return self
    def generation_signature(self):
        from .research_runtime import expert_signature
        return expert_signature(self.smoother)+expert_signature(self.transfer)
    def expert_signature(self,name):
        from .research_runtime import expert_signature
        return expert_signature(getattr(self,name))
    def component_signatures(self):
        """Signatures of actual component state, independent of metadata."""
        return {name: self.expert_signature(name)
                for name in ('smoother','transfer','detector','controller')}
    def mark_policy_stale(self, reason='expert_weights_changed_requires_policy_refit'):
        """Retain old policy weights for inspection, never certify new experts."""
        self.metadata['policy_stale']=True
        self.metadata['policy_stale_reason']=reason
        self.metadata['certificate']={'validated':False,'reason':reason}
        self.metadata['final_time_improvement_certified']=False
        return self
    def signature(self):
        return self.generation_signature()+module_signature(self.detector)+module_signature(self.controller)
    def save(self,path,extra=None):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        s=self.smoother
        from .research_runtime import research_spec
        rs,rp=research_spec(s),research_spec(self.transfer)
        payload={'format_version':14 if rs or rp else 13,'contract':CONTRACT,'metadata':self.metadata,
            'smoother_spec':rs or dict(hidden=s.head.in_channels,basis_count=s.basis_count,
                coefficient_scale=s.coefficient_scale,split_direction_gain=s.split_direction_gain,
                direction_epsilon=s.direction_epsilon,body_kind=s.body_kind),
            'transfer_spec':rp or dict(hidden=self.transfer.hidden,candidates=self.transfer.candidates,body_kind=self.transfer.body_kind),
            'detector_spec':dict(inputs=self.detector.inputs,outputs=self.detector.outputs,hidden=self.detector.hidden),
            'controller_spec':dict(inputs=self.controller.inputs,outputs=self.controller.outputs,hidden=self.controller.hidden),
            'states':{n:{k:v.detach().cpu() for k,v in getattr(self,n).state_dict().items()}
                      for n in ('smoother','transfer','detector','controller')},'extra':extra or {}}
        tmp=path.with_name(path.name+'.tmp')
        torch.save(payload,tmp); os.replace(tmp,path)
    @classmethod
    def load(cls,path):
        # weights_only prevents executing arbitrary Python objects from checkpoints.
        p=torch.load(path,map_location='cpu',weights_only=True)
        if p.get('contract')!=CONTRACT: raise ValueError('not v6.7; use explicit migrate_v66')
        from .research_runtime import restore_expert
        ss,ps=p['smoother_spec'],p['transfer_spec']
        c=cls(restore_expert(ss) if 'family' in ss else NeuralSmootherNet(**ss),
              restore_expert(ps) if 'family' in ps else TransferNet(**ps),
              TinyNet(**p['detector_spec']),TinyNet(**p['controller_spec']),p['metadata'])
        for n,s in p['states'].items(): getattr(c,n).load_state_dict(s,strict=True)
        return c.eval()
