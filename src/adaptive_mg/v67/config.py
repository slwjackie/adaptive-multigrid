"""v6.7 deployment contract. Old exact-K policy is deliberately not reused."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from ..config import MGConfig

@dataclass(frozen=True)
class AdaptiveConfig:
    mg: MGConfig = field(default_factory=lambda: MGConfig(mode='classical', pre_steps=2, post_steps=2))
    mode: str = 'production'  # production (certified), research, hybrid, classical
    branch: str = 'auto'  # explicit branches force temporal entry, retaining safety
    use_smoother: bool = True
    use_transfer: bool = True
    application: str = 'replace'  # additive is ablation only
    replace_pre: int = 1
    replace_post: int = 0
    auto_device_min_cells: int = 4096  # conservative starting value, tune per hardware
    inference_device: str = 'cpu'  # cpu, mps, cuda, auto
    inference_dtype: str = 'float32'
    lazy: bool = True
    spatial: bool = True
    gate_mode: str = 'learned'  # learned / open / closed, latter two for contracts
    gate_on: float = .65
    gate_off: float = .45
    gate_refresh: int = 4
    detector_max_side: int = 16
    detector_features: str = 'current'  # current, restriction_lagged
    active_skip: float = .05
    active_dense: float = .50
    min_dwell: int = 2
    hybrid_horizon: int = 4
    max_hybrid_entries: int = 4
    break_even: bool = True
    win_margin: float = .05  # relative TIME reduction, not speedup percentage
    exit_efficiency_ratio: float = 1.0
    underperformance_patience: int = 2
    cycle_budget_ratio: float = 1.15  # target only; measured, never assumed
    use_learned_controller: bool = True
    record_trace: bool = True

    def __post_init__(self):
        if self.branch not in {'auto','C','H_S','H_P','H_SP'}:
            raise ValueError('branch must be auto/C/H_S/H_P/H_SP')
        if self.mode not in {'production','research','hybrid','burst','classical'}:
            raise ValueError('unknown v6.7 mode')
        if self.application not in {'replace','additive'}:
            raise ValueError('application must be replace/additive')
        if self.mode == 'production' and self.application != 'replace':
            raise ValueError('additive is diagnostic only')
        if self.gate_mode not in {'learned','open','closed'}:
            raise ValueError('invalid gate_mode')
        if self.inference_dtype not in {'float32','float64'}:
            raise ValueError('NN dtype must be float32 or float64')
        if isinstance(self.auto_device_min_cells,bool) or not isinstance(self.auto_device_min_cells,int) or self.auto_device_min_cells<1:
            raise ValueError('auto_device_min_cells must be positive integer')
        if self.inference_device not in {'cpu','cuda','mps','auto'}:
            raise ValueError('invalid inference_device')
        if self.inference_device == 'mps' and self.inference_dtype != 'float32':
            raise ValueError('MPS operator generation must use FP32; solver is CPU FP64')
        if not 0 <= self.replace_pre <= self.mg.pre_steps or not 0 <= self.replace_post <= self.mg.post_steps:
            raise ValueError('cannot replace more sweeps than classical performs')
        if not 0 <= self.gate_off <= self.gate_on <= 1:
            raise ValueError('invalid hysteresis thresholds')
        if not 0 <= self.active_skip < self.active_dense <= 1 or not 0 <= self.win_margin < 1:
            raise ValueError('invalid active fraction / margin')
        if min(self.gate_refresh,self.detector_max_side,self.min_dwell,self.hybrid_horizon,
               self.max_hybrid_entries,self.underperformance_patience) < 1:
            raise ValueError('counts must be positive')
        if self.detector_features not in {'current','restriction_lagged'}:
            raise ValueError('invalid feature source')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        d = dict(value)
        if isinstance(d.get('mg'), dict): d['mg'] = MGConfig(**d['mg'])
        return cls(**d)

    def certification_scope(self):
        d=self.to_dict()
        for key in ('mode','record_trace'): d.pop(key)
        # Preserve the exact scope of existing combined-policy certificates.
        if self.branch == 'auto': d.pop('branch')
        d['mg'].pop('verbose',None)
        return d
