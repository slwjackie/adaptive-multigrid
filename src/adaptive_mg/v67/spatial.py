"""Small block detector. A closed smoother mask alone does NOT reset learned P."""
from time import perf_counter
import numpy as np
from .native import block_statistics


def features_from_statistics(raw,shape,blocks):
    count=np.maximum(raw[:,5],1)
    mean_energy=max(float(raw[:,0].sum()/count.sum()),1e-300)
    e=raw[:,0]/count
    bx,by=blocks
    xx,yy=np.meshgrid(np.linspace(0,1,bx),np.linspace(0,1,by),indexing='ij')
    return np.column_stack((np.log(np.maximum(e/mean_energy,1e-12)),
        raw[:,1]/count/np.sqrt(mean_energy),raw[:,2]/np.sqrt(mean_energy),
        raw[:,3]/count/mean_energy,raw[:,4]/count/mean_energy,
        xx.ravel(),yy.ravel(),np.full(bx*by,np.log2(max(shape))))).astype(np.float32)


def expand_mask(mask,shape,blocks):
    nx,ny=shape;bx,by=blocks
    return np.asarray(mask).reshape(bx,by)[np.arange(nx)*bx//nx][:,np.arange(ny)*by//ny].ravel()

class SpatialState:
    def __init__(self,components,cfg,net_factory=None):
        self.cfg=cfg
        self.components=components
        self.net_factory=net_factory
        # No model conversion or inference merely to enter a classical solve.
        self.net=None
        self.masks={};self.last={};self.raw_cache={};self.block_masks={}
        self.last_features={}
    def invalidate(self):
        self.last.clear()
    def gate(self,level,residual,cycle,stats,refresh=False):
        cfg=self.cfg;N=residual.size
        if not cfg.spatial or cfg.gate_mode=='open': return np.ones(N,bool)
        if cfg.gate_mode=='closed': return np.zeros(N,bool)
        key=level.index
        due=key not in self.last or cycle-self.last[key]>=cfg.gate_refresh or refresh
        if not due: return self.masks[key]
        t=perf_counter()
        if cfg.detector_features=='restriction_lagged' and key in self.raw_cache:
            raw,blocks=self.raw_cache[key];stats.lagged_detector_refreshes+=1
        else:
            scale=max(float(np.max(np.abs(residual),initial=0)),1e-300)
            _,raw,blocks=block_statistics(residual/scale,level.shape,cfg.detector_max_side)
        features=features_from_statistics(raw,level.shape,blocks)
        stats.detector_feature_seconds+=perf_counter()-t
        t=perf_counter()
        if self.net is None and self.components is not None:
            self.net=self.net_factory() if self.net_factory is not None else self.components.detector.compile()
        score=1/(1+np.exp(-np.clip(self.net(features)[:,0],-40,40))) if self.net else np.zeros(len(features))
        prior=self.block_masks.get(key,np.zeros(len(score),bool))
        selected=np.where(prior,score>=cfg.gate_off,score>=cfg.gate_on)
        mask=expand_mask(selected,level.shape,blocks)
        if mask.mean()<cfg.active_skip: mask[:]=False;selected[:]=False
        self.last[key]=cycle;self.block_masks[key]=selected;self.masks[key]=mask
        self.last_features[key]=features
        stats.detector_calls+=1;stats.detector_refreshes+=1
        stats.detector_seconds+=perf_counter()-t
        return mask
    def restriction(self,level,residual,stats):
        """Optional one-cycle-lagged features generated in the restriction pass."""
        t=perf_counter()
        # Use actual residual for restriction, normalize statistics afterward.
        rc,raw,blocks=block_statistics(residual,level.shape,self.cfg.detector_max_side,level.p)
        self.raw_cache[level.index]=(raw,blocks)
        stats.fused_restriction_calls+=1;stats.transfer_applies+=1
        stats.work_flops+=2*level.p.nnz
        stats.transfer_seconds+=perf_counter()-t
        return rc
