"""Optional early-prefix smoother pretraining, separate from measured switch training.

The policy's FINAL objective is measured T + instability penalty. This optional
stage only supplies an initialization to that black-box timing objective; its
residual surrogate is not reported as a measured time improvement.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .hierarchy import WorkStats, build_fixed_hierarchy, neural_level_enabled
from .strategy import get_strategy
from .models import NeuralSmootherNet
from .torch_ops import make_torch_problem, scipy_to_torch_sparse, apply_a, classical_correction


@dataclass
class TLevel:
    problem: object
    strategy: object
    p: object = None
    coarse: object = None
    directions: object = None
    gains: object = None


def prepare_torch_hierarchy(a, shape, smoother, config, *, device='cpu', dtype=torch.float64):
    base=build_fixed_hierarchy(a,shape,get_strategy(config.strategy_name),config,WorkStats())
    def convert(level):
        t=TLevel(make_torch_problem(level.a,level.shape,device=device,dtype=dtype),level.strategy)
        if level.coarse is not None:
            t.p=scipy_to_torch_sparse(level.p,device=device,dtype=dtype)
            t.coarse=convert(level.coarse)
            if neural_level_enabled(level,config):
                t.directions,t.gains=smoother.direction_and_gain(t.problem.features.to(dtype=next(smoother.parameters()).dtype))
        return t
    return convert(base)


def _spmv(a,x): return torch.sparse.mm(a,x.reshape(-1,1)).reshape(-1)


def torch_cycle(root,x,b,config,*,neural=False):
    p=root.problem
    if root.coarse is None: return torch.linalg.solve(p.a.to_dense(),b)
    def step(value,reverse):
        residual=b-apply_a(p,value)
        delta=classical_correction(p,residual,smoother=root.strategy.smoother,omega=config.jacobi_omega,
            chebyshev_degree=config.chebyshev_degree,chebyshev_lower_fraction=config.chebyshev_lower_fraction,reverse=reverse)
        value=value+delta
        if neural and root.directions is not None:
            residual=b-apply_a(p,value)
            inp=(residual/p.diag.abs().clamp_min(1e-14)).reshape(1,1,*p.shape)
            dirs=root.directions.to(dtype=b.dtype); gains=root.gains.to(dtype=b.dtype)
            d=NeuralSmootherNet.apply_coefficients(dirs,inp)[0,0].reshape(-1)
            value=value+config.smoother_gain_multiplier*gains[0,0]*d
        return value
    for _ in range(config.pre_steps): x=step(x,False)
    r=b-apply_a(p,x); rc=_spmv(root.p.transpose(0,1),r)
    ec=torch_cycle(root.coarse,torch.zeros_like(rc),rc,config,neural=neural)
    x=x+_spmv(root.p,ec)
    for _ in range(config.post_steps): x=step(x,True)
    return x


def prefix_rollout(root,b,config,*,k_nn=2,tail_cycles=2,x0=None):
    if k_nn<1 or tail_cycles<1: raise ValueError('pretraining must include a neural prefix and a classical tail')
    x=torch.zeros_like(b) if x0 is None else x0
    r0=torch.linalg.vector_norm(b-apply_a(root.problem,x)).clamp_min(1e-100)
    history=[]
    for i in range(k_nn+tail_cycles):
        x=torch_cycle(root,x,b,config,neural=i<k_nn)
        history.append(torch.linalg.vector_norm(b-apply_a(root.problem,x))/r0)
    weights=torch.arange(1,len(history)+1,device=b.device,dtype=b.dtype)
    weights=weights/weights.sum()
    loss=(weights*torch.log(torch.stack(history).clamp_min(config.tolerance))).sum()
    return loss,x,history


def pretrain_smoother(components,examples,config,*,epochs=1,learning_rate=1e-4,k_nn=2,tail_cycles=2,seed=7,device='cpu'):
    import numpy as np
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    m=components.smoother.to(device)
    optimizer=torch.optim.Adam(m.parameters(),lr=learning_rate)
    records=[]
    for epoch in range(1,epochs+1):
        losses=[]; grads=[]; skipped=0
        for index in rng.permutation(len(examples)):
            e=examples[index]; optimizer.zero_grad(set_to_none=True)
            root=prepare_torch_hierarchy(e.a,(e.n,e.n),m,config,device=device)
            b=torch.tensor(e.b,device=device,dtype=torch.float64)
            loss,_,_=prefix_rollout(root,b,config,k_nn=k_nn,tail_cycles=tail_cycles)
            if not torch.isfinite(loss): skipped+=1; continue
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(m.parameters(),2.)
            if not torch.isfinite(grad): skipped+=1; continue
            optimizer.step(); losses.append(float(loss.detach())); grads.append(float(grad))
        record=dict(epoch=epoch,mean_surrogate_loss=float(np.mean(losses)) if losses else None,
                    mean_gradient_norm=float(np.mean(grads)) if grads else None,skipped=skipped,
                    objective='optional_prefix_plus_classical_tail_residual_surrogate')
        records.append(record); print('[smoother]',record,flush=True)
    m.to('cpu').eval()
    return records
