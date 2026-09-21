"""Replacement V-cycle training graph, shared learned operators across cycles."""
from dataclasses import dataclass
import numpy as np
import torch
from ..grid import terminal,next_shape
from ..transfer import (OFFSETS_9,build_transfer_pattern,baseline_weights,weights_from_deltas_torch,
    coarse_fine_indices,scipy_prolongation_from_weights,matrix_feature_array)
from ..hierarchy import baseline_kwargs
from ..models import NeuralSmootherNet
from ..smoothers import LineSmootherCache,_zebra_order,classical_smoothing_correction
from ..strategy import get_strategy
from .autograd_sparse import SparseTensor
from .banks import selected_level,resolve_device
from .spatial import features_from_statistics,expand_mask
from .native import block_statistics

@dataclass
class TLevel:
    a: SparseTensor
    shape: tuple
    index: int
    strategy: object
    p: SparseTensor=None
    coarse: object=None
    dirs: object=None
    gains: object=None
    mask: object=None
    mask_hard: object=None
    raw_scipy: object=None


def _features(a,shape):
    """Matches runtime FP32 features, differentiable through sparse Ac values."""
    nx,ny=shape;diag=a.diagonal().abs().clamp_min(1e-14)
    log=torch.log(diag/torch.quantile(diag,.5).clamp_min(1e-14)+1e-12)
    channels=[log]
    di=a.cols//ny-a.rows//ny;dj=a.cols%ny-a.rows%ny
    for oi,oj in OFFSETS_9:
        mask=(di==oi)&(dj==oj);rr=a.rows[mask]
        vals=a.values[torch.tensor(mask)]/diag[torch.tensor(rr)]
        channels.append(torch.zeros(nx*ny,dtype=a.values.dtype).index_add(0,torch.tensor(rr),vals))
    return torch.stack(channels).reshape(1,10,nx,ny).float()


def make_graph(a,shape,components,cfg,*,learned=True):
    device=resolve_device(cfg,cells=shape[0]*shape[1]); dtype=torch.float32 if cfg.inference_dtype=='float32' else torch.float64
    for net in (components.smoother,components.transfer):net.to(device=device,dtype=dtype)
    strategy=get_strategy(cfg.mg.strategy_name)
    def build(at,sh,index):
        raw=at.numpy();level=TLevel(at,sh,index,strategy,raw_scipy=raw)
        if terminal(sh,cfg.mg.coarsest_n):return level
        cs=next_shape(sh,strategy.coarsening,cfg.mg.coarsest_n,level_index=index)
        pattern=build_transfer_pattern(sh,cs)
        # Classical support/drop selection is frozen inside this differentiable
        # surrogate; it is recomputed from the actual Ac on every forward pass.
        base=baseline_weights(raw,sh,strategy.transfer,coarse=cs,**baseline_kwargs(cfg.mg))
        enabled=learned and selected_level(index,cfg)
        if enabled and cfg.use_transfer and hasattr(components.transfer,'complexity_caps'):
            from .research_transfer import transfer_pattern_for_model
            pattern,base=transfer_pattern_for_model(pattern,components.transfer,base)
        level.pattern=pattern;level.base_weights=base
        feat=_features(at,sh).to(device=device,dtype=dtype)
        if enabled and cfg.use_transfer:
            d=(components.transfer.forward_graph(at,pattern,base)
               if hasattr(components.transfer,'forward_graph') else components.transfer(feat))
            bt=torch.tensor(base,device=device,dtype=dtype)
            if hasattr(components.transfer,'complexity_caps'):
                from .research_transfer import project_transfer_weights
                w=project_transfer_weights(components.transfer,pattern,d,base)
            else:w=weights_from_deltas_torch(pattern,d,bt)
            values=torch.tensor(base,dtype=torch.float64)+(w-bt).to('cpu',dtype=torch.float64)
            valid=torch.tensor(pattern.columns>=0)
            if hasattr(components.transfer,'complexity_caps'):
                support=torch.tensor(base!=0) if getattr(components.transfer,'support','standard')=='support_preserving' else (w.detach().to('cpu')!=0)
                valid=valid & support
                masked=torch.where(valid,values,torch.zeros_like(values))
                # Exact sparse numerical forward, preserving the projection's
                # STE to candidate edges currently pruned/zero in the bank.
                values=values+(masked-values).detach()
            target=torch.tensor(base.sum(1),dtype=torch.float64)
            values=values+(target-values.sum(1))[:,None]*valid/valid.sum(1).clamp_min(1)[:,None]
            inject=np.zeros(base.shape[0],bool);inject[coarse_fine_indices(pattern)]=True
            values=torch.where(torch.tensor(inject)[:,None],torch.tensor(base),values)
            level.interpolation_weights=values
            cols=pattern.columns.ravel();rr=np.repeat(np.arange(base.shape[0]),base.shape[1]);ok=cols>=0
            level.p=SparseTensor.from_coo(rr[ok],cols[ok],values.ravel()[torch.tensor(ok)],(base.shape[0],pattern.n_coarse_unknowns))
        else:
            level.p=SparseTensor.from_scipy(scipy_prolongation_from_weights(pattern,base))
            level.interpolation_weights=torch.tensor(base,dtype=torch.float64)
        ac=level.p.transpose().product(at.product(level.p)).symmetrize()
        if enabled and cfg.use_smoother:
            dirs,gains=components.smoother.direction_and_gain(feat)
            level.dirs=dirs.to('cpu',dtype=torch.float64);level.gains=gains.to('cpu',dtype=torch.float64)*cfg.mg.smoother_gain_multiplier
        level.coarse=build(ac,cs,index+1)
        return level
    return build(SparseTensor.from_scipy(a),shape,0)


def _lines(level,direction):
    key=('lines',direction)
    if key in level.a.cache:return level.a.cache[key]
    cache=LineSmootherCache(level.raw_scipy,level.shape)
    # Construct graph-preserving submatrices once, reused in all unrolled cycles.
    blocks=[]
    for ix in cache._line_sets(direction):
        blocks.append((ix,level.a.slice(ix,ix),level.a.slice(np.arange(level.a.shape[0]),ix)))
    level.a.cache[key]=blocks
    return blocks


def neural_delta(level,r):
    inp=(r/level.a.diagonal().abs().clamp_min(1e-14)).reshape(1,1,*level.shape)
    return level.gains[0,0]*NeuralSmootherNet.apply_coefficients(level.dirs,inp)[0,0].ravel()


def smooth(level,r,cfg,reverse=False,gate=None):
    kind=level.strategy.smoother;has=gate is not None and level.dirs is not None
    hard_mask=level.mask_hard if level.mask_hard is not None else (gate.detach()>=cfg.gate_on if has else None)
    if has and bool(hard_mask.all()):
        dn=neural_delta(level,r)
        if not cfg.spatial or cfg.gate_mode=='open':return dn
        dc=smooth(level,r,cfg,reverse,None)
        sp=(1-gate.mean())*dc+gate.mean()*dn
        return sp+(dn-sp).detach()
    diag=level.a.diagonal().abs().clamp_min(1e-14)
    if kind=='jacobi':
        dc=cfg.mg.jacobi_omega*r/diag
        if not has:return dc
        soft=gate;hard=hard_mask.double();dn=neural_delta(level,r)
        # Product STE: closed forward branches still train all participating heads.
        sp=(1-soft)*dc+soft*dn;hp=(1-hard)*dc+hard*dn
        return sp+(hp-sp).detach()
    if kind=='chebyshev':
        row=level.a.rows;rs=torch.zeros(r.numel(),dtype=r.dtype).index_add(0,torch.tensor(row),level.a.values.abs())
        lm=(1.05*torch.max(rs/diag)).clamp_min(1e-6).detach();lo=cfg.mg.chebyshev_lower_fraction*lm
        center=(lm+lo)/2;rad=(lm-lo)/2
        roots=sorted([center-rad*np.cos((2*k-1)*np.pi/(2*cfg.mg.chebyshev_degree)) for k in range(1,cfg.mg.chebyshev_degree+1)],key=float,reverse=True)
        d=torch.zeros_like(r);cur=r
        for root in roots:
            inc=cur/(diag*root);d=d+inc;cur=cur-level.a.apply(inc)
        return d
    dirs={'line_x':('x',),'line_y':('y',),'line_diag45':('diag45',),
          'line_alt':('y','x') if reverse else ('x','y')}[kind]
    d=torch.zeros_like(r);cur=r
    for direction in dirs:
        blocks=_lines(level,direction)
        for i in _zebra_order(len(blocks),reverse):
            ix,block,columns=blocks[i];idx=torch.tensor(ix)
            ci=block.solve(cur[idx])
            if has:
                # Actual deployment replaces only whole selected lines.
                hard=hard_mask[idx].all().double();soft=gate[idx].mean()
                ni=neural_delta(level,cur)[idx]
                sp=(1-soft)*ci+soft*ni;hp=(1-hard)*ci+hard*ni
                ci=sp+(hp-sp).detach()
            d=d.index_add(0,idx,ci);cur=cur-columns.apply(ci)
    return d


def graph_gate(level,r,components,cfg):
    if not cfg.spatial or cfg.gate_mode=='open':
        level.mask_hard=torch.ones_like(r,dtype=torch.bool);return torch.ones_like(r)
    if cfg.gate_mode=='closed':
        level.mask_hard=torch.zeros_like(r,dtype=torch.bool);return torch.zeros_like(r)
    rd=r.detach().numpy();scale=max(np.max(np.abs(rd),initial=0),1e-300)
    _,raw,blocks=block_statistics(rd/scale,level.shape,cfg.detector_max_side)
    f=features_from_statistics(raw,level.shape,blocks)
    net=components.detector.cpu().float();p=torch.sigmoid(net(torch.tensor(f)).ravel())
    nx,ny=level.shape;bx,by=blocks
    ids=(np.arange(nx)[:,None]*bx//nx)*by+np.arange(ny)[None,:]*by//ny
    soft=p[torch.tensor(ids.ravel())].double()
    prior=level.mask_hard if level.mask_hard is not None else torch.zeros_like(r,dtype=torch.bool)
    hard=torch.where(prior,soft.detach()>=cfg.gate_off,soft.detach()>=cfg.gate_on)
    if float(hard.double().mean())<cfg.active_skip:hard=torch.zeros_like(hard)
    level.mask_hard=hard
    return soft


def cycle(level,x,b,components,cfg,step=0):
    if level.coarse is None:return level.a.solve(b)
    gate=None
    if level.dirs is not None:
        if level.mask is None or step%cfg.gate_refresh==0:
            level.mask=graph_gate(level,b-level.a.apply(x),components,cfg)
        gate=level.mask
    for j in range(cfg.mg.pre_steps):
        r=b-level.a.apply(x);x=x+smooth(level,r,cfg,False,gate if j<cfg.replace_pre else None)
    r=b-level.a.apply(x);rc=level.p.transpose().apply(r)
    ec=cycle(level.coarse,torch.zeros_like(rc),rc,components,cfg,step)
    x=x+level.p.apply(ec)
    for j in range(cfg.mg.post_steps):
        r=b-level.a.apply(x);x=x+smooth(level,r,cfg,True,gate if j<cfg.replace_post else None)
    return x


def rollout(example,components,cfg,prefix=2,tail=2,compute_ratio=1.,lambda_compute=.02,hierarchical=False,schedule=None):
    """Shared operators in a differentiable multi-cycle prefix + classical tail.

    The time proxy is constant for a fixed action schedule. Gate-dependent
    expected work is differentiable only during hierarchical mask fine-tuning.
    No gradient of perf_counter() is claimed.
    """
    from dataclasses import replace
    # Operator stages deliberately expose learned components. Spatial mask
    # learning is enabled only in the final stage and validated separately.
    traincfg=cfg if hierarchical else replace(cfg,spatial=False,gate_mode='open')
    learned=make_graph(example.a,(example.n,example.n),components,traincfg,learned=True)
    classical=make_graph(example.a,(example.n,example.n),components,traincfg,learned=False)
    actions=list(schedule) if schedule is not None else ['H']*prefix+['C']*tail
    if not actions or any(a not in {'C','H'} for a in actions):raise ValueError('invalid training action schedule')
    b=torch.tensor(example.b,dtype=torch.float64);x=torch.zeros_like(b)
    r0=torch.linalg.vector_norm(b).clamp_min(1e-100);history=[]
    for k,action in enumerate(actions):
        root=learned if action=='H' else classical
        oldx=x
        x=cycle(root,x,b,components,traincfg,k)
        if hierarchical and action=='H' and learned.dirs is not None and learned.mask_hard is not None and not bool(learned.mask_hard.any()):
            # Runtime's all-gates-closed route resets BOTH S and P to classical.
            xc=cycle(classical,oldx,b,components,traincfg,k)
            probability=learned.mask.mean()
            surrogate=(1-probability)*xc+probability*x
            x=surrogate+(xc-surrogate).detach()
        history.append(torch.linalg.vector_norm(b-learned.a.apply(x))/r0)
    weights=torch.arange(1,len(history)+1,dtype=torch.float64);weights/=weights.sum()
    residual_loss=(weights*torch.log(torch.stack(history).clamp_min(cfg.mg.tolerance))).sum()
    n_h=actions.count('H');n_c=actions.count('C')
    cost=torch.tensor((n_h*compute_ratio+n_c)/len(actions),dtype=torch.float64)
    if hierarchical:
        gate_terms=[];level=learned
        while level.coarse is not None:
            if level.mask is not None:gate_terms.append(level.mask.mean())
            level=level.coarse
        if gate_terms:cost=1.+(compute_ratio-1.)*torch.stack(gate_terms).mean()*n_h/len(actions)
    return residual_loss+lambda_compute*cost,dict(residual=residual_loss,cost=cost,history=history,actions=actions),x
