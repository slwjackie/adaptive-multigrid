"""One immutable classical hierarchy shared by neural prefix and classical tail."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from .config import MGConfig
from .grid import GridShape, terminal, next_shape
from .strategy import MGStrategy
from .smoothers import LineSmootherCache, classical_smoothing_correction
from .transfer import (build_transfer_pattern, baseline_weights, scipy_prolongation_from_weights,
                       galerkin_coarse_operator, matrix_feature_array, OFFSETS_9)
from .native_stencil import NativeStencil, native_available

@dataclass
class WorkStats:
    base_hierarchy_builds: int = 0
    coarse_factorizations: int = 0
    line_factorizations: int = 0
    coarse_l_nnz: int = 0
    coarse_u_nnz: int = 0
    hierarchy_setup_seconds: float = 0.0
    nn_setup_seconds: float = 0.0
    neural_setup_calls: int = 0
    smoother_nn_calls: int = 0
    classical_smooth_seconds: float = 0.0
    neural_apply_seconds: float = 0.0
    transfer_seconds: float = 0.0
    coarse_solve_seconds: float = 0.0
    classical_smooth_calls: int = 0
    neural_apply_calls: int = 0
    native_stencil_calls: int = 0
    coarse_solves: int = 0
    level_visits: int = 0
    matvecs: int = 0
    transfer_applies: int = 0
    cache_hits: int = 0
    work_flops: float = 0.0
    attempted_neural_cycles: int = 0
    accepted_neural_cycles: int = 0
    classical_cycles: int = 0
    rejected_neural_cycles: int = 0
    classical_recovery_cycles: int = 0
    predictor_calls: int = 0
    predictor_seconds: float = 0.0
    predictor_feature_seconds: float = 0.0
    freshness_checks: int = 0
    freshness_seconds: float = 0.0
    # Zero-valued contract counters, no corresponding runtime components.
    detector_calls: int = 0
    spatial_gate_calls: int = 0
    residual_hierarchy_rebuilds: int = 0
    switch_hierarchy_builds: int = 0

    def add_sparse_flops(self, flops, *, safety=False):
        self.work_flops += float(flops)

    def to_dict(self):
        return asdict(self)

@dataclass
class FixedLevel:
    a: sp.csr_matrix
    shape: GridShape
    index: int
    strategy: MGStrategy
    diagonal: np.ndarray
    p: sp.csr_matrix | None = None
    r: sp.csr_matrix | None = None
    coarse: "FixedLevel | None" = None
    lu: object | None = None
    cache: LineSmootherCache | None = None
    pattern: object | None = None
    base_weights: np.ndarray | None = None
    features: np.ndarray | None = None
    neural_stencil: object | None = None

@dataclass
class StencilBank:
    csr: sp.csr_matrix
    native: NativeStencil | None = None

    def apply(self, r, stats):
        start = perf_counter()
        if self.native is not None:
            out = self.native.apply(r)[:,0]
            stats.native_stencil_calls += 1
        else:
            out = np.asarray(self.csr @ r).reshape(-1)
        stats.neural_apply_calls += 1
        stats.work_flops += 2*self.csr.nnz
        stats.neural_apply_seconds += perf_counter()-start
        return out

def baseline_kwargs(config):
    return dict(operator_sweeps=config.operator_sweeps,operator_omega=config.operator_omega,
                operator_blend=config.operator_blend,operator_drop_tolerance=config.operator_drop_tolerance,
                operator_candidate_topk=config.operator_candidate_topk)


def build_fixed_hierarchy(a, shape, strategy, config, stats, *, index=0, auxiliary=False):
    start=perf_counter()
    a=a.tocsr().astype(np.float64,copy=True);a.sum_duplicates();a.sort_indices()
    level=FixedLevel(a,shape,index,strategy,np.maximum(np.abs(a.diagonal()),1e-14))
    if terminal(shape,config.coarsest_n):
        level.lu=spla.splu(a.tocsc())
        stats.coarse_factorizations+=1
        stats.coarse_l_nnz+=level.lu.L.nnz;stats.coarse_u_nnz+=level.lu.U.nnz
    else:
        coarse_shape=next_shape(shape,strategy.coarsening,config.coarsest_n,level_index=index)
        level.pattern=build_transfer_pattern(shape,coarse_shape)
        level.base_weights=baseline_weights(a,shape,strategy.transfer,coarse=coarse_shape,**baseline_kwargs(config))
        level.p=scipy_prolongation_from_weights(level.pattern,level.base_weights)
        level.r=level.p.T.tocsr()
        level.cache=LineSmootherCache(a,shape)
        directions={"line_x":("x",),"line_y":("y",),"line_alt":("x","y"),"line_diag45":("diag45",)}.get(strategy.smoother,())
        for direction in directions:
            level.cache.get(direction)
        stats.line_factorizations+=level.cache.factorization_count
        level.coarse=build_fixed_hierarchy(galerkin_coarse_operator(a,level.p),coarse_shape,strategy,config,stats,index=index+1,auxiliary=auxiliary)
    # Count entire banks only at the caller; per-level times would double count.
    if index==0:
        stats.base_hierarchy_builds+=1
        stats.hierarchy_setup_seconds+=perf_counter()-start
    return level


def matvec(a, x, stats):
    stats.matvecs+=1;stats.work_flops+=2*a.nnz
    return np.asarray(a@x).reshape(-1)


def apply_transfer(p, x, stats):
    start=perf_counter();out=np.asarray(p@x).reshape(-1)
    stats.transfer_seconds+=perf_counter()-start
    stats.transfer_applies+=1;stats.work_flops+=2*p.nnz
    return out


def classical_step(level, residual, config, stats, *, reverse):
    start=perf_counter()
    out=classical_smoothing_correction(level.a,level.shape,residual,level.strategy.smoother,
        omega=config.jacobi_omega,chebyshev_degree=config.chebyshev_degree,
        chebyshev_lower_fraction=config.chebyshev_lower_fraction,cache=level.cache,
        reverse=reverse,work=stats)
    stats.classical_smooth_calls+=1
    stats.classical_smooth_seconds+=perf_counter()-start
    return out


def ensure_features(level):
    if level.features is None:
        level.features=matrix_feature_array(level.a,level.shape)
    return level.features


def stencil_from_coefficients(values, gains, diagonal, shape):
    nx,ny=shape; rows=[];cols=[];data=[]
    source=np.arange(nx*ny).reshape(shape)
    for k,(di,dj) in enumerate(OFFSETS_9):
        i0,i1=max(0,-di),min(nx,nx-di);j0,j1=max(0,-dj),min(ny,ny-dj)
        rr=source[i0:i1,j0:j1].ravel();cc=source[i0+di:i1+di,j0+dj:j1+dj].ravel()
        rows.append(rr);cols.append(cc)
        data.append(values[k,i0:i1,j0:j1].ravel()*gains/diagonal[cc])
    p=sp.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(nx*ny,nx*ny)).tocsr()
    p.eliminate_zeros()
    return p


def ensure_stencil(level, smoother, config, stats):
    if level.neural_stencil is not None:
        stats.cache_hits+=1;return level.neural_stencil
    start=perf_counter();module=smoother
    dtype=next(module.parameters()).dtype
    features=torch.as_tensor(ensure_features(level),dtype=dtype).unsqueeze(0)
    with torch.inference_mode():
        directions,gains=module.direction_and_gain(features)
    d=directions[0,:1].detach().double().numpy()
    g=gains[0,:1].detach().double().numpy()*config.smoother_gain_multiplier
    csr=stencil_from_coefficients(d[0],float(g[0]),level.diagonal,level.shape)
    native=None
    use=config.stencil_backend=="native" or (config.stencil_backend=="auto" and level.a.shape[0]>=config.native_min_cells and native_available())
    if use:
        native=NativeStencil.from_directions(d,diagonal=level.diagonal,gains=g,threads=config.native_threads,parallel_min=config.native_parallel_min)
    level.neural_stencil=StencilBank(csr,native)
    stats.neural_setup_calls+=1;stats.smoother_nn_calls+=1
    stats.nn_setup_seconds+=perf_counter()-start
    return level.neural_stencil


def classical_cycle(level, x, b, config, stats):
    """One standalone V-cycle; this function is also the exact gate-off path."""
    stats.level_visits+=1
    if level.lu is not None:
        start=perf_counter();out=np.asarray(level.lu.solve(b)).reshape(-1)
        stats.coarse_solves+=1;stats.coarse_solve_seconds+=perf_counter()-start
        stats.work_flops+=2*(level.lu.L.nnz+level.lu.U.nnz)
        return out
    for _ in range(config.pre_steps):
        r=b-matvec(level.a,x,stats)
        x=x+classical_step(level,r,config,stats,reverse=False)
    r=b-matvec(level.a,x,stats)
    rc=apply_transfer(level.r,r,stats)
    ec=classical_cycle(level.coarse,np.zeros_like(rc),rc,config,stats)
    x=x+apply_transfer(level.p,ec,stats)
    for _ in range(config.post_steps):
        r=b-matvec(level.a,x,stats)
        x=x+classical_step(level,r,config,stats,reverse=True)
    return x



def neural_level_enabled(level, config):
    return (level.coarse is not None and config.smoother_gain_multiplier > 0
            and (config.nn_levels == -1 or level.index < config.nn_levels))


def neural_cycle(level, x, b, smoother, config, stats):
    """Classical MG plus full-grid NN increments on the SAME hierarchy.

    At each configured smoothing step: x_c=x+B_c r, then
    x_new=x_c+g*C*(D^-1*(b-A*x_c)). No extra learned coarse hierarchy,
    no patch masks, no per-cycle CNN inference after bank preparation.
    """
    if not neural_level_enabled(level, config):
        return classical_cycle(level, x, b, config, stats)
    stats.level_visits += 1
    bank = ensure_stencil(level, smoother, config, stats)

    def smooth(value, reverse):
        residual = b-matvec(level.a, value, stats)
        value = value + classical_step(level, residual, config, stats, reverse=reverse)
        residual = b-matvec(level.a, value, stats)
        return value + bank.apply(residual, stats)

    for _ in range(config.pre_steps): x = smooth(x, False)
    residual = b-matvec(level.a, x, stats)
    rc = apply_transfer(level.r, residual, stats)
    ec = neural_cycle(level.coarse, np.zeros_like(rc), rc, smoother, config, stats)
    x = x + apply_transfer(level.p, ec, stats)
    for _ in range(config.post_steps): x = smooth(x, True)
    return x


def clear_neural_banks(root):
    while root is not None:
        root.neural_stencil = None
        root = root.coarse


def hierarchy_identity(root):
    """Cheap identity tuple for regression tests; no matrix hashing in a cycle."""
    result=[]
    while root is not None:
        result.append((id(root), id(root.a), id(root.p), id(root.r), id(root.lu)))
        root=root.coarse
    return tuple(result)
