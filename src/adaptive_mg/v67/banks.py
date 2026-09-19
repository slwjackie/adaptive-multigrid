"""Eager classical / lazy learned operator banks; no residual-dependent Galerkin."""
from dataclasses import dataclass,replace
from copy import copy,deepcopy
from time import perf_counter
import numpy as np
import scipy.sparse.linalg as sla
import torch
from ..hierarchy import (WorkStats,FixedLevel,StencilBank,build_fixed_hierarchy,
    baseline_kwargs,stencil_from_coefficients,matvec,apply_transfer,classical_step,classical_cycle)
from ..smoothers import LineSmootherCache,_zebra_order
from ..grid import terminal,next_shape
from ..transfer import (matrix_feature_array,baseline_weights,build_transfer_pattern,
    weights_from_deltas_torch,scipy_prolongation_from_weights,galerkin_coarse_operator,
    coarse_fine_indices)
from ..native_stencil import NativeStencil,native_available
from .native import apply_rows

@dataclass
class Stats(WorkStats):
    smoother_operator_cache_hits: int = 0
    smoother_bank_builds: int = 0
    transfer_bank_builds: int = 0
    smoother_bank_cache_hits: int = 0
    transfer_bank_cache_hits: int = 0
    classical_bank_cache_hits: int = 0
    learned_transfer_apply_calls: int = 0
    controller_nn_calls: int = 0
    gate_closed_fallback_cycles: int = 0
    classical_bypass_cycles: int = 0
    classical_lock_cycles: int = 0
    branch_C_cycles: int = 0
    branch_H_S_cycles: int = 0
    branch_H_P_cycles: int = 0
    branch_H_SP_cycles: int = 0
    branch_H_S_attempts: int = 0
    branch_H_P_attempts: int = 0
    branch_H_SP_attempts: int = 0
    branch_setup_seconds: float = 0.
    learned_hierarchy_builds: int = 0
    setup_failures: int = 0
    setup_failure_cache_hits: int = 0
    transfer_failure_cache_hits: int = 0
    smoother_failure_cache_hits: int = 0
    learned_factorizations: int = 0
    transfer_nn_calls: int = 0
    cpu_device_transfer_seconds: float = 0.
    operator_generation_seconds: float = 0.
    learned_hierarchy_seconds: float = 0.
    learned_transfer_seconds: float = 0.
    replacement_calls: int = 0
    replaced_line_solves: int = 0
    remaining_line_solves: int = 0
    selected_row_calls: int = 0
    dense_masked_calls: int = 0
    detector_refreshes: int = 0
    detector_seconds: float = 0.
    detector_feature_seconds: float = 0.
    lagged_detector_refreshes: int = 0
    fused_restriction_calls: int = 0
    controller_calls: int = 0
    controller_seconds: float = 0.
    hybrid_entries: int = 0
    trial_cycles: int = 0
    rollback_count: int = 0
    lock_count: int = 0
    safeguard_seconds: float = 0.
    active_rows: int = 0
    possible_rows: int = 0
    coarse_path_count: int = 0



@dataclass(frozen=True)
class GenerationFailure:
    """Compact failed-generation evidence; never retain traceback/hierarchy tensors."""
    error_type: str
    message: str

    @classmethod
    def from_error(cls,error):
        return error.failure if isinstance(error,CachedGenerationFailure) else cls(type(error).__name__,str(error))

    def reject_cached(self,stats,component):
        stats.cache_hits+=1;stats.setup_failure_cache_hits+=1
        name=component+'_failure_cache_hits'
        setattr(stats,name,getattr(stats,name)+1)
        raise CachedGenerationFailure(self,component)


class CachedGenerationFailure(RuntimeError):
    def __init__(self,failure,component):
        self.failure=failure;self.component=component
        super().__init__(f'cached {component} setup failure [{failure.error_type}]: {failure.message}')

def selected_level(index,cfg):
    return cfg.mg.nn_levels==-1 or index<cfg.mg.nn_levels

def resolve_device(cfg):
    d=cfg.inference_device
    if d=='auto': d='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if d=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    if d=='mps' and not torch.backends.mps.is_available(): raise ValueError('MPS unavailable')
    if d=='mps' and cfg.inference_dtype!='float32': raise ValueError('MPS cannot generate FP64 tensors')
    return torch.device(d)

def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize(device)
    elif device.type=='mps': torch.mps.synchronize()


def _inference(module,features,device,dtype,fn,stats):
    from .research_runtime import require_generation_allowed
    require_generation_allowed(module)
    start=perf_counter()
    # Generation must not alter checkpoint weights/dtype or invalidate a policy
    # merely by preparing a bank. Conversion is charged to this setup.
    tensors=tuple(module.parameters())+tuple(module.buffers())
    if any(t.device!=device or (t.is_floating_point() and t.dtype!=dtype) for t in tensors):
        module=deepcopy(module).to(device=device,dtype=dtype)
    module.eval()
    f=torch.as_tensor(features,dtype=dtype,device=device).unsqueeze(0)
    synchronize(device)
    stats.cpu_device_transfer_seconds+=perf_counter()-start
    start=perf_counter()
    try:
        with torch.inference_mode(): values=fn(module,f)
        synchronize(device)
    finally:stats.operator_generation_seconds+=perf_counter()-start
    start=perf_counter()
    def cp(t): return t.detach().cpu().double().numpy()
    result=tuple(cp(v) for v in values) if isinstance(values,tuple) else cp(values)
    synchronize(device)
    stats.cpu_device_transfer_seconds+=perf_counter()-start
    return result


def generated_stencil(level,components,cfg,stats,device,dtype):
    stats.smoother_nn_calls+=1;stats.neural_setup_calls+=1
    d,g=_inference(components.smoother,matrix_feature_array(level.a,level.shape),device,dtype,
                   lambda m,f:m.direction_and_gain(f),stats)
    d=d[0,:1];g=g[0,:1]*cfg.mg.smoother_gain_multiplier
    if not np.isfinite(d).all() or not np.isfinite(g).all(): raise ValueError('nonfinite generated smoother')
    csr=stencil_from_coefficients(d[0],float(g[0]),level.diagonal,level.shape)
    native=None
    if cfg.mg.stencil_backend=='native' or (cfg.mg.stencil_backend=='auto' and level.a.shape[0]>=cfg.mg.native_min_cells and native_available()):
        native=NativeStencil.from_directions(d,diagonal=level.diagonal,gains=g,
            threads=cfg.mg.native_threads,parallel_min=cfg.mg.native_parallel_min)
    return StencilBank(csr,native)


def generated_p(level,components,cfg,stats,device,dtype):
    stats.transfer_nn_calls+=1;stats.neural_setup_calls+=1
    def evaluate(m,f):
        delta=m.forward_graph(level.a,level.pattern,level.base_weights) if hasattr(m,'forward_graph') else m(f)
        base=torch.tensor(level.base_weights,device=device,dtype=dtype)
        if hasattr(m,'complexity_caps'):
            from .research_transfer import project_transfer_weights
            learned=project_transfer_weights(m,level.pattern,delta,base)
        else:learned=weights_from_deltas_torch(level.pattern,delta,base)
        # Cast a learned DELTA back to FP64. Zero head preserves baseline exactly.
        return (learned-base,learned!=0) if hasattr(m,'complexity_caps') else learned-base
    delta=_inference(components.transfer,matrix_feature_array(level.a,level.shape),device,dtype,evaluate,stats)
    support=None
    if isinstance(delta,tuple):delta,support=delta;support=support.astype(bool)
    weights=level.base_weights+delta
    valid=level.pattern.columns>=0
    if support is not None:valid=valid & support
    weights[~valid]=0.
    # Preserve partition sums and injection after the mixed-precision boundary.
    weights+=(level.base_weights.sum(1)-weights.sum(1))[:,None]*valid/np.maximum(valid.sum(1),1)[:,None]
    coarse=coarse_fine_indices(level.pattern);weights[coarse]=level.base_weights[coarse]
    if not np.isfinite(weights).all(): raise ValueError('nonfinite learned P')
    if np.max(np.sum(np.abs(weights),axis=1),initial=0)>8.:
        raise ValueError('learned transfer row magnitude exceeds safety limit')
    return scipy_prolongation_from_weights(level.pattern,weights)


def prepare_transfer_bank(base,components,cfg,stats):
    """Construct P/R/Ac only. Factors are formed from each actual learned Ac."""
    start=perf_counter();gen0=stats.operator_generation_seconds;copy0=stats.cpu_device_transfer_seconds
    device=resolve_device(cfg);dtype=torch.float32 if cfg.inference_dtype=='float32' else torch.float64
    def build(a,shape,index,base_level):
        same=(a is base_level.a) if base_level is not None else False
        level=FixedLevel(a,shape,index,base.strategy,np.maximum(np.abs(a.diagonal()),1e-14))
        level.learned_transfer=False
        if terminal(shape,cfg.mg.coarsest_n):
            if same: level.lu=base_level.lu
            else: level.lu=sla.splu(a.tocsc());stats.learned_factorizations+=1
            return level
        shape_c=next_shape(shape,base.strategy.coarsening,cfg.mg.coarsest_n,level_index=index)
        level.pattern=base_level.pattern if same else build_transfer_pattern(shape,shape_c)
        level.base_weights=base_level.base_weights if same else baseline_weights(a,shape,base.strategy.transfer,coarse=shape_c,**baseline_kwargs(cfg.mg))
        classical_p=scipy_prolongation_from_weights(level.pattern,level.base_weights) if hasattr(components.transfer,'complexity_caps') else None
        level.cache=base_level.cache if same else LineSmootherCache(a,shape)
        if not same:
            for direction in {'line_x':('x',),'line_y':('y',),'line_alt':('x','y'),'line_diag45':('diag45',)}.get(base.strategy.smoother,()):
                level.cache.get(direction)
            stats.learned_factorizations+=level.cache.factorization_count
        if selected_level(index,cfg):
            if hasattr(components.transfer,'complexity_caps'):
                from .research_transfer import transfer_pattern_for_model
                level.pattern,level.base_weights=transfer_pattern_for_model(level.pattern,components.transfer,level.base_weights)
            level.p=generated_p(level,components,cfg,stats,device,dtype)
            level.learned_transfer=True
            ac=galerkin_coarse_operator(a,level.p)
            if hasattr(components.transfer,'complexity_caps'):
                from .research_transfer import enforce_transfer_complexity
                level.transfer_complexity_report=enforce_transfer_complexity(a,level.p,ac,
                    baseline_p=classical_p,baseline_ac=galerkin_coarse_operator(a,classical_p),
                    caps=components.transfer.complexity_caps)
        elif same:
            level.p=base_level.p;ac=base_level.coarse.a
        else:
            level.p=scipy_prolongation_from_weights(level.pattern,level.base_weights)
            ac=galerkin_coarse_operator(a,level.p)
        level.r=level.p.T.tocsr()
        level.coarse=build(ac,shape_c,index+1,base_level.coarse if base_level is not None else None)
        return level
    try:
        root=build(base.a,base.shape,0,base)
        if hasattr(components.transfer,'complexity_caps'):
            from .research_transfer import TransferComplexityError
            item=root;nnz_a=0;nnz_p=0
            while item is not None:
                nnz_a+=item.a.count_nonzero();nnz_p+=item.p.count_nonzero() if item.p is not None else 0
                item=item.coarse
            complexity=nnz_a/max(root.a.count_nonzero(),1)
            cap=components.transfer.complexity_caps['max_operator_complexity']
            if complexity>cap:
                raise TransferComplexityError(['aggregate_operator_complexity'],dict(operator_complexity=complexity,limit=cap))
            root.research_complexity=dict(operator_complexity=complexity,transfer_complexity=nnz_p/root.a.shape[0])
        stats.transfer_bank_builds+=1
        return root
    finally:
        elapsed=perf_counter()-start;stats.nn_setup_seconds+=elapsed
        stats.learned_hierarchy_seconds+=max(0.,elapsed-(stats.operator_generation_seconds-gen0)-(stats.cpu_device_transfer_seconds-copy0))


def prepare_smoother_bank(base,components,cfg,stats,operator_cache=None):
    """Overlay stencils on an immutable actual hierarchy; never generate P."""
    start=perf_counter();device=resolve_device(cfg)
    dtype=torch.float32 if cfg.inference_dtype=='float32' else torch.float64
    from ..provenance import operator_digest
    generation_key=(components.expert_signature('smoother'),str(device),str(dtype),
        cfg.mg.smoother_gain_multiplier,cfg.mg.stencil_backend,cfg.mg.native_threads,
        cfg.mg.native_min_cells,cfg.mg.native_parallel_min)
    def build(level):
        result=copy(level)
        if level.coarse is not None:
            result.coarse=build(level.coarse)
            if selected_level(level.index,cfg) and cfg.mg.smoother_gain_multiplier>0 and (cfg.replace_pre or cfg.replace_post):
                key=(operator_digest(level.a),level.shape,generation_key)
                if operator_cache is not None and key in operator_cache:
                    cached=operator_cache[key]
                    if isinstance(cached,GenerationFailure):cached.reject_cached(stats,'smoother')
                    result.neural_stencil=cached;stats.smoother_operator_cache_hits+=1
                else:
                    try:
                        result.neural_stencil=generated_stencil(level,components,cfg,stats,device,dtype)
                    except (ValueError,RuntimeError,FloatingPointError) as error:
                        if operator_cache is not None:operator_cache[key]=GenerationFailure.from_error(error)
                        raise
                    if operator_cache is not None:operator_cache[key]=result.neural_stencil
        return result
    try:
        root=build(base);stats.smoother_bank_builds+=1
        return root
    finally:stats.nn_setup_seconds+=perf_counter()-start


def prepare_learned_bank(base,components,cfg,stats):
    """Compatibility entry point; components are independently lazy."""
    if cfg.branch!='auto':
        cfg=replace(cfg,use_smoother=cfg.branch in {'H_S','H_SP'},use_transfer=cfg.branch in {'H_P','H_SP'})
    if not (cfg.use_smoother or cfg.use_transfer):return base
    root=prepare_transfer_bank(base,components,cfg,stats) if cfg.use_transfer else base
    if cfg.use_smoother:root=prepare_smoother_bank(root,components,cfg,stats)
    stats.learned_hierarchy_builds+=1
    return root


def _bank_apply(bank,r,stats,mask=None,dense_cut=.5):
    if mask is None or bool(np.all(mask)): return bank.apply(r,stats)
    rows=np.flatnonzero(mask);out=np.zeros_like(r)
    if len(rows)==0: return out
    start=perf_counter()
    if len(rows)/r.size>=dense_cut:
        # Dense backend computes all rows but NEVER changes the semantic mask.
        out=bank.apply(r,stats);out[~mask]=0.;stats.dense_masked_calls+=1
        return out
    if bank.native is not None:
        out[rows]=apply_rows(bank.native,r,rows);stats.native_stencil_calls+=1
    else: out[rows]=np.asarray(bank.csr[rows]@r).ravel()
    stats.selected_row_calls+=1;stats.neural_apply_calls+=1
    stats.work_flops+=18*len(rows);stats.neural_apply_seconds+=perf_counter()-start
    return out


def replacement_step(level,r,mask,cfg,stats,reverse=False):
    """Replace, not mask a fully-computed classical sweep.

    Arbitrary line masks cannot preserve a block solve. Whole selected lines are
    replaced; other lines retain their factor solve, zebra order and coupling.
    For line_alt a partial region may contain NO replaceable full lines.
    """
    mask=np.asarray(mask,bool)
    if level.neural_stencil is None or not mask.any():
        return classical_step(level,r,cfg.mg,stats,reverse=reverse)
    stats.possible_rows+=r.size
    bank=level.neural_stencil
    if cfg.application=='additive':
        dc=classical_step(level,r,cfg.mg,stats,reverse=reverse)
        return dc+_bank_apply(bank,r-level.a@dc,stats,mask,cfg.active_dense)
    stats.replacement_calls+=1
    if mask.all():
        stats.active_rows+=r.size
        return _bank_apply(bank,r,stats)
    smoother=level.strategy.smoother
    if smoother=='jacobi':
        d=np.zeros_like(r);d[~mask]=cfg.mg.jacobi_omega*r[~mask]/level.diagonal[~mask]
        stats.active_rows+=int(mask.sum())
        return d+_bank_apply(bank,r,stats,mask,cfg.active_dense)
    if smoother=='chebyshev':
        # A polynomial application is globally coupled. No hidden masked full
        # polynomial + NN double work is called "replacement".
        return classical_step(level,r,cfg.mg,stats,reverse=reverse)
    directions={'line_x':('x',),'line_y':('y',),'line_diag45':('diag45',),
                'line_alt':('y','x') if reverse else ('x','y')}[smoother]
    d=np.zeros_like(r);current=r.copy()
    for direction in directions:
        blocks=level.cache.get(direction)
        for j in _zebra_order(len(blocks),reverse):
            block=blocks[j];rows=block.indices
            if mask[rows].all():
                t=perf_counter()
                inc=apply_rows(bank.native,current,rows) if bank.native is not None else np.asarray(bank.csr[rows]@current).ravel()
                stats.neural_apply_seconds+=perf_counter()-t
                stats.neural_apply_calls+=1;stats.selected_row_calls+=1;stats.replaced_line_solves+=1
                stats.active_rows+=len(rows)
            else:
                t=perf_counter();inc=block.solve(current[rows])
                stats.classical_smooth_seconds+=perf_counter()-t;stats.remaining_line_solves+=1
            d[rows]+=inc;current-=np.asarray(block.columns@inc).ravel()
            stats.work_flops+=2*block.columns.nnz+2*block.block_nnz
    return d


def hybrid_cycle(level,x,b,cfg,stats,spatial,cycle,refresh=False,root_gate=None):
    if level.coarse is None:
        return classical_cycle(level,x,b,cfg.mg,stats)
    stats.level_visits+=1
    enabled=cfg.use_smoother and level.neural_stencil is not None
    if enabled:
        residual=b-matvec(level.a,x,stats)
        if root_gate is None:stats.spatial_gate_calls+=1
        gate=root_gate if root_gate is not None else spatial.gate(level,residual,cycle,stats,refresh)
    else: gate=np.zeros_like(b,dtype=bool)
    for i in range(cfg.mg.pre_steps):
        r=b-matvec(level.a,x,stats)
        d=replacement_step(level,r,gate,cfg,stats) if enabled and i<cfg.replace_pre else classical_step(level,r,cfg.mg,stats,reverse=False)
        x=x+d
    r=b-matvec(level.a,x,stats)
    transfer_start=perf_counter()
    if cfg.spatial and cfg.detector_features=='restriction_lagged' and enabled and (cycle%cfg.gate_refresh==0 or refresh):
        rc=spatial.restriction(level,r,stats)
    else: rc=apply_transfer(level.r,r,stats)
    if getattr(level,'learned_transfer',False):stats.learned_transfer_apply_calls+=1
    stats.learned_transfer_seconds+=perf_counter()-transfer_start
    stats.coarse_path_count+=1
    ec=hybrid_cycle(level.coarse,np.zeros_like(rc),rc,cfg,stats,spatial,cycle,refresh)
    transfer_start=perf_counter()
    x=x+apply_transfer(level.p,ec,stats)
    if getattr(level,'learned_transfer',False):stats.learned_transfer_apply_calls+=1
    stats.learned_transfer_seconds+=perf_counter()-transfer_start
    for i in range(cfg.mg.post_steps):
        r=b-matvec(level.a,x,stats)
        d=replacement_step(level,r,gate,cfg,stats,True) if enabled and i<cfg.replace_post else classical_step(level,r,cfg.mg,stats,reverse=True)
        x=x+d
    return x
