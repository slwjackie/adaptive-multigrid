"""Standalone MG with an irreversible NEURAL -> CLASSICAL state machine.

No global strategy search, no local detector, no spatial masks and no outer
solver. K is chosen once per RHS. Classical P/R/Ac/LU are never rebuilt at a
phase transition. Growth/nonfinite neural trials are rolled back, then the
same cached classical cycle is run from the last finite accepted iterate.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from time import perf_counter
import json
import numpy as np
import scipy.sparse as sp
from .config import MGConfig
from .grid import as_shape, validate_root_shape
from .strategy import get_strategy, global_context_features
from .hierarchy import WorkStats, build_fixed_hierarchy, classical_cycle, neural_cycle, clear_neural_banks
from .policy import choose_k
from .provenance import operator_digest, module_signature, stable_norm, hardware_environment, json_safe


@dataclass
class SolveResult:
    x: np.ndarray
    converged: bool
    stop_reason: str
    mg_cycles: int
    residual_history: list
    relative_residual_history: list
    cycle_seconds: list
    cycle_path: list
    setup_seconds: float
    solve_seconds: float
    strategy_name: str
    decision: dict
    stats: dict
    setup_stats: dict
    switch_events: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    stopping_threshold: float = 0.0
    solver: str = 'standalone_one_way_temporal_mg'

    @property
    def elapsed_seconds(self): return self.setup_seconds+self.solve_seconds
    @property
    def relative_residual(self): return self.relative_residual_history[-1]
    @property
    def final_true_residual(self): return self.residual_history[-1]
    @property
    def convergence_factor(self):
        return (self.residual_history[-1]/max(self.residual_history[0],1e-300))**(1/max(self.mg_cycles,1))
    @property
    def cycles(self): return self.mg_cycles
    @property
    def k_nn(self): return self.decision['k']
    @property
    def executed_mg_cycles(self):
        return self.stats['attempted_neural_cycles']+self.stats['classical_cycles']

    def to_dict(self, *, include_solution=False):
        d={k:v for k,v in self.__dict__.items() if k!='x'}
        d.update(total_seconds=self.elapsed_seconds,executed_mg_cycles=self.executed_mg_cycles,relative_residual=self.relative_residual,
                 final_true_residual=self.final_true_residual,convergence_factor=self.convergence_factor)
        if include_solution: d['x']=self.x.tolist()
        return json_safe(d)


def validate_matrix(a, shape):
    if not sp.issparse(a): raise TypeError('A must be a SciPy sparse matrix')
    if a.shape!=(shape[0]*shape[1],)*2: raise ValueError('operator/grid shape mismatch')
    if not np.isfinite(a.data).all() or np.any(a.diagonal()<=0): raise ValueError('finite positive diagonal required')
    skew=(a-a.T).tocsr()
    if skew.nnz and np.max(np.abs(skew.data))>1e-10*max(np.max(np.abs(a.data)),1):
        raise ValueError('this model assumes a symmetric elliptic operator')
    # Symmetry/positive diagonal are necessary, NOT a proof of SPD.


class PreparedTemporalMGSolver:
    """Reusable A-dependent base hierarchy and optional NN coefficient bank.

    Not thread-safe. Different concurrent solves require different instances.
    Mutation checks are once per solve/batch, never in the cycle loop. A model
    update drops only NN banks; an operator update rebuilds the base hierarchy.
    """
    def __init__(self, a, n, components=None, config=None):
        self.source_a=a; self.shape=as_shape(n); self.components=components
        self.config=config or MGConfig(); self.cache_rebuilds=0; self.total_solves=0
        validate_root_shape(self.shape,self.config.coarsest_n)
        self._build()

    def _model_signatures(self):
        inactive=self.config.mode=='classical' or (self.config.mode=='fixed' and self.config.fixed_k==0)
        if inactive or self.components is None: return ('none','none','{}')
        return (module_signature(self.components.smoother),module_signature(self.components.switch),
                json.dumps(self.components.metadata.get('temporal_certificate',{}),sort_keys=True))

    def _compile_policy(self):
        self.switch_weights=None; self.operator_context=None; self.environment=None
        if (self.config.mode in {'temporal','predicted'} and self.components is not None and self.components.switch is not None
                and (self.config.mode=='predicted' or self.components.metadata.get('temporal_certificate',{}).get('validated'))):
            self.switch_weights=self.components.switch.numpy_weights()
            self.operator_context=global_context_features(self.a,self.shape)
            self.environment=hardware_environment()

    def _build(self):
        start=perf_counter(); validate_matrix(self.source_a,self.shape)
        self.a=self.source_a.tocsr().astype(np.float64,copy=True)
        self.a.sum_duplicates(); self.a.sort_indices()
        self.setup_stats=WorkStats(); self.strategy=get_strategy(self.config.strategy_name)
        self.root=build_fixed_hierarchy(self.a,self.shape,self.strategy,self.config,self.setup_stats)
        if self.components is not None:
            if next(self.components.smoother.parameters()).device.type!='cpu':
                raise ValueError('runtime smoother must reside on CPU')
            self.components.eval()
        self._config_snapshot=self.config
        self._a_signature=operator_digest(self.source_a)
        self.smoother_signature,self.switch_signature,self._meta_signature=self._model_signatures()
        self._compile_policy()
        self.initial_setup_seconds=perf_counter()-start; self._setup_pending=True
        self.cache_rebuilds+=1

    def _ensure_fresh(self):
        sig=operator_digest(self.source_a)
        if sig!=self._a_signature or self.config!=self._config_snapshot:
            self._build(); return self.initial_setup_seconds
        smooth,sw,meta=self._model_signatures()
        if (smooth,sw,meta)!=(self.smoother_signature,self.switch_signature,self._meta_signature):
            if smooth!=self.smoother_signature: clear_neural_banks(self.root)
            self.smoother_signature,self.switch_signature,self._meta_signature=smooth,sw,meta
            self._compile_policy()
        # Certificates are hardware/thread-scoped, including BLAS thread count.
        if self.config.mode=='temporal' and self.environment is not None:
            self.environment=hardware_environment()
        return 0.0

    def replace_operator(self, a, n=None):
        self.source_a=a
        if n is not None: self.shape=as_shape(n)
        validate_root_shape(self.shape,self.config.coarsest_n); self._build()

    def solve(self, b, *, x0=None):
        t=perf_counter(); rebuild_seconds=self._ensure_fresh(); cost=perf_counter()-t
        result=self._solve_fresh(b,x0=x0)
        # A rebuild is already charged to setup_seconds, not charged twice.
        result.solve_seconds+=max(0.0,cost-rebuild_seconds); result.stats['freshness_checks']=1; result.stats['freshness_seconds']=cost
        return result

    def solve_many(self, right_hand_sides, *, x0=None):
        bs=np.asarray(right_hand_sides,dtype=np.float64)
        if bs.ndim!=2 or bs.shape[1]!=self.a.shape[0]: raise ValueError('RHS batch must be (batch,N)')
        if x0 is None: xs=[None]*len(bs)
        else:
            xs=np.asarray(x0,dtype=np.float64)
            if xs.shape!=bs.shape: raise ValueError('x0 batch shape mismatch')
        t=perf_counter(); rebuild_seconds=self._ensure_fresh(); cost=perf_counter()-t
        results=[self._solve_fresh(b,x0=x) for b,x in zip(bs,xs)]
        if results:
            results[0].solve_seconds+=max(0.0,cost-rebuild_seconds); results[0].stats['freshness_checks']=1; results[0].stats['freshness_seconds']=cost
        return results

    def _solve_fresh(self,b,*,x0=None):
        start=perf_counter(); cfg=self.config; stats=WorkStats()
        b=np.asarray(b,dtype=np.float64)
        if b.ndim!=1 or b.size!=self.a.shape[0] or not np.isfinite(b).all(): raise ValueError('invalid RHS')
        x=np.zeros_like(b) if x0 is None else np.asarray(x0,dtype=np.float64).copy()
        if x.shape!=b.shape or not np.isfinite(x).all(): raise ValueError('invalid x0')
        residual=b-self.a@x; norm0=stable_norm(residual)
        if not np.isfinite(norm0): raise ValueError('initial true residual is nonfinite')
        reference=norm0 if cfg.residual_reference=='initial' else stable_norm(b)
        threshold=max(cfg.absolute_tolerance,cfg.tolerance*reference)
        denom=max(reference,cfg.absolute_tolerance,1e-300)
        hist=[norm0]; rel=[norm0/denom]; times=[]; paths=[]; events=[]; attempts=[]
        reason='max_cycles'; bad_count=0; decision={'requested_k':0,'k':0,'reason':'initially_converged','verified':False}
        if norm0>threshold:
            if self.root.coarse is None or cfg.nn_levels==0 or cfg.smoother_gain_multiplier==0:
                decision={'requested_k':0,'k':0,'reason':'no_configured_neural_work','verified':False}
            else:
                decision=choose_k(self,residual,threshold,stats).to_dict()
        k=decision['k']; phase='neural' if k>0 else 'classical'
        if phase=='neural' and self.components is None: raise ValueError('positive K requires a smoother model')

        def switch(reason, after_cycle, **detail):
            nonlocal phase
            if phase=='classical': return
            phase='classical'
            events.append(dict(reason=reason,after_accepted_cycle=after_cycle,
                               first_classical_cycle=after_cycle+1,**detail))

        for cycle in range(1,cfg.max_cycles+1):
            if hist[-1]<=threshold: break
            ct=perf_counter()
            if phase=='neural' and cfg.near_tolerance_factor and hist[-1]<=cfg.near_tolerance_factor*threshold:
                switch('near_tolerance',cycle-1)
            used_nn=phase=='neural'
            exc=None
            if used_nn:
                stats.attempted_neural_cycles+=1
                try:
                    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                        candidate=neural_cycle(self.root,x,b,self.components.smoother,cfg,stats)
                except (FloatingPointError, ValueError) as e:
                    # Only numerical/setup value errors become a logged safe
                    # fallback; programming/runtime errors are not swallowed.
                    candidate=np.full_like(x,np.nan); exc=str(e)
            else:
                candidate=classical_cycle(self.root,x,b,cfg,stats)
            with np.errstate(over='ignore',invalid='ignore'):
                new_r=b-self.a@candidate
            stats.matvecs+=1; stats.work_flops+=2*self.a.nnz
            norm=stable_norm(new_r); rho=norm/max(hist[-1],1e-300)
            bad=(not np.isfinite(norm)) or rho>cfg.safety_growth*(1+cfg.safety_rtol_slack)
            if used_nn:
                attempts.append(dict(cycle=cycle,norm_before=hist[-1],candidate_norm=norm,candidate_rho=rho,rejected=bool(bad),error=exc))
            if used_nn and bad:
                stats.rejected_neural_cycles+=1
                switch('nonfinite_neural' if not np.isfinite(norm) else 'residual_growth',cycle-1,candidate_rho=rho)
                # Transaction rollback: x has not been overwritten by candidate.
                candidate=classical_cycle(self.root,x,b,cfg,stats)
                new_r=b-self.a@candidate; norm=stable_norm(new_r); rho=norm/max(hist[-1],1e-300)
                stats.classical_recovery_cycles+=1; stats.classical_cycles+=1
                stats.matvecs+=1; stats.work_flops+=2*self.a.nnz
                path='classical_recovery'
            elif used_nn:
                stats.accepted_neural_cycles+=1; path='neural'
            else:
                stats.classical_cycles+=1; path='classical'
            x=candidate; residual=new_r
            hist.append(norm); rel.append(norm/denom); paths.append(path)
            if phase=='neural':
                bad_count=bad_count+1 if rho>=cfg.stagnation_rho else 0
                if norm>threshold:
                    if bad_count>=cfg.stagnation_patience: switch('stagnation',cycle,last_rho=rho)
                    elif stats.attempted_neural_cycles>=k: switch('planned_prefix_complete',cycle)
            times.append(perf_counter()-ct)
            if cfg.verbose: print(f'[MG] cycle={cycle:04d} path={path:18s} relres={rel[-1]:.3e} K={k}')
            if not np.isfinite(norm): reason='nonfinite_classical'; break
            if norm>cfg.divergence_factor*max(norm0,threshold): reason='diverged_classical'; break
        converged=bool(np.isfinite(hist[-1]) and hist[-1]<=threshold)
        if converged: reason='converged'
        setup=self.initial_setup_seconds if self._setup_pending else 0.
        setup_stats=self.setup_stats.to_dict() if self._setup_pending else WorkStats().to_dict()
        self._setup_pending=False; self.total_solves+=1
        return SolveResult(x,converged,reason,len(times),hist,rel,times,paths,setup,perf_counter()-start,
                           cfg.strategy_name,decision,stats.to_dict(),setup_stats,events,attempts,threshold)


# Familiar API name retained, but only the new temporal contract is accepted.
PreparedNeuralMGSolver=PreparedTemporalMGSolver


def solve(a,b,n,components=None,config=None,*,x0=None):
    return PreparedTemporalMGSolver(a,n,components,config).solve(b,x0=x0)
