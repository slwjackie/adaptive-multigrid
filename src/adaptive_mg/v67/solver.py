"""Production v6.7: C <-> H -> LOCK; standalone, transactional trial updates."""
from dataclasses import dataclass,field,replace
from time import perf_counter
import json
import hashlib
import numpy as np
from ..grid import as_shape,validate_root_shape
from ..strategy import get_strategy,global_context_features
from ..hierarchy import build_fixed_hierarchy,classical_cycle
from ..solver import validate_matrix
from ..provenance import operator_digest,module_signature,stable_norm,hardware_environment,json_safe
from .config import AdaptiveConfig
from .banks import (Stats,prepare_transfer_bank,prepare_smoother_bank,hybrid_cycle,resolve_device,
                    GenerationFailure,CachedGenerationFailure)
from .controller import OnlineHistory
from .spatial import SpatialState

@dataclass
class AdaptiveResult:
    x: np.ndarray
    converged: bool
    stop_reason: str
    residual_history: list
    relative_residual_history: list
    cycle_seconds: list
    cycle_path: list
    trace: list
    setup_seconds: float
    solve_seconds: float
    stats: dict
    setup_stats: dict
    stopping_threshold: float
    executed_cycles: int
    cache_key: str
    certificate_used: bool=False
    requested_branch: str='auto'
    actually_executed_branches: list=field(default_factory=list)
    p_bank: str='classical'
    branch_policy_status: str='not_applicable'
    abstention: dict=field(default_factory=dict)
    @property
    def elapsed_seconds(self): return self.setup_seconds+self.solve_seconds
    @property
    def cycles(self): return len(self.cycle_path)
    @property
    def mg_cycles(self): return self.cycles
    @property
    def relative_residual(self): return self.relative_residual_history[-1]
    @property
    def final_true_residual(self): return self.residual_history[-1]
    @property
    def convergence_factor(self):
        return (self.residual_history[-1]/max(self.residual_history[0],1e-300))**(1/max(self.cycles,1))
    def to_dict(self,include_solution=False):
        d={k:v for k,v in self.__dict__.items() if k!='x'}
        d.update(total_seconds=self.elapsed_seconds,cycles=self.cycles,relative_residual=self.relative_residual,
            final_true_residual=self.final_true_residual,convergence_factor=self.convergence_factor,
            solver='standalone_v67_hierarchical_replacement')
        if include_solution:d['x']=self.x.tolist()
        return json_safe(d)

class PreparedAdaptiveMG:
    """Reusable per-operator banks. Mutation is checked once per solve/batch.

    Prepared instances are not thread-safe; each concurrent solve needs its own
    instance. Every RHS gets a fresh temporal and spatial policy state.
    """
    def __init__(self,a,n,components=None,config=None):
        self.source_a=a;self.shape=as_shape(n);self.components=components;self.config=config or AdaptiveConfig()
        validate_root_shape(self.shape,self.config.mg.coarsest_n)
        self.cache_rebuilds=0;self.learned_builds_total=0;self._build()
    def _effective_config(self):
        branch=self.config.branch
        if branch=='auto':return self.config
        return replace(self.config,use_smoother=branch in {'H_S','H_SP'},use_transfer=branch in {'H_P','H_SP'})
    def _branch(self):
        cfg=self._effective_config()
        return ('H_SP' if cfg.use_smoother and cfg.use_transfer else 'H_S' if cfg.use_smoother else 'H_P' if cfg.use_transfer else 'C') if cfg.branch=='auto' else cfg.branch
    def _potential_neural(self):
        cfg=self._effective_config()
        return (self.components is not None and cfg.mode!='classical' and cfg.branch!='C'
            and cfg.gate_mode!='closed' and (cfg.use_smoother or cfg.use_transfer) and cfg.mg.nn_levels!=0
            and (cfg.mode!='production' or self.components.metadata.get('certificate',{}).get('validated',False)))
    def _signature(self): return self.components.signature() if self._potential_neural() else 'none'
    def _base_scope(self):
        mg=self.config.mg
        return (self.shape,mg.strategy_name,mg.coarsest_n,mg.operator_sweeps,mg.operator_omega,
                mg.operator_blend,mg.operator_drop_tolerance,mg.operator_candidate_topk)
    def _component_key(self,name):
        module=getattr(self.components,name)
        return (self.components.expert_signature(name),tuple((str(v.device),str(v.dtype)) for v in tuple(module.parameters())+tuple(module.buffers())))
    def _generation_scope(self,cfg):
        return (cfg.mg.nn_levels,str(resolve_device(cfg,cells=self.a.shape[0])),cfg.inference_dtype,'actual_A/reference_classical_A:fp64:v1')
    def _refresh_key(self):
        payload=(self.a_digest,self.shape,self.model_digest,self.config.to_dict())
        self.cache_key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    def _build(self):
        t=perf_counter();validate_matrix(self.source_a,self.shape)
        self.a=self.source_a.tocsr().astype(np.float64,copy=True);self.a.sum_duplicates();self.a.sort_indices()
        self.setup_stats=Stats();self.strategy=get_strategy(self.config.mg.strategy_name)
        self.classical=build_fixed_hierarchy(self.a,self.shape,self.strategy,self.config.mg,self.setup_stats)
        self.learned=None;self._learned_branch=None;self.a_digest=operator_digest(self.source_a)
        self.transfer_banks={};self.smoother_banks={};self.branch_banks={}
        self.generated_stencil_cache={}
        self.failed_transfer_banks={};self.failed_smoother_banks={}
        self.model_digest=self._signature();self.cfg_snapshot=self.config;self.base_snapshot=self._base_scope()
        self.device_snapshot=self._device_snapshot()
        self.compiled_controller=None;self.compiled_detector=None;self.context=None
        self.environment=hardware_environment() if self.components and self.components.metadata.get('certificate',{}).get('validated') else None
        self._refresh_key()
        self.initial_setup_seconds=perf_counter()-t;self.setup_pending=True;self.cache_rebuilds+=1
    def _device_snapshot(self):
        return tuple((str(p.device),str(p.dtype)) for m in self.components.modules() for p in tuple(m.parameters())+tuple(m.buffers())) if self._potential_neural() else ()
    def _ensure_fresh(self):
        self.environment=hardware_environment() if self.components and self.components.metadata.get('certificate',{}).get('validated') else None
        if operator_digest(self.source_a)!=self.a_digest or self._base_scope()!=self.base_snapshot:
            self._build();return
        signature=self._signature()
        devices=self._device_snapshot()
        if signature!=self.model_digest or self.config!=self.cfg_snapshot or devices!=self.device_snapshot:
            # Independent keys preserve unaffected transfer/hierarchy banks after
            # an S-only change and preserve the classical bank after any NN change.
            self.learned=None;self._learned_branch=None;self.compiled_detector=None;self.compiled_controller=None
            self.model_digest=signature;self.cfg_snapshot=self.config;self.device_snapshot=devices;self._refresh_key()
    def replace_operator(self,a,n=None):
        self.source_a=a
        if n is not None:self.shape=as_shape(n)
        validate_root_shape(self.shape,self.config.mg.coarsest_n);self._build()
    def certificate_valid(self):
        if self.components is None:return False
        c=self.components.metadata.get('certificate',{})
        return bool(c.get('validated') and c.get('model_signature')==self.model_digest
            and c.get('scope')==self.config.certification_scope() and c.get('hardware')==self.environment)
    def ensure_branch(self,branch,stats):
        if branch not in {'C','H_S','H_P','H_SP'}:raise ValueError('unknown explicit branch')
        if branch=='C':stats.classical_bank_cache_hits+=1;return self.classical
        if self.components is None:raise ValueError('learned mode requires components')
        from .research_runtime import require_components_allowed
        require_components_allowed(self.components,self.config,branch)
        start=perf_counter();cfg=replace(self.config,use_smoother=branch in {'H_S','H_SP'},use_transfer=branch in {'H_P','H_SP'})
        shared=(self.a_digest,self.shape,self.base_snapshot,self._generation_scope(cfg))
        pkey=shared+(self._component_key('transfer'),) if cfg.use_transfer else ('classical',self.a_digest,self.base_snapshot)
        skey=(pkey,self._component_key('smoother'),self._generation_scope(cfg),cfg.mg.smoother_gain_multiplier,
              cfg.mg.stencil_backend,cfg.mg.native_threads,cfg.mg.native_min_cells,cfg.mg.native_parallel_min,
              bool(cfg.replace_pre or cfg.replace_post)) if cfg.use_smoother else None
        key=(branch,pkey,skey)
        if key in self.branch_banks:
            stats.cache_hits+=1
            if cfg.use_transfer:stats.transfer_bank_cache_hits+=1
            else:stats.classical_bank_cache_hits+=1
            if cfg.use_smoother:stats.smoother_bank_cache_hits+=1
            self.learned=self.branch_banks[key];self._learned_branch=branch;return self.learned
        try:
            root=self.classical
            if cfg.use_transfer:
                if pkey in self.failed_transfer_banks:self.failed_transfer_banks[pkey].reject_cached(stats,'transfer')
                if pkey not in self.transfer_banks:
                    try:self.transfer_banks[pkey]=prepare_transfer_bank(self.classical,self.components,cfg,stats)
                    except (ValueError,RuntimeError,FloatingPointError) as error:
                        self.failed_transfer_banks[pkey]=GenerationFailure.from_error(error)
                        raise
                else:stats.transfer_bank_cache_hits+=1
                root=self.transfer_banks[pkey]
            else:stats.classical_bank_cache_hits+=1
            if cfg.use_smoother:
                if skey in self.failed_smoother_banks:self.failed_smoother_banks[skey].reject_cached(stats,'smoother')
                if skey not in self.smoother_banks:
                    try:self.smoother_banks[skey]=prepare_smoother_bank(root,self.components,cfg,stats,self.generated_stencil_cache)
                    except (ValueError,RuntimeError,FloatingPointError) as error:
                        self.failed_smoother_banks[skey]=GenerationFailure.from_error(error)
                        raise
                else:stats.smoother_bank_cache_hits+=1
                root=self.smoother_banks[skey]
            self.branch_banks[key]=root;self.learned=root;self._learned_branch=branch
            stats.learned_hierarchy_builds+=1;self.learned_builds_total+=1
            return root
        finally:stats.branch_setup_seconds+=perf_counter()-start
    def ensure_learned(self,stats):
        return self.ensure_branch(self._branch(),stats)
    def _detector_net(self):
        if self.compiled_detector is None:self.compiled_detector=self.components.detector.compile()
        return self.compiled_detector
    def _policy_status(self,cfg,branch):
        if cfg.branch=='C' or cfg.mode=='classical':return 'not_applicable'
        if cfg.branch!='auto' or cfg.mode in {'hybrid','burst'}:return 'forced_component_with_safety'
        if self.components is None:return 'components_unavailable'
        meta=self.components.metadata
        if meta.get('policy_stale'):return 'stale_policy'
        if not cfg.use_learned_controller or not meta.get('controller_trained'):return 'controller_disabled_or_untrained'
        for name in ('controller_expert_signatures','detector_expert_signatures'):
            expected=meta.get(name)
            if expected is not None and any(expected.get(k)!=module_signature(getattr(self.components,k)) for k in ('smoother','transfer')):
                return 'stale_expert_policy'
        for name in ('policy_generation_signature','controller_generation_signature','detector_generation_signature'):
            expected=meta.get(name)
            if expected is not None and expected!=self.components.generation_signature():return 'stale_expert_policy'
        entry=meta.get('branch_calibration',{}).get(branch)
        if entry is None:return 'legacy_combined_policy' if branch=='H_SP' else 'uncalibrated_branch'
        expected=dict(generation_signature=self.components.generation_signature(),config_scope=cfg.certification_scope(),
                      controller_signature=module_signature(self.components.controller),detector_signature=module_signature(self.components.detector))
        if not entry.get('validated') or any(entry.get(k)!=v for k,v in expected.items()):return 'stale_branch_calibration'
        if entry.get('hardware')!=hardware_environment():return 'stale_branch_calibration'
        return 'calibrated_branch_policy'
    def _classical_tail(self,b,x,history,rel,paths,times,stats,threshold,denom,norm0,executed,locked):
        """Pure classical continuation from the last accepted x and budget.

        No temporal history, spatial detector, model hash, branch decision or
        learned setup is called inside this loop. Cold bypass uses this same loop.
        """
        mg=self.config.mg;count=0;stop='max_cycles'
        while executed<mg.max_cycles and history[-1]>threshold:
            ct=perf_counter();x=classical_cycle(self.classical,x,b,mg,stats)
            norm=stable_norm(b-self.a@x)
            stats.matvecs+=1;stats.work_flops+=2*self.a.nnz
            history.append(norm);rel.append(norm/denom);paths.append('classical');times.append(perf_counter()-ct)
            executed+=1;count+=1
            if not np.isfinite(norm):stop='nonfinite_classical';break
            if norm>mg.divergence_factor*max(norm0,threshold):stop='diverged_classical';break
        stats.trial_cycles+=count;stats.classical_cycles+=count;stats.branch_C_cycles+=count
        if locked:stats.classical_lock_cycles+=count
        else:stats.classical_bypass_cycles+=count
        return x,executed,stop
    def solve(self,b,x0=None):
        t=perf_counter();before=self.cache_rebuilds;self._ensure_fresh();dt=perf_counter()-t
        r=self._solve(b,x0)
        # A full rebuild is charged by pending setup, not charged a second time.
        extra=max(0.,dt-(self.initial_setup_seconds if self.cache_rebuilds>before else 0))
        r.solve_seconds+=extra;r.stats['freshness_checks']=1;r.stats['freshness_seconds']=dt
        return r
    def solve_many(self,bs,x0=None):
        bs=np.asarray(bs,np.float64)
        if bs.ndim!=2 or bs.shape[1]!=self.a.shape[0]:raise ValueError('expected RHS (batch,N)')
        xs=[None]*len(bs) if x0 is None else np.asarray(x0,np.float64)
        if x0 is not None and xs.shape!=bs.shape:raise ValueError('x0 batch mismatch')
        t=perf_counter();before=self.cache_rebuilds;self._ensure_fresh();dt=perf_counter()-t
        out=[self._solve(b,x) for b,x in zip(bs,xs)]
        if out:
            extra=max(0.,dt-(self.initial_setup_seconds if self.cache_rebuilds>before else 0))
            out[0].solve_seconds+=extra;out[0].stats['freshness_checks']=1;out[0].stats['freshness_seconds']=dt
        return out
    def _solve(self,b,x0=None):
        start=perf_counter();cfg=self._effective_config();mg=cfg.mg;stats=Stats();branch=self._branch()
        b=np.asarray(b,np.float64);x=np.zeros_like(b) if x0 is None else np.array(x0,np.float64,copy=True)
        if b.shape!=(self.a.shape[0],) or x.shape!=b.shape or not np.isfinite(b).all() or not np.isfinite(x).all():
            raise ValueError('invalid b/x0')
        r=b-self.a@x;norm0=stable_norm(r)
        if not np.isfinite(norm0):raise ValueError('nonfinite initial residual')
        ref=norm0 if mg.residual_reference=='initial' else stable_norm(b)
        threshold=max(mg.absolute_tolerance,mg.tolerance*ref);denom=max(ref,mg.absolute_tolerance,1e-300)
        history=[norm0];rel=[norm0/denom];paths=[];times=[];trace=[]
        cert=self.certificate_valid() if cfg.mode=='production' else False
        status=self._policy_status(cfg,branch)
        reason=None
        if branch=='C' or cfg.mode=='classical':reason='explicit_classical'
        elif cfg.mode=='production' and not cert:reason='invalid_production_certificate'
        elif cfg.gate_mode=='closed':reason='static_closed_gate'
        elif (not cfg.use_smoother and not cfg.use_transfer) or mg.nn_levels==0 or self.classical.coarse is None:reason='no_configured_neural_work'
        elif not cfg.use_transfer and (mg.smoother_gain_multiplier==0 or not (cfg.replace_pre or cfg.replace_post)):reason='no_configured_smoother_application'
        elif cfg.branch=='auto' and cfg.mode not in {'hybrid','burst'} and status not in {'legacy_combined_policy','calibrated_branch_policy'}:reason=status
        bypass=reason is not None
        if not bypass and self.components is not None:
            # Check once per RHS even when a teacher-generated bank was cached
            # inside a prior offline context. Never add this to each V-cycle.
            from .research_runtime import require_components_allowed
            require_components_allowed(self.components,cfg,branch)
        # State is RHS-local. It is not even constructed for permanent cold C.
        spatial=None if bypass else SpatialState(self.components,cfg,net_factory=self._detector_net)
        online=None if bypass else OnlineHistory()
        if not bypass and cfg.branch=='auto' and cfg.mode not in {'hybrid','burst'}:
            if self.context is None:self.context=global_context_features(self.a,self.shape)
            if self.compiled_controller is None:self.compiled_controller=self.components.controller.compile()
        lock_at=None
        if not bypass and not cfg.lazy and norm0>threshold:
            try:self.ensure_branch(branch,stats)
            except (ValueError,FloatingPointError,RuntimeError) as exc:
                stats.setup_failures+=int(not isinstance(exc,CachedGenerationFailure));stats.lock_count+=1;online.mode='CLASSICAL_LOCK'
                lock_at=0
                reason='eager_setup_failure';trace.append(dict(cycle=0,state='CLASSICAL_LOCK',reason=reason,error=str(exc)))
        executed=0;stop='max_cycles'
        while executed<mg.max_cycles and history[-1]>threshold:
            if bypass or online.mode=='CLASSICAL_LOCK':
                if not bypass and lock_at is None:lock_at=executed
                x,executed,stop=self._classical_tail(b,x,history,rel,paths,times,stats,threshold,denom,norm0,executed,not bypass)
                break
            ct=perf_counter();cycle=len(paths)+1
            if cfg.branch!='auto':
                state='CLASSICAL_LOCK' if history[-1]<=mg.near_tolerance_factor*threshold else 'HYBRID'
                detail=dict(reason='near_tolerance' if state=='CLASSICAL_LOCK' else 'forced_component_with_safety')
            else:
                t=perf_counter()
                def counted_net(features):
                    stats.controller_nn_calls+=1
                    return self.compiled_controller(features)
                state,detail=online.select(cfg,counted_net,
                    bool(cfg.use_learned_controller and self.components and self.components.metadata.get('controller_trained')),self.context,
                    history[-1],threshold,norm0,cycle,self.learned is not None)
                stats.controller_calls+=1;stats.controller_seconds+=perf_counter()-t
            entered=state=='HYBRID' and online.mode!='HYBRID'
            if state!=online.mode:
                online.dwell=0
                if entered:online.entries+=1;stats.hybrid_entries+=1;spatial.invalidate()
                if state=='CLASSICAL_LOCK':stats.lock_count+=1
                online.mode=state
            if state=='CLASSICAL_LOCK':
                lock_at=executed
                reason=detail['reason'];trace.append(dict(cycle=cycle,state=state,decision=detail))
                continue
            old_norm=history[-1];nn_setup_before=stats.nn_setup_seconds
            used_h=False;exception=None;gate_fraction=0.;application_before=stats.neural_apply_calls+stats.learned_transfer_apply_calls
            try:
                if state=='HYBRID':
                    used_h=True;stats.attempted_neural_cycles+=1
                    setattr(stats,'branch_'+branch+'_attempts',getattr(stats,'branch_'+branch+'_attempts')+1)
                    if self.learned is not None and self._learned_branch==branch:
                        root=self.learned;stats.cache_hits+=1
                    else:root=self.ensure_branch(branch,stats)
                    if cfg.use_smoother:stats.spatial_gate_calls+=1
                    rootgate=spatial.gate(root,r,cycle,stats,entered) if cfg.use_smoother else None
                    gate_fraction=float(rootgate.mean()) if rootgate is not None else 0.
                    # Preserve legacy closed-root behavior; explicit H_P never
                    # builds or evaluates smoother-only spatial state.
                    if rootgate is not None and not rootgate.any():
                        used_h=False;stats.attempted_neural_cycles-=1
                        stats.gate_closed_fallback_cycles+=1
                        setattr(stats,'branch_'+branch+'_attempts',getattr(stats,'branch_'+branch+'_attempts')-1)
                        candidate=classical_cycle(self.classical,x,b,mg,stats);path='classical';detail['reason']='all_gates_closed'
                    else:
                        with np.errstate(over='ignore',invalid='ignore'):
                            candidate=hybrid_cycle(root,x,b,cfg,stats,spatial,cycle,entered,rootgate)
                        path='hybrid'
                else:candidate=classical_cycle(self.classical,x,b,mg,stats);path='classical'
            except (ValueError,FloatingPointError,RuntimeError) as exc:
                if state!='HYBRID':raise
                exception=f'{type(exc).__name__}: {exc}';candidate=np.full_like(x,np.nan);path='hybrid'
                stats.setup_failures+=int(not isinstance(exc,CachedGenerationFailure) and stats.neural_apply_calls+stats.learned_transfer_apply_calls==application_before)
            executed+=1;stats.trial_cycles+=1
            t=perf_counter()
            with np.errstate(over='ignore',invalid='ignore'):newr=b-self.a@candidate
            norm=stable_norm(newr);rho=norm/max(old_norm,1e-300)
            stats.matvecs+=1;stats.work_flops+=2*self.a.nnz
            bad=not np.isfinite(norm) or rho>=mg.safety_growth*(1+mg.safety_rtol_slack)
            trial_rho=rho
            if used_h and bad:
                stats.rollback_count+=1;stats.rejected_neural_cycles+=1
                stats.lock_count+=1;online.mode='CLASSICAL_LOCK';online.dwell=0
                lock_at=executed
                detail['reason']='hybrid_numerical_rejection';reason=detail['reason']
                if executed>=mg.max_cycles:
                    stats.safeguard_seconds+=perf_counter()-t
                    trace.append(dict(cycle=cycle,state='CLASSICAL_LOCK',rejected=True,trial_rho=trial_rho,error=exception))
                    stop='cycle_budget_after_rejected_trial';break
                candidate=classical_cycle(self.classical,x,b,mg,stats)
                newr=b-self.a@candidate;norm=stable_norm(newr);rho=norm/max(old_norm,1e-300)
                stats.matvecs+=1;stats.work_flops+=2*self.a.nnz
                executed+=1;stats.trial_cycles+=1;stats.classical_recovery_cycles+=1;path='classical_recovery'
            stats.safeguard_seconds+=perf_counter()-t
            if path=='hybrid':
                stats.accepted_neural_cycles+=1;setattr(stats,'branch_'+branch+'_cycles',getattr(stats,'branch_'+branch+'_cycles')+1)
            else:stats.classical_cycles+=1;stats.branch_C_cycles+=1
            x=candidate;r=newr;history.append(norm);rel.append(norm/denom);paths.append(path)
            elapsed=perf_counter()-ct;times.append(elapsed)
            steady_seconds=max(elapsed-(stats.nn_setup_seconds-nn_setup_before),1e-9)
            if online.mode!='CLASSICAL_LOCK':online.update('HYBRID' if path=='hybrid' else 'CLASSICAL',rho,steady_seconds)
            if cfg.record_trace:
                trace.append(dict(cycle=cycle,executed_cycles=executed,state=online.mode,path=path,branch=branch if path=='hybrid' else 'C',rho=rho,
                    trial_rho=trial_rho,rejected=bool(used_h and bad),seconds=elapsed,
                    eta=-np.log(max(rho,1e-300))/steady_seconds,active_fraction=gate_fraction,
                    learned_bank_ready=self.learned is not None,decision=detail,error=exception,
                    learned_numerical_applications=stats.neural_apply_calls+stats.learned_transfer_apply_calls-application_before))
            if not np.isfinite(norm):stop='nonfinite_classical';break
            if norm>mg.divergence_factor*max(norm0,threshold):stop='diverged_classical';break
        converged=bool(np.isfinite(history[-1]) and history[-1]<=threshold)
        if converged:stop='converged'
        setup=self.initial_setup_seconds if self.setup_pending else 0.
        setupstats=self.setup_stats.to_dict() if self.setup_pending else Stats().to_dict()
        self.setup_pending=False
        values=stats.to_dict();values['learned_operator_applications']=stats.neural_apply_calls+stats.learned_transfer_apply_calls
        values['nn_forward_calls']=stats.smoother_nn_calls+stats.transfer_nn_calls+stats.controller_nn_calls+stats.detector_calls
        values['fallback_cycles']=stats.classical_recovery_cycles+stats.gate_closed_fallback_cycles
        actual=[name for name in ('C','H_S','H_P','H_SP') if getattr(stats,'branch_'+name+'_cycles') or (name!='C' and getattr(stats,'branch_'+name+'_attempts'))]
        return AdaptiveResult(x,converged,stop,history,rel,times,paths,trace,setup,perf_counter()-start,
            values,setupstats,threshold,executed,self.cache_key,cert,cfg.branch,actual,
            'learned' if stats.learned_transfer_apply_calls else 'classical',status,
            dict(cold_classical_only=bypass,reason=reason,locked_after_attempt=lock_at,
                 statement='per-cycle learned-controller overhead removed after abstention' if bypass or stats.lock_count else 'adaptive reentry remains enabled'))
