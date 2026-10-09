"""Transactional time-varying hierarchies using the repository's MG kernels.

REUSE_P reuses interpolation, NEVER stale Ac/LU when A changes. REFRESH_FINE
rebuilds the finest transfer and keeps deeper P. REBUILD_H uses the unchanged
prepared neural bank builders. C_REBUILD is the current A's selected baseline.
"""
from __future__ import annotations
from dataclasses import dataclass, replace, field
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from ...hierarchy import FixedLevel, build_fixed_hierarchy, classical_cycle, baseline_kwargs
from ...smoothers import LineSmootherCache
from ...transfer import (build_transfer_pattern,baseline_weights,scipy_prolongation_from_weights,
                         galerkin_coarse_operator,coarse_fine_indices)
from ...grid import terminal,next_shape
from ...strategy import get_strategy
from ...provenance import stable_norm
from ..config import AdaptiveConfig
from ..banks import (Stats,prepare_transfer_bank,prepare_smoother_bank,hybrid_cycle,
                     generated_p,resolve_device)
from ..spatial import SpatialState
from ..strong import select_strong_strategy,StrongRules
from ..research_transfer import precheck_transfer_support,enforce_transfer_complexity
from .data import Snapshot,digest

ACTIONS=('REBUILD_C','REUSE_P','REFRESH_FINE','REBUILD_H')
DIRECTIONS={'line_x':('x',),'line_y':('y',),'line_alt':('x','y'),'line_diag45':('diag45',)}

def levels(root):
    while root is not None:
        yield root;root=root.coarse

def hierarchy_signature(root):
    from ...provenance import operator_digest
    return digest([operator_digest(l.p) for l in levels(root) if l.p is not None])

@dataclass(frozen=True)
class Bank:
    root: FixedLevel
    matrix_digest: str
    topology: str
    plan: str
    contract: str
    p_age: int = 0
    neural_transfer: bool = False

@dataclass
class StepResult:
    x: np.ndarray
    success: bool
    action: str
    actual_action: str
    cycles: int
    residuals: list
    threshold: float
    setup_seconds: float
    solve_seconds: float
    total_seconds: float
    bank: Bank | None
    fallback: bool = False
    reason: str = ''
    stats: dict = field(default_factory=dict)

    def record(self):
        return dict(success=self.success,requested_action=self.action,actual_action=self.actual_action,
             cycles=self.cycles,final_true_residual=self.residuals[-1],threshold=self.threshold,
             residuals=self.residuals,setup_seconds=self.setup_seconds,solve_seconds=self.solve_seconds,
             total_seconds=self.total_seconds,fallback=self.fallback,reason=self.reason,stats=self.stats,
             hierarchy_p_digest=hierarchy_signature(self.bank.root) if self.bank else None,
             p_age=self.bank.p_age if self.bank else 0,plan=self.bank.plan if self.bank else None,
             neural_transfer=bool(self.bank and self.bank.neural_transfer),certified=False)

class SequenceBackend:
    """Each independent stream owns its bank; candidate trials do not mutate it."""
    def __init__(self,cfg=None,rules=None,expert=None,*,expert_branch='H_P',max_complexity=8.):
        self.cfg=cfg or AdaptiveConfig(mode='classical',branch='C')
        self.rules=rules or StrongRules()
        if expert_branch not in ('H_S','H_P','H_SP'):raise ValueError('invalid expert branch')
        self.expert=expert.frozen_inference_copy() if expert is not None else None
        self.expert_branch=expert_branch;self.max_complexity=float(max_complexity)
        if not np.isfinite(self.max_complexity) or self.max_complexity<1:raise ValueError('invalid complexity')
        self.expert_signature=self.expert.signature() if self.expert else None
        self.contract=digest(dict(config=self.cfg.to_dict(),rules=self.rules.to_dict(),
              expert=self.expert_signature,branch=expert_branch,max_complexity=max_complexity))

    def select(self,s):
        selection=select_strong_strategy(s.a,s.shape,self.rules)
        cfg=replace(self.cfg,mg=replace(self.cfg.mg,strategy_name=selection.strategy_name))
        return selection,cfg

    def compatible(self,s,bank,plan):
        return bool(bank and bank.topology==s.topology_key and bank.plan==plan and bank.contract==self.contract)

    def available(self,s,bank,selection=None):
        selection=selection or self.select(s)[0]
        ok=self.compatible(s,bank,selection.strategy_name)
        return np.array([True,ok,ok,self.expert is not None],dtype=bool)

    def _factor(self,level,stats):
        if terminal(level.shape,self.cfg.mg.coarsest_n):
            level.lu=sla.splu(level.a.tocsc());stats.coarse_factorizations+=1
        else:
            level.cache=LineSmootherCache(level.a,level.shape)
            for direction in DIRECTIONS.get(level.strategy.smoother,()):level.cache.prepare(direction)
            stats.line_factorizations+=level.cache.factorization_count

    def _updated(self,s,old,cfg,stats,refresh):
        strategy=get_strategy(cfg.mg.strategy_name)
        def build(a,shape,index,previous):
            a=sp.csr_matrix(a,dtype=np.float64,copy=True);a.sum_duplicates();a.eliminate_zeros();a.sort_indices()
            level=FixedLevel(a,shape,index,strategy,np.maximum(a.diagonal(),1e-14));level.learned_transfer=False
            self._factor(level,stats)
            if terminal(shape,cfg.mg.coarsest_n):return level
            cshape=next_shape(shape,strategy.coarsening,cfg.mg.coarsest_n,level_index=index)
            if previous is None or previous.shape!=shape or previous.coarse.shape!=cshape:
                raise ValueError('inherited hierarchy shape mismatch')
            if refresh and index==0:
                level.pattern=build_transfer_pattern(shape,cshape)
                level.base_weights=baseline_weights(a,shape,strategy.transfer,coarse=cshape,**baseline_kwargs(cfg.mg))
                p0=scipy_prolongation_from_weights(level.pattern,level.base_weights);level._classical_p=p0
                if self.expert is not None and self.expert_branch in ('H_P','H_SP'):
                    caps=self.expert.transfer.complexity_caps
                    precheck_transfer_support(a,level.pattern,level.base_weights!=0,baseline_p=p0,caps=caps)
                    dtype=torch.float32 if cfg.inference_dtype=='float32' else torch.float64
                    level.p=generated_p(level,self.expert,cfg,stats,resolve_device(cfg,cells=a.shape[0]),dtype)
                    level.learned_transfer=True
                    ac=galerkin_coarse_operator(a,level.p)
                    enforce_transfer_complexity(a,level.p,ac,baseline_p=p0,
                        baseline_ac=galerkin_coarse_operator(a,p0),caps=caps)
                else:level.p=p0;ac=galerkin_coarse_operator(a,level.p)
            else:
                # P is immutable by contract. Copy numerical buffers so a later
                # trial cannot modify another arm's interpolation.
                level.p=previous.p.copy();level.pattern=previous.pattern
                level.base_weights=np.array(previous.base_weights,copy=True)
                level.learned_transfer=bool(getattr(previous,'learned_transfer',False))
                ac=galerkin_coarse_operator(a,level.p)
            level.r=level.p.T.tocsr()
            if not np.isfinite(level.p.data).all() or np.max(np.asarray(abs(level.p).sum(1)))>8.+1e-10:
                raise ValueError('invalid inherited/refreshed P')
            fixed=coarse_fine_indices(level.pattern)
            inject=(level.p[fixed]-sp.eye(len(fixed),format='csr')).tocsr()
            if inject.nnz and np.max(np.abs(inject.data))>1e-12:
                raise ValueError('coarse injection/full-rank contract changed')
            level.coarse=build(ac,cshape,index+1,previous.coarse)
            return level
        return build(s.a,s.shape,0,old.root)

    def build(self,s,old,action,selection,cfg):
        stats=Stats();plan=selection.strategy_name
        if action not in ACTIONS:raise ValueError('unknown action')
        if action in ('REUSE_P','REFRESH_FINE') and not self.compatible(s,old,plan):
            raise ValueError('mesh/topology/boundary/plan/expert changed: rebuild required')
        if action=='REBUILD_H' and self.expert is None:raise ValueError('no neural expert checkpoint')
        if self.expert is not None and self.expert.signature()!=self.expert_signature:
            raise ValueError('expert modified after world-model contract creation')
        if action=='REUSE_P' and old.matrix_digest==s.matrix_digest:
            return replace(old,p_age=old.p_age+1),dict(stats.to_dict(),exact_matrix_cache_hit=True)
        if action in ('REBUILD_C','REBUILD_H'):
            base=build_fixed_hierarchy(s.a,s.shape,get_strategy(plan),cfg.mg,stats)
            root=base
            if action=='REBUILD_H':
                hcfg=replace(cfg,branch=self.expert_branch,mode='research',
                    use_transfer=self.expert_branch in ('H_P','H_SP'),use_smoother=self.expert_branch in ('H_S','H_SP'))
                if hcfg.use_transfer:root=prepare_transfer_bank(base,self.expert,hcfg,stats)
                if hcfg.use_smoother:root=prepare_smoother_bank(root,self.expert,hcfg,stats)
        else:
            root=self._updated(s,old,cfg,stats,action=='REFRESH_FINE')
            # A-conditioned S is NEVER reused on a numerically changed A.
            if self.expert is not None and self.expert_branch in ('H_S','H_SP') and any(l.neural_stencil is not None for l in levels(old.root)):
                root=prepare_smoother_bank(root,self.expert,replace(cfg,use_smoother=True),stats)
        complexity=sum(l.a.nnz for l in levels(root))/s.a.nnz
        if complexity>self.max_complexity:raise ValueError('temporal hierarchy complexity safety cap')
        neural=any(getattr(l,'learned_transfer',False) for l in levels(root))
        age=old.p_age+1 if action in ('REUSE_P','REFRESH_FINE') else 0
        return Bank(root,s.matrix_digest,s.topology_key,plan,self.contract,age,neural),dict(stats.to_dict(),
                 operator_complexity=complexity,exact_matrix_cache_hit=False,
                 numeric_refactorized=True,temporal_scope='current A; inherited P only; not current-parent optimality')

    def solve(self,s,old,action,*,selection=None,cfg=None):
        started=perf_counter();s.validate(spd_check=False)
        if selection is None:selection,cfg=self.select(s)
        if cfg is None:raise ValueError('selection requires resolved config')
        requested=action;reason='';fallback=False;setup=0.;stats=Stats();build_stats=[]
        if not self.available(s,old,selection)[ACTIONS.index(action)]:
            action='REBUILD_C';reason='incompatible_or_missing_bank';fallback=requested!='REBUILD_C'
        before=perf_counter()
        try:bank,st=self.build(s,old,action,selection,cfg)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            if action=='REBUILD_C':raise
            fallback=True;reason='setup_rejection:'+str(exc);action='REBUILD_C'
            bank,st=self.build(s,None,action,selection,cfg)
        setup+=perf_counter()-before;build_stats.append(st)
        x=s.x0.copy();r=s.b-s.a@x;initial=stable_norm(r)
        reference=initial if cfg.mg.residual_reference=='initial' else stable_norm(s.b)
        threshold=max(cfg.mg.absolute_tolerance,cfg.mg.tolerance*reference)
        history=[initial];attempts=0;stagnant=0;solve_start=perf_counter()
        reserve=1;trial_budget=cfg.mg.max_cycles-reserve
        def run(root,x):
            learned=any(l.neural_stencil is not None or getattr(l,'learned_transfer',False) for l in levels(root))
            if not learned:return classical_cycle(root,x,s.b,cfg.mg,stats)
            hcfg=replace(cfg,mode='research',branch=self.expert_branch,use_transfer=True,
                         use_smoother=any(l.neural_stencil is not None for l in levels(root)),spatial=False,gate_mode='open')
            return hybrid_cycle(root,x,s.b,hcfg,stats,SpatialState(self.expert,hcfg),attempts+1)
        while attempts<cfg.mg.max_cycles and history[-1]>threshold:
            if action!='REBUILD_C' and (attempts>=trial_budget or stagnant>=cfg.mg.stagnation_patience):
                t=perf_counter();bank,st=self.build(s,None,'REBUILD_C',selection,cfg)
                setup+=perf_counter()-t;build_stats.append(st);action='REBUILD_C';fallback=True;reason='stagnation_or_budget_recovery'
            previous=history[-1]
            try:
                with np.errstate(over='ignore',invalid='ignore'):proposal=run(bank.root,x);norm=stable_norm(s.b-s.a@proposal)
            except (ValueError,RuntimeError,FloatingPointError):proposal=x;norm=np.inf
            attempts+=1
            bad=not np.isfinite(norm) or norm>previous*cfg.mg.safety_growth*(1+cfg.mg.safety_rtol_slack)
            if bad and action!='REBUILD_C':
                # Roll back only this trial, retain last accepted x. Same total
                # attempt budget and ORIGINAL stopping threshold are preserved.
                t=perf_counter();bank,st=self.build(s,None,'REBUILD_C',selection,cfg)
                setup+=perf_counter()-t;build_stats.append(st);action='REBUILD_C';fallback=True;reason='residual_rejection'
                continue
            if not np.isfinite(norm):reason='nonfinite_classical';break
            x=proposal;history.append(norm)
            rho=norm/max(previous,1e-300)
            stagnant=stagnant+1 if rho>=cfg.mg.stagnation_rho else 0
            if norm>cfg.mg.divergence_factor*max(initial,threshold):reason='classical_divergence';break
        total=perf_counter()-started
        success=bool(stable_norm(s.b-s.a@x)<=threshold)
        # Build fallback cost is separated from solve cost and always included
        # in total; phase counters do not pretend failed trial setup was free.
        solve_time=max(0.,total-setup)
        return StepResult(x,success,requested,action,attempts,history,threshold,setup,solve_time,total,
                          bank if success else None,fallback,reason or ('converged' if success else 'cycle_limit'),
                          dict(stats.to_dict(),builds=build_stats,attempt_budget=cfg.mg.max_cycles))
