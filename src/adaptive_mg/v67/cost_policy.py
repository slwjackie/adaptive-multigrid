"""Continuous-size, empirical cost policy for a FROZEN H_S expert.

No exact-n lookup, test labels, online counterfactual H/C solves, or speed
certificate. Training labels come from actual distinct-RHS workloads. The
operator-level tune optimism margin is empirical, not a coverage theorem.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
import json
import math
import numpy as np

from ..hierarchy import classical_cycle
from ..provenance import hardware_environment, stable_norm, json_safe, write_json
from .banks import Stats
from .config import AdaptiveConfig
from .models import Components
from .research_data import _hash
from .strong import PreparedStrongMG

VERSION = 'continuous-cost-policy-v1'
NUMERIC_FEATURES = ('log_N', 'log_nnz_per_row', 'log_depth', 'log_complexity',
                    'log_rhs', 'cached', 'log_anisotropy', 'log_contrast',
                    'sin_2angle', 'cos_2angle', 'orientation_variation',
                    'local_anisotropic_fraction')


def config_scope(config):
    value = config.to_dict()
    for key in ('mode', 'branch', 'use_smoother', 'use_transfer', 'record_trace'):
        value.pop(key, None)
    value['mg'].pop('strategy_name', None)
    value['mg'].pop('verbose', None)
    return json_safe(value)


def context_from_prepared(prepared, rhs_count, cached):
    level = prepared.classical
    depth, total_nnz = 0, 0
    while level is not None:
        depth += 1; total_nnz += level.a.nnz; level = level.coarse
    selection = prepared.selection
    return dict(N=int(prepared.a.shape[0]), nnz=int(prepared.a.nnz), depth=depth,
                complexity=total_nnz/max(prepared.a.nnz, 1), rhs_count=int(rhs_count),
                cached=bool(cached), strategy=selection.strategy_name,
                rule_id=selection.rule_id, operator_features=dict(selection.features),
                classical_coverage_fallback=bool(selection.rule_evidence.get('fallback_for_coverage', False)))


def context_from_record(example, run, count, cached):
    selection = run.get('selection') or example.strong_selection
    hierarchy = run['classical_hierarchy']
    return dict(N=example.a.shape[0], nnz=example.a.nnz, depth=len(hierarchy),
                complexity=sum(v['operator_nnz'] for v in hierarchy)/example.a.nnz,
                rhs_count=count, cached=cached, strategy=selection['strategy_name'],
                rule_id=selection['rule_id'], operator_features=selection['features'],
                classical_coverage_fallback=bool(selection.get('rule_evidence',{}).get('fallback_for_coverage',False)))


def feature_vector(context, strategies):
    def val(key, default=0.):
        value = context.get('operator_features', {}).get(key, default)
        return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else float(default)
    angle = np.deg2rad(val('principal_angle_deg'))
    numeric = [np.log(max(context['N'], 1)), np.log(max(context['nnz']/context['N'], 1.)),
               np.log(max(context['depth'], 1)), np.log(max(context['complexity'], 1.)),
               np.log(max(context['rhs_count'], 1)), float(context['cached']),
               np.log1p(max(val('tensor_anisotropy_ratio',1.),0)),
               np.log1p(max(val('diagonal_contrast_proxy',1.),0)),
               np.sin(2*angle), np.cos(2*angle), val('orientation_variation'),
               val('local_anisotropic_fraction')]
    result = np.array(numeric+[float(context['strategy']==s) for s in strategies], np.float64)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite policy context')
    return result


def fit_cost_model(train, tune, *, settings=None):
    """Fit TRAIN and choose the empirical optimism margin using TUNE only.

    Each operator has total training weight one irrespective of its number of
    workloads. Repeats are reduced before this function, never independent data.
    """
    settings = dict(settings or {})
    for key,default in (('ridge',1.),('uncertainty_floor',.02),
                        ('extrapolation_penalty',.05),('minimum_speedup_margin',.03)):
        value=float(settings.get(key,default))
        if not np.isfinite(value) or value<0 or (key=='ridge' and value==0):
            raise ValueError('invalid policy setting '+key)
    if not 0<float(settings.get('optimism_quantile',.95))<=1:
        raise ValueError('optimism_quantile must be in (0,1]')
    if not np.isfinite(settings.get('max_size_extrapolation',4.)) or settings.get('max_size_extrapolation',4.)<1:
        raise ValueError('invalid size extrapolation bound')
    if int(settings.get('minimum_strategy_operators',1))<1:
        raise ValueError('invalid strategy coverage')
    tr_ids={r['operator'] for r in train};tu_ids={r['operator'] for r in tune}
    if not tr_ids or not tu_ids or tr_ids & tu_ids:
        raise ValueError('nonempty, operator-disjoint policy fit/tune required')
    minimum=int(settings.get('minimum_operators',3))
    if min(len(tr_ids),len(tu_ids))<minimum:
        raise ValueError('insufficient policy operator coverage')
    usable=lambda r: r['C_success'] and r['H_success'] and r['neural_used'] and r['C_seconds']>0 and r['H_seconds']>0
    good=[r for r in train if usable(r)];valid=[r for r in tune if usable(r)]
    if len(good)<2 or not valid:
        raise ValueError('no common-success real neural policy measurements')
    strategies=sorted({r['context']['strategy'] for r in good})
    x=np.stack([feature_vector(r['context'],strategies) for r in good])
    mean=x.mean(0);scale=np.maximum(x.std(0),.1)
    z=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    y=np.log([r['C_seconds']/r['H_seconds'] for r in good])
    counts={g:sum(r['operator']==g for r in good) for g in tr_ids}
    w=np.array([1/counts[r['operator']] for r in good]);w/=w.mean()
    ridge=float(settings.get('ridge',1.))
    penalty=np.eye(z.shape[1])*ridge;penalty[0,0]=1e-9
    beta=np.linalg.solve(z.T@(w[:,None]*z)+penalty,z.T@(w*y))
    # One optimism residual per operator, worst over tested workloads.
    optimism={}
    for row in valid:
        xx=np.r_[1.,(feature_vector(row['context'],strategies)-mean)/scale]
        error=float(xx@beta-np.log(row['C_seconds']/row['H_seconds']))
        optimism[row['operator']]=max(optimism.get(row['operator'],-np.inf),error)
    q=float(settings.get('optimism_quantile',.95))
    margin=max(float(settings.get('uncertainty_floor',.02)),
               float(np.quantile(list(optimism.values()),q,method='higher')))
    blocked=sorted({r['context']['strategy'] for r in tune if r['C_success'] and not r['H_success']})
    coverage={strategy:len({r['operator'] for r in valid if r['context']['strategy']==strategy}) for strategy in strategies}
    return dict(version=VERSION,strategies=strategies,feature_names=list(NUMERIC_FEATURES)+['strategy:'+s for s in strategies],
        mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),optimism_margin=margin,
        blocked_strategies=blocked,tune_strategy_operators=coverage,
        minimum_strategy_operators=int(settings.get('minimum_strategy_operators',1)),
        N_range=[min(r['context']['N'] for r in good),max(r['context']['N'] for r in good)],
        rhs_range=[min(r['context']['rhs_count'] for r in good),max(r['context']['rhs_count'] for r in good)],
        trained_cache_states=sorted({bool(r['context']['cached']) for r in good}),
        max_size_extrapolation=float(settings.get('max_size_extrapolation',4.)),
        extrapolation_penalty=float(settings.get('extrapolation_penalty',.05)),
        minimum_log_gain=float(np.log1p(settings.get('minimum_speedup_margin',.03))),
        train_operator_ids=sorted(tr_ids),tune_operator_ids=sorted(tu_ids),
        train_log_rmse=float(np.sqrt(np.average((z@beta-y)**2,weights=w))),
        tune_optimism_by_operator=optimism,
        scope='empirical cost extrapolation with conservative abstention; not a statistical/generalization certificate')


def predict_cost(model, context):
    strategy=context['strategy']
    reason=None
    if strategy not in model['strategies']:reason='unseen_classical_strategy'
    elif strategy in model['blocked_strategies']:reason='tune_classical_success_lost'
    elif model['tune_strategy_operators'].get(strategy,0)<model['minimum_strategy_operators']:reason='insufficient_strategy_coverage'
    elif bool(context['cached']) not in model['trained_cache_states']:reason='unseen_cache_state'
    elif not model['rhs_range'][0]<=context['rhs_count']<=model['rhs_range'][1]:reason='rhs_outside_support'
    lo,hi=model['N_range'];n=context['N']
    extrap=max(n/hi,lo/n,1.)
    if extrap>model['max_size_extrapolation']:reason='size_outside_extrapolation_limit'
    if reason:return 'C',dict(reason=reason,size_extrapolation=extrap)
    x=feature_vector(context,model['strategies'])
    prediction=float(np.r_[1.,(x-np.asarray(model['mean']))/np.asarray(model['scale'])]@model['beta'])
    lower=prediction-model['optimism_margin']-model['extrapolation_penalty']*np.log(extrap)
    branch='H_S' if lower>model['minimum_log_gain'] else 'C'
    return branch,dict(reason='positive_empirical_margin' if branch=='H_S' else 'uncertain_or_unfavorable_cost',
        predicted_log_speedup=prediction,lower_log_speedup=lower,size_extrapolation=extrap,
        out_of_training_size=extrap>1,certificate=False)


@dataclass
class ContinuousCostPolicy:
    model: dict
    expert: Components
    rules_digest: str
    solver_scope: dict
    hardware: dict
    expert_signature: str
    provenance: dict
    probe_cycles: int = 0
    continuous_size_policy: bool = True

    @property
    def models(self): return {'H_S':self.expert}

    def to_dict(self):
        return dict(version=VERSION,model=self.model,rules_digest=self.rules_digest,solver_scope=self.solver_scope,
                    hardware=self.hardware,expert_signature=self.expert_signature,provenance=self.provenance,
                    probe_cycles=self.probe_cycles,performance_certified=False)

    def digest(self): return _hash(self.to_dict())

    def save(self,path): write_json(path,self.to_dict())

    @classmethod
    def load(cls,path,expert):
        d=json.loads(Path(path).read_text())
        if d.pop('version')!=VERSION:raise ValueError('stale policy version')
        d.pop('performance_certified',None)
        if d['expert_signature']!=expert.signature():raise ValueError('expert changed; refit policy')
        if not 0<=d.get('probe_cycles',0)<=2:raise ValueError('probe_cycles must be 0,1,2')
        return cls(expert=expert,**d)


class PreparedCostMG(PreparedStrongMG):
    """One shared C* hierarchy and lazy H_S bank; decisions are RHS-local.

    Optional probes are a CONSERVATIVE easy-case veto, not an uncalibrated way
    to turn an uncertain C decision into H. The iterate is never restarted.
    """
    def __init__(self,a,n,*,policy,config,rules,expected_rhs=1):
        self.policy=policy;self.expected_rhs=expected_rhs
        if config_scope(config)!=policy.solver_scope or rules.digest()!=policy.rules_digest:
            raise ValueError('policy solver/rules contract mismatch')
        if hardware_environment()!=policy.hardware:
            raise ValueError('policy hardware/thread/library contract mismatch')
        if policy.expert.signature()!=policy.expert_signature:
            raise ValueError('expert changed; refit policy')
        self._policy_valid=True
        self.policy_id=policy.digest()
        super().__init__(a,n,policy.expert,replace(config,mode='research',branch='auto',use_transfer=False),rules)

    def _build(self):
        super()._build()
        self._cost_context=context_from_prepared(self,1,False)
        self._prediction_cache={}
        self._scope_config=self.config
        self._scope_ok=(config_scope(self.config)==self.policy.solver_scope)

    def _ensure_fresh(self):
        super()._ensure_fresh()
        if self.config != self._scope_config:
            self._scope_config=self.config
            self._scope_ok=(config_scope(self.config)==self.policy.solver_scope)
        # Parent already checks revisions, device/dtype and source A once per
        # solve/batch. Reuse those checks; do not hash the same models twice.
        self._policy_valid=(self.model_digest==self.policy.expert_signature and
                            self._rules_snapshot==self.policy.rules_digest and self._scope_ok)

    def prepare_warm(self):
        st=Stats();self.ensure_branch('H_S',st)
        self.warm_preparation_stats=st.to_dict()

    def solve_many(self,bs,x0=None):
        self.expected_rhs=len(bs)
        return super().solve_many(bs,x0)

    def _solve(self,b,x0=None):
        begin=perf_counter();cfg=self.config
        cached=bool(self.smoother_banks)
        context=dict(self._cost_context,rhs_count=self.expected_rhs,cached=cached)
        prediction_key=(self.expected_rhs,cached,self._policy_valid)
        if prediction_key not in self._prediction_cache:
            self._prediction_cache[prediction_key]=(predict_cost(self.policy.model,context) if self._policy_valid else
                                                   ('C',dict(reason='stale_policy',certificate=False)))
        branch,saved_detail=self._prediction_cache[prediction_key];detail=dict(saved_detail)
        tdec=perf_counter()-begin
        norms=[];times=[];st=Stats()
        # Avoid unnecessary probes on abstaining/unsupported cases. Probe cost
        # is fully online, including residual checks and any final veto.
        if branch=='H_S' and self.policy.probe_cycles:
            x=np.zeros_like(b,dtype=np.float64) if x0 is None else np.array(x0,np.float64,copy=True)
            b=np.asarray(b,np.float64)
            if b.shape!=(self.a.shape[0],) or x.shape!=b.shape or not np.isfinite(b).all() or not np.isfinite(x).all():
                raise ValueError('invalid probe RHS/initial guess')
            norm0=stable_norm(b-self.a@x)
            ref=norm0 if cfg.mg.residual_reference=='initial' else stable_norm(b)
            threshold=max(cfg.mg.absolute_tolerance,cfg.mg.tolerance*ref)
            norms=[norm0]
            for _ in range(min(self.policy.probe_cycles,cfg.mg.max_cycles-1)):
                if norms[-1]<=threshold:break
                t=perf_counter();x=classical_cycle(self.classical,x,b,cfg.mg,st)
                norms.append(stable_norm(b-self.a@x));st.matvecs+=1;st.work_flops+=2*self.a.nnz
                times.append(perf_counter()-t)
            if len(norms)>1:
                rho=(norms[-1]/max(norm0,1e-300))**(1/len(times))
                needed=(np.log(max(threshold,1e-300)/max(norms[-1],1e-300))/np.log(rho)
                        if 0<rho<1 and norms[-1]>threshold else 0. if norms[-1]<=threshold else np.inf)
                detail.update(probe_rho=float(rho),probe_remaining_cycles=float(needed) if np.isfinite(needed) else None)
                if needed<=2:
                    branch='C';detail['reason']='probe_easy_case_veto'
        overhead=perf_counter()-begin
        chosen=replace(cfg,branch=branch,mode='classical' if branch=='C' else 'research')
        if times:
            # Keep the ORIGINAL stopping threshold and shared attempt budget.
            mg=replace(chosen.mg,max_cycles=chosen.mg.max_cycles-len(times),
                       tolerance=np.finfo(float).tiny,absolute_tolerance=threshold)
            chosen=replace(chosen,mg=mg)
        try:
            self.config=chosen
            result=super()._solve(b,x if times else x0)
        finally:
            self.config=cfg
        result.solve_seconds+=overhead
        if times:
            result.residual_history=norms+result.residual_history[1:]
            result.relative_residual_history=[v/max(ref,cfg.mg.absolute_tolerance,1e-300) for v in result.residual_history]
            result.cycle_path=['classical_probe']*len(times)+result.cycle_path
            result.cycle_seconds=times+result.cycle_seconds
            result.executed_cycles+=len(times)
            for key,value in st.to_dict().items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):
                    result.stats[key]=result.stats.get(key,0)+value
            for key in ('classical_cycles','branch_C_cycles','trial_cycles'):
                result.stats[key]=result.stats.get(key,0)+len(times)
            if 'C' not in result.actually_executed_branches:result.actually_executed_branches.insert(0,'C')
        result.stats['controller_seconds']=result.stats.get('controller_seconds',0)+tdec
        result.stats['controller_calls']=result.stats.get('controller_calls',0)+1
        result.stats['policy_probe_cycles']=len(times)
        result.stats['policy_probe_seconds']=sum(times)
        result.requested_branch='auto'
        result.branch_policy_status='empirical_continuous_size_policy_not_certified'
        result.abstention.update(cost_policy=detail,cost_policy_context=context,
                                 chosen_branch=branch,policy_digest=self.policy_id)
        if cfg.record_trace:result.trace.insert(0,dict(state='COST_SELECTION',**detail))
        return result
