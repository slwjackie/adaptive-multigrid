"""One-shot prefix-length decision using A and the initial residual context."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from .strategy import STRATEGIES, global_context_features
from .models import switch_numpy
from .checkpoint import CONTRACT
from .provenance import hardware_environment

N_FEATURES = 18 + len(STRATEGIES) + 10


def initial_features(operator_context, shape, residual, threshold, strategy_name, *, neural_bank_cached=False):
    nx, ny = shape
    r = np.asarray(residual, np.float64).reshape(shape)
    scale=max(float(np.max(np.abs(r), initial=0)),1e-300)
    z=r/scale
    norm=max(float(np.linalg.norm(z)),1e-300)
    # Dimensionless, RHS-aware statistics; absolute small residual is NOT
    # assumed to be intrinsically more difficult for a linear fixed operator.
    dx=float(np.linalg.norm(np.diff(z,axis=0)))/norm
    dy=float(np.linalg.norm(np.diff(z,axis=1)))/norm
    count=z.size
    dynamic=np.array([
        np.clip((np.log(max(scale,1e-300))+np.log(norm)-np.log(max(threshold,1e-300)))/np.log(10),0,300)/20,
        dx,dy,float(z.sum())/(norm*np.sqrt(count)),
        float(np.abs(z).sum())/(norm*np.sqrt(count)),
        float((z[:-1,:]*z[1:,:]).sum())/(norm*norm),
        float((z[:,:-1]*z[:,1:]).sum())/(norm*norm),
        float(np.max(np.abs(z),initial=0))/norm,
        float(neural_bank_cached),1.0],np.float64)
    strategy=np.array([s.name==strategy_name for s in STRATEGIES],np.float64)
    result=np.concatenate((operator_context,strategy,dynamic))
    if result.size != N_FEATURES or not np.isfinite(result).all():
        raise ValueError('invalid temporal policy features')
    return result


@dataclass
class SwitchDecision:
    requested_k: int
    k: int
    reason: str
    verified: bool = False
    logits: list | None = None

    def to_dict(self): return asdict(self)


def certificate_rejection(components, config, shape, smoother_signature, switch_signature, env, *, cache_ready):
    record=components.metadata.get('temporal_certificate',{})
    if not record.get('validated'): return 'policy_not_validated'
    if record.get('contract')!=CONTRACT: return 'wrong_policy_contract'
    if record.get('smoother_signature')!=smoother_signature: return 'smoother_changed'
    if record.get('switch_signature')!=switch_signature: return 'predictor_changed'
    if record.get('environment')!=env: return 'hardware_or_threads_changed'
    if record.get('scope')!=config.numerical_scope(): return 'numerical_scope_changed'
    if not record.get('min_n',0)<=max(shape)<=record.get('max_n',0): return 'grid_out_of_audit_scope'
    if int(cache_ready) not in record.get('cache_regimes',[0]): return 'cache_regime_not_audited'
    return None


def choose_k(prepared, residual, threshold, stats):
    config=prepared.config
    if config.mode=='classical': return SwitchDecision(0,0,'explicit_classical')
    if config.mode=='fixed':
        return SwitchDecision(config.fixed_k,min(config.fixed_k,config.permitted_k),'fixed_k_ablation')
    comp=prepared.components
    if comp is None or comp.switch is None:
        return SwitchDecision(0,0,'no_trained_predictor')
    cached=prepared.root.neural_stencil is not None
    if config.mode=='temporal':
        reason=certificate_rejection(comp,config,prepared.shape,prepared.smoother_signature,
                                     prepared.switch_signature,prepared.environment,cache_ready=cached)
        if reason: return SwitchDecision(0,0,reason)
    start=perf_counter()
    feature=initial_features(prepared.operator_context,prepared.shape,residual,threshold,config.strategy_name,neural_bank_cached=cached)
    stats.predictor_feature_seconds+=perf_counter()-start
    start=perf_counter()
    logits=switch_numpy(prepared.switch_weights,feature)
    if not np.isfinite(logits).all():
        stats.predictor_calls+=1; stats.predictor_seconds+=perf_counter()-start
        return SwitchDecision(0,0,'nonfinite_prediction')
    valid=np.asarray(comp.switch.k_values)<=config.permitted_k
    if not valid.any(): return SwitchDecision(0,0,'no_valid_action')
    masked=np.where(valid,logits,-np.inf)
    k=comp.switch.k_values[int(np.argmax(masked))]
    stats.predictor_calls+=1
    stats.predictor_seconds+=perf_counter()-start
    return SwitchDecision(k,k,'audited_policy' if config.mode=='temporal' else 'unverified_prediction',config.mode=='temporal',logits.tolist())
