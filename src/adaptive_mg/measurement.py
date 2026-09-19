"""Paired wall-clock measurements. Failed solves are censored, never speed wins."""
from __future__ import annotations
from dataclasses import replace
from time import perf_counter
import math
import numpy as np
from .solver import PreparedTemporalMGSolver
from .policy import initial_features
from .strategy import global_context_features
from .provenance import stable_norm


def run_measured(example, components, config, *, regime='cold'):
    if regime not in {'cold','warm'}: raise ValueError('regime must be cold/warm')
    if regime=='warm':
        prepared=PreparedTemporalMGSolver(example.a,example.n,components,config)
        prepared.solve(example.b)  # Explicit out-of-timer preparation, both arms.
        start=perf_counter(); result=prepared.solve(example.b); wall=perf_counter()-start
    else:
        start=perf_counter()
        prepared=PreparedTemporalMGSolver(example.a,example.n,components,config)
        result=prepared.solve(example.b); wall=perf_counter()-start
    record=result.to_dict()
    record['wall_seconds']=wall
    record['time_scope']='cold_constructor_plus_solve' if regime=='cold' else 'warm_solve_only'
    record['relative_solution_error']=stable_norm(result.x-example.exact)/max(stable_norm(example.exact),1e-300)
    return record


def measured_actions(example, components, config, k_values, *, repeats=3, warmups=1, seed=0, regime='cold'):
    if repeats < 1 or warmups < 0: raise ValueError('invalid repetitions')
    if any(k>config.permitted_k for k in k_values): raise ValueError('candidate K exceeds allowed prefix budget')
    rng=np.random.default_rng(seed); output={k:[] for k in k_values}
    for _ in range(warmups):
        for k in k_values:
            run_measured(example,components,replace(config,mode='fixed',fixed_k=k),regime=regime)
    for _ in range(repeats):
        for k in rng.permutation(k_values).tolist():
            output[k].append(run_measured(example,components,replace(config,mode='fixed',fixed_k=k),regime=regime))
    actions=[]
    for k in k_values:
        records=output[k]; times=[r['wall_seconds'] for r in records]
        representative=records[int(np.argsort(times)[len(times)//2])]
        attempts=representative['attempts']
        actions.append(dict(k=k,times=times,median_seconds=float(np.median(times)),
                            converged=all(r['converged'] for r in records),result=representative,
                            nonfinite_attempts=sum(a['candidate_norm'] is None for a in attempts),
                            rejected_attempts=sum(a['rejected'] for a in attempts),
                            growth_log=sum(math.log(max(a['candidate_rho'],1.)) for a in attempts if a['candidate_rho'] is not None)))
    reference=stable_norm(example.b)
    threshold=max(config.absolute_tolerance,config.tolerance*reference)
    context=global_context_features(example.a,(example.n,example.n))
    # Warm regime with a previously prepared NN bank. Each action's standalone
    # K=0 arm has no bank cost either way; it need not materialize NN banks.
    feature=initial_features(context,(example.n,example.n),example.b,threshold,config.strategy_name,neural_bank_cached=regime=='warm')
    return dict(example=example.manifest(),regime=regime,features=feature.tolist(),actions=actions)


def attach_objective(row, *, lambda_instability=2., failure_penalty=100., regression_penalty=25., tie_margin=.01):
    """J=T/T_C+lambda*I, equivalent to (T+lambda*T_C*I)/T_C.

    T is actual measured time-to-tolerance for successful actions. Failure has
    no finite time-to-tolerance; its observed run time is only a lower bound,
    and receives a censoring penalty. Oracle selection excludes all failures.
    """
    if lambda_instability<=0 or failure_penalty<=0 or regression_penalty<0:
        raise ValueError('invalid instability costs')
    baseline=next(a for a in row['actions'] if a['k']==0)
    tc=max(baseline['median_seconds'],1e-12)
    valid=[]
    for a in row['actions']:
        penalty=(failure_penalty*int(not a['converged'])
                 +regression_penalty*int(baseline['converged'] and not a['converged'])
                 +2.*a['nonfinite_attempts']+.25*a['rejected_attempts']+a['growth_log'])
        a['instability_penalty']=penalty
        a['relative_time']=a['median_seconds']/tc
        a['objective']=a['relative_time']+lambda_instability*penalty
        a['time_to_tolerance_seconds']=a['median_seconds'] if a['converged'] else None
        if a['converged']: valid.append(a)
    if valid:
        best=min(valid,key=lambda a:(a['objective'],a['k']))
        if baseline['converged'] and best['objective']>=(1-tie_margin)*baseline['objective']:
            best=baseline
        row['oracle_k']=best['k']; row['oracle_status']='best_tested_prefix_not_global_optimum'
        row['training_costs']=[a['objective'] for a in row['actions']]
    else:
        row['oracle_k']=0; row['oracle_status']='all_actions_failed_abstain'
        # Do not learn that a quicker failure is a good neural prefix.
        row['training_costs']=[0. if a['k']==0 else 1. for a in row['actions']]
    return row


def geometric_mean(values):
    values=[v for v in values if v is not None and math.isfinite(v) and v>0]
    return float(np.exp(np.mean(np.log(values)))) if values else None


def paired_policy_evaluation(examples, components, config, *, repeats=3, seed=0, regime='cold'):
    rows=[]; rng=np.random.default_rng(seed)
    for e in examples:
        modes=['classical','predicted']; raw={m:[] for m in modes}
        for m in modes: run_measured(e,components,replace(config,mode=m),regime=regime)
        for _ in range(repeats):
            for m in rng.permutation(modes).tolist():
                raw[m].append(run_measured(e,components,replace(config,mode=m),regime=regime))
        bc=raw['classical']; pn=raw['predicted']
        tc=float(np.median([r['wall_seconds'] for r in bc])); tn=float(np.median([r['wall_seconds'] for r in pn]))
        c=all(r['converged'] for r in bc); n=all(r['converged'] for r in pn)
        rows.append(dict(name=e.name,n=e.n,classical_converged=c,predicted_converged=n,
                         classical_seconds=tc,predicted_seconds=tn,speedup=tc/tn if c and n else None,
                         k=pn[0]['decision']['k'],neural_cycles=pn[0]['stats']['accepted_neural_cycles'],
                         predictor_calls=pn[0]['stats']['predictor_calls'],raw=raw))
    speedups=[r['speedup'] for r in rows if r['speedup'] is not None]
    lower=None
    if speedups:
        logs=np.log(speedups)
        boot=np.exp(np.mean(rng.choice(logs,(2000,len(logs)),replace=True),axis=1))
        lower=float(np.quantile(boot,.025))
    return dict(rows=rows,geometric_speedup=geometric_mean(speedups),
                paired_case_bootstrap_95_lower=lower,common_success=len(speedups),
                all_predicted_converged=bool(rows) and all(r['predicted_converged'] for r in rows),
                new_failures=sum(r['classical_converged'] and not r['predicted_converged'] for r in rows),
                nonzero_k_cases=sum(r['k']>0 for r in rows),
                actual_neural_cases=sum(r['neural_cycles']>0 for r in rows),
                regime=regime,repeats=repeats)
