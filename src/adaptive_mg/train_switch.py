"""Train a one-shot temporal policy from full-solve cost tables, not a timer gradient."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .models import SwitchNet
from .policy import N_FEATURES
from .provenance import write_json, module_signature, hardware_environment
from .checkpoint import CONTRACT
from .measurement import measured_actions, attach_objective, paired_policy_evaluation


def collect_table(examples, components, solver_config, k_values, *, repeats=3, warmups=1,
                  lambda_instability=2., seed=7, regimes=('cold',), output=None):
    rows=[]
    for index,e in enumerate(examples):
        for regime in regimes:
            row=measured_actions(e,components,solver_config,k_values,repeats=repeats,warmups=warmups,seed=seed+index,regime=regime)
            attach_objective(row,lambda_instability=lambda_instability)
            rows.append(row)
            print(f"[oracle] {e.name:30s} {regime} K*={row['oracle_k']} successes={sum(a['converged'] for a in row['actions'])}/{len(k_values)}",flush=True)
            if output: write_json(output,rows)
    return rows


def fit_switch(components, train_rows, tune_rows, *, epochs=600, hidden=24, learning_rate=.004, seed=7):
    if not train_rows or not tune_rows: raise ValueError('separate nonempty train and tuning sets required')
    if epochs<1: raise ValueError('epochs must be positive')
    torch.manual_seed(seed)
    k_values=tuple(a['k'] for a in train_rows[0]['actions'])
    for r in train_rows+tune_rows:
        if tuple(a['k'] for a in r['actions'])!=k_values: raise ValueError('inconsistent action tables')
    x=torch.tensor([r['features'] for r in train_rows],dtype=torch.float64)
    costs=torch.tensor([r['training_costs'] for r in train_rows],dtype=torch.float64)
    vx=torch.tensor([r['features'] for r in tune_rows],dtype=torch.float64)
    vc=torch.tensor([r['training_costs'] for r in tune_rows],dtype=torch.float64)
    net=SwitchNet(N_FEATURES,k_values,hidden=hidden).double()
    with torch.no_grad():
        net.feature_mean.copy_(x.mean(0)); net.feature_scale.copy_(x.std(0,unbiased=False).clamp_min(.05))
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
    history=[]; best=deepcopy(net.state_dict()); best_score=float('inf')
    for epoch in range(1,epochs+1):
        net.train(); optimizer.zero_grad(set_to_none=True)
        # Exact empirical expected objective. No BCE, class balancing, forced
        # NN activation, or gradients through discrete hard solver decisions.
        probability=torch.softmax(net(x),-1)
        loss=(probability*costs).sum(-1).mean()
        loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(net.parameters(),5.))
        optimizer.step(); net.eval()
        with torch.no_grad():
            action=net(vx).argmax(-1)
            val=vc[torch.arange(len(vx)),action].mean().item()
            train_action=net(x).argmax(-1)
            train_actual=costs[torch.arange(len(x)),train_action].mean().item()
        if val<best_score-1e-12:
            best_score=val; best=deepcopy(net.state_dict()); best_epoch=epoch
        if epoch==1 or epoch%50==0 or epoch==epochs:
            record=dict(epoch=epoch,expected_objective=float(loss.detach()),gradient_norm=grad,
                        train_discrete_objective=train_actual,tune_discrete_objective=val,
                        tune_nonzero_fraction=float((action!=0).double().mean()))
            history.append(record); print('[switch]',record,flush=True)
    net.load_state_dict(best); components.switch=net.eval()
    components.metadata.update(contract=CONTRACT,policy_validated=False,fast_path_quality_certified=False,
        switch_training={'loss':'E_policy[T_measured/T_classical + lambda_instability * I]',
                         'best_epoch':best_epoch,'best_tune_objective':best_score,'epochs':epochs,
                         'k_values':list(k_values),'train_records':len(train_rows),'tune_records':len(tune_rows),
                         'train_oracle_nonzero':sum(r['oracle_k']>0 for r in train_rows),
                         'all_failure_train_records':sum(r['oracle_status'].startswith('all_actions') for r in train_rows),
                         'history':history})
    return components.metadata['switch_training']


def audit_policy(components, tune_examples, audit_examples, config, *, repeats=3,
                 min_speedup=1.01, seed=1007, output_dir=None, regimes=('cold',)):
    """Both gates must pass; audit is not used to change parameters/actions."""
    reports={}
    for name,examples in [('tune',tune_examples),('audit',audit_examples)]:
        reports[name]={regime:paired_policy_evaluation(examples,components,config,repeats=repeats,seed=seed+j,regime=regime) for j,regime in enumerate(regimes)}
    def passes(r):
        return (r['all_predicted_converged'] and r['new_failures']==0 and r['common_success']>=2
                and r['actual_neural_cases']>0 and r['geometric_speedup'] is not None
                and r['geometric_speedup']>=min_speedup
                and r['paired_case_bootstrap_95_lower'] is not None and r['paired_case_bootstrap_95_lower']>1.)
    valid=all(passes(r) for split in reports.values() for r in split.values())
    sizes=[e.n for e in tune_examples+audit_examples]
    cert={'contract':CONTRACT,'validated':valid,'min_speedup':min_speedup,
          'smoother_signature':module_signature(components.smoother),
          'switch_signature':module_signature(components.switch),
          'environment':hardware_environment(),'scope':config.numerical_scope(),
          'min_n':min(sizes),'max_n':max(sizes),
          'cache_regimes':[0 if x=='cold' else 1 for x in regimes],
          'criterion':'all_accuracy_no_regression_positive_NN_use_paired_speedup_and_case_bootstrap',
          'interpretation':'empirical held-out evidence, not a mathematical convergence guarantee'}
    components.metadata['temporal_certificate']=cert
    components.metadata['policy_validated']=valid
    components.metadata['fast_path_quality_certified']=False  # No claim about an infinite neural-only run.
    if output_dir:
        write_json(Path(output_dir)/'policy_audit.json',reports)
        write_json(Path(output_dir)/'temporal_certificate.json',cert)
    return cert,reports
