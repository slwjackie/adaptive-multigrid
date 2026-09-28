"""Development-only P headroom -> classical baselines -> S/P factorial study.

New diagnostic artifacts are intentionally not ResearchPolicy training/final
artifacts. A per-operator fitted P is an offline oracle-like candidate, never a
trained generator. Existing solver, P2 safeguards and frozen C*(A) are unchanged.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from ..hierarchy import build_fixed_hierarchy, classical_cycle, classical_step
from ..transfer import coarse_fine_indices, galerkin_coarse_operator
from ..provenance import json_safe, stable_norm, hardware_environment
from .banks import Stats, prepare_smoother_bank, GenerationFailure
from .config import AdaptiveConfig
from .models import Components
from .strong import PreparedStrongMG, load_strong_rules
from .research_data import _restore
from .research_evaluation import manufactured_rhs
from .research_transfer import (enforce_transfer_complexity, hierarchy_complexity_report,
                                precheck_transfer_support)
from .p_headroom import (AffineSupport, energy_minimize, least_squares, slow_vectors,
    dense_budget, spectral_headroom, DiagnosticBudgetError, paired_speedups, factorial_interaction, measured_break_even)

VERSION='p-headroom-v1'
PROJECT=Path(__file__).resolve().parents[3]
METHODS=('classical','energy_min','ls_uniform','ls_energy','direct','nn')


def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix_digest(p):
    p=p.tocsr(copy=True);p.sum_duplicates();p.eliminate_zeros();p.sort_indices()
    return digest(dict(shape=p.shape,indptr=p.indptr.tolist(),indices=p.indices.tolist(),data=p.data.tolist()))


def read(path): return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(json_safe(value),indent=2,allow_nan=False)+'\n');temp.replace(path)


def source_digest():
    paths=sorted((PROJECT/'src').rglob('*.py'))+[PROJECT/'scripts/run_v6_7_p_headroom.py']
    return digest({str(p.relative_to(PROJECT)):file_digest(p) for p in paths if p.exists()})


def level_at(root,index):
    node=root
    for _ in range(index):
        if node.coarse is None:raise ValueError('requested interpolation level is absent')
        node=node.coarse
    if node.coarse is None:raise ValueError('requested level is the terminal solve, not a transfer')
    return node


def hierarchy_counts(root):
    values=[]
    while root is not None:values.append(root.a.count_nonzero());root=root.coarse
    return values


def branch_config(cfg,branch,level):
    return replace(cfg,mode='classical' if branch=='C' else 'research',branch=branch,
        use_smoother=branch in ('H_S','H_SP'),use_transfer=branch in ('H_P','H_SP'),
        transfer_levels=(level,),spatial=False,gate_mode='open',use_learned_controller=False,record_trace=False)


def constraints_for(level):
    return AffineSupport.build(level.p,coarse_fine_indices(level.pattern))


def check_candidate(level,p,caps):
    constraints_for(level).validate(p)
    if np.max(np.asarray(abs(p).sum(1)))>8.:raise ValueError('P row magnitude safety limit')
    ac=galerkin_coarse_operator(level.a,p)
    local=enforce_transfer_complexity(level.a,p,ac,baseline_p=level.p,baseline_ac=level.coarse.a,caps=caps)
    return ac,local


def install_interpolation(base,p,index,cfg,caps,stats):
    """Copy the path, preserve C*, and rebuild factors for the ACTUAL new Ac."""
    target=level_at(base,index);ac,local=check_candidate(target,p,caps)
    def visit(node):
        result=copy(node)
        if node.index==index:
            result.p=p;result.r=p.T.tocsr();result.learned_transfer=True
            result.coarse=build_fixed_hierarchy(ac,node.coarse.shape,node.strategy,cfg.mg,stats,index=node.index+1)
        else:
            result.coarse=visit(node.coarse)
        return result
    result=visit(base)
    total=hierarchy_complexity_report(hierarchy_counts(result),hierarchy_counts(base),caps)
    return result,dict(local_complexity=local,hierarchy_complexity=total)


class AlternativePPrepared(PreparedStrongMG):
    """Experimental P bank with the EXISTING guarded solve and C* recovery.

    Runtime calls this a transfer replacement even for EM/LS. Public study
    records explicitly classify EM/LS as classical, direct as offline fitting,
    and only a frozen generator as NN. Never export these rows as policy labels.
    """
    def __init__(self,a,n,components,config,rules,*,builder,index,caps):
        self.p_builder=builder;self.p_index=index;self.p_caps=caps
        super().__init__(a,n,components,config,rules)

    def _build(self):
        super()._build();self.p_diagnostic={};self.p_value=None
        self._alternative=None;self._alternative_error=None;self.alternative_setup_seconds=0.

    def ensure_branch(self,branch,stats):
        if branch not in ('H_P','H_SP'):return super().ensure_branch(branch,stats)
        if self._alternative_error is not None:self._alternative_error.reject_cached(stats,'transfer')
        if self._alternative is not None:
            stats.cache_hits+=1;self.learned=self._alternative;self._learned_branch=branch
            return self._alternative
        start=perf_counter()
        try:
            cfg=branch_config(self.config,branch,self.p_index);node=level_at(self.classical,self.p_index)
            # Baseline support is known before any generator or numeric Galerkin.
            precheck_transfer_support(node.a,node.pattern,node.base_weights!=0,baseline_p=node.p,caps=self.p_caps)
            p,detail=self.p_builder(self,node,cfg,stats)
            bank,checks=install_interpolation(self.classical,p,self.p_index,cfg,self.p_caps,stats)
            self.p_value=p;self.p_diagnostic=dict(detail,**checks,P_digest=matrix_digest(p))
            if branch=='H_SP':bank=prepare_smoother_bank(bank,self.components,cfg,stats,self.generated_stencil_cache)
            self._alternative=bank;self.learned=bank;self._learned_branch=branch
            stats.transfer_bank_builds+=1;stats.learned_hierarchy_builds+=1;self.learned_builds_total+=1
            return bank
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            self._alternative_error=GenerationFailure.from_error(exc)
            self.p_diagnostic=dict(error=str(exc),status='projection_cap_or_build_failure')
            raise
        finally:
            elapsed=perf_counter()-start
            stats.branch_setup_seconds+=elapsed
            self.alternative_setup_seconds+=elapsed


def build_classical_p(node,kind,cfg,settings):
    start=perf_counter();constraint=constraints_for(node)
    if kind=='classical':return node.p,dict(method=kind,setup_seconds=0.)
    if kind=='energy_min':return energy_minimize(node.a,constraint,maxiter=settings['energy_iterations'])
    if kind not in ('ls_uniform','ls_energy'):raise ValueError('unknown classical interpolation')
    # One sweep is the actual selected classical relaxation, not a V-cycle.
    def smooth(e):return e+classical_step(node,-node.a@e,cfg.mg,Stats(),reverse=False)
    v,vs=slow_vectors(node.a,smooth,count=settings['test_vectors'],sweeps=settings['test_sweeps'],seed=settings['seed'])
    p,ls=least_squares(node.a,constraint,v,coarse_fine_indices(node.pattern),
                      weighting='uniform' if kind=='ls_uniform' else 'energy',ridge=settings['ls_ridge'])
    return p,dict(**vs,**ls,setup_seconds=perf_counter()-start)


class DirectLogits(nn.Module):
    """Per-operator projection parameters: a diagnostic, NOT a generalizing NN."""
    support='support_preserving'
    training_only=False  # Used only by this diagnostic; it is never checkpointed.
    def __init__(self,shape,caps,limit):
        super().__init__();self.logits=nn.Parameter(torch.zeros(shape,dtype=torch.float64))
        self.complexity_caps=dict(caps);self.limit=limit
    def forward_graph(self,a,pattern,baseline):
        if self.logits.shape!=baseline.shape:raise ValueError('direct P used on a different level')
        d=self.logits if self.limit is None else self.limit*torch.tanh(self.logits)
        nx,ny=pattern.fine_shape
        return d.reshape(nx,ny,-1).permute(2,0,1)[None]


def optimize_direct(example,cfg,rules,caps,settings):
    """Fit root/selected-level logits through the actual MULTILEVEL V-cycle.

    No exact-two-grid objective masquerades as a deployed V-cycle objective.
    Independent random error probes are held out from optimizer updates.
    """
    from .research_training import create_research_components, transfer_feasibility
    from .unroll import make_graph,cycle
    index=settings['level'];base=PreparedStrongMG(example.a,example.n,None,branch_config(cfg,'C',index),rules)
    node=level_at(base.classical,index)
    if example.a.shape[0]>settings['direct_max_dofs']:
        raise DiagnosticBudgetError('direct full-V-cycle optimization exceeds direct_max_dofs')
    cfg=branch_config(replace(cfg,mg=replace(cfg.mg,strategy_name=base.selection.strategy_name)),'H_P',index)
    # Use the same inference precision/projection as deployment, not a hidden
    # FP64-only training head. Sparse matrix arithmetic remains FP64.
    model=create_research_components(smoother='student_cnn',transfer='small_gnn',
        smoother_hidden=4,transfer_hidden=4,support='support_preserving',complexity_caps=caps,seed=settings['seed'])
    model.transfer=DirectLogits(node.base_weights.shape,caps,settings['direct_logit_limit'])
    model.transfer.to(dtype=torch.float32 if cfg.inference_dtype=='float32' else torch.float64)
    for parameter in model.smoother.parameters():parameter.requires_grad_(False)
    generator=np.random.default_rng(settings['seed']+23011)
    train=torch.tensor(generator.normal(size=(settings['direct_probes'],example.a.shape[0])),dtype=torch.float64)
    held=torch.tensor(generator.normal(size=(settings['direct_probes'],example.a.shape[0])),dtype=torch.float64)
    opt=torch.optim.Adam(model.transfer.parameters(),lr=settings['direct_lr'])
    best=float('inf');best_p=None;best_logits=None;rows=[];start=perf_counter()
    def evaluate(probes):
        graph=make_graph(example.a,(example.n,example.n),model,cfg)
        reference=make_graph(example.a,(example.n,example.n),model,cfg,learned=False)
        feasible,repair=transfer_feasibility(graph,model,cfg,reference)
        if not feasible['feasible']:return None,repair,graph,feasible
        terms=[]
        for target in probes:
            b=graph.a.apply(target);x=torch.zeros_like(target)
            denominator=(target*b).sum().clamp_min(1e-100)
            for k in range(settings['direct_cycles']):
                x=cycle(graph,x,b,model,cfg,k)
                error=target-x
                terms.append(((error*graph.a.apply(error)).sum().clamp_min(1e-100)/denominator).log())
        return torch.stack(terms).mean(),repair,graph,feasible
    for step in range(settings['direct_steps']+1):
        opt.zero_grad(set_to_none=True);loss,repair,graph,feasible=evaluate(train)
        if loss is not None and torch.isfinite(loss):
            value=float(loss.detach())
            if value<best:
                best=value;target=graph
                for _ in range(index):target=target.coarse
                best_p=target.p.numpy().copy();best_logits=model.transfer.logits.detach().clone()
            objective=loss
        else:
            value=None;objective=repair
        rows.append(dict(step=step,train_log_energy=value,feasible=feasible['feasible']))
        if step==settings['direct_steps']:break
        if objective.requires_grad and torch.isfinite(objective):
            objective.backward();torch.nn.utils.clip_grad_norm_(model.transfer.parameters(),2.);opt.step()
    if best_p is None:raise ValueError('no feasible direct-P iterate')
    with torch.no_grad():model.transfer.logits.copy_(best_logits)
    with torch.no_grad():held_loss,_,_,held_feasible=evaluate(held)
    constraints_for(node).validate(best_p)
    return best_p,dict(method='direct_projection_best_found',offline_only=True,generator_trained=False,
        optimization_seconds=perf_counter()-start,iterations=settings['direct_steps'],train_log_energy=best,
        heldout_log_energy=float(held_loss) if held_loss is not None else None,trace=rows,
        projection='baseline support, existing train/runtime projection',logit_limit=settings['direct_logit_limit'],
        global_optimum_proved=False,objective='m actual V-cycles, same hierarchy depth/smoother; mean log energy-squared ratio')


def smoothing_maps(node,cfg,settings):
    dense_budget(node.a.shape[0],max_dofs=settings['dense_max_dofs'],max_bytes=settings['dense_max_bytes'])
    n=node.a.shape[0];eye=np.eye(n);maps=[]
    for reverse,steps in ((False,cfg.mg.pre_steps),(True,cfg.mg.post_steps)):
        out=eye.copy()
        for j in range(n):
            for _ in range(steps):out[:,j]+=classical_step(node,-node.a@out[:,j],cfg.mg,Stats(),reverse=reverse)
        maps.append(out)
    return maps


def observed_vcycle_decay(root,cfg,*,count=4,cycles=3,seed=1):
    """Held-out errors, deployed multilevel recursion; not an operator norm bound."""
    ratios=[];rng=np.random.default_rng(seed)
    for _ in range(count):
        e=rng.normal(size=root.a.shape[0]);energy=float(e@(root.a@e));x=e.copy()
        for _ in range(cycles):x=classical_cycle(root,x,np.zeros_like(x),cfg.mg,Stats())
        after=float(x@(root.a@x));ratios.append(np.sqrt(max(after,0.)/energy))
    return dict(m_cycles=cycles,probes=count,geometric_error_ratio=float(np.exp(np.mean(np.log(np.maximum(ratios,1e-300))))),
                scope='independent random error probes; fixed classical smoother; actual recursive V-cycle, no safeguard or worst-case certificate')


def make_plan(source_run,out,*,split='train',limit=7,sizes=None,methods=None,level=0,
              repeats=3,rhs_count=4,direct_steps=20,direct_max_dofs=225,
              dense_max_dofs=225,dense_max_bytes=536870912,p_checkpoint=None):
    if split not in ('train','validation'):raise ValueError('only development train/validation may be inspected')
    if any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in (limit,repeats,rhs_count,direct_steps,direct_max_dofs,dense_max_dofs,dense_max_bytes)):
        raise ValueError('positive integer budgets required')
    if level<0:raise ValueError('nonnegative transfer level required')
    source=Path(source_run).resolve();out=Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('diagnostic output must be separate from original run')
    if out.exists() and any(out.iterdir()):raise FileExistsError('use a fresh diagnostic directory')
    settings=read(source/'configuration.json');caps=settings.get('complexity_caps',{})
    if settings.get('support')!='support_preserving' or caps.get('complexity_reference')!='parent':
        raise ValueError('P2 baseline-support / parent-relative contract required before headroom study')
    rules=load_strong_rules(source/'selector_rules.json')
    data_path=source/'data/development_manifest.json';data=read(data_path)
    records=data['splits'][split]
    if any(r['split']!=split for r in records):raise ValueError('split label mismatch')
    # Reconstruct before selecting: all identities must match the saved A/RHS.
    examples=_restore(records,rules)
    grouped={}
    for r,e in zip(records,examples):
        if sizes is None or e.n in sizes:grouped.setdefault((e.n,e.research_family),[]).append(r)
    chosen=[]
    while len(chosen)<limit and any(grouped.values()):
        for key in sorted(grouped):
            if grouped[key] and len(chosen)<limit:chosen.append(grouped[key].pop(0))
    if not chosen:raise ValueError('no matching development operators')
    selected_methods=list(methods or ('classical','energy_min','ls_uniform','ls_energy','direct'))
    if len(set(selected_methods))!=len(selected_methods) or not set(selected_methods)<=set(METHODS):raise ValueError('invalid methods')
    if 'classical' not in selected_methods:raise ValueError('classical control is mandatory')
    pinned=None
    if p_checkpoint is not None:
        path=Path(p_checkpoint).resolve();model=Components.load(path)
        if model.metadata.get('training_rules_digest')!=rules.digest():raise ValueError('P checkpoint uses different frozen rules')
        if getattr(model.transfer,'support',None)!='support_preserving':raise ValueError('P checkpoint is not post-P2 support-preserving')
        pinned=dict(path=str(path),sha256=file_digest(path))
    if 'nn' in selected_methods and pinned is None:raise ValueError('nn comparison requires an existing trained P checkpoint')
    torch.set_num_threads(int(settings.get('torch_threads',1)))
    controls=dict(level=level,repeats=repeats,rhs_count=rhs_count,direct_steps=direct_steps,
        direct_max_dofs=direct_max_dofs,dense_max_dofs=dense_max_dofs,dense_max_bytes=dense_max_bytes,
        seed=2026092807,energy_iterations=150,test_vectors=8,test_sweeps=4,ls_ridge=1e-8,
        direct_probes=3,direct_cycles=3,direct_lr=.03,direct_logit_limit=2.)
    manifest=dict(version=VERSION,source_run=str(source),source_split=split,records=chosen,
        source_inputs={str(p):file_digest(p) for p in (source/'configuration.json',source/'selector_rules.json',data_path)},
        original_settings=settings,controls=controls,methods=selected_methods,p_checkpoint=pinned,
        source_digest=source_digest(),hardware=hardware_environment(),rules_digest=rules.digest(),
        scope='inspected development diagnostic, not held-out generalization or policy-fit evidence',
        final_ood_opened=False,automatic_model_promotion=False)
    manifest['plan_digest']=digest(manifest);out.mkdir(parents=True,exist_ok=True)
    save(out/'plan.json',manifest)
    print('Planned',len(chosen),'operators; no final/OOD data or source-run files changed.',flush=True)
    return manifest


def load_plan(out):
    out=Path(out).resolve();plan=read(out/'plan.json');stored=plan.pop('plan_digest')
    if plan.get('version')!=VERSION or digest(plan)!=stored:raise ValueError('plan integrity mismatch')
    plan['plan_digest']=stored;torch.set_num_threads(int(plan['original_settings'].get('torch_threads',1)))
    if source_digest()!=plan['source_digest'] or hardware_environment()!=plan['hardware']:
        raise ValueError('source/hardware changed; use a new headroom run')
    for p,h in plan['source_inputs'].items():
        if file_digest(p)!=h:raise ValueError('original data/rules/config changed')
    if plan['p_checkpoint'] and file_digest(plan['p_checkpoint']['path'])!=plan['p_checkpoint']['sha256']:
        raise ValueError('P checkpoint changed')
    if any(r['split'] not in ('train','validation') for r in plan['records']):raise ValueError('held-out data prohibited')
    rules=load_strong_rules(Path(plan['source_run'])/'selector_rules.json')
    cfg=AdaptiveConfig.from_dict(plan['original_settings']['solver'])
    return out,plan,cfg,rules,_restore(plan['records'],rules)


def builder_for(kind,settings,*,frozen_p=None,p_model=None):
    def builder(prepared,node,cfg,stats):
        if frozen_p is not None:return frozen_p.copy(),dict(method=kind,offline_only=kind=='direct')
        if kind=='nn':
            # Reuse the current generated_p projection and FP32->FP64 contract.
            from .banks import generated_p,resolve_device
            start=perf_counter()
            p=generated_p(node,p_model,cfg,stats,resolve_device(cfg,cells=node.a.shape[0]),
                          torch.float32 if cfg.inference_dtype=='float32' else torch.float64)
            return p,dict(method='frozen_NN_P',generator_seconds=perf_counter()-start)
        return build_classical_p(node,kind,cfg,settings)
    return builder


def prepared_for(example,cfg,rules,caps,settings,kind,*,s_model=None,frozen_p=None,p_model=None):
    index=settings['level']
    if kind=='classical':
        branch='C' if s_model is None else 'H_S'
        return PreparedStrongMG(example.a,example.n,s_model,branch_config(cfg,branch,index),rules)
    branch='H_P' if s_model is None else 'H_SP'
    return AlternativePPrepared(example.a,example.n,s_model,branch_config(cfg,branch,index),rules,
        builder=builder_for(kind,settings,frozen_p=frozen_p,p_model=p_model),index=index,caps=caps)


def measured(example,factory,settings,kind,regime,*,uses_s=False):
    count=settings['rhs_count'];rhs,exact=manufactured_rhs(example,count+1)
    rhs=np.asarray(rhs);exact=np.asarray(exact);prime=rhs[-1]
    if any(np.array_equal(prime,b) for b in rhs[:-1]):raise ValueError('prime duplicates timed RHS')
    start=perf_counter();prepared=factory();constructor_seconds=perf_counter()-start;prime_seconds=0.;prime_stats={}
    if regime=='warm_multiple':
        # Includes all lazy work in an independent, excluded prime solve.
        prime_result=prepared.solve(prime,np.zeros_like(prime));prime_stats=prime_result.stats;prime_seconds=perf_counter()-start
        start=perf_counter()
    results=prepared.solve_many(rhs[:-1],np.zeros_like(rhs[:-1]));elapsed=perf_counter()-start
    records=[]
    for b,u,result in zip(rhs[:-1],exact[:-1],results):
        norm=stable_norm(b-example.a@result.x);threshold=max(prepared.config.mg.absolute_tolerance,
                                                              prepared.config.mg.tolerance*stable_norm(b))
        good=bool(result.converged and np.isfinite(norm) and norm<=threshold and result.executed_cycles<=prepared.config.mg.max_cycles)
        error=result.x-u
        records.append(dict(success=good,true_residual=norm,threshold=threshold,cycles=result.executed_cycles,
            stop_reason=result.stop_reason,relative_solution_error=stable_norm(error)/max(stable_norm(u),1e-300),
            stats=result.stats))
    transfer_calls=sum(r.stats.get('learned_transfer_apply_calls',0) for r in results)
    smoother_calls=sum(r.stats.get('neural_apply_calls',0) for r in results)
    applied=(kind=='classical' or (getattr(prepared,'p_value',None) is not None and transfer_calls>0))
    if uses_s:applied=applied and smoother_calls>0
    bank_setup=(prepared.alternative_setup_seconds if isinstance(prepared,AlternativePPrepared) else
                prime_stats.get('nn_setup_seconds',0.)+sum(r.stats.get('nn_setup_seconds',0.) for r in results))
    return dict(success=all(r['success'] for r in records),candidate_applied=applied,wall_seconds=elapsed,
        seconds_per_rhs=elapsed/count,warm_prime_seconds=prime_seconds,rhs=records,
        recorded_setup_seconds=constructor_seconds+bank_setup,constructor_seconds=constructor_seconds,bank_setup_seconds=bank_setup,
        timed_transfer_calls=transfer_calls,timed_smoother_calls=smoother_calls,
        actual_NN_numerical_use=bool((uses_s and smoother_calls>0) or (kind=='nn' and transfer_calls>0)),
        setup_scope='excluded_after_independent_prime' if regime=='warm_multiple' else 'included_once',
        intervention_kind='offline_fitted_P' if kind=='direct' else 'NN_P' if kind=='nn' else 'classical_interpolation',
        actual_NN_component=bool(uses_s or kind=='nn'),offline_oracle=kind=='direct',
        P_detail=getattr(prepared,'p_diagnostic',{}),selected_strategy=prepared.selection.strategy_name,
        rules_digest=prepared.selection.rules_digest,hierarchy_depth=len(hierarchy_counts(prepared.classical)),
        note='guard/fallback remain original C*; only genuine candidate applications may support its speedup')


def run_study(out,*,resume=False):
    out,plan,cfg,rules,examples=load_plan(out);settings=plan['controls'];caps=plan['original_settings']['complexity_caps']
    if not resume and (out/'case_results').exists():raise FileExistsError('results exist; use --resume')
    p_model=Components.load(plan['p_checkpoint']['path']).frozen_inference_copy() if plan['p_checkpoint'] else None
    for number,example in enumerate(examples,1):
        path=out/'case_results'/f'{example.group_digest}.json'
        if path.exists():
            old=read(path);h=old.pop('payload_digest')
            if digest(old)!=h or old['plan_digest']!=plan['plan_digest']:raise ValueError('case record integrity mismatch')
            continue
        print(f'[P-headroom] {number}/{len(examples)} {example.name}',flush=True)
        base=PreparedStrongMG(example.a,example.n,None,branch_config(cfg,'C',settings['level']),rules)
        case=dict(plan_digest=plan['plan_digest'],case=example.name,operator=example.group_digest,n=example.n,
                  selected_strategy=base.selection.strategy_name,runs={},candidates={},scope=plan['scope'])
        try:node=level_at(base.classical,settings['level'])
        except ValueError as exc:
            case.update(status='no_transfer_at_level',reason=str(exc));case['payload_digest']=digest(case);save(path,case);continue
        case['level_shape']=node.shape;case['coarse_dimension']=node.p.shape[1]
        candidates={'classical':node.p};direct=None
        for kind in plan['methods']:
            if kind=='classical':continue
            try:
                if kind=='direct':p,detail=optimize_direct(example,cfg,rules,caps,settings);direct=p
                elif kind=='nn':p,detail=builder_for(kind,settings,p_model=p_model)(base,node,branch_config(cfg,'H_P',settings['level']),Stats())
                else:p,detail=build_classical_p(node,kind,cfg,settings)
                bank,checks=install_interpolation(base.classical,p,settings['level'],cfg,caps,Stats())
                candidates[kind]=p;sp.save_npz(out/f'P_{example.group_digest}_{kind}.npz',p)
                case['candidates'][kind]=dict(status='feasible',detail=detail,checks=checks,P_digest=matrix_digest(p),
                    P_relative_change=float(sp.linalg.norm(p-node.p)/max(sp.linalg.norm(node.p),1e-300)),
                    Vcycle_error=observed_vcycle_decay(bank,cfg,seed=settings['seed']+993))
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                case['candidates'][kind]=dict(status='skipped_budget' if isinstance(exc,DiagnosticBudgetError) else 'projection_cap_or_optimization_failure',reason=str(exc))
        case['classical_Vcycle_error']=observed_vcycle_decay(base.classical,cfg,seed=settings['seed']+993)
        try:
            pre,post=smoothing_maps(node,cfg,settings)
            case['spectral']=spectral_headroom(node.a,pre,post,candidates,max_dofs=settings['dense_max_dofs'],max_bytes=settings['dense_max_bytes'])
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            case['spectral']=dict(status='skipped_budget' if isinstance(exc,DiagnosticBudgetError) else 'not_applicable',reason=str(exc))
        active=[k for k in plan['methods'] if k in candidates]
        rng=np.random.default_rng(settings['seed']+number)
        for regime in ('warm_multiple','multiple'):
            case['runs'][regime]={k:[] for k in active}
            for repeat in range(settings['repeats']):
                for kind in rng.permutation(active):
                    kind=str(kind)
                    factory=lambda kind=kind:prepared_for(example,cfg,rules,caps,settings,kind,
                        frozen_p=direct if kind=='direct' else None,p_model=p_model)
                    result=measured(example,factory,settings,kind,regime)
                    if kind!='classical' and result['candidate_applied']:
                        if result['P_detail']['P_digest']!=matrix_digest(candidates[kind]):raise ValueError('non-reproducible P builder')
                    case['runs'][regime][kind].append(result)
        case['status']='complete';case['payload_digest']=digest(case);save(path,case)
        save(out/'progress.json',dict(completed=number,total=len(examples),status='running'))
    report=report_study(out)
    save(out/'progress.json',dict(completed=len(examples),total=len(examples),status='complete'))
    return report


def load_results(out,plan,examples):
    cases=[]
    for e in examples:
        p=out/'case_results'/f'{e.group_digest}.json'
        if not p.exists():raise ValueError('complete run before report/factorial')
        row=read(p);h=row.pop('payload_digest')
        if row['plan_digest']!=plan['plan_digest'] or digest(row)!=h:raise ValueError('result integrity mismatch')
        for method,candidate in row.get('candidates',{}).items():
            if candidate['status']=='feasible':
                mat=sp.load_npz(out/f'P_{e.group_digest}_{method}.npz')
                if matrix_digest(mat)!=candidate['P_digest']:raise ValueError('saved P changed')
        cases.append(row)
    return cases


def report_study(out):
    out,plan,cfg,rules,examples=load_plan(out);cases=load_results(out,plan,examples);summary={};gates=[]
    for regime in ('warm_multiple','multiple'):
        arms={}
        for method in plan['methods']:
            rows={}
            for c in cases:
                runs=c.get('runs',{}).get(regime,{}).get(method,[])
                if runs:rows[c['operator']]=dict(success=all(r['success'] and r['candidate_applied'] for r in runs),seconds=float(np.median([r['wall_seconds'] for r in runs])))
                else:rows[c['operator']]=dict(success=False,seconds=None)
            arms[method]=rows
        summary[regime]={m:dict(**paired_speedups(arms['classical'],arm),offline_oracle=m=='direct') for m,arm in arms.items()}
    for c in cases:
        reason='headroom_inconclusive';spc=c.get('spectral',{})
        if spc.get('deployed_optimality_applicable'):
            qc=spc['candidates']['classical']['deployed_two_grid_A_norm'];qo=spc['symmetric_two_grid_optimal_A_norm']
            if qc-qo<=.05*max(qc,1e-12):reason='little_two_grid_headroom_only'
            else:reason='headroom_remains_compare_classical_then_learned'
        if any(v['status']=='projection_cap_or_optimization_failure' for v in c.get('candidates',{}).values()):
            reason='inspect_candidate_constraints_or_optimizer_before_generator_claim'
        gates.append(dict(operator=c['operator'],recommendation=reason,automatic_stop=False))
    economics=[]
    for c in cases:
        for m in plan['methods']:
            if m in ('classical','direct'):continue
            wr=c.get('runs',{}).get('warm_multiple',{});cr=c.get('runs',{}).get('multiple',{})
            if not all(k in wr and k in cr for k in ('classical',m)):continue
            if not all(r['success'] and r['candidate_applied'] for k in ('classical',m) for r in wr[k]+cr[k]):continue
            saved=float(np.median([r['seconds_per_rhs'] for r in wr['classical']])-np.median([r['seconds_per_rhs'] for r in wr[m]]))
            extra=float(np.median([r['recorded_setup_seconds'] for r in cr[m]])-np.median([r['recorded_setup_seconds'] for r in cr['classical']]))
            economics.append(dict(operator=c['operator'],method=m,observed_setup_difference=extra,observed_warm_seconds_saved_per_rhs=saved,
                linearized_break_even_rhs=measured_break_even(extra,saved),scope='observed batch-average rates; not a tested crossover at other RHS counts'))
    classical_winners=[m for m in ('energy_min','ls_uniform','ls_energy') if m in summary['warm_multiple']
        and not summary['warm_multiple'][m]['new_failures'] and len(summary['warm_multiple'][m]['common'])>=3
        and summary['warm_multiple'][m]['ci95'] and summary['warm_multiple'][m]['ci95'][0]>1.03]
    report=dict(version=VERSION,summary=summary,case_gates=gates,economics=economics,classical_candidates_to_validate=classical_winners,
        next_step='independent_classical_validation_before_NN' if classical_winners else 'inspect_headroom_and_direct_best_found_gap',
        learned_LWLS_implemented=False,alternating_training_implemented=False,final_ood_opened=False,
        automatic_promotion=False,scope='development decision support; direct P is per-A optimization, not generalization; spectral headroom is not a wall-clock bound')
    save(out/'report.json',report)
    for regime,methods in summary.items():
        for method,r in methods.items():print(regime,method,'speedup=',r['geometric_speedup'],'new_failures=',len(r['new_failures']),'offline_oracle=',method=='direct',flush=True)
    return report


def factorial_study(out,s_selection,*,p_kind='energy_min',resume=False):
    out,plan,base_cfg,rules,examples=load_plan(out);cases=load_results(out,plan,examples)
    if p_kind=='classical' or p_kind not in plan['methods']:raise ValueError('choose a measured non-classical P intervention')
    selected_path=Path(s_selection).resolve();selected=read(selected_path)
    checkpoint=Path(selected['checkpoint'])
    if file_digest(checkpoint)!=selected['checkpoint_sha256'] or selected['rules_digest']!=rules.digest():
        raise ValueError('selected S checkpoint/rules changed or use a different parent selector')
    s_model=Components.load(checkpoint).frozen_inference_copy();s_cfg=AdaptiveConfig.from_dict(selected['solver'])
    if s_model.signature()!=selected['expert_signature']:raise ValueError('S expert signature mismatch')
    original=base_cfg.mg.to_dict() if hasattr(base_cfg.mg,'to_dict') else base_cfg.to_dict()['mg']
    chosen=s_cfg.mg.to_dict() if hasattr(s_cfg.mg,'to_dict') else s_cfg.to_dict()['mg']
    for d in (original,chosen):
        d.pop('strategy_name',None);d.pop('verbose',None)
    if original!=chosen:raise ValueError('factorial requires the same classical solver/tolerance/precision configuration')
    settings=plan['controls'];caps=plan['original_settings']['complexity_caps']
    p_model=Components.load(plan['p_checkpoint']['path']).frozen_inference_copy() if plan['p_checkpoint'] else None
    target=out/('factorial_'+p_kind);manifest=dict(version=VERSION,plan_digest=plan['plan_digest'],
        S_selection=str(selected_path),S_selection_sha256=file_digest(selected_path),
        S_checkpoint_sha256=file_digest(checkpoint),P_kind=p_kind,solver=s_cfg.to_dict(),
        scope='fixed weights and same C*; diagnostic factorial, no joint training or promotion')
    if (target/'manifest.json').exists():
        if not resume or read(target/'manifest.json')!=json_safe(manifest):raise ValueError('factorial protocol changed')
    else:
        if resume:raise FileNotFoundError('no factorial run to resume')
        save(target/'manifest.json',manifest)
    result_rows=[]
    for example,case in zip(examples,cases):
        candidate=case.get('candidates',{}).get(p_kind,{})
        if candidate.get('status')!='feasible':
            result_rows.append(dict(operator=example.group_digest,status='P_not_feasible'));continue
        p=sp.load_npz(out/f'P_{example.group_digest}_{p_kind}.npz')
        path=target/(example.group_digest+'.json')
        if path.exists():
            row=read(path);h=row.pop('payload_digest')
            if digest(row)!=h or row['manifest_digest']!=digest(manifest):raise ValueError('factorial result changed')
            result_rows.append(row);continue
        row=dict(operator=example.group_digest,case=example.name,manifest_digest=digest(manifest),runs={},interactions={})
        for regime in ('warm_multiple','multiple'):
            measurements={name:[] for name in ('C','H_P','H_S','H_SP')}
            for repeat in range(settings['repeats']):
                order=np.random.default_rng(settings['seed']+repeat).permutation(list(measurements))
                for arm in order:
                    use_s=arm in ('H_S','H_SP');method=p_kind if arm in ('H_P','H_SP') else 'classical'
                    factory=lambda use_s=use_s,method=method:prepared_for(example,s_cfg,rules,caps,settings,method,
                        s_model=s_model if use_s else None,frozen_p=p if method=='direct' else None,p_model=p_model)
                    measurement=measured(example,factory,settings,method,regime,uses_s=use_s)
                    if method!='classical' and measurement['candidate_applied']:
                        if measurement['P_detail']['P_digest']!=candidate['P_digest']:raise ValueError('S/P factorial did not use identical P')
                    measurements[arm].append(measurement)
            row['runs'][regime]=measurements
            if all(all(r['success'] and r['candidate_applied'] for r in values) for values in measurements.values()):
                times=[float(np.median([r['wall_seconds'] for r in measurements[name]])) for name in ('C','H_P','H_S','H_SP')]
                row['interactions'][regime]=factorial_interaction(*times)
            else:row['interactions'][regime]=dict(status='not_all_four_successful',no_failure_speedup=True)
        row['status']='complete';row['payload_digest']=digest(row);save(path,row);row.pop('payload_digest');result_rows.append(row)
    report=dict(rows=result_rows,automatic_joint_training=False,final_certificate=False,
        scope='development 2x2; inspect P gain in BOTH S contexts; a conditional P benefit is not logically excluded by weak standalone H_P')
    save(target/'report.json',report)
    print('Factorial saved:',target/'report.json','; no weights or selector were modified.',flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('plan');p.add_argument('--source-run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--split',choices=('train','validation'),default='train');p.add_argument('--limit',type=int,default=7)
    p.add_argument('--sizes',nargs='+',type=int);p.add_argument('--methods',nargs='+',choices=METHODS)
    p.add_argument('--level',type=int,default=0);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--rhs-count',type=int,default=4);p.add_argument('--direct-steps',type=int,default=20)
    p.add_argument('--direct-max-dofs',type=int,default=225);p.add_argument('--dense-max-dofs',type=int,default=225)
    p.add_argument('--dense-max-bytes',type=int,default=536870912);p.add_argument('--p-checkpoint')
    for name in ('run','report','factorial'):
        p=commands.add_parser(name);p.add_argument('--run-dir',required=True)
        if name!='report':p.add_argument('--resume',action='store_true')
        if name=='factorial':p.add_argument('--s-selection',required=True);p.add_argument('--p-kind',choices=METHODS,default='energy_min')
    a=parser.parse_args(argv)
    if a.command=='plan':
        values=vars(a).copy();values.pop('command');return make_plan(**values)
    if a.command=='run':return run_study(a.run_dir,resume=a.resume)
    if a.command=='report':return report_study(a.run_dir)
    return factorial_study(a.run_dir,a.s_selection,p_kind=a.p_kind,resume=a.resume)
