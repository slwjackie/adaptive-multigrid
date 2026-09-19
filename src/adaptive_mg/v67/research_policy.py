"""Offline strong-aware tiny policies and conservative research-only deployment.

WHICH is a small A-only bucket utility table; WHEN/WHERE retain the existing
compiled tiny networks and transactional CLASSICAL LOCK numerical path. No
counterfactual solve is performed by a deployed policy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..hierarchy import classical_cycle, classical_step
from ..provenance import hardware_environment, json_safe, operator_digest, stable_norm, write_json
from .banks import Stats
from .models import Components, TinyNet, DETECTOR_INPUTS, CONTROLLER_INPUTS
from .native import block_statistics
from .research_runtime import is_teacher
from .spatial import features_from_statistics
from .strong import PreparedStrongMG, StrongRules, select_strong_strategy
from .training import temporal_table, fit_controller

BRANCHES = ('H_S', 'H_P', 'H_SP')
RHS_COUNTS = (1, 4, 16, 64)


def _hash(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _branch_config(cfg, branch, *, forced=False):
    return replace(cfg, branch=branch if forced else 'auto', mode='research',
                   use_smoother=branch in {'H_S','H_SP'}, use_transfer=branch in {'H_P','H_SP'})


def _scope(cfg):
    value=cfg.to_dict()
    for name in ('mode','record_trace','branch','use_smoother','use_transfer'):
        value.pop(name,None)
    value['mg'].pop('strategy_name',None)
    value['mg'].pop('verbose',None)
    return value


def _bucket(selection, shape):
    n=max(shape) if isinstance(shape,tuple) else shape
    return str(selection['rule_id'] if isinstance(selection,dict) else selection.rule_id)+'|n='+str(n)


def _selection(example, rules):
    actual=select_strong_strategy(example.a,example.n,rules)
    stored=getattr(example,'strong_selection',None)
    if not isinstance(stored,dict) or any(stored.get(k)!=getattr(actual,k) for k in ('strategy_name','rule_id','rules_digest')):
        raise ValueError('Policy training requires the immutable recorded C*(A) selection')
    return actual


def _validate_development(train, validation, rules, manifest):
    if not isinstance(manifest,dict) or manifest.get('rules_digest')!=rules.digest():
        raise ValueError('A committed development manifest with matching frozen rules is required')
    if set(manifest.get('splits',{}))!={'train','validation'}:
        raise ValueError('Policies accept train and validation only; no final/OOD labels')
    seen=set(manifest.get('historical_index',{}).get('normalized_operator_digests',[]))
    for split,examples in (('train',train),('validation',validation)):
        expected={r['operator_digest']:r for r in manifest['splits'][split]}
        if len(expected)!=len(manifest['splits'][split]):
            raise ValueError('Duplicate development operator records')
        for e in examples:
            digest=operator_digest(e.a);group=operator_digest(e.a,scale_invariant=True)
            record=expected.get(digest)
            if record is None or group!=record['normalized_operator_digest'] or group in seen:
                raise ValueError('Policy split overlap, historical leakage or undeclared operator')
            if hashlib.sha256(np.asarray(e.b).tobytes()).hexdigest()!=record['rhs_digest']:
                raise ValueError('Policy RHS differs from committed development manifest')
            if 'exact_digest' in record and hashlib.sha256(np.asarray(e.exact).tobytes()).hexdigest()!=record['exact_digest']:
                raise ValueError('Policy exact solution differs from committed development manifest')
            seen.add(group);_selection(e,rules)
    if not train or not validation:
        raise ValueError('Policy fit requires nonempty disjoint training and validation sets')


def _spatial_data(models, examples, cfg, rules, settings):
    features=[];labels=[];records=[]
    # A shared detector sees both S-only and actual P+S hierarchy contexts. If the
    # two S experts differ, neither expert is silently assigned the other labels.
    for branch,model in models.items():
        if branch not in {'H_S','H_SP'}:continue
        for e in examples:
            selection=_selection(e,rules)
            chosen=replace(_branch_config(cfg,branch,forced=True),spatial=False,gate_mode='open',
                           mg=replace(cfg.mg,strategy_name=selection.strategy_name))
            prepared=PreparedStrongMG(e.a,e.n,model,chosen,rules);st=Stats()
            try:
                root=prepared.ensure_branch(branch,st);x=np.zeros_like(e.b)
                if root.neural_stencil is None:continue
                for state in range(int(settings.get('detector_states',2))):
                    residual=e.b-root.a@x
                    scale=max(float(np.max(np.abs(residual),initial=0)),1e-300)
                    _,raw,blocks=block_statistics(residual/scale,root.shape,cfg.detector_max_side)
                    feature=features_from_statistics(raw,root.shape,blocks)
                    tc=[];tn=[]
                    for _ in range(int(settings.get('timing_repeats',3))):
                        t=perf_counter();dc=classical_step(root,residual,chosen.mg,Stats(),reverse=False);tc.append(perf_counter()-t)
                        t=perf_counter();dn=root.neural_stencil.apply(residual,Stats());tn.append(perf_counter()-t)
                    _,rawc,_=block_statistics((residual-root.a@dc)/scale,root.shape,cfg.detector_max_side)
                    _,rawn,_=block_statistics((residual-root.a@dn)/scale,root.shape,cfg.detector_max_side)
                    energy=np.maximum(raw[:,0],1e-30)
                    ec=-.5*np.log(np.maximum(rawc[:,0],1e-30)/energy)/max(float(np.median(tc)),1e-9)
                    en=-.5*np.log(np.maximum(rawn[:,0],1e-30)/energy)/max(float(np.median(tn)),1e-9)
                    target=((en>0)&(en>(1+cfg.win_margin)*ec)&(rawn[:,0]<rawc[:,0])).astype(np.float32)
                    finite=np.isfinite(feature).all(1)&np.isfinite(ec)&np.isfinite(en)
                    features.extend(feature[finite]);labels.extend(target[finite])
                    records.append(dict(operator=e.digest,branch=branch,strategy=selection.strategy_name,state=state,
                                        blocks=int(finite.sum()),positives=int(target[finite].sum()),
                                        classical_seconds=float(np.median(tc)),neural_seconds=float(np.median(tn))))
                    x=classical_cycle(prepared.classical,x,e.b,chosen.mg,Stats())
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                records.append(dict(operator=e.digest,branch=branch,error=str(exc),strategy=selection.strategy_name))
    return np.asarray(features,np.float32),np.asarray(labels,np.float32),records


def _fit_shared_detector(models,train,cfg,rules,settings,out):
    x,y,records=_spatial_data(models,train,cfg,rules,settings)
    torch.manual_seed(int(settings.get('seed',71)))
    detector=TinyNet(DETECTOR_INPUTS,1)
    trained=bool(len(y) and np.any(y>0))
    if len(y):
        xt=torch.tensor(x);yt=torch.tensor(y);detector.normalize_on(xt)
    if trained:
        optimizer=torch.optim.Adam(detector.parameters(),lr=float(settings.get('policy_lr',.002)))
        for _ in range(int(settings.get('epochs',{}).get('detector',100))):
            optimizer.zero_grad(set_to_none=True)
            logits=detector(xt).ravel()
            positive=max(float(yt.sum()),1.)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,yt,pos_weight=yt.new_tensor(max((len(yt)-positive)/positive,1.)))
            loss.backward();optimizer.step()
    else:
        with torch.no_grad():detector.fc2.weight.zero_();detector.fc2.bias.fill_(-20.)
    for model in models.values():
        model.detector=deepcopy(detector).eval()
        model.metadata['detector_trained']=trained
        model.metadata['detector_expert_signatures']={k:model.expert_signature(k) for k in ('smoother','transfer')}
    write_json(out/'spatial_oracle.json',dict(records=records,trained=trained,examples=len(y),positives=int(y.sum()),
               scope='training-only local efficiency proxy against recorded C*(A); no accuracy certificate'))
    if len(y):np.savez_compressed(out/'spatial_data.npz',features=x,targets=y)


def _run_candidate(example, model, cfg, rules, branch):
    """Real cold and repeated-same-A solves. Extrapolated RHS totals are labels."""
    chosen=replace(cfg,branch=branch,mode='classical' if branch=='C' else 'research')
    total_start=perf_counter();t=total_start
    cold_wall=None
    try:
        prepared=PreparedStrongMG(example.a,example.n,model,chosen,rules)
        cold=prepared.solve(example.b);cold_wall=perf_counter()-t
        t=perf_counter();warm=prepared.solve(example.b);warm_wall=perf_counter()-t
        residual_c=stable_norm(example.b-example.a@cold.x)
        residual_w=stable_norm(example.b-example.a@warm.x)
        good=bool(cold.converged and warm.converged and residual_c<=cold.stopping_threshold and residual_w<=warm.stopping_threshold)
        used=bool(cold.stats.get('accepted_neural_cycles',0)>0 or warm.stats.get('accepted_neural_cycles',0)>0)
        return dict(cold_seconds=cold_wall,warm_seconds=warm_wall,success=good,neural_used=used,
                    cold=cold.to_dict(),warm=warm.to_dict(),recomputed_residual=[residual_c,residual_w])
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        return dict(cold_seconds=cold_wall if cold_wall is not None else perf_counter()-total_start,
                    warm_seconds=None,failed_seconds=perf_counter()-total_start,success=False,neural_used=False,error=str(exc))


def collect_which_rows(examples, models, cfg, rules, settings, *, split):
    if split not in {'train','validation'}:raise ValueError('No final/OOD policy labels')
    rows=[];rng=np.random.default_rng(int(settings.get('seed',71)))
    repeats=int(settings.get('timing_repeats',3))
    if repeats<1:raise ValueError('timing_repeats must be positive')
    for e in examples:
        selected=_selection(e,rules);runs={b:[] for b in ('C',*models)}
        for repeat in range(repeats):
            for branch in rng.permutation(list(runs)):
                runs[branch].append(_run_candidate(e,models.get(branch),cfg,rules,branch))
        rows.append(dict(split=split,operator=e.digest,normalized_operator_digest=e.group_digest,
                         bucket=_bucket(selected,e.n),strategy_name=selected.strategy_name,runs=runs))
    return rows


def _utility(row, branch, rhs, cached):
    c=row['runs'].get('C',[]);h=row['runs'].get(branch,[])
    if not c or not h or not all(v['success'] for v in c+h) or not any(v['neural_used'] for v in h):return None
    def total(run):
        return rhs*run['warm_seconds'] if cached else run['cold_seconds']+(rhs-1)*run['warm_seconds']
    ct=np.median([total(v) for v in c]);ht=np.median([total(v) for v in h])
    if not np.isfinite(ct+ht) or min(ct,ht)<=0:return None
    return float(np.log(ct/ht))


def _lower(values,z,floor):
    if not values:return -math.inf
    values=np.asarray(values,float)
    uncertainty=max(float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else floor,floor)
    return float(values.mean()-z*uncertainty)


def fit_which_table(training_rows, validation_rows, *, branches=BRANCHES, minimum_cases=3, margin=.05, uncertainty_z=1.96, uncertainty_floor=.02):
    """Fit only train utilities; separately require positive validation evidence.

    Any failure or absence of actual neural work makes that bucket abstain. Each
    operator contributes one median, not artificially independent timing repeats.
    """
    if minimum_cases<1 or not 0<=margin<1 or uncertainty_z<0 or uncertainty_floor<=0:
        raise ValueError('Invalid conservative policy thresholds')
    if any(r.get('split')!='train' for r in training_rows) or any(r.get('split')!='validation' for r in validation_rows):
        raise ValueError('WHICH table requires declared train and validation rows')
    train_groups={r['normalized_operator_digest'] for r in training_rows}
    val_groups={r['normalized_operator_digest'] for r in validation_rows}
    if len(train_groups)!=len(training_rows) or len(val_groups)!=len(validation_rows) or train_groups&val_groups:
        raise ValueError('WHICH operator split overlap or duplication')
    table={};threshold=math.log(1+margin)
    for bucket in sorted({r['bucket'] for r in training_rows}):
        tr=[r for r in training_rows if r['bucket']==bucket];vr=[r for r in validation_rows if r['bucket']==bucket]
        table[bucket]={}
        for branch in branches:
            entry={}
            for cached in (False,True):
                for rhs in RHS_COUNTS:
                    tv=[_utility(r,branch,rhs,cached) for r in tr];vv=[_utility(r,branch,rhs,cached) for r in vr]
                    complete=bool(len(tv)>=minimum_cases and len(vv)>=minimum_cases and all(v is not None for v in tv+vv))
                    tl=_lower(tv,uncertainty_z,uncertainty_floor) if complete else -math.inf
                    vl=_lower(vv,uncertainty_z,uncertainty_floor) if complete else -math.inf
                    entry[f'{"cached" if cached else "cold"}:{rhs}']=dict(eligible=bool(complete and min(tl,vl)>threshold),
                        train_cases=len(tr),validation_cases=len(vr),train_lower_log_speedup=tl if np.isfinite(tl) else None,
                        validation_lower_log_speedup=vl if np.isfinite(vl) else None,
                        utility=min(tl,vl) if complete else None,
                        reason='supported' if complete and min(tl,vl)>threshold else 'insufficient_or_unfavorable_development_evidence')
            table[bucket][branch]=entry
    return table


@dataclass
class ResearchPolicy:
    models: dict
    table: dict
    rules_digest: str
    config_scope: dict
    hardware: dict
    signatures: dict
    provenance: dict=field(default_factory=dict)
    final_certificate: dict | None=None
    _evidence_cache: dict=field(default_factory=dict,init=False,repr=False)

    def _verified_evidence(self, entry):
        if not isinstance(entry,dict) or not isinstance(entry.get('sha256'),str) or len(entry['sha256'])!=64:
            raise ValueError('Final certificate needs a pinned evidence file')
        path=Path(entry['path']).resolve();stat=path.stat()
        identity=(str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,entry['sha256'])
        if identity not in self._evidence_cache:
            payload=path.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=entry['sha256']:
                raise ValueError('Final certificate evidence file hash mismatch')
            self._evidence_cache[identity]=json.loads(payload)
        return self._evidence_cache[identity]

    def _check_certificate_contract(self, certificate):
        if not isinstance(certificate,dict) or certificate.get('schema')!=1 or certificate.get('validated') is not True:
            raise ValueError('Untouched-final certificate is absent or unvalidated')
        expected=dict(policy_digest=self.digest(),rules_digest=self.rules_digest,
                      config_scope=self.config_scope,hardware=self.hardware)
        if any(certificate.get(k)!=v for k,v in expected.items()):
            raise ValueError('Final certificate policy/rules/config/hardware mismatch')
        if any(self.signatures.get(b)!=m.signature() for b,m in self.models.items()):
            raise ValueError('Final certificate expert/controller weights changed')
        freeze=self._verified_evidence(certificate.get('final_freeze'))
        result=self._verified_evidence(certificate.get('final_result'))
        if freeze.get('rules_digest')!=self.rules_digest:
            raise ValueError('Final freeze uses different classical rules')
        acceptance=certificate.get('acceptance')
        if not isinstance(acceptance,dict) or result.get('acceptance')!=acceptance:
            raise ValueError('Final certificate acceptance differs from pinned assessment')
        counts=('final_cases','repeats','strong_successes','neural_successes','new_failures','neural_used_cases')
        if any(isinstance(acceptance.get(k),bool) or not isinstance(acceptance.get(k),int) for k in counts):
            raise ValueError('Final certificate needs integer success/sample counts')
        total=acceptance['final_cases'];strong=acceptance['strong_successes'];neural=acceptance['neural_successes']
        if (acceptance.get('same_fp64_tolerance') is not True or total<20 or acceptance['repeats']<5
                or not 0<strong<=neural<=total or acceptance['new_failures']!=0
                or not 0<acceptance['neural_used_cases']<=neural):
            raise ValueError('Final certificate does not meet tolerance/success/sample/neural-use criteria')
        minimum=float(acceptance['minimum_speedup']);speed=float(acceptance['geometric_speedup'])
        ci=list(acceptance['ci95'])
        required=max(1.03,1./(1.-float(self.config_scope['win_margin'])))
        if (len(ci)!=2 or not np.isfinite([minimum,speed,*ci]).all() or minimum<required
                or speed<minimum or not 1.<float(ci[0])<=float(ci[1])):
            raise ValueError('Final certificate does not meet the speed/uncertainty criteria')
        if any(is_teacher(m.smoother) or is_teacher(m.transfer) for m in self.models.values()):
            raise ValueError('Teacher cannot be certified for deployment')
        return True

    def attach_final_certificate(self, certificate):
        """Export a separate certified copy; never mutate frozen policy/checkpoints.

        The strict final-run certifier establishes provenance and acceptance.
        This second guard verifies its pinned evidence and deployed identity.
        A local artifact is not a cryptographically authenticated certificate.
        """
        result=deepcopy(self)
        result.final_certificate=deepcopy(certificate)
        result._evidence_cache={}
        result._check_certificate_contract(result.final_certificate)
        return result

    def certificate_valid(self,cfg,rules,hardware=None):
        try:
            env=hardware_environment() if hardware is None else hardware
            if self.config_scope!=_scope(cfg) or self.rules_digest!=rules.digest() or self.hardware!=env:
                return False
            if not self.models or any(not self.valid_branch(b,cfg,rules,env) for b in self.models):
                return False
            return self._check_certificate_contract(self.final_certificate)
        except (ValueError,TypeError,KeyError,OSError,OverflowError):
            return False

    def valid_branch(self,branch,cfg,rules,hardware=None):
        if branch not in self.models:return False
        model=self.models[branch]
        return bool(self.rules_digest==rules.digest() and self.config_scope==_scope(cfg)
                    and self.hardware==(hardware_environment() if hardware is None else hardware)
                    and self.signatures.get(branch)==model.signature()
                    and model.metadata.get('controller_trained') and not model.metadata.get('policy_stale')
                    and not is_teacher(model.smoother) and not is_teacher(model.transfer))

    def choose(self,selection,shape,cfg,rules,*,expected_rhs=1,cached_branches=(),hardware=None):
        if isinstance(expected_rhs,bool) or not isinstance(expected_rhs,int) or expected_rhs<1:
            raise ValueError('expected_rhs must be a positive integer')
        rhs=max(k for k in RHS_COUNTS if k<=expected_rhs)
        bucket=_bucket(selection,shape);candidates=[]
        env=hardware_environment() if hardware is None else hardware
        if cfg.mode=='production' and not self.certificate_valid(cfg,rules,env):
            return 'C',dict(bucket=bucket,expected_rhs=expected_rhs,table_rhs=rhs,candidate_count=0,
                            cache_available=list(cached_branches),reason='invalid_untouched_final_certificate')
        for branch,entries in self.table.get(bucket,{}).items():
            if not self.valid_branch(branch,cfg,rules,env):continue
            key=f'{"cached" if branch in cached_branches else "cold"}:{rhs}'
            evidence=entries.get(key,{})
            if evidence.get('eligible') and np.isfinite(evidence.get('utility',float('nan'))):
                candidates.append((float(evidence['utility']),branch))
        selected=max(candidates)[1] if candidates else 'C'
        return selected,dict(bucket=bucket,expected_rhs=expected_rhs,table_rhs=rhs,
                             cache_available=list(cached_branches),candidate_count=len(candidates),
                             reason='validation_supported_utility' if candidates else 'conservative_classical_abstention')

    def digest(self):
        return _hash(dict(table=self.table,rules_digest=self.rules_digest,config_scope=self.config_scope,
                          hardware=self.hardware,signatures=self.signatures,provenance=self.provenance))

    def save(self,path):
        path=Path(path)
        if (path/'policy.json').exists():raise FileExistsError('Policy artifact already exists')
        path.mkdir(parents=True,exist_ok=True)
        for branch,model in self.models.items():model.save(path/(branch+'.pt'))
        write_json(path/'policy.json',dict(schema=1,table=self.table,rules_digest=self.rules_digest,
            config_scope=self.config_scope,hardware=self.hardware,signatures=self.signatures,provenance=self.provenance,
            branches=list(self.models),final_certificate=self.final_certificate,production_certified=self.final_certificate is not None))

    @classmethod
    def load(cls,path):
        path=Path(path);record=json.loads((path/'policy.json').read_text())
        if record.get('schema')!=1 or set(record['branches'])-set(BRANCHES):raise ValueError('Unsupported research policy')
        models={b:Components.load(path/(b+'.pt')) for b in record['branches']}
        if any(record['signatures'].get(b)!=m.signature() for b,m in models.items()):raise ValueError('Policy component signature mismatch')
        return cls(models,record['table'],record['rules_digest'],record['config_scope'],record['hardware'],record['signatures'],record['provenance'],record.get('final_certificate'))


def refit_research_policy(branch_models,train_examples,validation_examples,cfg,rules,settings,out,*,dataset_manifest=None):
    """Offline numerical labels from new TRAIN C*(A); validation gates deployment."""
    if not branch_models or set(branch_models)-set(BRANCHES):raise ValueError('Provide explicit H_S/H_P/H_SP Student experts')
    _validate_development(train_examples,validation_examples,rules,dataset_manifest)
    if any(is_teacher(m.smoother) or is_teacher(m.transfer) for m in branch_models.values()):
        raise ValueError('Teacher is not a deployable policy expert')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Policy output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    models={b:deepcopy(m) for b,m in branch_models.items()}
    _fit_shared_detector(models,train_examples,cfg,rules,settings,out)
    for branch,model in models.items():
        torch.manual_seed(int(settings.get('seed',71)))
        model.controller=TinyNet(CONTROLLER_INPUTS,4)
        rows=[]
        for index,e in enumerate(train_examples):
            selection=_selection(e,rules)
            chosen=replace(_branch_config(cfg,branch),mg=replace(cfg.mg,strategy_name=selection.strategy_name))
            case_rows=temporal_table(model,[e],chosen,settings,out/branch/f'case_{index:04d}')
            for row in case_rows:row.update(strong_strategy=selection.strategy_name,strong_rules_digest=rules.digest(),split='train')
            rows.extend(case_rows)
        fit_controller(model,rows,settings,out/branch/'temporal_fit')
        model.metadata['research_policy_refit']=True
        # The legacy trainer compares the exact current expert signatures; this
        # adapter's independent signature records additionally pin architectures.
        model.metadata['policy_stale']=not bool(rows)
        write_json(out/branch/'strong_action_table.json',rows)
    training=collect_which_rows(train_examples,models,cfg,rules,settings,split='train')
    validation=collect_which_rows(validation_examples,models,cfg,rules,settings,split='validation')
    write_json(out/'which_train.json',training);write_json(out/'which_validation.json',validation)
    table=fit_which_table(training,validation,branches=tuple(models),minimum_cases=int(settings.get('minimum_policy_bucket_cases',3)),
                         margin=float(settings.get('policy_speedup_margin',cfg.win_margin)),
                         uncertainty_z=float(settings.get('policy_uncertainty_z',1.96)),
                         uncertainty_floor=float(settings.get('policy_uncertainty_floor',.02)))
    policy=ResearchPolicy(models,table,rules.digest(),_scope(cfg),hardware_environment(),{b:m.signature() for b,m in models.items()},
        dict(development_manifest_digest=_hash(dataset_manifest),training_operators=[e.digest for e in train_examples],
             validation_operators=[e.digest for e in validation_examples],
             method='A-only rule/grid utility table + branch temporal TinyNets + shared spatial TinyNet',
             utility_scope='measured cold/warm solves; RHS totals extrapolated for offline labels, not multiple-RHS benchmark',
             no_online_shadow_solves=True,final_test_used=False,production_certified=False))
    policy.save(out/'deployable_policy')
    return policy


class PreparedResearchMG(PreparedStrongMG):
    """Conservative WHICH adapter; existing WHEN, WHERE and LOCK remain intact."""
    def __init__(self,a,n,components=None,config=None,rules=None,*,policy=None,expected_rhs=1):
        self.policy=policy;self.expected_rhs=expected_rhs;self._selected_branch='C';self._which_info={}
        self._policy_hardware=hardware_environment();self._auto_requested=config is None or config.branch=='auto'
        super().__init__(a,n,components,config,rules)

    def _refresh_key(self):
        super()._refresh_key()
        policy_key=self.policy.digest() if self.policy is not None else 'none'
        final_key=_hash(self.policy.final_certificate) if self.policy is not None and self.policy.final_certificate else 'none'
        self.cache_key=_hash((self.cache_key,policy_key,self._selected_branch,self.expected_rhs,final_key))

    def _branch(self):
        return self._selected_branch if self._auto_requested else super()._branch()

    def _effective_config(self):
        cfg=super()._effective_config()
        if not self._auto_requested:return cfg
        return replace(cfg,use_smoother=self._selected_branch in {'H_S','H_SP'},use_transfer=self._selected_branch in {'H_P','H_SP'})

    def _potential_neural(self):
        if self.config.mode!='production':return super()._potential_neural()
        cfg=self._effective_config()
        return bool(self._auto_requested and self.components is not None and self._selected_branch!='C'
                    and cfg.gate_mode!='closed' and cfg.mg.nn_levels!=0
                    and (cfg.use_smoother or cfg.use_transfer) and self.certificate_valid())

    def certificate_valid(self):
        return bool(self._auto_requested and self.policy is not None
                    and self.policy.certificate_valid(self.config,self.rules,self._policy_hardware))

    def _ensure_fresh(self):
        super()._ensure_fresh()
        if not self._auto_requested:return
        self._policy_hardware=hardware_environment()
        if self.policy is None or self.config.mode=='classical':
            selected='C';info=dict(reason='policy_unavailable_or_uncertified_production')
        else:
            cached={key[0] for key in self.branch_banks}
            selected,info=self.policy.choose(self.selection,self.shape,self.config,self.rules,
                expected_rhs=self.expected_rhs,cached_branches=cached,hardware=self._policy_hardware)
        self._selected_branch=selected;self._which_info=info
        self.components=None if selected=='C' else self.policy.models[selected]
        super()._ensure_fresh()
        self._refresh_key()

    def _policy_status(self,cfg,branch):
        if not self._auto_requested:return super()._policy_status(cfg,branch)
        if branch=='C':return 'not_applicable'
        if self.policy and self.policy.valid_branch(branch,cfg,self.rules,self._policy_hardware):return 'calibrated_branch_policy'
        return 'stale_strong_research_policy'

    def _solve(self,b,x0=None):
        result=super()._solve(b,x0)
        result.abstention['research_which']=dict(self._which_info)
        result.abstention['research_selected_branch']=self._selected_branch if self._auto_requested else self._branch()
        result.abstention['strong_policy_calibrated']=bool(result.branch_policy_status=='calibrated_branch_policy')
        result.abstention['policy_final_certified']=bool(result.certificate_used)
        return result

    def solve_many(self,bs,x0=None):
        before=self.expected_rhs
        try:
            self.expected_rhs=len(bs)
            return super().solve_many(bs,x0)
        finally:self.expected_rhs=before
