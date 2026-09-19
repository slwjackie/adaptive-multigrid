"""Policy fits use new strong-aware labels and abstain outside validated scope."""
from dataclasses import replace
from copy import deepcopy
import hashlib

import numpy as np
import pytest
import torch

from adaptive_mg.config import MGConfig
from adaptive_mg.data import make_examples
from adaptive_mg.provenance import hardware_environment,operator_digest
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.strong import StrongRules,PreparedStrongMG,select_strong_strategy
from adaptive_mg.v67.research_policy import (ResearchPolicy,PreparedResearchMG,fit_which_table,
    refit_research_policy,_scope,_bucket,_validate_development)


def examples(seed,count=2):
    result=make_examples([7],per_size=count,seed=seed)
    rules=StrongRules()
    for e in result:e.strong_selection=select_strong_strategy(e.a,e.n,rules).to_dict()
    return result


def manifest(train,validation,rules):
    def record(e):
        return dict(operator_digest=e.digest,normalized_operator_digest=e.group_digest,
                    rhs_digest=hashlib.sha256(e.b.tobytes()).hexdigest())
    return dict(rules_digest=rules.digest(),splits=dict(train=[record(e) for e in train],
                validation=[record(e) for e in validation]),historical_index=dict(normalized_operator_digests=[]))


def utility_row(name,split='train',*,ratio=2.,success=True,used=True,bucket='near_isotropic|n=7'):
    def run(cold,warm,used):return dict(cold_seconds=cold,warm_seconds=warm,success=success,neural_used=used)
    return dict(split=split,operator=name,normalized_operator_digest=name,bucket=bucket,
                runs={'C':[run(2,1,False)],'H_S':[run(2/ratio,1/ratio,used)]})


def policy_for(e,cfg,rules):
    model=Components.create(hidden=4)
    model.metadata.update(controller_trained=True,policy_stale=False,detector_trained=False)
    # Force a tiny controller's positive prediction so tests exercise the real
    # cached branch and LOCK path; production acceptance is never claimed.
    with torch.no_grad():
        model.controller.fc2.bias.copy_(torch.tensor([2.,-2.,0.,20.]))
    selection=select_strong_strategy(e.a,e.n,rules)
    bucket=_bucket(selection,e.n)
    train=[utility_row('t'+str(i),bucket=bucket) for i in range(3)]
    validation=[utility_row('v'+str(i),'validation',bucket=bucket) for i in range(3)]
    table=fit_which_table(train,validation,branches=('H_S',))
    return ResearchPolicy({'H_S':model},table,rules.digest(),_scope(cfg),hardware_environment(),{'H_S':model.signature()})


def config(**kwargs):
    return AdaptiveConfig(mg=MGConfig(mode='classical',max_cycles=50,nn_levels=1,pre_steps=2,post_steps=2,
                                     native_threads=1),mode='research',branch='auto',spatial=False,gate_mode='open',**kwargs)


def test_which_requires_unseen_validation_success_neural_use_and_uncertainty():
    train=[utility_row('t'+str(i)) for i in range(3)]
    val=[utility_row('v'+str(i),'validation') for i in range(3)]
    def entry(v):return fit_which_table(train,v,branches=('H_S',))['near_isotropic|n=7']['H_S']['cold:1']
    assert entry(val)['eligible']
    assert not entry(val[:1])['eligible']
    bad=deepcopy(val);bad[0]['runs']['H_S'][0]['success']=False
    assert not entry(bad)['eligible']
    bad=deepcopy(val);bad[0]['runs']['H_S'][0]['neural_used']=False
    assert not entry(bad)['eligible']
    weak=[utility_row('v'+str(i),'validation',ratio=1.01) for i in range(3)]
    assert not entry(weak)['eligible']
    overlap=deepcopy(val);overlap[0]['normalized_operator_digest']='t0'
    with pytest.raises(ValueError,match='overlap'):entry(overlap)


def test_expected_rhs_and_cache_are_deployment_inputs_not_case_labels():
    e=examples(110)[0];cfg=config();rules=StrongRules();policy=policy_for(e,cfg,rules)
    selected=select_strong_strategy(e.a,e.n,rules)
    for rhs in (1,4,16,64):
        branch,info=policy.choose(selected,e.n,cfg,rules,expected_rhs=rhs,cached_branches=('H_S',))
        assert branch=='H_S' and info['table_rhs']==rhs
    assert policy.choose(selected,31,cfg,rules)[0]=='C'
    with pytest.raises(ValueError):policy.choose(selected,e.n,cfg,rules,expected_rhs=0)


def test_policy_stale_expert_config_rules_hardware_abstains():
    e=examples(111)[0];cfg=config();rules=StrongRules();policy=policy_for(e,cfg,rules)
    selection=select_strong_strategy(e.a,e.n,rules)
    assert policy.choose(selection,e.n,cfg,rules)[0]=='H_S'
    assert policy.choose(selection,e.n,replace(cfg,replace_pre=2),rules)[0]=='C'
    assert policy.choose(selection,e.n,cfg,rules,hardware={})[0]=='C'
    changed=rules.replace_strategies({'near_isotropic':'chebyshev_bilinear_full'},provenance='different')
    assert policy.choose(selection,e.n,cfg,changed)[0]=='C'
    with torch.no_grad():policy.models['H_S'].smoother.head.weight.add_(.01)
    assert policy.choose(selection,e.n,cfg,rules)[0]=='C'


def test_policy_save_load_is_independent_and_metadata_pins_cache(tmp_path):
    e=examples(112)[0];cfg=config();rules=StrongRules();policy=policy_for(e,cfg,rules)
    policy.save(tmp_path/'saved');restored=ResearchPolicy.load(tmp_path/'saved')
    assert policy.digest()==restored.digest()
    assert restored.models['H_S'] is not policy.models['H_S']
    prepared=PreparedResearchMG(e.a,e.n,config=cfg,rules=rules,policy=policy)
    old=prepared.cache_key
    policy.table={};prepared._ensure_fresh()
    assert prepared.cache_key!=old and prepared._selected_branch=='C'


def test_classical_abstention_preserves_exact_strong_path_and_no_generators():
    e=examples(113)[0];cfg=config();rules=StrongRules()
    expected=PreparedStrongMG(e.a,e.n,config=replace(cfg,branch='C'),rules=rules).solve(e.b)
    actual=PreparedResearchMG(e.a,e.n,config=cfg,rules=rules).solve(e.b)
    np.testing.assert_array_equal(expected.x,actual.x)
    assert actual.stats['smoother_nn_calls']==actual.stats['transfer_nn_calls']==0
    assert actual.abstention['research_selected_branch']=='C'
    assert not actual.certificate_used


def test_policy_auto_uses_once_generated_student_and_never_teacher_or_locked_calls():
    e=examples(114)[0];cfg=config();rules=StrongRules();policy=policy_for(e,cfg,rules)
    prepared=PreparedResearchMG(e.a,e.n,config=cfg,rules=rules,policy=policy)
    first=prepared.solve(e.b);second=prepared.solve(e.b)
    assert first.abstention['research_selected_branch']=='H_S'
    assert first.branch_policy_status=='calibrated_branch_policy'
    assert first.stats['smoother_nn_calls']==1
    assert second.stats['smoother_nn_calls']==0
    assert first.stats['transfer_nn_calls']==second.stats['transfer_nn_calls']==0
    locks=[v for v in first.trace if v.get('state')=='CLASSICAL_LOCK']
    if locks:
        locked_cycle=locks[0]['cycle']
        assert all(v.get('path')!='hybrid' for v in first.trace if v['cycle']>locked_cycle)
    assert not first.certificate_used
    prod=PreparedResearchMG(e.a,e.n,config=replace(cfg,mode='production'),rules=rules,policy=policy).solve(e.b)
    assert prod.stats['smoother_nn_calls']==0


def test_committed_split_selection_and_rhs_are_required():
    train=examples(115);val=examples(116);rules=StrongRules();m=manifest(train,val,rules)
    _validate_development(train,val,rules,m)
    with pytest.raises(ValueError,match='manifest'):_validate_development(train,val,rules,None)
    changed=deepcopy(m);changed['splits']['final']=[]
    with pytest.raises(ValueError,match='final'):_validate_development(train,val,rules,changed)
    changed=deepcopy(m);changed['historical_index']['normalized_operator_digests']=[train[0].group_digest]
    with pytest.raises(ValueError,match='leakage'):_validate_development(train,val,rules,changed)
    train[0].b[0]+=.01
    with pytest.raises(ValueError,match='RHS'):_validate_development(train,val,rules,m)


def test_tiny_actual_refit_uses_selected_strategy_and_independent_controllers(tmp_path):
    torch.set_num_threads(1)
    train=examples(117,count=1);val=examples(118,count=1);rules=StrongRules();cfg=config()
    initial=Components.create(hidden=4);before=initial.signature()
    settings=dict(seed=11,detector_states=1,controller_states=1,timing_repeats=1,
                  epochs=dict(detector=1,controller=1),minimum_policy_bucket_cases=3)
    policy=refit_research_policy({'H_S':initial},train,val,cfg,rules,settings,tmp_path/'refit',dataset_manifest=manifest(train,val,rules))
    assert initial.signature()==before
    assert policy.models['H_S'] is not initial
    assert policy.provenance['final_test_used'] is False
    assert (tmp_path/'refit/H_S/strong_action_table.json').exists()
    import json
    rows=json.loads((tmp_path/'refit/H_S/strong_action_table.json').read_text())
    assert rows and all(row['strong_strategy']==train[0].strong_selection['strategy_name'] for row in rows)
    assert all(row['split']=='train' for row in rows)
    assert policy.choose(select_strong_strategy(val[0].a,val[0].n,rules),val[0].n,cfg,rules)[0]=='C'


def final_certificate_fixture(policy,tmp_path):
    """Synthetic certificate documents for runtime guarding, not final research."""
    import json
    acceptance=dict(same_fp64_tolerance=True,final_cases=20,repeats=5,strong_successes=20,
                    neural_successes=20,new_failures=0,neural_used_cases=20,
                    geometric_speedup=1.3,ci95=[1.2,1.4],minimum_speedup=1/(1-policy.config_scope['win_margin']))
    freeze=tmp_path/'freeze.json';result=tmp_path/'assessment.json'
    freeze.write_text(json.dumps(dict(rules_digest=policy.rules_digest)))
    result.write_text(json.dumps(dict(acceptance=acceptance)))
    def pin(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(schema=1,validated=True,policy_digest=policy.digest(),rules_digest=policy.rules_digest,
                config_scope=deepcopy(policy.config_scope),hardware=deepcopy(policy.hardware),
                final_freeze=pin(freeze),final_result=pin(result),acceptance=acceptance)


def test_final_certificate_exports_copy_and_enables_only_matching_production(tmp_path):
    e=examples(191)[0];cfg=config();rules=StrongRules();policy=policy_for(e,cfg,rules)
    certificate=final_certificate_fixture(policy,tmp_path)
    certified=policy.attach_final_certificate(certificate)
    assert policy.final_certificate is None
    assert certified is not policy and certified.models['H_S'] is not policy.models['H_S']
    assert certified.digest()==policy.digest()
    production=replace(cfg,mode='production')
    assert certified.certificate_valid(production,rules)
    certified.save(tmp_path/'certified_export')
    restored=ResearchPolicy.load(tmp_path/'certified_export')
    assert restored.certificate_valid(production,rules)
    prepared=PreparedResearchMG(e.a,e.n,config=production,rules=rules,policy=restored)
    first=prepared.solve(e.b);second=prepared.solve(e.b)
    assert first.certificate_used and first.abstention['policy_final_certified']
    assert first.stats['smoother_nn_calls']==1 and second.stats['smoother_nn_calls']==0
    assert first.stats['transfer_nn_calls']==0
    assert first.abstention['research_selected_branch']=='H_S'
    assert restored.models['H_S'].metadata['certificate']['validated'] is False


@pytest.mark.parametrize('field',['policy_digest','rules_digest','config_scope','hardware','sha256','acceptance'])
def test_forged_or_mismatched_final_certificate_is_blocked(tmp_path,field):
    e=examples(192)[0];cfg=config();rules=StrongRules();policy=policy_for(e,cfg,rules)
    certificate=final_certificate_fixture(policy,tmp_path)
    if field=='sha256':certificate['final_result']['sha256']='0'*64
    elif field=='acceptance':certificate['acceptance']['geometric_speedup']=.9
    else:certificate[field]='forged'
    with pytest.raises(ValueError):policy.attach_final_certificate(certificate)
    # Loading an externally changed artifact cannot make the runtime trust it.
    policy.final_certificate=certificate
    assert not policy.certificate_valid(replace(cfg,mode='production'),rules)
    result=PreparedResearchMG(e.a,e.n,config=replace(cfg,mode='production'),rules=rules,policy=policy).solve(e.b)
    assert not result.certificate_used and result.stats['smoother_nn_calls']==0


def test_modified_final_evidence_or_policy_weights_revoke_production(tmp_path):
    e=examples(193)[0];cfg=config();rules=StrongRules();policy=policy_for(e,cfg,rules)
    certificate=final_certificate_fixture(policy,tmp_path)
    certified=policy.attach_final_certificate(certificate)
    assert certified.certificate_valid(cfg,rules)
    path=tmp_path/'assessment.json';path.write_text(path.read_text()+' ')
    assert not certified.certificate_valid(cfg,rules)
    # Restoring bytes restores evidence, but changed numerical weights still fail.
    path.write_text(path.read_text().rstrip())
    assert certified.certificate_valid(cfg,rules)
    with torch.no_grad():certified.models['H_S'].smoother.head.weight.add_(.01)
    assert not certified.certificate_valid(cfg,rules)


def test_passing_final_certificate_never_enables_teacher_or_wrong_runtime_scope(tmp_path):
    e=examples(194)[0];cfg=config();rules=StrongRules();policy=policy_for(e,cfg,rules)
    certified=policy.attach_final_certificate(final_certificate_fixture(policy,tmp_path))
    assert not certified.certificate_valid(replace(cfg,inference_dtype='float64'),rules)
    assert not certified.certificate_valid(cfg,rules,hardware={})
    certified.models['H_S'].smoother.training_only=True
    assert not certified.certificate_valid(cfg,rules)
