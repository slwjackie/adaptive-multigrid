"""Stage promotion is paired, failure-aware, and distinct from final audit."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from adaptive_mg import MGConfig
from adaptive_mg.v67 import AdaptiveConfig, Components
from adaptive_mg.v67.acceptance import evaluate_stage_acceptance
from adaptive_mg.v67.evaluation import certify


@pytest.fixture
def comparison():
    torch.set_num_threads(1)
    incumbent=Components.create(hidden=4,seed=5)
    candidate=deepcopy(incumbent)
    with torch.no_grad():next(candidate.smoother.parameters()).add_(.1)
    cfg=AdaptiveConfig(mg=MGConfig(mode='classical',max_cycles=10,tolerance=1e-8),
                       mode='hybrid',use_transfer=False,spatial=False,gate_mode='open')
    return incumbent,candidate,cfg


def run(success=True,seconds=1.,neural=True):
    residual=1e-10 if success else 1.
    return {'converged':success,'stopping_threshold':1e-8,
            'final_true_residual':residual,'executed_cycles':2,'cycles':2,
            'residual_history':[1.,residual],'relative_residual_history':[1.,residual],
            'wall_seconds':seconds,'time_scope':'cold',
            'stats':{'accepted_neural_cycles':int(neural),
                     'attempted_neural_cycles':int(neural),
                     'neural_apply_calls':int(neural),
                     'neural_setup_calls':int(neural)}}


def rows(total=6):
    return [{'example':{'name':f'case-{i}','case':{'n':7}},
             'runs':{'previous':[run(seconds=2.) for _ in range(3)],
                     'candidate':[run() for _ in range(3)]}} for i in range(total)]


def evaluate(data,comparison,settings=None,stage='smoother'):
    incumbent,candidate,cfg=comparison
    return evaluate_stage_acceptance(data,candidate,incumbent,cfg,settings,stage=stage)


def fail(data,index,arm):
    data[index]['runs'][arm]=[run(False,seconds=.001) for _ in range(3)]


def test_both_failed_case_allows_relative_promotion_but_strict_reproduces_rejection(comparison):
    data=rows()
    fail(data,5,'previous');fail(data,5,'candidate')
    report=evaluate(data,comparison)
    assert report['accepted'] and report['stage_promoted']
    assert report['common_successes']==5 and report['both_failed_case_ids']==['case-5']
    assert report['geometric_speedup_vs_incumbent']==2.
    assert report['required_speedup']==1/(1-.03)
    assert report['speedup_ci95']==[2.,2.]
    assert not report['final_time_improvement_certified']
    strict=evaluate(data,comparison,{'stage_acceptance_policy':'strict_v1'})
    assert not strict['accepted']
    assert strict['failed_conditions']==['all_common_success']


def test_equal_success_counts_cannot_hide_lost_case(comparison):
    data=rows(7)
    fail(data,5,'candidate');fail(data,6,'previous')
    report=evaluate(data,comparison)
    assert report['incumbent_successes']==report['candidate_successes']==6
    assert not report['accepted']
    assert report['new_failure_case_ids']==['case-5']
    assert report['rescued_case_ids']==['case-6']
    assert report['failed_conditions']==['no_new_failures']


def test_all_failed_and_no_common_has_no_time_improvement(comparison):
    data=rows()
    for i in range(len(data)):
        fail(data,i,'candidate');fail(data,i,'previous')
    report=evaluate(data,comparison)
    assert not report['accepted']
    assert report['common_successes']==0
    assert report['geometric_speedup_vs_incumbent'] is None
    assert report['speedup_ci95']==[None,None]
    assert not report['checks']['actual_neural_use']
    assert all(row['candidate_successful_seconds'] is None for row in report['case_results'])


def test_metadata_signature_and_unrelated_weights_do_not_count(comparison):
    incumbent,_,cfg=comparison
    candidate=deepcopy(incumbent)
    candidate.metadata.update(signature='new',smoother_trained=True)
    candidate.signature=lambda:'spoofed-signature'
    with torch.no_grad():candidate.controller.fc2.weight.add_(1.)
    report=evaluate(rows(),(incumbent,candidate,cfg))
    assert not report['checks']['model_weights_changed']
    assert report['changed_component_signatures']['controller']['parameters_changed']
    assert not report['accepted']


def test_real_tensor_change_counts_even_if_signature_method_claims_equal(comparison):
    incumbent,candidate,_=comparison
    candidate.signature=lambda:incumbent.signature()
    assert evaluate(rows(),comparison)['accepted']


def test_minimum_common_count_and_fraction_are_both_enforced(comparison):
    data=rows(12)
    for i in range(5,12):
        fail(data,i,'previous');fail(data,i,'candidate')
    report=evaluate(data,comparison)
    assert report['minimum_common_samples']==6
    assert not report['accepted']
    assert not evaluate(rows(4),comparison)['accepted']
    smoke=evaluate(rows(2),comparison,{'stage_min_common_success':1})
    assert smoke['accepted'] and not smoke['performance_certified']


def test_partially_failed_repeats_are_not_cherry_picked(comparison):
    data=rows()
    data[0]['runs']['candidate']=[run(seconds=.001),run(False,seconds=50.),run(seconds=.001)]
    report=evaluate(data,comparison)
    assert not report['accepted'] and report['new_failure_case_ids']==['case-0']
    first=report['case_results'][0]
    assert first['candidate_successful_repeats']==2
    assert first['candidate_successful_seconds'] is None
    assert first['speedup_vs_incumbent'] is None
    assert report['geometric_speedup_vs_incumbent']==2.


def test_failed_only_or_setup_only_neural_use_cannot_support_promotion(comparison):
    data=rows()
    for row in data:
        for value in row['runs']['candidate']:
            value['stats']['neural_apply_calls']=0
    report=evaluate(data,comparison)
    assert report['neural_counts']['candidate']['neural_setup_calls']==18
    assert report['neural_used_cases']==6
    assert not report['checks']['actual_neural_use']
    assert not report['accepted']


@pytest.mark.parametrize('change',['residual','threshold','budget','time','repeat','protocol'])
def test_bad_measurement_evidence_cannot_promote(comparison,change):
    data=rows()
    value=data[0]['runs']['candidate'][0]
    if change=='residual':value['final_true_residual']=.1
    elif change=='threshold':value['stopping_threshold']=.2
    elif change=='budget':value['executed_cycles']=11
    elif change=='time':value['wall_seconds']=float('nan')
    elif change=='repeat':data[0]['runs']['candidate'].pop()
    else:value['time_scope']='warm'
    assert not evaluate(data,comparison)['accepted']


def test_final_certificate_remains_strict_when_stage_promotes(comparison):
    data=rows()
    fail(data,5,'previous');fail(data,5,'candidate')
    assert evaluate(data,comparison)['accepted']
    for row in data:
        row['runs']={'classical':row['runs']['previous'],'adaptive':row['runs']['candidate']}
    _,candidate,cfg=comparison
    certificate=certify(candidate,cfg,data,min_cases=5,margin=.03)
    assert not certificate['validated']
    assert certificate['audit_summary']['common_successes']==5


def test_transfer_application_uses_p_counter_or_recorded_legacy_coarse_path(comparison):
    incumbent,candidate,cfg=comparison
    cfg=replace(cfg,use_transfer=True,use_smoother=False)
    with torch.no_grad():next(candidate.transfer.parameters()).add_(.1)
    data=rows()
    for row in data:
        for value in row['runs']['candidate']:
            value['stats'].update(neural_apply_calls=0,transfer_nn_calls=1,coarse_path_count=2)
    assert evaluate(data,(incumbent,candidate,cfg),stage='transfer')['accepted']
    for row in data:
        for value in row['runs']['candidate']:value['stats']['coarse_path_count']=0
    assert not evaluate(data,(incumbent,candidate,cfg),stage='transfer')['accepted']


def test_duplicate_case_id_rejected(comparison):
    data=rows();data[1]['example']['name']=data[0]['example']['name']
    with pytest.raises(ValueError,match='unique'):
        evaluate(data,comparison)


def test_mark_policy_stale_preserves_weights_and_invalidates_evidence(comparison):
    _,model,_=comparison
    signatures=model.component_signatures()
    model.metadata['certificate']={'validated':True}
    model.mark_policy_stale()
    assert model.component_signatures()==signatures
    assert model.metadata['policy_stale']
    assert not model.metadata['certificate']['validated']


def test_fresh_measured_runs_use_operator_identity_not_opaque_model_cache_key(tmp_path,comparison):
    import json
    from adaptive_mg.v67.data import dataset
    from adaptive_mg.v67.training import accept_operator
    incumbent,candidate,cfg=comparison
    cfg=replace(cfg,branch='H_S',mg=replace(cfg.mg,nn_levels=1,max_cycles=12,
                                         strategy_name='jacobi_bilinear_full'))
    examples=dataset([7],1,72)
    _,report=accept_operator(candidate,incumbent,examples,cfg,
                            {'validation_repeats':1,'stage_min_common_success':1},
                            tmp_path,stage='smoother')
    data=json.loads((tmp_path/'raw.json').read_text())
    previous=data[0]['runs']['previous'][0]
    new=data[0]['runs']['candidate'][0]
    assert previous['cache_key'] != new['cache_key']
    assert previous['operator_digest']==new['operator_digest']
    assert report['checks']['same_measurement_protocol']
    assert report['checks']['complete_repeats']
    assert report['protocol_sources']==['recorded_measurement_config']


def test_existing_candidate_cannot_be_overwritten_by_new_training(tmp_path,comparison):
    from adaptive_mg.v67.training import operator_stage
    _,candidate,cfg=comparison
    path=tmp_path/'candidate.pt'
    candidate.save(path)
    saved=path.read_bytes()
    with pytest.raises(FileExistsError,match='candidate already preserved'):
        operator_stage(candidate,[],cfg,{'epochs':{'smoother':0}},'smoother',tmp_path)
    assert path.read_bytes()==saved


def test_rhs_reference_zero_initial_residual_keeps_real_rhs_tolerance(comparison):
    incumbent,candidate,cfg=comparison
    cfg=replace(cfg,mg=replace(cfg.mg,residual_reference='rhs'))
    data=rows()
    for row in data:
        for arm in row['runs'].values():
            for value in arm:
                value.update(residual_history=[0.],relative_residual_history=[0.],
                             final_true_residual=0.,residual_reference_norm=1.)
    report=evaluate(data,(incumbent,candidate,cfg))
    assert report['common_successes']==6
    assert report['checks']['same_measurement_protocol']
