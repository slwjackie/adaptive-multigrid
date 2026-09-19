import json
from dataclasses import replace
from pathlib import Path
import pytest

from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.strong import StrongRules
from adaptive_mg.v67.strong_calibration import calibrate_leaf_mapping, calibrate_from_run


def measured(strategy, seconds, success=True):
    cfg = AdaptiveConfig(mode='classical', branch='C')
    cfg = replace(cfg, mg=replace(cfg.mg, strategy_name=strategy))
    return dict(measurement_config=cfg.to_dict(), converged=success, wall_seconds=seconds,
        stopping_threshold=cfg.mg.tolerance, final_true_residual=cfg.mg.tolerance/2 if success else 1.,
        recomputed_true_residual=cfg.mg.tolerance/2 if success else 1.,
        residual_history=[1., cfg.mg.tolerance/2 if success else 1.],
        relative_residual_history=[1., cfg.mg.tolerance/2 if success else 1.], executed_cycles=4,
        operator_digest='operator', rhs_digest='rhs', x0_digest='zero', stats={})


def row(index, rules, fixed=True, alternative=True):
    return dict(example=dict(name=f'case{index}', operator_digest='operator', normalized_operator_digest=f'normal{index}',
                             case={'n':7}, seed=11, scale=1.),
        runs={
            'fixed_C':[measured('line_alt_bilinear_full', 2., fixed)],
            'strong_C':[dict(measured('jacobi_bilinear_full',4.,fixed),selection=dict(rule_id='near_isotropic',rules_digest=rules.digest()))],
            'oracle::jacobi_bilinear_full':[measured('jacobi_bilinear_full',4.,fixed)],
            'oracle::line_alt_bilinear_full':[measured('line_alt_bilinear_full',1.,alternative)],
        })


def test_offline_mapping_changes_leaf_only_and_never_uses_case_names():
    rules = StrongRules(); rows = [row(i,rules) for i in range(3)]
    selected, reports = calibrate_leaf_mapping(rows,rules,repeats=1)
    assert selected['near_isotropic']=='line_alt_bilinear_full'
    assert reports['near_isotropic']['samples']==3
    updated=rules.replace_strategies(selected)
    assert updated.anisotropy_min==rules.anisotropy_min
    assert updated.alignment_angle_deg==rules.alignment_angle_deg
    for i,r in enumerate(rows):r['example']['name']=f'arbitrary_label{i}'
    assert calibrate_leaf_mapping(rows,rules,repeats=1)[0]==selected


def test_one_new_failure_or_success_swap_cannot_buy_speed():
    rules=StrongRules()
    rows=[row(0,rules,True,False),row(1,rules,False,True),row(2,rules)]
    mapping,report=calibrate_leaf_mapping(rows,rules,repeats=1)
    assert mapping['near_isotropic']=='jacobi_bilinear_full'
    other=next(c for c in report['near_isotropic']['candidates'] if c['strategy']=='line_alt_bilinear_full')
    assert not other['eligible'] and other['new_failure_case_ids']==['case0']


def test_undersampled_leaf_and_all_failed_runtime_keep_prior():
    rules=StrongRules()
    assert calibrate_leaf_mapping([row(0,rules)],rules,repeats=1)[0]==dict(rules.strategy_by_rule)
    rows=[row(i,rules,False,False) for i in range(3)]
    mapping,report=calibrate_leaf_mapping(rows,rules,repeats=1)
    assert mapping['near_isotropic']=='jacobi_bilinear_full'
    assert report['near_isotropic']['timing_selection']=='not_available_no_fixed_success'


def test_partial_repeat_and_stale_rule_evidence_rejected():
    rules=StrongRules();rows=[row(i,rules) for i in range(3)]
    rows[0]['runs']['oracle::line_alt_bilinear_full']=[]
    with pytest.raises(ValueError,match='partial'):
        calibrate_leaf_mapping(rows,rules,repeats=1)
    rows=[row(i,rules) for i in range(3)]
    rows[0]['runs']['strong_C'][0]['selection']['rules_digest']='stale'
    with pytest.raises(ValueError,match='selector evidence'):
        calibrate_leaf_mapping(rows,rules,repeats=1)


def fixture_run(tmp_path, split='tune'):
    project=tmp_path/'project';research=project/'artifacts/my_v6_7_research'
    run=project/'artifacts/my_v6_7_strong_baseline/bank'
    research.mkdir(parents=True);run.mkdir(parents=True)
    rules=StrongRules();rows=[row(i,rules) for i in range(3)]
    (research/'split_manifest.json').write_text(json.dumps({'splits':{'train':[], 'tune':[r['example'] for r in rows], 'audit':[]}}))
    (research/'configuration.json').write_text(json.dumps({'solver':AdaptiveConfig().to_dict()}))
    (run/'run_manifest.json').write_text(json.dumps({'provenance':{'split':split},'repeats':1,'rules_digest':rules.digest(),
        'case_inputs':[dict(r['example'],rhs_digest='rhs',x0_digest='zero') for r in rows]}))
    (run/'progress.json').write_text(json.dumps({'status':'complete'}))
    (run/'raw_results.json').write_text(json.dumps(rows))
    return research,run,rules,project/'artifacts/my_v6_7_strong_baseline/frozen'


def test_frozen_rules_provenance_and_immutable_output(tmp_path):
    research,run,rules,out=fixture_run(tmp_path)
    frozen,evidence=calibrate_from_run(run,research,rules,out)
    assert not evidence['audit_used'] and not evidence['thresholds_changed']
    assert dict(frozen.strategy_by_rule)['near_isotropic']=='line_alt_bilinear_full'
    assert json.loads(frozen.provenance)['raw_sha256']==evidence['raw_sha256']
    assert (out/'selector_rules.json').exists()
    with pytest.raises(FileExistsError):calibrate_from_run(run,research,rules,out)


@pytest.mark.parametrize('split',['audit','fresh','smoke'])
def test_audit_or_test_measurements_cannot_calibrate(tmp_path,split):
    research,run,rules,out=fixture_run(tmp_path,split)
    with pytest.raises(ValueError,match='only original train/tune'):
        calibrate_from_run(run,research,rules,out)


def test_mislabeled_cases_and_changed_tolerance_rejected(tmp_path):
    research,run,rules,out=fixture_run(tmp_path)
    rows=json.loads((run/'raw_results.json').read_text())
    rows[0]['example']['operator_digest']='other'
    (run/'raw_results.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='specification'):calibrate_from_run(run,research,rules,out)
    rows[0]['example']['operator_digest']='operator'
    rows[0]['runs']['oracle::line_alt_bilinear_full'][0]['measurement_config']['mg']['tolerance']=1e-2
    (run/'raw_results.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='protocol'):calibrate_from_run(run,research,rules,out)
