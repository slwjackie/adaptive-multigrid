"""Project integration; runs with the actual repository solver, not a mock MG."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.strong import StrongRules,PreparedStrongMG
from adaptive_mg.v67.research_data import _generate_split
from adaptive_mg.v67.research_training import create_research_components
from adaptive_mg.v67 import p_headroom_study as study


CAPS=dict(max_row_nnz=16,max_p_ratio=1.,max_ac_ratio=1.15,max_operator_complexity=3.,
          complexity_reference='parent',max_operator_complexity_ratio=1.15,hard_max_operator_complexity=None)


def fixture_run(tmp_path,n=7):
    torch.set_num_threads(1)
    src=tmp_path/'original';src.mkdir()
    cfg=AdaptiveConfig(mg=MGConfig(mode='classical',pre_steps=2,post_steps=2,max_cycles=80,
                       nn_levels=1,stencil_backend='csr'),mode='research',branch='auto',spatial=False,gate_mode='open')
    rules=StrongRules()
    examples,records,_=_generate_split('train',dict(sizes=[n],per_family=1,families=['near_isotropic'],seed=414767),set(),rules)
    study.save(src/'configuration.json',dict(support='support_preserving',complexity_caps=CAPS,solver=cfg.to_dict(),torch_threads=1))
    study.save(src/'selector_rules.json',rules.to_dict())
    study.save(src/'data/development_manifest.json',dict(splits={'train':records,'validation':[]}))
    return src,cfg,rules,examples[0]


def test_end_to_end_headroom_reuses_actual_development_operator_without_mutating_source(tmp_path):
    src,cfg,rules,e=fixture_run(tmp_path);out=tmp_path/'p_study'
    original={str(p):study.file_digest(p) for p in src.rglob('*') if p.is_file()}
    plan=study.make_plan(src,out,limit=1,repeats=1,rhs_count=2,direct_steps=2,dense_max_dofs=49)
    report=study.run_study(out)
    assert not report['automatic_promotion'] and not report['final_ood_opened']
    case=study.read(out/'case_results'/f'{e.group_digest}.json')
    assert case['status']=='complete' and case['spectral']['status']=='computed'
    for method in ('energy_min','ls_uniform','ls_energy','direct'):
        assert case['candidates'][method]['status']=='feasible'
        assert case['runs']['warm_multiple'][method][0]['candidate_applied']
        assert case['runs']['warm_multiple'][method][0]['success']
    assert case['runs']['warm_multiple']['direct'][0]['offline_oracle']
    assert not case['runs']['warm_multiple']['energy_min'][0]['actual_NN_component']
    assert case['candidates']['direct']['detail']['objective'].startswith('m actual V-cycles')
    assert original=={str(p):study.file_digest(p) for p in src.rglob('*') if p.is_file()}
    again=study.run_study(out,resume=True)
    assert again==report
    with pytest.raises(FileExistsError):study.run_study(out)
    record_path=out/f'P_{e.group_digest}_energy_min.npz'
    record_path.write_bytes(b'bad')
    with pytest.raises(Exception):study.report_study(out)


def test_absent_levels_and_dense_budget_are_explicit_not_fake_oracles(tmp_path):
    src,cfg,rules,e=fixture_run(tmp_path);out=tmp_path/'limits'
    study.make_plan(src,out,limit=1,repeats=1,rhs_count=1,direct_steps=1,
                    direct_max_dofs=1,dense_max_dofs=1)
    report=study.run_study(out);case=study.read(out/'case_results'/f'{e.group_digest}.json')
    assert case['spectral']['status']=='skipped_budget'
    assert case['candidates']['direct']['status']=='skipped_budget'
    assert report['summary']['warm_multiple']['direct']['geometric_speedup'] is None
    assert not report['case_gates'][0]['automatic_stop']


def test_p2_and_final_scope_guards_and_stale_input_rejection(tmp_path):
    src,_,_,_=fixture_run(tmp_path)
    with pytest.raises(ValueError):study.make_plan(src,tmp_path/'x',split='final')
    with pytest.raises(ValueError):study.make_plan(src,src/'nested')
    out=tmp_path/'good';study.make_plan(src,out,limit=1)
    conf=study.read(src/'configuration.json');conf['support']='standard';study.save(src/'configuration.json',conf)
    with pytest.raises(ValueError,match='changed'):study.load_plan(out)
    with pytest.raises(ValueError,match='P2'):study.make_plan(src,tmp_path/'bad',limit=1)


def test_invalid_interpolation_uses_original_selected_classical_recovery(tmp_path):
    src,cfg,rules,e=fixture_run(tmp_path)
    c=PreparedStrongMG(e.a,e.n,None,study.branch_config(cfg,'C',0),rules);reference=c.solve(e.b)
    def bad(prepared,node,config,stats):return node.p*100.,{}
    p=study.AlternativePPrepared(e.a,e.n,None,study.branch_config(cfg,'H_P',0),rules,
                                builder=bad,index=0,caps=CAPS)
    result=p.solve(e.b)
    assert result.converged and p.p_value is None
    assert result.stats['setup_failures']>0
    np.testing.assert_allclose(result.x,reference.x,rtol=1e-11,atol=1e-12)
    assert p.selection.strategy_name==c.selection.strategy_name


def test_factorial_preserves_selected_S_and_identical_P_under_both_smoothers(tmp_path):
    src,cfg,rules,e=fixture_run(tmp_path);out=tmp_path/'factorial'
    study.make_plan(src,out,limit=1,repeats=1,rhs_count=1,direct_steps=1,methods=['classical','energy_min'])
    study.run_study(out)
    model=create_research_components(smoother='student_cnn',smoother_hidden=4,transfer_hidden=4,
                                     support='support_preserving',complexity_caps=CAPS,seed=14)
    ckpt=tmp_path/'S.pt';model.save(ckpt);frozen=model.frozen_inference_copy()
    selection=tmp_path/'selection.json'
    study.save(selection,dict(checkpoint=str(ckpt),checkpoint_sha256=study.file_digest(ckpt),
        rules_digest=rules.digest(),expert_signature=frozen.signature(),solver=cfg.to_dict()))
    before=study.file_digest(ckpt)
    report=study.factorial_study(out,selection)
    row=report['rows'][0];assert row['status']=='complete'
    for regime,runs in row['runs'].items():
        assert runs['H_P'][0]['P_detail']['P_digest']==runs['H_SP'][0]['P_detail']['P_digest']
        assert not runs['H_P'][0]['actual_NN_component'] and runs['H_SP'][0]['actual_NN_component']
    assert study.file_digest(ckpt)==before and not report['automatic_joint_training']
    assert study.factorial_study(out,selection,resume=True)==report
