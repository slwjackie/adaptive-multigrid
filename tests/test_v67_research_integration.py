from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from adaptive_mg.v67.banks import Stats
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.strong import PreparedStrongMG, StrongRules, select_strong_strategy
from adaptive_mg.v67.research_runtime import offline_teacher_generation
from adaptive_mg.v67.research_training import (create_research_components, sample_config,
    full_cycle_objective, train_expert, validation_adoption)


def example():
    e=dataset([7],7,19321)[0]
    e.strong_selection=select_strong_strategy(e.a,e.n,StrongRules()).to_dict()
    e.research_split='train'
    return e


@pytest.mark.parametrize('s,p',[('ordinary_control','existing'),('d4','small_gnn'),('o2','edge_mlp'),('teacher_cnn','gnn_teacher')])
def test_research_components_roundtrip_and_independent_snapshots(tmp_path,s,p):
    model=create_research_components(smoother=s,transfer=p,smoother_hidden=4,transfer_hidden=4)
    target=tmp_path/'expert.pt';model.save(target)
    restored=Components.load(target)
    assert restored.generation_signature()==model.generation_signature()
    assert restored.signature()==model.signature()
    with torch.no_grad():next(restored.smoother.parameters()).add_(.1)
    assert restored.generation_signature()!=model.generation_signature()


def test_teacher_is_blocked_even_when_production_metadata_is_forged():
    e=example();rules=StrongRules()
    model=create_research_components(smoother='teacher_cnn',smoother_hidden=4)
    cfg=sample_config(e,AdaptiveConfig(),rules,'H_S')
    p=PreparedStrongMG(e.a,e.n,model,cfg,rules)
    with pytest.raises(RuntimeError,match='training-only'):p.ensure_branch('H_S',Stats())
    with offline_teacher_generation():
        root=p.ensure_branch('H_S',Stats())
        assert root.neural_stencil is not None
        production=PreparedStrongMG(e.a,e.n,model,replace(cfg,mode='production'),rules)
        with pytest.raises(RuntimeError,match='training-only'):production.ensure_branch('H_S',Stats())


@pytest.mark.parametrize('branch',['H_S','H_P','H_SP'])
def test_new_student_banks_reuse_per_level_across_rhs_and_keep_branch_separation(branch):
    torch.set_num_threads(1)
    e=example();rules=StrongRules()
    model=create_research_components(smoother='o2',smoother_hidden=4,transfer='small_gnn',transfer_hidden=4)
    cfg=sample_config(e,AdaptiveConfig(),rules,branch)
    p=PreparedStrongMG(e.a,e.n,model,cfg,rules)
    first=p.solve(e.b);second=p.solve(e.b*1.3)
    assert second.stats['smoother_nn_calls']==second.stats['transfer_nn_calls']==0
    if branch=='H_S':assert first.stats['transfer_nn_calls']==0
    if branch=='H_P':assert first.stats['smoother_nn_calls']==0
    assert first.stats['smoother_nn_calls']<=cfg.mg.nn_levels
    assert first.stats['transfer_nn_calls']<=cfg.mg.nn_levels
    assert first.converged and second.converged


@pytest.mark.parametrize('p',['existing','small_gnn','edge_mlp'])
def test_tight_support_budget_survives_fp32_to_fp64_boundary(p):
    e=example();rules=StrongRules()
    model=create_research_components(smoother_hidden=4,transfer=p,transfer_hidden=4,
        complexity_caps=dict(max_row_nnz=4,max_p_ratio=4.,max_ac_ratio=9.,max_operator_complexity=9.))
    with torch.no_grad():
        for parameter in model.transfer.parameters():parameter.add_(.007)
    cfg=sample_config(e,AdaptiveConfig(),rules,'H_P')
    prepared=PreparedStrongMG(e.a,e.n,model,cfg,rules)
    root=prepared.ensure_branch('H_P',Stats())
    assert max(np.diff(root.p.indptr))<=4
    np.testing.assert_allclose(root.p.sum(1),prepared.classical.p.sum(1),atol=1e-14)
    np.testing.assert_allclose(root.r.toarray(),root.p.toarray().T,atol=0)
    np.testing.assert_allclose(root.coarse.a.toarray(),(root.p.T@root.a@root.p).toarray(),atol=1e-12)
    assert np.linalg.eigvalsh(root.coarse.a.toarray()).min()>0


@pytest.mark.parametrize('branch',['H_S','H_P','H_SP'])
def test_full_cycle_gradient_and_kd_teacher_detached(branch):
    torch.set_num_threads(1);e=example();rules=StrongRules()
    student=create_research_components(smoother_hidden=4,transfer='small_gnn',transfer_hidden=4)
    teacher=create_research_components(smoother='teacher_cnn',smoother_hidden=6,transfer='gnn_teacher',transfer_hidden=6)
    cfg=sample_config(e,AdaptiveConfig(),rules,branch)
    loss,details,_=full_cycle_objective(e,student,cfg,prefix=1,tail=1,teacher=teacher,lambda_kd=.2)
    assert torch.isfinite(loss)
    loss.backward()
    names=('smoother',) if branch=='H_S' else ('transfer',) if branch=='H_P' else ('smoother','transfer')
    assert all(any(p.grad is not None and torch.isfinite(p.grad).all() for p in getattr(student,n).parameters()) for n in names)
    assert all(p.grad is None for m in teacher.modules() for p in m.parameters())
    assert len(details['history'])==2


def test_direct_training_leak_guard_and_resumability(tmp_path):
    torch.set_num_threads(1);e=example();rules=StrongRules()
    model=create_research_components(smoother_hidden=4,transfer_hidden=4)
    settings=dict(seed=19321,updates=1,prefix_cycles=1,tail_cycles=1)
    trained,status=train_expert(model,[e],AdaptiveConfig(),rules,settings,tmp_path/'direct',branch='H_S')
    assert status['updates']==1 and status['all_updates_completed']
    assert trained.generation_signature()!=model.generation_signature()
    restored,same=train_expert(model,[e],AdaptiveConfig(),rules,settings,tmp_path/'direct',branch='H_S',resume=True)
    assert same==status and restored.generation_signature()==trained.generation_signature()
    e.research_split='validation'
    with pytest.raises(ValueError,match='TRAIN'):train_expert(model,[e],AdaptiveConfig(),rules,settings,tmp_path/'bad',branch='H_S')


def test_adoption_preserves_success_ids_and_teacher_cost_is_not_student_acceptance():
    old={'a':dict(success=True,seconds=2.,contraction=.5),'b':dict(success=False,seconds=.01,contraction=2.)}
    swapped={'a':dict(success=False,seconds=.001,contraction=2.),'b':dict(success=True,seconds=.1,contraction=.1)}
    assert not validation_adoption(swapped,old,min_cases=1)['adopted']
    teacher=deepcopy(old);teacher['a'].update(seconds=5.,contraction=.2)
    assert validation_adoption(teacher,old,task_reduction=.05,min_cases=1)['adopted']
    assert not validation_adoption(teacher,old,min_cases=1)['adopted']
