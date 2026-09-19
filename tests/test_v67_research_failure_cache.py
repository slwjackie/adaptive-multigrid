"""Failed setup is A/expert/config dependent and must not rerun per RHS."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from adaptive_mg.v67.banks import Stats,CachedGenerationFailure,GenerationFailure
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.strong import PreparedStrongMG,StrongRules,select_strong_strategy
from adaptive_mg.v67.research_training import create_research_components,sample_config
from adaptive_mg.v67.research_transfer import TransferComplexityError


def setup(branch='H_P',*,reject_p=True):
    torch.set_num_threads(1)
    e=dataset([7],7,17391)[0];rules=StrongRules()
    e.strong_selection=select_strong_strategy(e.a,e.n,rules).to_dict()
    model=create_research_components(smoother_hidden=4,transfer='small_gnn',transfer_hidden=4,
        complexity_caps=dict(max_operator_complexity=1. if reject_p else 9.,max_ac_ratio=9.,max_p_ratio=9.,max_row_nnz=16))
    cfg=sample_config(e,AdaptiveConfig(),rules,branch)
    cfg=replace(cfg,mg=replace(cfg.mg,nn_levels=1,max_cycles=50))
    return e,model,cfg,rules,PreparedStrongMG(e.a,e.n,model,cfg,rules)


def test_rejected_p_setup_is_attempted_once_across_same_a_rhs_and_branches():
    e,model,cfg,rules,prepared=setup()
    first=prepared.solve(e.b);second=prepared.solve(1.3*e.b)
    assert first.converged and second.converged
    assert first.stats['transfer_nn_calls']==1 and first.stats['setup_failures']==1
    assert second.stats['transfer_nn_calls']==0 and second.stats['setup_failures']==0
    assert second.stats['setup_failure_cache_hits']==second.stats['transfer_failure_cache_hits']==1
    assert second.stats['accepted_neural_cycles']==0
    assert len(prepared.failed_transfer_banks)==1
    failure=next(iter(prepared.failed_transfer_banks.values()))
    assert isinstance(failure,GenerationFailure) and not hasattr(failure,'__traceback__')
    prepared.config=replace(cfg,branch='H_SP')
    third=prepared.solve(e.b)
    assert third.converged
    assert third.stats['smoother_nn_calls']==third.stats['transfer_nn_calls']==0
    assert third.stats['transfer_failure_cache_hits']==1
    assert any('cached transfer setup failure' in (r.get('error') or '') for r in third.trace)


@pytest.mark.parametrize('changed',['expert','precision','operator'])
def test_changed_generation_key_retries_failed_p(changed):
    e,model,cfg,rules,prepared=setup()
    prepared.solve(e.b)
    if changed=='expert':
        with torch.no_grad():model.transfer.delta_head.weight.add_(.01)
    elif changed=='precision':prepared.config=replace(cfg,inference_dtype='float64')
    else:prepared.replace_operator(e.a*1.1)
    result=prepared.solve(e.b)
    assert result.stats['transfer_nn_calls']==1
    assert result.stats['transfer_failure_cache_hits']==0


def test_failed_s_stencil_is_shared_across_hs_hsp_when_actual_a_matches():
    e,model,cfg,rules,prepared=setup('H_S',reject_p=False)
    with torch.no_grad():model.smoother.head.weight.fill_(float('nan'))
    first=prepared.solve(e.b);second=prepared.solve(e.b)
    assert first.converged and second.converged
    assert first.stats['smoother_nn_calls']==1
    assert second.stats['smoother_nn_calls']==0 and second.stats['smoother_failure_cache_hits']==1
    prepared.config=replace(cfg,branch='H_SP')
    third=prepared.solve(e.b)
    assert third.converged
    assert third.stats['smoother_nn_calls']==0 and third.stats['smoother_failure_cache_hits']==1
    assert third.stats['transfer_nn_calls']==1
    # Changed finite weights can generate again; the failure is not a permanent
    # branch ban and does not poison the exact-A numerical transfer cache.
    with torch.no_grad():model.smoother.head.weight.zero_()
    fourth=prepared.solve(e.b)
    assert fourth.stats['smoother_nn_calls']==1
    assert fourth.stats['smoother_failure_cache_hits']==0
    assert fourth.stats['transfer_nn_calls']==0


def test_failed_generation_does_not_change_classical_or_successful_cache_path():
    e,model,cfg,rules,prepared=setup()
    reference=PreparedStrongMG(e.a,e.n,config=replace(cfg,branch='C'),rules=rules).solve(e.b)
    prepared.solve(e.b)
    prepared.config=replace(cfg,branch='C')
    classical=prepared.solve(e.b)
    np.testing.assert_array_equal(classical.x,reference.x)
    assert classical.stats['smoother_nn_calls']==classical.stats['transfer_nn_calls']==0
    assert classical.stats['setup_failure_cache_hits']==0
    good_e,_,_,_,good=setup(reject_p=False)
    first=good.solve(good_e.b);second=good.solve(good_e.b)
    assert first.stats['transfer_nn_calls']==1 and second.stats['transfer_nn_calls']==0
    assert first.stats['setup_failure_cache_hits']==second.stats['setup_failure_cache_hits']==0


def test_direct_ensure_records_only_actual_failure_and_preserves_original_error():
    _,_,_,_,prepared=setup()
    first=Stats()
    with pytest.raises(TransferComplexityError):prepared.ensure_branch('H_P',first)
    second=Stats()
    with pytest.raises(CachedGenerationFailure,match='TransferComplexityError'):
        prepared.ensure_branch('H_P',second)
    assert first.transfer_nn_calls==1 and second.transfer_nn_calls==0
    assert second.setup_failure_cache_hits==1 and second.branch_setup_seconds>0
