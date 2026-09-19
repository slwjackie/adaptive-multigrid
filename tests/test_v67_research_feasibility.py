from dataclasses import replace

import numpy as np
import torch

from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.strong import StrongRules, select_strong_strategy, PreparedStrongMG
from adaptive_mg.v67.research_training import create_research_components, sample_config, full_cycle_objective


def test_infeasible_transfer_task_uses_classical_but_keeps_repair_gradient():
    torch.set_num_threads(1)
    e=dataset([15],7,17931)[0];rules=StrongRules()
    e.strong_selection=select_strong_strategy(e.a,e.n,rules).to_dict()
    model=create_research_components(transfer='small_gnn',smoother_hidden=4,transfer_hidden=4,
        complexity_caps=dict(max_row_nnz=8,max_p_ratio=4.,max_ac_ratio=1.01,max_operator_complexity=3.))
    with torch.no_grad():
        for parameter in model.transfer.parameters():parameter.add_(.007)
    cfg=sample_config(e,AdaptiveConfig(),rules,'H_P')
    loss,details,x=full_cycle_objective(e,model,cfg,prefix=2,tail=1)
    assert not details['proposal_feasibility']['feasible']
    assert details['task_branch']=='classical_fallback' and details['actions']==['C']*3
    assert details['feasibility_repair']>0
    _,reference,xc=full_cycle_objective(e,model,replace(cfg,use_transfer=False,use_smoother=False,branch='C'),prefix=2,tail=1)
    np.testing.assert_allclose(x.detach().numpy(),xc.detach().numpy(),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(details['history'].detach().numpy(),reference['history'].detach().numpy(),rtol=1e-12)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in model.transfer.parameters())
    prepared=PreparedStrongMG(e.a,e.n,model,cfg,rules)
    result=prepared.solve(e.b)
    assert result.stats['setup_failures']==1
    assert result.stats['accepted_neural_cycles']==0


def test_feasible_smoother_keeps_full_learned_prefix():
    torch.set_num_threads(1)
    e=dataset([7],7,17932)[0];rules=StrongRules()
    e.strong_selection=select_strong_strategy(e.a,e.n,rules).to_dict()
    model=create_research_components(smoother_hidden=4,transfer_hidden=4)
    loss,details,_=full_cycle_objective(e,model,sample_config(e,AdaptiveConfig(),rules,'H_S'),prefix=2,tail=1)
    assert details['proposal_feasibility']['feasible'] and details['actions']==['H','H','C']
    assert details['feasibility_repair']==0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.smoother.parameters())
