"""Interrupt after an atomic operator checkpoint, then resume the optimizer."""
import json
import pytest
import torch
from adaptive_mg import MGConfig
from adaptive_mg.v67 import AdaptiveConfig, Components
from adaptive_mg.v67.data import dataset
import adaptive_mg.v67.training as training


def test_operator_optimizer_resume_after_interrupt(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    model=Components.create(hidden=4,seed=62)
    cfg=AdaptiveConfig(mg=MGConfig(mode='classical',strategy_name='jacobi_bilinear_full',
        nn_levels=1,max_cycles=8,pre_steps=1,post_steps=1),mode='hybrid',spatial=False,gate_mode='open')
    examples=dataset([7],2,31)
    settings={'seed':11,'epochs':{'joint':1},'unroll_prefix':1,'unroll_tail':1,
        'checkpoint_every':1,'lambda_compute':0.,'operator_lr':1e-4}
    complete,_,_=training.operator_stage(model,examples,cfg,settings,'joint',tmp_path/'complete')
    original=training.rollout;calls=0
    def interrupted(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise KeyboardInterrupt('simulated interruption after first saved update')
        return original(*args,**kwargs)
    monkeypatch.setattr(training,'rollout',interrupted)
    with pytest.raises(KeyboardInterrupt):
        training.operator_stage(model,examples,cfg,settings,'joint',tmp_path/'resumed')
    assert (tmp_path/'resumed/resume.pt').exists()
    monkeypatch.setattr(training,'rollout',original)
    resumed,_,report=training.operator_stage(model,examples,cfg,settings,'joint',tmp_path/'resumed',resume=True)
    assert report['cumulative_updates']==2
    assert report['updates_this_invocation']==1
    for name in ('smoother','transfer'):
        for key,value in getattr(complete,name).state_dict().items():
            torch.testing.assert_close(value,getattr(resumed,name).state_dict()[key],rtol=0,atol=0)
