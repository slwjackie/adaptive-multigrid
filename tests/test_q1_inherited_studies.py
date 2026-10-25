"""The inherited warm policy and demo paths retain the chosen discretization."""
from copy import deepcopy
import json

import pytest

from adaptive_mg.v67 import three_pillars as base, warm_study as warm
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.research_data import _hash
from adaptive_mg.v67.strong import StrongRules


@pytest.mark.parametrize('discretization', ['p1', 'q1'])
def test_policy_splits_inherit_discretization_without_mutating_settings(tmp_path, monkeypatch, discretization):
    out=tmp_path/'run';out.mkdir()
    (out/'calibration_manifest.json').write_text(json.dumps({'splits':{}}))
    settings={'seed':12913, 'historical_roots':[str(tmp_path)],
              'study':{'policy_splits':{name:dict(sizes=[7], per_family=1,
                families=['near_isotropic'], seed=912923+100003*i)
                for i,name in enumerate(('policy_fit','policy_tune','policy_validation'))}}}
    if discretization!='p1':settings['discretization']=discretization
    before=deepcopy(settings)
    rules=StrongRules()
    monkeypatch.setattr(warm,'_load',lambda _: (out,settings,AdaptiveConfig(),rules))
    monkeypatch.setattr(warm,'_development',lambda *args: ([],[],{'splits':{}}))
    result=warm.prepare_data(out)
    repeated=warm.prepare_data(out)
    assert settings==before
    for name in settings['study']['policy_splits']:
        assert len(result[name])==1
        assert result[name][0].case.discretization==discretization
        assert result[name][0].digest==repeated[name][0].digest
    saved=json.loads((out/'study_split_manifest.json').read_text())
    if discretization=='p1':
        assert saved['plan_digest']==_hash(settings['study']['policy_splits'])
        assert all('discretization' not in row['case']
                   for rows in saved['splits'].values() for row in rows)
    else:
        assert saved['plan_digest']!=_hash(settings['study']['policy_splits'])
        assert all(row['case']['discretization']=='q1'
                   for rows in saved['splits'].values() for row in rows)
        settings['study']['policy_splits']['policy_fit']['discretization']='p1'
        with pytest.raises(ValueError,match='discretization'):
            warm.prepare_data(out)


def test_three_pillars_demo_solves_the_selected_q1_operator(tmp_path, monkeypatch):
    settings={'seed':12719,'discretization':'q1'}
    cfg=AdaptiveConfig()
    rules=StrongRules()
    monkeypatch.setattr(base,'_load_run',lambda _: (tmp_path,settings,cfg))
    monkeypatch.setattr(base,'load_strong_rules',lambda _: rules)
    generate=base._generate_split
    seen=[]
    def tracked(*args,**kwargs):
        result=generate(*args,**kwargs)
        seen.extend(example.case.discretization for example in result[0])
        return result
    monkeypatch.setattr(base,'_generate_split',tracked)
    result=base.demo(tmp_path,'C',family='near_isotropic',n=7)
    assert seen==['q1']
    assert result.converged
