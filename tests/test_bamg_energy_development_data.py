from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from adaptive_mg.config import MGConfig
from adaptive_mg.v67 import bamg_energy_development_data as data
from adaptive_mg.v67.fixed_classical import BilinearClassicalBaseline
from adaptive_mg.v67.research_data import make_research_plan, load_development_data


def plan(count=20, *, small=False):
    value = make_research_plan(seed=2026110821, discretization='q1')
    value['development_sampling'] = data.SAMPLING
    for split, per_family in [('train', count), ('validation', 10)]:
        spec = value['splits'][split]
        spec.update(per_family=per_family)
        if small:
            spec.update(sizes=[7], families=['aligned_anisotropy', 'channel'])
        spec['count'] = len(spec['sizes']) * len(spec['families']) * per_family
    return value


def baseline():
    return BilinearClassicalBaseline.from_config(SimpleNamespace(mg=MGConfig(
        strategy_name='line_alt_bilinear_full__v22', pre_steps=2, post_steps=2)))


def test_design_is_nested_and_validation_unchanged_without_assembly(monkeypatch):
    monkeypatch.setattr(data, '_example', lambda *a: pytest.fail('Metadata planning assembled an operator'))
    small, large = plan(20), plan(40)
    train_small, train_large = list(data.specifications(small, 'train')), list(data.specifications(large, 'train'))
    assert len(train_small) == 560 and len(train_large) == 1120
    assert train_small == train_large[:560]
    assert list(data.specifications(small, 'validation')) == list(data.specifications(large, 'validation'))
    assert len(list(data.specifications(small, 'validation'))) == 280
    assert len({r['name'] for r in train_large}) == 1120


def test_all_supports_categories_and_joint_hard_strata():
    value = plan(40)
    for split in ('train', 'validation'):
        rows = list(data.specifications(value, split))
        for n in value['splits'][split]['sizes']:
            for family in data.FAMILIES:
                cases = [r for r in rows if r['case']['n'] == n and r['family'] == family]
                angles = Counter(r['case']['angle_deg'] for r in cases)
                assert set(angles) == (set((0., 90.)) if family == 'aligned_anisotropy' else set(data.TRAIN_ANGLES))
                assert max(angles.values()) - min(angles.values()) <= 1
                checkers = Counter(r['case']['checker_cells'] for r in cases)
                assert set(checkers) == {3, 4, 5}
                assert max(checkers.values()) - min(checkers.values()) <= 1
                if split == 'train' and family in data.FAMILIES[3:]:
                    hard = [r['case'] for r in cases if r['stratification']['paired_hard_stratum']]
                    assert len(hard) == 2
                    assert all(c['epsilon'] < 10**-2.9 and c['contrast'] >= 10**2.9 for c in hard)
        for row in rows:
            c = row['case']
            assert c['topology'] == 'standard' and c['discretization'] == 'q1'
            assert c['n'] in (15, 31, 63, 127)
            assert (.8 <= c['epsilon'] < 1) if c['family'] == 'near_isotropic' else (.001 <= c['epsilon'] < .1)
            assert (c['contrast'] == 1) if c['family'] in data.FAMILIES[:3] else (10 <= c['contrast'] < 1000)
            x0, x1, y0, y1 = c['patch_bounds']
            assert .35 <= (x0+x1)/2 <= .65 and .35 <= (y0+y1)/2 <= .65
            assert .13 <= (x1-x0)/2 <= .24 and .13 <= (y1-y0)/2 <= .24
            assert .06 <= c['channel_width'] < .13
            assert .7 <= c['frequency'] < 2.5
            assert 0 <= c['phase'] < 2*np.pi


def test_latin_hypercube_marginals_and_retries_preserve_strata():
    spec = plan()['splits']['train']
    for retry in (0, 1):
        cases = [data._specification('train', spec, 31, 'channel', i, retry)['case'] for i in range(20)]
        dimensions = [
            [(np.log10(c['epsilon'])+3)/2 for c in cases],
            [(np.log10(c['contrast'])-1)/2 for c in cases],
            [(c['channel_width']-.06)/.07 for c in cases],
            [(c['frequency']-.7)/1.8 for c in cases],
            [c['phase']/(2*np.pi) for c in cases],
            [((c['patch_bounds'][0]+c['patch_bounds'][1])/2-.35)/.3 for c in cases],
        ]
        for values in dimensions:
            assert sorted(int(u * 20) for u in values) == list(range(20))
    for index in range(20):
        a, b = [data._specification('train', spec, 31, 'channel', index, retry) for retry in (0, 1)]
        assert a['case']['angle_deg'] == b['case']['angle_deg']
        assert a['case']['checker_cells'] == b['case']['checker_cells']
        assert a['stratification'] == b['stratification']
        assert a['case']['epsilon'] != b['case']['epsilon']


def test_materialized_nesting_shared_validation_legacy_restore_and_history(tmp_path):
    p20, p40 = plan(small=True), plan(40, small=True)
    rules = baseline()
    forbidden = data._example(next(data.specifications(p20, 'train'))).group_digest
    history = tmp_path / 'exclusion_manifest.json'
    history.write_text(json.dumps({'operators': [{'normalized_operator_digest': forbidden}]}))
    out20, out40 = tmp_path / '20', tmp_path / '40'
    t20, v20, m20 = data.prepare_development_data(out20, rules, plan=p20, historical_roots=[history])
    t40, v40, m40 = data.prepare_development_data(out40, rules, plan=p40, historical_roots=[history])
    assert m20['splits']['validation'] == m40['splits']['validation']
    assert m20['splits']['train'] == m40['splits']['train'][:40]
    assert m20['rejected_normalized_duplicates']['train'] >= 1
    assert m20['splits']['train'][0]['retry'] == 1
    for examples in (t20+v20, t40+v40):
        digests = [e.group_digest for e in examples]
        assert len(set(digests)) == len(digests)
        assert forbidden not in digests
        for example in examples:
            np.testing.assert_allclose(example.a @ example.exact, example.b)
    restored_train, restored_validation, restored_manifest = load_development_data(out40, rules)
    assert [e.digest for e in restored_train] == [e.digest for e in t40]
    assert [e.digest for e in restored_validation] == [e.digest for e in v40]
    assert restored_manifest == m40
    assert not m40['final_operators_materialized']
    assert not (out40/'final_data_manifest.json').exists()
    with pytest.raises(FileExistsError, match='already committed'):
        data.prepare_development_data(out40, rules, plan=p40, historical_roots=[history])


def test_invalid_plan_fails_before_generation():
    value = plan()
    for change in ({'development_sampling': 'random'}, {'discretization': 'p1'}):
        changed = deepcopy(value)
        changed.update(change)
        with pytest.raises(ValueError):
            list(data.specifications(changed, 'train'))
    changed = deepcopy(value)
    changed['splits']['train'].update(per_family=21, count=21*28)
    with pytest.raises(ValueError, match='complete stratified blocks'):
        list(data.specifications(changed, 'train'))
    with pytest.raises(ValueError, match='Only TRAIN and validation'):
        list(data.specifications(value, 'final'))


def test_solver_or_assembly_failure_is_not_filtered(tmp_path, monkeypatch):
    value = plan(small=True)
    def fail(record):
        raise RuntimeError('assembly failed')
    monkeypatch.setattr(data, '_example', fail)
    with pytest.raises(RuntimeError, match='assembly failed'):
        data.prepare_development_data(tmp_path/'data', baseline(), plan=value, historical_roots=[tmp_path])
    assert not (tmp_path/'data/development_manifest.json').exists()
