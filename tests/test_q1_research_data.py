"""Q1 family coverage and discretization-bound dataset provenance."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from adaptive_mg.pde import (DiffusionCase, assemble_q1_stiffness_reference,
                             assemble_stiffness)
from adaptive_mg.v67.data import VariableCase, variable_stiffness
import adaptive_mg.v67.research_data as data
from adaptive_mg.v67.strong import StrongRules


def compact_plan(*, discretization='q1', smoke=False):
    plan = data.make_research_plan(seed=1709, smoke=smoke,
                                   discretization=discretization)
    for spec in plan['splits'].values():
        spec.update(sizes=[7], per_family=1, count=len(spec['families']))
    return plan


def test_p1_plan_hash_and_legacy_specification_compatibility():
    # Golden metadata hash from the pre-Q1 generator at commit 23d2ed0.
    plan = data.make_research_plan(seed=1709)
    assert data._hash(plan) == 'ad2a80fbd5c46caacf82223efb2716ea669f97bbc35ddb202fcde1c41aa729c9'
    assert 'discretization' not in plan
    assert all('discretization' not in spec for spec in plan['splits'].values())
    for family in data.FAMILIES:
        record = data._specification('train', plan['splits']['train'], 7, family, 0, 0)
        assert 'discretization' not in record['case']
        old = data._example(record)
        explicit = copy.deepcopy(record)
        explicit['case']['discretization'] = 'p1'
        new = data._example(explicit)
        assert old.digest == new.digest
        np.testing.assert_array_equal(old.b, new.b)


@pytest.mark.parametrize('family', data.FAMILIES)
def test_q1_all_standard_families_are_spd_and_preserve_manufactured_rhs(family):
    plan = compact_plan()
    record = data._specification('train', plan['splits']['train'], 7, family, 0, 0)
    assert record['case']['discretization'] == 'q1'
    q1 = data._example(record)
    p1_record = copy.deepcopy(record)
    p1_record['case']['discretization'] = 'p1'
    p1 = data._example(p1_record)
    assert q1.digest != p1.digest
    np.testing.assert_array_equal(q1.exact, p1.exact)
    np.testing.assert_allclose(q1.a @ q1.exact, q1.b, rtol=0., atol=0.)
    np.testing.assert_allclose(q1.a.toarray(), q1.a.toarray().T, rtol=0., atol=0.)
    assert np.linalg.eigvalsh(q1.a.toarray()).min() > 0
    assert q1.a.nnz <= 9 * q1.n * q1.n


def test_variable_q1_samples_coefficient_at_quadrature_points():
    case = VariableCase(7, .013, 37., 1.9, 1.7, .4, discretization='q1')
    angle = np.deg2rad(case.angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    tensor = rotation @ np.diag([1., case.epsilon]) @ rotation.T
    def scalar_field(x, y):
        coefficient = np.exp(case.amplitude * np.sin(2*np.pi*case.frequency*x + case.phase)
                             * np.cos(2*np.pi*y - case.phase))
        return coefficient * tensor
    reference = assemble_q1_stiffness_reference(case.n, scalar_field)
    np.testing.assert_allclose(variable_stiffness(case).toarray(), reference.toarray(),
                               rtol=2e-14, atol=2e-14)
    constant = replace(case, amplitude=0.)
    expected = assemble_stiffness(DiffusionCase(n=7, epsilon=case.epsilon,
        angle_deg=case.angle_deg, discretization='q1'))
    np.testing.assert_allclose(variable_stiffness(constant).toarray(), expected.toarray(),
                               rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize('family', data.FAMILIES[3:])
def test_q1_ood_topology_uses_same_field_as_scalar_reference(family):
    plan = compact_plan()
    record = data._specification('ood_topology', plan['splits']['ood_topology'],
                                 7, family, 0, 0)
    case = data.ResearchCase(**record['case'])
    q1 = data._assemble_case(case)
    reference = assemble_q1_stiffness_reference(case.n,
        lambda x, y: data._ood_tensor_field(case, x, y))
    np.testing.assert_allclose(q1.toarray(), reference.toarray(), rtol=2e-14, atol=2e-12)
    assert np.linalg.eigvalsh(q1.toarray()).min() > 0
    standard = data._assemble_case(replace(case, topology='standard'))
    assert not np.allclose(q1.toarray(), standard.toarray())


def test_selector_validation_rejects_mixed_or_unknown_plans():
    with pytest.raises(ValueError, match='discretization'):
        data.make_research_plan(discretization='q2')
    plan = compact_plan()
    plan['splits']['train'].pop('discretization')
    with pytest.raises(ValueError, match='discretization'):
        data._validate_plan(plan)
    with pytest.raises(ValueError, match='discretization'):
        variable_stiffness(VariableCase(7, .1, 45., 1., 1., 0., discretization='q2'))


def _prepare(tmp_path, discretization='q1'):
    history = tmp_path / 'history'
    history.mkdir()
    output = tmp_path / 'run'
    rules = StrongRules()
    train, validation, manifest = data.prepare_development_data(output, rules,
        plan=compact_plan(discretization=discretization), historical_roots=[history])
    return output, rules, train, validation, manifest


def test_q1_development_restore_and_plan_mismatch_rejection(tmp_path):
    output, rules, train, validation, manifest = _prepare(tmp_path)
    restored, restored_validation, again = data.load_development_data(output, rules)
    assert manifest == again
    assert [e.digest for e in restored] == [e.digest for e in train]
    assert [e.digest for e in restored_validation] == [e.digest for e in validation]
    assert all(e.case.discretization == 'q1' for e in restored + restored_validation)
    altered = copy.deepcopy(manifest)
    altered['splits']['train'][0]['case']['discretization'] = 'p1'
    (output / 'development_manifest.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='discretization'):
        data.load_development_data(output, rules)
    (output / 'development_manifest.json').write_text(json.dumps(manifest))
    plan = json.loads((output / 'dataset_plan.json').read_text())
    plan.pop('discretization')
    for spec in plan['splits'].values():
        spec.pop('discretization')
    (output / 'dataset_plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Plan'):
        data.load_development_data(output, rules)


def test_legacy_p1_manifests_remain_loadable_without_selector(tmp_path):
    output, rules, train, validation, manifest = _prepare(tmp_path, 'p1')
    assert all('discretization' not in r['case']
               for records in manifest['splits'].values() for r in records)
    restored, _, again = data.load_development_data(output, rules)
    assert again == manifest
    assert [e.digest for e in restored] == [e.digest for e in train]


def test_q1_final_and_all_ood_splits_materialize_and_restore(tmp_path):
    output, rules, *_ = _prepare(tmp_path)
    rules_path = tmp_path / 'rules.json'
    rules_path.write_text(json.dumps(rules.to_dict()))
    checkpoint = tmp_path / 'student.pt'
    checkpoint.write_bytes(b'test checkpoint for provenance only')
    source = tmp_path / 'source.py'
    source.write_text('# frozen source fixture\n')
    freeze = data.freeze_research(output, {'student': checkpoint}, rules_path,
        {}, [source], {'discretization': 'q1', 'repeats': 1})
    with data.claim_final_evaluation(output, freeze, {'discretization': 'q1'}) as claim:
        splits, manifest = data.materialize_final_data(output, claim, rules)
        restored, repeated = data.materialize_final_data(output, claim, rules)
        assert repeated == manifest
        assert set(splits) == set(data.HOLDOUT)
        for name, examples in splits.items():
            assert all(e.case.discretization == 'q1' for e in examples)
            assert [e.digest for e in examples] == [e.digest for e in restored[name]]
