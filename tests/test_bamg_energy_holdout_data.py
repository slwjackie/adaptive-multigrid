from copy import deepcopy
import json

import numpy as np
import pytest

from adaptive_mg.provenance import operator_digest
from adaptive_mg.v67 import bamg_energy_holdout_data as data


FREEZE = dict(freeze_sha256='f' * 64, claim_fingerprint='c' * 64)


def small_plan():
    return data.make_holdout_plan(smoke=True)


def test_metadata_plan_has_expanded_balanced_cells_without_operator_assembly(monkeypatch):
    monkeypatch.setattr(data, '_assemble', lambda *a: pytest.fail('Planning assembled holdout data'))
    plan = data.make_holdout_plan()
    assert plan['cases_expected'] == 1270
    assert {k: v['count'] for k, v in plan['splits'].items()} == dict(
        final=560, ood_grid=140, ood_rotation=150, ood_anisotropy=180,
        ood_contrast=120, ood_topology=120)
    specs = data.specifications(plan)
    assert len(specs) == len({v['case_id'] for v in specs}) == 1270
    assert all(s['n'] == 255 for s in specs if s['split'] == 'ood_grid')
    assert plan['splits']['final']['per_family'] == 20
    assert not plan['selection_allowed']
    assert plan['materialization_requires_freeze']
    assert data.make_holdout_plan(smoke=True)['cases_expected'] == 70


def test_parameter_stratification_and_single_axis_supports():
    plan = data.make_holdout_plan()
    specs = data.specifications(plan)
    parameters = [(s, data._parameters(plan, s, 0)) for s in specs]
    epsilon_strata = []
    for spec, p in parameters:
        case, split, family = p['case'], spec['split'], spec['family']
        assert p == data._parameters(plan, spec, 0)
        if split == 'ood_anisotropy':
            assert 1e-5 <= case['epsilon'] < 10**-3.1
        elif family == 'near_isotropic':
            assert .8 <= case['epsilon'] < 1.
        else:
            assert .001 <= case['epsilon'] < .1
        if split == 'ood_rotation':
            assert min(abs(case['angle_deg']-a) for a in data.TRAIN_ANGLES) >= 5.5
            assert family in data.FAMILIES[2:]
        elif family == 'aligned_anisotropy':
            assert case['angle_deg'] in (0., 90.)
        else:
            assert case['angle_deg'] in data.TRAIN_ANGLES
        if split == 'ood_contrast':
            assert 10**3.2 <= case['contrast'] < 1e5
        elif family in data.FAMILIES[:3]:
            assert case['contrast'] == 1.
        else:
            assert 10 <= case['contrast'] < 1000
        assert (p['geometry'] is not None) == (split == 'ood_topology')
        if split == 'final' and spec['n'] == 31 and family == 'rotated_anisotropy':
            epsilon_strata.append(int((np.log10(case['epsilon'])+3)/2*20))
    assert sorted(epsilon_strata) == list(range(20))
    for family, variants in data.TOPOLOGIES.items():
        found = {p['geometry']['variant'] for s, p in parameters
                 if s['split'] == 'ood_topology' and s['family'] == family and s['n'] == 31}
        assert found == set(variants)


@pytest.mark.parametrize('family', data.FAMILIES[3:])
def test_novel_topology_variants_are_spd_and_distinct(family):
    plan = small_plan()
    specs = [s for s in data.specifications(plan) if s['split'] == 'ood_topology' and s['family'] == family]
    hashes, variants = set(), set()
    for spec in specs:
        parameters = data._parameters(plan, spec, 0)
        a, rhs, exact = data._assemble(parameters)
        hashes.add(operator_digest(a, scale_invariant=True))
        variants.add(parameters['geometry']['variant'])
        assert np.linalg.eigvalsh(a.toarray())[0] > 0
        np.testing.assert_allclose(a.toarray(), a.T.toarray(), atol=1e-12)
        np.testing.assert_allclose(a @ exact, rhs['manufactured'], atol=1e-12)
        assert set(rhs) == set(data.RHS_KINDS)
        for vector in rhs.values():
            assert np.linalg.norm(vector) == pytest.approx(1.)
            assert np.isfinite(vector).all()
        assert np.all(rhs['localized'] > 0)
        assert not np.allclose(rhs['manufactured'], rhs['random'])
        assert not np.allclose(rhs['localized'], rhs['random'])
    assert variants == set(data.TOPOLOGIES[family])
    assert len(hashes) == 3


def test_gate_streaming_reconstruction_and_metadata_only_cache_audit(tmp_path, monkeypatch):
    plan = small_plan()
    spec = data.specifications(plan)[0]
    out = tmp_path / 'holdout'
    with pytest.raises(ValueError, match='frozen evaluation'):
        data.load_case(out, spec, plan=plan, forbidden_digests=[], freeze_identity={})
    assert not out.exists()
    state = data.validate_cache(out, plan=plan, forbidden_digests=[], freeze_identity=FREEZE)
    assert not state['cases'] and not out.exists()
    first = data.load_case(out, spec, plan=plan, forbidden_digests=[], freeze_identity=FREEZE)
    files = {str(p): data._file_hash(p) for p in out.rglob('*.json')}
    second = data.load_case(out, spec, plan=plan, forbidden_digests=[], freeze_identity=FREEZE)
    assert first['case_id'] == second['case_id'] == spec['case_id']
    assert first['normalized_operator_digest'] == second['normalized_operator_digest']
    np.testing.assert_array_equal(first['A'].data, second['A'].data)
    for kind in data.RHS_KINDS:
        np.testing.assert_array_equal(first['rhs'][kind], second['rhs'][kind])
    assert files == {str(p): data._file_hash(p) for p in out.rglob('*.json')}
    assert all(p.stat().st_size < 50000 for p in out.rglob('*.json'))
    monkeypatch.setattr(data, '_assemble', lambda *a: pytest.fail('Cache audit assembled A'))
    state = data.validate_cache(out, plan=plan, forbidden_digests=[], freeze_identity=FREEZE)
    assert len(state['cases']) == 1
    assert not state['completed']


def test_duplicate_retries_keep_planned_id_and_do_not_select_on_solver_outcomes(tmp_path):
    plan = small_plan()
    spec = data.specifications(plan)[0]
    a, _, _ = data._assemble(data._parameters(plan, spec, 0))
    forbidden = [operator_digest(a, scale_invariant=True)]
    result = data.load_case(tmp_path, spec, plan=plan, forbidden_digests=forbidden, freeze_identity=FREEZE)
    assert result['case_id'] == spec['case_id']
    assert result['parameters']['retry'] == 1
    assert result['normalized_operator_digest'] not in forbidden
    again = data.load_case(tmp_path, spec, plan=plan, forbidden_digests=forbidden, freeze_identity=FREEZE)
    assert again['data_record_sha256'] == result['data_record_sha256']


def test_cache_hash_and_changed_protocol_rejected(tmp_path):
    plan = small_plan()
    spec = data.specifications(plan)[0]
    data.load_case(tmp_path, spec, plan=plan, forbidden_digests=[], freeze_identity=FREEZE)
    for changed in [dict(freeze_sha256='0'*64, claim_fingerprint='c'*64),
                    dict(freeze_sha256='f'*64, claim_fingerprint='0'*64)]:
        with pytest.raises(ValueError, match='identity changed'):
            data.load_case(tmp_path, spec, plan=plan, forbidden_digests=[], freeze_identity=changed)
    with pytest.raises(ValueError, match='identity changed'):
        data.load_case(tmp_path, spec, plan=plan, forbidden_digests=['a'*64], freeze_identity=FREEZE)
    path = tmp_path / 'cases' / (spec['case_id'] + '.json')
    with path.open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='hash changed'):
        data.validate_cache(tmp_path, plan=plan, forbidden_digests=[], freeze_identity=FREEZE)


def test_orphan_atomic_case_is_recovered_without_reassembling(tmp_path, monkeypatch):
    plan = small_plan()
    spec = data.specifications(plan)[0]
    data.load_case(tmp_path, spec, plan=plan, forbidden_digests=[], freeze_identity=FREEZE)
    manifest = data._read(tmp_path / 'data_manifest.json')
    manifest['cases'].clear()
    data._write_json(tmp_path / 'data_manifest.json', manifest)
    monkeypatch.setattr(data, '_assemble', lambda *a: pytest.fail('Orphan index recovery assembled A'))
    state = data.validate_cache(tmp_path, plan=plan, forbidden_digests=[], freeze_identity=FREEZE)
    assert list(state['cases']) == [spec['case_id']]


@pytest.mark.parametrize('mutation', [
    lambda p: p['splits']['ood_grid'].update(sizes=[7]),
    lambda p: p['splits']['ood_rotation']['families'].append('aligned_anisotropy'),
    lambda p: p['support'].update(epsilon=[1e-8, 1.]),
    lambda p: p.update(selection_allowed=True),
])
def test_invalid_axis_or_misdeclared_support_fails(mutation):
    plan = small_plan()
    mutation(plan)
    with pytest.raises(ValueError):
        data.validate_plan(plan)
