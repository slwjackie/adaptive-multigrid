"""Synthetic unit fixtures ONLY: exercise wire/CFD declarations, not flame physics.

These deliberately small manufactured LDU arrays are not a combustion dataset.
Imported provenance remains producer-declared with combustion_verified=False.
"""
from copy import deepcopy
import io
import json
import struct

import numpy as np
import pytest
import scipy.sparse as sp

from adaptive_mg import MGConfig
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.h2_fixed_p import data, service
from adaptive_mg.v67.world_model.adapter import solution_to_native
from adaptive_mg.v67.world_model.data import load_trajectories


def cfg():
    return AdaptiveConfig(mg=MGConfig(mode='classical', strategy_name='line_alt_energymin_full__em5__v11',
                         max_cycles=80, tolerance=1e-8, absolute_tolerance=1e-12, stencil_backend='csr'),
                         use_transfer=False, spatial=False, gate_mode='open')


def contract():
    return dict(schema='h2-fixed-p-case-v1', case_id='unit_fixture', case_group='unit_fixture',
                split='train', shape=[7,7], chemistry_sha256='a'*64,
                physics=dict(fuel='H2', oxidizer='air', dimension=2, fixed_grid=True,
                             evidence='TEST_DECLARATION_ONLY_NOT_REAL_CFD'))


def request(index=0, mode='classical'):
    n = 7
    line = sp.diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1,0,1], format='csr')
    a = (sp.kron(sp.eye(n), line) + sp.kron(line, sp.eye(n))).tocsr()
    a.setdiag(a.diagonal() + index*np.linspace(.01,.2,n*n))
    exact = np.sin(np.arange(n*n)*.7)
    x0 = exact*.2
    order = np.random.default_rng(13).permutation(n*n)
    inverse = np.argsort(order)
    native = a[inverse][:,inverse].tocsr()
    upper = sp.triu(native, k=1).tocoo()
    lo, up = upper.row, upper.col
    probes = np.column_stack((np.sin(np.arange(n*n)+.2), np.cos(np.arange(n*n)*.3)))
    c = cfg()
    value = dict(schema=service.SCHEMA, source_kind='external_cfd', mode=mode,
                 op='record' if mode=='native' else 'solve', shape=[n,n], index=index, time=index*.01,
                 mesh_id='unit-grid', boundary_id='unit-dirichlet', boundary_finalized=True,
                 nullspace='none', coupled_interfaces=0, diag=native.diagonal().tolist(),
                 lower_addr=lo.tolist(), upper_addr=up.tolist(), upper=upper.data.tolist(),
                 lower=np.asarray(native[up,lo]).ravel().tolist(), b=(a@exact)[inverse].tolist(),
                 x0=x0[inverse].tolist(), structured_to_native=order.tolist(),
                 probe_vectors=probes.tolist(), probe_products=(native@probes).tolist(),
                 rtol=c.mg.tolerance, atol=c.mg.absolute_tolerance, max_cycles=c.mg.max_cycles,
                 context={'test_fixture': True}, x_native_reference=exact[inverse].tolist())
    return value, a, exact


class FragmentSocket:
    """Byte transport fake: framing tests require no AF_UNIX permission."""
    def __init__(self, contents=b'', chunk=5):
        self.input = io.BytesIO(contents)
        self.output = bytearray()
        self.chunk = chunk
    def recv(self, count):
        return self.input.read(min(count, self.chunk))
    def sendall(self, value):
        self.output.extend(value)


def test_wire_fragmentation_roundtrip_and_truncation():
    value, _, _ = request()
    writer = FragmentSocket()
    service.send(writer, value)
    assert service.receive(FragmentSocket(bytes(writer.output))) == value
    with pytest.raises(EOFError):
        service.receive(FragmentSocket(bytes(writer.output[:-1])))
    with pytest.raises(ValueError, match='frame size'):
        service.receive(FragmentSocket(struct.pack('!Q', service.MAX_FRAME+1)))
    payload = b'[1,2]'
    with pytest.raises(ValueError, match='object request'):
        service.receive(FragmentSocket(struct.pack('!Q',len(payload))+payload))


def test_decode_native_LDU_permutation_and_solution_roundtrip():
    value, expected_a, exact = request()
    s, native, raw = service.decode_system(value, contract())
    np.testing.assert_allclose(s.a.toarray(), expected_a.toarray())
    np.testing.assert_allclose(s.b, expected_a@exact)
    np.testing.assert_allclose(s.x0, .2*exact)
    xn = solution_to_native(exact, raw['structured_to_native'])
    np.testing.assert_allclose(native@xn, raw['b'])
    assert s.source_kind == 'external_cfd'
    assert s.context['case_contract_digest'] == data.digest(contract())
    assert not s.a.data.flags.writeable and not s.b.flags.writeable


@pytest.mark.parametrize('change,match', [
    ({'boundary_finalized': False}, 'finalized'),
    ({'coupled_interfaces': 1}, 'serial'),
    ({'nullspace': 'constant'}, 'anchor'),
    ({'index': .5}, 'integer'),
    ({'shape': [7.,7]}, 'integer'),
    ({'source_kind': 'synthetic_elliptic'}, 'external_cfd'),
])
def test_decode_rejects_incompatible_contract(change, match):
    value, _, _ = request()
    value.update(change)
    with pytest.raises(ValueError, match=match):
        service.decode_system(value, contract())


def test_decode_rejects_changed_witness_invalid_map_and_nonSPD():
    value, _, _ = request()
    bad = deepcopy(value)
    bad['probe_products'][0][0] += 1.
    with pytest.raises(ValueError, match='Amul'):
        service.decode_system(bad, contract())
    bad = deepcopy(value)
    bad['structured_to_native'][0] = bad['structured_to_native'][1]
    with pytest.raises(ValueError, match='permutation'):
        service.decode_system(bad, contract())
    # Symmetric positive-diagonal can still be indefinite. Recompute witnesses
    # so it is SPD admission (not witness disagreement) which rejects it.
    bad = deepcopy(value)
    bad['diag'] = [.1]*49
    lo, up = np.asarray(bad['lower_addr']), np.asarray(bad['upper_addr'])
    a = sp.coo_matrix((np.r_[bad['diag'],bad['upper'],bad['lower']],
        (np.r_[np.arange(49),lo,up],np.r_[np.arange(49),up,lo])),shape=(49,49)).tocsr()
    bad['probe_products'] = (a@np.asarray(bad['probe_vectors'])).tolist()
    with pytest.raises(np.linalg.LinAlgError):
        service.decode_system(bad, contract())


def test_live_session_solves_two_current_systems_same_P_and_records_evidence(tmp_path):
    log = tmp_path/'service.jsonl'
    session = service.Session(cfg(), contract(), mode='classical', evidence=log)
    results = []
    for index in range(2):
        value, a, exact = request(index)
        result = session.handle(value)
        results.append(result)
        assert result['success'] and result['final_residual'] <= result['threshold']
        np.testing.assert_allclose(np.asarray(result['x_native'])[value['structured_to_native']], exact, atol=1e-6)
        assert result['step']['stats']['hierarchy_p_preserved']
    assert results[0]['p_digest'] == results[1]['p_digest']
    assert results[0]['matrix_digest'] != results[1]['matrix_digest']
    assert results[1]['step']['stats']['builds'][0]['numeric_refactorized']
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(entries) == 2 and all('x_native' not in row for row in entries)
    with pytest.raises(ValueError, match='nonmonotone'):
        session.handle(request(1)[0])


def test_live_session_stopping_arm_and_native_residual_contracts():
    session = service.Session(cfg(), contract(), mode='classical')
    value, _, _ = request()
    for key in ('rtol','atol','max_cycles'):
        bad = deepcopy(value)
        bad[key] *= 2
        with pytest.raises(ValueError, match='stopping criterion'):
            session.handle(bad)
    bad = dict(value, native_threshold=1e9)
    with pytest.raises(ValueError, match='raw residual contract'):
        session.handle(bad)
    with pytest.raises(ValueError, match='arm mismatch'):
        session.handle(dict(value, mode='hs'))
    native = service.Session(cfg(), contract(), mode='native')
    bad, _, _ = request(mode='native')
    bad['x_native_reference'] = [0.]*49
    with pytest.raises(ValueError, match='shared raw L2'):
        native.handle(bad)
    assert native.last_index == -1


def test_cancellation_roundoff_is_allowed_only_for_native_diagnostic_agreement():
    value, _, _ = request(mode='native')
    _, a_native, _ = service.decode_system(value, contract())
    exact = np.full(a_native.shape[0], 1e12)
    value['b'] = (a_native@exact).tolist()
    value['x_native_reference'] = exact.tolist()
    value['x0'] = (exact-1e-4).tolist()
    s, _, _ = service.decode_system(value, contract())
    ax = s.a@s.x0
    initial = service.stable_norm(s.b-ax)
    threshold = max(cfg().mg.absolute_tolerance, cfg().mg.tolerance*initial)
    roundoff = 128*np.finfo(float).eps*(service.stable_norm(s.b)+service.stable_norm(ax))
    assert roundoff > initial*1e-10
    value['native_initial_residual'] = initial + roundoff*.5
    value['native_threshold'] = threshold + cfg().mg.tolerance*roundoff*.5
    session = service.Session(cfg(), contract(), mode='native')
    result = session.handle(value)
    assert result['success'] and result['threshold'] == threshold
    assert result['final_residual'] <= threshold  # original, unrelaxed target
    bad = deepcopy(value)
    bad['native_initial_residual'] = initial + roundoff*2
    with pytest.raises(ValueError, match='raw residual contract'):
        service.Session(cfg(), contract(), mode='native').handle(bad)
    bad = deepcopy(value)
    bad['native_threshold'] = threshold + cfg().mg.tolerance*roundoff*2
    with pytest.raises(ValueError, match='raw residual contract'):
        service.Session(cfg(), contract(), mode='native').handle(bad)
    bad = deepcopy(value)
    bad['x_native_reference'][0] += 1.
    with pytest.raises(ValueError, match='shared raw L2'):
        service.Session(cfg(), contract(), mode='native').handle(bad)


def collect_fixture(root):
    session = service.Session(cfg(), contract(), mode='native', recording=root)
    for index in range(2):
        value, _, _ = request(index, mode='native')
        assert session.handle(value)['success']


def test_collection_import_preserves_declared_provenance_and_detects_source_tampering(tmp_path):
    raw, output = tmp_path/'raw', tmp_path/'dataset'
    collect_fixture(raw)
    manifest = data.import_recordings([raw], output)
    assert manifest['source_kind'] == 'external_cfd' and manifest['combustion_verified'] is False
    data.verify_dataset(output)
    cohorts = load_trajectories(output, 'train')
    assert len(cohorts) == 1 and len(cohorts[0][1]) == 2
    assert all(s.context['test_fixture'] for s in cohorts[0][1])
    assert cohorts[0][0]['chemistry_sha256'] == contract()['chemistry_sha256']
    copied = next((output/'sources').rglob('*.npz'))
    copied.write_bytes(copied.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='source recording changed'):
        data.verify_dataset(output)


def test_import_rejects_contract_swap_and_invalid_case_ids(tmp_path):
    raw = tmp_path/'raw'
    collect_fixture(raw)
    document = data.read(raw/'case_contract.json')
    document['chemistry_sha256'] = 'b'*64
    (raw/'case_contract.json').write_text(json.dumps(document))
    with pytest.raises(ValueError, match='snapshot and case contract'):
        data.import_recordings([raw], tmp_path/'output')
    with pytest.raises(ValueError, match='case_id'):
        data.validate_case_contract(dict(contract(), case_id='../bad'))
    with pytest.raises(ValueError, match='case_id'):
        data.validate_case_contract(dict(contract(), case_id='bad.name'))
    with pytest.raises(ValueError, match='shape'):
        data.validate_case_contract(dict(contract(), shape=[7.,7]))
