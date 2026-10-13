"""Admission/observable fixtures, never evidence that OpenFOAM was executed."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_mg.v67.h2_fixed_p.data import (
    CROSSFLOW_KIND, SLOSHING_KIND, FIELD_PROFILES, validate_case_contract, experiment_kind,
)
from adaptive_mg.v67.h2_fixed_p.evaluation import (
    nonreacting_field_evidence, sloshing_field_evidence, compare_coupled_runs,
)
from adaptive_mg.v67.h2_fixed_p.fullrun import collect_fields, _verify_prepared, _digest_files
from adaptive_mg.v67.smoother_reuse.physics import flow_metrics


def slosh_contract():
    return dict(schema='lox-he-sloshing-case-v1', experiment_kind=SLOSHING_KIND,
        case_id='slosh_fixture', case_group='slosh_fixture', split='validation', shape=[3, 3],
        species=[], phases=['lox', 'he'], thermophysical_sha256='a'*64,
        thermophysical_files=['constant/phaseProperties'], geometry_m=[1., 1., .1],
        pressure_field='p_rgh', pressure_gauge='reference_fixed', pressure_reference=dict(cell=0, value=2e5),
        physics=dict(kind=SLOSHING_KIND, reacting=False, dimension=2, fixed_grid=True,
                     isothermal=True, phase_change=False, gas_phase_approximation='pure_helium'),
        initial_liquid_fraction=.5, end_time=.02, delta_t=.01,
        physics_checks=dict(alpha_bounds_tolerance=1e-8, liquid_volume_relative_tolerance=1e-6,
                            interface_motion_threshold=1e-8))


def slosh_fields():
    alpha = np.array([[.9]*4+[.5]+[.1]*4]*2)
    alpha[1, 0] -= .05
    alpha[1, 8] += .05
    return {'alpha.lox': alpha, 'U': np.ones((2, 9, 3))*.01,
            'p': np.ones((2, 9))*2e5, 'p_rgh': np.ones((2, 9))*2e5}


def test_new_flow_contracts_keep_hydrogen_inert_and_explicit_slosh_approximation():
    contract = slosh_contract()
    assert experiment_kind(validate_case_contract(contract)) == SLOSHING_KIND
    for key, value in [('phase_change', True), ('gas_phase_approximation', 'oxygen_vapour')]:
        bad = deepcopy(contract)
        bad['physics'][key] = value
        with pytest.raises(ValueError):
            validate_case_contract(bad)
    bad = deepcopy(contract)
    del bad['pressure_reference']
    with pytest.raises(ValueError, match='reference'):
        validate_case_contract(bad)
    h2 = dict(schema='h2-crossflow-case-v1', experiment_kind=CROSSFLOW_KIND, case_id='h2', case_group='h2',
        split='train', shape=[7, 7], species=['H2', 'N2'], reaction_count=0,
        thermophysical_files=['constant/physicalProperties'], thermophysical_sha256='b'*64,
        physics=dict(kind=CROSSFLOW_KIND, reacting=False, dimension=2, fixed_grid=True))
    assert experiment_kind(validate_case_contract(h2)) == CROSSFLOW_KIND
    h2['reaction_count'] = 1
    with pytest.raises(ValueError, match='zero reactions'):
        validate_case_contract(h2)


def test_slosh_checks_conserve_against_initial_not_first_saved_volume():
    fields, contract = slosh_fields(), slosh_contract()
    result = sloshing_field_evidence(fields, contract, fields['alpha.lox'][0])
    assert result['passed'] and not result['thermal_physics_validated']
    fields['alpha.lox'] += .01
    result = sloshing_field_evidence(fields, contract)
    assert result['interface_moves'] and not result['liquid_volume_conserved'] and not result['passed']


@pytest.mark.parametrize('corruption', ['stationary', 'bounds', 'nonfinite', 'initial'])
def test_sloshing_cannot_pass_bad_interface_or_fields(corruption):
    fields, contract = slosh_fields(), slosh_contract()
    if corruption == 'stationary':
        fields['alpha.lox'][1] = fields['alpha.lox'][0]
    elif corruption == 'bounds':
        fields['alpha.lox'][1, 0] = 1.01
    elif corruption == 'nonfinite':
        fields['p_rgh'][1, 0] = np.nan
        with pytest.raises(ValueError, match='finite'):
            sloshing_field_evidence(fields, contract)
        return
    elif corruption == 'initial':
        with pytest.raises(ValueError, match='initial alpha'):
            sloshing_field_evidence(fields, contract, np.zeros(9))
        return
    assert not sloshing_field_evidence(fields, contract)['passed']


def write_field(path, values):
    values = np.asarray(values)
    if values.ndim == 2:
        body = ' '.join('(' + ' '.join(map(str, row)) + ')' for row in values)
        kind = 'vector'
    else:
        body, kind = ' '.join(map(str, values)), 'scalar'
    path.write_text('FoamFile { format ascii; }\ninternalField nonuniform List<' + kind + '> '
                    + str(len(values)) + ' (' + body + ');\nboundaryField {}\n')


def test_collect_slosh_does_not_invent_zero_heat_release(tmp_path):
    values, contract = slosh_fields(), slosh_contract()
    (tmp_path/'0').mkdir()
    write_field(tmp_path/'0/alpha.lox', values['alpha.lox'][0])
    for index, time in enumerate(['.01', '.02']):
        folder = tmp_path/time
        folder.mkdir()
        for name, field in values.items():
            write_field(folder/name, field[index])
    result = collect_fields(tmp_path, contract)
    assert result['physics_evidence']['passed']
    assert not result['reaction_evidence']['zero_heat_release_observed']
    assert not result['reaction_evidence']['finite_saved_Qdot']
    metrics = flow_metrics(result['fields'], contract, result['sample_times'])
    assert metrics['samples'][0]['liquid_volume_m3'] == pytest.approx(.05)


def test_coupled_slosh_rechecks_arrays_and_does_not_mean_align_fixed_reference():
    contract, fields = slosh_contract(), slosh_fields()
    c = dict(schema='h2-fixed-p-coupled-run-v1', arm='classical', source_kind='external_cfd',
        execution_mode='online_coupled', success=True, case_id='slosh_fixture', mesh_sha256='mesh',
        thermophysical_sha256='a'*64, initial_conditions_sha256='init', boundary_conditions_sha256='bc',
        dt_schedule=[.01, .01], stopping_criteria={'rtol': 1e-8}, end_time=.02,
        fixed_p_sha256='P', hardware={'cpu':'fixture'}, execution_threads=1, pressure_gauge='reference_fixed',
        continuity={'passed':True}, experiment_kind=SLOSHING_KIND, physics_contract=contract,
        comparison_fields=list(FIELD_PROFILES[SLOSHING_KIND]), fields=fields, sample_times=[.01, .02],
        physics_evidence=sloshing_field_evidence(fields, contract),
        evidence={'coupled_run_attested':True, 'solver_log_sha256':'log', 'field_files_sha256':{'p':'hash'}},
        timing={'wall_seconds':10., 'pressure_seconds':6.})
    h = deepcopy(c)
    h['arm'] = 'H_S'
    h['evidence'].update(expert_checkpoint_sha256='checkpoint', neural_apply_calls=2)
    tol = dict(declared_before_evaluation=True, pressure_alignment='none',
               fields={name:dict(atol=1e-8, rtol=1e-7) for name in FIELD_PROFILES[SLOSHING_KIND]})
    assert compare_coupled_runs(c, h, tol)['success']
    tol['pressure_alignment'] = 'subtract_spatial_mean'
    assert not compare_coupled_runs(c, h, tol)['speed_claim_eligible']
    tol['pressure_alignment'] = 'none'
    # Identical corrupted paired fields cannot masquerade as physical validity.
    c['fields']['alpha.lox'] += .01
    h['fields']['alpha.lox'] += .01
    assert not compare_coupled_runs(c, h, tol)['speed_claim_eligible']


def test_inert_hydrogen_species_evidence_uses_h2_not_missing_helium():
    h2 = np.array([[.1, .9], [.2, .8]])
    fields = dict(H2=h2, N2=1-h2, T=np.ones((2,2))*300, rho=np.array([[1.,.2],[.9,.3]]))
    assert nonreacting_field_evidence(fields, ('H2', 'N2'))['passed']
    with pytest.raises(KeyError):
        nonreacting_field_evidence(fields)


@pytest.mark.parametrize('generator_name', ['generate_crossflow', 'generate_sloshing'])
def test_new_generated_cases_match_prepared_contracts_without_executing_cfd(tmp_path, generator_name):
    path = Path(__file__).resolve().parents[1]/'integrations/openfoam13/case'/f'{generator_name}.py'
    spec = importlib.util.spec_from_file_location(generator_name+'_physics_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    n = '31' if generator_name == 'generate_crossflow' else '7'
    args = module.parser().parse_args(['--output', str(tmp_path/'case'), '--case-id', 'fixture',
        '--case-group', 'fixture', '--split', 'validation', '--nx', n, '--ny', n])
    case = module.generate(args)
    contract = json.loads((case/'contract.json').read_text())
    validate_case_contract(contract)
    # Native mesh hashes are test fixtures, not a checkMesh/foamRun attestation.
    mesh = case/'constant/polyMesh/faces'
    mesh.parent.mkdir()
    mesh.write_text('unit-test mesh hash fixture')
    contract.update(status='prepared_not_executed', foam_conversion_verified=True,
                    mesh_sha256=_digest_files(case, ['constant/polyMesh/faces']))
    (case/'contract.json').write_text(json.dumps(contract))
    assert _verify_prepared(case)['case_id'] == 'fixture'
    if generator_name == 'generate_sloshing':
        solution = case/'system/fvSolution'
        solution.write_text(solution.read_text().replace('pRefCell 48;', 'pRefCell 0;'))
        with pytest.raises(ValueError, match='pressure reference changed'):
            _verify_prepared(case)


def test_negative_native_laplacian_sign_preserves_solution_and_true_residual():
    from test_h2_fixed_p_service import request, cfg
    from adaptive_mg.v67.h2_fixed_p import service
    value, positive, exact = request()
    contract = slosh_contract()
    contract['shape'] = [7, 7]
    value['pressure_field'] = 'p_rgh'
    value['context']['native_to_spd_scale'] = -1
    # Wire data represent -A_native, -b_native and -Amul_native, with unchanged x.
    # Native negative convention is reconstructed here to check the invariant.
    native_b = -np.asarray(value['b'])
    native_products = -np.asarray(value['probe_products'])
    snapshot, canonical, raw = service.decode_system(value, contract)
    np.testing.assert_allclose(-canonical @ np.asarray(value['probe_vectors']), native_products)
    response = service.Session(cfg(), contract, mode='classical').handle(value)
    x_native = np.asarray(response['x_native'])
    original_residual = np.linalg.norm(native_b - (-canonical) @ x_native)
    assert response['success'] and original_residual <= response['threshold']
    np.testing.assert_allclose(x_native[value['structured_to_native']], exact, atol=1e-6)
    assert snapshot.context['native_to_spd_scale'] == -1
    # No witness-only partial sign flip can be admitted.
    bad = deepcopy(value)
    bad['probe_products'] = native_products.tolist()
    with pytest.raises(ValueError, match='Amul'):
        service.decode_system(bad, contract)
    # A negative definite operator is not admitted without canonicalization.
    bad = deepcopy(value)
    for key in ('diag', 'lower', 'upper', 'b', 'probe_products'):
        bad[key] = (-np.asarray(bad[key])).tolist()
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        service.decode_system(bad, contract)
    bad = deepcopy(value)
    del bad['pressure_field']
    with pytest.raises(ValueError, match='pressure field'):
        service.decode_system(bad, contract)
