"""Protocol/field fixtures only; these tests do not simulate an inert gas flow."""
from copy import deepcopy
from pathlib import Path
import hashlib

import numpy as np
import pytest
import scipy.sparse as sp

from adaptive_mg import DiffusionCase, assemble_stiffness, MGConfig
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.h2_fixed_p import study as W
from adaptive_mg.v67.h2_fixed_p.data import (
    FIELD_PROFILES, VARIABLE_DENSITY_KIND, experiment_kind, import_recordings,
    input_fingerprint, validate_case_contract, verify_dataset,
)
from adaptive_mg.v67.h2_fixed_p.evaluation import compare_coupled_runs, nonreacting_field_evidence
from adaptive_mg.v67.h2_fixed_p.fullrun import collect_fields, _verify_prepared, _digest_files, summarize_service_timing
from adaptive_mg.v67.world_model.adapter import FinalizedLduRecorder
from adaptive_mg.v67.world_model.data import digest, write_json


def contract(kind=VARIABLE_DENSITY_KIND):
    common = dict(case_id='inert_fixture', case_group='inert_fixture', split='train', shape=[7, 7],
                  test_fixture_only=True, actual_CFD=False)
    if kind == VARIABLE_DENSITY_KIND:
        return dict(common, schema='variable-density-case-v1',
                    physics=dict(kind=kind, reacting=False, dimension=2, fixed_grid=True),
                    species=['He', 'N2'], reaction_count=0, thermophysical_sha256='a' * 64,
                    thermophysical_files=['constant/physicalProperties'])
    return dict(common, schema='h2-fixed-p-case-v1', chemistry_sha256='b' * 64,
                physics=dict(fuel='H2', oxidizer='air', dimension=2, fixed_grid=True))


def test_inert_contract_uses_thermophysical_provenance_without_fictitious_chemistry():
    value = validate_case_contract(contract())
    assert experiment_kind(value) == VARIABLE_DENSITY_KIND
    assert input_fingerprint(value) == ('thermophysical_sha256', 'a' * 64)
    assert 'chemistry_sha256' not in value


@pytest.mark.parametrize('change', ['reacting', 'reaction_count', 'species', 'hash', 'path', 'kind'])
def test_nonreacting_declaration_cannot_hide_reactions_or_unbound_inputs(change):
    value = contract()
    if change == 'reacting': value['physics']['reacting'] = True
    elif change == 'reaction_count': value['reaction_count'] = 1
    elif change == 'species': value['species'] = ['H2', 'N2']
    elif change == 'hash': value['thermophysical_sha256'] = ''
    elif change == 'path': value['thermophysical_files'] = ['../escape']
    elif change == 'kind': value['experiment_kind'] = 'h2_combustion'
    with pytest.raises(ValueError):
        validate_case_contract(value)


def fixture_recording(root, split='train', offset=0):
    value = contract()
    value.update(case_id='inert_fixture_' + split, case_group='inert_fixture_' + split, split=split)
    recorder = FinalizedLduRecorder(root, producer={'kind': 'unit_test_fixture', 'actual_CFD': False},
                                   physics=value['physics'])
    write_json(root / 'case_contract.json', value)
    for index in range(2):
        a = assemble_stiffness(DiffusionCase(7, epsilon=.3 + offset * .01, angle_deg=20 + offset + index, contrast=3, pattern='channel'))
        upper = sp.triu(a, k=1).tocoo()
        probes = np.column_stack([np.ones(49), np.linspace(-1, 1, 49)])
        recorder.record(trajectory=value['case_id'], case_group=value['case_group'], split=split,
            index=index, time=index * .01, shape=(7, 7), mesh_id='fixture-grid', boundary_id='anchored',
            diag=a.diagonal(), lower_addr=upper.row, upper_addr=upper.col, lower=upper.data, upper=upper.data,
            b=np.ones(49) + index * .1, x0=np.zeros(49), probe_vectors=probes, probe_products=a @ probes,
            structured_to_native=np.arange(49), boundary_finalized=True, nullspace='none', coupled_interfaces=0,
            context={'test_fixture_only': True, 'case_contract_digest': digest(value), 'physics': value['physics']})
    return root


def test_import_preserves_inert_physics_and_checks_thermophysical_provenance(tmp_path):
    raw = fixture_recording(tmp_path / 'raw')
    output = tmp_path / 'data'
    result = import_recordings([raw], output)
    assert result['experiment_kind'] == VARIABLE_DENSITY_KIND
    assert 'nonreacting' in result['physics']
    assert result['physics_verified'] is False
    assert result['trajectories'][0]['thermophysical_sha256'] == 'a' * 64
    assert 'chemistry_sha256' not in result['trajectories'][0]
    assert verify_dataset(output)['experiment_kind'] == VARIABLE_DENSITY_KIND
    result['trajectories'][0]['thermophysical_sha256'] = 'b' * 64
    write_json(output / 'sequence_manifest.json', result)
    with pytest.raises(ValueError, match='producer contract'):
        verify_dataset(output)


def test_import_refuses_mixing_primary_inert_and_supplementary_h2(tmp_path):
    sources = [tmp_path / 'inert', tmp_path / 'h2']
    for root, kind in zip(sources, [VARIABLE_DENSITY_KIND, 'h2_combustion']):
        root.mkdir()
        write_json(root / 'case_contract.json', contract(kind))
    with pytest.raises(ValueError, match='mix experiment kinds'):
        import_recordings(sources, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def fields():
    he = np.array([[.1, .9], [.2, .8]])
    return dict(T=np.full((2, 2), 300.), He=he, N2=1. - he,
                rho=np.array([[1., .2], [.9, .3]]), U=np.ones((2, 2, 3)), p=np.full((2, 2), 101325.))


def test_spatial_density_contrast_alone_is_not_temporal_change():
    value = fields()
    value['rho'][1] = value['rho'][0]
    evidence = nonreacting_field_evidence(value)
    assert evidence['density_spatially_nonuniform']
    assert not evidence['density_evolves']
    assert not evidence['passed']


def write_field(path, values):
    values = np.asarray(values)
    if values.ndim == 2:
        body = ' '.join('(' + ' '.join(map(str, row)) + ')' for row in values)
        kind = 'vector'
    else:
        body, kind = ' '.join(map(str, values)), 'scalar'
    path.write_text('FoamFile { format ascii; }\ninternalField nonuniform List<' + kind + '> '
                    + str(len(values)) + ' (' + body + ');\nboundaryField {}\n')


def test_saved_inert_fields_require_zero_heat_release_and_actual_species_density_evidence(tmp_path):
    values = fields()
    for i, t in enumerate(['0.01', '0.02']):
        folder = tmp_path / t
        folder.mkdir()
        for name, array in values.items():
            write_field(folder / name, array[i])
        write_field(folder / 'Qdot', [0., 0.])
    declaration = dict(contract(), shape=[2, 1], end_time=.02)
    result = collect_fields(tmp_path, declaration)
    assert result['physics_evidence']['passed']
    assert result['reaction_evidence']['zero_heat_release_observed']
    assert result['fields']['rho'].shape == (2, 2)
    assert len(result['hashes']) == 14
    # A negative Qdot must also fail; checking only positive heat release would miss it.
    write_field(tmp_path / '0.02/Qdot', [-1., 0.])
    assert not collect_fields(tmp_path, declaration)['physics_evidence']['passed']


def coupled_fixture():
    values = fields()
    evidence = nonreacting_field_evidence(values)
    evidence.update(zero_heat_release_observed=True, maximum_absolute_Qdot_by_time=[0., 0.])
    classical = dict(schema='h2-fixed-p-coupled-run-v1', arm='classical', source_kind='external_cfd',
        execution_mode='online_coupled', success=True, case_id='fixture', mesh_sha256='mesh',
        thermophysical_sha256='thermo', initial_conditions_sha256='init', boundary_conditions_sha256='bc',
        dt_schedule=[.01, .01], stopping_criteria={'rtol': 1.e-8, 'atol': 1.e-12}, end_time=.02,
        fixed_p_sha256='P', hardware={'cpu': 'fixture'}, execution_threads=1, pressure_gauge='absolute',
        continuity={'passed': True}, experiment_kind=VARIABLE_DENSITY_KIND,
        comparison_fields=list(FIELD_PROFILES[VARIABLE_DENSITY_KIND]), physics_evidence=evidence,
        evidence={'coupled_run_attested': True, 'solver_log_sha256': 'log', 'field_files_sha256': {'rho': 'hash'}},
        timing={'wall_seconds': 10., 'pressure_seconds': 6., 'foam_process_wall_seconds': 9.,
                'bridge_startup_seconds': 1., 'scope': 'fixture', 'pressure_scope': 'fixture'},
        fields=values, sample_times=[.01, .02])
    neural = deepcopy(classical)
    neural['arm'] = 'H_S'
    neural['evidence'].update(expert_checkpoint_sha256='checkpoint', neural_apply_calls=2)
    tolerance = dict(declared_before_evaluation=True, pressure_alignment='none',
                     fields={name: dict(atol=1.e-8, rtol=1.e-7) for name in FIELD_PROFILES[VARIABLE_DENSITY_KIND]})
    return classical, neural, tolerance


def test_coupled_inert_comparison_uses_thermophysical_and_all_inert_fields():
    result = compare_coupled_runs(*coupled_fixture())
    assert result['success']
    assert set(result['fields']) == set(FIELD_PROFILES[VARIABLE_DENSITY_KIND])
    assert 'H2' not in result['fields']
    assert result['timing']['classical']['foam_process_wall_seconds'] == 9.
    assert result['runs']['classical']['physics_evidence']['density_evolves']


@pytest.mark.parametrize('change', ['kind', 'profile', 'thermo', 'rho', 'static', 'fractions', 'sum', 'qdot', 'missing_evidence'])
def test_inert_comparison_cannot_claim_speedup_after_physics_or_provenance_failure(change):
    c, h, tol = coupled_fixture()
    if change == 'kind': h['experiment_kind'] = 'h2_combustion'
    elif change == 'profile': h['comparison_fields'].remove('rho')
    elif change == 'thermo': h['thermophysical_sha256'] = 'changed'
    elif change == 'rho': h['fields']['rho'][0, 0] = -1.
    elif change == 'static': h['fields']['rho'][1] = h['fields']['rho'][0]
    elif change == 'fractions': h['fields']['He'][0, 0] = 1.1
    elif change == 'sum': h['fields']['N2'][0, 0] -= .1
    elif change == 'qdot': h['physics_evidence']['maximum_absolute_Qdot_by_time'][0] = 1.
    elif change == 'missing_evidence': del h['physics_evidence']
    result = compare_coupled_runs(c, h, tol)
    assert not result['success'] and result['blockers']
    assert result['full_cfd_speedup'] is None and result['pressure_speedup'] is None


def test_service_phase_totals_preserve_nested_measured_costs():
    row = dict(service_seconds=.4, admission_seconds=.1,
               step=dict(setup_seconds=.1, solve_seconds=.1, total_seconds=.3,
                         stats=dict(feature_seconds=.02, verification_seconds=.03,
                                    recovery_seconds=.04, orchestration_seconds=.01)))
    totals = summarize_service_timing([row, deepcopy(row)])
    assert totals['service_seconds'] == .8
    assert totals['mg_total_seconds'] == .6
    assert totals['mg_setup_seconds'] == .2
    assert totals['mg_recovery_seconds'] == .08
    assert 'must not be added' in totals['phase_scope']
    row['step']['stats']['recovery_seconds'] = float('nan')
    with pytest.raises(ValueError, match='measured service phase'):
        summarize_service_timing([row])


def prepared_fixture(root):
    """Only hash/dictionary fixtures: no mesh creation or CFD execution."""
    value = contract()
    thermo = {
        'constant/physicalProperties': 'fixture thermophysical input',
        'constant/chemistryProperties': 'chemistry off;',
        'constant/combustionProperties': 'combustionModel none;',
    }
    for name, text in {**thermo, 'constant/polyMesh/faces': 'fixture mesh'}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    initial = [f'0/{name}' for name in ['He', 'N2', 'T', 'p', 'U']]
    (root / '0').mkdir()
    for name in initial:
        (root / name).write_text('FoamFile {format ascii;} internalField uniform 1; boundaryField {}')
    boundary = '\n'.join((root / name).read_text().split('boundaryField', 1)[1] for name in sorted(initial))
    (root / 'system').mkdir()
    (root / 'system/controlDict').write_text('startFrom startTime; startTime 0; adjustTimeStep no; '
        'writeFormat ascii; writeCompression off; endTime .02; deltaT .01; writeControl timeStep; writeInterval 1;')
    value.update(status='prepared_not_executed', foam_conversion_verified=True,
                 preparation_kind='native_thermophysical', thermophysical_files=list(thermo),
                 thermophysical_sha256=_digest_files(root, list(thermo)),
                 initial_conditions_sha256=_digest_files(root, initial),
                 boundary_conditions_sha256=hashlib.sha256(boundary.encode()).hexdigest(),
                 mesh_sha256=_digest_files(root, ['constant/polyMesh/faces']), end_time=.02, delta_t=.01)
    write_json(root / 'contract.json', value)
    return value


def test_prepared_inert_case_checks_native_inputs_without_chemkin_conversion(tmp_path):
    value = prepared_fixture(tmp_path)
    assert _verify_prepared(tmp_path)['preparation_kind'] == 'native_thermophysical'
    assert 'converted_chemistry_sha256' not in value
    (tmp_path / 'constant/physicalProperties').write_text('changed transport')
    with pytest.raises(ValueError, match='thermophysical_sha256'):
        _verify_prepared(tmp_path)


def test_prepared_inert_case_cannot_reenable_chemistry_even_with_updated_hash(tmp_path):
    value = prepared_fixture(tmp_path)
    (tmp_path / 'constant/chemistryProperties').write_text('chemistry on;')
    value['thermophysical_sha256'] = _digest_files(tmp_path, value['thermophysical_files'])
    write_json(tmp_path / 'contract.json', value)
    with pytest.raises(ValueError, match='disable chemistry'):
        _verify_prepared(tmp_path)


def study_settings(kind=VARIABLE_DENSITY_KIND):
    cfg = AdaptiveConfig(mg=MGConfig(mode='classical', strategy_name='jacobi_energymin_full__em5__v11',
                                     max_cycles=100, nn_levels=1, stencil_backend='csr'),
                         mode='research', branch='H_S', use_transfer=False, spatial=False,
                         gate_mode='open', use_learned_controller=False, inference_device='cpu')
    return dict(schema='h2-fixed-p-study-v1', solver=cfg.to_dict(), world_model_enabled=False,
                benchmark=dict(role='primary' if kind == VARIABLE_DENSITY_KIND else 'supplementary', experiment_kind=kind),
                torch_threads=1, seed=17, training=dict(updates=2, cycles=2, hidden=4, seed=17))


def test_inert_protocol_train_validation_freeze_test_preserves_kind_without_claiming_cfd(tmp_path, monkeypatch):
    # Synthetic numerical fixtures exercise the recording protocol only. Source
    # digest is pinned while other agents may edit unrelated source files.
    monkeypatch.setattr(W, 'source_digest', lambda: 'a' * 64)
    inputs = [fixture_recording(tmp_path / split, split, offset)
              for split, offset in [('train', 0), ('validation', 17), ('test', 33)]]
    config = tmp_path / 'config.json'
    write_json(config, study_settings())
    run = tmp_path / 'run'
    manifest = W.prepare(config, inputs, run)
    assert manifest['experiment_kind'] == VARIABLE_DENSITY_KIND
    assert manifest['benchmark']['role'] == 'primary'
    W.train(run)
    assert W.read(run / 'expert/status.json')['optimizer_updates'] == 2
    with pytest.raises(FileNotFoundError):
        W.evaluate(run, split='test', repeats=1)
    validation = W.evaluate(run, split='validation', repeats=1)
    assert validation['success']
    assert not validation['full_cfd_wall_clock_measured']
    assert not validation['physical_observables_validated']
    assert verify_dataset(run / 'data')['experiment_kind'] == VARIABLE_DENSITY_KIND
    W.freeze(run)
    assert W.check_freeze(run)
    result = W.evaluate(run, split='test', repeats=1)
    assert result['success']
    assert W.load_run(run)[3]['experiment_kind'] == VARIABLE_DENSITY_KIND
    assert not result['full_cfd_wall_clock_measured']
    assert result['diagnostic']['world_model_enabled'] is False


def test_inert_recordings_reject_h2_study_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(W, 'source_digest', lambda: 'a' * 64)
    config = tmp_path / 'config.json'
    write_json(config, study_settings('h2_combustion'))
    inputs = [fixture_recording(tmp_path / split, split, offset)
              for split, offset in [('train', 0), ('validation', 17)]]
    with pytest.raises(ValueError, match='configuration benchmark'):
        W.prepare(config, inputs, tmp_path / 'run')
    assert not (tmp_path / 'run/run_manifest.json').exists()


@pytest.mark.parametrize('source', ['config', 'run_manifest', 'run_config'])
@pytest.mark.parametrize('case_kind', [VARIABLE_DENSITY_KIND, 'h2_combustion'])
def test_manual_serve_rejects_cross_kind_before_expert_session_or_socket(tmp_path, monkeypatch, source, case_kind):
    """Direct CLI service entry has the same physics boundary as run_paired."""
    from adaptive_mg.v67.h2_fixed_p import service

    opposite = 'h2_combustion' if case_kind == VARIABLE_DENSITY_KIND else VARIABLE_DENSITY_KIND
    settings = study_settings(case_kind if source == 'run_manifest' else opposite)
    cfg = AdaptiveConfig.from_dict(settings['solver'])
    declaration = validate_case_contract(contract(case_kind))
    monkeypatch.setattr(W, 'read', lambda path: declaration)
    monkeypatch.setattr(W, 'configuration', lambda path: (settings, cfg))
    manifest = {'experiment_kind': opposite if source == 'run_manifest' else case_kind}
    monkeypatch.setattr(W, 'load_run', lambda path: (tmp_path, settings, cfg, manifest))

    def forbidden(*args, **kwargs):
        pytest.fail('cross-kind service reached expert loading, Session construction, or socket access')

    monkeypatch.setattr(W, 'load_expert', forbidden)
    monkeypatch.setattr(service, 'Session', forbidden)
    monkeypatch.setattr(service, 'serve', forbidden)
    argv = ['serve', '--case-contract', 'fixture-contract.json', '--socket', str(tmp_path / 'service.sock')]
    if source == 'config':
        argv += ['--config', 'fixture-config.json', '--mode', 'native']
        message = 'service case and configuration experiment kinds disagree'
    else:
        argv += ['--run-dir', str(tmp_path), '--mode', 'hs']
        message = ('service case and trained dataset experiment kinds disagree'
                   if source == 'run_manifest' else 'service case and configuration experiment kinds disagree')
    with pytest.raises(ValueError, match=message):
        W.main(argv)
    assert not (tmp_path / 'service.sock').exists()
