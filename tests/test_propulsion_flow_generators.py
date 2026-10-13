"""Generator physics arithmetic and native-preparation protocol, not CFD results."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from adaptive_mg.v67.h2_fixed_p.data import validate_case_contract
from adaptive_mg.v67.h2_fixed_p.fullrun import read_internal_field, _verify_prepared

REPO = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / 'integrations/openfoam13/case' / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


jet = module('test_crossflow_generator', 'generate_crossflow.py')
slosh = module('test_sloshing_generator', 'generate_sloshing.py')
case_runner = module('test_new_flow_case_runner', 'run_case.py')


def args(generator, path, *extra):
    return generator.parser().parse_args(['--output', str(path), '--case-id', 'unit-flow',
                                         '--case-group', 'unit-flow', '--split', 'train', *extra])


def test_crossflow_complementary_species_and_impermeable_noninjecting_wall(tmp_path):
    a = args(jet, tmp_path / 'case')
    root = jet.generate(a)
    contract = validate_case_contract(json.loads((root/'contract.json').read_text()))
    assert contract['experiment_kind'] == 'h2_crossflow_nonreacting'
    assert contract['physics']['oxidizer'] == 'none' and contract['reaction_count'] == 0
    h2 = read_internal_field(root/'0/H2', a.nx*a.ny)
    n2 = read_internal_field(root/'0/N2', a.nx*a.ny)
    np.testing.assert_allclose(h2+n2, 1.)
    assert not (root/'0/He').exists()
    assert 'species (H2 N2)' in (root/'constant/thermo.compressibleGas').read_text()
    for name in ('H2', 'N2', 'T'):
        text = (root/'0'/name).read_text()
        assert 'codedMixed' in text and 'this->refGrad() = 0' in text
        assert 'this->valueFraction()[i] = Foam::mag(z)<1 ? 1 : 0' in text
    assert jet.slot_profile(a, 0.) == jet.slot_profile(a, a.width) == 0.
    assert jet.slot_profile(a, a.jet_center) == 1.
    for time in np.linspace(0., a.end_time, 201):
        u, y = jet.jet_state(a, time)
        assert 0 < u <= a.jet_speed*(1+a.forcing_amplitude)
        assert 0 < y <= 1.
    for s in (.01, .10, .5, .95):
        assert jet.jet_state(a, s*a.end_time) == (a.jet_speed, a.light_mass_fraction)
    with pytest.raises(FileExistsError):
        jet.generate(a)


def test_slosh_discrete_volume_hydrostatic_pressure_and_property_binding(tmp_path):
    a = args(slosh, tmp_path / 'case', '--fill-fraction', '.43')
    root = slosh.generate(a)
    contract = validate_case_contract(json.loads((root/'contract.json').read_text()))
    alpha = read_internal_field(root/'0/alpha.lox', a.nx*a.ny)
    p = read_internal_field(root/'0/p', a.nx*a.ny)
    prgh = read_internal_field(root/'0/p_rgh', a.nx*a.ny)
    np.testing.assert_allclose(alpha.mean(), .43, atol=1e-14)
    assert 0 <= alpha.min() < alpha.max() <= 1.
    rho = alpha*slosh.PROPERTIES['lox']['rho'] + (1-alpha)*slosh.PROPERTIES['he']['rho']
    y = np.repeat((np.arange(a.ny)+.5)*a.height/a.ny, a.nx)
    np.testing.assert_allclose(p, prgh-rho*a.gravity*y, atol=1e-9)
    assert p[-1] == contract['pressure_reference']['value'] == a.pressure
    assert contract['initial_liquid_fraction'] == pytest.approx(alpha.mean())
    assert contract['physics']['phase_change'] is False
    assert contract['physics']['gas_phase_approximation'] == 'pure_helium'
    assert contract['pressure_field'] == 'p_rgh'
    assert contract['natural_slosh_periods'] > 4
    assert not (root/'0/T').exists() and not (root/'constant/chemistryProperties').exists()
    assert 'p_rghFinal { $p_rgh; }' in (root/'system/fvSolution').read_text()
    assert 'pRefCell 960;' in (root/'system/fvSolution').read_text()
    assert 'nSubCycles 2;' in (root/'system/fvSolution').read_text()
    assert 'type fixedFluxPressure;' in (root/'0/p_rgh').read_text()
    basis = json.loads((root/'constant/property_basis.json').read_text())
    assert basis['version'] == '7.2.0' and basis['pressure_Pa'] == a.pressure


def test_tank_velocity_table_differentiates_to_acceleration_with_true_quiet_gaps(tmp_path):
    a = args(slosh, tmp_path/'unused', '--phase', '1.4', '--pulse-shift', '.03')
    values = np.asarray(slosh.motion_table(a))
    acceleration = np.diff(values[:, 1]) / a.delta_t
    midpoints = .5*(values[:-1, 0]+values[1:, 0])
    np.testing.assert_allclose(acceleration, [slosh.tank_acceleration(a,t) for t in midpoints], atol=1e-13)
    quiet = np.array([slosh.c.pulse_envelope(t,a.end_time,a.pulse_shift)==0 for t in midpoints])
    assert quiet.sum() > 100
    np.testing.assert_array_equal(acceleration[quiet], 0.)
    assert np.max(np.abs(acceleration)) > .1
    assert np.max(np.abs(acceleration)) <= a.acceleration_amplitude


@pytest.mark.parametrize('generator,extra', [
    (jet, ['--delta-t', '.0001']), (jet, ['--composition-amplitude', '.3']),
    (jet, ['--jet-width', '.001']), (jet, ['--jet-center', '.001']),
    (jet, ['--nx', '15']), (slosh, ['--inlet-temperature', '300']),
    (slosh, ['--pressure', '101325']), (slosh, ['--fill-fraction', '.99']),
    (slosh, ['--gravity', '0']), (slosh, ['--acceleration-amplitude', '5']),
])
def test_invalid_physics_rejected_before_case_creation(tmp_path, generator, extra):
    with pytest.raises(ValueError):
        generator.generate(args(generator, tmp_path/'invalid', *extra))
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('generator', [jet, slosh])
def test_new_native_case_preparation_never_calls_chemkin(tmp_path, monkeypatch, generator):
    root = generator.generate(args(generator, tmp_path/'case'))
    invoked = []
    def fake_command(case, argv, log):
        invoked.append(argv[0])
        (case/log).write_text('Mesh OK.\n' if argv[0]=='checkMesh' else 'dictionary fixture\n')
        if argv[0]=='blockMesh':
            (case/'constant/polyMesh').mkdir()
            (case/'constant/polyMesh/faces').write_text('mesh fixture, not real CFD')
    monkeypatch.setenv('WM_PROJECT_VERSION', '13')
    monkeypatch.setattr(case_runner.shutil, 'which', lambda name: '/fixture/'+name)
    monkeypatch.setattr(case_runner, 'command', fake_command)
    monkeypatch.setattr(sys, 'argv', ['run_case.py', '--case', str(root), '--prepare-only'])
    case_runner.main()
    assert invoked == ['blockMesh', 'checkMesh']
    prepared = _verify_prepared(root)
    assert prepared['status'] == 'prepared_not_executed'
    assert 'converted_chemistry_sha256' not in prepared
    # Every constant property/forcing input is bound by the preparation hash.
    changed = 'constant/fvModels' if generator is slosh else 'constant/thermo.compressibleGas'
    (root/changed).write_text((root/changed).read_text()+'\n// changed\n')
    with pytest.raises(ValueError, match='thermophysical'):
        _verify_prepared(root)
