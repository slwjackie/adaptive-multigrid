"""End-to-end protocol fixtures, explicitly NOT H2 combustion simulations."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg import DiffusionCase, assemble_stiffness, MGConfig
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.world_model.adapter import FinalizedLduRecorder
from adaptive_mg.v67.world_model.data import digest, file_hash, write_json
from adaptive_mg.v67.h2_fixed_p import study as W
from adaptive_mg.v67.h2_fixed_p.data import read


def settings(*, device='cpu'):
    cfg = AdaptiveConfig(mg=MGConfig(mode='classical', strategy_name='jacobi_energymin_full__em5__v11',
                                     max_cycles=100, nn_levels=1, stencil_backend='csr'),
                         mode='research', branch='H_S', use_transfer=False, spatial=False,
                         gate_mode='open', use_learned_controller=False, inference_device=device)
    return dict(schema='h2-fixed-p-study-v1', solver=cfg.to_dict(), world_model_enabled=False,
                torch_threads=1, seed=17, training=dict(updates=2, cycles=2, hidden=4, seed=17))


def make_fixture_recording(path, split, offset):
    """Exercise the external format with truthfully identified test matrices.

    Physics labels only test schema admission. Producer and snapshot metadata
    mark the actual origin, and no result is reported as real flame validation.
    """
    physics = dict(fuel='H2', oxidizer='air', dimension=2, fixed_grid=True)
    contract = dict(schema='h2-fixed-p-case-v1', case_id='fixture_' + split,
                    case_group='test_only_' + split, split=split, shape=[7, 7],
                    chemistry_sha256='0' * 64, physics=physics,
                    test_fixture_only=True, actual_combustion_simulation=False)
    recorder = FinalizedLduRecorder(path,
        producer={'kind': 'unit_test_fixture', 'actual_CFD': False}, physics=physics)
    write_json(path / 'case_contract.json', contract)
    for index in range(2):
        n = 7
        a = assemble_stiffness(DiffusionCase(n, epsilon=.3 + offset * .01,
            angle_deg=20 + offset + index * 4, contrast=3 + index * .2, pattern='channel'))
        upper = sp.triu(a, k=1).tocoo()
        probes = np.column_stack((np.ones(n * n), np.linspace(-1, 1, n * n)))
        b = 1 + np.sin(np.linspace(0, 5, n * n)) + index * .2
        recorder.record(trajectory=contract['case_id'], case_group=contract['case_group'], split=split,
            index=index, time=index * .01, shape=(n, n), mesh_id='fixture-grid', boundary_id='anchored',
            diag=a.diagonal(), lower_addr=upper.row, upper_addr=upper.col,
            lower=upper.data, upper=upper.data, b=b, x0=np.linspace(0, .002, n * n),
            probe_vectors=probes, probe_products=a @ probes, structured_to_native=np.arange(n * n),
            boundary_finalized=True, nullspace='none', coupled_interfaces=0,
            context={'test_fixture_only': True, 'actual_CFD': False,
                     'case_contract_digest': digest(contract), 'physics': physics})
    return path


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # Source identity tests below exercise the real hash separately. Fix it here
    # so concurrently authored source files cannot disrupt protocol unit tests.
    monkeypatch.setattr(W, 'source_digest', lambda: 'a' * 64)
    config = tmp_path / 'config.json'
    write_json(config, settings())
    inputs = [make_fixture_recording(tmp_path / split, split, offset)
              for split, offset in [('train', 0), ('validation', 15), ('test', 30)]]
    run = tmp_path / 'run'
    W.prepare(config, inputs, run)
    return run


@pytest.fixture
def trained(prepared):
    W.train(prepared)
    return prepared


def test_prepare_train_validate_freeze_test_pipeline_is_honest(trained):
    run = trained
    status = read(run / 'expert/status.json')
    assert status['optimizer_updates'] == 2
    assert status['initial_signatures']['smoother'] != status['final_signatures']['smoother']
    model = W.load_expert(run, W.load_run(run)[2])
    assert model.metadata['actual_exported_rhs']
    assert model.metadata['training_source_kind'] == 'external_cfd'
    assert status['cfd_validation_completed'] is False
    with pytest.raises(FileNotFoundError):
        W.evaluate(run, split='test', repeats=1)
    validation = W.evaluate(run, split='validation', repeats=1)
    assert validation['success']
    assert all(row['trained_neural_smoothing_used'] for row in validation['rows'])
    assert not validation['full_cfd_wall_clock_measured']
    assert not validation['physical_observables_validated']
    assert not validation['combustion_authenticity_verified']
    W.freeze(run)
    W.check_freeze(run)
    with pytest.raises(ValueError, match='frozen'):
        W.train(run)
    test = W.evaluate(run, split='test', repeats=1)
    assert test['success']
    assert test['diagnostic']['world_model_enabled'] is False
    with pytest.raises(FileExistsError, match='already exists'):
        W.evaluate(run, split='test', repeats=1)
    assert W.check_freeze(run)  # writing final reports does not alter source/frozen inputs


@pytest.mark.parametrize('which', ['configuration', 'source', 'provenance'])
def test_run_rejects_configuration_source_and_case_provenance_tampering(prepared, monkeypatch, which):
    run = prepared
    if which == 'configuration':
        config = read(run / 'configuration.json')
        config['training']['updates'] += 1
        write_json(run / 'configuration.json', config)
    elif which == 'source':
        monkeypatch.setattr(W, 'source_digest', lambda: 'b' * 64)
    else:
        case = run / 'data/sources/0000/case_contract.json'
        data = read(case)
        data['chemistry_sha256'] = '1' * 64
        write_json(case, data)
    with pytest.raises(ValueError, match='changed'):
        W.load_run(run)


@pytest.mark.parametrize('which', ['checkpoint', 'training_records', 'training_manifest'])
def test_modified_training_evidence_is_rejected(trained, which):
    run = trained
    cfg = W.load_run(run)[2]
    if which == 'checkpoint':
        with (run / 'expert/candidate.pt').open('ab') as stream:
            stream.write(b'tampered')
    elif which == 'training_records':
        records = read(run / 'expert/training.json')
        records[0]['gradient_norm'] = 0
        write_json(run / 'expert/training.json', records)
    else:
        evidence = read(run / 'expert/training_manifest.json')
        evidence['trajectories'][0]['samples'][0]['rhs_digest'] = '0' * 64
        write_json(run / 'expert/training_manifest.json', evidence)
    with pytest.raises(ValueError):
        W.load_expert(run, cfg)


def test_candidate_cannot_be_relabelled_for_a_different_imported_rhs(trained):
    run = trained
    cfg = W.load_run(run)[2]
    manifest = read(run / 'data/sequence_manifest.json')
    entry = next(t for t in manifest['trajectories'] if t['split'] == 'train')['snapshots'][0]
    from adaptive_mg.v67.world_model.data import load_snapshot, save_snapshot
    from dataclasses import replace
    path = run / 'data' / entry['path']
    state = load_snapshot(path)
    save_snapshot(path, replace(state, b=state.b + np.linspace(-.1, .2, len(state.b))))
    entry['sha256'] = file_hash(path)
    write_json(run / 'data/sequence_manifest.json', manifest)
    # Even if an operator rewrites the dataset file hash, load_expert binds the
    # candidate evidence to actual recorded TRAIN b/x0 values.
    run_manifest = read(run / 'run_manifest.json')
    run_manifest['dataset_sha256'] = file_hash(run / 'data/sequence_manifest.json')
    write_json(run / 'run_manifest.json', run_manifest)
    with pytest.raises(ValueError, match='training snapshot'):
        W.load_expert(run, cfg)


def test_live_expert_provenance_is_streamed_without_spd_readmission(trained, monkeypatch):
    from adaptive_mg.v67.world_model import data as D
    import scipy.linalg
    import scipy.sparse.linalg
    cfg = W.load_run(trained)[2]
    def forbidden(*args, **kwargs):
        pytest.fail('service startup must not load all snapshots or repeat SPD admission')
    monkeypatch.setattr(W, 'load_trajectories', forbidden)
    monkeypatch.setattr(D, 'load_snapshot', forbidden)
    monkeypatch.setattr(scipy.linalg, 'cholesky', forbidden)
    monkeypatch.setattr(scipy.sparse.linalg, 'eigsh', forbidden)
    original_load = np.load
    opened, active, peak = [], [0], [0]
    class TrackedArchive:
        def __init__(self, path, **kwargs):
            assert kwargs.get('allow_pickle') is False
            self.archive = original_load(path, **kwargs)
            opened.append(Path(path))
        def __enter__(self):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            return self.archive
        def __exit__(self, *_):
            self.archive.close()
            active[0] -= 1
    monkeypatch.setattr(np, 'load', TrackedArchive)
    model = W.load_expert(trained, cfg)
    assert model.metadata['optimizer_updates'] == 2
    assert peak == [1] and active == [0]
    manifest = read(trained / 'data/sequence_manifest.json')
    expected = {trained / 'data' / e['path'] for t in manifest['trajectories']
                if t['split'] == 'train' for e in t['snapshots']}
    assert set(opened) == expected


def test_streamed_training_file_hash_is_checked_before_array_loading(trained, monkeypatch):
    cfg = W.load_run(trained)[2]
    manifest = read(trained / 'data/sequence_manifest.json')
    entry = next(t for t in manifest['trajectories'] if t['split'] == 'train')['snapshots'][0]
    path = trained / 'data' / entry['path']
    with path.open('ab') as stream:
        stream.write(b'tampered')
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('unverified snapshot opened'))
    with pytest.raises(ValueError, match='snapshot hash changed'):
        W.load_expert(trained, cfg)


def test_freeze_requires_bound_validation_and_protects_all_training_records(trained):
    run = trained
    with pytest.raises(ValueError, match='validation'):
        W.freeze(run)
    W.evaluate(run, repeats=1)
    report_path = run / 'validation/replay.json'
    report = read(report_path)
    saved = deepcopy(report)
    report['evidence']['checkpoint_sha256'] = '0' * 64
    write_json(report_path, report)
    with pytest.raises(ValueError, match='validation evidence'):
        W.freeze(run)
    write_json(report_path, saved)
    W.freeze(run)
    frozen = read(run / 'freeze.json')
    assert 'expert/training.json' in frozen['artifacts']
    assert 'expert/initial.pt' in frozen['artifacts']
    trace = run / 'expert/training.json'
    trace.write_text(trace.read_text() + '\n')
    with pytest.raises(ValueError, match='frozen artifact changed'):
        W.check_freeze(run)


def test_empty_freeze_manifest_cannot_bypass_test_gate(trained):
    write_json(trained / 'freeze.json', dict(schema='h2-fixed-p-study-v1',
               source_digest=W.source_digest(), world_model_enabled=False, artifacts={}))
    with pytest.raises(ValueError, match='freeze manifest'):
        W.evaluate(trained, split='test', repeats=1)


def test_evaluate_cli_and_api_reject_tune_without_mutation(prepared, capsys):
    with pytest.raises(SystemExit) as error:
        W.main(['evaluate', '--run-dir', str(prepared), '--split', 'tune'])
    assert error.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
    with pytest.raises(ValueError, match='validation or test'):
        W.evaluate(prepared, split='tune')
    assert not (prepared / 'tune').exists()


def test_auto_inference_configuration_can_load_its_genuinely_trained_candidate(prepared, monkeypatch):
    # Exercise the CPU resolution branch irrespective of developer GPU access.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    config = read(prepared / 'configuration.json')
    config['solver']['inference_device'] = 'auto'
    write_json(prepared / 'configuration.json', config)
    manifest = read(prepared / 'run_manifest.json')
    from adaptive_mg.v67.world_model.data import digest
    manifest['configuration_digest'] = digest(config)
    write_json(prepared / 'run_manifest.json', manifest)
    W.train(prepared)
    model = W.load_expert(prepared, W.load_run(prepared)[2])
    assert model.metadata['optimizer_updates'] == 2


def test_source_digest_hashes_code_not_outputs_or_cache(tmp_path, monkeypatch):
    package = tmp_path / 'src/adaptive_mg'
    study = package / 'v67/h2_fixed_p/study.py'
    study.parent.mkdir(parents=True)
    study.write_text('source = 1\n')
    monkeypatch.setattr(W, '__file__', str(study))
    before = W.source_digest()
    for directory in ['artifacts', 'outputs', '__pycache__', '.cache', 'build', 'dist']:
        output = package / directory
        output.mkdir()
        (output / 'generated.py').write_text('not_package_source = 1\n')
    (study.parent / 'status.json').write_text('{"complete": true}')
    assert W.source_digest() == before
    study.write_text('source = 2\n')
    assert W.source_digest() != before
    before_native = W.source_digest()
    native = tmp_path / 'integrations/openfoam13/AdaptiveFixedP/AdaptiveFixedP.C'
    native.parent.mkdir(parents=True)
    native.write_text('// bridge source\n')
    assert W.source_digest() != before_native
    pinned = W.source_digest()
    (native.parent / 'generated.o').write_bytes(b'compiler output')
    assert W.source_digest() == pinned


@pytest.mark.parametrize('invalid', ['world_model', 'float64'])
def test_unsupported_experiment_modes_are_rejected(tmp_path, invalid):
    value = settings()
    if invalid == 'world_model':
        value['world_model_enabled'] = True
    else:
        value['solver']['inference_dtype'] = 'float64'
    path = tmp_path / 'config.json'
    write_json(path, value)
    with pytest.raises(ValueError, match='World Model|FP32'):
        W.configuration(path)
