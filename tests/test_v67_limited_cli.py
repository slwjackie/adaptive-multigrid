"""Small integration checks for no-training selection and frozen-operator refit."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.inspection import preservation_manifest, verify_preservation
from adaptive_mg.v67.limited import (diagnostic_cases, forced_config, make_arm,
                                    run_benchmark, safe_output)
from adaptive_mg.v67.models import Components


def script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / name
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def archived_run(tmp_path):
    root = tmp_path / 'original'
    cfg = AdaptiveConfig()
    settings = {'seed': 13, 'hidden': 4, 'torch_threads': 1,
        'solver': cfg.to_dict(), 'validation_repeats': 1, 'stage_validation_cases': 5,
        'stage_time_margin': .03, 'train_sizes': [7], 'train_per_size': 1,
        'epochs': {'detector': 1, 'controller': 1}, 'detector_states': 1,
        'controller_states': 1, 'timing_repeats': 1}
    save_json(root / 'configuration.json', settings)
    initial = Components.create(hidden=4, seed=13)
    initial.save(root / 'controller/controller.pt')
    initial.save(root / 'checkpoint_v6_7_final.pt')
    save_json(root / 'certificate.json', {'validated': False, 'original': True})
    for stage in ('smoother', 'transfer', 'joint', 'hierarchical'):
        candidate = deepcopy(initial)
        with torch.no_grad():
            next(getattr(candidate, 'transfer' if stage == 'transfer' else 'smoother').parameters()).add_(.01)
        candidate.save(root / stage / 'candidate.pt')
        save_json(root / stage / 'status.json', {'acceptance': {'accepted': False}})
        stagecfg = replace(cfg, use_smoother=stage != 'transfer', use_transfer=stage != 'smoother',
                           spatial=stage == 'hierarchical', gate_mode='learned' if stage == 'hierarchical' else 'open')
        rows = []
        for index in range(5):
            def run(model, seconds):
                return {'converged': True, 'final_true_residual': cfg.mg.tolerance / 2,
                    'stopping_threshold': cfg.mg.tolerance, 'residual_history': [1., cfg.mg.tolerance / 2],
                    'relative_residual_history': [1., cfg.mg.tolerance / 2],
                    'executed_cycles': 3, 'wall_seconds': seconds, 'time_scope': 'cold',
                    'cache_key': f'operator{index}:{model.signature()}:' + json.dumps(stagecfg.certification_scope()),
                    'stats': {'neural_apply_calls': 3, 'learned_transfer_apply_calls': 2,
                              'accepted_neural_cycles': 2}}
            rows.append({'example': {'name': f'case{index}', 'case': {'n': 7}},
                         'runs': {'previous': [run(initial, 2.)], 'candidate': [run(candidate, 1.)]}})
        save_json(root / stage / 'validation/raw.json', rows)
    return root, settings, cfg


def test_archived_selection_never_trains_and_preserves_original(archived_run, tmp_path, monkeypatch):
    root, settings, cfg = archived_run
    cli = script('reevaluate_v6_7_candidates.py')
    def forbidden(*args, **kwargs):
        raise AssertionError('training or fresh timing started during archived reevaluation')
    monkeypatch.setattr(cli, 'accept_operator', forbidden)
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    original = preservation_manifest(root.rglob('*'))
    out = tmp_path / 'selected'
    argv = ['--research-dir', str(root), '--output-dir', str(out), '--stages', 'smoother']
    cli.main(argv)
    status = json.loads((out / 'smoother/status.json').read_text())
    assert status['stage_promoted'] and status['no_training']
    selected = Path(status['selected_checkpoint'])
    assert selected.is_absolute() and selected.exists()
    model = Components.load(selected)
    assert model.metadata['policy_stale']
    assert model.metadata['certificate']['validated'] is False
    assert model.generation_signature() == Components.load(root / 'smoother/candidate.pt').generation_signature()
    selected_before = selected.read_bytes()
    cli.main(argv + ['--resume'])
    assert selected.read_bytes() == selected_before
    assert verify_preservation(original)['all_unchanged']
    with pytest.raises(ValueError, match='[Rr]esume'):
        cli.main(argv + ['--resume', '--policy', 'strict_v1'])
    selected.unlink()
    with pytest.raises(FileNotFoundError, match='[Ss]elected'):
        cli.main(argv + ['--resume'])


def test_archived_candidate_must_match_measured_weights(archived_run, tmp_path):
    root, _, _ = archived_run
    candidate = Components.load(root / 'smoother/candidate.pt')
    with torch.no_grad(): next(candidate.smoother.parameters()).add_(.02)
    candidate.save(root / 'smoother/candidate.pt')
    with pytest.raises(ValueError, match='candidate|signature|recorded|comparison'):
        script('reevaluate_v6_7_candidates.py').main([
            '--research-dir', str(root), '--output-dir', str(tmp_path / 'different'), '--stages', 'smoother'])


def test_missing_candidate_is_explicit_and_final_is_not_substituted(archived_run, tmp_path):
    root, _, _ = archived_run
    (root / 'smoother/candidate.pt').unlink()
    out = tmp_path / 'missing'
    script('reevaluate_v6_7_candidates.py').main([
        '--research-dir', str(root), '--output-dir', str(out), '--stages', 'smoother'])
    status = json.loads((out / 'smoother/status.json').read_text())
    assert status['status'] == 'missing_candidate'
    assert status['stage_promoted'] is False
    assert not (out / 'smoother/selected.pt').exists()


def test_completed_resume_rejects_changed_hierarchical_entry(archived_run, tmp_path):
    root, _, _ = archived_run
    cli = script('reevaluate_v6_7_candidates.py')
    argv = ['--research-dir', str(root), '--output-dir', str(tmp_path / 'hierarchical'), '--stages', 'hierarchical']
    cli.main(argv)
    entry = Components.load(root / 'controller/controller.pt')
    with torch.no_grad(): next(entry.controller.parameters()).add_(.02)
    entry.save(root / 'controller/controller.pt')
    with pytest.raises(ValueError, match='[Rr]esume|entry|signature|recorded'):
        cli.main(argv + ['--resume'])


def test_benchmark_resume_skips_completed_measurements_and_rejects_changed_config(archived_run, tmp_path, monkeypatch):
    root, settings, cfg = archived_run
    examples, provenance = diagnostic_cases(root, settings, 'smoke', limit=1, sizes=[7])
    checkpoint = root / 'smoother/candidate.pt'
    arms = {'classical': make_arm(None, forced_config(cfg, 'C'), 'C', 'pure classical'),
            'smoother': make_arm(checkpoint, forced_config(cfg, 'H_S'), 'H_S', 'forced component + safety')}
    out = tmp_path / 'benchmark'
    report = run_benchmark(examples, provenance, arms, out, root, repeats=2, warmups=0)
    assert (out / 'comparison.csv').exists()
    assert (out / 'branch_statistics.json').exists()
    assert all(row['repeats'] == 2 for row in report['table'])
    assert report['provenance']['independent_certification_evidence'] is False
    def forbidden(*args, **kwargs): raise AssertionError('completed case was timed again')
    monkeypatch.setattr('adaptive_mg.v67.limited.measured', forbidden)
    resumed = run_benchmark(examples, provenance, arms, out, root, repeats=2, warmups=0, resume=True)
    assert resumed['table'] == report['table']
    saved_case = out / 'cases/0000.json'
    original_case = saved_case.read_text()
    changed = json.loads(original_case)
    changed['runs']['smoother'][0]['requested_branch'] = 'H_SP'
    save_json(saved_case, changed)
    with pytest.raises(ValueError, match='[Rr]esume'):
        run_benchmark(examples, provenance, arms, out, root, repeats=2, warmups=0, resume=True)
    saved_case.write_text(original_case)
    arms['smoother']['config'] = replace(arms['smoother']['config'], gate_on=.7)
    with pytest.raises(ValueError, match='[Rr]esume'):
        run_benchmark(examples, provenance, arms, out, root, repeats=2, warmups=0, resume=True)


def test_reconstructed_audit_rejects_changed_case_digest(archived_run):
    root, settings, _ = archived_run
    settings.update(audit_sizes=[7], audit_per_size=1)
    cases = dataset([7], 1, settings['seed'] + 200002)
    stored = [e.manifest() for e in cases]
    save_json(root / 'split_manifest.json', {'splits': {'audit': stored}})
    _, provenance = diagnostic_cases(root, settings, 'audit')
    assert provenance['scope'] == 'diagnostic re-evaluation of previously inspected audit'
    stored[0]['operator_digest'] = 'different'
    save_json(root / 'split_manifest.json', {'splits': {'audit': stored}})
    with pytest.raises(ValueError, match='digest'):
        diagnostic_cases(root, settings, 'audit')


def test_tiny_policy_refit_keeps_operator_tensors_and_invalidates_certificate(archived_run, tmp_path):
    root, _, _ = archived_run
    checkpoint = root / 'smoother/candidate.pt'
    before = Components.load(checkpoint).generation_signature()
    immutable = preservation_manifest(root.rglob('*'))
    out = tmp_path / 'refit'
    cli = script('refit_v6_7_limited_policy.py')
    argv = ['--research-dir', str(root), '--checkpoint', str(checkpoint),
            '--branch', 'H_S', '--output-dir', str(out), '--limit', '1']
    cli.main(argv)
    model = Components.load(out / 'calibrated.pt')
    status = json.loads((out / 'status.json').read_text())
    assert model.generation_signature() == before
    assert status['operator_weights_unchanged']
    assert not status['calibrated']
    assert model.metadata['policy_stale']
    assert model.metadata['certificate']['validated'] is False
    assert set(model.metadata['branch_calibration']) == {'H_S'}
    assert Path(status['checkpoint']).is_absolute()
    cli.main(argv + ['--resume'])
    assert verify_preservation(immutable)['all_unchanged']


def test_all_cli_outputs_reject_original_subtrees_and_parent(tmp_path):
    research = tmp_path / 'research'; research.mkdir()
    link = tmp_path / 'link'; link.symlink_to(research, target_is_directory=True)
    for output in (research, research / 'child', link / 'child', tmp_path):
        with pytest.raises(ValueError): safe_output(output, research)


def test_empty_refit_cannot_validate_inherited_stale_controller(archived_run, tmp_path, monkeypatch):
    root, _, _ = archived_run
    checkpoint = root / 'smoother/candidate.pt'
    model = Components.load(checkpoint)
    model.metadata.update(controller_trained=True, policy_stale=True)
    model.save(checkpoint)
    cli = script('refit_v6_7_limited_policy.py')
    def empty_table(model, examples, cfg, settings, output):
        save_json(Path(output) / 'action_table.json', [])
        return []
    monkeypatch.setattr(cli, 'temporal_table', empty_table)
    out = tmp_path / 'empty-refit'
    cli.main(['--research-dir', str(root), '--checkpoint', str(checkpoint),
              '--branch', 'H_P', '--output-dir', str(out)])
    status = json.loads((out / 'status.json').read_text())
    selected = Components.load(out / 'calibrated.pt')
    assert not status['calibrated']
    assert selected.metadata['policy_stale']
    assert not selected.metadata['branch_calibration']['H_P']['validated']
