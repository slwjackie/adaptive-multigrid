import json
from pathlib import Path

import pytest
import torch

from adaptive_mg.v67.inspection import (assert_new_output, diagnose_original_stage,
    preservation_manifest, recover_initial_entry, tensor_weight_signatures, verify_preservation)
from adaptive_mg.v67.models import Components


def run(ok, seconds, signature='old', accepted=1):
    return {'converged': ok, 'wall_seconds': seconds, 'cache_key': f'operator:{signature}:{{}}',
            'stats': {'accepted_neural_cycles': accepted}}


def test_original_both_failure_is_visible_and_blocks_original_rule():
    rows = [{'example': {'name': 'success'}, 'runs': {'previous': [run(True, 2)],
                                                  'candidate': [run(True, 1, 'new')]}},
            {'example': {'name': 'both-fail'}, 'runs': {'previous': [run(False, .01)],
                                                    'candidate': [run(False, .001, 'new')]}}]
    report = diagnose_original_stage('smoother', rows, {'stage_time_margin': .03})
    assert report['both_failed_case_ids'] == ['both-fail']
    assert report['geometric_speedup_vs_incumbent'] == pytest.approx(2)
    assert report['original_failed_conditions'] == ['all_cases_common_success']
    assert report['recomputed_original_accepted'] is False


def test_some_failed_repeats_are_not_cherry_picked():
    rows = [{'example': {'name': 'partial'}, 'runs': {'previous': [run(True, 4), run(True, 4)],
        'candidate': [run(True, .1, 'new'), run(False, .01, 'new')]}}]
    report = diagnose_original_stage('smoother', rows, {})
    assert report['common_successes'] == 0
    assert report['new_failure_case_ids'] == ['partial']
    assert report['geometric_speedup_vs_incumbent'] is None


def test_tensor_weight_signature_ignores_metadata():
    model = Components.create(hidden=4, seed=4)
    before = tensor_weight_signatures(model)
    model.metadata['smoother_trained'] = True
    assert tensor_weight_signatures(model) == before
    with torch.no_grad():
        next(model.smoother.parameters()).add_(1)
    assert tensor_weight_signatures(model)['smoother'] != before['smoother']


def test_output_boundary_handles_symlinks(tmp_path):
    research = tmp_path / 'research'; research.mkdir()
    link = tmp_path / 'alias'; link.symlink_to(research, target_is_directory=True)
    with pytest.raises(ValueError): assert_new_output(research, link / 'new')
    assert assert_new_output(research, tmp_path / 'new') == tmp_path / 'new'


def test_preservation_manifest_detects_changes(tmp_path):
    checkpoint = tmp_path / 'candidate.pt'; checkpoint.write_bytes(b'original')
    manifest = preservation_manifest([checkpoint])
    assert verify_preservation(manifest)['all_unchanged']
    checkpoint.write_bytes(b'changed')
    assert not verify_preservation(manifest)['all_unchanged']


def test_seed_reconstruction_requires_every_archived_entry_signature(tmp_path):
    model = Components.create(hidden=4, seed=13)
    (tmp_path / 'configuration.json').write_text(json.dumps({'hidden': 4, 'seed': 13}))
    for stage in ('smoother', 'transfer', 'joint'):
        path = tmp_path / stage / 'validation'; path.mkdir(parents=True)
        (path.parent / 'status.json').write_text(json.dumps({'acceptance': {'accepted': False}}))
        (path / 'raw.json').write_text(json.dumps([{'runs': {'previous': [run(True, 1, model.signature())]}}]))
    restored, evidence = recover_initial_entry(tmp_path)
    assert restored is not None and evidence['verified']
    assert evidence['original_snapshot_exists'] is False
    path = tmp_path / 'joint/validation/raw.json'
    path.write_text(json.dumps([{'runs': {'previous': [run(True, 1, 'different')]}}]))
    restored, evidence = recover_initial_entry(tmp_path)
    assert restored is None and not evidence['verified']
    assert not list(tmp_path.rglob('*.pt'))
