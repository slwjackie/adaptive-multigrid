"""Fixed-parent provenance, leakage protection and real training contracts."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.fixed_classical import (
    FixedClassicalBaseline, FIXED_STRATEGY_NAME, load_baseline,
)
from adaptive_mg.v67 import research_data as data
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.research_training import sample_config, train_expert
from adaptive_mg.v67.unet_transfer import CompactTransferStudent


def config():
    return AdaptiveConfig(
        mg=MGConfig(mode='classical', strategy_name=FIXED_STRATEGY_NAME,
                    em_fixed_iterations=True, stencil_backend='csr'),
        mode='research', branch='H_P', use_smoother=False, use_transfer=True,
        transfer_levels='all', use_learned_controller=False,
        spatial=False, gate_mode='open', break_even=False,
    )


def plan():
    result = data.make_research_plan(seed=827191, smoke=False)
    for spec in result['splits'].values():
        spec.update(sizes=[7], families=['variable_coefficient'], per_family=1, count=1)
    return result


def development(tmp_path):
    cfg = config()
    baseline = FixedClassicalBaseline.from_config(cfg)
    history = tmp_path / 'history'
    history.mkdir()
    target = tmp_path / 'data'
    examples, validation, manifest = data.prepare_development_data(
        target, baseline, plan=plan(), historical_roots=[history])
    return cfg, baseline, target, examples, validation, manifest


def test_frozen_contract_binds_all_numerical_settings_and_roundtrips(tmp_path):
    cfg = config()
    baseline = FixedClassicalBaseline.from_config(cfg)
    path = tmp_path / 'baseline.json'
    path.write_text(json.dumps(baseline.to_dict()))
    assert load_baseline(path) == baseline
    baseline.validate_config(replace(cfg, branch='C', mode='classical'))
    baseline.validate_config(replace(cfg, mg=replace(cfg.mg, verbose=True)))
    for key, value in [('tolerance', 1e-7), ('coarsest_n', 1),
                       ('em_rtol', 1e-6), ('em_fixed_iterations', False),
                       ('near_tolerance_factor', 5.), ('native_threads', 2)]:
        with pytest.raises(ValueError, match='frozen baseline'):
            baseline.validate_config(replace(cfg, mg=replace(cfg.mg, **{key: value})))
    assert baseline.selection_record() == {
        'kind': 'fixed_classical', 'strategy_name': FIXED_STRATEGY_NAME,
        'baseline_digest': baseline.digest(),
    }
    malformed = baseline.to_dict()
    malformed['mg']['pre_steps'] = 1
    with pytest.raises(ValueError, match='Fixed classical'):
        FixedClassicalBaseline.from_dict(malformed)


def test_fixed_data_restore_and_training_selection_fail_closed(tmp_path):
    cfg, baseline, output, examples, validation, manifest = development(tmp_path)
    assert manifest['baseline_digest'] == baseline.digest()
    assert 'rules' not in manifest and 'rules_digest' not in manifest
    restored, _, loaded = data.load_development_data(output, baseline)
    assert loaded == manifest
    for original, recovered in zip(examples, restored):
        assert not hasattr(original, 'strong_selection')
        assert recovered.classical_selection == baseline.selection_record()
        np.testing.assert_array_equal(original.a.toarray(), recovered.a.toarray())
        for branch in ('C', 'H_P'):
            assert sample_config(recovered, cfg, baseline, branch).mg == cfg.mg
    changed_cfg = replace(cfg, mg=replace(cfg.mg, tolerance=2e-8))
    changed_baseline = FixedClassicalBaseline.from_config(changed_cfg)
    with pytest.raises(ValueError, match='baseline/selector changed'):
        data.load_development_data(output, changed_baseline)
    examples[0].classical_selection['strategy_name'] = 'line_alt_bilinear_full'
    with pytest.raises(ValueError, match='saved fixed classical'):
        sample_config(examples[0], cfg, baseline, 'H_P')


def test_fixed_baseline_freeze_and_final_data_keep_same_parent(tmp_path):
    _, baseline, output, examples, validation, manifest = development(tmp_path)
    parent_file = tmp_path / 'fixed_classical.json'
    parent_file.write_text(json.dumps(baseline.to_dict()))
    checkpoint = tmp_path / 'candidate.pt'
    checkpoint.write_bytes(b'test-checkpoint')
    source = tmp_path / 'source.py'
    source.write_text('# test source\n')
    frozen = data.freeze_research(output, {'student': checkpoint}, parent_file,
                                 {}, [source], {'regime': 'cold'})
    record = json.loads(frozen.read_text())
    assert record['baseline_digest'] == baseline.digest()
    assert 'baseline' in record['files'] and 'rules' not in record['files']
    with data.claim_final_evaluation(output, frozen, {'regime': 'cold'}) as claim:
        splits, final = data.materialize_final_data(output, claim, baseline)
        assert final['baseline_digest'] == baseline.digest()
        seen = {e.group_digest for e in examples + validation}
        for group in splits.values():
            for example in group:
                assert example.classical_selection == baseline.selection_record()
                assert example.group_digest not in seen
                seen.add(example.group_digest)
        restored, _ = data.materialize_final_data(output, claim, baseline)
        assert {k: [e.digest for e in v] for k, v in restored.items()} == {
            k: [e.digest for e in v] for k, v in splits.items()}


def test_fixed_parent_real_training_provenance_and_resume(tmp_path):
    torch.set_num_threads(1)
    cfg, baseline, _, examples, _, _ = development(tmp_path)
    initial = Components.create(hidden=4, seed=1927)
    initial.transfer = CompactTransferStudent(hidden=4)
    settings = dict(seed=473, updates=2, prefix_cycles=1, tail_cycles=0,
                    lambda_compute=0., learning_rate=1e-4)
    target = tmp_path / 'train'
    model, status = train_expert(initial, examples, cfg, baseline, settings,
                                target, branch='H_P')
    assert status['updates'] == 2
    assert status['baseline_digest'] == baseline.digest()
    assert 'rules_digest' not in status
    assert model.metadata['training_baseline_digest'] == baseline.digest()
    assert 'training_rules_digest' not in model.metadata
    assert model.generation_signature() != initial.generation_signature()
    manifest = json.loads((target / 'training_manifest.json').read_text())
    assert manifest['baseline_digest'] == baseline.digest()
    assert 'rules_digest' not in manifest
    recovered, status2 = train_expert(initial, examples, cfg, baseline, settings,
                                     target, branch='H_P', resume=True)
    assert status2 == status
    assert recovered.generation_signature() == model.generation_signature()


def test_fixed_modules_do_not_import_selector_even_when_import_is_forbidden():
    project = Path(__file__).resolve().parents[1]
    script = '''
import importlib.abc
import sys
class BlockStrong(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('adaptive_mg.v67.strong'):
            raise AssertionError('fixed path imported selector: ' + fullname)
sys.meta_path.insert(0, BlockStrong())
import adaptive_mg.v67.fixed_classical
import adaptive_mg.v67.research_data
import adaptive_mg.v67.research_training
'''
    script = 'import sys; sys.path.insert(0, ' + repr(str(project / 'src')) + ')\n' + script
    result = subprocess.run([sys.executable, '-c', script], cwd=project,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
