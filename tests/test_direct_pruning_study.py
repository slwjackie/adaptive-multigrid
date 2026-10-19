"""Pruning import isolation, data lineage, resumable measurements and final gates."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from adaptive_mg.v67 import direct_pruning_study as study
from adaptive_mg.v67 import unet_transfer_study as source_study
from adaptive_mg.v67.limited import digest_file
from adaptive_mg.v67.research_data import _digests, make_research_plan


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def tree_hashes(path):
    return {str(p.relative_to(path)): digest_file(p) for p in path.rglob('*') if p.is_file()}


@pytest.fixture(scope='module')
def trained_source(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('pruning_source')
    settings = json.loads((ROOT / 'configs/v6_7_unet_transfer_smoke.json').read_text())
    settings.update(training_sizes=[7], validation_sizes=[7], training_per_family=1, validation_per_family=1)
    settings['training'].update(updates=2, kd_updates=1)
    settings['unet_transfer']['student_hidden'] = 4
    config = write(tmp / 'source_config.json', settings)
    source = tmp / 'source'
    source_study.prepare(config, source)
    source_study.train(source, 'direct')
    teacher = source / 'checkpoints/teacher'
    teacher.mkdir()
    (teacher / 'untouched.txt').write_text('The pruning study must never read or train this teacher.')
    return source


def configuration(tmp, *, smoke=True):
    settings = json.loads((ROOT / 'configs/v6_7_direct_pruning_smoke.json').read_text())
    settings['smoke'] = smoke
    return write(tmp / 'pruning_config.json', settings)


def test_source_import_is_immutable_and_new_data_are_disjoint(tmp_path, trained_source):
    before = tree_hashes(trained_source)
    out = tmp_path / 'pruning'
    config = configuration(tmp_path)
    report = study.prepare(config, trained_source, out)
    assert tree_hashes(trained_source) == before
    assert report['original_hidden'] == 4 and report['hidden'] == 2
    assert report['pruned_parameter_count'] < report['original_parameter_count']
    old = json.loads((trained_source / 'data/development_manifest.json').read_text())
    new = json.loads((out / 'data/development_manifest.json').read_text())
    assert not _digests(old['splits']) & _digests(new['splits'])
    assert set(new['historical_index']['normalized_operator_digests']) >= _digests(old['splits'])
    assert not (out / 'data/final_data_manifest.json').exists()
    assert not (out / 'checkpoints/teacher').exists()
    study.prepare(config, trained_source, out, resume=True)
    with pytest.raises(FileExistsError):
        study.prepare(config, trained_source, out)
    with pytest.raises(ValueError, match='separate run'):
        study.prepare(config, trained_source, trained_source / 'pruning')


def test_source_hash_and_imported_snapshot_corruption_are_rejected(tmp_path, trained_source):
    source = tmp_path / 'source_copy'
    shutil.copytree(trained_source, source)
    config = configuration(tmp_path)
    out = tmp_path / 'pruning'
    study.prepare(config, source, out)
    imported = out / 'imported/source/checkpoints/direct/candidate.pt'
    imported.write_bytes(imported.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Imported input changed'):
        study.load(out)
    checkpoint = source / 'checkpoints/direct/candidate.pt'
    checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        study.prepare(config, source, tmp_path / 'another_run')


def test_source_frozen_weight_binding_is_verified_without_old_code_requirement(tmp_path, trained_source):
    source = tmp_path / 'source_copy'
    shutil.copytree(trained_source, source)
    baseline = study.load_baseline(source / 'fixed_classical.json')
    frozen = dict(config={'old_source_version': 1}, files={})
    frozen['config_digest'] = study._hash(frozen['config'])
    for key, rel in [('checkpoint:direct_CNN', 'checkpoints/direct/candidate.pt'),
                     ('baseline', 'fixed_classical.json'), ('plan', 'data/dataset_plan.json'),
                     ('development', 'data/development_manifest.json')]:
        frozen['files'][key] = dict(path='/old/machine/' + rel, sha256=digest_file(source / rel))
    frozen['files']['source:0'] = dict(path='/old/machine/source.py', sha256='0' * 64)
    write(source / 'data/research_freeze.json', frozen)
    _, files = study._source_snapshot(source, baseline)
    assert 'data/research_freeze.json' in files
    frozen['files']['checkpoint:direct_CNN']['sha256'] = '0' * 64
    write(source / 'data/research_freeze.json', frozen)
    with pytest.raises(ValueError, match='Source frozen import changed'):
        study._source_snapshot(source, baseline)


def test_pruning_train_benchmark_and_resume_measure_ttt_separately(tmp_path, trained_source):
    out = tmp_path / 'pruning'
    study.prepare(configuration(tmp_path), trained_source, out)
    original = digest_file(out / 'imported/pruned_initial.pt')
    status = study.train(out)
    assert status['updates'] == 2 and not status['teacher_used_for_training']
    assert status['all_updates_completed'] and status['feasible_task_updates'] > 0
    assert digest_file(out / 'imported/pruned_initial.pt') == original
    assert study.train(out, resume=True)['checkpoint_sha256'] == status['checkpoint_sha256']
    _, settings, _, baseline = study.load(out)
    arms = study._arms(out, settings, baseline)
    assert tuple(arms) == study.ARMS
    assert arms['pruned_CNN']['model'].signature() == arms['pruned_CNN_TTT']['model'].signature()
    assert 'ttt' not in arms['pruned_CNN']
    protocol = dict(repeats=1, warmups=0, regimes=['cold', 'warm', 'multiple'], rhs_counts=[1, 2])
    decisions = study.benchmark(out, **protocol)
    assert decisions and all(v['setup_included'] for v in decisions.values())
    target = out / 'benchmarks/validation'
    paired = json.loads((target / 'paired_pruning_ttt.json').read_text())
    assert 'cold/rhs1/pruned_CNN_TTT_vs_pruned_CNN' in paired
    assert not paired['warm/rhs1/pruned_CNN_TTT_vs_pruned_CNN']['setup_included']
    raw_hash = digest_file(target / 'raw_results.json')
    assert study.benchmark(out, resume=True, **protocol) == decisions
    assert digest_file(target / 'raw_results.json') == raw_hash
    with pytest.raises(ValueError, match='setup-inclusive'):
        study.freeze(out, repeats=1, warmups=0, regimes=['warm'], rhs_counts=[1])
    with pytest.raises(ValueError, match='Smoke'):
        study.freeze(out, **protocol)
    for changes in ({'repeats': 0}, {'warmups': -1}, {'regimes': ['cold', 'cold']},
                    {'rhs_counts': [1, 0]}, {'splits': ['final', 'final']}):
        with pytest.raises(ValueError):
            study.freeze(out, **{**protocol, **changes})
        assert not (out / 'data/research_freeze.json').exists()


def test_fresh_final_is_frozen_and_single_use(tmp_path, trained_source, monkeypatch):
    def tiny_plan(**kwargs):
        plan = make_research_plan(**kwargs)
        for name, spec in plan['splits'].items():
            if name not in ('train', 'validation'):
                spec.update(sizes=[7], families=['near_isotropic'] if name != 'ood_topology' else ['local_patch'],
                            per_family=1, count=1)
        return plan

    monkeypatch.setattr(study, 'make_research_plan', tiny_plan)
    out = tmp_path / 'pruning'
    study.prepare(configuration(tmp_path, smoke=False), trained_source, out)
    study.train(out)
    protocol = dict(repeats=1, warmups=0, regimes=['cold'], rhs_counts=[1])
    study.benchmark(out, **protocol)
    study.freeze(out, splits=['final'], **protocol)
    assert not (out / 'data/final_data_manifest.json').exists()
    with pytest.raises(ValueError, match='frozen'):
        study.train(out, resume=True)
    study.final(out)
    ledger = json.loads((out / 'data/final_claim.json').read_text())
    assert ledger['status'] == 'completed'
    assert (out / 'final/final/paired_pruning_ttt.json').exists()
    assert not (out / 'final/ood_grid').exists()
    with pytest.raises(ValueError, match='completed'):
        study.final(out, resume=True)


def test_settings_reject_distillation_and_uncounted_final(tmp_path):
    settings = json.loads(configuration(tmp_path).read_text())
    study.validate_settings(settings)
    bad = deepcopy(settings)
    bad['training']['lambda_kd'] = .25
    with pytest.raises(ValueError, match='does not use teacher'):
        study.validate_settings(bad)
    bad = deepcopy(settings)
    bad['solver']['use_smoother'] = True
    with pytest.raises(ValueError, match='forced H_P'):
        study.validate_settings(bad)
    bad = deepcopy(settings)
    bad['ttt']['max_updates'] = 0
    with pytest.raises(ValueError, match='at least one online update'):
        study.validate_settings(bad)
