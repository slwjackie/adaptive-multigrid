"""Real small-grid joint pipeline, interruption recovery and CLI boundaries."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from adaptive_mg.v67 import joint_transfer_study as study
from adaptive_mg.v67 import research_data
from adaptive_mg.v67.fixed_classical import BILINEAR_STRATEGY_NAME, FIXED_STRATEGY_NAME
from adaptive_mg.v67.limited import digest_file

ROOT = Path(__file__).resolve().parents[1]


def settings():
    spec = json.loads((ROOT / 'configs/v6_7_joint_transfer_smoke.json').read_text())
    spec.update(training_sizes=[7], validation_sizes=[7], training_per_family=1,
                validation_per_family=1, target_rhs_count=1)
    spec['training']['updates'] = 2
    spec['search'].update(starts=2, updates=2, repeats=1)
    spec['distillation'].update(updates=2, finetune_updates=2)
    return spec


def prepare(tmp, spec=None):
    path = tmp / 'settings.json'
    path.write_text(json.dumps(settings() if spec is None else spec))
    out = tmp / 'run'
    study.main(['prepare', '--config', str(path), '--run-dir', str(out)])
    return out


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture(scope='module')
def completed_joint_pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('joint_pipeline')
    out = prepare(tmp)
    study.main(['train', '--run-dir', str(out)])
    digests = {name: digest_file(out / 'checkpoints' / name / 'candidate.pt')
               for name in ('learned_P', 'learned_S', 'joint')}
    study.main(['train', '--run-dir', str(out), '--resume'])
    assert digests == {name: digest_file(out / 'checkpoints' / name / 'candidate.pt')
                       for name in digests}
    study.main(['search', '--run-dir', str(out)])
    study.main(['distill', '--run-dir', str(out)])
    checkpoint = out / 'checkpoints/distilled_joint/candidate.pt'
    before = digest_file(checkpoint)
    study.main(['search', '--run-dir', str(out), '--resume'])
    study.main(['distill', '--run-dir', str(out), '--resume'])
    assert digest_file(checkpoint) == before
    return out


def test_real_two_update_training_search_and_two_plus_two_distillation(completed_joint_pipeline):
    out, spec, cfg, baseline = study.load(completed_joint_pipeline)
    assert baseline.strategy_name == BILINEAR_STRATEGY_NAME
    assert cfg.transfer_levels == cfg.smoother_levels == 'all'
    initial = study._initial(out)
    arms = study._arms(out, spec, cfg, baseline)
    assert set(arms) == {'fixed_C', 'em5_C', 'learned_P', 'learned_S', 'joint', 'distilled_joint'}
    assert arms['em5_C']['reference_only']
    assert arms['em5_C']['config'].mg.strategy_name == FIXED_STRATEGY_NAME
    assert arms['em5_C']['config'].mg.em_fixed_iterations
    assert arms['em5_C']['config'].mg.em_iterations == 5
    assert arms['learned_P']['model'].expert_signature('smoother') == initial.expert_signature('smoother')
    assert arms['learned_S']['model'].expert_signature('transfer') == initial.expert_signature('transfer')
    for name, (_, trainable) in study.STAGES.items():
        folder = out / 'checkpoints' / name
        status, history = read(folder / 'status.json'), read(folder / 'training.json')
        assert status['status'] == 'complete' and status['all_updates_completed']
        assert status['skipped'] == 0 and status['updates'] == (4 if name == 'distilled_joint' else 2)
        assert arms[name]['model'].metadata['trainable_components'] == list(trainable)
        assert all(row['metrics']['fallback_used'] is False for row in history)
        assert all(set(row['metrics']['probe_labels']) == {
            'random', 'classical_smoothed', 'learned_hard'} for row in history)
        if name == 'distilled_joint':
            assert [row['phase'] for row in history] == ['distill', 'distill', 'task', 'task']
            assert [row['cycles'] for row in history] == [3, 6, 6, 6]
            assert arms[name]['model'].metadata['offline_oracle_distillation']
        else:
            assert [row['cycles'] for row in history] == [3, 6]
    search = read(out / 'search/report.json')
    assert search['status'] == 'complete' and search['training_only']
    assert search['fallback_allowed'] is False and search['coverage'] > 0
    assert search['total'] == 7
    for row in search['rows']:
        assert len(row['starts']) == 2
        for start in row['starts']:
            if start['status'] == 'complete':
                assert start['updates'] == 2
                assert {item['label'] for item in start['candidates']} == {'initial', 'optimized'}
                assert all(item['timing']['fallback_used'] is False for item in start['candidates'])
    assert not (out / 'data/research_freeze.json').exists()
    assert not (out / 'data/final_claim.json').exists()


def test_six_arm_setup_inclusive_benchmark_resume_and_post_validation_lock(completed_joint_pipeline):
    out = completed_joint_pipeline
    argv = ['benchmark', '--run-dir', str(out), '--repeats', '1', '--warmups', '0',
            '--rhs-counts', '1', '--regimes', 'multiple']
    result = study.main(argv)
    target = out / 'benchmarks/validation'
    raw_digest = digest_file(target / 'raw_results.json')
    replay = study.main(argv + ['--resume'])
    assert digest_file(target / 'raw_results.json') == raw_digest
    assert replay['summary'] == result['summary']
    assert read(target / 'progress.json') == {
        'status': 'complete', 'completed_measurements': 42, 'total_measurements': 42}
    for row in result['rows']:
        for name in ('fixed_C', 'em5_C', *study.STAGES):
            run = row['runs'][name]['multiple']['1'][0]
            assert run['setup_scope'] == 'included_once'
            assert run['wall_seconds'] > 0 and run['total_setup_seconds'] > 0
            expected = FIXED_STRATEGY_NAME if name == 'em5_C' else BILINEAR_STRATEGY_NAME
            assert run['measurement_config']['mg']['strategy_name'] == expected
    raw = read(target / 'raw_contractions.json')
    assert raw['fallback_allowed'] is False
    assert all(set(row['arms']) == set(result['manifest']['arms']) for row in raw['rows'])
    selection = read(target / 'selection.json')
    assert selection['regime'] == 'multiple' and selection['rhs_count'] == 1
    assert selection['total_cases'] == 7 and not selection['final_certificate']
    assert read(out / 'benchmark_evidence.json')['performance_certified'] is False
    for command in ('train', 'search', 'distill'):
        with pytest.raises(ValueError, match='Validation committed'):
            study.main([command, '--run-dir', str(out), '--resume'])
    with pytest.raises(ValueError, match='Smoke'):
        study.freeze(out, repeats=1, warmups=0, rhs_counts=[1], regimes=['multiple'])
    assert not (out / 'data/final_claim.json').exists()


def test_partial_training_resumes_exactly_from_saved_optimizer(tmp_path, monkeypatch):
    out = prepare(tmp_path)
    original = study.joint_objective
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated interruption after one committed update')
        return original(*args, **kwargs)

    monkeypatch.setattr(study, 'joint_objective', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        study.main(['train', '--run-dir', str(out)])
    status = read(out / 'checkpoints/learned_P/status.json')
    assert status['status'] == 'failed' and status['updates'] == 1
    monkeypatch.setattr(study, 'joint_objective', original)
    study.main(['train', '--run-dir', str(out), '--resume'])
    out, spec, cfg, baseline = study.load(out)
    actual = study._checkpoint(out, 'learned_P', baseline)
    examples, _, _ = study._development(out, spec, baseline)
    reference_out = tmp_path / 'uninterrupted_reference'
    reference_out.mkdir()
    reference = study._run_training(reference_out, 'learned_P', study._initial(out),
        examples, cfg, baseline, spec, resume=False)
    assert actual.component_signatures() == reference.component_signatures()
    actual_history = read(out / 'checkpoints/learned_P/training.json')
    reference_history = read(reference_out / 'checkpoints/learned_P/training.json')
    assert actual_history == reference_history


def test_source_change_initial_checkpoint_and_frozen_development_are_rejected(tmp_path, monkeypatch):
    out = prepare(tmp_path)
    source = ROOT / 'scripts/run_v6_7_joint_transfer_study.py'
    real_digest = study.digest_file

    def changed_digest(path):
        return '0' * 64 if Path(path).resolve() == source.resolve() else real_digest(path)

    with monkeypatch.context() as patch:
        patch.setattr(study, 'digest_file', changed_digest)
        with pytest.raises(ValueError, match='Sources/settings/environment changed'):
            study.load(out)
    initial = out / 'initial.pt'
    original = initial.read_bytes()
    initial.write_bytes(original + b'changed')
    with pytest.raises(ValueError, match='Initial checkpoint changed'):
        study._initial(out)
    initial.write_bytes(original)
    (out / 'data/research_freeze.json').write_text('{}')
    with pytest.raises(ValueError, match='frozen'):
        study.main(['train', '--run-dir', str(out), '--resume'])
    with pytest.raises(ValueError, match='frozen'):
        study.main(['benchmark', '--run-dir', str(out), '--repeats', '1', '--warmups', '0',
                    '--rhs-counts', '1', '--regimes', 'multiple'])


@pytest.mark.parametrize('mutation', ['classical_parent', 'partial_level', 'pre_post', 'controller'])
def test_configuration_rejects_changes_to_the_declared_experiment(mutation):
    spec = deepcopy(settings())
    if mutation == 'classical_parent':
        spec['solver']['mg']['strategy_name'] = FIXED_STRATEGY_NAME
    elif mutation == 'partial_level':
        spec['solver']['smoother_levels'] = [0]
    elif mutation == 'pre_post':
        spec['solver']['replace_post'] = 1
    else:
        spec['solver']['use_learned_controller'] = True
    with pytest.raises(ValueError):
        study.validate_settings(spec)


def test_benchmark_cli_keeps_explicit_target_rhs_and_resume(monkeypatch, tmp_path):
    captured = {}

    def benchmark(output, **kwargs):
        captured.update(output=output, **kwargs)

    monkeypatch.setattr(study, 'benchmark', benchmark)
    study.main(['benchmark', '--run-dir', str(tmp_path), '--resume', '--repeats', '2',
                '--warmups', '0', '--rhs-counts', '1', '4', '--regimes', 'cold', 'multiple'])
    assert captured == dict(output=str(tmp_path), resume=True, repeats=2, warmups=0,
                            rhs_counts=[1, 4], regimes=['cold', 'multiple'])


def test_small_non_smoke_freeze_final_contract_and_no_final_selection(tmp_path, monkeypatch):
    # Reduce the plan BEFORE it is committed. This tests the actual non-smoke
    # freeze/claim lifecycle without repurposing a smoke run or seeing final A.
    original_plan = study.make_research_plan

    def compact_plan(*, smoke, seed, discretization="p1"):
        assert smoke is False
        plan = original_plan(smoke=False, seed=seed, discretization=discretization)
        for split in plan['splits'].values():
            # Keep a family valid for each declared OOD mechanism (topology
            # requires heterogeneous coefficients).
            split.update(sizes=[7], families=split['families'][:1], per_family=1, count=1)
        return plan

    monkeypatch.setattr(study, 'make_research_plan', compact_plan)
    calls = []
    generate = research_data._generate_split

    def tracked(name, *args, **kwargs):
        calls.append(name)
        return generate(name, *args, **kwargs)

    monkeypatch.setattr(research_data, '_generate_split', tracked)
    spec = settings()
    spec['smoke'] = False
    spec['seed'] += 10009
    out = prepare(tmp_path, spec)
    assert calls == ['train', 'validation']
    assert not (out / 'data/final_data_manifest.json').exists()
    for command in ('train', 'search', 'distill'):
        study.main([command, '--run-dir', str(out)])
    study.main(['benchmark', '--run-dir', str(out), '--repeats', '1', '--warmups', '0',
                '--rhs-counts', '1', '--regimes', 'multiple'])
    selection_path = out / 'benchmarks/validation/selection.json'
    selection_digest = digest_file(selection_path)
    study.main(['freeze', '--run-dir', str(out), '--repeats', '1', '--warmups', '0',
                '--rhs-counts', '1', '--regimes', 'multiple', '--splits', 'final', 'ood_grid'])
    assert calls == ['train', 'validation']
    assert not (out / 'data/final_data_manifest.json').exists()
    frozen_path = out / 'data/research_freeze.json'
    frozen = read(frozen_path)
    assert frozen['no_final_operator_observation']
    assert frozen['config']['arms'] == ['fixed_C', 'em5_C', *study.STAGES]
    assert set(name for name in frozen['files'] if name.startswith('checkpoint:')) == {
        f'checkpoint:{name}' for name in study.STAGES}
    checkpoint = out / 'checkpoints/joint/candidate.pt'
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b'changed')
    with pytest.raises(ValueError, match='Frozen input changed'):
        research_data._verify_freeze(frozen_path, out / 'data')
    checkpoint.write_bytes(original)
    original_hash = research_data._file_hash
    source_path = next(Path(item['path']) for key, item in frozen['files'].items()
                       if key.startswith('source:'))

    def changed_source(path):
        return '0' * 64 if Path(path).resolve() == source_path.resolve() else original_hash(path)

    with monkeypatch.context() as patch:
        patch.setattr(research_data, '_file_hash', changed_source)
        with pytest.raises(ValueError, match='Frozen input changed'):
            study.main(['final', '--run-dir', str(out)])
    assert not (out / 'data/final_claim.json').exists()

    def forbidden_selection(*args, **kwargs):
        pytest.fail('Final evaluation must not select learned models')

    monkeypatch.setattr(study, 'select_measured_candidate', forbidden_selection)
    study.main(['final', '--run-dir', str(out)])
    assert read(out / 'data/final_claim.json')['status'] == 'completed'
    assert digest_file(selection_path) == selection_digest
    assert set(calls[2:]) == set(research_data.HOLDOUT)
    for split in ('final', 'ood_grid'):
        manifest = read(out / 'final' / split / 'run_manifest.json')
        assert manifest['arms']['em5_C']['reference_only']
        assert manifest['arms']['em5_C']['config']['mg']['strategy_name'] == FIXED_STRATEGY_NAME
        assert not (out / 'final' / split / 'selection.json').exists()
        assert read(out / 'final' / split / 'progress.json')['status'] == 'complete'
    with pytest.raises(ValueError, match='already completed'):
        study.main(['final', '--run-dir', str(out), '--resume'])
