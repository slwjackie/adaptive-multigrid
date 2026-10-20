"""Real staged EM/line training, deterministic resume and measured selection."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from adaptive_mg.v67 import em_line_study as study
from adaptive_mg.v67.em_line_models import AffineTransferCNN
from adaptive_mg.v67.fixed_classical import BILINEAR_STRATEGY_NAME, FIXED_STRATEGY_NAME
from adaptive_mg.v67.line_smoother import LineSmootherCNN
from adaptive_mg.v67.limited import digest_file
from adaptive_mg.v67.models import Components

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def settings():
    return read(ROOT/'configs/v6_7_em_line_smoke.json')


def prepare(tmp, spec=None):
    config = tmp/'settings.json'
    config.write_text(json.dumps(settings() if spec is None else spec))
    out = tmp/'run'
    study.main(['prepare', '--config', str(config), '--run-dir', str(out)])
    return out


@pytest.fixture(scope='module')
def trained_pipeline(tmp_path_factory):
    out = prepare(tmp_path_factory.mktemp('em_line_pipeline'))
    study.main(['train', '--run-dir', str(out)])
    before = {name: digest_file(out/'checkpoints'/name/'candidate.pt') for name in study.STAGES}
    study.main(['train', '--run-dir', str(out), '--resume'])
    assert before == {name: digest_file(out/'checkpoints'/name/'candidate.pt') for name in study.STAGES}
    return out


def test_real_sequential_two_update_stages_and_serialized_metadata(trained_pipeline):
    out, spec, cfg, baseline = study.load(trained_pipeline)
    assert spec['smoke'] and baseline.strategy_name == BILINEAR_STRATEGY_NAME
    assert cfg.smoother_levels == cfg.transfer_levels == 'all'
    arms = study._arms(out, cfg, baseline)
    assert tuple(arms) == ('fixed_C', 'em5_C', *study.STAGES)
    assert arms['em5_C']['reference_only']
    assert arms['em5_C']['config'].mg.strategy_name == FIXED_STRATEGY_NAME
    assert arms['em5_C']['config'].mg.em_iterations == 5
    assert arms['em5_C']['config'].mg.em_fixed_iterations
    previous = Components.load(out/'initial.pt')
    for name, (branch, trainable) in study.STAGES.items():
        model = arms[name]['model']
        assert isinstance(model.transfer, AffineTransferCNN)
        assert isinstance(model.smoother, LineSmootherCNN)
        folder = out/'checkpoints'/name
        status, history = read(folder/'status.json'), read(folder/'training.json')
        assert status['status'] == 'complete' and status['all_updates_completed']
        assert status['updates'] == 2 and status['skipped'] == 0
        assert model.metadata['em_line_stage'] == name
        assert model.metadata['previous_stage'] == previous.metadata.get('em_line_stage')
        assert model.metadata['previous_signature'] == previous.signature()
        assert model.metadata['training_branch'] == branch
        assert model.metadata['trainable_components'] == list(trainable)
        assert model.metadata['transfer_trained']
        assert model.metadata['smoother_trained'] == (name != 'em_P')
        assert not model.metadata['online_em'] and not model.metadata['final_test_seen']
        assert model.metadata['offline_em_iteration_budget'] == 5
        assert all(row['metrics']['fallback_used'] is False for row in history)
        assert [row['cycles'] for row in history] == ([0, 0] if name == 'em_P' else [3, 6])
        for key in study.MODULES:
            if key in trainable:
                assert model.expert_signature(key) != previous.expert_signature(key)
            else:
                assert model.expert_signature(key) == previous.expert_signature(key)
        if name == 'em_P':
            assert all(row['metrics']['teacher_targets_detached'] for row in history)
            assert all(row['metrics']['em_iteration_budget'] == 5 for row in history)
        else:
            assert all(set(row['metrics']['probe_labels']) == {
                'random', 'classical_smoothed', 'learned_hard'} for row in history)
            assert all(row['metrics']['pre_post_coefficients_shared'] for row in history)
        previous = model
    assert not (out/'data/final_claim.json').exists()


def test_real_five_arm_setup_selection_warm_secondary_and_resume(trained_pipeline):
    out = trained_pipeline
    args = ['benchmark', '--run-dir', str(out), '--repeats', '1', '--warmups', '0',
            '--rhs-counts', '1', '--regimes', 'warm', 'multiple']
    result = study.main(args)
    target = out/'benchmarks/validation'
    before = digest_file(target/'raw_results.json')
    replay = study.main(args+['--resume'])
    assert digest_file(target/'raw_results.json') == before
    assert replay['summary'] == result['summary']
    assert read(target/'progress.json') == {
        'status': 'complete', 'completed_measurements': 70, 'total_measurements': 70}
    for row in result['rows']:
        for name in ('fixed_C', 'em5_C', *study.STAGES):
            counted = row['runs'][name]['multiple']['1'][0]
            warm = row['runs'][name]['warm']['1'][0]
            assert counted['setup_scope'] == 'included_once'
            assert counted['wall_seconds'] > 0 and counted['total_setup_seconds'] > 0
            assert warm['setup_scope'] == 'excluded_prepared_operator'
            expected = FIXED_STRATEGY_NAME if name == 'em5_C' else BILINEAR_STRATEGY_NAME
            assert counted['measurement_config']['mg']['strategy_name'] == expected
    raw = read(target/'raw_contractions.json')
    assert raw['fallback_allowed'] is False
    assert all(set(row['arms']) == set(result['manifest']['arms']) for row in raw['rows'])
    selection = read(target/'selection.json')
    assert selection['regime'] == 'multiple' and selection['rhs_count'] == 1
    assert selection['total_cases'] == 7 and not selection['final_certificate']
    assert set(selection['warm_secondary']) == {'warm'}
    # Altering only the cached-solve timings cannot change the setup-inclusive choice.
    distorted = deepcopy(result)
    for row in distorted['rows']:
        for name in row['runs']:
            for run in row['runs'][name]['warm']['1']:
                run['wall_seconds'] = 1e-20 if name == 'joint_line' else 1e20
    checked = study.select_measured_candidate(distorted, rhs_count=1,
                                             candidate_arms=tuple(study.STAGES))
    assert checked['recommended_deployment_arm'] == selection['recommended_deployment_arm']
    export = read(out/'selected/deployment.json')
    assert export['arm'] == selection['recommended_deployment_arm']
    if export['arm'] in study.STAGES:
        assert digest_file(export['checkpoint']) == export['sha256']
        assert export['sha256'] == digest_file(out/'checkpoints'/export['arm']/'candidate.pt')
        assert export['branch'] == study.STAGES[export['arm']][0]
    else:
        assert export['checkpoint'] is None
    assert read(out/'benchmark_evidence.json')['performance_certified'] is False
    with pytest.raises(ValueError, match='Validation'):
        study.main(['train', '--run-dir', str(out), '--resume'])
    with pytest.raises(ValueError, match='Smoke'):
        study.freeze(out, repeats=1, warmups=0, rhs_counts=[1], regimes=['multiple'])
    assert not (out/'data/research_freeze.json').exists()
    assert not (out/'data/final_claim.json').exists()


def test_interrupted_frozen_p_training_resumes_identical_optimizer(tmp_path, monkeypatch):
    out = prepare(tmp_path)
    actual_objective = study.line_vcycle_objective
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated interruption after one smoother update')
        return actual_objective(*args, **kwargs)
    monkeypatch.setattr(study, 'line_vcycle_objective', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        study.main(['train', '--run-dir', str(out)])
    assert read(out/'checkpoints/em_P/status.json')['status'] == 'complete'
    status = read(out/'checkpoints/frozen_P_line/status.json')
    assert status['status'] == 'failed' and status['updates'] == 1
    assert not (out/'checkpoints/joint_line').exists()
    em_digest = digest_file(out/'checkpoints/em_P/candidate.pt')
    monkeypatch.setattr(study, 'line_vcycle_objective', actual_objective)
    study.main(['train', '--run-dir', str(out), '--resume'])
    assert em_digest == digest_file(out/'checkpoints/em_P/candidate.pt')
    out, spec, cfg, baseline = study.load(out)
    examples, _, _ = study._development(out, spec, baseline)
    initial = study._checkpoint(out, 'em_P', baseline)
    reference = tmp_path/'reference'
    reference.mkdir()
    for name in ('frozen_P_line', 'joint_line'):
        expected = study._run_training(reference, name, initial, examples, cfg,
                                       baseline, spec, resume=False)
        actual = study._checkpoint(out, name, baseline)
        assert actual.component_signatures() == expected.component_signatures()
        assert actual.metadata == expected.metadata
        assert read(out/'checkpoints'/name/'training.json') == read(reference/'checkpoints'/name/'training.json')
        initial = expected
    digests = {name: digest_file(out/'checkpoints'/name/'candidate.pt') for name in study.STAGES}
    study.main(['train', '--run-dir', str(out), '--resume'])
    assert digests == {name: digest_file(out/'checkpoints'/name/'candidate.pt') for name in study.STAGES}


def test_validation_start_blocks_training_before_any_measurement_completes(tmp_path):
    out = prepare(tmp_path)
    marker = out/'benchmarks/validation/run_manifest.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')
    with pytest.raises(ValueError, match='Validation'):
        study.main(['train', '--run-dir', str(out)])
    assert not (out/'checkpoints').exists()


def test_initial_and_stage_training_manifest_mutations_rejected(trained_pipeline):
    out, _, cfg, baseline = study.load(trained_pipeline)
    manifest = out/'checkpoints/frozen_P_line/training_manifest.json'
    original = manifest.read_bytes()
    changed = read(manifest)
    changed['settings']['learning_rate'] *= 2
    try:
        manifest.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            study._arms(out, cfg, baseline)
    finally:
        manifest.write_bytes(original)
    initial = out/'initial.pt'
    original = initial.read_bytes()
    try:
        initial.write_bytes(original+b'changed')
        with pytest.raises(ValueError):
            study._arms(out, cfg, baseline)
    finally:
        initial.write_bytes(original)


@pytest.mark.parametrize('change', [
    'em_budget', 'em_adaptive', 'partial_levels', 'classical_parent', 'pre_post',
    'controller', 'decreasing_curriculum', 'zero_probes', 'negative_hard_cycles',
    'nan_stability', 'nan_learning_rate', 'missing_stage', 'terminal_grid',
])
def test_settings_reject_changes_to_the_declared_experiment(change):
    spec = settings()
    if change == 'em_budget':
        spec['em_teacher']['iterations'] = 4
    elif change == 'em_adaptive':
        spec['em_teacher']['fixed_iterations'] = False
    elif change == 'partial_levels':
        spec['solver']['smoother_levels'] = [0]
    elif change == 'classical_parent':
        spec['solver']['mg']['strategy_name'] = FIXED_STRATEGY_NAME
    elif change == 'pre_post':
        spec['solver']['replace_post'] = 1
    elif change == 'controller':
        spec['solver']['use_learned_controller'] = True
    elif change == 'decreasing_curriculum':
        spec['stages']['joint_line']['end_cycles'] = 1
    elif change == 'zero_probes':
        spec['stages']['joint_line']['probes_per_kind'] = 0
    elif change == 'negative_hard_cycles':
        spec['stages']['joint_line']['hard_cycles'] = -1
    elif change == 'nan_stability':
        spec['stages']['joint_line']['lambda_stability'] = float('nan')
    elif change == 'nan_learning_rate':
        spec['stages']['em_P']['learning_rate'] = float('nan')
    elif change == 'missing_stage':
        del spec['stages']['frozen_P_line']
    elif change == 'terminal_grid':
        spec['training_sizes'] = [3]
    with pytest.raises(ValueError):
        study.validate_settings(spec)


def test_warm_only_selection_and_missing_primary_rhs_are_rejected(tmp_path):
    out = prepare(tmp_path)
    for regimes, counts in ((['warm'], [1]), (['multiple'], [4])):
        with pytest.raises(ValueError, match='setup-inclusive selection'):
            study.benchmark(out, repeats=1, warmups=0, regimes=regimes, rhs_counts=counts)
    assert not (out/'benchmarks').exists()
