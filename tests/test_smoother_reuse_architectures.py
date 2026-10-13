"""World architecture/causality contracts; not evidence of CFD acceleration."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from adaptive_mg.v67.smoother_reuse import ACTIONS
from adaptive_mg.v67.smoother_reuse import study, world
from adaptive_mg.v67.smoother_reuse.policy import FEATURES
from test_smoother_reuse import model_episode, reuse_settings


@pytest.mark.parametrize('hidden', [16, 32, 64])
@pytest.mark.parametrize('memory,head', [
    ('shared', 'linear'), ('shared', 'role_mlp'), ('independent_roles', 'role_mlp')])
def test_requested_architectures_preserve_action_shapes_and_train_all_roles(hidden, memory, head):
    net = world.Dynamics(hidden, memory=memory, head=head, head_hidden=8)
    state = net.observe(torch.randn(2, len(FEATURES)), torch.zeros(2, 2))
    costs, success, next_obs = net.predict(state)
    assert costs.shape == success.shape == (2, 2)
    assert next_obs.shape == (2, 2, len(FEATURES))
    # Every branch participates in the same supervised objectives.
    (costs.square().mean()+success.square().mean()+next_obs.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert state.shape == ((2, hidden) if memory == 'shared' else (2, 3, hidden))


def test_independent_role_recurrences_have_no_hidden_parameter_sharing():
    net = world.Dynamics(4, memory='independent_roles', head='role_mlp', head_hidden=3)
    state = net.observe(torch.randn(1, len(FEATURES)), torch.zeros(1, 2))
    net.predict(state)[0].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in net.observe_nets['cost'].parameters())
    for role in ('success', 'next_observation'):
        assert all(p.grad is None or not p.grad.any() for p in net.observe_nets[role].parameters())


@pytest.mark.parametrize('recurrent', [True, False])
@pytest.mark.parametrize('memory,head', [
    ('shared', 'linear'), ('shared', 'role_mlp'),
    ('independent_roles', 'linear'), ('independent_roles', 'role_mlp')])
def test_fit_roundtrip_rollout_and_reset_for_every_architecture(tmp_path, recurrent, memory, head):
    episode = model_episode()
    artifact = world.fit([episode], hidden=4, epochs=1, recurrent=recurrent,
                         memory=memory, head=head, head_hidden=3)
    path = tmp_path/'model.pt'
    torch.save(artifact, path)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    predictor = world.Predictor(loaded)
    assert artifact['parameter_count'] == sum(p.numel() for p in predictor.net.parameters())
    for index, raw in enumerate(episode['obs']):
        action, detail = predictor.choose(raw, 'refresh', horizon=2)
        assert action in ACTIONS
        assert len(detail['predicted_seconds']) == len(detail['rollout_seconds']) == 2
        assert np.isfinite(detail['rollout_seconds']).all()
        predictor.accept(action)
    predictor.reset()
    assert predictor.state is None and not predictor.previous.any()


def test_legacy_linear_checkpoint_keeps_module_names_and_exact_predictions():
    artifact = world.fit([model_episode()], hidden=4, epochs=1)
    legacy = deepcopy(artifact)
    legacy.pop('architecture')
    legacy.pop('parameter_count')
    assert set(legacy['state_dict']) == {
        'observe_net.weight_ih', 'observe_net.weight_hh', 'observe_net.bias_ih',
        'observe_net.bias_hh', 'head.weight', 'head.bias'}
    old, current = world.Predictor(legacy), world.Predictor(artifact)
    for raw in model_episode()['obs']:
        assert old.choose(raw, 'refresh') == current.choose(raw, 'refresh')
        old.accept('reuse')
        current.accept('reuse')


@pytest.mark.parametrize('memory,head', [('shared', 'role_mlp'), ('independent_roles', 'role_mlp')])
def test_history_is_causal_and_stateless_ablation_discards_hidden_memory(memory, head):
    artifact = world.fit([model_episode()], hidden=4, epochs=1, memory=memory, head=head)
    predictor = world.Predictor(artifact)
    raw = model_episode()['obs'][0]
    predictor.choose(raw, 'refresh')
    first = predictor.state.clone()
    predictor.choose(raw, 'refresh')
    assert not torch.allclose(first, predictor.state)
    predictor.choose(raw, 'refresh', stateless=True)
    torch.testing.assert_close(predictor.state, first)
    # The public input remains the current observation and previous action;
    # no future matrix or forcing schedule enters these architecture variants.
    for parameter in predictor.net.parameters():
        with torch.no_grad():
            parameter.fill_(float('nan'))
    action, detail = predictor.choose(raw, 'reuse')
    assert action == 'reuse' and detail['reason'] == 'nonfinite_prediction'
    assert predictor.state is None


@pytest.mark.parametrize('key,value', [
    ('world_memory', 'typo'), ('world_head', 'typo'), ('world_head_hidden', 0),
    ('world_head_hidden', True)])
def test_bad_architecture_is_rejected_before_collection(key, value):
    settings = reuse_settings()
    settings[key] = value
    with pytest.raises(ValueError, match='world'):
        study.settings_from({'smoother_reuse': settings})


def test_truncated_and_overlapping_probe_windows_do_not_open_training_gate():
    def rows(cost, count=4):
        return [dict(total_seconds=cost, success=True, pressure_calls=count)]*3
    # Six apparent wins overlap into two independent pressure-call windows.
    ep = dict(trajectory='fixture', probe_horizon=4, available=[[True, True]]*8,
        horizon_measurements=[{'0': rows(1.), '1': rows(2.)}]*3 +
                             [{'0': rows(2.), '1': rows(1.)}]*5)
    report = study.tradeoff_report([ep], .05)
    assert report['nonoverlapping_wins'] == dict(refresh_wins=1, reuse_wins=1)
    assert not report['world_model_candidate']
    # Even a low-cost ending window is not a full four-call payoff observation.
    ep['horizon_measurements'][4] = {'0': rows(2., 3), '1': rows(1., 3)}
    report = study.tradeoff_report([ep], .05)
    assert report['nonoverlapping_wins']['reuse_wins'] == 0
    assert all(r['end_position_exclusive'] <= 8 for r in report['selected_windows'])


def test_world_branch_pins_and_reuses_the_same_measured_baseline(tmp_path, monkeypatch):
    from adaptive_mg.v67.h2_fixed_p import study as base
    from adaptive_mg.v67.h2_fixed_p.data import read
    from adaptive_mg.v67.world_model.data import write_json
    from test_h2_fixed_p_study import settings, make_fixture_recording
    script = Path(__file__).resolve().parents[1]/'scripts/run_v6_7_flow_comparison.py'
    spec = importlib.util.spec_from_file_location('architecture_flow_runner', script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(base, 'source_digest', lambda: 'a'*64)
    cfg = settings()
    cfg['smoother_reuse'] = reuse_settings()
    config_path = tmp_path/'config.json'
    write_json(config_path, cfg)
    inputs = [make_fixture_recording(tmp_path/split, split, offset) for split, offset in
              [('train', 0), ('tune', 7), ('validation', 15), ('test', 30)]]
    parent = tmp_path/'parent'
    base.prepare(config_path, inputs, parent)
    base.train(parent)
    probe = study.tradeoff_report
    # Test the inherited-evidence path; this forced gate is no speed claim.
    monkeypatch.setattr(study, 'tradeoff_report', lambda episodes, margin:
                        dict(probe(episodes, margin), world_model_candidate=True))
    study.collect(parent)
    original = study.train(parent)
    variant = dict(hidden=8, world_memory='independent_roles', world_head='role_mlp', world_head_hidden=3)
    child = runner.clone_world_study(parent, tmp_path/'child', variant)
    with monkeypatch.context() as patch:
        patch.setattr(study, 'run_episode', lambda *a, **k: pytest.fail('world branch retimed the cheap baseline'))
        result = study.train(child)
    assert result['baseline_scores'] == original['baseline_scores']
    assert result['selected'] == original['selected']
    assert result['inherited_baseline']['measured_again'] is False
    assert result['model_details']['world']['hidden'] == 8
    base.evaluate(child, split='validation', repeats=1)
    study.evaluate(child, split='validation', repeats=1)
    base.freeze(child)
    frozen = study.freeze(child)
    assert 'world_parent.json' in frozen['artifacts']
    assert study.check_freeze(child)
    pins = child/'world_parent.json'
    pins.write_text(pins.read_text()+' ')
    with pytest.raises(ValueError, match='inherited temporal baseline evidence changed'):
        study.trained(child)
    bad = runner.clone_world_study(parent, tmp_path/'bad', variant)
    changed = read(bad/'configuration.json')
    changed['smoother_reuse']['minimum_gain'] = .2
    write_json(bad/'configuration.json', changed)
    collection = read(bad/'reuse/collection.json')
    with pytest.raises(ValueError, match='architecture changes only'):
        study.inherited_baselines(bad, collection['contract'], collection)
