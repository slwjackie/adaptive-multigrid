"""Independent level CNNs: real gradient paths, Galerkin inputs and runtime parity.

All systems here are numerical unit fixtures, not validated fluid simulations.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from adaptive_mg.hierarchy import build_fixed_hierarchy
from adaptive_mg.strategy import get_strategy
from adaptive_mg.transfer import matrix_feature_array
from adaptive_mg.v67.banks import Stats, hybrid_cycle, prepare_smoother_bank
from adaptive_mg.v67.h2_fixed_p import training as training
from adaptive_mg.v67.h2_fixed_p.backend import FixedPSolver
from adaptive_mg.v67.level_smoothers import PerLevelSmoother
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.spatial import SpatialState
from adaptive_mg.v67.unroll import cycle

from test_h2_fixed_p_training import config, recorded_fixture, trajectory


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_separate_weights_receive_their_current_galerkin_features_and_gradients():
    cfg = replace(config(), smoother_levels=(0, 1))
    state = recorded_fixture(1, n=15)
    reference = build_fixed_hierarchy(recorded_fixture(0, n=15).a, state.shape,
                                     get_strategy(cfg.mg.strategy_name), cfg.mg, Stats())
    model = Components.create(hidden=4, seed=17, smoother_architecture='per_level',
                              smoother_levels=(0, 1))
    networks = model.smoother.networks
    assert not {p.data_ptr() for p in networks['0'].parameters()} & {
        p.data_ptr() for p in networks['1'].parameters()}
    inputs = {}
    hooks = [network.body.register_forward_pre_hook(
        lambda module, args, key=key: inputs.__setitem__(key, args[0].detach().clone()))
        for key, network in networks.items()]
    loss, _, _ = training.residual_objective(state, reference, model, cfg, cycles=2)
    loss.backward()
    for hook in hooks:
        hook.remove()
    coarse = reference.p.T @ state.a @ reference.p
    for key, a, shape in [('0', state.a, state.shape), ('1', coarse, reference.coarse.shape)]:
        np.testing.assert_allclose(inputs[key][0].numpy(), matrix_feature_array(a.tocsr(), shape),
                                   rtol=2e-6, atol=1e-7)
        assert sum(float(p.grad.square().sum()) for p in networks[key].parameters()
                   if p.grad is not None) > 0
    untouched = {name: value.clone() for name, value in networks['1'].state_dict().items()}
    torch.optim.SGD(networks['0'].parameters(), lr=.01).step()
    assert all(torch.equal(value, networks['1'].state_dict()[name])
               for name, value in untouched.items())


@pytest.mark.parametrize('architecture', ['shared', 'per_level'])
def test_trained_multilevel_checkpoint_roundtrip_matches_runtime_and_solver(tmp_path, architecture):
    cfg = config()
    # Explicit levels take precedence even if the legacy prefix is disabled.
    cfg = replace(cfg, mg=replace(cfg.mg, nn_levels=0), smoother_levels=(0, 1))
    states = [recorded_fixture(0, n=15), recorded_fixture(1, n=15)]
    model, status = training.train_smoother(trajectory(states), cfg, tmp_path / architecture,
        updates=2, hidden=4, smoother_architecture=architecture)
    assert status['smoother_architecture'] == architecture
    assert status['smoother_level_indices'] == [0, 1]
    if architecture == 'per_level':
        assert status['optimizer_updates_by_level'] == {'0': 2, '1': 2}
        for key in ('0', '1'):
            assert status['initial_level_signatures'][key] != status['final_level_signatures'][key]
    loaded = Components.load(status['checkpoint'])
    assert loaded.component_signatures() == model.component_signatures()
    state = states[1]
    base = build_fixed_hierarchy(states[0].a, state.shape,
                                get_strategy(cfg.mg.strategy_name), cfg.mg, Stats())
    graph = training.fixed_p_graph(states[0].a, base, loaded, cfg)
    predicted = cycle(graph, torch.tensor(state.x0), torch.tensor(state.b), loaded, cfg)
    stats = Stats()
    bank = prepare_smoother_bank(base, loaded, cfg, stats)
    actual = hybrid_cycle(bank, state.x0.copy(), state.b.copy(), cfg, Stats(),
                          SpatialState(loaded, cfg), 0, True)
    np.testing.assert_allclose(predicted.detach().numpy(), actual, rtol=3e-6, atol=1e-10)
    assert stats.smoother_nn_calls == 2
    solver = FixedPSolver(cfg, loaded)
    solved = solver.step(states[0])
    assert sum(b.get('smoother_nn_calls', 0) for b in solved.stats['builds']) == 2
    assert solved.stats['actual_neural_used']
    assert solved.stats['hierarchy_p_preserved']
    reused = solver.step(states[1], smoother_action='reuse')
    assert sum(b.get('smoother_nn_calls', 0) for b in reused.stats['builds']) == 0
    assert reused.stats['hierarchy_p_preserved']


def test_legacy_shared_checkpoint_without_architecture_metadata_still_loads(tmp_path):
    model = Components.create(hidden=4)
    model.metadata.pop('smoother_architecture')
    path = tmp_path / 'legacy.pt'
    model.save(path)
    payload = torch.load(path, weights_only=True)
    assert payload['format_version'] == 13
    loaded = Components.load(path)
    assert loaded.component_signatures() == model.component_signatures()
    assert not isinstance(loaded.smoother, PerLevelSmoother)


def test_missing_level_fails_and_does_not_share_another_network():
    model = PerLevelSmoother([0], hidden=4)
    with pytest.raises(ValueError, match='no CNN for level 1'):
        model.for_level(1)
    with pytest.raises(ValueError, match='distinct nonnegative'):
        PerLevelSmoother([0, 0], hidden=4)


def test_untrained_coarse_level_is_rejected_before_neural_trial(tmp_path):
    cfg = config()
    model, _ = training.train_smoother(trajectory([recorded_fixture(n=7)]), cfg,
        tmp_path / 'single_level', updates=1, hidden=4, smoother_architecture='per_level')
    assert model.smoother.level_indices == (0,)
    with pytest.raises(ValueError, match='missing requested levels: \\[1\\]'):
        FixedPSolver(cfg, model).step(recorded_fixture(n=15))


def test_per_level_checkpoint_runs_through_live_ldu_service(tmp_path):
    from test_h2_fixed_p_service import cfg as service_cfg, contract, request
    from adaptive_mg.v67.h2_fixed_p.service import Session
    cfg = service_cfg()
    _, status = training.train_smoother(trajectory([recorded_fixture(n=7)]), cfg,
        tmp_path / 'service_model', updates=1, hidden=4, smoother_architecture='per_level')
    session = Session(cfg, contract(), mode='hs', expert=Components.load(status['checkpoint']))
    value, _, exact = request(mode='hs')
    result = session.handle(value)
    assert result['success']
    assert result['step']['stats']['actual_neural_used']
    assert sum(b.get('smoother_nn_calls', 0) for b in result['step']['stats']['builds']) == 1
    np.testing.assert_allclose(np.asarray(result['x_native'])[value['structured_to_native']],
                               exact, atol=1e-6)
