"""Numerical and provenance contracts; fixture matrices are NOT physical CFD."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from adaptive_mg import DiffusionCase, assemble_stiffness, MGConfig
from adaptive_mg.hierarchy import build_fixed_hierarchy
from adaptive_mg.strategy import get_strategy
from adaptive_mg.v67.banks import Stats, prepare_smoother_bank, hybrid_cycle
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.spatial import SpatialState
from adaptive_mg.v67.unroll import cycle
from adaptive_mg.v67.world_model.data import snapshot
from adaptive_mg.v67.world_model.backend import hierarchy_signature, levels
from adaptive_mg.v67.h2_fixed_p import training as T


def config():
    return AdaptiveConfig(mg=MGConfig(mode='classical', strategy_name='jacobi_energymin_full__em5__v11',
                                     stencil_backend='csr', nn_levels=2),
                          mode='research', branch='H_S', use_transfer=False,
                          spatial=False, gate_mode='open', use_learned_controller=False)


def recorded_fixture(index=0, n=7, *, source_kind='external_cfd', x0=None):
    # Synthetic unit fixture explicitly declares itself nonphysical. Calling
    # the interface external_cfd tests admission, not combustion correctness.
    a = assemble_stiffness(DiffusionCase(n, epsilon=.3, angle_deg=30 + 9 * index,
                                        contrast=3 + index, pattern='channel'))
    xx, yy = np.meshgrid(np.linspace(.1, .9, n), np.linspace(.1, .9, n), indexing='ij')
    b = (1 + np.sin(3 * xx) * np.cos(2 * yy) + index * yy).ravel()
    if x0 is None:
        x0 = (.02 * xx * yy).ravel()
    return snapshot(a, b, x0=x0, shape=(n, n), time=index * .01, index=index,
                    mesh_id='fixed-grid', boundary_id='anchored-pressure',
                    source_kind=source_kind, context={'test_fixture_only': True, 'physical_CFD': False})


def trajectory(states=None, split='train'):
    return [({'id': 'fixture-case', 'case_group': 'unit-fixture', 'split': split},
             states or [recorded_fixture(0), recorded_fixture(1)])]


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_current_A_galerkin_with_all_initial_P_fixed(monkeypatch):
    cfg = config()
    first, second = recorded_fixture(0, n=15), recorded_fixture(1, n=15)
    reference = T._freeze_p(build_fixed_hierarchy(first.a, first.shape,
                         get_strategy(cfg.mg.strategy_name), cfg.mg, Stats()))
    old_signature = hierarchy_signature(reference)
    model = Components.create(hidden=4, seed=17)
    monkeypatch.setattr(model.transfer, 'forward', lambda *_: pytest.fail('learned P called'))
    # If target graphs rebuild EM, fail rather than silently allowing drift.
    import adaptive_mg.energymin as energymin
    monkeypatch.setattr(energymin, 'energy_weights', lambda *a, **kw: pytest.fail('P rebuilt'))
    graph = T.fixed_p_graph(second.a, reference, model, cfg)
    graph_levels = list(T._graph_levels(graph))
    for parent, current in zip(levels(reference), graph_levels):
        if parent.p is not None:
            np.testing.assert_array_equal(current.p.numpy().toarray(), parent.p.toarray())
            np.testing.assert_allclose(current.coarse.raw_scipy.toarray(),
                     (parent.p.T @ current.raw_scipy @ parent.p).toarray(), rtol=1e-13, atol=1e-12)
            assert not current.p.values.requires_grad
            assert not parent.p.data.flags.writeable
    assert hierarchy_signature(reference) == old_signature
    assert not np.allclose(graph.coarse.raw_scipy.toarray(), reference.coarse.a.toarray())


@pytest.mark.parametrize('plan', ['jacobi_energymin_full__em5__v11', 'line_alt_energymin_full__em5__v11'])
def test_differentiable_cycle_matches_runtime_actual_nonzero_x0(plan):
    cfg = config()
    cfg = replace(cfg, mg=replace(cfg.mg, strategy_name=plan))
    state = recorded_fixture()
    model = Components.create(hidden=4, seed=19)
    reference = build_fixed_hierarchy(state.a, state.shape, get_strategy(cfg.mg.strategy_name), cfg.mg, Stats())
    learned = T.fixed_p_graph(state.a, reference, model, cfg)
    b = torch.tensor(state.b)
    x = cycle(learned, torch.tensor(state.x0), b, model, cfg)
    runtime_root = prepare_smoother_bank(reference, model, cfg, Stats())
    expected = hybrid_cycle(runtime_root, state.x0.copy(), state.b.copy(), cfg, Stats(), SpatialState(model, cfg), 0, True)
    np.testing.assert_allclose(x.detach().numpy(), expected, rtol=2e-6, atol=1e-10)
    loss, details, _ = T.residual_objective(state, reference, model, cfg, cycles=2)
    assert details['initial_residual_norm'] == pytest.approx(np.linalg.norm(state.b - state.a @ state.x0))
    assert details['effective_x0_digest'] == T._vector_digest(state.x0)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.smoother.parameters())
    assert all(p.grad is None for p in model.transfer.parameters())


def test_training_updates_only_HS_preserves_real_rhs_x0_and_P(tmp_path, monkeypatch):
    cfg = config()
    calls = []
    original_build = T.build_fixed_hierarchy
    def record_build(*args, **kwargs):
        calls.append(args[0])
        return original_build(*args, **kwargs)
    monkeypatch.setattr(T, 'build_fixed_hierarchy', record_build)
    states = [recorded_fixture(0), recorded_fixture(1)]
    model, status = T.train_smoother(trajectory(states), cfg, tmp_path / 'train', updates=3,
                                    hidden=4, learning_rate=1e-3)
    assert len(calls) == 1
    assert status['optimizer_updates'] == 3
    assert status['final_signatures']['smoother'] != status['initial_signatures']['smoother']
    for name in ('transfer', 'detector', 'controller'):
        assert status['final_signatures'][name] == status['initial_signatures'][name]
    manifest = json.loads((tmp_path / 'train' / 'training_manifest.json').read_text())
    records = json.loads((tmp_path / 'train' / 'training.json').read_text())
    assert manifest['random_rhs_generated'] is False
    pd = manifest['trajectories'][0]['fixed_p_digest']
    for record in records:
        state = states[record['snapshot_index']]
        assert record['rhs_digest'] == T._vector_digest(state.b)
        assert record['exported_x0_digest'] == T._vector_digest(state.x0)
        assert record['initial_residual_digest'] == T._vector_digest(state.b - state.a @ state.x0)
        assert record['fixed_p_digest'] == pd
        assert record['gradient_norm'] > 0
    loaded = Components.load(status['checkpoint'])
    assert loaded.metadata['training_branch'] == 'H_S'
    assert loaded.metadata['optimizer_updates'] == 3
    assert loaded.metadata['fixed_p_contract'] == T.FIXED_P_CONTRACT
    assert not status['performance_certified']
    assert not status['cfd_validation_completed']
    assert loaded.component_signatures() == model.component_signatures()


@pytest.mark.parametrize('bad', ['validation', 'synthetic', 'zero_updates', 'changed_mesh', 'learned_transfer', 'zero_gain'])
def test_invalid_training_contracts_fail_without_candidate(tmp_path, bad):
    cfg = config()
    data = trajectory()
    kwargs = dict(updates=1, hidden=4)
    if bad == 'validation':
        data = trajectory(split='validation')
    elif bad == 'synthetic':
        data = trajectory([recorded_fixture(source_kind='synthetic_elliptic')])
    elif bad == 'zero_updates':
        kwargs['updates'] = 0
    elif bad == 'changed_mesh':
        metadata, states = data[0]
        data = [(metadata, [states[0], replace(states[1], mesh_id='changed')])]
    elif bad == 'learned_transfer':
        cfg = replace(cfg, use_transfer=True)
    elif bad == 'zero_gain':
        cfg = replace(cfg, mg=replace(cfg.mg, smoother_gain_multiplier=0))
    with pytest.raises(ValueError):
        T.train_smoother(data, cfg, tmp_path / bad, **kwargs)
    assert not (tmp_path / bad / 'candidate.pt').exists()


def test_training_rejects_already_solved_data_without_fake_update(tmp_path):
    state = recorded_fixture()
    solved = replace(state, b=state.a @ state.x0)
    with pytest.raises(ValueError, match='no unconverged'):
        T.train_smoother(trajectory([solved]), config(), tmp_path / 'solved', updates=1)
    assert not (tmp_path / 'solved' / 'candidate.pt').exists()


def test_changed_exported_rhs_changes_training_fingerprint_and_weights(tmp_path):
    states = [recorded_fixture(0), recorded_fixture(1)]
    model, status = T.train_smoother(trajectory(states), config(), tmp_path / 'a', updates=1, hidden=4)
    # Only physical input RHS is perturbed; identical seed and matrix cannot
    # erase this change through manufactured/random-RHS replacement.
    changed = [replace(s, b=s.b + np.linspace(-.5, .7, len(s.b))) for s in states]
    other, other_status = T.train_smoother(trajectory(changed), config(), tmp_path / 'b', updates=1, hidden=4)
    assert status['training_fingerprint'] != other_status['training_fingerprint']
    assert model.component_signatures()['smoother'] != other.component_signatures()['smoother']
