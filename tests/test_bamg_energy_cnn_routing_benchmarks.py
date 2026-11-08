from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg.config import MGConfig
from adaptive_mg.v67 import bamg_energy_cnn_routing as routing
from adaptive_mg.v67 import bamg_energy_cnn_routing_benchmark as validation
from adaptive_mg.v67 import bamg_energy_cnn_routing_holdout_benchmark as holdout
from adaptive_mg.v67.bamg_teacher import TeacherConfig
from adaptive_mg.v67.bamg_transfer import WeightCNN
from test_bamg_energy_cnn_benchmark import examples, settings as validation_settings
from test_bamg_energy_cnn_holdout_benchmark import cases, example, identity, settings as holdout_settings


def policy(arm='scratch', seeds=None, sizes=None):
    seeds = [6101, 6102, 6103] if seeds is None else seeds
    sizes = [3, 7] if sizes is None else sizes
    return dict(version=routing.VERSION, arm=arm, stage='finetuned',
        candidates=[10., 30., 50.], model_seeds=seeds, validation_sizes=sizes,
        grid_to_lambda={str(n): 10. if n == min(sizes) else 50. for n in sizes},
        global_lambda=30., unseen_grid_policy='lower_anchor_clamp',
        granularity='finest_grid_whole_hierarchy',
        grid_eligible=dict.fromkeys(map(str, sizes), True), global_eligible=True)


def descriptor(tmp_path, *, arm='scratch', seeds=None, sizes=None):
    paths = {}
    for strength in (10, 30, 50):
        path = tmp_path / f'weight_{strength}.pt'
        path.write_bytes(f'fake checkpoint {strength}'.encode())
        paths[str(strength)] = path
    return dict(kind='grid_router_v1', policy=policy(arm, seeds, sizes), checkpoints=paths)


class ScalarBank:
    def __init__(self, seconds=.01, fail=False):
        self.seconds, self.fail = seconds, fail

    def cycle(self, x, b):
        return np.zeros_like(b)

    def solve(self, b, *, x0, timeout_seconds):
        assert np.array_equal(x0, np.zeros_like(b))
        return dict(converged=not self.fail, warm_seconds=self.seconds, cycles=4,
            tolerance=1e-8, residuals=[1., 1. if self.fail else 1e-10],
            reason='failed' if self.fail else 'converged', fallback_used=False)


def test_checkpoint_identity_binds_every_route_alternative_and_policy(tmp_path):
    route = descriptor(tmp_path)
    paths = {'routed': route, 'fixed': route['checkpoints']['10']}
    hashes = validation.checkpoint_hashes(paths)
    assert hashes['routed']['policy'] == route['policy']
    assert set(hashes['routed']['checkpoint_hashes']) == {'10', '30', '50'}
    assert validation.checkpoint_files(paths) == sorted(route['checkpoints'].values())
    route['checkpoints']['30'].write_bytes(b'changed unused candidate')
    assert validation.checkpoint_hashes(paths) != hashes
    with pytest.raises(ValueError, match='every candidate'):
        validation.checkpoint_hashes({'routed': {**route,
            'checkpoints': {'10': route['checkpoints']['10']}}})
    with pytest.raises(ValueError, match='descriptor'):
        validation.checkpoint_files({'routed': {**route, 'arbitrary': True}})
    path = route['checkpoints']['50']
    path.unlink()
    assert validation.checkpoint_hashes(paths, allow_missing=True)['routed']['checkpoint_hashes']['50'] is None
    with pytest.raises(FileNotFoundError):
        validation.checkpoint_hashes(paths)


def test_route_resolves_finest_grid_once_for_every_learned_level(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    s = validation_settings()
    s['models']['weight'].update(hidden=2, depth=1)
    route = descriptor(tmp_path, seeds=[6101], sizes=[7, 15])
    for strength, path in route['checkpoints'].items():
        torch.manual_seed(int(strength))
        model = WeightCNN(**s['models']['weight']).double()
        torch.save(dict(weight_spec=model.spec, weight_state=model.state_dict()), path)
    n = 15
    line = sp.diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1], format='csr')
    a = sp.kronsum(line, line).tocsr()
    ex = dict(A=a, shape=(n, n), b=np.ones(n*n))
    loaded, routed = [], []
    load, choose = validation._load_weight, routing.route_lambda
    def load_once(path, settings):
        loaded.append(Path(path))
        return load(path, settings)
    def choose_once(rule, n):
        routed.append(n)
        return choose(rule, n)
    monkeypatch.setattr(validation, '_load_weight', load_once)
    monkeypatch.setattr(routing, 'route_lambda', choose_once)
    actual = validation._build_bank(ex, 'routed', route, s)
    assert loaded == [route['checkpoints']['50']]
    assert routed == [15]  # Never reselect lambda=10 on the coarse n=7 level.
    expected = validation._build_bank(ex, 'fixed', route['checkpoints']['50'], s)
    left, right, levels = actual.root, expected.root, 0
    while left.p is not None:
        np.testing.assert_array_equal(left.p.toarray(), right.p.toarray())
        np.testing.assert_allclose(left.coarse.a.toarray(), (left.p.T @ left.a @ left.p).toarray())
        left, right = left.coarse, right.coarse
        levels += 1
    assert levels == 2
    for shape in ((15, 7), (15., 15.), (True, True)):
        with pytest.raises(ValueError, match='square'):
            validation.resolve_checkpoint({'shape': shape}, route)


def test_bootstrap_control_uses_original_selected_teacher_pass(tmp_path):
    torch.set_num_threads(1)
    s = validation_settings()
    s['numerical_baselines'] = ['fixed_C', 'em5_C', 'bootstrap_C']
    s['teacher'].update(vectors=4, setup_passes=2, selection_vectors=2, audit_vectors=2,
                        eigen_count=1, eigen_refine=0, adaptive_vectors=0, augment_vectors=0)
    n = 7
    line = sp.diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1], format='csr')
    a = sp.kronsum(line, line).tocsr()
    ex = dict(A=a, shape=(n, n), b=np.ones(n*n))
    tc = TeacherConfig(**{**s['teacher'], 'k': s['k'], 'l1_cap': s['l1_cap'],
                         'support_mode': 'fixed_bilinear'})
    teacher = validation.build_teacher(a, (n, n), config=tc, mg_config=MGConfig(**s['mg']))
    bank = validation._build_bank(ex, 'bootstrap_C', None, s)
    np.testing.assert_array_equal(bank.root.p.toarray(), teacher.levels[0].P.toarray())
    np.testing.assert_allclose(bank.root.coarse.a.toarray(), (bank.root.p.T @ a @ bank.root.p).toarray())
    info = bank._benchmark_setup_metadata['bootstrap_teacher']
    assert info['selected_pass'] == teacher.diagnostics['selected_pass']
    assert info['failed_passes'] == teacher.diagnostics['failed_passes']
    assert validation._measurement(bank, ex['b'], s['benchmark'])['fallback_used'] is False


def test_validation_measures_router_independently_retains_bootstrap_failures_and_bounds_banks(tmp_path, monkeypatch):
    s = validation_settings()
    s['benchmark']['cache_hierarchies'] = False
    s['numerical_baselines'] = ['fixed_C', 'em5_C', 'bootstrap_C']
    route = descriptor(tmp_path)
    paths = {'routed': route, 'fixed_cnn': route['checkpoints']['10']}
    calls = []
    def build(ex, method, path, settings):
        calls.append((ex['case_id'], method, path))
        if method == 'bootstrap_C':
            raise MemoryError('deliberate bootstrap failure')
        if method == 'routed':
            assert path == route['checkpoints']['10']
        return ScalarBank(.013 if method == 'routed' else .005)
    monkeypatch.setattr(validation, '_build_bank', build)
    out = tmp_path / 'validation'
    result = validation.benchmark(out, examples=examples(), settings=s,
                                  checkpoints=paths, identity={})
    assert result['completed'] and result['methods']['bootstrap_C']['failures'] == 4
    assert result['methods']['bootstrap_C']['warm_seconds'] == 120.
    assert result['methods']['routed']['warm_seconds'] == .013  # Not the chosen fixed CNN's .005.
    assert sum(method == 'routed' for _, method, _ in calls) == 4
    assert sum(method == 'bootstrap_C' for _, method, _ in calls) == 2  # No retries/fallback for failed setup.
    for ex in examples():
        state = validation.read_json(validation._case_path(out, ex['case_id']))
        selected = state['setups']['routed']['routing']
        assert selected['selected_lambda'] == 10. and selected['finest_n'] == 3
        assert selected['one_model_for_entire_hierarchy']
        assert selected['selected_checkpoint_sha256'] == validation.file_digest(paths['fixed_cnn'])
    route['policy']['grid_to_lambda']['3'] = 30.
    with pytest.raises(ValueError, match='identity changed'):
        validation.benchmark(out, examples=examples(), settings=s,
                             checkpoints=paths, identity={}, resume=True)


def test_validation_rejects_checkpoint_mutation_during_measurement(tmp_path, monkeypatch):
    route = descriptor(tmp_path)
    def build(ex, method, path, settings):
        if method == 'routed':
            route['checkpoints']['50'].write_bytes(b'changed unused model while running')
        return ScalarBank()
    monkeypatch.setattr(validation, '_build_bank', build)
    with pytest.raises(ValueError, match='changed while evaluating'):
        validation.benchmark(tmp_path / 'changed', examples=examples(), settings=validation_settings(),
                             checkpoints={'routed': route}, identity={})


def test_holdout_frozen_routing_cohorts_global_reference_and_resume(tmp_path, monkeypatch):
    s = holdout_settings()
    s.update(lambdas=[10, 30, 50], model_seeds=[6101, 6102, 6103],
             numerical_baselines=['fixed_C', 'em5_C', 'bootstrap_C'])
    routes = {arm: descriptor(tmp_path, arm=arm) for arm in ('scratch', 'warm')}
    paths = {routing.method_name(strength, arm, seed): routes[arm]['checkpoints'][str(strength)]
             for strength in s['lambdas'] for arm in ('scratch', 'warm') for seed in s['model_seeds']}
    paths.update({routing.routed_name(arm, seed): routes[arm]
                  for arm in ('scratch', 'warm') for seed in s['model_seeds']})
    frozen = {**identity(), 'routing_selection': {'arms': {arm: route['policy'] for arm, route in routes.items()}}}
    builds = []
    def build(ex, method, path, settings):
        builds.append((ex['case_id'], method, path))
        seed_scale = next((i + 1 for i, seed in enumerate(s['model_seeds']) if f'seed{seed}' in method), 1)
        seconds = .003 if method.startswith('routed') else .006
        return ScalarBank(seconds * seed_scale * (ex['index'] + 1))
    monkeypatch.setattr(holdout, '_build_bank', build)
    out = tmp_path / 'holdout'
    kwargs = dict(cases=cases(), load_example=example, settings=s, checkpoints=paths, identity=frozen)
    first = holdout.benchmark(out, **kwargs, max_cases=1)
    assert not first['completed'] and first['selected_method'] is None
    result = holdout.benchmark(out, **kwargs, resume=True)
    assert result['completed'] and result['selected_method'] is None
    assert result['cohorts']['routed_scratch_finetuned']['seed_count'] == 3
    assert result['cohorts']['routed_scratch_finetuned']['warm_seconds'] == pytest.approx(.009)
    pair = result['routing_comparisons']['scratch']
    assert pair['reference'] == 'lambda_30_scratch_finetuned'
    assert pair['candidate'] == 'routed_scratch_finetuned'
    assert pair['fixed_over_routed_time_ratio'] == pytest.approx(2.)
    assert pair['paired_ci95']['time_ratio_reference_over_candidate'] == pytest.approx([2., 2.])
    assert pair['paired_ci95']['independent_operators'] == 2
    assert not pair['heldout_selection']
    assert 'lambda_10_finetuned' in result['cohort_comparisons']
    assert 'routed_finetuned' in result['cohort_comparisons']
    assert result['cohort_comparisons']['routed_finetuned']['scratch_over_warm_time_ratio'] == pytest.approx(1.)
    state = holdout.read_json(holdout._case_path(out, cases()[0]['case_id']))
    for row in state['setups'][routing.routed_name('scratch', 6101)].values():
        assert row['routing']['selected_lambda'] == 10.
        assert row['routing']['one_model_for_entire_hierarchy']
    count = len(builds)
    holdout.benchmark(out, **kwargs, resume=True)
    assert len(builds) == count
    routes['scratch']['checkpoints']['50'].write_bytes(b'changed candidate unused for n3')
    with pytest.raises(ValueError, match='changed'):
        holdout.benchmark(out, **kwargs, resume=True)


def test_holdout_rejects_missing_seed_and_inconsistent_frozen_router(tmp_path):
    s = holdout_settings()
    s.update(lambdas=[10, 30, 50], model_seeds=[6101, 6102, 6103])
    route = descriptor(tmp_path)
    paths = {routing.routed_name('scratch', seed): route for seed in s['model_seeds'][:-1]}
    with pytest.raises(ValueError, match='omits a prespecified seed'):
        holdout._manifest(cases(), s, paths, identity())
    paths = {routing.method_name(30, arm, seed): route['checkpoints']['30']
             for arm in ('scratch', 'warm') for seed in s['model_seeds']}
    for arm in ('scratch', 'warm'):
        paths.update({routing.routed_name(arm, seed): {**route, 'policy': policy(arm)}
                      for seed in s['model_seeds']})
    frozen = {**identity(), 'routing_selection': {'arms': {arm: policy(arm) for arm in ('scratch', 'warm')}}}
    changed = deepcopy(frozen)
    changed['routing_selection']['arms']['scratch']['grid_to_lambda']['3'] = 50.
    with pytest.raises(ValueError, match='differs from the frozen'):
        holdout._manifest(cases(), s, paths, changed)


def test_bootstrap_builder_exception_is_a_recorded_setup_failure(tmp_path, monkeypatch):
    s = validation_settings()
    s['numerical_baselines'] = ['fixed_C', 'em5_C', 'bootstrap_C']
    def fail(*args, **kwargs):
        raise RuntimeError('no feasible initial bootstrap hierarchy')
    monkeypatch.setattr(validation, 'build_teacher', fail)
    result = validation.benchmark(tmp_path / 'failed_bootstrap', examples=examples(),
                                  settings=s, checkpoints={}, identity={})
    assert result['methods']['bootstrap_C']['failures'] == 4
    assert result['methods']['fixed_C']['failures'] == 0
    state = validation.read_json(validation._case_path(tmp_path / 'failed_bootstrap', 'validation_0'))
    assert 'no feasible initial bootstrap hierarchy' in state['setups']['bootstrap_C']['reason']
    assert state['setups']['bootstrap_C']['fallback_used'] is False
