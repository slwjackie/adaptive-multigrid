"""Measured warm adaptation, failure-inclusive selection and exact resume."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from adaptive_mg.data import make_examples
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.em_line_models import make_em_line_components
from adaptive_mg.v67.fixed_classical import BilinearClassicalBaseline
from adaptive_mg.v67 import light_smoother_evaluation as ev


def setup():
    root = Path(__file__).resolve().parents[1]
    cfg = AdaptiveConfig.from_dict(json.loads((root/'configs/v6_7_em_line_smoke.json').read_text())['solver'])
    baseline = BilinearClassicalBaseline.from_config(cfg)
    example = make_examples([7], per_size=1, seed=31)[0]
    example.classical_selection = baseline.selection_record()
    example.research_split = 'validation'
    model = make_em_line_components(hidden=8, seed=27)
    arms = dict(fixed_C=dict(model=None, branch='C', selector=False),
                learned_P_line=dict(model=model, branch='H_P', selector=False, light=True))
    return cfg, baseline, example, arms


@pytest.mark.parametrize('regime,expected_wall,expected_setup', [('cold', 7., 2.), ('warm', 5., 0.)])
def test_online_cost_is_inside_wall_including_warm(monkeypatch, regime, expected_wall, expected_setup):
    clock = [0.]
    events = []
    cfg = SimpleNamespace(inference_dtype='float32', to_dict=lambda: {})
    example = SimpleNamespace(a=sp.eye(2, format='csr'), n=2, digest='a', group_digest='a')
    baseline = SimpleNamespace(selection_record=lambda: {})
    class Prepared:
        classical = learned = None
        config = cfg
        online_enabled = True
        def __init__(self):
            clock[0] += 2.
        def prepare_warm(self):
            events.append('prepare')
        def reset_online(self):
            events.append('reset')
        def solve(self, b, x):
            events.append(('solve', self.online_enabled, b.tolist()))
            clock[0] += 5. if self.online_enabled else 1.
            return dict(verified_success=True, convergence_factor=.2, cycles=2, executed_cycles=2,
                stats=dict(online_adaptation_seconds=4. if self.online_enabled else 0.,
                           online_adaptation_calls=int(self.online_enabled),
                           online_adaptation_accepted=0, online_adaptation_rejected=int(self.online_enabled),
                           branch_H_P_cycles=2), setup_stats={})
    monkeypatch.setattr(ev, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(ev, '_construct_light', lambda *args: Prepared())
    monkeypatch.setattr(ev, 'manufactured_rhs', lambda example, count: (
        np.array([[i+1., i+2.] for i in range(count)]), np.ones((count, 2))))
    monkeypatch.setattr(ev, '_result_record', lambda result, *args: result)
    record = ev.measured_light(example, dict(light=True, branch='H_P'), cfg, baseline, regime=regime)
    assert record['wall_seconds'] == expected_wall
    assert record['total_setup_seconds'] == expected_setup
    assert record['solve_seconds'] == 5.
    assert record['online_adaptation_seconds'] == 4.
    assert record['online_adaptation_rejected'] == 1
    assert record['wall_seconds'] == record['total_setup_seconds'] + record['solve_seconds']
    assert record['successful']
    if regime == 'warm':
        assert record['warm_prime_seconds'] == 3.
        assert record['warm_prime_counters']['online_adaptation_calls'] == 0
        assert events == ['prepare', ('solve', False, [2., 3.]), 'reset', ('solve', True, [1., 2.])]
        assert record['warm_ttt_seconds'] == 4.


def test_real_warm_basic_arm_does_not_turn_on_adaptation():
    cfg, baseline, example, arms = setup()
    # This invokes the actual eager hierarchy, prime, reset and timed solver.
    # The default online config is off and must stay off after warm priming.
    record = ev.measured_light(example, arms['learned_P_line'], cfg, baseline, regime='warm')
    assert record['successful']
    assert record['total_setup_seconds'] == 0.
    assert record['online_adaptation_calls'] == 0
    assert record['online_adaptation_seconds'] == 0.
    assert record['warm_prime_counters']['online_adaptation_calls'] == 0
    assert record['coefficient_generation_count'] == 0
    assert record['factor_rebuilds'] == record['hierarchy_rebuilds'] == 0


def test_real_warm_mlp_ttt_is_charged_and_preserves_predictor():
    import torch
    from adaptive_mg.v67.light_smoother_models import MLPPredictor, Standardizer, TinyMLP
    cfg, baseline, example, arms = setup()
    torch.set_num_threads(1)
    predictor = MLPPredictor(TinyMLP(bounds=(.8, 1.2)), Standardizer.fit(np.zeros((2, 12))))
    frozen = deepcopy(predictor.to_dict())
    arm = dict(arms['learned_P_line'], predictor=predictor,
        online=dict(enabled=True, method='mlp_ttt', trigger_window=1, trigger_rho=.00001,
                    min_remaining_cycles=.01, near_tolerance_factor=1., cooldown_cycles=1,
                    gradient_steps=1, max_calls=1, time_budget_seconds=10.))
    record = ev.measured_light(example, arm, cfg, baseline, regime='warm')
    assert record['successful']
    assert record['warm_prime_counters']['online_adaptation_calls'] == 0
    assert record['online_adaptation_calls'] == 1
    assert 0 < record['online_adaptation_seconds'] <= record['solve_seconds'] == record['wall_seconds']
    assert record['all_executed_cycles'] > record['executed_cycles']
    assert record['mlp_inference_calls'] == 0  # only once in excluded setup
    assert record['factor_rebuilds'] == record['hierarchy_rebuilds'] == 0
    assert predictor.to_dict() == frozen


def test_deployment_setup_failure_uses_and_reports_classical_recovery():
    cfg, baseline, example, arms = setup()
    class InvalidPredictor:
        kind = 'regression'
        bounds = (.8, 1.2)
        def predict(self, features):
            return np.array([np.nan, 1.])
    record = ev.measured_light(example,
        dict(arms['learned_P_line'], predictor=InvalidPredictor()), cfg, baseline)
    assert record['successful']
    assert record['setup_fallback'] and record['setup_fallback_reason']
    assert record['setup_failures'] == 1
    assert record['fallback_count'] > 0
    assert not record['actual_neural_used']
    assert record['total_setup_seconds'] > 0
    assert record['wall_seconds'] >= record['total_setup_seconds']


def synthetic_run(success=True, wall=2.):
    return dict(successful=success, success=success, wall_seconds=wall, seconds_per_rhs=wall,
        warm_prime_seconds=0., setup_scope='included_once', total_setup_seconds=.5,
        solve_seconds=wall-.5, rhs_results=[dict(relative_solution_error=1e-10, relative_energy_error=1e-10)],
        actual_neural_used=False, **{key: 0. for key in ev.EXTRA_METRICS})


def test_measurement_resume_retains_records_and_failure(monkeypatch, tmp_path):
    cfg, baseline, example, arms = setup()
    calls = []
    def measure(example, arm, cfg, baseline, **kwargs):
        calls.append(arm['branch'])
        if len(calls) == 3:
            raise KeyboardInterrupt('interrupted')
        return synthetic_run(success=arm['branch'] == 'C', wall=2. if arm['branch'] == 'C' else .1)
    monkeypatch.setattr(ev, 'measured_light', measure)
    args = dict(repeats=2, warmups=0, rhs_counts=[1], regimes=['multiple'])
    with pytest.raises(KeyboardInterrupt):
        ev.evaluate_light([example], arms, cfg, baseline, tmp_path, **args)
    saved = {p.name: p.read_bytes() for p in (tmp_path/'repeat_records').glob('*.json')}
    assert len(saved) == 2
    monkeypatch.setattr(ev, 'measured_light', lambda e, a, *x, **kw: synthetic_run(a['branch'] == 'C', 2. if a['branch'] == 'C' else .1))
    result = ev.evaluate_light([example], arms, cfg, baseline, tmp_path, resume=True, **args)
    assert {name: (tmp_path/'repeat_records'/name).read_bytes() for name in saved} == saved
    assert result['summary']['multiple']['1']['learned_P_line']['successes'] == 0
    selection = ev.select_light_candidate(result)
    assert selection['recommended_deployment_arm'] == 'fixed_C'
    assert not selection['candidates']['learned_P_line']['eligible']
    raw = (tmp_path/'raw_results.json').read_bytes()
    replay = ev.evaluate_light([example], arms, cfg, baseline, tmp_path, resume=True, **args)
    assert (tmp_path/'raw_results.json').read_bytes() == raw
    assert replay['summary'] == result['summary']
    assert json.loads((tmp_path/'progress.json').read_text())['completed_measurements'] == 4
    with pytest.raises(ValueError, match='mismatch'):
        ev.evaluate_light([example], arms, cfg, baseline, tmp_path, resume=True, **dict(args, repeats=3))


def test_changed_measurement_digest_is_rejected(monkeypatch, tmp_path):
    cfg, baseline, example, arms = setup()
    monkeypatch.setattr(ev, 'measured_light', lambda *args, **kwargs: synthetic_run())
    args = dict(repeats=1, warmups=0, rhs_counts=[1], regimes=['multiple'])
    ev.evaluate_light([example], arms, cfg, baseline, tmp_path, **args)
    path = next((tmp_path/'repeat_records').glob('*.json'))
    record = json.loads(path.read_text())
    record['run']['wall_seconds'] *= .1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='record changed'):
        ev.evaluate_light([example], arms, cfg, baseline, tmp_path, resume=True, **args)


def test_optional_line_cnn_must_share_identical_p():
    cfg, baseline, _, arms = setup()
    changed = deepcopy(arms['learned_P_line']['model'])
    import torch
    with torch.no_grad():
        changed.transfer.head.bias.add_(.1)
    arms['frozen_P_line'] = dict(model=changed, branch='H_SP', selector=False)
    with pytest.raises(ValueError, match='identical P'):
        ev._validated_arms(arms, cfg, baseline)


def selection_result():
    arms = ('fixed_C', 'learned_P_line', 'learned_P_MLP')
    rows = []
    for i in range(2):
        runs = {name: {'multiple': {'1': [synthetic_run(wall=t) for _ in range(2)]}}
                for name, t in zip(arms, (3., 2., 1.))}
        rows.append(dict(example=dict(name=f'v{i}', case=dict(n=7)), research_split='validation', runs=runs))
    return dict(manifest=dict(repeats=2, arms={name: {} for name in arms}), rows=rows,
                summary={'multiple': {'1': {name: {} for name in arms}}})


def test_selection_uses_total_and_all_repeats_not_warm():
    result = selection_result()
    assert ev.select_light_candidate(result)['selected_arm'] == 'learned_P_MLP'
    result['rows'][0]['runs']['learned_P_MLP']['multiple']['1'][0]['successful'] = False
    assert ev.select_light_candidate(result)['selected_arm'] == 'learned_P_line'
    result['rows'][1]['runs']['learned_P_line']['multiple']['1'].pop()
    assert ev.select_light_candidate(result)['recommended_deployment_arm'] == 'fixed_C'


def test_pairwise_report_separates_predictor_and_online_effects():
    result = selection_result()
    result['manifest']['arms']['learned_P_MLP_TTT'] = {}
    for row in result['rows']:
        row['runs']['learned_P_MLP_TTT'] = {'multiple': {'1': [synthetic_run(wall=.5) for _ in range(2)]}}
    result['rows'][0]['runs']['learned_P_MLP_TTT']['multiple']['1'][0]['successful'] = False
    effects = ev.paired_effects(result)
    basic = effects['multiple/rhs1/learned_P_MLP_vs_learned_P_line']
    online = effects['multiple/rhs1/learned_P_MLP_TTT_vs_learned_P_MLP']
    assert basic['paired_speedup']['geometric_speedup'] == 2.
    assert online['paired_speedup']['geometric_speedup'] == 2.
    assert online['candidate_successes'] == 1
    assert online['excluded_failure_case_ids'] == ['v0']
    assert len(online['rows']) == 2


@pytest.mark.parametrize('split', ['train', 'final', 'ood_grid'])
def test_selection_refuses_nonvalidation(split):
    result = selection_result()
    result['rows'][0]['research_split'] = split
    with pytest.raises(ValueError, match='Only validation'):
        ev.select_light_candidate(result)


def test_selection_refuses_excluded_setup():
    result = selection_result()
    result['rows'][0]['runs']['learned_P_MLP']['multiple']['1'][0]['setup_scope'] = 'excluded_prepared_operator'
    with pytest.raises(ValueError, match='exclude setup'):
        ev.select_light_candidate(result)
