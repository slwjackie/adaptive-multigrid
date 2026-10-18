"""Fixed EM parent provenance, measured setup cost and classical recovery."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.v67 import AdaptiveConfig, Components
from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.fixed_classical import FixedClassicalBaseline, FIXED_STRATEGY_NAME
from adaptive_mg.v67.unet_transfer import CompactTransferStudent
import adaptive_mg.v67.research_evaluation as evaluation


def _inputs():
    torch.set_num_threads(1)
    cfg = AdaptiveConfig(mg=MGConfig(
        mode='classical', strategy_name=FIXED_STRATEGY_NAME,
        pre_steps=2, post_steps=2, em_iterations=5, em_fixed_iterations=True,
        em_rtol=1e-8, tolerance=1e-8, absolute_tolerance=1e-12,
        max_cycles=60, coarsest_n=3, nn_levels=1, near_tolerance_factor=0.,
        stencil_backend='csr'), mode='research', branch='H_P',
        use_smoother=False, use_transfer=True, spatial=False,
        gate_mode='open', inference_device='cpu', inference_dtype='float32')
    baseline = FixedClassicalBaseline.from_config(cfg)
    e = dataset([7], 7, 781421)[3]
    e.research_split = 'validation'
    e.classical_selection = baseline.selection_record()
    model = Components.create(hidden=4, seed=981321)
    model.transfer = CompactTransferStudent(hidden=4)
    specs = {'fixed_C': dict(model=None, branch='C', selector=False),
             'direct_CNN': dict(model=model, branch='H_P', selector=False)}
    return e, cfg, baseline, specs


def _evaluate(tmp_path, *, examples=None, specs=None, cfg=None, baseline=None, **kwargs):
    e, default_cfg, default_baseline, default_specs = _inputs()
    return evaluation.evaluate_research(
        examples or [e], specs or default_specs, cfg or default_cfg,
        baseline or default_baseline, tmp_path, repeats=1, warmups=0,
        regimes=('cold',), rhs_counts=(1,), reference_arm='fixed_C', **kwargs)


def test_fixed_comparison_names_and_parent_provenance(tmp_path):
    e, cfg, baseline, specs = _inputs()
    report = _evaluate(tmp_path, examples=[e], specs=specs, cfg=cfg, baseline=baseline)
    assert report['reference_arm'] == 'fixed_C'
    assert report['manifest']['baseline_digest'] == baseline.digest()
    assert report['manifest']['baseline'] == baseline.to_dict()
    assert 'rules' not in report['manifest'] and 'rules_digest' not in report['manifest']
    assert 'strong' not in json.dumps(report)
    for name in specs:
        metadata = report['manifest']['arms'][name]
        assert metadata['baseline_digest'] == baseline.digest()
        assert not metadata['selector']
        run = report['rows'][0]['runs'][name]['cold']['1'][0]
        assert run['success']
        assert run['selection'] == baseline.selection_record()
        assert run['measurement_config']['mg'] == cfg.to_dict()['mg']
        assert run['counters']['em_iterations'] == 5
        assert run['classical_hierarchy'][0]['transfer_setup_report']['iterations'] == 5
        assert not run['selector_in_wall_time']
        summary = report['summary']['cold']['1'][name]
        assert 'speedup_vs_fixed' in summary
        assert 'common_cohort_speedup_vs_fixed' in summary
        assert 'accuracy_worse_vs_fixed_case_ids' in summary
    assert report['summary']['cold']['1']['fixed_C']['geometric_speedup'] == 1.
    assert 'speedup_vs_fixed' in (tmp_path / 'comparison.csv').read_text()


@pytest.mark.parametrize('arm', ['fixed_C', 'direct_CNN'])
def test_selector_and_policy_are_forbidden_for_every_fixed_arm(tmp_path, arm):
    e, cfg, baseline, specs = _inputs()
    for key, value in [('selector', True), ('policy', object())]:
        altered = deepcopy(specs)
        altered[arm][key] = value
        with pytest.raises(ValueError, match='selector|polic'):
            _evaluate(tmp_path, examples=[e], specs=altered, cfg=cfg, baseline=baseline)


def test_fixed_parent_drift_and_wrong_labels_are_rejected(tmp_path):
    e, cfg, baseline, specs = _inputs()
    for changed_mg in (replace(cfg.mg, em_rtol=1e-6),
                       replace(cfg.mg, em_fixed_iterations=False),
                       replace(cfg.mg, coarsest_n=1),
                       replace(cfg.mg, strategy_name='jacobi_bilinear_full')):
        with pytest.raises(ValueError, match='frozen baseline'):
            _evaluate(tmp_path, examples=[e], specs=specs,
                      cfg=replace(cfg, mg=changed_mg), baseline=baseline)
    specs['direct_CNN']['config'] = replace(cfg, mg=replace(cfg.mg, max_cycles=100))
    with pytest.raises(ValueError, match='frozen baseline'):
        _evaluate(tmp_path, examples=[e], specs=specs, cfg=cfg, baseline=baseline)
    specs['direct_CNN'].pop('config')
    e.classical_selection['baseline_digest'] = 'other-parent'
    with pytest.raises(ValueError, match='labelled'):
        _evaluate(tmp_path, examples=[e], specs=specs, cfg=cfg, baseline=baseline)
    e.classical_selection = baseline.selection_record()
    e.strong_selection = {'rules_digest': 'legacy-selector'}
    with pytest.raises(ValueError, match='selector-labelled'):
        _evaluate(tmp_path, examples=[e], specs=specs, cfg=cfg, baseline=baseline)


@pytest.mark.parametrize('regime,count', [('cold', 1), ('multiple', 3)])
def test_em_cnn_galerkin_and_failed_trial_recovery_all_inside_wall(monkeypatch, regime, count):
    """Advance a virtual clock inside real numerical setup/solve operations.

    Failing NN trials must recover with exactly the fixed EM hierarchy, count
    every wasted trial and recovery in wall time, and lock only that RHS.
    """
    import adaptive_mg.energymin as energymin
    import adaptive_mg.hierarchy as hierarchy
    import adaptive_mg.v67.banks as banks
    import adaptive_mg.v67.solver as solver

    e, cfg, baseline, specs = _inputs()
    model = specs['direct_CNN']['model']
    with torch.no_grad():
        model.transfer.head.bias.copy_(torch.linspace(-.01, .01, 16))
    reference = solver.PreparedAdaptiveMG(e.a, e.n, None, replace(cfg, branch='C'))
    expected_p = reference.classical.p.copy()
    clock, events = [0.], []

    def tick(event, seconds):
        clock[0] += seconds
        events.append((event, seconds))

    def wrap(module, name, event, seconds):
        original = getattr(module, name)
        def measured(*args, **kwargs):
            tick(event, seconds)
            return original(*args, **kwargs)
        monkeypatch.setattr(module, name, measured)

    monkeypatch.setattr(evaluation, 'perf_counter', lambda: clock[0])
    wrap(energymin, 'energy_weights', 'EM', 7.)
    wrap(CompactTransferStudent, 'forward', 'CNN', 11.)
    wrap(hierarchy, 'galerkin_coarse_operator', 'classical_Galerkin', 17.)
    wrap(banks, 'galerkin_coarse_operator', 'learned_Galerkin', 19.)
    original_cycle = solver.classical_cycle
    def classical(level, x, b, mg, stats):
        tick('classical_recovery_or_tail', 13.)
        assert level.strategy.name == FIXED_STRATEGY_NAME
        np.testing.assert_array_equal(level.p.toarray(), expected_p.toarray())
        assert mg == cfg.mg
        return original_cycle(level, x, b, mg, stats)
    monkeypatch.setattr(solver, 'classical_cycle', classical)
    def rejected(level, x, *args, **kwargs):
        tick('rejected_NN_trial', 5.)
        return np.full_like(x, np.nan)
    monkeypatch.setattr(solver, 'hybrid_cycle', rejected)
    original_verify = evaluation._result_record
    def verify(*args, **kwargs):
        tick('post_timer_verification', 1000.)
        return original_verify(*args, **kwargs)
    monkeypatch.setattr(evaluation, '_result_record', verify)

    run = evaluation.measured_research(
        e, specs['direct_CNN'], cfg, baseline, regime=regime, rhs_count=count)
    assert run['success'], run.get('error')
    assert run['wall_seconds'] == sum(seconds for event, seconds in events
                                     if event != 'post_timer_verification')
    assert run['constructor_seconds'] == 7. + 17.
    assert run['counters']['em_iterations'] == 5
    assert sum(event == 'EM' for event, _ in events) == 1
    assert sum(event == 'CNN' for event, _ in events) == 1
    assert any(event == 'learned_Galerkin' for event, _ in events)
    assert sum(event == 'rejected_NN_trial' for event, _ in events) == count
    assert run['rollback_count'] == count
    assert run['fallback_count'] == count
    assert run['counters']['classical_lock_cycles'] > 0
    assert run['setup_scope'] == 'included_once'
    assert not run['actual_neural_used']
    for result in run['rhs_results']:
        assert result['cycle_path'][0] == 'classical_recovery'
        assert set(result['cycle_path'][1:]) == {'classical'}
        assert result['stats']['branch_H_P_attempts'] == 1
        assert result['measurement_config']['mg'] == cfg.to_dict()['mg']


def test_fixed_completed_resume_preserves_evidence_without_new_solves(tmp_path, monkeypatch):
    e, cfg, baseline, specs = _inputs()
    original = _evaluate(tmp_path, examples=[e], specs=specs, cfg=cfg, baseline=baseline)
    evidence = (tmp_path / 'comparison.json').read_bytes()
    monkeypatch.setattr(evaluation, 'measured_research',
        lambda *args, **kwargs: pytest.fail('Completed fixed resume reran a solve'))
    resumed = _evaluate(tmp_path, examples=[e], specs=specs, cfg=cfg,
                        baseline=baseline, resume=True)
    assert resumed['rows'] == original['rows']
    assert (tmp_path / 'comparison.json').read_bytes() == evidence
