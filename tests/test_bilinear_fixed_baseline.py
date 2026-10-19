"""Bilinear parent isolation, immutable provenance, and independent EM5 reference."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from adaptive_mg import MGConfig
from adaptive_mg.v67 import AdaptiveConfig, Components
from adaptive_mg.v67.banks import Stats
from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.fixed_classical import (
    BilinearClassicalBaseline, FixedClassicalBaseline, BILINEAR_STRATEGY_NAME,
    FIXED_STRATEGY_NAME, VERSION, baseline_metadata, load_baseline,
)
from adaptive_mg.v67.research_data import (
    make_research_plan, prepare_development_data, load_development_data,
)
from adaptive_mg.v67.research_training import sample_config
from adaptive_mg.v67.solver import PreparedAdaptiveMG
from adaptive_mg.v67.unroll import make_graph
import adaptive_mg.v67.research_evaluation as evaluation


def inputs(n=7):
    torch.set_num_threads(1)
    cfg = AdaptiveConfig(mg=MGConfig(
        mode='classical', strategy_name=BILINEAR_STRATEGY_NAME,
        coarsest_n=3, max_cycles=60, stencil_backend='csr', near_tolerance_factor=0.),
        mode='research', branch='H_P', use_smoother=False, use_transfer=True,
        transfer_levels='all', spatial=False, gate_mode='open',
        use_learned_controller=False, break_even=False)
    baseline = BilinearClassicalBaseline.from_config(cfg)
    example = dataset([n], 7, 981743)[0]
    example.research_split = 'validation'
    example.classical_selection = baseline.selection_record()
    model = Components.create(hidden=4, seed=143)
    return cfg, baseline, example, model


def em_reference(cfg):
    em_cfg = replace(cfg, mg=replace(cfg.mg, strategy_name=FIXED_STRATEGY_NAME,
                                    em_iterations=5, em_fixed_iterations=True, em_rtol=1e-8))
    return dict(model=None, branch='C', selector=False, reference_only=True,
                config=em_cfg, rules=FixedClassicalBaseline.from_config(em_cfg))


def test_bilinear_and_em_contract_roundtrips_are_distinct_and_strict(tmp_path):
    cfg, baseline, _, _ = inputs()
    assert isinstance(baseline, FixedClassicalBaseline)
    em = em_reference(cfg)['rules']
    assert em.to_dict()['version'] == VERSION
    assert baseline.digest() != em.digest()
    for contract in (baseline, em):
        path = tmp_path / (contract.version + '.json')
        path.write_text(json.dumps(contract.to_dict()))
        restored = load_baseline(path)
        assert restored == contract and type(restored) is type(contract)
        assert baseline_metadata(contract, include_contract=True) == {
            'baseline_digest': contract.digest(), 'baseline': contract.to_dict()}
    with pytest.raises(ValueError, match='Unsupported'):
        FixedClassicalBaseline.from_dict(baseline.to_dict())
    with pytest.raises(ValueError, match='Unsupported'):
        BilinearClassicalBaseline.from_dict(em.to_dict())
    with pytest.raises(ValueError, match='requires'):
        BilinearClassicalBaseline.from_config(em_reference(cfg)['config'])
    for key, value in [('tolerance', 1e-7), ('coarsest_n', 1), ('em_iterations', 3)]:
        with pytest.raises(ValueError, match='frozen baseline'):
            baseline.validate_config(replace(cfg, mg=replace(cfg.mg, **{key: value})))
    malformed = baseline.to_dict()
    malformed['mg']['post_steps'] = 1
    with pytest.raises(ValueError, match='pre/post'):
        BilinearClassicalBaseline.from_dict(malformed)


def test_bilinear_data_labels_and_training_config_remain_frozen(tmp_path):
    cfg, baseline, _, _ = inputs()
    plan = make_research_plan(seed=189247)
    for spec in plan['splits'].values():
        spec.update(sizes=[7], families=['variable_coefficient'], per_family=1, count=1)
    history = tmp_path / 'history'
    history.mkdir()
    examples, validation, manifest = prepare_development_data(
        tmp_path / 'data', baseline, plan=plan, historical_roots=[history])
    recovered, restored_validation, restored = load_development_data(tmp_path / 'data', baseline)
    assert restored == manifest
    assert len(restored_validation) == len(validation)
    for example in examples + recovered:
        assert example.classical_selection == baseline.selection_record()
        assert sample_config(example, cfg, baseline, 'H_P').mg == cfg.mg


def test_no_em_in_bilinear_parent_learned_hierarchy_or_training_graph(monkeypatch):
    import adaptive_mg.energymin as energymin
    cfg, baseline, example, model = inputs(15)
    monkeypatch.setattr(energymin, 'energy_weights',
                        lambda *args, **kwargs: pytest.fail('Bilinear experiment invoked EM'))
    prepared = PreparedAdaptiveMG(example.a, example.n, model, cfg)
    stats = Stats()
    root = prepared.ensure_learned(stats)
    graph = make_graph(example.a, (example.n, example.n), model, cfg, learned=True)
    assert graph.coarse.coarse is not None  # More than one generated level.
    for hierarchy in (prepared.classical, root):
        while hierarchy.coarse is not None:
            assert hierarchy.strategy.transfer == 'bilinear'
            assert hierarchy.strategy.coarsening == 'full'
            hierarchy = hierarchy.coarse
    assert stats.em_iterations == 0
    result = prepared.solve(example.b)
    assert result.converged


def test_failed_neural_cycle_recovers_only_with_bilinear_parent(monkeypatch):
    import adaptive_mg.energymin as energymin
    import adaptive_mg.v67.solver as solver
    cfg, baseline, example, model = inputs()
    monkeypatch.setattr(energymin, 'energy_weights',
                        lambda *args, **kwargs: pytest.fail('Fallback invoked EM'))
    original = solver.classical_cycle
    visited = []
    def classical(level, x, b, mg, stats):
        visited.append(level.strategy.name)
        return original(level, x, b, mg, stats)
    monkeypatch.setattr(solver, 'classical_cycle', classical)
    monkeypatch.setattr(solver, 'hybrid_cycle',
                        lambda level, x, *args, **kwargs: np.full_like(x, np.nan))
    run = evaluation.measured_research(example,
        dict(model=model, branch='H_P', selector=False), cfg, baseline)
    assert run['success']
    assert run['rollback_count'] == 1 and run['fallback_count'] == 1
    assert visited and set(visited) == {BILINEAR_STRATEGY_NAME}
    assert run['counters']['em_iterations'] == 0
    assert run['selection'] == baseline.selection_record()
    assert 'no energy minimization' in run['classical_setup_scope']


def test_em5_reference_is_measured_separately_with_its_own_parent(tmp_path, monkeypatch):
    import adaptive_mg.energymin as energymin
    cfg, baseline, example, model = inputs()
    em = em_reference(cfg)
    calls = []
    original = energymin.energy_weights
    def measured(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(energymin, 'energy_weights', measured)
    report = evaluation.evaluate_research([example], {
        'fixed_C': dict(model=None, branch='C', selector=False),
        'direct_CNN': dict(model=model, branch='H_P', selector=False),
        'EM5_C': em,
    }, cfg, baseline, tmp_path, repeats=1, warmups=0, regimes=('cold',),
        rhs_counts=(1,), reference_arm='fixed_C')
    assert len(calls) == 1 and calls[0]['maxiter'] == 5 and calls[0]['fixed_iterations']
    runs = report['rows'][0]['runs']
    for name in ('fixed_C', 'direct_CNN'):
        run = runs[name]['cold']['1'][0]
        assert run['success'] and run['counters']['em_iterations'] == 0
        assert run['selection'] == baseline.selection_record()
    run = runs['EM5_C']['cold']['1'][0]
    # The EM5 budget may stop at a machine-zero projected gradient.
    assert run['success'] and 0 < run['counters']['em_iterations'] <= 5
    assert run['selection'] == em['rules'].selection_record()
    assert report['manifest']['arms']['EM5_C']['reference_only']
    assert report['manifest']['arms']['EM5_C']['baseline_digest'] == em['rules'].digest()


@pytest.mark.parametrize('change', ['unlabelled', 'neural', 'tolerance'])
def test_em_reference_cannot_change_parent_or_numerical_protocol(tmp_path, change):
    cfg, baseline, example, model = inputs()
    em = em_reference(cfg)
    if change == 'unlabelled':
        em['reference_only'] = False
    elif change == 'neural':
        em.update(branch='H_P', model=model)
    else:
        em['config'] = replace(em['config'], mg=replace(em['config'].mg, tolerance=1e-6))
        em['rules'] = FixedClassicalBaseline.from_config(em['config'])
    with pytest.raises(ValueError, match='different-parent|protocol'):
        evaluation.evaluate_research([example], {
            'fixed_C': dict(model=None, branch='C', selector=False), 'EM5_C': em,
        }, cfg, baseline, tmp_path, repeats=1, warmups=0,
            regimes=('cold',), rhs_counts=(1,), reference_arm='fixed_C')
