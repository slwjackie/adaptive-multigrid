from dataclasses import replace
import json
import pytest
from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.limited import aggregate_comparison, safe_output, diagnostic_cases, forced_config


def run(success, seconds):
    cfg = AdaptiveConfig()
    return dict(converged=success, wall_seconds=seconds, cycles=4, stats={}, setup_stats={},
        final_true_residual=cfg.mg.tolerance / 2 if success else 1., stopping_threshold=cfg.mg.tolerance,
        residual_history=[1., cfg.mg.tolerance / 2 if success else 1.],
        relative_residual_history=[1., cfg.mg.tolerance / 2 if success else 1.], executed_cycles=4)


def test_failed_repeat_never_fast_success():
    arms = {name: dict(checkpoint_id=name, branch='C', scope='test', config=AdaptiveConfig())
            for name in ('classical', 'candidate')}
    rows = [dict(example={'name': 'a', 'case': {'n': 7}}, runs={
        'classical': [run(True, 10), run(True, 10)],
        'candidate': [run(True, 1), run(False, .01)]})]
    result = aggregate_comparison(rows, arms)
    candidate = result['table'][1]
    assert candidate['successful_repeats'] == 1 and not candidate['success']
    assert candidate['successful_time_to_tolerance_seconds'] is None
    assert candidate['speedup_vs_classical'] is None
    assert candidate['speedup_vs_incumbent'] is None
    assert result['summary']['candidate']['new_failure_case_ids'] == ['a']


def test_success_speedup_keeps_reference_fields_separate():
    cfg = AdaptiveConfig()
    arms = {n: dict(checkpoint_id=n, branch='C', scope='test', config=cfg) for n in ('classical', 'candidate')}
    result = aggregate_comparison([dict(example={'name':'a', 'case':{'n':7}},
        runs={'classical':[run(True,2)], 'candidate':[run(True,1)]})], arms)
    assert result['table'][1]['speedup_vs_classical'] == 2
    assert result['table'][1]['speedup_vs_incumbent'] is None
    assert result['summary']['candidate']['final_time_improvement_certified'] is False


@pytest.mark.parametrize('change', [
    {'final_true_residual': 1.}, {'executed_cycles': 100000},
    {'final_true_residual': None}, {'wall_seconds': 0.},
])
def test_invalid_residual_budget_or_time_never_yields_speedup(change):
    cfg = AdaptiveConfig()
    arms = {n: dict(checkpoint_id=n, branch='C', scope='test', config=cfg) for n in ('classical', 'candidate')}
    candidate = dict(run(True, 1), **change)
    result = aggregate_comparison([dict(example={'name':'a', 'case':{'n':7}},
        runs={'classical':[run(True,2)], 'candidate':[candidate]})], arms)
    assert result['table'][1]['speedup_vs_classical'] is None
    assert result['table'][1]['successful_time_to_tolerance_seconds'] is None


def test_original_tree_cannot_be_output(tmp_path):
    research = tmp_path / 'research'; research.mkdir()
    for out in (research, research / 'child', tmp_path):
        with pytest.raises(ValueError): safe_output(out, research)


def test_exact_audit_reconstruction_is_diagnostic_and_mismatch_rejected(tmp_path):
    from adaptive_mg.v67.data import dataset
    settings = dict(seed=10, audit_sizes=[7], audit_per_size=2)
    examples = dataset([7], 2, 200012)
    manifest = {'splits': {'audit': [e.manifest() for e in examples]}}
    (tmp_path / 'split_manifest.json').write_text(json.dumps(manifest))
    _, provenance = diagnostic_cases(tmp_path, settings, 'audit')
    assert provenance['scope'] == 'diagnostic re-evaluation of previously inspected audit'
    assert not provenance['independent_certification_evidence']
    manifest['splits']['audit'][0]['operator_digest'] = 'wrong'
    (tmp_path / 'split_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): diagnostic_cases(tmp_path, settings, 'audit')


def test_branches_preserve_baseline_and_replacement_budget():
    cfg = AdaptiveConfig()
    for branch in ('C', 'H_S', 'H_P', 'H_SP'):
        c = forced_config(cfg, branch)
        assert c.mg == cfg.mg
        assert (c.replace_pre, c.replace_post) == (cfg.replace_pre, cfg.replace_post)
        assert c.use_smoother == (branch in ('H_S', 'H_SP'))
        assert c.use_transfer == (branch in ('H_P', 'H_SP'))
