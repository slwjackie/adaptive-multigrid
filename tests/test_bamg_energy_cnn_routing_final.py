from copy import deepcopy
from pathlib import Path

import pytest

from adaptive_mg.v67 import bamg_energy_cnn_routing as routing
from adaptive_mg.v67 import bamg_energy_cnn_routing_final as final
from adaptive_mg.v67 import bamg_energy_cnn_routing_study as study
from adaptive_mg.v67.bamg_study import _file_hash, _read, _write
from test_bamg_energy_cnn_study import inputs


def ready(tmp_path):
    source, teacher, config = inputs(tmp_path)
    settings = _read(config)
    settings.update(version=study.VERSION, lambdas=[10., 30., 50.], sampling='epoch_shuffle',
        numerical_baselines=['fixed_C', 'em5_C', 'bootstrap_C'],
        routing=dict(unseen_grid_policy='lower_anchor_clamp', stage='finetuned'))
    settings['supervised']['updates'] = 1
    settings['finetune'].update(updates=1, curriculum_step=1)
    settings['holdout']['splits'] = {
        name: dict(families=['local_patch' if name in ('ood_contrast', 'ood_topology')
                            else 'rotated_anisotropy'], per_family=1)
        for name in ('final', 'ood_grid', 'ood_rotation', 'ood_anisotropy', 'ood_contrast', 'ood_topology')}
    _write(config, settings)
    out = tmp_path / 'routing'
    study.run(config, out, source_study=source, teacher_comparison=teacher)
    return out, source, teacher, config


def test_routing_freeze_full_holdout_and_exact_resume(tmp_path, monkeypatch):
    out, source, teacher, config = ready(tmp_path)
    original = {str(p): _file_hash(p) for p in source.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='freeze'):
        final.evaluate_final(out)
    assert not (out / 'final').exists()
    frozen = final.freeze(out)
    assert frozen['selection'] == _read(out / 'routing_selection.json')
    assert frozen['holdout_selection_permitted'] is False
    assert frozen['all_prespecified_seeds_retained'] is True
    assert not frozen['final_operators_materialized'] and not (out / 'final').exists()
    assert {'a' * 64, 'b' * 64} <= set(frozen['forbidden_operator_digests'])
    assert final.freeze(out) == frozen
    with pytest.raises(ValueError, match='frozen'):
        study.run(config, out, source_study=source, teacher_comparison=teacher, resume=True)
    first = final.evaluate_final(out, max_cases=1)
    assert not first['completed'] and first['benchmark']['cases_completed'] == 1
    shard = next((out / 'final/benchmark/benchmark_cases').glob('*.json'))
    first_hash = _file_hash(shard)
    result = final.evaluate_final(out, resume=True)
    assert result['completed'] and result['benchmark']['cases_expected'] == 6
    assert _file_hash(shard) == first_hash
    assert result['selection'] == frozen['selection']
    assert result['holdout_selection_permitted'] is False
    assert result['benchmark']['selected_method'] is None
    assert result['evaluation_scope'] == 'smoke_only'
    manifest = _read(out / 'final/benchmark/benchmark_manifest.json')
    assert {'fixed_C', 'em5_C', 'bootstrap_C'} <= set(manifest['methods'])
    for path in (out / 'final/benchmark/benchmark_cases').glob('*.json'):
        state = _read(path)
        for arm, policy in frozen['selection']['arms'].items():
            for seed in manifest['model_seeds']:
                name = routing.routed_name(arm, seed)
                for setup in state['setups'][name].values():
                    choice = setup['routing']
                    assert choice['selected_lambda'] == routing.route_lambda(policy, state['case']['n'])
                    assert Path(choice['selected_checkpoint']).name == 'deployment.pt'
    data = _read(out / 'final/data/data_manifest.json')
    assert data['completed'] and len(data['cases']) == 6
    assert set(frozen['forbidden_operator_digests']).isdisjoint(
        item['normalized_operator_digest'] for item in data['cases'].values())
    monkeypatch.setattr(final.heldout_benchmark, 'benchmark',
                        lambda *args, **kwargs: pytest.fail('completed final was retimed'))
    assert final.evaluate_final(out, resume=True) == result
    with pytest.raises(FileExistsError, match='claimed'):
        final.evaluate_final(out)
    final.report_final(out, group='ood_grid')
    assert original == {str(p): _file_hash(p) for p in source.rglob('*') if p.is_file()}
    with shard.open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='changed'):
        final.report_final(out)


def test_routing_freeze_rejects_changed_selection_and_provenance(tmp_path):
    out, *_ = ready(tmp_path)
    bank_path = out / 'teacher_bank_manifest.json'
    bank = _read(bank_path)
    changed_bank = deepcopy(bank)
    changed_bank['policy'] = 'changed vector-bank policy'
    _write(bank_path, changed_bank)
    with pytest.raises(ValueError, match='provenance'):
        final.freeze(out)
    _write(bank_path, bank)
    case_path = next((out / 'validation/benchmark_cases').glob('*.json'))
    case = _read(case_path)
    changed_case = deepcopy(case)
    changed_case['measurements']['fixed_C']['0']['warm_seconds'] *= 2
    _write(case_path, changed_case)
    with pytest.raises(ValueError, match='unchanged fixed validation'):
        final.freeze(out)
    _write(case_path, case)
    route_path = out / 'routing_selection.json'
    original = _read(route_path)
    changed = deepcopy(original)
    policy = changed['arms']['scratch']
    first = str(policy['validation_sizes'][0])
    policy['grid_to_lambda'][first] = 50. if policy['grid_to_lambda'][first] != 50. else 10.
    _write(route_path, changed)
    with pytest.raises(ValueError, match='recomputed'):
        final.freeze(out)
    _write(route_path, original)
    manifest_path = out / 'validation_routed/benchmark_manifest.json'
    manifest = _read(manifest_path)
    changed = deepcopy(manifest)
    changed['identity']['routing_selection_sha256'] = 'f' * 64
    _write(manifest_path, changed)
    with pytest.raises(ValueError):
        final.freeze(out)
    _write(manifest_path, manifest)
    frozen = final.freeze(out)
    # Every candidate is frozen, including a lambda not routed for this grid.
    chosen = frozen['selection']['arms']['scratch']['grid_to_lambda']['7']
    alternative = next(value for value in (10., 30., 50.) if value != chosen)
    checkpoint = out / 'runs' / routing.run_name(alternative, 'scratch', 6101) / 'deployment.pt'
    with checkpoint.open('ab') as stream:
        stream.write(b'changed-unused-routing-alternative')
    with pytest.raises(ValueError, match='artifact changed'):
        final.evaluate_final(out)


def test_routing_freeze_rejects_missing_seed_and_holdout_cache_loss(tmp_path):
    out, *_ = ready(tmp_path)
    path = out / 'comparison.json'
    original = _read(path)
    changed = deepcopy(original)
    changed['runs'].pop(next(iter(changed['runs'])))
    _write(path, changed)
    with pytest.raises(ValueError, match='lambda/seed/arm'):
        final.freeze(out)
    _write(path, original)
    final.freeze(out)
    final.evaluate_final(out, max_cases=1)
    path = out / 'final/data/data_manifest.json'
    data = _read(path)
    key = next(iter(data['cases']))
    (out / 'final/data/cases' / (key + '.json')).unlink()
    del data['cases'][key]
    _write(path, data)
    with pytest.raises(ValueError, match='saved data case'):
        final.evaluate_final(out, resume=True)


def test_frozen_grid_policy_cannot_change_after_final_claim(tmp_path):
    out, *_ = ready(tmp_path)
    final.freeze(out)
    final.evaluate_final(out, max_cases=1)
    path = out / 'routing_selection.json'
    value = _read(path)
    value['arms']['warm']['unseen_grid_policy'] = 'choose_fastest_on_test'
    _write(path, value)
    with pytest.raises(ValueError, match='artifact changed'):
        final.evaluate_final(out, resume=True)
    with pytest.raises(ValueError, match='artifact changed'):
        final.report_final(out)
