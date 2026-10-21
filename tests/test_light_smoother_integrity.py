"""Cross-run import, train-only exclusions, and deployment lineage gates."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from adaptive_mg.v67 import em_line_study as em
from adaptive_mg.v67 import light_smoother_study as study
from adaptive_mg.v67 import research_data as data
from adaptive_mg.v67.limited import digest_file

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture(scope='module')
def source_em(tmp_path_factory):
    root = tmp_path_factory.mktemp('light_integrity_em')
    settings = read(ROOT/'configs/v6_7_em_line_smoke.json')
    for stage in settings['stages'].values():
        stage['updates'] = 1
    write(root/'settings.json', settings)
    out = root/'em'
    em.prepare(root/'settings.json', out)
    em.train(out)
    return out


def light_settings(seed=2026102171):
    settings = read(ROOT/'configs/v6_7_light_smoother_smoke.json')
    settings['seed'] = seed
    settings['search'].update(grid=[1.], repeats=1, rhs_count=1, confirmation_rhs_count=1)
    settings['fit']['updates'] = 1
    return settings


def prepared(root, source, *, settings=None, fit=True):
    spec = light_settings() if settings is None else settings
    write(root/'config.json', spec)
    out = root/'run'
    study.prepare(root/'config.json', out, em_run_dir=source)
    if fit:
        study.search(out)
        study.fit(out)
    return out


def test_completed_import_is_snapshot_with_matched_frozen_p(source_em, tmp_path):
    before = digest_file(source_em/'checkpoints/em_P/candidate.pt')
    out = prepared(tmp_path, source_em)
    _, settings, cfg, baseline = study.load(out)
    imported = read(out/'import_manifest.json')
    assert imported['frozen_line']['included']
    assert not imported['joint_line_included']
    arms = study._arms(out, settings, cfg, baseline)
    signatures = {arm['model'].expert_signature('transfer') for arm in arms.values() if arm['model']}
    assert signatures == {imported['transfer_signature']}
    assert 'joint_line' not in arms
    assert digest_file(source_em/'checkpoints/em_P/candidate.pt') == before
    assert digest_file(out/'imported/em_line/checkpoints/em_P/candidate.pt') == before
    # A later move/loss of the source does not change the complete local import.
    source_manifest = imported['files']['em_line_manifest.json']
    assert source_manifest['source_path'] != str(out/source_manifest['path'])
    study.prepare(tmp_path/'config.json', out, em_run_dir=source_em, resume=True)


def test_intact_fit_from_another_run_cannot_be_attached(source_em, tmp_path):
    left = prepared(tmp_path/'left', source_em, settings=light_settings(2026102172))
    right = prepared(tmp_path/'right', source_em, settings=light_settings(2026102173))
    shutil.rmtree(left/'fit')
    shutil.copytree(right/'fit', left/'fit')
    out, settings, cfg, baseline = study.load(left)
    with pytest.raises(ValueError, match='fit manifest/status'):
        study._arms(out, settings, cfg, baseline)


def test_source_final_and_ancestral_operator_ids_are_excluded(source_em, tmp_path):
    source = tmp_path/'historical_source'
    shutil.copytree(source_em, source)
    settings = light_settings(2026102174)
    plan = data.make_research_plan(smoke=True, seed=settings['seed']+7100003)
    train = plan['splits']['train']
    train.update(sizes=[7], per_family=1, count=len(data.FAMILIES))
    # These are exactly the first candidates the fresh light generator would
    # otherwise use. One was in an ancestor and one in this source's final set.
    records = [data._specification('train', train, 7, family, 0, 0)
               for family in ('near_isotropic', 'channel')]
    ancestor, heldout = [data._example(record).group_digest for record in records]
    development_path = source/'data/development_manifest.json'
    development = read(development_path)
    history = development['historical_index']
    history['normalized_operator_digests'] = sorted(set(history['normalized_operator_digests']) | {ancestor})
    history['count'] = len(history['normalized_operator_digests'])
    write(development_path, development)
    write(source/'data/final_data_manifest.json', {
        'splits': {'final': [{'normalized_operator_digest': heldout}]}})
    out = prepared(tmp_path/'light', source, settings=settings, fit=False)
    excluded = {r['normalized_operator_digest'] for r in read(out/'imported/exclusion_manifest.json')['operators']}
    assert {ancestor, heldout} <= excluded
    current = read(out/'data/development_manifest.json')
    observed = {r['normalized_operator_digest'] for rows in current['splits'].values() for r in rows}
    assert not observed & {ancestor, heldout}
    assert current['rejected_normalized_duplicates']['train'] >= 2
    training = current['splits']['train']
    assert all(r['retry'] >= 1 for r in training if r['family'] in ('near_isotropic', 'channel'))


def test_deployment_paths_and_online_controls_are_frozen(source_em, tmp_path):
    out = prepared(tmp_path, source_em, settings=light_settings(2026102175))
    study.benchmark(out, repeats=1, warmups=0, rhs_counts=[1], regimes=['multiple'])
    path = out/'selected/deployment.json'
    original = read(path)
    for field, value in (('mlp', '/wrong/checkpoint.json'),
                         ('online_settings', {'enabled': True, 'time_budget_seconds': 999}),
                         ('include_online', not original['include_online'])):
        changed = deepcopy(original)
        changed[field] = value
        write(path, changed)
        with pytest.raises(ValueError, match='Deployment differs'):
            study.freeze(out, repeats=1, warmups=0, rhs_counts=[1],
                         regimes=['multiple'], splits=['final'])
    write(path, original)
    # A valid export reaches the intended prohibition on using smoke for final.
    with pytest.raises(ValueError, match='Smoke'):
        study.freeze(out, repeats=1, warmups=0, rhs_counts=[1],
                     regimes=['multiple'], splits=['final'])


def test_failed_search_case_is_retained_but_cannot_become_fit_label(tmp_path):
    example = SimpleNamespace(name='failed_train_operator', group_digest='a'*64)
    result_path = tmp_path/'search/failed_train_operator/result.json'
    record = dict(status='complete', fallback_allowed=False,
        label_eligible=False, selected_candidate_success=False,
        selection_outcome='no_successful_candidate',
        features=[[0.]*12], selected_coefficients=[[1., 1.]],
        confirmation=dict(selected_index=0, summaries=[dict(eligible=False)]),
        trials=[dict(success=False, stop_reason='max_cycles')])
    write(result_path, record)
    write(tmp_path/'search/summary.json', dict(status='complete', rows=[dict(
        case=example.name, normalized_operator_digest=example.group_digest,
        path=str(result_path.relative_to(tmp_path)), sha256=digest_file(result_path))]))
    before = result_path.read_bytes()
    with pytest.raises(ValueError, match='No successful search label'):
        study._search_data(tmp_path, [example])
    assert result_path.read_bytes() == before
