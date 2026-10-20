"""Real CLI path, immutable selection and one-use final evaluation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from adaptive_mg.v67 import em_line_study as em
from adaptive_mg.v67 import joint_transfer_study as joint
from adaptive_mg.v67 import light_smoother_study as study
from adaptive_mg.v67.limited import digest_file

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp('light_cli')
    # A tiny *test fixture* with real final gates, never a performance run.
    # Keep all numerical components and lineages real; reduce only the held-out
    # dataset plan so the final ledger can be exercised in a unit-test budget.
    original_plan = joint.make_research_plan
    def tiny_plan(**kwargs):
        plan = original_plan(**kwargs)
        for name in study.HOLDOUT:
            family = plan['splits'][name]['families'][0]
            plan['splits'][name].update(sizes=[7], families=[family], per_family=1, count=1)
        return plan
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(joint, 'make_research_plan', tiny_plan)
        spec = read(ROOT/'configs/v6_7_em_line_smoke.json')
        spec['smoke'] = False
        for stage in spec['stages'].values():
            stage['updates'] = 1
        source_config = root/'source.json'
        source_config.write_text(json.dumps(spec))
        source = root/'source'
        em.prepare(source_config, source)
        em.train(source)
        light = read(ROOT/'configs/v6_7_light_smoother_smoke.json')
        light['smoke'] = False
        light['search'].update(grid=[1.], repeats=1, rhs_count=1, confirmation_rhs_count=1)
        light['fit']['updates'] = 2
        config = root/'config.json'
        config.write_text(json.dumps(light))
        out = root/'run'
        study.main(['prepare', '--config', str(config), '--em-run-dir', str(source), '--run-dir', str(out)])
        study.main(['search', '--run-dir', str(out)])
        search_hash = digest_file(out/'search/summary.json')
        study.main(['search', '--run-dir', str(out), '--resume'])
        assert digest_file(out/'search/summary.json') == search_hash
        study.main(['fit', '--run-dir', str(out)])
        fits = {key: digest_file(out/f'fit/{key}.json') for key in ('mlp', 'regression')}
        study.main(['fit', '--run-dir', str(out), '--resume'])
        assert fits == {key: digest_file(out/f'fit/{key}.json') for key in fits}
    return out


def test_real_eight_arm_cli_resume_and_sealed_final(pipeline):
    out = pipeline
    arguments = ['benchmark', '--run-dir', str(out), '--include-online',
                 '--repeats', '1', '--warmups', '0', '--rhs-counts', '1',
                 '--regimes', 'cold', 'warm', 'multiple']
    result = study.main(arguments)
    assert set(result['manifest']['arms']) == {'fixed_C', 'em5_C', *study.BASE_ARMS,
                                             *study.ONLINE_ARMS, 'frozen_P_line'}
    assert len(result['rows']) == 7
    assert read(out/'benchmarks/validation/progress.json')['completed_measurements'] == 168
    previous = digest_file(out/'benchmarks/validation/raw_results.json')
    repeated = study.main(arguments+['--resume'])
    assert repeated['summary'] == result['summary']
    assert digest_file(out/'benchmarks/validation/raw_results.json') == previous
    for row in result['rows']:
        for name in study.BASE_ARMS:
            assert row['runs'][name]['warm']['1'][0]['online_adaptation_calls'] == 0
        assert row['runs']['learned_P_line']['cold']['1'][0]['feature_seconds'] == 0
    with pytest.raises(ValueError, match='Validation'):
        study.fit(out, resume=True)
    with pytest.raises(ValueError):
        study.main([arg for arg in arguments if arg != '--include-online']+['--resume'])
    study.main(['freeze', '--run-dir', str(out), '--repeats', '1', '--warmups', '0',
                '--rhs-counts', '1', '--regimes', 'multiple', '--splits', 'final'])
    selected = deepcopy(read(out/'selected/deployment.json'))
    original_evaluate = study.evaluate_light
    def interrupted(*args, **kwargs):
        original_evaluate(*args, **kwargs)
        raise KeyboardInterrupt('after measurements, before final completion ledger')
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(study, 'evaluate_light', interrupted)
        with pytest.raises(KeyboardInterrupt):
            study.main(['final', '--run-dir', str(out)])
    assert read(out/'data/final_claim.json')['status'] == 'claimed'
    study.main(['final', '--run-dir', str(out), '--resume'])
    assert read(out/'data/final_claim.json')['status'] == 'completed'
    assert read(out/'selected/deployment.json') == selected
    assert read(out/'final/final/progress.json')['completed_measurements'] == 8
    with pytest.raises(ValueError, match='already completed'):
        study.main(['final', '--run-dir', str(out), '--resume'])
    with pytest.raises(ValueError, match='frozen'):
        study.search(out, resume=True)
