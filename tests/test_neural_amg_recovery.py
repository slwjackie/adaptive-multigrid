"""Historical v2 repairs cannot silently resume the changed v3 objective."""
from pathlib import Path
import runpy

import pytest
from neural_amg import study, teacher

ROOT = Path(__file__).resolve().parents[1]
RECOVERIES = runpy.run_path(str(ROOT / 'scripts/recover_neural_amg_bootstrap.py'))


@pytest.mark.parametrize('name', ['recover', 'recover_teacher_l1'])
def test_historical_repairs_require_original_version(tmp_path, name):
    source = tmp_path / 'old'
    source.mkdir()
    output = tmp_path / 'new'
    with pytest.raises(ValueError, match='start_neural_amg_cancellation'):
        RECOVERIES[name](source, output)
    assert not output.exists()
    assert not list(source.iterdir())


def test_historical_source_pins_remain_bound_to_migration_source():
    original = study.read_json(ROOT / 'scripts/cancellation_start_sources.json')['source']
    for name in ('bootstrap_depth_recovery_sources.json', 'teacher_l1_recovery_sources.json'):
        pins = study.read_json(ROOT / 'scripts' / name)
        assert pins['after'] == original
    assert original != study._source_identity()


def test_bootstrap_depth_propagates_to_labels_and_runtime(tmp_path, monkeypatch):
    config = study.load_config(ROOT / 'configs/neural_amg_smoke.json')
    config['hierarchy']['max_levels'] = 17
    config['teacher']['maxiter'] = 2
    data = study.prepare(config, tmp_path / 'run')
    real = teacher.fit_teacher
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs['bootstrap_max_levels'])
        return real(*args, **kwargs)
    monkeypatch.setattr(study, 'fit_teacher', capture)
    monkeypatch.setattr(teacher, 'fit_teacher', capture)
    study._labels(config, data['train'][0], tmp_path / 'run')
    assert seen and set(seen) == {17}
    seen.clear()
    case = study._load_case(tmp_path / 'run', data['train'][0])
    study._hierarchy(config, case.a, method='teacher')
    assert seen and set(seen) == {17}
