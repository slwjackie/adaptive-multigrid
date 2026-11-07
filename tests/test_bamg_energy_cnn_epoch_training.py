"""Every epoch covers TRAIN; random probes and resume positions cannot alter order."""
import numpy as np
import pytest

from adaptive_mg.v67.bamg_energy_cnn_epoch_training import EpochShuffle


@pytest.mark.parametrize('count', [1, 7, 280, 560, 1120])
def test_every_epoch_is_a_permutation_with_at_most_one_visit_difference(count):
    sampler = EpochShuffle(count, 6101, 'supervised')
    visits = np.zeros(count, dtype=int)
    for update in range(3 * count + count // 2):
        index, report = sampler.select(update)
        visits[index] += 1
        assert report['epoch'] == update // count + 1
        assert report['epoch_position'] == update % count + 1
        assert visits.max() - visits.min() <= 1
        if (update + 1) % count == 0:
            assert len(set(visits)) == 1
    result = sampler.coverage(int(visits.sum()))
    assert result['minimum_visits'] == int(visits.min())
    assert result['maximum_visits'] == int(visits.max())
    assert result['all_cases_seen'] is True


def test_saved_update_reconstructs_order_without_touching_probe_rng():
    uninterrupted = EpochShuffle(29, 6102, 'finetune')
    expected = [uninterrupted.select(i) for i in range(100)]
    rng = np.random.default_rng(77)
    untouched = np.random.default_rng(77)
    for start in [0, 1, 28, 29, 30, 58, 71, 99]:
        restored = EpochShuffle(29, 6102, 'finetune')
        assert [restored.select(i) for i in range(start, 100)] == expected[start:]
    assert np.array_equal(rng.standard_normal(100), untouched.standard_normal(100))
    supervised = EpochShuffle(29, 6102, 'supervised')
    other_seed = EpochShuffle(29, 6103, 'finetune')
    assert [supervised.select(i)[0] for i in range(29)] != [p[0] for p in expected[:29]]
    assert [other_seed.select(i)[0] for i in range(29)] != [p[0] for p in expected[:29]]


@pytest.mark.parametrize('args', [(0, 1, 'supervised'), (3, -1, 'supervised'),
                                  (3, 1, 'other'), (True, 1, 'supervised')])
def test_invalid_sampler_contract_rejected(args):
    with pytest.raises(ValueError):
        EpochShuffle(*args)
