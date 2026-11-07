from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg.provenance import operator_digest
from adaptive_mg.v67 import bamg_energy_cnn_holdout_benchmark as study
from adaptive_mg.v67.bamg_transfer import WeightCNN


def settings():
    root = Path(__file__).resolve().parents[1]
    values = json.loads((root / 'configs/v6_7_bamg_compact_research_q1.json').read_text())
    values['holdout_benchmark'] = dict(repetitions=2, warmup_cycles=0,
        timeout_seconds=120., failure_penalty_seconds=120.,
        rhs_kinds=['manufactured', 'random', 'localized'],
        bootstrap_resamples=100, bootstrap_seed=7)
    return values


def cases(count=2):
    return [dict(case_id=f'final_near_isotropic_n3_{i}', n=3, family='near_isotropic',
                 split='final', index=i) for i in range(count)]


def example(case):
    n, index = case['n'], case['index']
    diagonal = np.ones(n * n)
    diagonal[0] = index + 2
    a = sp.diags(diagonal, format='csr')
    rhs = dict(manufactured=np.ones(n*n), random=np.linspace(1., 2., n*n),
               localized=np.linspace(.2, .5, n*n))
    return dict(**case, A=a, shape=(n, n), b=rhs['manufactured'], rhs=rhs,
                normalized_operator_digest=operator_digest(a, scale_invariant=True))


def checkpoints(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    path.write_bytes(b'mocked_weight_checkpoint')
    return {'old_cnn': path}


def identity():
    return dict(freeze_sha256='a' * 64, claim_fingerprint='frozen-run')


class Bank:
    def __init__(self, ex, method, calls, *, fail=False, seconds=None):
        self.ex, self.method, self.calls = ex, method, calls
        self.fail, self.seconds = fail, seconds

    def cycle(self, x, b):
        return np.zeros_like(b)

    def solve(self, b, *, x0, timeout_seconds):
        assert np.array_equal(x0, np.zeros_like(b))
        kind = next(name for name, value in self.ex['rhs'].items() if np.array_equal(b, value))
        self.calls.append((self.ex['case_id'], self.method, kind))
        scale = 1 + self.ex['index']
        seconds = self.seconds if self.seconds is not None else scale * {'fixed_C': .01, 'em5_C': .008, 'old_cnn': .005}[self.method]
        return dict(converged=not self.fail, warm_seconds=seconds, cycles=4,
            reason='max_cycles' if self.fail else 'converged', tolerance=1e-8,
            residuals=[1., 1. if self.fail else 1e-10], fallback_used=False,
            x=np.ones(50_000), cycle_seconds=[.1] * 50_000)


def test_grouped_paired_operator_intervals_and_compact_results(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(study, '_build_bank', lambda ex, method, p, s: Bank(ex, method, calls))
    out = tmp_path / 'run'
    result = study.benchmark(out, cases=cases(), load_example=example, settings=settings(),
                             checkpoints=checkpoints(tmp_path), identity=identity())
    assert result['completed'] and result['selected_method'] is None and not result['heldout_selection']
    assert len(calls) == 2 * 3 * 2 * 3
    old = result['methods']['old_cnn']
    assert old['attempts'] == 12 and old['independent_operators'] == 2
    assert old['warm_seconds'] == pytest.approx(.0075)
    assert old['speedup_vs_fixed'] == pytest.approx(2.)
    assert old['paired_ci95']['speedup_vs_fixed'] == pytest.approx([2., 2.])
    assert old['paired_ci95']['independent_operators'] == 2
    assert old['operator_failure_wilson_ci95'][1] > .5  # zero/2 is weak reliability evidence
    assert result['groups']['final/n3/near_isotropic']['cases_expected'] == 2
    assert 'eligible' not in old
    for path in (out / 'benchmark_cases').glob('*.json'):
        text = path.read_text()
        assert len(text) < 25_000
        for forbidden in ('"residuals"', '"x"', '"A"', '"P"', '"cycle_seconds"'):
            assert forbidden not in text
    study.print_summary(result, 'final/n3/near_isotropic')


@pytest.mark.parametrize('seconds', [.001, 125.])
def test_failure_penalty_distinct_operators_and_no_speedup_claim(tmp_path, monkeypatch, seconds):
    calls = []
    def build(ex, method, p, s):
        fail = ex['index'] == 0 and method == 'old_cnn'
        return Bank(ex, method, calls, fail=fail, seconds=seconds if fail else None)
    monkeypatch.setattr(study, '_build_bank', build)
    result = study.benchmark(tmp_path / 'run', cases=cases(), load_example=example, settings=settings(),
                             checkpoints=checkpoints(tmp_path), identity=identity())
    old = result['methods']['old_cnn']
    assert old['failures'] == 6 and old['distinct_failed_operators'] == 1
    assert old['operator_failure_rate'] == .5 and old['missing'] == 0
    assert old['warm_seconds'] == pytest.approx((max(120., seconds) + .01) / 2)
    assert old['speedup_vs_fixed'] is None and old['paired_ci95'] is None
    assert old['penalized_time_ratio_vs_fixed'] is not None
    assert result['selected_method'] is None


def test_resume_skips_completed_cases_and_saved_rhs_measurements(tmp_path, monkeypatch):
    calls, loaded = [], []
    interrupted = False
    class InterruptedBank(Bank):
        def solve(self, *args, **kwargs):
            nonlocal interrupted
            if len(calls) == 20 and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return super().solve(*args, **kwargs)
    monkeypatch.setattr(study, '_build_bank', lambda ex, m, p, s: InterruptedBank(ex, m, calls))
    def load(case):
        loaded.append(case['case_id'])
        return example(case)
    out, paths = tmp_path / 'run', checkpoints(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        study.benchmark(out, cases=cases(), load_example=load, settings=settings(),
                        checkpoints=paths, identity=identity())
    partial = study.summarize(out)
    assert not partial['completed'] and partial['cases_completed'] == 1
    assert partial['selected_method'] is None
    before_first = study.file_digest(study._case_path(out, cases()[0]['case_id']))
    loaded.clear()
    result = study.benchmark(out, cases=cases(), load_example=load, settings=settings(),
                             checkpoints=paths, identity=identity(), resume=True)
    assert result['completed']
    assert loaded == [cases()[1]['case_id']]
    assert before_first == study.file_digest(study._case_path(out, cases()[0]['case_id']))
    assert len(calls) == 36
    assert all(v == 2 for v in Counter(calls).values())
    study.benchmark(out, cases=cases(), load_example=lambda c: pytest.fail('complete A rebuilt'),
                     settings=settings(), checkpoints=paths, identity=identity(), resume=True)
    paths['old_cnn'].write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        study.benchmark(out, cases=cases(), load_example=example, settings=settings(),
                         checkpoints=paths, identity=identity(), resume=True)


def test_max_cases_keeps_missing_penalties_and_rhs_identity_is_bound(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(study, '_build_bank', lambda ex, m, p, s: Bank(ex, m, calls))
    out, paths = tmp_path / 'run', checkpoints(tmp_path)
    result = study.benchmark(out, cases=cases(), load_example=example, settings=settings(),
                             checkpoints=paths, identity=identity(), max_cases=1)
    assert not result['completed'] and result['cases_completed'] == 1
    assert result['methods']['fixed_C']['missing'] == 6
    assert result['methods']['fixed_C']['warm_seconds'] == pytest.approx((.01 + 120) / 2)
    path = study._case_path(out, cases()[0]['case_id'])
    state = study.read_json(path)
    state['measurements']['fixed_C'].pop('0:random')
    study.write_json(path, state)
    def altered(case):
        ex = example(case)
        ex['rhs']['random'] = ex['rhs']['random'] * 2
        return ex
    with pytest.raises(ValueError, match='RHS changed'):
        study.benchmark(out, cases=cases(), load_example=altered, settings=settings(),
                         checkpoints=paths, identity=identity(), resume=True)


def test_freeze_and_split_contracts_and_single_case_no_ci(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(study, '_build_bank', lambda ex, m, p, s: Bank(ex, m, calls))
    paths = checkpoints(tmp_path)
    with pytest.raises(ValueError, match='freeze'):
        study.benchmark(tmp_path / 'bad', cases=cases(), load_example=example, settings=settings(),
                         checkpoints=paths, identity={})
    bad = cases()
    bad[0]['split'] = 'validation'
    with pytest.raises(ValueError, match='final/OOD'):
        study.benchmark(tmp_path / 'bad', cases=bad, load_example=example, settings=settings(),
                         checkpoints=paths, identity=identity())
    result = study.benchmark(tmp_path / 'single', cases=cases(1), load_example=example,
                             settings=settings(), checkpoints=paths, identity=identity())
    assert result['methods']['old_cnn']['paired_ci95'] is None


def test_actual_small_runtime_all_rhs(tmp_path):
    s = settings()
    s['holdout_benchmark'].update(repetitions=1, warmup_cycles=1)
    case = dict(case_id='final_rotated_n7_0', n=7, family='rotated_anisotropy', split='final', index=0)
    n = 7
    tridiagonal = sp.diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1], format='csr')
    a = sp.kronsum(tridiagonal, tridiagonal).tocsr()
    ex = example(case)
    ex['A'] = a
    ex['normalized_operator_digest'] = operator_digest(a, scale_invariant=True)
    torch.manual_seed(71)
    weight = WeightCNN(**s['models']['weight']).double()
    path = tmp_path / 'real.pt'
    torch.save(dict(weight_spec=weight.spec, weight_state=weight.state_dict()), path)
    result = study.benchmark(tmp_path / 'real', cases=[case], load_example=lambda c: ex,
                             settings=s, checkpoints={'old_cnn': path}, identity=identity())
    assert result['completed'] and result['methods']['fixed_C']['failures'] == 0
    assert result['methods']['em5_C']['failures'] == 0
    assert all(v['attempts'] == 3 for v in result['methods'].values())


def test_cohorts_average_all_seeds_before_paired_operator_bootstrap(tmp_path, monkeypatch):
    s = settings()
    s['model_seeds'] = [6101, 6102, 6103]
    path = tmp_path / 'cohort.pt'
    path.write_bytes(b'mocked-model')
    paths = {f'{arm}_seed{seed}_{stage}': path for arm in ('scratch', 'warm')
             for seed in s['model_seeds'] for stage in ('supervised', 'finetuned')}
    monkeypatch.setattr(study, '_build_bank', lambda ex, method, p, settings: (ex, method))
    times = {'scratch': [.012, .020, .040], 'warm': [.012, .005, .019]}
    def measurement(bank, rhs, bench):
        ex, method = bank
        if method in ('fixed_C', 'em5_C'):
            seconds = .01
        else:
            arm, seed, stage = method.split('_')
            seconds = times[arm][s['model_seeds'].index(int(seed.removeprefix('seed')))]
        return dict(success=True, warm_seconds=seconds * (ex['index'] + 1), cycles=4,
                    reason='converged', tolerance=1e-8, final_residual=1e-10, fallback_used=False)
    monkeypatch.setattr(study, '_measurement', measurement)
    result = study.benchmark(tmp_path / 'cohorts', cases=cases(), load_example=example,
                             settings=s, checkpoints=paths, identity=identity())
    scratch = result['cohorts']['scratch_finetuned']
    warm = result['cohorts']['warm_finetuned']
    assert scratch['warm_seconds'] == pytest.approx(.024 * 1.5)
    assert warm['warm_seconds'] == pytest.approx(.012 * 1.5)
    assert scratch['seed_count'] == 3 and scratch['independent_operators'] == 2
    assert scratch['attempts'] == 2 * 3 * 2 * 3
    pair = result['cohort_comparisons']['finetuned']
    assert pair['scratch_over_warm_time_ratio'] == pytest.approx(2.)
    assert pair['paired_ci95']['time_ratio_reference_over_candidate'] == pytest.approx([2., 2.])
    assert pair['paired_ci95']['independent_operators'] == 2  # neither 6 seeds nor 36 RHS/repeats
    assert not pair['paired_ci95']['includes_training_seed_uncertainty']
    assert result['groups']['final/n3/near_isotropic']['cohort_comparisons']['finetuned'] == pair
    manifest = study.read_json(tmp_path / 'cohorts/benchmark_manifest.json')
    assert len(manifest['model_groups']['scratch_finetuned']) == 3
    partial_paths = {key: value for key, value in paths.items() if key != 'scratch_seed6103_finetuned'}
    with pytest.raises(ValueError, match='omits a prespecified seed'):
        study.benchmark(tmp_path / 'incomplete', cases=cases(), load_example=example,
                         settings=s, checkpoints=partial_paths, identity=identity())
