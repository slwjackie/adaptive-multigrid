from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from adaptive_mg.v67 import bamg_energy_cnn_benchmark as study
from adaptive_mg.v67.bamg_transfer import WeightCNN


def settings():
    value = json.loads((Path(__file__).resolve().parents[1] /
                        'configs/v6_7_bamg_compact_research_q1.json').read_text())
    value['benchmark'].update(repetitions=2, warmup_cycles=0)
    return value


def examples():
    return [dict(case_id=f'validation_{i}', family='near_isotropic', shape=(3, 3),
                 A=sp.eye(9, format='csr') * (i + 1), b=np.ones(9),
                 normalized_operator_digest=f'operator_{i}', split='validation')
            for i in range(2)]


class Bank:
    def __init__(self, case, method, calls, *, fail=False):
        self.case, self.method, self.calls, self.fail = case, method, calls, fail

    def cycle(self, x, b):
        return np.zeros_like(b)

    def solve(self, b, *, x0, timeout_seconds):
        assert b.shape == (9,)
        assert np.array_equal(x0, np.zeros_like(b))
        self.calls.append((self.case, self.method))
        seconds = {'fixed_C': .010, 'em5_C': .008, 'old_weight': .006}[self.method]
        return dict(converged=not self.fail, warm_seconds=seconds,
                    reason='max_cycles' if self.fail else 'converged', cycles=4,
                    tolerance=1e-8, residuals=[1., 1. if self.fail else 1e-10],
                    fallback_used=False, x=np.arange(100_000),
                    cycle_seconds=[.001] * 100_000)


def checkpoint(tmp_path):
    path = tmp_path / 'weights.pt'
    path.write_bytes(b'fake checkpoint; the runtime builder is monkeypatched')
    return {'old_weight': path}


@pytest.mark.parametrize('failed_seconds', [.006, 125.])
def test_failure_penalty_and_compact_scalar_storage(tmp_path, monkeypatch, failed_seconds):
    calls = []
    class FailedTimingBank(Bank):
        def solve(self, *args, **kwargs):
            result = super().solve(*args, **kwargs)
            if self.fail:
                result['warm_seconds'] = failed_seconds
            return result
    def build(example, method, path, cfg):
        return FailedTimingBank(example['case_id'], method, calls,
                                fail=method == 'old_weight' and example['case_id'] == 'validation_0')
    monkeypatch.setattr(study, '_build_bank', build)
    out = tmp_path / 'run'
    result = study.benchmark(out, examples=examples(), settings=settings(),
                             checkpoints=checkpoint(tmp_path), identity={'sources': 'v1'})
    assert result['completed'] and result['rhs_count'] == 1
    assert len(calls) == 12
    old = result['methods']['old_weight']
    assert old['attempts'] == 4 and old['failures'] == 2 and old['missing'] == 0
    failed_cost = max(120., failed_seconds)
    assert old['warm_seconds'] == pytest.approx((failed_cost + .006) / 2)
    assert old['cycles'] is None and old['speedup'] is None and not old['eligible']
    assert result['selected_method'] == 'em5_C'
    assert result['methods']['em5_C']['speedup'] == pytest.approx(1.25)
    assert result['case_results'][0]['methods']['old_weight']['warm_seconds'] == failed_cost
    for path in out.rglob('*.json'):
        text = path.read_text()
        assert len(text) < 15_000
        for forbidden in ('"residuals"', '"x"', '"A"', '"P"', '"cycle_seconds"'):
            assert forbidden not in text


def test_partial_validation_never_selects_and_resume_skips_saved_solves(tmp_path, monkeypatch):
    calls, completed_calls = [], []
    interrupted = False
    class InterruptingBank(Bank):
        def solve(self, *args, **kwargs):
            nonlocal interrupted
            if len(completed_calls) == 2 and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            result = super().solve(*args, **kwargs)
            completed_calls.append((self.case, self.method))
            return result
    monkeypatch.setattr(study, '_build_bank',
                        lambda ex, m, p, s: InterruptingBank(ex['case_id'], m, calls))
    out, paths = tmp_path / 'run', checkpoint(tmp_path)
    kwargs = dict(examples=examples(), settings=settings(), checkpoints=paths,
                  identity={'sources': {'module.py': 'sha256'}})
    with pytest.raises(KeyboardInterrupt):
        study.benchmark(out, **kwargs)
    partial = study.summarize(out)
    assert not partial['completed'] and partial['selected_method'] is None
    assert not any(row['eligible'] for row in partial['methods'].values())
    assert sum(row['missing'] for row in partial['methods'].values()) == 10
    assert sum(row['failures'] for row in partial['methods'].values()) == 10
    result = study.benchmark(out, **kwargs, resume=True)
    assert result['completed']
    assert Counter(completed_calls) == Counter({(ex['case_id'], m): 2
        for ex in examples() for m in ('fixed_C', 'em5_C', 'old_weight')})
    # Remove one scalar observation. Exactly this method/case is rebuilt.
    path = study._case_path(out, 'validation_0')
    saved = study.read_json(path)
    saved['measurements']['old_weight'].pop('1')
    study.write_json(path, saved)
    rebuilt = []
    def build_missing(ex, method, path, cfg):
        assert ex['case_id'] == 'validation_0' and method == 'old_weight'
        rebuilt.append((ex['case_id'], method))
        return Bank(ex['case_id'], method, calls)
    monkeypatch.setattr(study, '_build_bank', build_missing)
    study.benchmark(out, **kwargs, resume=True)
    assert rebuilt == [('validation_0', 'old_weight')]
    before = {p.name: p.read_bytes() for p in (out / 'benchmark_cases').glob('*.json')}
    monkeypatch.setattr(study, '_build_bank', lambda *a: pytest.fail('Finished runtime rebuilt'))
    study.benchmark(out, **kwargs, resume=True)
    assert before == {p.name: p.read_bytes() for p in (out / 'benchmark_cases').glob('*.json')}


def test_resume_rejects_source_checkpoint_or_rhs_change(tmp_path, monkeypatch):
    monkeypatch.setattr(study, '_build_bank', lambda ex, m, p, s: Bank(ex['case_id'], m, []))
    out, paths = tmp_path / 'run', checkpoint(tmp_path)
    kwargs = dict(examples=examples(), settings=settings(), checkpoints=paths,
                  identity={'sources': 'original'})
    study.benchmark(out, **kwargs)
    with pytest.raises(ValueError, match='identity changed'):
        study.benchmark(out, **{**kwargs, 'identity': {'sources': 'modified'}}, resume=True)
    changed = examples()
    changed[0]['b'][0] = 2.
    with pytest.raises(ValueError, match='identity changed'):
        study.benchmark(out, **{**kwargs, 'examples': changed}, resume=True)
    paths['old_weight'].write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='identity changed'):
        study.benchmark(out, **kwargs, resume=True)


def test_setup_failure_is_penalized_without_retry_or_fallback(tmp_path, monkeypatch):
    builds = []
    def build(ex, method, path, cfg):
        builds.append((ex['case_id'], method))
        if method == 'old_weight':
            raise FileNotFoundError('Checkpoint missing')
        return Bank(ex['case_id'], method, [])
    monkeypatch.setattr(study, '_build_bank', build)
    out = tmp_path / 'run'
    paths = {'old_weight': tmp_path / 'absent.pt'}
    result = study.benchmark(out, examples=examples(), settings=settings(),
                             checkpoints=paths, identity={})
    assert result['methods']['old_weight']['failures'] == 4
    assert result['methods']['old_weight']['warm_seconds'] == 120.
    assert len(builds) == 6
    state = study.read_json(study._case_path(out, 'validation_0'))
    assert state['measurements']['old_weight']['0']['reason'].startswith('setup_failed:')
    assert not any(row['fallback_used'] for row in state['measurements']['old_weight'].values())


@pytest.mark.parametrize('change,reason', [
    ({'warm_seconds': 121.}, 'timeout'),
    ({'warm_seconds': float('nan')}, 'invalid_warm_seconds'),
    ({'residuals': [1., float('nan')]}, 'nonfinite_or_invalid_residual_tolerance'),
    ({'residuals': [1., .1]}, 'convergence_claim_failed_residual_check'),
    ({'fallback_used': True}, 'forbidden_fallback'),
])
def test_measurement_rejects_invalid_success(change, reason):
    class InvalidBank(Bank):
        def solve(self, *args, **kwargs):
            return {**super().solve(*args, **kwargs), **change}
    value = study._measurement(InvalidBank('case', 'fixed_C', []), np.ones(9),
                               settings()['benchmark'])
    assert not value['success'] and value['reason'] == reason
    json.dumps(value, allow_nan=False)


def test_real_baseline_and_cnn_runtime_share_rhs_and_actual_galerkin(tmp_path):
    torch.set_num_threads(1)
    cfg = settings()
    cfg['models']['weight'].update(hidden=2, depth=1)
    model = WeightCNN(**cfg['models']['weight']).double()
    path = tmp_path / 'weight.pt'
    torch.save(dict(weight_spec=model.spec, weight_state=model.state_dict()), path)
    n = 7
    line = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format='csr')
    a = (sp.kron(sp.eye(n), line) + sp.kron(line, sp.eye(n))).tocsr()
    example = dict(A=a, shape=(n, n), b=np.ones(n * n), case_id='validation_smoke',
                   family='near_isotropic', normalized_operator_digest='smoke')
    for method in ('fixed_C', 'em5_C', 'old_weight'):
        bank = study._build_bank(example, method, path if method == 'old_weight' else None, cfg)
        np.testing.assert_allclose(bank.root.coarse.a.toarray(),
                                   (bank.root.p.T @ a @ bank.root.p).toarray(), atol=1e-12)
        row = study._measurement(bank, example['b'], cfg['benchmark'])
        assert row['fallback_used'] is False
        assert row['final_residual'] is not None
        if method in study.BASELINES:
            assert row['success']
