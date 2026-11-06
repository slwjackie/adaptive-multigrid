from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_mg.v67 import bamg_energy_study as study
from adaptive_mg.v67.bamg_energy import array_digest
from adaptive_mg.v67.bamg_solver import runtime_from_prolongations
from adaptive_mg.v67.bamg_teacher import build_teacher
from adaptive_mg.v67.bamg_transfer import direct_pattern, fixed_bilinear_mask


def settings():
    value = json.loads((study.PROJECT / 'configs/v6_7_bamg_energy_teacher_q1.json').read_text())
    value.update(smoke=True, lambdas=[0., .1])
    value['data'].update(sizes=[7], per_family=1, families=['near_isotropic', 'rotated_anisotropy'])
    value['teacher'].update(vectors=4, sweeps=[1, 2], selection_vectors=2, audit_vectors=2,
                            eigen_count=2, eigen_refine=1, adaptive_vectors=1,
                            augment_vectors=0, max_vector_retries=0)
    value['benchmark'].update(repetitions=2, rhs_count=2)
    return value


def test_common_vectors_support_and_actual_galerkin_operators():
    s = settings()
    cfg, tc = study.validate(s)
    record = study.specifications(s)[1]
    example = study._example(record)
    reference = build_teacher(example.a, (7, 7), config=tc, mg_config=cfg)
    vectors = [l.vectors.copy() for l in reference.levels[:-1]]
    ps = [l.P.copy() for l in reference.levels[:-1]]
    expected_z = [array_digest(z) for z in vectors]
    outputs = {}
    for method, strength in study.method_table(s).items():
        matrices, diagnostics = study.candidate_hierarchy(example.a, (7, 7), method,
            strength, vectors, ps, tc, cfg, s['optimizer'])
        outputs[method] = matrices
        assert diagnostics[0]['support_digest'] == array_digest(fixed_bilinear_mask(direct_pattern(7)).numpy())
        if method == 'em5_C':
            assert diagnostics[0]['native_support_matches_common']
        if method not in ('fixed_C', 'em5_C'):
            assert diagnostics[0]['bank_vectors_digest'] == expected_z[0]
            assert diagnostics[0]['normalizers_fixed']
        runtime = runtime_from_prolongations(example.a, (7, 7), matrices, cfg)
        np.testing.assert_allclose(runtime.root.coarse.a.toarray(),
            (matrices[0].T @ example.a @ matrices[0]).toarray(), atol=2e-12)
    np.testing.assert_allclose(outputs['bootstrap_C'][0].toarray(),
                               outputs['hybrid_lambda_0'][0].toarray(), atol=1e-9)
    assert expected_z == [array_digest(z) for z in vectors]


def test_resume_uses_cached_bank_and_never_selects_from_partial_validation(tmp_path, monkeypatch):
    s = settings()
    config = tmp_path / 'config.json'
    study.write_json(config, s)
    out = tmp_path / 'comparison'
    first = study.run(config, out, max_cases=1)
    assert not first['completed'] and first['selected_lambda'] is None
    assert not any(m['eligible'] for m in first['methods'].values())
    result = study.run(config, out, resume=True)
    assert result['completed']
    assert result['cases_expected'] == 2
    assert result['selected_lambda'] in (0., .1)
    assert all(m['attempts'] == 8 for m in result['methods'].values())
    for v in result['groups'].values():
        assert v['selected_lambda'] is None
    # Simulate interruption after one measured RHS. Bank must not be regenerated.
    case_path = next((out / 'cases').glob('*/results.json'))
    case = study.read_json(case_path)
    case['completed'] = False
    case['measurements']['hybrid_lambda_0.1'].pop()
    study.write_json(case_path, case)
    monkeypatch.setattr(study, 'build_teacher', lambda *a, **k: pytest.fail('cached bank regenerated'))
    resumed = study.run(config, out, resume=True)
    assert resumed['completed']
    assert len(study.read_json(case_path)['measurements']['hybrid_lambda_0.1']) == 4
    # A complete repeat must leave measurement files unchanged.
    before = study.file_digest(case_path)
    study.run(config, out, resume=True)
    assert study.file_digest(case_path) == before
    altered = deepcopy(s)
    altered['lambdas'].append(1.)
    study.write_json(config, altered)
    with pytest.raises(ValueError, match='changed'):
        study.run(config, out, resume=True)


def test_multilevel_candidates_share_vectors_but_rebuild_different_coarse_A():
    s = settings()
    s['data']['sizes'] = [15]
    cfg, tc = study.validate(s)
    example = study._example(study.specifications(s)[1])
    reference = build_teacher(example.a, (15, 15), config=tc, mg_config=cfg)
    vectors = [l.vectors for l in reference.levels[:-1]]
    ps = [l.P for l in reference.levels[:-1]]
    outputs = []
    for method, strength in [('bootstrap_C', None), ('hybrid_lambda_1', 1.)]:
        matrices, report = study.candidate_hierarchy(example.a, (15, 15), method,
            strength, vectors, ps, tc, cfg, s['optimizer'])
        actual = example.a
        for p, info, z in zip(matrices, report, vectors):
            assert info['A_digest'] == study.matrix_digest(actual)
            assert info['bank_vectors_digest'] == array_digest(z)
            actual = (p.T @ actual @ p).tocsr()
            actual.sum_duplicates(); actual.sort_indices()
        outputs.append(report)
    assert outputs[0][0]['A_digest'] == outputs[1][0]['A_digest']
    assert outputs[0][1]['A_digest'] != outputs[1][1]['A_digest']
    assert outputs[0][1]['support_digest'] == outputs[1][1]['support_digest']


def test_failed_fast_candidate_cannot_win_and_failures_are_not_dropped():
    s = settings()
    s['benchmark'].update(repetitions=1, rhs_count=1, failure_penalty_seconds=100.)
    methods = study.method_table(s)
    case = dict(completed=True, setups={}, measurements={})
    for method in methods:
        case['measurements'][method] = [dict(repeat=0, rhs=0, success=True, cycles=3, warm_seconds=2.)]
    case['measurements']['hybrid_lambda_0.1'][0].update(success=False, warm_seconds=.001)
    result = study.summarize([case], s, 1)
    failed = result['methods']['hybrid_lambda_0.1']
    assert failed['failures'] == 1 and not failed['eligible']
    assert failed['mean_case_median_warm_seconds'] == 100.
    assert result['selected_lambda'] == 0.
    assert failed['speedup_vs_fixed'] is None
    # Even if failures are allowed, reliability is ranked before speed.
    s['benchmark']['allowed_failure_rate'] = 1.
    s['benchmark']['failure_penalty_seconds'] = .01
    assert study.summarize([case], s, 1)['selected_lambda'] == 0.
    incomplete = study.summarize([case], s, 2)
    assert incomplete['selected_lambda'] is None
    assert incomplete['methods']['fixed_C']['failures'] == 1


def test_bank_corruption_is_reported_not_silently_regenerated(tmp_path):
    s = settings()
    cfg, tc = study.validate(s)
    e = study._example(study.specifications(s)[0])
    study.vector_bank(e, tmp_path, tc, cfg)
    with (tmp_path / 'vectors.npz').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='Corrupt'):
        study.vector_bank(e, tmp_path, tc, cfg)
