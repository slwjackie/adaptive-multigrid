"""Figures preserve the existing FE operator, coordinates and data provenance."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from adaptive_mg.v67.config import AdaptiveConfig
from adaptive_mg.v67.fixed_classical import BilinearClassicalBaseline
from adaptive_mg.v67 import research_data as data

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'pde_figures', ROOT/'scripts/run_v6_7_pde_figures.py')
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def record_for(family, n=7):
    plan = data.make_research_plan(seed=413907)
    record = data._specification('validation', plan['splits']['validation'],
                                 n, family, 0, 0)
    example = data._example(record)
    record.update(example.manifest(),
        rhs_digest=hashlib.sha256(example.b.tobytes()).hexdigest(),
        exact_digest=hashlib.sha256(example.exact.tobytes()).hexdigest())
    return record, example


@pytest.mark.parametrize('family', data.FAMILIES)
@pytest.mark.parametrize('n', [7, 31, 63])
def test_visualized_tensor_and_p1_geometry_reassemble_actual_operator(family, n):
    record, example = record_for(family, n)
    points, triangles, gradients = figures.triangle_mesh(example.n)
    vertices = points[triangles]
    # Gradients reproduce coordinate derivatives and a constant exactly.
    np.testing.assert_allclose(gradients.sum(axis=1), 0., atol=1e-13)
    np.testing.assert_allclose(np.einsum('tik,tij->tkj', vertices, gradients),
                               np.broadcast_to(np.eye(2), (len(triangles), 2, 2)),
                               atol=1e-13)
    edges = vertices[:, 1:]-vertices[:, :1]
    areas = np.abs(np.linalg.det(edges))/2
    tensors = figures.tensor_at(example.case, example.scale,
                                vertices.mean(axis=1))
    assert np.linalg.eigvalsh(tensors).min() > 0
    local = areas[:, None, None]*np.einsum(
        'tik,tkl,tjl->tij', gradients, tensors, gradients)
    full_indices = np.rint(points*(example.n+1)).astype(int)
    gi, gj = full_indices[triangles, 0], full_indices[triangles, 1]
    interior = ((gi >= 1) & (gi <= example.n) &
                (gj >= 1) & (gj <= example.n))
    ids = (gi-1)*example.n+gj-1
    keep = interior[:, :, None] & interior[:, None, :]
    rows = np.broadcast_to(ids[:, :, None], local.shape)[keep]
    cols = np.broadcast_to(ids[:, None, :], local.shape)[keep]
    reconstructed = sp.coo_matrix((local[keep], (rows, cols)),
                                  shape=example.a.shape).tocsr()
    error = reconstructed-example.a
    assert np.max(np.abs(error.data), initial=0) <= (
        5e-12+5e-13*np.max(np.abs(example.a.data)))


def test_nodal_axes_zero_boundary_and_element_flux_are_consistent():
    _, example = record_for('rotated_anisotropy')
    n = example.n
    coord = np.arange(1, n+1)/(n+1)
    x, y = np.meshgrid(coord, coord, indexing='ij')
    # Asymmetric field catches silent x/y transposition.
    u = (x+3*y).ravel()
    fields = figures.element_fields(example.case, example.scale, u)
    points, triangles = fields['points'], fields['triangles']
    full = np.asarray(fields['u_full']).reshape(-1)
    boundary = np.any((points == 0) | (points == 1), axis=1)
    np.testing.assert_array_equal(full[boundary], 0.)
    np.testing.assert_allclose(full[~boundary],
                               points[~boundary, 0]+3*points[~boundary, 1])
    internal_triangles = ~boundary[triangles].any(axis=1)
    np.testing.assert_allclose(fields['gradient'][internal_triangles],
                               np.tile([1., 3.], (internal_triangles.sum(), 1)),
                               atol=1e-13)
    expected = -np.einsum('tij,tj->ti', fields['tensors'], fields['gradient'])
    np.testing.assert_allclose(fields['flux'], expected, atol=1e-13)


def test_constant_source_load_is_integral_of_p1_basis():
    n = 7
    points, triangles, _ = figures.triangle_mesh(n)
    vertices = points[triangles]
    areas = np.abs(np.linalg.det(vertices[:, 1:]-vertices[:, :1]))/2
    full_load = np.zeros(len(points))
    np.add.at(full_load, triangles.ravel(), np.repeat(areas/3, 3))
    interior = np.all((points > 0) & (points < 1), axis=1)
    expected = full_load[interior]
    actual = figures.constant_load(n)
    assert actual.shape == (n*n,)
    np.testing.assert_allclose(actual, expected, rtol=2e-15)
    np.testing.assert_allclose(actual, 1/(n+1)**2, rtol=2e-15)


@pytest.mark.parametrize('key', [
    'operator_digest', 'normalized_operator_digest', 'rhs_digest', 'exact_digest'])
def test_modified_operator_or_manufactured_target_digest_is_rejected(key):
    record, original = record_for('checkerboard')
    restored = figures.restore_example(record)
    np.testing.assert_array_equal(restored.b, original.b)
    corrupt = deepcopy(record)
    corrupt[key] = '0'*64
    with pytest.raises(ValueError):
        figures.restore_example(corrupt)


@pytest.mark.parametrize('split', data.HOLDOUT)
def test_figure_case_loader_does_not_materialize_holdout(split, monkeypatch):
    monkeypatch.setattr(data, '_assemble_case',
                        lambda _: pytest.fail('holdout was assembled'))
    with pytest.raises(ValueError):
        figures.load_cases(None, ROOT/'configs/v6_7_light_smoother_research.json',
                           split, 63, 0)


def tree_hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def test_run_import_uses_exact_record_and_never_changes_source(tmp_path):
    run = tmp_path/'existing_run'
    run.mkdir()
    settings = json.loads((ROOT/'configs/v6_7_light_smoother_smoke.json').read_text())
    cfg = AdaptiveConfig.from_dict(settings['solver'])
    baseline = BilinearClassicalBaseline.from_config(cfg)
    (run/'configuration.json').write_text(json.dumps(settings))
    (run/'fixed_classical.json').write_text(json.dumps(baseline.to_dict()))
    plan = data.make_research_plan(smoke=True, seed=settings['seed']+7100003)
    for split in data.DEVELOPMENT:
        plan['splits'][split].update(sizes=[7], per_family=1, count=7)
    history = tmp_path/'history'
    history.mkdir()
    _, _, manifest = data.prepare_development_data(
        run/'data', baseline, plan=plan, historical_roots=[history])
    # Existing evaluation/freeze records are unrelated and must survive intact.
    (run/'unrelated_evaluation.json').write_text('{"completed": 288}')
    before = tree_hashes(run)
    records, loaded_settings, origin = figures.load_cases(run, None, 'validation', 7, 0)
    assert loaded_settings == settings
    assert len(records) == 7
    expected = {r['family']: r for r in manifest['splits']['validation']}
    for record in records:
        assert record == expected[record['family']]
        figures.restore_example(record)
    assert tree_hashes(run) == before
    assert origin


def test_output_validation_preserves_old_artifacts_and_source_run(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    old = source/'old_result.json'
    old.write_text('{"finished": true}')
    before = tree_hashes(source)
    for destination in (source, source/'new_figure'):
        with pytest.raises((ValueError, FileExistsError)):
            figures.validate_output(destination, source)
    destination = tmp_path/'new_figure'
    assert figures.validate_output(destination, source) == destination.resolve()
    assert not destination.exists()
    destination.mkdir()
    (destination/'old.png').write_bytes(b'preserve me')
    with pytest.raises(FileExistsError):
        figures.validate_output(destination, source)
    assert tree_hashes(source) == before


def test_undeclared_grid_is_rejected_before_operator_assembly(monkeypatch):
    monkeypatch.setattr(figures, '_example',
                        lambda _: pytest.fail('invalid size was assembled'))
    with pytest.raises(ValueError):
        figures.load_cases(None, ROOT/'configs/v6_7_light_smoother_smoke.json',
                           'validation', 63, 0)


def test_isotropic_material_has_no_preferred_axis_drawn():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _, example = record_for('local_patch')
    fields = figures.element_fields(example.case, example.scale, example.exact)
    fields['anisotropy'] = np.ones(len(fields['triangles']))
    fig, ax = plt.subplots()
    try:
        figures._directions(ax, fields, fields['principal_direction'], axes=True)
        assert len(ax.collections[-1].get_segments()) == 0
        fields['anisotropy'][:] = 2
        figures._directions(ax, fields, fields['principal_direction'], axes=True)
        assert len(ax.collections[-1].get_segments()) > 0
    finally:
        plt.close(fig)


def test_both_rhs_pipeline_retains_mg_failure_and_correct_numeric_fields(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    from adaptive_mg.v67 import solver
    original = solver.PreparedAdaptiveMG
    constructions = []
    def one_setup_failure(*args, **kwargs):
        constructions.append(1)
        if len(constructions) == 1:
            raise ValueError('test-only fixed_C setup failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(solver, 'PreparedAdaptiveMG', one_setup_failure)
    # Root workflow separately renders/visually checks the real n=63 figures.
    # This test focuses on run bookkeeping and actual numerical data.
    monkeypatch.setattr(figures, 'overview', lambda *args, **kwargs: None)
    monkeypatch.setattr(figures, 'detail', lambda *args, **kwargs: None)
    output = tmp_path/'figures'
    config = ROOT/'configs/v6_7_light_smoother_smoke.json'
    original_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    args = SimpleNamespace(output=output, run_dir=None, config=config,
        split='validation', n=7, sample_index=0, rhs='both', with_mg=True, dpi=50)
    rows = figures.run(args)
    assert len(rows) == 14
    assert all(row['direct_success'] for row in rows)
    failed = [row for row in rows if not row['fixed_C_success']]
    assert len(failed) >= 2
    assert {row['rhs_mode'] for row in failed if row['family'] == data.FAMILIES[0]} == {
        'study', 'unit_source'}
    for mode in ('study', 'unit_source'):
        retained = json.loads((output/mode/(data.FAMILIES[0]+'_mg.json')).read_text())
        assert retained['stage'] == 'setup' and not retained['converged']
        for family in data.FAMILIES:
            a = sp.load_npz(output/mode/(family+'_A.npz'))
            fields = np.load(output/mode/(family+'_fields.npz'))
            relres = np.linalg.norm(a@fields['u']-fields['rhs'])/np.linalg.norm(fields['rhs'])
            assert relres < 1e-8
            if mode == 'unit_source':
                np.testing.assert_array_equal(fields['rhs'], figures.constant_load(7))
                assert fields['study_target'].size == 0
            else:
                np.testing.assert_allclose(fields['u'], fields['study_target'], atol=1e-11)
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['diagnostic_only'] and not manifest['model_selection']
    assert not manifest['final_ood_accessed'] and not manifest['source_run_mutated']
    assert manifest['origin']['kind'] == 'configuration_generated_illustration'
    assert hashlib.sha256(config.read_bytes()).hexdigest() == original_hash
    with pytest.raises(FileExistsError):
        figures.run(args)
