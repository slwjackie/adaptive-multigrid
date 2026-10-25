"""Q1 figures use bilinear fields and retain the discretization provenance."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from adaptive_mg.v67 import research_data as data

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'q1_pde_figures', ROOT/'scripts/run_v6_7_pde_figures.py')
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def q1_config(tmp_path):
    settings = json.loads((ROOT/'configs/v6_7_light_smoother_smoke.json').read_text())
    settings['discretization'] = 'q1'
    path = tmp_path/'q1.json'
    path.write_text(json.dumps(settings))
    return path


def test_q1_bilinear_gradient_and_flux_at_cell_centers():
    plan = data.make_research_plan(smoke=True, discretization='q1')
    record = data._specification('validation', plan['splits']['validation'],
                                 7, 'rotated_anisotropy', 0, 0)
    example = data._example(record)
    x, y = np.meshgrid(np.arange(1, 8)/8, np.arange(1, 8)/8, indexing='ij')
    fields = figures.element_fields(example.case, example.scale, (x*y).ravel())
    assert fields['discretization'] == 'q1'
    assert 'triangles' not in fields
    points, cells = fields['points'], fields['cells']
    boundary = np.any((points == 0) | (points == 1), axis=1)
    np.testing.assert_array_equal(fields['u_full'][boundary], 0)
    inside = ~boundary[cells].any(axis=1)
    centers = fields['centroids'][inside]
    # grad(x*y)=(y,x), which varies within each Q1 cell. The center sample
    # must reproduce this rather than either of the two P1 gradients.
    np.testing.assert_allclose(fields['gradient'][inside], centers[:, ::-1], atol=1e-14)
    expected = -np.einsum('eij,ej->ei', fields['tensors'][inside], centers[:, ::-1])
    np.testing.assert_allclose(fields['flux'][inside], expected, atol=1e-13)


def test_q1_constant_load_integrates_four_corner_shape_functions():
    n = 7
    points, cells = figures.quadrilateral_mesh(n)
    full_load = np.zeros(len(points))
    # Each bilinear corner shape integrates to area/4; four cells meet
    # at every interior node, including the first layer next to boundary.
    np.add.at(full_load, cells.ravel(), 1/(4*(n+1)**2))
    inside = np.all((points > 0) & (points < 1), axis=1)
    np.testing.assert_allclose(figures.constant_load(n), full_load[inside], rtol=1e-15)


def test_q1_solution_plot_samples_xy_without_a_diagonal():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import QuadMesh
    points, cells = figures.quadrilateral_mesh(3)
    fields = dict(points=points, cells=cells, discretization='q1')
    fig, ax = plt.subplots()
    try:
        art = figures._draw(ax, fields, points[:, 0]*points[:, 1], nodal=True)
        assert isinstance(art, QuadMesh)
        values = np.asarray(art.get_array()).reshape(16, 16)
        centers = (np.arange(16)+.5)/16
        np.testing.assert_allclose(values, centers[:, None]*centers[None, :], atol=1e-15)
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_q1_figure_pipeline_preserves_operator_and_reports_sampling(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    config = q1_config(tmp_path)
    records, settings, _ = figures.load_cases(None, config, 'validation', 7, 0)
    assert len(records) == 7
    assert all(r['case']['discretization'] == 'q1' for r in records)
    for record in records:
        assert figures.restore_example(record).case.discretization == 'q1'
    monkeypatch.setattr(figures, 'overview', lambda *args, **kwargs: None)
    monkeypatch.setattr(figures, 'detail', lambda *args, **kwargs: None)
    output = tmp_path/'figures'
    rows = figures.run(SimpleNamespace(output=output, run_dir=None, config=config,
        split='validation', n=7, sample_index=0, rhs='both', with_mg=False, dpi=50))
    assert len(rows) == 14
    assert all(r['direct_success'] for r in rows)
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['protocol']['discretization'] == 'q1'
    assert '2x2 Gauss' in manifest['discretization']
    assert 'cell centers' in manifest['field_sampling']
    assert 'not cell-constant' in manifest['flux']
    assert 'Q1' in (output/'CAPTIONS.md').read_text()
    for family in data.FAMILIES:
        fields = np.load(output/'unit_source'/f'{family}_fields.npz')
        assert str(fields['discretization']) == 'q1'
        assert fields['cells'].shape == (64, 4)
        np.testing.assert_array_equal(fields['rhs'], figures.constant_load(7))
