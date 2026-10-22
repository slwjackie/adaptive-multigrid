#!/usr/bin/env python3
"""Reproducible P1 diffusion field figures; never alters a study or checkpoint.

The original study load is b=A*u_star, not point samples of a smooth f.
An optional f=1 problem is a separate illustration on the same operators.
This standalone script is deliberately outside src to preserve study hashes.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from adaptive_mg.config import MGConfig
from adaptive_mg.pde import DiffusionCase, diffusion_tensor
from adaptive_mg.provenance import hardware_environment, json_safe
from adaptive_mg.v67.research_data import (
    FAMILIES, _example, _hash, _specification, make_research_plan,
)

VERSION = 'p1-diffusion-figures-v1'
LABELS = dict(zip(FAMILIES, ('Near isotropic', 'Aligned anisotropy',
    'Rotated anisotropy', 'Local patch', 'Checkerboard', 'Channel',
    'Variable coefficient')))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector_hash(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2,
        ensure_ascii=False, allow_nan=False) + '\n')


def restore_example(record):
    if record.get('split') not in ('train', 'validation'):
        raise ValueError('Only train/validation illustrations are permitted')
    if record['case'].get('topology', 'standard') != 'standard':
        raise ValueError('Only the seven standard development families are supported')
    example = _example(record)
    expected = dict(operator_digest=example.digest,
        normalized_operator_digest=example.group_digest,
        rhs_digest=vector_hash(example.b), exact_digest=vector_hash(example.exact))
    for name, value in expected.items():
        if record.get(name) != value:
            raise ValueError(f'{record["name"]}: reconstructed {name} mismatch')
    return example


def load_cases(run_dir, config_path, split, n, sample_index):
    """Choose a predeclared index, without inspecting solve outcomes."""
    if split not in ('train', 'validation'):
        raise ValueError('Final/OOD data are never opened by this figure script')
    if n < 3 or n & (n + 1) or sample_index < 0:
        raise ValueError('Require n=2**L-1 >=3 and nonnegative sample index')
    if run_dir is not None:
        if config_path is not None:
            raise ValueError('Use --run-dir or --config, not both')
        source = Path(run_dir).resolve()
        config = source / 'configuration.json'
        manifest_path = source / 'data/development_manifest.json'
        plan_path = source / 'data/dataset_plan.json'
        settings = json.loads(config.read_text())
        manifest = json.loads(manifest_path.read_text())
        plan = json.loads(plan_path.read_text())
        if manifest['plan_digest'] != _hash(plan):
            raise ValueError('Development plan digest mismatch')
        spec = plan['splits'][split]
        if n not in spec['sizes'] or sample_index >= spec['per_family']:
            raise ValueError('Requested grid/index is outside the declared development split')
        origin = dict(kind='imported_development_records', source_run=str(source),
            files={str(p): file_hash(p) for p in (config, manifest_path, plan_path)},
            validation_claim='Actual recorded operators; deterministic index selection, no outcome filtering')
        records = []
        for family in FAMILIES:
            prefix = f'{split}_{family}_n{n}_{sample_index}_r'
            selected = [r for r in manifest['splits'][split]
                if r['name'].startswith(prefix) and r['family'] == family
                and r['split'] == split and r['case']['n'] == n]
            if len(selected) != 1:
                raise ValueError(f'Need exactly one recorded case matching {prefix}')
            restore_example(selected[0])
            records.append(selected[0])
    else:
        config = Path(config_path or ROOT / 'configs/v6_7_light_smoother_research.json').resolve()
        settings = json.loads(config.read_text())
        # Same development seed schedule as joint_transfer_study._development.
        plan = make_research_plan(smoke=settings['smoke'], seed=settings['seed'] + 7100003)
        spec = plan['splits'][split]
        spec.update(sizes=settings['training_sizes' if split == 'train' else 'validation_sizes'],
                    per_family=settings['training_per_family' if split == 'train' else 'validation_per_family'])
        if n not in spec['sizes'] or sample_index >= spec['per_family']:
            raise ValueError('Requested grid/index is outside the declared development split')
        origin = dict(kind='configuration_generated_illustration',
            files={str(config): file_hash(config)}, retry=0,
            validation_claim='Same generator and seed schedule; NOT a verified existing run. Historical duplicate retries are not reconstructed.')
        records = []
        for family in FAMILIES:
            record = _specification(split, spec, n, family, sample_index, 0)
            example = _example(record)
            record.update(example.manifest(), rhs_digest=vector_hash(example.b),
                          exact_digest=vector_hash(example.exact))
            records.append(record)
    if n not in spec['sizes'] or sample_index >= spec['per_family']:
        raise ValueError('Requested grid/index is outside the declared development split')
    return records, settings, origin


def triangle_mesh(n):
    """Full nodal coordinates, original triangles, physical P1 gradients."""
    h = 1. / (n + 1)
    x, y = np.meshgrid(np.arange(n + 2) * h, np.arange(n + 2) * h, indexing='ij')
    points = np.column_stack((x.ravel(), y.ravel()))
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    start = i.ravel() * (n + 2) + j.ravel()
    t0 = np.column_stack((start, start + n + 2, start + n + 3))
    t1 = np.column_stack((start, start + n + 3, start + 1))
    gradients = np.concatenate((
        np.broadcast_to(np.array([[-1., 0.], [1., -1.], [0., 1.]]) / h, (len(start), 3, 2)),
        np.broadcast_to(np.array([[0., -1.], [1., 0.], [-1., 1.]]) / h, (len(start), 3, 2))))
    return points, np.concatenate((t0, t1)), gradients


def tensor_at(case, scale, xy):
    """Exact standard-family coefficient definitions, sampled at supplied points."""
    if case.topology != 'standard' or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Require standard development topology and positive finite scale')
    xy = np.asarray(xy, dtype=float)
    if case.pattern == 'variable_coefficient':
        theta = np.deg2rad(case.angle_deg)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        base = rotation @ np.diag([1., case.epsilon]) @ rotation.T
        a = np.exp(case.amplitude * np.sin(2*np.pi*case.frequency*xy[:, 0]+case.phase)
                   * np.cos(2*np.pi*xy[:, 1]-case.phase))
        return scale * a[:, None, None] * base
    native = DiffusionCase(case.n, case.epsilon, case.angle_deg, case.contrast,
        case.pattern, tuple(case.patch_bounds), case.checker_cells, case.channel_width)
    return scale * np.asarray([diffusion_tensor(native, float(x), float(y)) for x, y in xy])


def constant_load(n):
    """Exact interior P1 load integral for f=1 on this triangulation: h**2."""
    return np.full(n*n, 1. / (n+1)**2)


def element_fields(case, scale, u):
    points, triangles, gradients = triangle_mesh(case.n)
    values = np.zeros((case.n+2, case.n+2))
    values[1:-1, 1:-1] = np.asarray(u).reshape(case.n, case.n)
    values = values.ravel()
    centroids = points[triangles].mean(axis=1)
    tensors = tensor_at(case, scale, centroids)
    gradient = np.einsum('ti,tij->tj', values[triangles], gradients)
    flux = -np.einsum('tij,tj->ti', tensors, gradient)
    eigvals, eigvecs = np.linalg.eigh(tensors)
    if not np.isfinite(tensors).all() or np.min(eigvals) <= 0:
        raise ValueError('Coefficient field is not finite SPD')
    return dict(points=points, triangles=triangles, centroids=centroids,
        tensors=tensors, gradient=gradient, flux=flux, u_full=values,
        lambda_max=eigvals[:, 1], anisotropy=eigvals[:, 1]/eigvals[:, 0],
        principal_direction=eigvecs[:, :, 1])


def validate_output(output, run_dir=None):
    target = Path(output).resolve()
    if run_dir is not None:
        source = Path(run_dir).resolve()
        if target == source or source in target.parents:
            raise ValueError('Figure output must be outside the source study directory')
    for protected in (ROOT / 'src', ROOT / 'scripts', ROOT / 'configs', ROOT / 'tests'):
        if target == protected or protected in target.parents:
            raise ValueError('Figure output cannot overwrite repository source directories')
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError('Use a new/empty figure output directory')
    return target


def _axes(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='$x$', ylabel='$y$', aspect='equal')
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.tick_params(labelsize=8)


def _draw(ax, fields, values, *, nodal=False, norm=None, cmap='viridis'):
    p, t = fields['points'], fields['triangles']
    kwargs = dict(cmap=cmap, norm=norm, rasterized=True)
    if nodal:
        artist = ax.tripcolor(p[:, 0], p[:, 1], t, values, shading='gouraud', **kwargs)
    else:
        artist = ax.tripcolor(p[:, 0], p[:, 1], t, facecolors=values, shading='flat', **kwargs)
    _axes(ax)
    return artist


def _directions(ax, fields, vectors, *, axes=False):
    n = int(np.sqrt(len(fields['points']))) - 2
    stride = max(1, (n+1)//8)
    ids = np.array([i*(n+1)+j for i in range(stride//2, n+1, stride)
                   for j in range(stride//2, n+1, stride)])
    xy, vec = fields['centroids'][ids], vectors[ids]
    norm = np.linalg.norm(vec, axis=1)
    valid = norm > np.max(norm, initial=0)*1e-12
    if axes:
        valid &= fields['anisotropy'][ids] > 1+1e-10
    xy, vec = xy[valid], vec[valid]/norm[valid, None]
    if axes:
        from matplotlib.collections import LineCollection
        # An eigenvector is an unoriented axis; do not imply a preferred sign.
        lines = np.stack((xy-.019*vec, xy+.019*vec), axis=1)
        ax.add_collection(LineCollection(lines, colors='white', linewidths=1.1))
    else:
        ax.quiver(xy[:, 0], xy[:, 1], vec[:, 0], vec[:, 1], color='white',
                  angles='xy', scale_units='xy', scale=27, width=.004)


def _save(fig, folder, stem, dpi):
    import matplotlib.pyplot as plt
    fig.savefig(folder / (stem+'.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(folder / (stem+'.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def overview(items, folder, stem, kind, rhs_label, origin, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    values = []
    for item in items:
        f = item['fields']
        values.append(f['u_full'] if kind == 'solution' else
            np.linalg.norm(f['flux'], axis=1) if kind == 'flux' else f[kind])
    positive = kind != 'solution'
    vmin = min(float(np.min(v[v > 0])) for v in values) if positive else min(float(np.min(v)) for v in values)
    vmax = max(float(np.max(v)) for v in values)
    if positive:
        norm = LogNorm(max(vmin, vmax*1e-7), max(vmax, vmin*(1+1e-6)))
    else:
        norm = Normalize(min(0., vmin), max(vmax, 1e-15))
    titles = dict(lambda_max='Diffusion strength', anisotropy='Anisotropy and principal axes',
                  solution='Numerical solution', flux='Diffusive flux')
    units = dict(lambda_max=r'$\lambda_{\max}(\widetilde K)$',
        anisotropy=r'$\lambda_{\max}/\lambda_{\min}$', solution=r'$u_h$', flux=r'$|\mathbf{q}_h|$')
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 7.7))
    fig.subplots_adjust(left=.055, right=.94, top=.83, bottom=.16, wspace=.35, hspace=.48)
    for index, (ax, item, value) in enumerate(zip(axes.ravel(), items, values)):
        fields, case = item['fields'], item['example'].case
        artist = _draw(ax, fields, value, nodal=kind == 'solution', norm=norm,
                       cmap='cividis' if kind == 'anisotropy' else 'viridis')
        if kind in ('lambda_max', 'anisotropy'):
            _directions(ax, fields, fields['principal_direction'], axes=True)
        elif kind == 'flux':
            _directions(ax, fields, fields['flux'])
        ax.set_title(f'({chr(97+index)}) {LABELS[case.family]}\n'
            rf'$\epsilon$={case.epsilon:.3g}, $\theta$={case.angle_deg:g}°, $s$={item["example"].scale:.3g}', fontsize=9)
    note_ax = axes.ravel()[-1]; note_ax.axis('off')
    label = 'Imported development cases' if origin['kind'].startswith('imported') else 'Configuration-generated illustration'
    extra = ('White segments: principal diffusion axes\nDirection has no preferred sign.'
        if kind in ('lambda_max', 'anisotropy') else 'White arrows: flux direction only\nArrow lengths are normalized.'
        if kind == 'flux' else 'Reference solution: sparse LU\nNo neural speed comparison.')
    note_ax.text(0, .95, f'{label}\n\nUnit square; $u=0$ on boundary\n'
        f'Interior grid: {items[0]["example"].n} × {items[0]["example"].n}\nP1 triangular finite elements\n\n{extra}',
        va='top', fontsize=10, linespacing=1.6, transform=note_ax.transAxes)
    fig.suptitle(f'Seven diffusion PDE families | {titles[kind]}', fontsize=17, y=.97)
    fig.text(.5, .905, rhs_label, ha='center', fontsize=11)
    cb = fig.colorbar(artist, cax=fig.add_axes([.22, .075, .55, .022]), orientation='horizontal')
    cb.set_label(units[kind]+'  (one shared scale across all families)', fontsize=10)
    _save(fig, folder, stem, dpi)


def detail(item, folder, mode, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    f = item['fields']; e = item['example']; case = e.case
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 8.6), layout='constrained')
    flux = np.linalg.norm(f['flux'], axis=1)
    panels = [(f['lambda_max'], False, 'Diffusion strength', True),
        (f['anisotropy'], False, 'Anisotropy ratio / principal axes', True),
        (f['u_full'], True, 'LU reference solution $u_h$', False)]
    for ax, (values, nodal, title, log) in zip(axs.ravel(), panels):
        low, high = float(np.min(values)), float(np.max(values))
        norm = LogNorm(low, max(high, low*(1+1e-6))) if log else Normalize(min(0., low), high)
        art = _draw(ax, f, values, nodal=nodal, norm=norm)
        fig.colorbar(art, ax=ax, shrink=.75); ax.set_title(title, fontsize=10)
    _directions(axs[0, 1], f, f['principal_direction'], axes=True)
    art = axs[1, 0].imshow(item['rhs'].reshape(e.n, e.n).T, origin='lower',
        extent=(.5/(e.n+1), 1-.5/(e.n+1), .5/(e.n+1), 1-.5/(e.n+1)), cmap='RdBu_r')
    _axes(axs[1, 0]); fig.colorbar(art, ax=axs[1, 0], shrink=.75)
    axs[1, 0].set_title('Assembled load $b_i$ (not pointwise $f$)', fontsize=10)
    art = _draw(axs[1, 1], f, flux)
    _directions(axs[1, 1], f, f['flux'])
    fig.colorbar(art, ax=axs[1, 1], shrink=.75)
    axs[1, 1].set_title(r'Diffusive flux $|\mathbf{q}_h|$; direction arrows', fontsize=10)
    last = axs[1, 2]
    if item['mg'] is not None and item['mg'].get('residual_history'):
        record = item['mg']; history = np.asarray(record['residual_history'])
        last.semilogy(np.arange(len(history)), np.maximum(history/history[0], 1e-300), label='fixed_C')
        last.axhline(record['stopping_threshold']/history[0], color='black', linestyle='--', label='Tolerance')
        last.set(xlabel='Executed V-cycles', ylabel=r'$\|b-Au_k\|_2/\|b\|_2$',
            title='fixed_C: '+('converged' if record['converged'] else 'FAILED: '+record['stop_reason']))
        last.grid(alpha=.2); last.legend(fontsize=8)
    else:
        last.axis('off')
        if item['mg'] is not None:
            last.set_title('fixed_C FAILED ('+item['mg'].get('stage', 'solve')+')', color='darkred', fontsize=10)
        last.text(0, .95, f'Direct residual / ||b||: {item["metrics"]["relative_residual"]:.3e}\n'
            f'Stopping threshold: {item["metrics"]["stopping_threshold"]:.3e}\n\n'
            'Fields use the LU reference.\nFlux is elementwise -K grad(u).\nThis is not fluid velocity.\n'
            'No continuum error is claimed.', va='top', fontsize=10, linespacing=1.8)
    fig.suptitle(f'{LABELS[case.family]} | {e.name}\n'
        + ('Original study load: $b=A u_*$' if mode == 'study' else 'Supplemental constant source: $f=1$'), fontsize=14)
    _save(fig, folder, case.family, dpi)


def run(args):
    output = validate_output(args.output, args.run_dir)
    records, settings, origin = load_cases(args.run_dir, args.config, args.split, args.n, args.sample_index)
    config = MGConfig(**settings['solver']['mg'])
    if args.with_mg and config.strategy_name != 'line_alt_bilinear_full__v22':
        raise ValueError('--with-mg requires the recorded fixed_C bilinear strategy')
    modes = ['study', 'unit_source'] if args.rhs == 'both' else [args.rhs]
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    from threadpoolctl import threadpool_limits
    from adaptive_mg.v67.solver import PreparedAdaptiveMG
    from adaptive_mg.v67.config import AdaptiveConfig
    from adaptive_mg.v67.limited import forced_config
    fixed_config = forced_config(AdaptiveConfig.from_dict(settings['solver']), 'C')
    source_files = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'src').rglob('*.cpp')) + [Path(__file__).resolve()]
    source_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in source_files}
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = dict(version=VERSION, origin=origin, source_commit=revision, source_sha256=source_hashes,
        protocol=dict(split=args.split, n=args.n, sample_index=args.sample_index, rhs_modes=modes,
            with_mg=args.with_mg, dpi=args.dpi, numerical_settings=asdict(config)),
        records=records, pde='-div(s*K*grad(u))=f on (0,1)^2; u=0 on boundary',
        discretization='P1 triangles, tensor at element centroids, eliminated boundary DOFs',
        field_solver='Full fine-grid scipy.sparse.linalg.splu reference; not a deployment benchmark',
        study_rhs='Original dataset RHS #0: b=A*u_star; u_star is a discrete manufactured target, not a continuum accuracy certificate',
        unit_source_rhs='Supplemental f=1: b_i=integral(phi_i)=h^2; not original benchmark forcing',
        flux='q_T=-s*K(centroid_T)*grad(u_h)|_T; broken P1 gradient, no interface smoothing, not velocity',
        diagnostic_only=True, model_selection=False, final_ood_accessed=False,
        source_run_mutated=False)
    write_json(output/'manifest.json', manifest)
    write_json(output/'progress.json', dict(status='running', completed=0, total=7*len(modes)))
    groups = {mode: [] for mode in modes}; metrics = []; completed = 0
    with threadpool_limits(limits=1):
        manifest['environment'] = hardware_environment(refresh=True)
        for record in records:
            example = restore_example(record)
            started = perf_counter(); factor = sla.splu(example.a.tocsc())
            factor_seconds = perf_counter()-started
            prepared, setup_error = None, None
            if args.with_mg:
                try:
                    prepared = PreparedAdaptiveMG(example.a, example.n, config=fixed_config)
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    setup_error = dict(converged=False, stage='setup',
                        error=f'{type(exc).__name__}: {exc}', residual_history=[])
            for mode in modes:
                folder = output/mode; folder.mkdir(exist_ok=True)
                rhs = example.b.copy() if mode == 'study' else constant_load(example.n)
                started = perf_counter(); u = factor.solve(rhs); solve_seconds = perf_counter()-started
                residual = np.linalg.norm(rhs-example.a@u); rhs_norm = np.linalg.norm(rhs)
                threshold = max(config.absolute_tolerance, config.tolerance*rhs_norm)
                row = dict(family=record['family'], name=example.name, rhs_mode=mode, n=example.n,
                    relative_residual=float(residual/rhs_norm), absolute_residual=float(residual),
                    stopping_threshold=float(threshold), direct_success=bool(np.isfinite(u).all() and residual <= threshold),
                    discrete_target_relative_error=float(np.linalg.norm(u-example.exact)/np.linalg.norm(example.exact)) if mode == 'study' else None,
                    factor_seconds=factor_seconds, direct_solve_seconds=solve_seconds,
                    fixed_C_success=None, fixed_C_cycles=None, fixed_C_relative_residual=None)
                mg = None
                if setup_error is not None:
                    mg = setup_error
                    row['fixed_C_success'] = False
                    write_json(folder/(record['family']+'_mg.json'), mg)
                elif prepared is not None:
                    try:
                        result = prepared.solve(rhs, x0=np.zeros_like(rhs)); mg = result.to_dict()
                        row.update(fixed_C_success=bool(result.converged), fixed_C_cycles=result.cycles,
                                   fixed_C_relative_residual=float(np.linalg.norm(rhs-example.a@result.x)/rhs_norm))
                        np.savez_compressed(folder/(record['family']+'_mg.npz'), u=result.x, rhs=rhs)
                    except (ValueError, RuntimeError, FloatingPointError) as exc:
                        mg = dict(converged=False, error=f'{type(exc).__name__}: {exc}', residual_history=[])
                        row['fixed_C_success'] = False
                    write_json(folder/(record['family']+'_mg.json'), mg)
                metrics.append(row)
                if not row['direct_success']:
                    write_json(output/'metrics.json', metrics)
                    write_json(output/'progress.json', dict(status='failed_reference', name=example.name, rhs_mode=mode))
                    raise RuntimeError('Reference failed the recorded tolerance; no successful field figure will be published')
                fields = element_fields(example.case, example.scale, u)
                sp.save_npz(folder/(record['family']+'_A.npz'), example.a)
                np.savez_compressed(folder/(record['family']+'_fields.npz'), **fields, rhs=rhs, u=u,
                    study_target=example.exact if mode == 'study' else np.empty(0))
                item = dict(example=example, rhs=rhs, fields=fields, metrics=row, mg=mg)
                groups[mode].append(item); detail(item, folder, mode, args.dpi)
                completed += 1
                write_json(output/'progress.json', dict(status='running', completed=completed, total=7*len(modes)))
                print(f'[{completed}/{7*len(modes)}] {example.name} {mode}: '
                    f'LU relres={row["relative_residual"]:.3e} fixed_C={row["fixed_C_success"]}', flush=True)
        first = groups[modes[0]]
        for kind in ('lambda_max', 'anisotropy'):
            overview(first, output, 'coefficients' if kind == 'lambda_max' else 'anisotropy',
                kind, r'$-\nabla\cdot(\widetilde K\nabla u)=f$,  $\widetilde K=sK$,  $u|_{\partial\Omega}=0$', origin, args.dpi)
        for mode, items in groups.items():
            rhs_label = ('Original study RHS #0: discrete manufactured load $b=A u_*$'
                if mode == 'study' else 'Supplemental common forcing $f=1$; same seven operators')
            overview(items, output, mode+'_solutions', 'solution', rhs_label, origin, args.dpi)
            overview(items, output, mode+'_flux', 'flux', rhs_label, origin, args.dpi)
        write_json(output/'manifest.json', manifest)
    write_json(output/'metrics.json', metrics)
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    imported = origin['kind'].startswith('imported')
    (output/'CAPTIONS.md').write_text(
        '# Figure captions and interpretation\n\n'
        + ('Actual cases imported from the recorded development split.\n\n' if imported else
           '**Configuration-generated illustrations, not verified cases from the user\'s existing run.**\n\n')
        + '각 family의 사전 지정 sample index에서 선택한 확산 텐서와 P1 유한요소 수치해이다. '
          '영역은 무차원 단위 정사각형이며 모든 경계에 u=0을 적용하였다.\n\n'
          '- coefficients: 최대 고유값(확산 강도)과 주 확산축. 흰 선분은 방향축이며 유속이 아니다.\n'
          '- anisotropy: 최대/최소 고유값 비와 주 확산축. 모든 family에 같은 색상 범위를 사용하였다.\n'
          '- study_solutions/flux: 기존 실험의 첫 RHS b=A u_*를 풀었다. 해의 family 간 차이는 제조해의 seed에도 의존한다.\n'
          '- unit_source_solutions/flux (선택): 같은 A에 공통 f=1을 가한 보조 문제. 기존 평가 RHS와 구분한다.\n'
          '- flux: 요소별 q=-K_tilde grad(u_h). 화살표 길이는 정규화했으며 색상은 flux 크기이다. '
          '불연속 계수 경계에서 flux를 평활화하거나 유체 속도로 해석하지 않는다.\n\n'
          '수치해 그림은 fine-grid LU reference이며 NN 성능 개선의 증거가 아니다. '
          '선택적 fixed_C 진단은 실제 잔차 이력과 실패를 그대로 보존한다. '
          'LU와 제조해 사이의 오차는 선형계 풀이 오차이며 연속 PDE의 이산화 오차가 아니다. '
          '기록된 시간은 시각화 진단용으로 단 한 번 측정했으며 성능 benchmark로 사용하지 않는다.\n')
    write_json(output/'progress.json', dict(status='complete', completed=completed, total=completed,
        direct_successes=sum(r['direct_success'] for r in metrics), diagnostic_only=True))
    print(f'Figures and numeric fields saved: {output}', flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--run-dir', type=Path, help='Read-only source EM-line/light/subset run; no checkpoint needed')
    source.add_argument('--config', type=Path, help='Generate clearly labelled illustrations; defaults to light research config')
    p.add_argument('--output', required=True, type=Path, help='New/empty directory outside the source run')
    p.add_argument('--split', choices=['train', 'validation'], default='validation')
    p.add_argument('--n', type=int, default=63, help='Interior nodes per coordinate direction')
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--rhs', choices=['study', 'unit_source', 'both'], default='study')
    p.add_argument('--with-mg', action='store_true', help='Also run unchanged fixed_C diagnostic and save every failure')
    p.add_argument('--dpi', type=int, default=180)
    args = p.parse_args()
    if not 50 <= args.dpi <= 600:
        p.error('--dpi must be between 50 and 600')
    run(args)


if __name__ == '__main__':
    main()
