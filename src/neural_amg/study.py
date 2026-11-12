"""Isolated neural AMG study: teacher supervision, multilevel tuning and frozen tests.

Seven diffusion families are assembled exclusively with P1 FEM on seeded
unstructured triangular meshes. No old Q1 or structured P1 run is resumed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time
import os

import numpy as np
import scipy
import pyamg
import scipy.sparse as sp
import torch
from threadpoolctl import threadpool_limits

from neural_amg.provenance import operator_digest
from neural_amg.autograd_sparse import SparseTensor
from .model import GraphAttentionTransfer, cancellation_penalty
from .solver import build_hierarchy, solve, multicycle_loss
from .teacher import fit_teacher
from .fem import DISCRETIZATION, TriangleMesh
from .data import FAMILIES, validate_data_config, specification, materialize

VERSION = 'neural-amg-p1-study-v3'
SPLITS = ('train', 'validation', 'final', 'ood_grid', 'ood_rotation',
          'ood_anisotropy', 'ood_contrast', 'ood_topology', 'ood_mesh')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        temporary = f.name
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = f.name
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def read_json(path):
    return json.loads(Path(path).read_text())


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / 'neural_amg').glob('*.py'))
    return {str(p.relative_to(root)): file_hash(p) for p in paths}


def validate_config(config):
    if config.get('version') != VERSION:
        raise ValueError('Unsupported neural AMG study version')
    if config['model'].get('feature_mode', 'original') not in ('original', 'row9', 'symmetric9'):
        raise ValueError('Unknown feature mode')
    if config['model'].get('depth', 3) != 3:
        raise ValueError('This protocol fixes A-graph information depth to three hops')
    for phase in ('supervised', 'finetune'):
        if not isinstance(config[phase]['epochs'], int) or config[phase]['epochs'] < 0:
            raise ValueError('Epoch count must be nonnegative')
    if sum(config[p]['epochs'] for p in ('supervised', 'finetune')) == 0:
        raise ValueError('At least one training epoch is required')
    for component in ('model', 'teacher'):
        if config[component].get('l1_cap') is not None:
            raise ValueError('Hard L1 caps were removed; use the cancellation warm-start command')
    for component in ('teacher', 'supervised', 'finetune'):
        value = config[component].get('cancellation_weight', .1)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError('cancellation_weight must be finite and nonnegative')
    if config['benchmark']['repetitions'] < 1:
        raise ValueError('At least one timing repetition is required')
    methods = config['benchmark'].get('methods', ['classical', 'teacher', 'learned'])
    if (not methods or len(set(methods)) != len(methods)
            or set(methods) - {'classical', 'teacher', 'learned'} or 'learned' not in methods):
        raise ValueError('Benchmark methods must be unique known methods including learned')
    if config['benchmark']['failure_penalty_seconds'] <= 0:
        raise ValueError('Failure penalty must be positive')
    validate_data_config(config['data'])
    return config


def load_config(path):
    return validate_config(read_json(path))


def identity(config):
    return dict(config=digest(config), source=_source_identity(),
                runtime=dict(numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__),
                             pyamg=pyamg.__version__))


def _specifications(config):
    data = config['data']
    validate_data_config(data)
    result = {}
    for index, (split, spec) in enumerate(data['splits'].items()):
        if split not in SPLITS:
            raise ValueError('Unknown P1 data split')
        families = spec.get('families', data.get('families', list(FAMILIES)))
        result[split] = [specification(split, data, index, n, family, k)
                         for n in spec['sizes'] for family in families for k in range(spec['per_family'])]
    return result


@dataclass
class Case:
    name: str
    a: sp.csr_matrix
    b: np.ndarray
    metadata: dict
    mesh: TriangleMesh


def _materialize(spec):
    return materialize(spec)


def _case_paths(run, spec):
    directory = Path(run) / 'data' / spec['split']
    return directory / (spec['name'] + '.A.npz'), directory / (spec['name'] + '.b.npy')


def _mesh_path(run, spec):
    return Path(run) / 'data' / spec['split'] / (spec['name'] + '.mesh.npz')


def _load_case(run, record):
    ap, bp = _case_paths(run, record['spec'])
    mp = _mesh_path(run, record['spec'])
    if (file_hash(ap) != record['a_file_sha256'] or file_hash(bp) != record['b_file_sha256']
            or file_hash(mp) != record['mesh_file_sha256']):
        raise ValueError('Cached case content changed')
    with np.load(mp, allow_pickle=False) as saved:
        mesh = TriangleMesh(saved['points'], saved['triangles'], saved['boundary'])
        if not np.array_equal(saved['interior'], mesh.interior):
            raise ValueError('Cached P1 degree-of-freedom map changed')
    if mesh.digest() != record['mesh_digest']:
        raise ValueError('Cached P1 mesh identity changed')
    a, b = sp.load_npz(ap), np.load(bp, allow_pickle=False)
    if a.shape != (len(mesh.interior),)*2 or b.shape != (len(mesh.interior),):
        raise ValueError('P1 mesh and operator dimensions differ')
    return Case(record['spec']['name'], a, b, record, mesh)


def _prepare_split(run, split, specs, forbidden):
    path = Path(run) / f'data_{split}.json'
    records = read_json(path)['records'] if path.exists() else []
    if len(records) > len(specs):
        raise ValueError('Unexpected cached case count')
    for i, record in enumerate(records):
        if digest(record['spec']) != digest(specs[i]):
            raise ValueError('Cached case specification changed')
        _load_case(run, record)
        if record['normalized_operator_digest'] in forbidden:
            raise ValueError('Operator overlap across data splits')
        forbidden.add(record['normalized_operator_digest'])
    for spec in specs[len(records):]:
        system = _materialize(spec)
        a, b, mesh = system.a, system.b, system.mesh
        group = operator_digest(a, scale_invariant=True)
        if group in forbidden:
            raise ValueError('Duplicate normalized operator: choose disjoint data seeds/operators')
        forbidden.add(group)
        ap, bp = _case_paths(run, spec); ap.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(ap, a); np.save(bp, b, allow_pickle=False)
        mp = _mesh_path(run, spec)
        np.savez_compressed(mp, points=mesh.points, triangles=mesh.triangles,
                            boundary=mesh.boundary, interior=mesh.interior)
        record = dict(spec=spec, normalized_operator_digest=group, operator_digest=operator_digest(a),
                      a_file_sha256=file_hash(ap), b_file_sha256=file_hash(bp),
                      mesh_file_sha256=file_hash(mp), mesh_digest=mesh.digest(),
                      mesh_statistics=mesh.statistics())
        records.append(record)
        atomic_json(path, dict(records=records))
    return records


def prepare(config, run, *, resume=False):
    validate_config(config); run = Path(run); run.mkdir(parents=True, exist_ok=True)
    ident = identity(config)
    if (run / 'identity.json').exists():
        if not resume:
            raise FileExistsError('Run exists; use --resume or a new output directory')
        if read_json(run / 'identity.json') != ident:
            raise ValueError('Config, source, P1 data or runtime changed; resume rejected')
    else:
        if any(run.iterdir()):
            raise FileExistsError('New run directory must be empty')
        atomic_json(run / 'config.json', config)
        atomic_json(run / 'identity.json', ident)
        atomic_json(run / 'data_plan.json', _specifications(config))
    _verify_initialization(run)
    plan = read_json(run / 'data_plan.json')
    if digest(plan) != digest(_specifications(config)):
        raise ValueError('Data plan changed')
    forbidden = set()
    result = {split: _prepare_split(run, split, plan[split], forbidden)
              for split in ('train', 'validation')}
    return result


def _hierarchy(config, a, *, model=None, method='learned', seed=0):
    return build_hierarchy(a, model=model, method=method, strength=config['teacher']['strength'],
                           seed=seed, teacher_options={k:v for k,v in config['teacher'].items() if k != 'strength'},
                           **config['hierarchy'])


def _solve(config, hierarchy, b):
    return solve(hierarchy, b, **config['solve'])


def benchmark(config, records, run, model, methods=('learned',)):
    """Cold setup+solve and a separately timed same-hierarchy warm solve.

    Data loading, model loading, offline labels and training are excluded.
    Features, NN inference, all Galerkin operations and coarse factorization
    belong to cold timing (factorization may occur lazily in the first solve).
    """
    output = []
    model.eval()
    with torch.no_grad():
        for record in records:
            case = _load_case(run, record)
            for method in methods:
                timings = []
                for repeat in range(config['benchmark']['repetitions']):
                    start = time.perf_counter()
                    try:
                        hierarchy = _hierarchy(config, case.a, model=model if method == 'learned' else None,
                                               method=method, seed=record['spec']['seed'] % (2**31))
                        setup = time.perf_counter() - start
                        result = _solve(config, hierarchy, case.b)
                        total = time.perf_counter() - start
                        warm_start = time.perf_counter()
                        warm = _solve(config, hierarchy, case.b)
                        warm_seconds = time.perf_counter() - warm_start
                        finite = bool(np.isfinite(result['x']).all())
                        converged = bool(result['converged'] and warm['converged'] and finite)
                        timings.append(dict(converged=converged, setup_seconds=setup,
                            total_seconds=total, warm_seconds=warm_seconds,
                            cycles=int(result['cycles']), residuals=[float(x) if np.isfinite(x) else None for x in result['residuals']],
                            guard_events=result.get('guard_events', []),
                            warm_guard_events=warm.get('guard_events', []),
                            level_sizes=result.get('level_sizes'),
                            terminal_size=result.get('terminal_size'),
                            operator_complexity=result.get('operator_complexity'),
                            warm_terminal_size=warm.get('terminal_size')))
                    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                        timings.append(dict(converged=False, error=str(error),
                            total_seconds=time.perf_counter()-start))
                success = all(t['converged'] for t in timings)
                median = float(np.median([t['total_seconds'] for t in timings]))
                output.append(dict(case=case.name, method=method, converged=success,
                    median_total_seconds=median if success else config['benchmark']['failure_penalty_seconds'],
                    repetitions=timings))
    return output


def _score(rows):
    return (sum(not row['converged'] for row in rows),
            float(np.mean([row['median_total_seconds'] for row in rows])))


def _labels(config, record, run):
    directory = Path(run) / 'labels'; directory.mkdir(exist_ok=True)
    path = directory / (record['spec']['name'] + '.pt')
    checksum = path.with_suffix('.json')
    if path.exists() and checksum.exists():
        if file_hash(path) != read_json(checksum)['sha256']:
            raise ValueError('Teacher cache checksum mismatch')
        return torch.load(path, map_location='cpu', weights_only=False)
    case = _load_case(run, record)
    hierarchy = _hierarchy(config, case.a, method='classical', seed=record['spec']['seed'] % (2**31))
    labels = []
    kwargs = dict(config['teacher']); kwargs.pop('strength')
    kwargs.setdefault('bootstrap_max_levels', config['hierarchy'].get('max_levels', 10))
    kwargs.setdefault('bootstrap_first_cycle_max_growth', config['hierarchy'].get('first_cycle_max_growth', 10.))
    for level_index, level in enumerate(hierarchy.levels):
        if level.p is None:
            continue
        result = fit_teacher(level.a.numpy(), level.structure,
                             strength=config['teacher']['strength'],
                             seed=(record['spec']['seed'] + level_index) % (2**31), **kwargs)
        labels.append(dict(a=level.a.numpy(), structure=level.structure, target=result.p,
                           vectors=np.asarray(result.vectors), coarse_vectors=np.asarray(result.coarse_vectors),
                           report=result.report))
    if not labels:
        raise ValueError('Training operator has no nonterminal MG level; lower max_coarse')
    atomic_torch(path, labels); atomic_json(checksum, dict(sha256=file_hash(path)))
    return labels


def supervised_loss(model, labels, action_weight=.1, cancellation_weight=.1):
    losses = []
    for sample in labels:
        p = model(SparseTensor.from_scipy(sample['a']), sample['structure'])
        target = torch.as_tensor(np.asarray(sample['target'][p.rows, p.cols]).ravel(), dtype=torch.float64)
        error = p.values - target
        weight = error.square().mean() / target.square().mean().clamp_min(1e-12)
        coarse = torch.tensor(sample['coarse_vectors'], dtype=torch.float64)
        action = p.apply(coarse) - torch.as_tensor(sample['target'] @ sample['coarse_vectors'], dtype=torch.float64)
        scale = torch.tensor(sample['vectors'], dtype=torch.float64).square().mean().clamp_min(1e-12)
        losses.append(weight + action_weight * action.square().mean() / scale
                      + cancellation_weight * cancellation_penalty(p, sample['structure']))
    return torch.stack(losses).mean()


def _model(config):
    return GraphAttentionTransfer(**config['model']).double()


def _state(model):
    return {k: value.detach().clone() for k, value in model.state_dict().items()}


def _check_run(config, run):
    if read_json(Path(run) / 'identity.json') != identity(config):
        raise ValueError('Config, source, P1 data or runtime changed')


def _verify_initialization(run, state=None):
    """Bind copied source-run evidence to its new experiment checkpoint."""
    run = Path(run)
    path = run / 'cancellation_start.json'
    if state is None and (run / 'training.pt').exists():
        state = torch.load(run / 'training.pt', map_location='cpu', weights_only=False)
    expected = (state or {}).get('cancellation_start_digest')
    if not path.exists():
        if expected is not None:
            raise ValueError('Cancellation initialization audit is missing')
        return
    report = read_json(path)
    if expected != digest(report):
        raise ValueError('Cancellation initialization audit changed')
    for name, checksum in report['archived_files'].items():
        if not (run / name).resolve().is_relative_to(run.resolve()):
            raise ValueError('Initialization archive path escapes run directory')
        if file_hash(run / name) != checksum:
            raise ValueError('Initialization archive changed: ' + name)


def train(config, run, *, resume=False, max_updates=None):
    data = prepare(config, run, resume=resume); run = Path(run)
    if (run / 'freeze.json').exists():
        raise ValueError('Frozen runs cannot be trained')
    torch.set_num_threads(config.get('torch_threads', 1))
    seed = config['seed']; torch.manual_seed(seed); np.random.seed(seed % (2**32)); random.seed(seed)
    model = _model(config)
    path = run / 'training.pt'
    phases = ('supervised', 'finetune')
    state = dict(identity=identity(config), development_digest=digest(data), phase=0, epoch=0, offset=0, updates=0, complete=False, best_score=None,
                 best_model=None, history=[], model=_state(model), optimizer=None)
    if path.exists():
        if not resume:
            raise FileExistsError('Training checkpoint exists; use --resume')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('identity') != identity(config) or state.get('development_digest') != digest(data):
            raise ValueError('Training checkpoint identity differs from this run')
        model.load_state_dict(state['model']); torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
    if state['complete']:
        return state
    done_here = 0
    with threadpool_limits(limits=config.get('torch_threads', 1)):
        for phase_index in range(state['phase'], len(phases)):
            phase = phases[phase_index]; pcfg = config[phase]
            optimizer = torch.optim.Adam(model.parameters(), lr=pcfg['learning_rate'])
            if phase_index == state['phase'] and state['optimizer'] is not None:
                optimizer.load_state_dict(state['optimizer'])
            if phase_index != state['phase']:
                state.update(phase=phase_index, epoch=0, offset=0, optimizer=None)
            def save():
                state.update(model=_state(model), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate())
                atomic_torch(path, state)
            def validate():
                rows = benchmark(config, data['validation'], run, model)
                score = _score(rows)
                if state['best_score'] is None or tuple(score) < tuple(state['best_score']):
                    state['best_score'] = score; state['best_model'] = _state(model)
                    state['best_update'] = state['updates']
                state['history'].append(dict(phase=phase, update=state['updates'], score=score, cases=rows))
                atomic_json(run / 'validation.json', state['history'])
                print(f"validation update={state['updates']} failures={score[0]} mean_total_ms={score[1]*1000:.3f}", flush=True)
            if state['best_model'] is None:
                validate(); save()
            for epoch in range(state['epoch'], pcfg['epochs']):
                order = np.random.default_rng(np.random.SeedSequence([seed, phase_index, epoch])).permutation(len(data['train']))
                for offset in range(state['offset'], len(order)):
                    record = data['train'][int(order[offset])]
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    if phase == 'supervised':
                        loss = supervised_loss(model, _labels(config, record, run),
                                               pcfg.get('action_weight', .1),
                                               pcfg.get('cancellation_weight', .1))
                    else:
                        case = _load_case(run, record)
                        hierarchy = _hierarchy(config, case.a, model=model, seed=record['spec']['seed'] % (2**31))
                        rng = np.random.default_rng(np.random.SeedSequence([seed, phase_index, epoch, offset]))
                        probes = torch.tensor(rng.normal(size=(case.a.shape[0], pcfg.get('probes', 3))), dtype=torch.float64)
                        # Keep a penalty on rejected predictions too: the first
                        # V-cycle may truncate the hierarchy after a guard fires.
                        penalties = [cancellation_penalty(level.p, level.structure)
                                     for level in hierarchy.levels if level.p is not None]
                        cancellation = torch.stack(penalties).mean() if penalties else probes.new_zeros(())
                        mg_loss = multicycle_loss(hierarchy, probes, cycles=pcfg.get('cycles', 3))
                        loss = mg_loss + pcfg.get('cancellation_weight', .1) * cancellation
                        for event in hierarchy.guard_events:
                            print('hierarchy_guard ' + json.dumps(dict(case=case.name, **event)), flush=True)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    optimized = loss.requires_grad
                    if optimized:
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), pcfg.get('gradient_clip', 2.))
                        if not torch.isfinite(grad):
                            raise FloatingPointError('Nonfinite training gradient')
                        optimizer.step()
                    else:
                        state['direct_only_cases'] = state.get('direct_only_cases', 0) + 1
                    state.update(epoch=epoch, offset=offset+1, updates=state['updates']+1)
                    done_here += 1; save()
                    detail = ''
                    if phase == 'finetune':
                        detail = f' mg={float(mg_loss.detach()):.6g} cancellation={float(cancellation.detach()):.6g} optimized={int(optimized)}'
                    print(f"{phase} epoch={epoch+1}/{pcfg['epochs']} case={offset+1}/{len(order)} loss={float(loss.detach()):.6g}{detail}", flush=True)
                    if max_updates is not None and done_here >= max_updates:
                        return state
                validate()
                state.update(epoch=epoch+1, offset=0); save()
            state.update(phase=phase_index+1, epoch=0, offset=0, optimizer=None)
            # Preserve the transition itself, including an interrupted zero-epoch phase.
            state.update(model=_state(model), torch_rng=torch.get_rng_state(),
                         numpy_rng=np.random.get_state(), python_rng=random.getstate())
            atomic_torch(path, state)
        state['complete'] = True; atomic_torch(path, state)
    return state


def freeze(config, run):
    run = Path(run); _check_run(config, run)
    path = run / 'freeze.json'
    if path.exists():
        return verify_freeze(config, run)
    state = torch.load(run / 'training.pt', map_location='cpu', weights_only=False)
    _verify_initialization(run, state)
    data = {s: read_json(run/f'data_{s}.json')['records'] for s in ('train','validation')}
    if state.get('identity') != identity(config) or state.get('development_digest') != digest(data):
        raise ValueError('Training checkpoint identity differs from this run')
    if not state['complete'] or state['best_model'] is None:
        raise ValueError('Complete training and validation before freeze')
    model_path = run / 'frozen_model.pt'
    atomic_torch(model_path, dict(spec=_model(config).spec, state_dict=state['best_model']))
    files = ['frozen_model.pt', 'config.json', 'identity.json', 'data_plan.json',
             'data_train.json', 'data_validation.json', 'validation.json']
    if (run / 'bootstrap_depth_recovery.json').exists():
        files.append('bootstrap_depth_recovery.json')
    if (run / 'teacher_l1_recovery.json').exists():
        files.append('teacher_l1_recovery.json')
    if (run / 'cancellation_start.json').exists():
        files.append('cancellation_start.json')
        files.extend(read_json(run / 'cancellation_start.json')['archived_files'])
    payload = dict(version=VERSION, identity=identity(config),
        files={f:file_hash(run/f) for f in files}, best_update=state['best_update'],
        best_score=state['best_score'], selection='failures_then_mean_case_median_setup_plus_solve')
    payload['digest'] = digest(payload); atomic_json(path, payload)
    return payload


def verify_freeze(config, run):
    run = Path(run); payload = read_json(run / 'freeze.json')
    stored = payload.pop('digest')
    if digest(payload) != stored or payload['identity'] != identity(config):
        raise ValueError('Frozen identity or content changed')
    for name, expected in payload['files'].items():
        if file_hash(run / name) != expected:
            raise ValueError('Frozen file changed: ' + name)
    for split in ('train', 'validation'):
        for record in read_json(run / f'data_{split}.json')['records']:
            _load_case(run, record)
    return dict(payload, digest=stored)


def evaluate(config, run, *, resume=False):
    run = Path(run); frozen = verify_freeze(config, run)
    path = run / 'evaluation.json'
    if path.exists() and not resume:
        raise FileExistsError('Evaluation exists; use --resume')
    result = read_json(path) if path.exists() else dict(freeze=frozen['digest'], results=[], complete=False)
    if path.exists():
        saved_digest = result.pop('ledger_digest', None)
        if saved_digest != digest(result):
            raise ValueError('Evaluation ledger changed')
    if result['freeze'] != frozen['digest']:
        raise ValueError('Evaluation freeze mismatch')
    checkpoint = torch.load(run/'frozen_model.pt', map_location='cpu', weights_only=False)
    model = GraphAttentionTransfer.from_spec(checkpoint['spec']).double()
    if model.spec != _model(config).spec:
        raise ValueError('Frozen model spec differs from configuration')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    torch.set_num_threads(config.get('torch_threads',1))
    forbidden = {r['normalized_operator_digest'] for split in ('train','validation')
                 for r in read_json(run/f'data_{split}.json')['records']}
    plan = read_json(run/'data_plan.json')
    allowed = {(s, spec['name'], method) for s, specs in plan.items()
               if s not in ('train','validation') for spec in specs
               for method in config['benchmark'].get('methods', ['classical','teacher','learned'])}
    completed = {(r['split'], r['case'], r['method']) for r in result['results']}
    if len(completed) != len(result['results']) or not completed <= allowed:
        raise ValueError('Unknown or duplicate evaluation record')
    def save_evaluation():
        atomic_json(path, dict(result, ledger_digest=digest(result)))
    with threadpool_limits(limits=config.get('torch_threads',1)):
        for split, specs in plan.items():
            if split in ('train','validation'):
                continue
            records = _prepare_split(run, split, specs, forbidden)
            for record in records:
                methods = [m for m in config['benchmark'].get('methods', ['classical','teacher','learned'])
                           if (split, record['spec']['name'], m) not in completed]
                for row in benchmark(config, [record], run, model, methods):
                    result['results'].append(dict(split=split, **row))
                    save_evaluation()
        result['complete'] = True; save_evaluation()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare','train','freeze','evaluate'):
        p = commands.add_parser(command)
        p.add_argument('--config', required=True)
        p.add_argument('--run-dir', required=True)
        if command in ('prepare','train','evaluate'):
            p.add_argument('--resume', action='store_true')
        if command == 'train':
            p.add_argument('--max-updates', type=int, default=None, help='Stop cleanly after this many updates; resume later')
    args = parser.parse_args(argv); config = load_config(args.config)
    kwargs = {'resume': args.resume} if hasattr(args,'resume') else {}
    if args.command == 'train':
        kwargs['max_updates'] = args.max_updates
    result = globals()[args.command](config, args.run_dir, **kwargs)
    print(json.dumps(dict(command=args.command, run_dir=args.run_dir,
                          complete=result.get('complete', True) if isinstance(result,dict) else True)))


if __name__ == '__main__':
    main()
