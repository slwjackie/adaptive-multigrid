"""Isolated neural AMG study: teacher supervision, multilevel tuning and frozen tests.

The data protocol uses the existing seven diffusion families, but coarsening,
features, teacher constraints and solver are algebraic. No legacy run is resumed.
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

from adaptive_mg.provenance import operator_digest
from adaptive_mg.v67.autograd_sparse import SparseTensor
from .model import GraphAttentionTransfer
from .solver import build_hierarchy, solve, multicycle_loss
from .teacher import fit_teacher

VERSION = 'neural-amg-study-v1'
SPLITS = ('train', 'validation', 'final', 'ood_grid', 'ood_rotation',
          'ood_anisotropy', 'ood_contrast', 'ood_topology')


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
    # Bind reused sparse differentiation and PDE generators as well.
    paths += sorted((root / 'adaptive_mg').rglob('*.py'))
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
    if config['model'].get('l1_cap', 8.) != config['teacher'].get('l1_cap', 8.):
        raise ValueError('Model and teacher row L1 caps must agree')
    if config['benchmark']['repetitions'] < 1:
        raise ValueError('At least one timing repetition is required')
    methods = config['benchmark'].get('methods', ['classical', 'teacher', 'learned'])
    if (not methods or len(set(methods)) != len(methods)
            or set(methods) - {'classical', 'teacher', 'learned'} or 'learned' not in methods):
        raise ValueError('Benchmark methods must be unique known methods including learned')
    if config['benchmark']['failure_penalty_seconds'] <= 0:
        raise ValueError('Failure penalty must be positive')
    if config['data']['kind'] not in ('generated', 'npz_manifest'):
        raise ValueError('Data kind must be generated or npz_manifest')
    return config


def load_config(path):
    config = validate_config(read_json(path))
    if config['data']['kind'] == 'npz_manifest':
        config['data']['manifest'] = str((Path(path).resolve().parent / config['data']['manifest']).resolve())
    return config


def _external_identity(config):
    if config['data']['kind'] != 'npz_manifest':
        return {}
    path = Path(config['data']['manifest']); manifest = read_json(path)
    result = {'manifest': file_hash(path)}
    for split, records in manifest['splits'].items():
        for k, record in enumerate(records):
            for field in ('A', 'b'):
                if field in record:
                    result[f'{split}/{k}/{field}'] = file_hash(path.parent / record[field])
    return result


def identity(config):
    return dict(config=digest(config), source=_source_identity(), external=_external_identity(config),
                runtime=dict(numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__),
                             pyamg=pyamg.__version__))


def _specifications(config):
    data = config['data']
    if data['kind'] == 'npz_manifest':
        path = Path(data['manifest']); manifest = read_json(path)
        if not {'train', 'validation', 'final'} <= manifest['splits'].keys():
            raise ValueError('External manifest requires train, validation and final splits')
        if any(s not in SPLITS and not s.startswith('ood_') for s in manifest['splits']):
            raise ValueError('Unknown external split')
        result = {}
        for split, records in manifest['splits'].items():
            result[split] = [dict(name=f'{split}_{k}', split=split, family=r.get('family','external'),
                seed=data['seed'] + k + 1000003 * list(manifest['splits']).index(split),
                external_A=str((path.parent / r['A']).resolve()),
                external_b=str((path.parent / r['b']).resolve()) if 'b' in r else None)
                for k, r in enumerate(records)]
        if any(not rows for rows in result.values()):
            raise ValueError('Empty split in external manifest')
        return result
    from adaptive_mg.v67.research_data import _specification, FAMILIES
    result = {}
    for index, (split, spec) in enumerate(data['splits'].items()):
        if split not in SPLITS:
            raise ValueError('Unknown generated split')
        families = spec.get('families', data.get('families', list(FAMILIES)))
        full = dict(spec, seed=data['seed'] + 1000003 * index, discretization='q1')
        result[split] = [_specification(split, full, n, family, k, 0)
                         for n in spec['sizes'] for family in families for k in range(spec['per_family'])]
    if not {'train', 'validation', 'final'} <= result.keys() or any(not x for x in result.values()):
        raise ValueError('Nonempty train, validation, final specifications required')
    return result


@dataclass
class Case:
    name: str
    a: sp.csr_matrix
    b: np.ndarray
    metadata: dict


def _materialize(spec):
    if 'external_A' in spec:
        a = sp.load_npz(spec['external_A']).tocsr().astype(np.float64)
        b = (np.load(spec['external_b'], allow_pickle=False) if spec['external_b']
             else np.random.default_rng(spec['seed']).normal(size=a.shape[0]))
    else:
        from adaptive_mg.v67.research_data import _example
        example = _example(spec); a, b = example.a, example.b
    a.sum_duplicates(); a.sort_indices(); a.eliminate_zeros()
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] != a.shape[1] or b.shape != (a.shape[0],) or not np.isfinite(b).all():
        raise ValueError('Invalid operator/RHS shape or values')
    return a, b


def _case_paths(run, spec):
    directory = Path(run) / 'data' / spec['split']
    return directory / (spec['name'] + '.A.npz'), directory / (spec['name'] + '.b.npy')


def _load_case(run, record):
    ap, bp = _case_paths(run, record['spec'])
    if file_hash(ap) != record['a_file_sha256'] or file_hash(bp) != record['b_file_sha256']:
        raise ValueError('Cached case content changed')
    return Case(record['spec']['name'], sp.load_npz(ap), np.load(bp, allow_pickle=False), record)


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
        a, b = _materialize(spec)
        group = operator_digest(a, scale_invariant=True)
        if group in forbidden:
            raise ValueError('Duplicate normalized operator: choose disjoint data seeds/operators')
        forbidden.add(group)
        ap, bp = _case_paths(run, spec); ap.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(ap, a); np.save(bp, b, allow_pickle=False)
        record = dict(spec=spec, normalized_operator_digest=group, operator_digest=operator_digest(a),
                      a_file_sha256=file_hash(ap), b_file_sha256=file_hash(bp))
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
            raise ValueError('Config, source, external data or runtime changed; resume rejected')
    else:
        if any(run.iterdir()):
            raise FileExistsError('New run directory must be empty')
        atomic_json(run / 'config.json', config)
        atomic_json(run / 'identity.json', ident)
        atomic_json(run / 'data_plan.json', _specifications(config))
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
                            cycles=int(result['cycles']), residuals=[float(x) if np.isfinite(x) else None for x in result['residuals']]))
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


def supervised_loss(model, labels, action_weight=.1):
    losses = []
    for sample in labels:
        p = model(SparseTensor.from_scipy(sample['a']), sample['structure'])
        target = torch.as_tensor(np.asarray(sample['target'][p.rows, p.cols]).ravel(), dtype=torch.float64)
        error = p.values - target
        weight = error.square().mean() / target.square().mean().clamp_min(1e-12)
        coarse = torch.tensor(sample['coarse_vectors'], dtype=torch.float64)
        action = p.apply(coarse) - torch.as_tensor(sample['target'] @ sample['coarse_vectors'], dtype=torch.float64)
        scale = torch.tensor(sample['vectors'], dtype=torch.float64).square().mean().clamp_min(1e-12)
        losses.append(weight + action_weight * action.square().mean() / scale)
    return torch.stack(losses).mean()


def _model(config):
    return GraphAttentionTransfer(**config['model']).double()


def _state(model):
    return {k: value.detach().clone() for k, value in model.state_dict().items()}


def _check_run(config, run):
    if read_json(Path(run) / 'identity.json') != identity(config):
        raise ValueError('Config, source, external data or runtime changed')


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
                        loss = supervised_loss(model, _labels(config, record, run), pcfg.get('action_weight', .1))
                    else:
                        case = _load_case(run, record)
                        hierarchy = _hierarchy(config, case.a, model=model, seed=record['spec']['seed'] % (2**31))
                        rng = np.random.default_rng(np.random.SeedSequence([seed, phase_index, epoch, offset]))
                        probes = torch.tensor(rng.normal(size=(case.a.shape[0], pcfg.get('probes', 3))), dtype=torch.float64)
                        loss = multicycle_loss(hierarchy, probes, cycles=pcfg.get('cycles', 3))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), pcfg.get('gradient_clip', 2.))
                    if not torch.isfinite(grad):
                        raise FloatingPointError('Nonfinite training gradient')
                    optimizer.step()
                    state.update(epoch=epoch, offset=offset+1, updates=state['updates']+1)
                    done_here += 1; save()
                    print(f"{phase} epoch={epoch+1}/{pcfg['epochs']} case={offset+1}/{len(order)} loss={float(loss.detach()):.6g}", flush=True)
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
    data = {s: read_json(run/f'data_{s}.json')['records'] for s in ('train','validation')}
    if state.get('identity') != identity(config) or state.get('development_digest') != digest(data):
        raise ValueError('Training checkpoint identity differs from this run')
    if not state['complete'] or state['best_model'] is None:
        raise ValueError('Complete training and validation before freeze')
    model_path = run / 'frozen_model.pt'
    atomic_torch(model_path, dict(spec=_model(config).spec, state_dict=state['best_model']))
    files = ['frozen_model.pt', 'config.json', 'identity.json', 'data_plan.json',
             'data_train.json', 'data_validation.json', 'validation.json']
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
