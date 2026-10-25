"""Fixed line_alt/P-only study, with explicit legacy joint-study compatibility.

Every experiment owns fresh development data and sealed final/OOD operators.
Search is offline and training-only; deployable models generate operators once
per hierarchy. All selection timings include setup and use the same tolerance.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..provenance import hardware_environment, json_safe
from ..pde import validate_discretization
from .config import AdaptiveConfig
from .direct_pruning_study import _ensure_development_open, _validate_protocol
from .fixed_classical import (BilinearClassicalBaseline, FixedClassicalBaseline,
                              FIXED_STRATEGY_NAME, load_baseline)
from .limited import digest_file, forced_config, initialize_timing_runtime
from .models import Components
from .research_data import (make_research_plan, prepare_development_data,
    load_development_data, freeze_research, claim_final_evaluation,
    materialize_final_data, complete_final_evaluation, HOLDOUT, _hash,
    _write_json as write_json)
from .research_evaluation import evaluate_research
from .joint_transfer_training import (make_joint_components, curriculum_cycles,
    joint_objective, make_oracle as _make_pair_oracle, export_oracle as _export_pair_oracle,
    import_oracle as _import_pair_oracle, measure_raw_candidate)
from .joint_transfer_evaluation import raw_contractions, report, select_measured_candidate
from .learned_p import (VERSION as P_VERSION, NoNeuralSmoother, make_p_components,
    frobenius_objective, make_p_oracle, export_p_oracle, import_p_oracle, require_p_only, p_diagnostic)

VERSION = 'joint-bilinear-transfer-smoother-v1'
PROJECT = Path(__file__).resolve().parents[3]
STAGES = {'learned_P': ('H_P', ('transfer',)),
          'learned_S': ('H_S', ('smoother',)),
          'joint': ('H_SP', ('transfer', 'smoother')),
          'distilled_joint': ('H_SP', ('transfer', 'smoother'))}
_THREAD_LIMIT = None


def _p_only(settings):
    return settings.get('version') == P_VERSION


def _stages(settings):
    return ({'learned_P': ('H_P', ('transfer',)), 'distilled_P': ('H_P', ('transfer',))}
            if _p_only(settings) else STAGES)


def _source_stage(settings):
    return 'learned_P' if _p_only(settings) else 'joint'


def make_oracle(example, model, cfg, **kwargs):
    return (make_p_oracle if isinstance(model.smoother, NoNeuralSmoother) else _make_pair_oracle)(
        example, model, cfg, **kwargs)


def export_oracle(model):
    return (export_p_oracle if isinstance(model.smoother, NoNeuralSmoother) else _export_pair_oracle)(model)


def import_oracle(payload):
    return (import_p_oracle if payload.get('version') == 'free-p-line-alt-v1' else _import_pair_oracle)(payload)


def _read(path):
    return json.loads(Path(path).read_text())


def _initialize(settings):
    global _THREAD_LIMIT
    torch.set_num_threads(settings['torch_threads'])
    initialize_timing_runtime()
    # Apply the declared limit to every loaded BLAS/OpenMP runtime, including
    # scipy's second pool on macOS. Resume checks still verify the environment.
    _THREAD_LIMIT = threadpool_limits(limits=settings['torch_threads'])


def validate_settings(settings):
    validate_discretization(settings.get('discretization', 'p1'))
    p_only = _p_only(settings)
    if settings.get('version') not in (VERSION, P_VERSION):
        raise ValueError('Use a joint bilinear study configuration')
    cfg = AdaptiveConfig.from_dict(settings['solver'])
    BilinearClassicalBaseline.from_config(cfg)
    if p_only:
        if (cfg.branch != 'H_P' or cfg.mode != 'research' or not cfg.use_transfer
                or cfg.use_smoother or cfg.replace_pre or cfg.replace_post
                or cfg.transfer_levels != 'all' or cfg.spatial or cfg.gate_mode != 'open'
                or cfg.use_learned_controller or not cfg.lazy or cfg.application != 'replace'):
            raise ValueError('P-only requires fixed line_alt, all-level P, no NN smoother/replacement')
        objective = settings.get('objective', {})
        if (set(objective) != {'kind', 'probes', 'diagnostic_probes'}
                or objective.get('kind') != 'vcycle_frobenius'):
            raise ValueError('P-only requires the full V-cycle Frobenius objective')
        for key in ('probes', 'diagnostic_probes'):
            if type(objective.get(key)) is not int or objective[key] < 1:
                raise ValueError(f'objective.{key} must be positive')
    elif (cfg.branch != 'H_SP' or cfg.mode != 'research' or not cfg.use_transfer
            or not cfg.use_smoother or cfg.smoother_levels != 'all'
            or cfg.transfer_levels != 'all' or cfg.replace_pre != 2 or cfg.replace_post != 2
            or cfg.replacement_group_pre != 1 or cfg.replacement_group_post != 1
            or cfg.spatial or cfg.gate_mode != 'open' or cfg.use_learned_controller
            or not cfg.lazy or cfg.application != 'replace'):
        raise ValueError('Require all-level P and two pre/two post sparse replacement steps')
    for key in ('seed', 'hidden', 'torch_threads', 'training_per_family',
                'validation_per_family', 'raw_cycles', 'raw_probes', 'target_rhs_count'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('training_sizes', 'validation_sizes'):
        values = settings[key]
        if (not values or len(set(values)) != len(values)
                or any(type(n) is not int or n < 3 or n & (n + 1) for n in values)):
            raise ValueError('Require distinct 2**L-1 grid sizes >=3')
    if p_only and min(settings['training_sizes']) <= cfg.mg.coarsest_n:
        raise ValueError('P learning requires at least one nonterminal level')
    for name in ('training', 'distillation'):
        spec = settings[name]
        keys = ('updates', 'start_cycles', 'end_cycles') if p_only else (
            'updates', 'start_cycles', 'end_cycles', 'probes_per_kind', 'hard_smoothing', 'hard_cycles')
        for key in keys:
            if type(spec[key]) is not int or spec[key] < 1:
                raise ValueError(f'{name}.{key} must be positive')
        if spec['end_cycles'] < spec['start_cycles'] or not np.isfinite(spec['learning_rate']) or spec['learning_rate'] <= 0:
            raise ValueError('Invalid curriculum or learning rate')
        if spec.get('gradient_clip', 0) <= 0:
            raise ValueError('Require positive gradient clipping')
        if p_only and (spec['start_cycles'] != 1 or spec['end_cycles'] != 1):
            raise ValueError('Frobenius training uses exactly one full V-cycle error operator')
    kd = settings['distillation']
    if (type(kd['finetune_updates']) is not int or kd['finetune_updates'] < 1
            or kd['lambda_action'] <= 0 or kd['lambda_operator'] <= 0):
        raise ValueError('Distillation needs action and operator losses followed by joint fine-tuning')
    search = settings['search']
    for key in ('starts', 'updates', 'cycles', 'repeats'):
        if type(search[key]) is not int or search[key] < 1:
            raise ValueError(f'search.{key} must be positive')
    if search['starts'] < 2 or search['learning_rate'] <= 0 or search['perturbation'] < 0:
        raise ValueError('Search needs network and Jacobi starts and a positive learning rate')
    if p_only and search['cycles'] != 1:
        raise ValueError('P-only search optimizes the same one-cycle Frobenius objective')
    return cfg


def _sources():
    return sorted([p for p in (PROJECT / 'src').rglob('*') if p.suffix in ('.py', '.cpp')]
                  + [PROJECT / 'scripts/run_v6_7_joint_transfer_study.py',
                     PROJECT / 'scripts/run_v6_7_learned_p_study.py'])


def _manifest(settings, baseline):
    return dict(version=settings['version'], settings_digest=_hash(settings), baseline_digest=baseline.digest(),
        source_digest=_hash({str(p.relative_to(PROJECT)): digest_file(p) for p in _sources()}),
        hardware=hardware_environment(refresh=True))


def load(output):
    out = Path(output).resolve()
    settings = _read(out / 'configuration.json')
    cfg = validate_settings(settings)
    _initialize(settings)
    baseline = load_baseline(out / 'fixed_classical.json')
    baseline.validate_config(cfg)
    if _read(out / 'joint_manifest.json') != json_safe(_manifest(settings, baseline)):
        raise ValueError('Sources/settings/environment changed; use a new joint study run')
    return out, settings, cfg, baseline


def _development(out, settings, baseline):
    if (out / 'data/development_manifest.json').exists():
        return load_development_data(out / 'data', baseline)
    plan = make_research_plan(smoke=settings['smoke'], seed=settings['seed'] + 7100003, discretization=settings.get('discretization', 'p1'))
    for name, key, count in (('train', 'training_sizes', 'training_per_family'),
                              ('validation', 'validation_sizes', 'validation_per_family')):
        spec = plan['splits'][name]
        spec.update(sizes=settings[key], per_family=settings[count])
        spec['count'] = len(spec['sizes']) * len(spec['families']) * spec['per_family']
    roots = list(dict.fromkeys([str(out.parent)] + settings.get('historical_roots', [])))
    return prepare_development_data(out / 'data', baseline, plan=plan, historical_roots=roots)


def prepare(config, output, *, resume=False):
    settings = _read(config)
    cfg = validate_settings(settings)
    _initialize(settings)
    baseline = BilinearClassicalBaseline.from_config(cfg)
    out = Path(output).resolve()
    if (out / 'joint_manifest.json').exists():
        if not resume:
            raise FileExistsError('Run exists; use --resume or a new directory')
        _, stored, _, _ = load(out)
        if settings != stored:
            raise ValueError('Prepare settings differ')
        _ensure_development_open(out)
    else:
        if resume:
            raise FileNotFoundError('No joint study to resume')
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Prepare requires an empty run directory')
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'configuration.json', settings)
        write_json(out / 'fixed_classical.json', baseline.to_dict())
        factory = make_p_components if _p_only(settings) else make_joint_components
        model = factory(hidden=settings['hidden'], seed=settings['seed'])
        model.save(out / 'initial.pt')
        write_json(out / 'initial.json', dict(sha256=digest_file(out / 'initial.pt')))
        write_json(out / 'joint_manifest.json', _manifest(settings, baseline))
    train_data, validation, _ = _development(out, settings, baseline)
    print(f'Prepared {"P-only line_alt" if _p_only(settings) else "joint"} study: '
          f'train={len(train_data)}, validation={len(validation)}; '
          'bilinear parent, EM5 evaluation only', flush=True)


@contextmanager
def _lease(out):
    fd = os.open(out / '.joint_work.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another training/search process owns this run') from exc
        yield
    finally:
        os.close(fd)


def _save_state(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _mutable(out):
    _ensure_development_open(out)
    if (out / 'benchmark_evidence.json').exists():
        raise ValueError('Validation committed; create a new run to change learned weights')


def _initial(out):
    if digest_file(out / 'initial.pt') != _read(out / 'initial.json')['sha256']:
        raise ValueError('Initial checkpoint changed')
    return Components.load(out / 'initial.pt')


def _checkpoint(out, name, baseline):
    path = out / 'checkpoints' / name / 'candidate.pt'
    status = _read(path.parent / 'status.json')
    training = _read(path.parent / 'training_manifest.json')
    spec = training['settings']
    expected_updates = spec['updates'] + (spec['finetune_updates'] if name.startswith('distilled') else 0)
    if (status.get('status') != 'complete' or status.get('baseline_digest') != baseline.digest()
            or not status.get('all_updates_completed') or status.get('skipped') != 0
            or status.get('updates') != expected_updates or training.get('stage') != name
            or training.get('baseline_digest') != baseline.digest()
            or status.get('checkpoint_sha256') != digest_file(path)):
        raise ValueError(f'Incomplete or changed checkpoint: {name}')
    model = Components.load(path)
    if model.metadata.get('joint_study_stage') != name or model.metadata.get('training_baseline_digest') != baseline.digest():
        raise ValueError('Checkpoint lineage differs')
    return model


def _run_training(out, name, initial, examples, cfg, baseline, settings, *, resume, teachers=None):
    branch, names = _stages(settings)[name]
    cfg = forced_config(cfg, branch)
    spec = settings['distillation' if name.startswith('distilled') else 'training']
    total = spec['updates'] + (spec['finetune_updates'] if teachers is not None else 0)
    target = out / 'checkpoints' / name
    manifest = dict(version=settings['version'], stage=name, config=cfg.to_dict(), settings=spec,
        baseline_digest=baseline.digest(), initial_signature=initial.signature(),
        operators=[e.group_digest for e in examples],
        oracle_hashes=({key: value['sha256'] for key, value in teachers.items()} if teachers is not None else None))
    if _p_only(settings):
        require_p_only(initial, cfg)
        manifest['objective'] = settings['objective']
    path = target / 'training_manifest.json'
    if path.exists():
        if not resume or _read(path) != json_safe(manifest):
            raise ValueError('Training exists; identical --resume required')
        if (target / 'status.json').exists() and _read(target / 'status.json').get('status') == 'complete':
            return _checkpoint(out, name, baseline)
    else:
        write_json(path, manifest)
    model = deepcopy(initial)
    for key in ('smoother', 'transfer', 'detector', 'controller'):
        module = getattr(model, key)
        module.train(key in names)
        for parameter in module.parameters():
            parameter.requires_grad_(key in names)
    params = [p for key in names for p in getattr(model, key).parameters()]
    optimizer = torch.optim.Adam(params, lr=spec['learning_rate'])
    statepath = target / 'resume.pt'
    history, start = [], 0
    if statepath.exists():
        state = torch.load(statepath, map_location='cpu', weights_only=True)
        if (state.get('manifest_digest') != _hash(manifest)
                or type(state.get('updates')) is not int or not 0 <= state['updates'] <= total
                or len(state.get('history', [])) != state['updates']
                or set(state.get('model', {})) != {'smoother', 'transfer', 'detector', 'controller'}):
            raise ValueError('Training resume belongs to a different protocol')
        for key, value in state['model'].items():
            getattr(model, key).load_state_dict(value)
        optimizer.load_state_dict(state['optimizer'])
        history, start = state['history'], state['updates']
    eligible = [e for e in examples if teachers is None or e.group_digest in teachers]
    if not eligible:
        raise ValueError('No successful training-only oracle targets; inspect search report')
    order = np.random.default_rng(settings['seed'] + list(_stages(settings)).index(name)).permutation(len(eligible))
    try:
        for update in range(start, total):
            kd = teachers is not None and update < spec['updates']
            pool = eligible if kd else examples
            index = int(order[update % len(order)]) if kd or teachers is None else update % len(pool)
            example = pool[index]
            teacher = None
            if kd:
                record = teachers[example.group_digest]
                if digest_file(record['path']) != record['sha256']:
                    raise ValueError('Oracle target changed')
                teacher = import_oracle(torch.load(record['path'], map_location='cpu', weights_only=True))
            cycles = (curriculum_cycles(update, spec['updates'],
                        start=spec['start_cycles'], end=spec['end_cycles'])
                      if update < spec['updates'] else spec['end_cycles'])
            optimizer.zero_grad(set_to_none=True)
            if _p_only(settings):
                loss, metrics = frobenius_objective(example, model, cfg,
                    probes=settings['objective']['probes'], seed=settings['seed']+update*104729,
                    teacher=teacher, lambda_action=spec.get('lambda_action', 1.),
                    lambda_operator=spec.get('lambda_operator', .1))
            else:
                loss, metrics = joint_objective(example, model, cfg, cycles=cycles,
                    seed=settings['seed'] + update * 104729,
                    probes_per_kind=spec['probes_per_kind'], hard_smoothing=spec['hard_smoothing'],
                    hard_cycles=spec['hard_cycles'], teacher=teacher,
                    lambda_action=spec.get('lambda_action', 1.), lambda_operator=spec.get('lambda_operator', .1))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, spec['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in params):
                raise FloatingPointError('Nonfinite model after optimizer update')
            row = dict(update=update + 1, case=example.name, cycles=cycles,
                phase='distill' if kd else 'task', loss=float(loss.detach()), gradient_norm=float(norm),
                metrics=json_safe(metrics))
            history.append(row)
            _save_state(statepath, dict(manifest_digest=_hash(manifest), updates=update + 1,
                model={key: getattr(model, key).state_dict()
                       for key in ('smoother', 'transfer', 'detector', 'controller')},
                optimizer=optimizer.state_dict(), history=history))
            write_json(target / 'status.json', dict(status='running', updates=update + 1, total=total))
            print(f'[TRAIN {name}] {update+1}/{total} {example.name}: '
                  f'{row["phase"]}, cycles={cycles}, loss={row["loss"]:.6g}', flush=True)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        write_json(target / 'status.json', dict(status='failed', updates=len(history), reason=str(error)))
        raise
    model.metadata.update(joint_study_stage=name, training_baseline_digest=baseline.digest(),
        training_branch=branch, trainable_components=list(names),
        smoother_trained='smoother' in names, transfer_trained='transfer' in names,
        training_operator_digests=[e.group_digest for e in examples],
        offline_oracle_distillation=teachers is not None, final_test_seen=False)
    if _p_only(settings):
        model.metadata['objective'] = settings['objective']
    model.save(target / 'candidate.pt')
    write_json(target / 'training.json', history)
    write_json(target / 'status.json', dict(status='complete', updates=total, all_updates_completed=True,
        skipped=0, baseline_digest=baseline.digest(), checkpoint_sha256=digest_file(target / 'candidate.pt')))
    return model


def train(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    _mutable(out)
    with _lease(out):
        examples, _, _ = _development(out, settings, baseline)
        initial = _initial(out)
        for name in (['learned_P'] if _p_only(settings) else ['learned_P', 'learned_S', 'joint']):
            _run_training(out, name, initial, examples, cfg, baseline, settings, resume=resume)


def _oracle_records(out):
    """Validate complete TRAIN coverage and lineage before loading oracle labels."""
    out, settings, cfg, baseline = load(out)
    examples, _, _ = load_development_data(out / 'data', baseline)
    source = _checkpoint(out, _source_stage(settings), baseline)
    expected_manifest = dict(version=settings.get('version', VERSION), settings=settings['search'], source=source.signature(),
        baseline_digest=baseline.digest(), operators=[e.group_digest for e in examples],
        target_rhs_count=settings['target_rhs_count'], training_only=True, fallback_allowed=False)
    if _p_only(settings):
        expected_manifest['objective'] = settings['objective']
    manifest = _read(out / 'search/manifest.json')
    if manifest != json_safe(expected_manifest):
        raise ValueError('Search source, training operators, or protocol changed')
    digest = _hash(manifest)
    evidence = _read(out / 'search/report.json')
    rows = evidence.get('rows', [])
    expected = {e.group_digest: e for e in examples}
    if (evidence.get('status') != 'complete' or evidence.get('manifest_digest') != digest
            or evidence.get('training_only') is not True or evidence.get('fallback_allowed') is not False
            or evidence.get('total') != len(expected) or len(rows) != len(expected)
            or len({r.get('operator_digest') for r in rows}) != len(rows)
            or {r.get('operator_digest') for r in rows} != set(expected)):
        raise ValueError('Search must cover every unique TRAIN operator, including recorded failures')
    records = {}
    for row in rows:
        example = expected[row['operator_digest']]
        if (row.get('manifest_digest') != digest or row.get('case') != example.name
                or row.get('matrix_digest') != example.digest
                or len(row.get('starts', [])) != settings['search']['starts']
                or {r.get('start') for r in row['starts']} != set(range(settings['search']['starts']))):
            raise ValueError('Oracle result belongs to a different operator or incomplete search')
        valid = []
        for start in row['starts']:
            if (start.get('manifest_digest') != digest
                    or start.get('operator_digest') != example.group_digest
                    or start.get('matrix_digest') != example.digest):
                raise ValueError('Oracle start lineage differs')
            candidates = start.get('candidates', [])
            successes = [c for c in candidates if c.get('timing', {}).get('successful') is True]
            for candidate in successes:
                timing = candidate['timing']
                seconds = timing.get('seconds')
                if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                        or not np.isfinite(seconds) or seconds <= 0
                        or timing.get('fallback_used') is not False):
                    raise ValueError('Oracle selection contains invalid or fallback-credited timing')
            best = min(successes, key=lambda c: c['timing']['seconds']) if successes else None
            if start.get('selected') != best:
                raise ValueError('Saved oracle start selection differs from measured candidates')
            if best is not None:
                valid.append(best)
        selected = min(valid, key=lambda c: c['timing']['seconds']) if valid else None
        if row.get('selected') != selected:
            raise ValueError('Saved oracle selection differs from measured starts')
        if selected is not None:
            path = (out / selected['path']).resolve()
            if out / 'search' not in path.parents or digest_file(path) != selected['sha256']:
                raise ValueError('Oracle checkpoint path/hash changed')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload.get('metadata', {}).get('oracle_binding') != dict(
                    operator_digest=example.digest, shape=[example.n, example.n]):
                raise ValueError('Oracle checkpoint belongs to another operator')
            records[example.group_digest] = {**selected, 'path': str(path)}
    if evidence.get('coverage') != len(records):
        raise ValueError('Oracle coverage differs from complete search evidence')
    return records


def search(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    _mutable(out)
    with _lease(out):
        examples, _, _ = _development(out, settings, baseline)
        source = _checkpoint(out, _source_stage(settings), baseline)
        spec = settings['search']
        manifest = dict(version=settings.get('version', VERSION), settings=spec, source=source.signature(),
            baseline_digest=baseline.digest(), operators=[e.group_digest for e in examples],
            target_rhs_count=settings['target_rhs_count'], training_only=True, fallback_allowed=False)
        if _p_only(settings):
            manifest['objective'] = settings['objective']
        path = out / 'search/manifest.json'
        if path.exists():
            if not resume or _read(path) != json_safe(manifest):
                raise ValueError('Search exists; identical --resume required')
        else:
            write_json(path, manifest)
        manifest_digest = _hash(manifest)

        def candidate_valid(candidate):
            timing = candidate.get('timing', {})
            seconds = timing.get('seconds')
            return (timing.get('successful') is True and timing.get('fallback_used') is False
                    and isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                    and np.isfinite(seconds) and seconds > 0)

        def best_candidate(candidates):
            valid = [c for c in candidates if candidate_valid(c)]
            return min(valid, key=lambda c: c['timing']['seconds']) if valid else None

        def verify_candidate(candidate, folder):
            label = candidate.get('label')
            if label not in ('initial', 'optimized'):
                raise ValueError('Unknown cached oracle candidate')
            candidate_path = folder / (label + '.pt')
            if (candidate.get('path') != str(candidate_path.relative_to(out))
                    or digest_file(candidate_path) != candidate.get('sha256')):
                raise ValueError('Cached oracle candidate path/hash changed')
            if candidate.get('timing', {}).get('successful') is True and not candidate_valid(candidate):
                raise ValueError('Cached oracle candidate has invalid success timing')

        def verify_start(record, identity, folder):
            if any(record.get(key) != value for key, value in identity.items()):
                raise ValueError('Cached oracle start belongs to a different protocol/operator')
            candidates = record.get('candidates', [])
            if len({c.get('label') for c in candidates}) != len(candidates):
                raise ValueError('Duplicate oracle candidate measurements')
            for candidate in candidates:
                verify_candidate(candidate, folder)
            if record.get('selected') != best_candidate(candidates):
                raise ValueError('Cached oracle selection differs from measured candidates')

        rows = []
        for index, example in enumerate(examples):
            directory = out / 'search' / example.group_digest
            result_path = directory / 'result.json'
            identity = dict(manifest_digest=manifest_digest, operator_digest=example.group_digest,
                            matrix_digest=example.digest)
            candidates = []
            for start in range(spec['starts']):
                folder = directory / f'start_{start}'
                result_file = folder / 'result.json'
                seed = settings['seed'] + index * 100003 + start * 1009
                start_identity = dict(identity, start=start, seed=seed)
                if result_file.exists():
                    record = _read(result_file)
                    verify_start(record, start_identity, folder)
                    candidates.append(record)
                    continue
                initial_path = folder / 'initial.pt'
                initial_measurement = folder / 'initial_measurement.json'
                statepath = folder / 'resume.pt'
                state = None
                if statepath.exists():
                    state = torch.load(statepath, map_location='cpu', weights_only=True)
                    if (state.get('identity') != start_identity or not initial_path.exists()
                            or type(state.get('updates')) is not int
                            or not 0 <= state['updates'] <= spec['updates']
                            or len(state.get('history', [])) != state['updates']):
                        raise ValueError('Oracle resume protocol/operator/history differs')
                saved_initial = None
                if initial_measurement.exists():
                    saved = _read(initial_measurement)
                    if saved.get('identity') != start_identity:
                        raise ValueError('Initial candidate timing belongs to a different search')
                    saved_initial = saved['candidate']
                    verify_candidate(saved_initial, folder)
                completed, history, measured = 0, [], []
                if saved_initial is not None:
                    measured.append(saved_initial)
                try:
                    # Save and measure the original candidate before any Adam
                    # update. A failed refinement must not erase this evidence.
                    if initial_path.exists():
                        initial = import_oracle(torch.load(initial_path, map_location='cpu', weights_only=True))
                    else:
                        initial = make_oracle(example, source, cfg,
                            initialization='jacobi' if start == 1 else 'network', seed=seed,
                            perturbation=spec['perturbation'] if start >= 2 else 0.)
                        _save_state(initial_path, export_oracle(initial))

                    def measure(label, candidate_path, candidate):
                        timing = measure_raw_candidate(example, candidate, cfg,
                            heldout_seed=settings['seed'] + 900000007 + index,
                            target_rhs_count=settings['target_rhs_count'], cycles=settings['raw_cycles'],
                            repeats=spec['repeats'], probe_model=source,
                            hard_smoothing=settings['training'].get('hard_smoothing', 4),
                            hard_cycles=settings['training'].get('hard_cycles', 4))
                        return dict(label=label, timing=timing,
                            path=str(candidate_path.relative_to(out)), sha256=digest_file(candidate_path))

                    if saved_initial is None:
                        saved_initial = measure('initial', initial_path, initial)
                        measured.append(saved_initial)
                        write_json(initial_measurement, dict(identity=start_identity, candidate=saved_initial))
                    oracle = import_oracle(state['oracle']) if state is not None else initial
                    names = ('transfer',) if _p_only(settings) else ('transfer', 'smoother')
                    params = [p for key in names for p in getattr(oracle, key).parameters()]
                    optimizer = torch.optim.Adam(params, lr=spec['learning_rate'])
                    if state is not None:
                        optimizer.load_state_dict(state['optimizer'])
                        completed, history = state['updates'], state['history']
                    for update in range(completed, spec['updates']):
                        optimizer.zero_grad(set_to_none=True)
                        if _p_only(settings):
                            loss, metrics = frobenius_objective(example, oracle, cfg,
                                probes=settings['objective']['probes'], seed=seed+update*104729)
                        else:
                            loss, metrics = joint_objective(example, oracle, cfg, cycles=spec['cycles'],
                                seed=seed + update * 104729,
                                probes_per_kind=settings['training']['probes_per_kind'],
                                hard_smoothing=settings['training']['hard_smoothing'],
                                hard_cycles=settings['training']['hard_cycles'])
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('Nonfinite offline search loss')
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, settings['training']['gradient_clip'], error_if_nonfinite=True)
                        optimizer.step()
                        if not all(bool(torch.isfinite(p).all()) for p in params):
                            raise FloatingPointError('Nonfinite operator after Adam update')
                        history.append(dict(update=update+1, loss=float(loss.detach())))
                        _save_state(statepath, dict(identity=start_identity, updates=update+1,
                            oracle=export_oracle(oracle), optimizer=optimizer.state_dict(), history=history))
                        if (update + 1) % 10 == 0 or update + 1 == spec['updates']:
                            print(f'[SEARCH {index+1}/{len(examples)} start {start+1}/{spec["starts"]}] '
                                  f'{update+1}/{spec["updates"]} loss={float(loss.detach()):.6g}', flush=True)
                    final_path = folder / 'optimized.pt'
                    _save_state(final_path, export_oracle(oracle))
                    measured.append(measure('optimized', final_path, oracle))
                    record = dict(start_identity, status='complete', updates=spec['updates'],
                                  selected=best_candidate(measured), candidates=measured)
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                    record = dict(start_identity, status='failed', updates=len(history),
                        selected=best_candidate(measured), candidates=measured,
                        reason=f'{type(error).__name__}: {error}')
                write_json(result_file, record)
                candidates.append(record)
            selected = best_candidate([c['selected'] for c in candidates if c.get('selected') is not None])
            row = dict(identity, case=example.name, selected=selected, starts=candidates)
            if result_path.exists() and _read(result_path) != json_safe(row):
                raise ValueError('Cached operator result differs from verified starts')
            write_json(result_path, row)
            rows.append(row)
            write_json(out / 'search/progress.json', dict(status='running', completed=len(rows), total=len(examples)))
        write_json(out / 'search/report.json', dict(status='complete', manifest_digest=manifest_digest,
            training_only=True, fallback_allowed=False,
            selection='held-out probes and measured setup+solve at fixed tolerance',
            coverage=sum(r['selected'] is not None for r in rows), total=len(rows), rows=rows))
        write_json(out / 'search/progress.json', dict(status='complete', completed=len(rows), total=len(examples)))


def distill(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    _mutable(out)
    with _lease(out):
        examples, _, _ = _development(out, settings, baseline)
        teachers = _oracle_records(out)
        name = 'distilled_P' if _p_only(settings) else 'distilled_joint'
        _run_training(out, name, _checkpoint(out, _source_stage(settings), baseline), examples,
                      cfg, baseline, settings, resume=resume, teachers=teachers)


def diagnose(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if not _p_only(settings):
        raise ValueError('Fixed Frobenius diagnostics require a new P-only run')
    training, validation, _ = _development(out, settings, baseline)
    initial = _initial(out)
    names = [name for name in _stages(settings)
             if (out / 'checkpoints' / name / 'candidate.pt').exists()]
    if not names:
        raise ValueError('Train the P generator before running diagnose')
    models = {name: _checkpoint(out, name, baseline) for name in names}
    manifest = dict(version=P_VERSION, objective=settings['objective'],
        models={name: model.signature() for name, model in models.items()}, initial=initial.signature(),
        config=cfg.to_dict(), rhs_count=settings['target_rhs_count'], repeats=3, warmups=1)
    path = out / 'diagnostics/p_frobenius.json'
    if resume and path.exists():
        result = _read(path)
        if result['manifest'] != json_safe(manifest):
            raise ValueError('Diagnostic protocol/weights changed; rerun without --resume')
        return result
    rows = []
    for example in training + validation:
        for name, model in models.items():
            row = p_diagnostic(example, initial, model, cfg,
                probes=settings['objective']['diagnostic_probes'],
                seed=settings['seed']+800000011+int(example.group_digest[:8], 16),
                rhs_count=settings['target_rhs_count'], repeats=manifest['repeats'], warmups=manifest['warmups'])
            rows.append(dict(row, arm=name))
    result = json_safe(dict(manifest=manifest, training=False, checkpoint_selection=False,
        scope='same A and frozen error vectors for before/after; TRAIN+validation only', rows=rows))
    write_json(path, result)
    for name in models:
        group = [r for r in rows if r['arm'] == name and r['split'] == 'validation']
        print(f'[P DIAG {name}] Frobenius improved '
              f'{sum(r["frobenius_improved"] is True for r in group)}/{len(group)}; '
              f'time improved {sum(r["time_improved"] is True for r in group)}/{len(group)}', flush=True)
    return result


def _arms(out, settings, cfg, baseline):
    emcfg = replace(cfg, mg=replace(cfg.mg, strategy_name=FIXED_STRATEGY_NAME,
        em_iterations=5, em_fixed_iterations=True, em_rtol=1e-8))
    arms = {'fixed_C': dict(model=None, branch='C', selector=False),
            'em5_C': dict(model=None, branch='C', selector=False, reference_only=True,
                          config=emcfg, rules=FixedClassicalBaseline.from_config(emcfg))}
    for name, (branch, _) in _stages(settings).items():
        arms[name] = dict(model=_checkpoint(out, name, baseline), branch=branch, selector=False)
        if _p_only(settings):
            require_p_only(arms[name]['model'], cfg)
    return arms


def benchmark(output, *, resume=False, **protocol):
    protocol = _validate_protocol(protocol)
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if 'multiple' not in protocol['regimes'] or settings['target_rhs_count'] not in protocol['rhs_counts']:
        raise ValueError('Include multiple and the declared target_rhs_count for setup-inclusive selection')
    _, examples, _ = _development(out, settings, baseline)
    arms = _arms(out, settings, cfg, baseline)
    target = out / 'benchmarks/validation'
    result = evaluate_research(examples, arms, cfg, baseline, target,
        reference_arm='fixed_C', resume=resume, **protocol)
    report(result, target)
    write_json(target / 'raw_contractions.json', raw_contractions(examples, arms, cfg,
        cycles=settings['raw_cycles'], probes=settings['raw_probes']))
    selection = select_measured_candidate(result, rhs_count=settings['target_rhs_count'],
        candidate_arms=tuple(_stages(settings)))
    write_json(target / 'selection.json', selection)
    files = {key: dict(path=str(target / key), sha256=digest_file(target / key))
        for key in ('raw_results.json', 'cycle_diagnostics.json', 'raw_contractions.json', 'selection.json')}
    write_json(out / 'benchmark_evidence.json', dict(files=files,
        checkpoints={n: a['model'].signature() for n, a in arms.items() if a['model']},
        selection=selection, performance_certified=False))
    return result


def freeze(output, **protocol):
    protocol = _validate_protocol(protocol, final=True)
    out, settings, cfg, baseline = load(output)
    _ensure_development_open(out)
    if 'multiple' not in protocol['regimes'] or settings['target_rhs_count'] not in protocol['rhs_counts']:
        raise ValueError('Final protocol must include the setup-inclusive target RHS workload')
    arms = _arms(out, settings, cfg, baseline)
    evidence = _read(out / 'benchmark_evidence.json')
    if evidence['checkpoints'] != {n: a['model'].signature() for n, a in arms.items() if a['model']}:
        raise ValueError('Checkpoint changed after validation')
    files = {}
    for key, record in evidence['files'].items():
        if digest_file(record['path']) != record['sha256']:
            raise ValueError('Validation evidence changed')
        files[key] = record['path']
    files.update(configuration=out / 'configuration.json', manifest=out / 'joint_manifest.json',
        evidence=out / 'benchmark_evidence.json', search=out / 'search/report.json')
    if (out / 'diagnostics/p_frobenius.json').exists():
        files['p_diagnostic'] = out / 'diagnostics/p_frobenius.json'
    checkpoints = {name: out / 'checkpoints' / name / 'candidate.pt' for name in _stages(settings)}
    evaluation = dict(version=settings['version'], solver=cfg.to_dict(), arms=list(arms),
        target_rhs_count=settings['target_rhs_count'], **protocol)
    return freeze_research(out / 'data', checkpoints, out / 'fixed_classical.json', files, _sources(), evaluation)


def final(output, *, resume=False):
    out, settings, cfg, baseline = load(output)
    path = out / 'data/research_freeze.json'
    evaluation = _read(path)['config']
    arms = _arms(out, settings, cfg, baseline)
    if (evaluation['version'] != settings['version'] or evaluation['solver'] != json_safe(cfg.to_dict())
            or evaluation['arms'] != list(arms) or evaluation['target_rhs_count'] != settings['target_rhs_count']):
        raise ValueError('Frozen joint evaluation changed')
    with claim_final_evaluation(out / 'data', path, evaluation, resume=resume) as claim:
        splits, _ = materialize_final_data(out / 'data', claim, baseline)
        outputs = []
        for name in evaluation['splits']:
            target = out / 'final' / name
            result = evaluate_research(splits[name], arms, cfg, baseline, target, reference_arm='fixed_C',
                resume=resume and (target / 'run_manifest.json').exists(),
                **{key: evaluation[key] for key in ('repeats', 'warmups', 'regimes', 'rhs_counts')})
            report(result, target)
            outputs.extend(target / key for key in ('raw_results.json', 'cycle_diagnostics.json'))
        complete_final_evaluation(out / 'data', claim, outputs)
    print('Final/OOD joint study evaluation sealed.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'train', 'diagnose', 'search', 'distill', 'benchmark', 'freeze', 'final'):
        p = sub.add_parser(name)
        p.add_argument('--run-dir', required=True)
        if name != 'freeze':
            p.add_argument('--resume', action='store_true')
        if name == 'prepare':
            p.add_argument('--config', default='configs/v6_7_learned_p_research.json')
        if name in ('benchmark', 'freeze'):
            p.add_argument('--repeats', type=int, default=3)
            p.add_argument('--warmups', type=int, default=1)
            p.add_argument('--rhs-counts', nargs='+', type=int, default=[1, 4, 16, 64])
            p.add_argument('--regimes', nargs='+', choices=('cold', 'warm', 'multiple', 'warm_multiple'),
                           default=['cold', 'warm', 'multiple'])
        if name == 'freeze':
            p.add_argument('--splits', nargs='+', choices=HOLDOUT, default=list(HOLDOUT))
    a = parser.parse_args(argv)
    if a.command == 'prepare':
        return prepare(a.config, a.run_dir, resume=a.resume)
    if a.command in ('train', 'diagnose', 'search', 'distill', 'final'):
        return globals()[a.command](a.run_dir, resume=a.resume)
    protocol = dict(repeats=a.repeats, warmups=a.warmups, rhs_counts=a.rhs_counts, regimes=a.regimes)
    if a.command == 'freeze':
        return freeze(a.run_dir, splits=a.splits, **protocol)
    return benchmark(a.run_dir, resume=a.resume, **protocol)
