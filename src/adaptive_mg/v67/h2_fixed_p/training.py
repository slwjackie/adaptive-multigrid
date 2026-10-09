"""Train only H_S on recorded CFD residuals with per-trajectory classical P.

The first matrix of EACH trajectory builds interpolation once. Every training
sample uses those exact P values and recomputes A_c = P.T @ A_current @ P at
all levels. There are no random right-hand sides or manufactured solutions.
This module never labels successful training as CFD validation or acceleration.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
import hashlib
import numpy as np
import torch
from torch.nn import functional as F

from ...hierarchy import build_fixed_hierarchy
from ...provenance import operator_digest
from ...strategy import get_strategy
from ...transfer import galerkin_coarse_operator, matrix_feature_array
from ..autograd_sparse import SparseTensor
from ..banks import Stats, resolve_device, selected_level
from ..models import Components
from ..unroll import TLevel, cycle
from ..world_model.backend import hierarchy_signature, levels
from ..world_model.data import digest, file_hash, write_json

FIXED_P_CONTRACT = 'h2-fixed-classical-p-v1'


def _vector_digest(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _freeze_p(root):
    for level in levels(root):
        if level.p is not None:
            for array in (level.p.data, level.p.indices, level.p.indptr):
                array.flags.writeable = False
    return root


def fixed_p_graph(a, reference, model, cfg, *, learned=True):
    """Differentiable smoother overlay; reference P never rebuilt or trained.

    Runtime uses the same ``matrix_feature_array`` and classical Galerkin
    function. A and P are constants in this graph; only smoother outputs carry
    gradients. The classical graph is also used as a same-P loss reference.
    """
    if cfg.use_transfer:
        raise ValueError('fixed-P H_S training forbids learned transfer')
    device = resolve_device(cfg, cells=a.shape[0])
    dtype = torch.float32 if cfg.inference_dtype == 'float32' else torch.float64
    model.smoother.to(device=device, dtype=dtype)

    def build(current, parent):
        current = current.tocsr(copy=True)
        level = TLevel(SparseTensor.from_scipy(current), parent.shape,
                       parent.index, parent.strategy, raw_scipy=current)
        if parent.coarse is None:
            return level
        if parent.p is None or parent.p.shape[0] != current.shape[0]:
            raise ValueError('incompatible fixed interpolation hierarchy')
        level.p = SparseTensor.from_scipy(parent.p)
        level.pattern = parent.pattern
        level.base_weights = parent.base_weights
        level.interpolation_weights = torch.tensor(parent.base_weights, dtype=torch.float64)
        if learned and cfg.use_smoother and selected_level(parent.index, cfg, 'smoother'):
            features = torch.tensor(matrix_feature_array(current, parent.shape),
                                    device=device, dtype=dtype).unsqueeze(0)
            cascade = hasattr(model.smoother, 'stage_directions_and_gains')
            directions, gains = (model.smoother.stage_directions_and_gains(features, parent.index)
                                 if cascade else model.smoother.direction_and_gain(features))
            level.is_residual_cascade = cascade
            level.dirs = directions.to('cpu', dtype=torch.float64)
            level.gains = gains.to('cpu', dtype=torch.float64) * cfg.mg.smoother_gain_multiplier
        level.coarse = build(galerkin_coarse_operator(current, parent.p), parent.coarse)
        return level

    return build(a, reference)


def residual_objective(state, reference, model, cfg, *, cycles=2,
                       classical_prefix_cycles=0, lambda_stability=.1,
                       lambda_noharm=.1):
    """Use actual exported b/x0 and real residuals along unrolled V-cycles."""
    learned = fixed_p_graph(state.a, reference, model, cfg, learned=True)
    classical = fixed_p_graph(state.a, reference, model, cfg, learned=False)
    b = torch.tensor(state.b, dtype=torch.float64)
    x = torch.tensor(state.x0, dtype=torch.float64)
    with torch.no_grad():
        for step in range(classical_prefix_cycles):
            x = cycle(classical, x, b, model, cfg, step)
    initial = b - learned.a.apply(x)
    norm0 = torch.linalg.vector_norm(initial).clamp_min(1e-100)
    initial_x = x.clone()
    history = []
    for step in range(cycles):
        x = cycle(learned, x, b, model, cfg, step)
        history.append(torch.linalg.vector_norm(b - learned.a.apply(x)) / norm0)
    h = torch.stack(history)
    weights = torch.arange(1, cycles + 1, dtype=torch.float64)
    weights /= weights.sum()
    residual_loss = (weights * h.clamp_min(1e-100).log()).sum()
    previous = torch.cat((h.new_ones(1), h[:-1]))
    stability = F.relu(h / previous.clamp_min(1e-100) - 1).square().mean()
    with torch.no_grad():
        xc = initial_x
        classical_history = []
        for step in range(cycles):
            xc = cycle(classical, xc, b, model, cfg, step)
            classical_history.append(torch.linalg.vector_norm(b - classical.a.apply(xc)) / norm0)
        hc = torch.stack(classical_history)
    noharm = (weights * F.relu(h.clamp_min(1e-100).log() - hc.clamp_min(1e-100).log()).square()).sum()
    loss = residual_loss + lambda_stability * stability + lambda_noharm * noharm
    details = dict(residual_loss=float(residual_loss.detach()), stability=float(stability.detach()),
                   noharm=float(noharm.detach()), relative_residuals=h.detach().tolist(),
                   classical_relative_residuals=hc.tolist(),
                   initial_residual_norm=float(norm0.detach()),
                   initial_residual_digest=_vector_digest(initial.detach().numpy()),
                   effective_x0_digest=_vector_digest(initial_x.numpy()),
                   fixed_p_digest=hierarchy_signature(reference),
                   coarse_operator_digests=[operator_digest(l.raw_scipy)
                                            for l in _graph_levels(learned)][1:])
    return loss, details, x


def _graph_levels(root):
    while root is not None:
        yield root
        root = root.coarse


def _positive_integer(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(name + ' must be a ' + ('nonnegative' if zero else 'positive') + ' integer')


def train_smoother(trajectories, cfg, output, *, updates=100, seed=20261010,
                   hidden=16, learning_rate=1e-3, cycles=2,
                   classical_prefix_cycles=0, lambda_stability=.1, lambda_noharm=.1):
    """Return a genuinely updated H_S candidate and auditable training status.

    Inputs are validated ``(trajectory_metadata, list[Snapshot])`` pairs. Only
    train/external_cfd data are admitted; provenance is not a claim that this
    function has independently validated the producer's combustion physics.
    Output must be a fresh directory. Any inactive/nonfinite gradient fails
    training rather than emitting a zero-update 'trained' checkpoint.
    """
    for name, value in [('updates', updates), ('hidden', hidden), ('cycles', cycles)]:
        _positive_integer(value, name)
    _positive_integer(classical_prefix_cycles, 'classical_prefix_cycles', zero=True)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not np.isfinite(lambda_stability + lambda_noharm) or min(lambda_stability, lambda_noharm) < 0:
        raise ValueError('loss weights must be finite and nonnegative')
    if cfg.use_transfer or cfg.application != 'replace':
        raise ValueError('fixed-P H_S training requires replacement and use_transfer=False')
    if cfg.replace_pre + cfg.replace_post < 1 or cfg.mg.smoother_gain_multiplier <= 0:
        raise ValueError('at least one active neural replacement is required')
    cfg = replace(cfg, mode='research', branch='H_S', use_smoother=True,
                  use_transfer=False, spatial=False, gate_mode='open', use_learned_controller=False)
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError('nonempty TRAIN trajectories required')
    samples, trajectory_records, ids = [], [], set()
    for metadata, states in trajectories:
        if metadata.get('split') != 'train':
            raise ValueError('only TRAIN trajectories may optimize the smoother')
        ident = metadata.get('id')
        if not ident or ident in ids or not metadata.get('case_group'):
            raise ValueError('unique trajectory id and physical case_group required')
        ids.add(ident)
        if not states:
            raise ValueError('empty training trajectory')
        if any(s.source_kind != 'external_cfd' for s in states):
            raise ValueError('recorded external_cfd snapshots required; synthetic RHS are forbidden')
        for j, state in enumerate(states):
            state.validate()
            if state.topology_key != states[0].topology_key:
                raise ValueError('fixed-P training requires unchanged mesh/boundary/sparsity within a trajectory')
            if j and (state.index <= states[j - 1].index or state.time < states[j - 1].time):
                raise ValueError('training snapshots must preserve chronological solve order')
        reference = _freeze_p(build_fixed_hierarchy(states[0].a, states[0].shape,
                                                   get_strategy(cfg.mg.strategy_name), cfg.mg, Stats()))
        if not any(l.p is not None and selected_level(l.index, cfg, 'smoother') for l in levels(reference)):
            raise ValueError('training hierarchy has no active nonterminal smoother level')
        p_digest = hierarchy_signature(reference)
        record = dict(id=ident, case_group=metadata['case_group'], split='train',
                      fixed_p_digest=p_digest, initial_matrix_digest=states[0].matrix_digest,
                      reference_builds=1, samples=[])
        for state in states:
            r0 = state.b - state.a @ state.x0
            threshold = max(cfg.mg.absolute_tolerance, cfg.mg.tolerance *
                            (np.linalg.norm(r0) if cfg.mg.residual_reference == 'initial' else np.linalg.norm(state.b)))
            entry = dict(index=state.index, time=state.time, matrix_digest=state.matrix_digest,
                         rhs_digest=_vector_digest(state.b), x0_digest=_vector_digest(state.x0),
                         residual_digest=_vector_digest(r0), initial_residual_norm=float(np.linalg.norm(r0)),
                         source_kind=state.source_kind, fixed_p_digest=p_digest,
                         eligible=bool(np.linalg.norm(r0) > threshold))
            record['samples'].append(entry)
            if entry['eligible']:
                samples.append((ident, state, reference, entry))
        trajectory_records.append(record)
    if not samples:
        raise ValueError('no unconverged recorded CFD residuals available for training')

    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('training output must be a fresh directory')
    out.mkdir(parents=True, exist_ok=True)
    model = Components.create(hidden=hidden, seed=seed)
    for name in ('smoother', 'transfer', 'detector', 'controller'):
        net = getattr(model, name)
        net.train(name == 'smoother')
        for parameter in net.parameters():
            parameter.requires_grad_(name == 'smoother')
    before = model.component_signatures()
    model.save(out / 'initial.pt', extra={'role': 'untrained_initialization'})
    # Move before optimizer creation: offline FP64 numerical operations stay CPU.
    device = resolve_device(cfg, cells=max(s.a.shape[0] for _, s, _, _ in samples))
    dtype = torch.float32 if cfg.inference_dtype == 'float32' else torch.float64
    model.smoother.to(device=device, dtype=dtype)
    # Keep optimizer state and parameters on one device across mixed grid sizes.
    # Runtime may still use its own declared 'auto' generation threshold.
    cfg = replace(cfg, inference_device=str(device))
    params = list(model.smoother.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    rng = np.random.default_rng(seed)
    schedule = []
    while len(schedule) < updates:
        schedule.extend(rng.permutation(len(samples)).tolist())
    schedule = schedule[:updates]
    manifest = dict(schema='h2-fixed-p-smoother-training-v1', contract=FIXED_P_CONTRACT,
                    config=cfg.to_dict(), seed=seed, hidden=hidden, updates=updates,
                    learning_rate=learning_rate, cycles=cycles, classical_prefix_cycles=classical_prefix_cycles,
                    lambda_stability=lambda_stability, lambda_noharm=lambda_noharm,
                    trajectories=trajectory_records,
                    schedule=[{'trajectory': samples[i][0], 'index': samples[i][1].index} for i in schedule],
                    actual_exported_rhs=True, actual_exported_x0=True, random_rhs_generated=False,
                    interpolation_rebuilt_per_sample=False, learned_transfer=False,
                    initial_signatures=before, final_test_seen=False)
    fingerprint = digest(manifest)
    manifest['training_fingerprint'] = fingerprint
    write_json(out / 'training_manifest.json', manifest)
    start = perf_counter()
    records = []
    for step, sample_index in enumerate(schedule):
        ident, state, reference, entry = samples[sample_index]
        optimizer.zero_grad(set_to_none=True)
        loss, details, _ = residual_objective(state, reference, model, cfg,
                    cycles=cycles, classical_prefix_cycles=classical_prefix_cycles,
                    lambda_stability=lambda_stability, lambda_noharm=lambda_noharm)
        if not loss.requires_grad or not torch.isfinite(loss):
            raise FloatingPointError('inactive or nonfinite smoother objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=2.)
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise FloatingPointError('zero or nonfinite gradient; no trained checkpoint emitted')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in params):
            raise FloatingPointError('nonfinite parameters after smoother update')
        if hierarchy_signature(reference) != entry['fixed_p_digest']:
            raise RuntimeError('fixed interpolation mutated during training')
        records.append(dict(step=step, trajectory=ident, snapshot_index=state.index,
                            matrix_digest=state.matrix_digest, rhs_digest=entry['rhs_digest'],
                            exported_x0_digest=entry['x0_digest'], loss=float(loss.detach()),
                            gradient_norm=float(norm), **details))
        write_json(out / 'training.json', records)
    model.eval()
    for net in model.modules():
        net.cpu()
    after = model.component_signatures()
    if after['smoother'] == before['smoother']:
        raise RuntimeError('smoother weights did not change; no trained checkpoint emitted')
    if any(after[name] != before[name] for name in ('transfer', 'detector', 'controller')):
        raise RuntimeError('a frozen non-smoother component changed')
    model.metadata.update(training_branch='H_S', smoother_trained=True, transfer_trained=False,
                          optimizer_updates=len(records), requested_optimizer_updates=updates,
                          training_kind='recorded_cfd_fixed_p_multicycle_residual',
                          fixed_p_contract=FIXED_P_CONTRACT, training_fingerprint=fingerprint,
                          training_case_groups=sorted({m['case_group'] for m, _ in trajectories}),
                          training_operator_digests=sorted({s.matrix_digest for _, states in trajectories for s in states}),
                          training_source_kind='external_cfd', learned_transfer=False,
                          actual_exported_rhs=True, actual_exported_x0=True, final_test_seen=False)
    model.mark_policy_stale('fixed_P_H_S_candidate_requires_independent_timing_and_CFD_validation')
    model.save(out / 'candidate.pt', extra={'role': 'unselected_H_S_candidate', 'performance_certified': False})
    status = dict(status='complete', training_branch='H_S', optimizer_updates=len(records),
                  updates=len(records), requested_updates=updates, training_fingerprint=fingerprint,
                  fixed_p_contract=FIXED_P_CONTRACT, initial_signatures=before, final_signatures=after,
                  checkpoint=str((out / 'candidate.pt').resolve()), checkpoint_sha256=file_hash(out / 'candidate.pt'),
                  training_seconds=perf_counter() - start, eligible_samples=len(samples),
                  trajectories=len(trajectory_records), final_test_seen=False, performance_certified=False,
                  cfd_validation_completed=False, world_model_trained=False)
    write_json(out / 'status.json', status)
    return model, status
