"""Train-only supervised fits for the two lightweight smoother predictors.

Both models consume exactly the same offline-search rows and standardizer.  The
MLP optimizes coefficient labels, not a V-cycle, and never receives or modifies
the shared P model.  Adam state is committed after every update so interruption
and resume preserve the numerical training trajectory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..provenance import json_safe
from .joint_transfer_study import _save_state
from .limited import digest_file
from .light_smoother_models import (FEATURE_VERSION, Standardizer, TinyMLP,
    MLPPredictor, fit_ridge, load_predictor, save_predictor)
from .research_data import _hash, _write_json

VERSION = 'light-smoother-supervised-fit-v1'
DEFAULT_SETTINGS = dict(updates=500, batch_size=64, learning_rate=.001,
    gradient_clip=2., ridge_lambda=.001, seed=1, bounds=[.8, 1.2])


def validate_fit_settings(settings):
    """Return the explicit, JSON-safe training protocol, rejecting unknown keys."""
    if not isinstance(settings, dict) or set(settings) - set(DEFAULT_SETTINGS):
        raise ValueError('Unknown lightweight fitting settings')
    spec = dict(DEFAULT_SETTINGS, **settings)
    for key in ('updates', 'batch_size'):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f'fit.{key} must be a positive integer')
    if type(spec['seed']) is not int or spec['seed'] < 0:
        raise ValueError('fit.seed must be a nonnegative integer')
    for key in ('learning_rate', 'gradient_clip'):
        if isinstance(spec[key], bool) or not np.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f'fit.{key} must be finite and positive')
    if (isinstance(spec['ridge_lambda'], bool) or not np.isfinite(spec['ridge_lambda'])
            or spec['ridge_lambda'] < 0):
        raise ValueError('fit.ridge_lambda must be finite and nonnegative')
    bounds = np.asarray(spec['bounds'], dtype=np.float64)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < 1 < bounds[1]:
        raise ValueError('Bounds must be positive and contain 1 strictly inside')
    spec['bounds'] = bounds.tolist()
    return json_safe(spec)


def _read(path):
    return json.loads(Path(path).read_text())


def _array_digest(array):
    canonical = np.ascontiguousarray(array, dtype='<f8')
    return _hash(dict(shape=list(canonical.shape), dtype='<f8',
        sha256=hashlib.sha256(canonical.tobytes()).hexdigest()))


def _data(x, y, spec, provenance):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != 12 or not len(x) or y.shape != (len(x), 2)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Require finite, nonempty feature (rows,12) and label (rows,2) arrays')
    if (y < spec['bounds'][0]).any() or (y > spec['bounds'][1]).any():
        raise ValueError('Offline-search labels are outside the shared coefficient bounds')
    if not isinstance(provenance, dict) or provenance.get('split') != 'train':
        raise ValueError('Only explicitly train-labelled offline-search rows may be fitted')
    # Conversion rejects unsupported/nonfinite provenance before writing a run.
    _hash(provenance)
    return x, y


def _manifest(x, y, spec, provenance, scaler):
    return json_safe(dict(version=VERSION, settings=spec, feature_version=FEATURE_VERSION,
        rows=len(x), input_digest=_array_digest(x), target_digest=_array_digest(y),
        provenance=provenance, scaler=scaler.to_dict(),
        objective='supervised squared coefficient error; P and all MG operators fixed',
        shared_training_rows=True, normalization_split='train'))


def _status(out, manifest, updates, offline_seconds, *, complete=False, losses=None):
    status = dict(version=VERSION, manifest_digest=_hash(manifest),
        status='complete' if complete else 'running', updates=updates,
        total_updates=manifest['settings']['updates'], all_updates_completed=complete,
        offline_fit_seconds=float(offline_seconds),
        timing_scope='offline fitting only; excluded from online setup/solve measurements')
    if complete:
        status.update(predictors={name: dict(path=f'{name}.json',
                          sha256=digest_file(out/f'{name}.json'))
                          for name in ('mlp', 'regression')},
            parameter_counts=dict(mlp=242, regression=26),
            coefficient_mse=losses, history_sha256=digest_file(out/'history.json'))
    _write_json(out/'status.json', status)
    return status


def load_fitted(output, expected_manifest=None):
    """Verify and restore a completed pair; never silently accept partial fits."""
    out = Path(output)
    manifest, status = _read(out/'fit_manifest.json'), _read(out/'status.json')
    spec = validate_fit_settings(manifest['settings'])
    if (manifest.get('version') != VERSION or manifest.get('feature_version') != FEATURE_VERSION
            or manifest.get('normalization_split') != 'train'
            or manifest.get('provenance', {}).get('split') != 'train'
            or status.get('version') != VERSION or status.get('status') != 'complete'
            or not status.get('all_updates_completed')
            or status.get('updates') != spec['updates']
            or status.get('total_updates') != spec['updates']
            or status.get('manifest_digest') != _hash(manifest)
            or (expected_manifest is not None and manifest != json_safe(expected_manifest))):
        raise ValueError('Incomplete or changed lightweight fit manifest/status')
    if status.get('history_sha256') != digest_file(out/'history.json'):
        raise ValueError('Lightweight training history changed')
    history = _read(out/'history.json')
    if len(history) != spec['updates'] or [r['update'] for r in history] != list(range(1, spec['updates']+1)):
        raise ValueError('Lightweight training history is incomplete')
    models = {}
    for name in ('mlp', 'regression'):
        record = status.get('predictors', {}).get(name, {})
        path = out/f'{name}.json'
        if record.get('path') != path.name or record.get('sha256') != digest_file(path):
            raise ValueError(f'Fitted {name} checkpoint changed')
        model = load_predictor(path)
        if (model.kind != name or model.metadata.get('fit_manifest_digest') != _hash(manifest)
                or model.metadata.get('split') != 'train'
                or model.scaler.to_dict() != manifest['scaler']
                or list(model.bounds) != spec['bounds']):
            raise ValueError(f'Fitted {name} lineage or standardization changed')
        models[name] = model
    return models


def fit_models(x, y, output, settings, provenance, *, resume=False):
    """Fit the 242-parameter MLP and 26-coefficient ridge model to shared labels.

    ``provenance['split']`` must equal ``'train'``.  The caller owns the run-wide
    mutation/freeze lease; this function independently verifies its immutable
    inputs, settings, provenance, state and saved predictor files.
    """
    started = perf_counter()
    spec = validate_fit_settings(settings)
    x, y = _data(x, y, spec, provenance)
    scaler = Standardizer.fit(x, split='train')
    manifest = _manifest(x, y, spec, provenance, scaler)
    out = Path(output)
    manifest_path = out/'fit_manifest.json'
    if manifest_path.exists():
        if not resume or _read(manifest_path) != manifest:
            raise ValueError('Fitting exists; identical inputs/protocol and --resume required')
        if (out/'status.json').exists() and _read(out/'status.json').get('status') == 'complete':
            return load_fitted(out, manifest)
    else:
        if resume:
            raise FileNotFoundError('No lightweight fit to resume')
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Fitting requires an empty output directory')
        _write_json(manifest_path, manifest)

    # Initialization cannot change the caller's global RNG state. Batches are
    # independently seeded by update number, so no implicit RNG state is lost.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec['seed'])
        model = TinyMLP(bounds=spec['bounds'])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=spec['learning_rate'])
    state_path = out/'resume.pt'
    history, completed, prior_seconds = [], 0, 0.
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        if (state.get('version') != VERSION or state.get('manifest_digest') != _hash(manifest)
                or type(state.get('updates')) is not int
                or not 0 <= state['updates'] <= spec['updates']
                or len(state.get('history', [])) != state['updates']
                or not np.isfinite(state.get('offline_fit_seconds', np.nan))
                or state['offline_fit_seconds'] < 0):
            raise ValueError('Invalid or mismatched lightweight fit resume state')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history, completed = state['history'], state['updates']
        if [r['update'] for r in history] != list(range(1, completed+1)):
            raise ValueError('Fit resume update history is inconsistent')
        prior_seconds = state['offline_fit_seconds']
    elif (out/'status.json').exists() and _read(out/'status.json').get('updates', 0) > 0:
        raise ValueError('Interrupted fit is missing its Adam resume state')

    z = torch.as_tensor(scaler.transform(x), dtype=torch.float64)
    targets = torch.as_tensor(y, dtype=torch.float64)

    def checkpoint():
        elapsed = prior_seconds + perf_counter()-started
        _save_state(state_path, dict(version=VERSION, manifest_digest=_hash(manifest),
            updates=completed, model=model.state_dict(), optimizer=optimizer.state_dict(),
            history=history, offline_fit_seconds=elapsed))
        _status(out, manifest, completed, elapsed)

    if not state_path.exists():
        checkpoint()
    try:
        for update in range(completed, spec['updates']):
            rng = np.random.default_rng(np.random.SeedSequence([spec['seed'], update]))
            indices = rng.permutation(len(x))[:min(spec['batch_size'], len(x))]
            index = torch.as_tensor(indices, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(z[index])-targets[index])**2)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite supervised MLP loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'])
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError('Nonfinite supervised MLP gradient')
            optimizer.step()
            history.append(dict(update=update+1, coefficient_mse=float(loss.detach()),
                                gradient_norm=float(gradient), batch_rows=indices.tolist()))
            completed = update+1
            checkpoint()
            if completed == 1 or completed == spec['updates'] or completed % 100 == 0:
                print(f'[FIT light MLP] {completed}/{spec["updates"]} coefficient_mse={float(loss.detach()):.6g}', flush=True)
        model.eval()
        metadata = dict(fit_manifest_digest=_hash(manifest), split='train',
            feature_version=FEATURE_VERSION, rows=len(x), input_digest=manifest['input_digest'],
            target_digest=manifest['target_digest'], shared_label_data=True)
        mlp = MLPPredictor(model, scaler, metadata=dict(metadata))
        regression = fit_ridge(x, y, bounds=spec['bounds'], regularization=spec['ridge_lambda'], scaler=scaler)
        regression.metadata = dict(metadata)
        save_predictor(mlp, out/'mlp.json')
        save_predictor(regression, out/'regression.json')
        _write_json(out/'history.json', history)
        losses = {name: float(np.mean((np.asarray(predictor.predict(x))-y)**2))
                  for name, predictor in (('mlp', mlp), ('regression', regression))}
        _status(out, manifest, completed, prior_seconds+perf_counter()-started,
                complete=True, losses=losses)
    except BaseException:
        # A signal can arrive *inside* Adam.step or checkpoint serialization.
        # Never persist potentially half-updated in-memory parameters. Reload
        # the last atomic state, update timing only and resume that full step.
        durable = torch.load(state_path, map_location='cpu', weights_only=True)
        durable['offline_fit_seconds'] = prior_seconds+perf_counter()-started
        _save_state(state_path, durable)
        _status(out, manifest, durable['updates'], durable['offline_fit_seconds'])
        raise
    return load_fitted(out, manifest)
