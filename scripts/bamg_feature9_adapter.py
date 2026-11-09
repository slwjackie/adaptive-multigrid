"""Process-local opt-in integration; importing this module changes no globals.

The adapter deliberately lives outside src: adding this experiment must not
invalidate the source hashes of an already-running 10-channel study. Patches
are scoped to feature9_mode and restored on exceptions. Use one study per
process (the legacy training pipeline is not a concurrent-thread API).
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

from adaptive_mg.v67 import bamg_transfer as transfer
from adaptive_mg.v67 import bamg_solver as solver
from adaptive_mg.v67 import bamg_training as training
from adaptive_mg.v67 import bamg_study as legacy_study
from adaptive_mg.v67 import bamg_energy_cnn_study as paired
from adaptive_mg.v67 import bamg_energy_cnn_training as arm_training
from adaptive_mg.v67 import bamg_energy_cnn_epoch_training as epoch_training
from adaptive_mg.v67 import bamg_energy_cnn_benchmark as benchmark
from adaptive_mg.v67 import bamg_energy_cnn_holdout_benchmark as holdout
from adaptive_mg.v67 import bamg_energy_cnn_routing_benchmark as routing_benchmark
from adaptive_mg.v67 import bamg_energy_cnn_routing_holdout_benchmark as routing_holdout

from bamg_feature9 import (FEATURE_VERSION, DIAGONAL_FLOOR, LOG_EPSILON,
                          Feature9WeightCNN, direct_features9,
                          transfer_legacy_weights, validate_mode)

_original_model = transfer.model_from_spec
_original_weights = solver._weights
_original_objective = training.supervised_objective
_original_initialize = arm_training._initialize_weight
_original_resolve = paired.resolve_settings
_original_sources = paired._sources
_original_bank = benchmark._build_bank
_original_routing_bank = routing_benchmark._build_bank
_active = False


def model_from_spec(spec):
    if 'feature_mode' not in spec:
        return _original_model(spec)
    values = dict(spec)
    if values.pop('kind', None) != 'weight':
        raise ValueError('feature9 is only implemented for WeightCNN')
    return Feature9WeightCNN(**values)


def _weights(a, shape, pattern, mask, model, l1_cap, timings):
    if not isinstance(model, Feature9WeightCNN):
        return _original_weights(a, shape, pattern, mask, model, l1_cap, timings)
    device, dtype = solver._placement(model)
    solver._sync(model)
    start = perf_counter()
    features = direct_features9(a, shape, feature_mode=model.feature_mode, device=device, dtype=dtype)
    solver._sync(model)
    timings['feature_seconds'] += perf_counter() - start
    start = perf_counter()
    raw = model(features, mask.to(device))
    solver._sync(model)
    timings['weight_cnn_seconds'] += perf_counter() - start
    start = perf_counter()
    weights = transfer.project_weights(raw, mask.to(device), pattern, l1_cap=l1_cap)
    solver._sync(model)
    timings['projection_seconds'] += perf_counter() - start
    start = perf_counter()
    p = transfer.sparse_from_weights(pattern, weights, mask)
    timings['assembly_transfer_seconds'] += perf_counter() - start
    if not bool(torch.isfinite(p.values).all()):
        raise FloatingPointError('Nonfinite generated P')
    return p, weights


def supervised_objective(mask_model, weight_model, levels, *, settings=None):
    if not isinstance(weight_model, Feature9WeightCNN):
        return _original_objective(mask_model, weight_model, levels, settings=settings)
    if mask_model is not None:
        raise ValueError('feature9 study requires fixed bilinear support, not a MaskCNN')
    extractor = partial(direct_features9, feature_mode=weight_model.feature_mode)
    with patch.object(training, 'direct_features', extractor):
        return _original_objective(mask_model, weight_model, levels, settings=settings)


def initialize_weight(weight, path):
    if not isinstance(weight, Feature9WeightCNN):
        return _original_initialize(weight, path)
    path = Path(path).resolve()
    before = legacy_study._file_hash(path)
    package = torch.load(path, map_location='cpu', weights_only=True)
    spec = package.get('weight_spec', package.get('specs', {}).get('weight'))
    state = package.get('weight_state', package.get('models', {}).get('weight'))
    if spec is None or state is None or legacy_study._file_hash(path) != before:
        raise ValueError('Initialization checkpoint is invalid or changed while reading')
    if spec == weight.spec:
        weight.load_state_dict(state, strict=True)
        rule, preserving = 'weights_only', True
    else:
        if 'feature_mode' in spec:
            raise ValueError('Cannot warm-transfer between different feature9 modes')
        old = _original_model(spec).double()
        old.load_state_dict(state, strict=True)
        transfer_legacy_weights(weight, old)
        rule, preserving = 'drop_first_conv_center_input_column_1', False
    if any(not bool(torch.isfinite(p).all()) for p in weight.parameters()):
        raise ValueError('Initialization contains nonfinite parameters')
    return dict(path=str(path), sha256=before, weight_spec=spec,
                target_weight_spec=weight.spec, transfer=rule,
                function_preserving=preserving, optimizer_reused=False,
                training_updates_reused=False)


def _without_mode(models):
    models = deepcopy(models)
    models['weight'].pop('feature_mode', None)
    return models


def resolve_settings(requested, source_settings, teacher_settings, cfg, tc):
    clean = deepcopy(requested)
    mode = validate_mode(clean['models']['weight'].pop('feature_mode', None))
    resolved = _original_resolve(clean, source_settings, teacher_settings, cfg, tc)
    resolved['models']['weight']['feature_mode'] = mode
    resolved['feature_experiment'] = dict(version=FEATURE_VERSION, mode=mode,
        matrix_channels=9, diagonal_floor=DIAGONAL_FLOOR, log_epsilon=LOG_EPSILON,
        signed_couplings=True, warm_transfer='drop_first_conv_center_input_column_1')
    return resolved


def _sources():
    sources = _original_sources()
    scripts = Path(__file__).resolve().parent
    for name in ('bamg_feature9.py', 'bamg_feature9_adapter.py',
                 'run_v6_7_bamg_energy_cnn_feature9.py', 'bamg_energy_cnn_memory.py'):
        path = scripts / name
        sources[str(path.relative_to(paired.PROJECT))] = legacy_study._file_hash(path)
    return sources


def _bank(original, example, method, checkpoint, settings):
    if method == 'old_cnn':
        settings = deepcopy(settings)
        settings['models'] = _without_mode(settings['models'])
    return original(example, method, checkpoint, settings)


def development_reference(directory, settings, source, teacher):
    """Same legacy cohort audit, allowing only the declared feature ablation.

    The routing layer separately checks version, lambdas and routing protocol.
    A reference whose final/OOD claim has opened remains forbidden.
    """
    root = Path(directory).resolve()
    if (root / 'final/claim.json').exists():
        raise ValueError('Choose feature settings on development validation before opening final/OOD')
    manifest = paired._read(root / 'study_manifest.json')
    previous = paired._read(root / 'configuration.json')
    if paired.numerical.digest(previous) != manifest['settings_digest']:
        raise ValueError('Development reference configuration changed')
    if manifest['source'] != source or manifest['teacher'] != teacher:
        raise ValueError('Development reference must use the same source model and numerical teacher')
    for relative, expected in paired._read(root / 'prepared_manifest.json').items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or paired._file_hash(path) != expected:
            raise ValueError('Development reference prepared input changed: ' + relative)
    before, after = deepcopy(previous['data']), deepcopy(settings['data'])
    old_count = before['train'].pop('per_family')
    new_count = after['train'].pop('per_family')
    if before != after or new_count < old_count or after.get('sampling') != 'stratified_nested_v1':
        raise ValueError('Development reference requires the same stratified cohort and nondecreasing TRAIN count')
    for key in ('model_seeds', 'models', 'benchmark', 'warm_validation', 'holdout',
                'discretization', 'support_mode', 'sampling', 'supervised', 'finetune',
                'caps', 'holdout_benchmark'):
        a, b = deepcopy(previous.get(key)), deepcopy(settings.get(key))
        if key == 'models':
            for models in (a, b):
                if 'feature_mode' in models['weight']:
                    validate_mode(models['weight']['feature_mode'])
            a, b = _without_mode(a), _without_mode(b)
        if key == 'warm_validation':
            if a['every'] * new_count != b['every'] * old_count:
                raise ValueError('Development reference validation cadence must match in epochs')
            a.pop('every', None)
            b.pop('every', None)
        if key in ('supervised', 'finetune'):
            if a['updates'] * new_count != b['updates'] * old_count:
                raise ValueError('Development reference must use equal training epochs')
            if key == 'finetune':
                sa = a.get('curriculum_step')
                sb = b.get('curriculum_step')
                sa = (a['updates'] + 1) // 2 if sa is None else sa
                sb = (b['updates'] + 1) // 2 if sb is None else sb
                if sa * b['updates'] != sb * a['updates']:
                    raise ValueError('Development reference curriculum fraction differs')
            for item in ('updates', 'curriculum_step'):
                a.pop(item, None)
                b.pop(item, None)
        if a != b:
            raise ValueError('Development reference comparison protocol differs: ' + key)
    files = ('inputs/exclusion_manifest.json', 'data/dataset_plan.json',
             'data/development_manifest.json', 'prepared_manifest.json',
             'configuration.json', 'study_manifest.json')
    return dict(path=str(root), hashes={name: paired._file_hash(root / name) for name in files},
                scope='shared development cohort; feature ablation; no final/OOD based selection')


@contextmanager
def feature9_mode():
    """Activate both new feature modes, leaving legacy models on their old path."""
    global _active
    if _active:
        raise RuntimeError('feature9_mode cannot be nested or shared between concurrent studies')
    _active = True
    try:
        with ExitStack() as stack:
            for module in (transfer, solver, legacy_study, benchmark, routing_benchmark):
                stack.enter_context(patch.object(module, 'model_from_spec', model_from_spec))
            changes = ((solver, '_weights', _weights),
                       (training, 'supervised_objective', supervised_objective),
                       (epoch_training, 'supervised_objective', supervised_objective),
                       (arm_training, 'WeightCNN', Feature9WeightCNN),
                       (arm_training, '_initialize_weight', initialize_weight),
                       (paired, 'resolve_settings', resolve_settings),
                       (paired, '_development_reference', development_reference),
                       (paired, '_sources', _sources))
            for module, name, value in changes:
                stack.enter_context(patch.object(module, name, value))
            for module in (benchmark, holdout):
                stack.enter_context(patch.object(module, '_build_bank', partial(_bank, _original_bank)))
            for module in (routing_benchmark, routing_holdout):
                stack.enter_context(patch.object(module, '_build_bank', partial(_bank, _original_routing_bank)))
            yield
    finally:
        _active = False
