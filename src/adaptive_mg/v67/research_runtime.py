"""Serialization and generation guards for the optional research experts.

The offline teacher context is explicit and never enables a production solve.
All generated numerical operators still use the original cached FP64 kernels.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json

_OFFLINE_TEACHER = ContextVar('v67_offline_teacher_generation', default=False)


@contextmanager
def offline_teacher_generation():
    token = _OFFLINE_TEACHER.set(True)
    try:
        yield
    finally:
        _OFFLINE_TEACHER.reset(token)


def is_teacher(module):
    return bool(getattr(module, 'training_only', False))


def require_generation_allowed(module, *, production=False):
    if is_teacher(module) and (production or not _OFFLINE_TEACHER.get()):
        raise RuntimeError('Teacher is training-only; deployment inference is forbidden')


def require_components_allowed(components, cfg, branch):
    for name, enabled in (('smoother', branch in ('H_S', 'H_SP')),
                          ('transfer', branch in ('H_P', 'H_SP'))):
        if enabled:
            require_generation_allowed(getattr(components, name), production=cfg.mode == 'production')


def research_spec(module):
    if hasattr(module, 'research_transfer_spec'):
        return dict(family='research_cnn_transfer',config=module.research_transfer_spec())
    if hasattr(module, 'architecture_spec'):
        return dict(family='research_smoother', config=module.architecture_spec())
    if hasattr(module, 'architecture_config'):
        return dict(family='graph_transfer', config=module.architecture_config())
    return None


def restore_expert(spec):
    if spec['family'] == 'research_cnn_transfer':
        from .research_controls import ControlledTransferCNN
        return ControlledTransferCNN(**spec['config'])
    if spec['family'] == 'research_smoother':
        from .research_smoothers import research_smoother_from_spec
        return research_smoother_from_spec(spec['config'])
    if spec['family'] == 'graph_transfer':
        from .research_transfer import make_graph_transfer
        return make_graph_transfer(**spec['config'])
    raise ValueError('Unknown research expert family')


def expert_signature(module):
    from ..provenance import module_signature
    signature = module_signature(module)
    spec = research_spec(module)
    if spec is not None:
        signature += hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return signature
