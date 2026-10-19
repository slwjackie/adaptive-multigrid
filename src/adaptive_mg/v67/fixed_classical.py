"""Immutable numerical contracts for fixed classical parents.

This path performs no classical strategy selection or calibration. Legacy
workflows may still load their own rules through the explicitly lazy adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from ..config import MGConfig


FIXED_STRATEGY_NAME = 'line_alt_energymin_full__em5__v22'
EM_ITERATIONS = 5
EM_RTOL = 1e-8
VERSION = 'fixed-classical-em-v1'
BILINEAR_STRATEGY_NAME = 'line_alt_bilinear_full__v22'
BILINEAR_VERSION = 'fixed-classical-bilinear-v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class FixedClassicalBaseline:
    """Frozen full MG numerical scope, independent of A and neural branch.

All hierarchy, smoothing, precision/backend, convergence and recovery controls
are bound. Only presentation/action-mode fields excluded by MGConfig's
numerical_scope are omitted. A new RHS or neural branch cannot alter the parent.
    """

    mg_contract_json: str = field(repr=False)
    kind = 'fixed_classical'
    version = VERSION
    strategy_name = FIXED_STRATEGY_NAME

    def __post_init__(self):
        scope = json.loads(self.mg_contract_json)
        required = dict(strategy_name=FIXED_STRATEGY_NAME, pre_steps=2,
                        post_steps=2, em_iterations=EM_ITERATIONS,
                        em_fixed_iterations=True, em_rtol=EM_RTOL)
        if any(scope.get(key) != value for key, value in required.items()):
            raise ValueError('Fixed classical requires line_alt + EM5 + full, pre/post=2/2, fixed iterations and em_rtol=1e-8')
        normalized = MGConfig(**scope).numerical_scope()
        if scope != normalized or self.mg_contract_json != _canonical(scope):
            raise ValueError('Fixed classical numerical contract must be complete and canonical')

    @classmethod
    def from_config(cls, cfg):
        return cls(_canonical(cfg.mg.numerical_scope()))

    @classmethod
    def from_dict(cls, value):
        if set(value) != {'kind', 'version', 'mg'} or value['kind'] != cls.kind or value['version'] != cls.version:
            raise ValueError('Unsupported fixed classical baseline contract')
        return cls(_canonical(value['mg']))

    def validate_config(self, cfg):
        if _canonical(cfg.mg.numerical_scope()) != self.mg_contract_json:
            raise ValueError('Fixed classical numerical settings differ from the frozen baseline')

    def to_dict(self):
        return dict(kind=self.kind, version=self.version, mg=json.loads(self.mg_contract_json))

    def digest(self):
        return hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()

    def selection_record(self):
        return dict(kind=self.kind, strategy_name=self.strategy_name,
                    baseline_digest=self.digest())


class BilinearClassicalBaseline(FixedClassicalBaseline):
    """Line-alternating, bilinear, full-coarsening parent with no EM work.

    MGConfig retains inactive EM controls for schema compatibility. They are
    frozen like every numerical field, but the bilinear strategy never calls
    energy minimization. The distinct version prevents old EM runs from being
    silently reinterpreted as this new experimental contract.
    """

    version = BILINEAR_VERSION
    strategy_name = BILINEAR_STRATEGY_NAME

    def __post_init__(self):
        scope = json.loads(self.mg_contract_json)
        required = dict(strategy_name=BILINEAR_STRATEGY_NAME, pre_steps=2, post_steps=2)
        if any(scope.get(key) != value for key, value in required.items()):
            raise ValueError('Fixed bilinear classical requires line_alt + bilinear + full, pre/post=2/2')
        normalized = MGConfig(**scope).numerical_scope()
        if scope != normalized or self.mg_contract_json != _canonical(scope):
            raise ValueError('Fixed classical numerical contract must be complete and canonical')


def load_baseline(path):
    """Load fixed provenance without importing the legacy selector package."""
    value = json.loads(Path(path).read_text())
    if value.get('kind') == FixedClassicalBaseline.kind:
        if value.get('version') == BILINEAR_VERSION:
            return BilinearClassicalBaseline.from_dict(value)
        return FixedClassicalBaseline.from_dict(value)
    from .strong import load_strong_rules
    return load_strong_rules(path)


def baseline_metadata(baseline, *, include_contract=False):
    """Use explicit fixed-baseline keys; preserve historical schemas otherwise."""
    fixed = isinstance(baseline, FixedClassicalBaseline)
    result = {('baseline_digest' if fixed else 'rules_digest'): baseline.digest()}
    if include_contract:
        result['baseline' if fixed else 'rules'] = baseline.to_dict()
    return result


def stored_baseline_digest(record, baseline):
    return record.get('baseline_digest' if isinstance(baseline, FixedClassicalBaseline) else 'rules_digest')
