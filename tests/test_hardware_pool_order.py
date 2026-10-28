import hashlib
import json
from types import SimpleNamespace

import pytest

from adaptive_mg import provenance


def _fingerprint(environment):
    return hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def pools(monkeypatch):
    records = [
        dict(internal_api='openblas', num_threads=1, version='0.3.31.dev', architecture='SkylakeX'),
        dict(internal_api='openmp', num_threads=1, version=None, architecture=None),
        dict(internal_api='openblas', num_threads=1, version='0.3.34.106.0', architecture='SkylakeX'),
    ]
    monkeypatch.setattr(provenance, '_POOL_CONTROLLER', SimpleNamespace(info=lambda: records))
    return records


def test_hardware_fingerprint_ignores_library_discovery_order(pools):
    original = provenance.hardware_environment()
    pools.reverse()
    reordered = provenance.hardware_environment()
    assert reordered == original
    assert _fingerprint(reordered) == _fingerprint(original)


@pytest.mark.parametrize(('field', 'value'), [('num_threads', 2), ('version', '0.3.35')])
def test_hardware_fingerprint_detects_changed_pool_fields(pools, field, value):
    original = _fingerprint(provenance.hardware_environment())
    pools[0][field] = value
    assert _fingerprint(provenance.hardware_environment()) != original


def test_hardware_fingerprint_preserves_duplicate_pools(pools):
    original = provenance.hardware_environment()
    duplicate = pools[0].copy()
    pools.append(duplicate)
    duplicated = provenance.hardware_environment()
    assert duplicated['pools'].count(duplicate) == 2
    assert len(duplicated['pools']) == len(original['pools']) + 1
    assert _fingerprint(duplicated) != _fingerprint(original)
