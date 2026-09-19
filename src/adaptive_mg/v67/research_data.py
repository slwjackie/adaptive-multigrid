"""Fresh strong-aware research data and a single-use final-test gate.

Only the existing deterministic selector labels examples.  Preparing development
data commits final specifications without assembling, selecting, or solving any
final/OOD operator.  This is reproducibility/leakage protection, not a security
boundary against a person deliberately editing Python or its JSON ledgers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import fcntl
import hashlib
import json
import os
import tempfile

import numpy as np
import scipy.sparse as sp

from ..data import Example
from ..pde import DiffusionCase, assemble_stiffness
from ..provenance import json_safe, operator_digest
from .data import VariableCase, variable_stiffness
from .strong import StrongRules, load_strong_rules, select_strong_strategy

VERSION = 'strong-aware-research-data-v1'
FAMILIES = ('near_isotropic', 'aligned_anisotropy', 'rotated_anisotropy',
            'local_patch', 'checkerboard', 'channel', 'variable_coefficient')
DEVELOPMENT = ('train', 'validation')
HOLDOUT = ('final', 'ood_grid', 'ood_rotation', 'ood_anisotropy', 'ood_contrast', 'ood_topology')
TRAIN_ANGLES = (15., 30., 45., 60., 75.)
OOD_ANGLES = (7.5, 22.5, 37.5, 52.5, 67.5, 82.5)


def _hash(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False)
    if exclusive:
        with path.open('x') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_research_plan(*, smoke=False, seed=2026091601):
    """Return metadata only; full train=280, validation=56, final/OOD=126.

    Smoke has 7 training and 14 validation operators and cannot run final tests.
    A caller may reduce sizes/counts in tests before committing this plan.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    specs = {
        'train': dict(sizes=[15] if smoke else [15, 31, 63, 127], per_family=1 if smoke else 10, families=list(FAMILIES)),
        'validation': dict(sizes=[15, 31] if smoke else [15, 31, 63, 127], per_family=1 if smoke else 2, families=list(FAMILIES)),
        'final': dict(sizes=[15, 31, 63, 127], per_family=2, families=list(FAMILIES)),
        'ood_grid': dict(sizes=[255], per_family=2, families=list(FAMILIES)),
        'ood_rotation': dict(sizes=[31, 63], per_family=2, families=list(FAMILIES[2:])),
        'ood_anisotropy': dict(sizes=[31, 63], per_family=1, families=list(FAMILIES[1:])),
        'ood_contrast': dict(sizes=[31, 63], per_family=1, families=list(FAMILIES[3:])),
        'ood_topology': dict(sizes=[31, 63], per_family=2, families=list(FAMILIES[3:])),
    }
    for index, (name, spec) in enumerate(specs.items()):
        spec.update(seed=seed + 1000003 * index, count=len(spec['sizes']) * len(spec['families']) * spec['per_family'])
    return dict(version=VERSION, smoke=bool(smoke), master_seed=seed, splits=specs,
                train_angles=list(TRAIN_ANGLES), rotation_ood_angles=list(OOD_ANGLES),
                final_materialization_requires_freeze=True,
                selector_mutation=False, selection_input='A and structured grid shape only')


def _validate_plan(plan):
    if plan.get('version') != VERSION or set(plan.get('splits', {})) != set(DEVELOPMENT + HOLDOUT):
        raise ValueError('Unsupported or incomplete research plan')
    seeds = []
    for name, spec in plan['splits'].items():
        sizes, families = spec['sizes'], spec['families']
        if not sizes or len(set(sizes)) != len(sizes) or any(isinstance(n, bool) or not isinstance(n, int) or n < 3 or (n + 1) & n for n in sizes):
            raise ValueError('Grid sizes must be distinct 2**L-1 integers >=3')
        if not families or len(set(families)) != len(families) or set(families) - set(FAMILIES):
            raise ValueError('Invalid or duplicate families')
        if isinstance(spec['per_family'], bool) or not isinstance(spec['per_family'], int) or spec['per_family'] < 1:
            raise ValueError('per_family must be a positive integer')
        if spec['count'] != len(sizes) * len(families) * spec['per_family']:
            raise ValueError('Declared split count differs from specifications')
        if isinstance(spec['seed'], bool) or not isinstance(spec['seed'], int) or spec['seed'] < 0:
            raise ValueError('Invalid split seed')
        seeds.append(spec['seed'])
    if len(set(seeds)) != len(seeds):
        raise ValueError('Every split must have an independent seed')
    if plan.get('train_angles') != list(TRAIN_ANGLES) or plan.get('rotation_ood_angles') != list(OOD_ANGLES):
        raise ValueError('Angle protocol differs from implemented generator')


def _digests(value):
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {'normalized_operator_digest', 'group_digest', 'normalized_digest'} and isinstance(child, str) and len(child) == 64:
                found.add(child)
            else:
                found.update(_digests(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_digests(child))
    return found


def historical_operator_index(roots, *, exclude=()):
    """Read targeted manifests and raw_results, including partial/new local runs.

    Directory traversal never reads checkpoints, logs, per-repeat records or
    array files.  New local user runs are found without hard-coded run names.
    Malformed candidate provenance fails closed instead of silently omitting it.
    """
    excluded = [Path(p).resolve() for p in exclude]
    candidates = set()
    resolved_roots = []
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            raise FileNotFoundError(f'Historical data root does not exist: {root}')
        resolved_roots.append(str(root))
        if root.is_file():
            candidates.add(root)
            continue
        for directory, names, files in os.walk(root):
            here = Path(directory)
            names[:] = [n for n in names if n not in {'.git', '.venv', '__pycache__', 'node_modules', 'records', 'repeat_records'}
                        and not any(here / n == p or p in (here / n).parents for p in excluded)]
            if any(here == p or p in here.parents for p in excluded):
                names[:] = []
                continue
            for name in files:
                if name.endswith('.json') and ('manifest' in name or name == 'raw_results.json'):
                    candidates.add(here / name)
    seen, sources = set(), []
    ordered = sorted(candidates, key=lambda p: (p.name == 'raw_results.json', str(p)))
    for path in ordered:
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f'Cannot verify historical provenance: {path}') from exc
        groups = _digests(payload)
        if groups:
            seen.update(groups)
            sources.append(dict(path=str(path), sha256=_file_hash(path), normalized_operators=len(groups)))
    return dict(roots=resolved_roots, sources=sources, normalized_operator_digests=sorted(seen),
                count=len(seen), scan_scope='JSON manifests and raw_results.json only')


@dataclass(frozen=True)
class ResearchCase:
    n: int
    epsilon: float
    angle_deg: float
    contrast: float
    pattern: str
    family: str
    topology: str = 'standard'
    patch_bounds: tuple = (.25, .75, .25, .75)
    checker_cells: int = 4
    channel_width: float = .09
    amplitude: float = 1.
    frequency: float = 1.
    phase: float = 0.
    geometry_phase: float = 0.


def _assemble_case(case):
    if case.topology == 'standard':
        if case.pattern == 'variable_coefficient':
            return variable_stiffness(VariableCase(case.n, case.epsilon, case.angle_deg,
                                                  case.amplitude, case.frequency, case.phase))
        return assemble_stiffness(DiffusionCase(case.n, case.epsilon, case.angle_deg, case.contrast,
            case.pattern, case.patch_bounds, case.checker_cells, case.channel_width))
    # OOD topology only.  The mesh, Dirichlet elimination and P1 element formula
    # are unchanged; only the positive coefficient field/tensor topology varies.
    n = case.n
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    i, j = i.ravel(), j.ravel()
    theta = np.deg2rad(case.angle_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    tensor = rotation @ np.diag([1., case.epsilon]) @ rotation.T
    rows, cols, values = [], [], []
    for offsets, gradients in ((np.array([[0, 0], [1, 0], [1, 1]]), np.array([[-1., 0], [1, -1], [0, 1]])),
                               (np.array([[0, 0], [1, 1], [0, 1]]), np.array([[0., -1], [1, 0], [-1, 1]]))):
        x = (i + offsets[:, 0].mean()) / (n + 1)
        y = (j + offsets[:, 1].mean()) / (n + 1)
        if case.pattern == 'local_patch':
            c = .04 * np.sin(case.geometry_phase)
            hard = (((x - .30 - c) / .20) ** 2 + ((y - .36) / .13) ** 2 < 1) | (((x - .72) / .12) ** 2 + ((y - .65 - c) / .23) ** 2 < 1)
        elif case.pattern == 'checkerboard':
            row = np.floor(y * (case.checker_cells + 1)).astype(int)
            col = np.floor(x * case.checker_cells + .5 * (row % 2) + case.geometry_phase / (2 * np.pi)).astype(int)
            hard = (row + col) % 2 == 0
        elif case.pattern == 'channel':
            curve = .20 * np.sin(2 * np.pi * x + case.geometry_phase)
            hard = (np.abs(y - .32 - curve) < case.channel_width) | (np.abs(y - .74 + .5 * curve) < .65 * case.channel_width)
        elif case.pattern == 'variable_coefficient':
            exponent = case.amplitude * (.65 * np.sin(2 * np.pi * case.frequency * x + case.phase) * np.cos(4 * np.pi * y)
                        + .35 * np.cos(6 * np.pi * x - 2 * np.pi * y + case.geometry_phase))
            tensors = np.exp(exponent)[:, None, None] * tensor
        else:
            raise ValueError('Topology OOD is only meaningful for heterogeneous families')
        if case.pattern != 'variable_coefficient':
            tensors = np.where(hard[:, None, None], case.contrast * tensor, np.eye(2))
        local = .5 * np.einsum('ik,nkl,jl->nij', gradients, tensors, gradients)
        gi, gj = i[:, None] + offsets[None, :, 0], j[:, None] + offsets[None, :, 1]
        valid = (gi >= 1) & (gi <= n) & (gj >= 1) & (gj <= n)
        idx = (gi - 1) * n + gj - 1
        mask = valid[:, :, None] & valid[:, None, :]
        rows.append(np.broadcast_to(idx[:, :, None], local.shape)[mask])
        cols.append(np.broadcast_to(idx[:, None, :], local.shape)[mask])
        values.append(local[mask])
    a = sp.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(n*n, n*n)).tocsr()
    a = (.5 * (a + a.T)).tocsr()
    a.eliminate_zeros()
    return a


def _specification(split, spec, n, family, index, retry):
    sample_seed = int(np.random.SeedSequence([spec['seed'], n, FAMILIES.index(family), index, retry]).generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(sample_seed)
    epsilon = float(10 ** rng.uniform(-3., -1.))
    angle = float(rng.choice(OOD_ANGLES if split == 'ood_rotation' else TRAIN_ANGLES))
    contrast = float(10 ** rng.uniform(1., 3.))
    if split == 'ood_anisotropy':
        epsilon = float(10 ** rng.uniform(-5., -3.1))
    if split == 'ood_contrast':
        contrast = float(10 ** rng.uniform(3.2, 5.))
    if family == 'near_isotropic':
        epsilon, contrast = float(rng.uniform(.8, 1.)), 1.
    elif family == 'aligned_anisotropy':
        angle, contrast = float(rng.choice([0., 90.])), 1.
    elif family == 'rotated_anisotropy':
        contrast = 1.
    center_x, center_y = rng.uniform(.35, .65, 2)
    width_x, width_y = rng.uniform(.13, .24, 2)
    pattern = 'uniform' if family in FAMILIES[:3] else family
    case = ResearchCase(n, epsilon, angle, contrast, pattern, family,
        'novel_' + family if split == 'ood_topology' else 'standard',
        tuple(map(float, (center_x-width_x, center_x+width_x, center_y-width_y, center_y+width_y))),
        int(rng.choice([3, 4, 5])), float(rng.uniform(.06, .13)),
        float(np.log(contrast) / 2), float(rng.uniform(.7, 2.5)),
        float(rng.uniform(0, 2*np.pi)), float(rng.uniform(0, 2*np.pi)))
    return dict(name=f'{split}_{family}_n{n}_{index}_r{retry}', case=asdict(case),
                seed=sample_seed, scale=float(10 ** rng.uniform(-.5, .5)), split=split,
                family=family, retry=retry)


def _example(record):
    case = ResearchCase(**{**record['case'], 'patch_bounds': tuple(record['case']['patch_bounds'])})
    a = record['scale'] * _assemble_case(case)
    coordinate = np.arange(1, case.n + 1) / (case.n + 1)
    x, y = np.meshgrid(coordinate, coordinate, indexing='ij')
    exact = np.sin(np.pi * x) * np.sin(np.pi * y)
    rng = np.random.default_rng(record['seed'] ^ 0x4D47524853)
    for _ in range(4):
        k, l = rng.integers(1, min(case.n, 12) + 1, size=2)
        exact += rng.uniform(-.3, .3) * np.sin(k * np.pi * x) * np.sin(l * np.pi * y)
    exact = exact.ravel()
    return Example(record['name'], case, a, np.asarray(a @ exact), exact,
                   operator_digest(a), operator_digest(a, scale_invariant=True), record['seed'], record['scale'])


def _selection_record(example, rules):
    selected = select_strong_strategy(example.a, example.n, rules)
    return dict(strategy_name=selected.strategy_name, rule_id=selected.rule_id,
                rules_digest=selected.rules_digest, features=selected.features,
                rule_evidence=selected.rule_evidence)


def _generate_split(name, spec, forbidden, rules):
    examples, records, rejected = [], [], 0
    for n in spec['sizes']:
        for family in spec['families']:
            for index in range(spec['per_family']):
                for retry in range(100):
                    record = _specification(name, spec, n, family, index, retry)
                    example = _example(record)
                    if example.group_digest in forbidden:
                        rejected += 1
                        continue
                    forbidden.add(example.group_digest)
                    selection = _selection_record(example, rules)
                    example.strong_selection = selection
                    example.research_split = name
                    example.research_family = family
                    example.research_ood = name.removeprefix('ood_') if name.startswith('ood_') else None
                    record.update(example.manifest(), selection=selection,
                                  rhs_digest=hashlib.sha256(example.b.tobytes()).hexdigest(),
                                  exact_digest=hashlib.sha256(example.exact.tobytes()).hexdigest())
                    examples.append(example)
                    records.append(record)
                    break
                else:
                    raise ValueError('Could not generate a normalized-operator-disjoint sample after 100 attempts')
    return examples, records, rejected


def _restore(records, rules):
    examples = []
    for record in records:
        example = _example(record)
        if example.digest != record['operator_digest'] or example.group_digest != record['normalized_operator_digest']:
            raise ValueError('Stored operator does not reconstruct exactly')
        if hashlib.sha256(example.b.tobytes()).hexdigest() != record['rhs_digest'] or hashlib.sha256(example.exact.tobytes()).hexdigest() != record['exact_digest']:
            raise ValueError('Stored manufactured RHS/solution does not reconstruct exactly')
        selected = _selection_record(example, rules)
        if json_safe(selected) != record['selection']:
            raise ValueError('Frozen C*(A) selection changed')
        example.strong_selection = selected
        example.research_split = record['split']
        example.research_family = record['family']
        example.research_ood = record['split'].removeprefix('ood_') if record['split'].startswith('ood_') else None
        examples.append(example)
    return examples


def prepare_development_data(output_dir, rules, *, plan=None, historical_roots=()):
    """Create only train/validation and return (train, validation, manifest).

    ``output_dir`` is the whole new research-run directory: keep all run results
    under it so historical scans can exclude this run's own recorded splits.
    Existing development manifests require load_development_data, never rewrite.
    """
    output = Path(output_dir).resolve()
    if (output / 'dataset_plan.json').exists() or (output / 'development_manifest.json').exists():
        raise FileExistsError('Development data already committed; load it instead')
    if not historical_roots:
        raise ValueError('Explicit historical artifact roots are required for leakage exclusion')
    plan = make_research_plan() if plan is None else plan
    _validate_plan(plan)
    history = historical_operator_index(historical_roots, exclude=[output])
    forbidden = set(history['normalized_operator_digests'])
    splits, records, rejected = {}, {}, {}
    for name in DEVELOPMENT:
        splits[name], records[name], rejected[name] = _generate_split(name, plan['splits'][name], forbidden, rules)
    manifest = dict(version=VERSION, plan_digest=_hash(plan), rules=rules.to_dict(), rules_digest=rules.digest(),
        historical_index=history, splits=records, rejected_normalized_duplicates=rejected,
        normalized_operator_disjoint=True, final_operators_materialized=False,
        final_operators_selected=False, final_operators_solved=False)
    manifest = json_safe(manifest)
    _write_json(output / 'dataset_plan.json', plan, exclusive=True)
    _write_json(output / 'development_manifest.json', manifest, exclusive=True)
    return splits['train'], splits['validation'], manifest


def load_development_data(output_dir, rules):
    output = Path(output_dir).resolve()
    plan = json.loads((output / 'dataset_plan.json').read_text())
    manifest = json.loads((output / 'development_manifest.json').read_text())
    _validate_plan(plan)
    if manifest['plan_digest'] != _hash(plan) or manifest['rules_digest'] != rules.digest():
        raise ValueError('Plan or frozen classical selector changed')
    seen = set(manifest['historical_index']['normalized_operator_digests'])
    for name in DEVELOPMENT:
        records = manifest['splits'][name]
        if len(records) != plan['splits'][name]['count']:
            raise ValueError('Development split is incomplete')
        for record in records:
            if record['normalized_operator_digest'] in seen:
                raise ValueError('Normalized-operator leakage in stored development data')
            seen.add(record['normalized_operator_digest'])
    return _restore(manifest['splits']['train'], rules), _restore(manifest['splits']['validation'], rules), manifest


def freeze_research(output_dir, checkpoint_paths, rules_path, controller_paths, source_paths, config):
    """Pin the complete declared deployable system before any final materialization.

    Checkpoints/controllers are name->path maps; source_paths lists all relevant
    source files.  Empty controller_paths explicitly means no controller weights;
    its fixed configuration still belongs in config.  No final data are generated.
    """
    output = Path(output_dir).resolve()
    if (output / 'final_claim.json').exists():
        raise ValueError('Final test already claimed; cannot change frozen research')
    plan = json.loads((output / 'dataset_plan.json').read_text())
    _validate_plan(plan)
    if plan['smoke']:
        raise ValueError('Smoke research cannot freeze or consume the untouched final test')
    if not checkpoint_paths or not source_paths or not config:
        raise ValueError('Freeze needs student checkpoints, source files and full evaluation configuration')
    manifest = json.loads((output / 'development_manifest.json').read_text())
    rules = load_strong_rules(rules_path)
    if manifest['plan_digest'] != _hash(plan) or rules.digest() != manifest['rules_digest']:
        raise ValueError('Dataset plan or C*(A) rules changed before freeze')
    files = {'rules': str(Path(rules_path).resolve()),
             'plan': str(output / 'dataset_plan.json'),
             'development': str(output / 'development_manifest.json')}
    files.update({'checkpoint:' + k: str(Path(v).resolve()) for k, v in checkpoint_paths.items()})
    files.update({'controller:' + k: str(Path(v).resolve()) for k, v in controller_paths.items()})
    files.update({'source:' + str(index): str(Path(p).resolve()) for index, p in enumerate(sorted(map(str, source_paths)))})
    frozen = dict(version=VERSION, output_dir=str(output), plan_digest=_hash(plan), rules_digest=rules.digest(),
                  files={name: dict(path=path, sha256=_file_hash(path)) for name, path in files.items()},
                  config=json_safe(config), config_digest=_hash(config), final_test_consumed=False,
                  controller_files_declared=len(controller_paths), no_final_operator_observation=True)
    path = output / 'research_freeze.json'
    _write_json(path, frozen, exclusive=True)
    return path


def _verify_freeze(path, output):
    frozen = json.loads(Path(path).read_text())
    if frozen.get('version') != VERSION or frozen.get('output_dir') != str(output):
        raise ValueError('Freeze belongs to another research run')
    if frozen['config_digest'] != _hash(frozen['config']):
        raise ValueError('Frozen configuration digest mismatch')
    for record in frozen['files'].values():
        if _file_hash(record['path']) != record['sha256']:
            raise ValueError(f'Frozen input changed: {record["path"]}')
    return frozen


@dataclass
class FinalEvaluationClaim:
    output_dir: Path
    fingerprint: str
    freeze_path: Path
    _lock_fd: int | None

    def close(self):
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def claim_final_evaluation(output_dir, freeze_path, invocation, *, resume=False):
    """Acquire a process lease and commit one final invocation before seeing A.

    Interrupted exact resumes are allowed; concurrent or completed re-use and
    changed checkpoints/configuration/command are rejected.  Close the returned
    claim in finally (or use a with block); complete_final_evaluation closes it.
    """
    output, freeze_path = Path(output_dir).resolve(), Path(freeze_path).resolve()
    frozen = _verify_freeze(freeze_path, output)
    if not invocation:
        raise ValueError('Final invocation must explicitly declare evaluation settings')
    fingerprint = _hash(dict(freeze_sha256=_file_hash(freeze_path), invocation=invocation))
    fd = os.open(output / '.final_evaluation.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('A final evaluation already holds the process lease') from exc
        ledger = output / 'final_claim.json'
        if ledger.exists():
            previous = json.loads(ledger.read_text())
            if previous.get('status') == 'completed':
                raise ValueError('Final evaluation already completed; it is single-use')
            if not resume or previous.get('fingerprint') != fingerprint:
                raise ValueError('Only an identical interrupted final invocation can resume')
        else:
            if resume:
                raise ValueError('Cannot resume an unclaimed final evaluation')
            _write_json(ledger, dict(version=VERSION, status='claimed', fingerprint=fingerprint,
                freeze_path=str(freeze_path), freeze_sha256=_file_hash(freeze_path),
                invocation=invocation, plan_digest=frozen['plan_digest']), exclusive=True)
        return FinalEvaluationClaim(output, fingerprint, freeze_path, fd)
    except BaseException:
        os.close(fd)
        raise


def _check_claim(output, claim):
    if not isinstance(claim, FinalEvaluationClaim) or claim._lock_fd is None or claim.output_dir != output:
        raise ValueError('An active final-evaluation claim is required')
    ledger = json.loads((output / 'final_claim.json').read_text())
    if ledger.get('status') != 'claimed' or ledger.get('fingerprint') != claim.fingerprint:
        raise ValueError('Final evaluation claim is stale')
    _verify_freeze(claim.freeze_path, output)
    if ledger.get('final_data_sha256') and _file_hash(output / 'final_data_manifest.json') != ledger['final_data_sha256']:
        raise ValueError('Committed final data changed')
    return ledger


def materialize_final_data(output_dir, claim, rules):
    """Return (dict of final/OOD Example lists, manifest), only under a valid lease."""
    output = Path(output_dir).resolve()
    ledger = _check_claim(output, claim)
    plan = json.loads((output / 'dataset_plan.json').read_text())
    development = json.loads((output / 'development_manifest.json').read_text())
    _validate_plan(plan)
    if plan['smoke'] or rules.digest() != development['rules_digest'] or _hash(plan) != ledger['plan_digest']:
        raise ValueError('Smoke, changed rules, or changed final plan cannot consume final data')
    final_path = output / 'final_data_manifest.json'
    if final_path.exists():
        manifest = json.loads(final_path.read_text())
        if manifest['claim_fingerprint'] != claim.fingerprint:
            raise ValueError('Final data belong to another invocation')
        if set(manifest['splits']) != set(HOLDOUT):
            raise ValueError('Final data are incomplete')
        seen = set(development['historical_index']['normalized_operator_digests']) | _digests(development['splits'])
        for name, records in manifest['splits'].items():
            if len(records) != plan['splits'][name]['count']:
                raise ValueError('Final data are incomplete')
            for record in records:
                if record['normalized_operator_digest'] in seen:
                    raise ValueError('Normalized-operator leakage in final data')
                seen.add(record['normalized_operator_digest'])
        restored = {name: _restore(records, rules) for name, records in manifest['splits'].items()}
        for examples in restored.values():
            for example in examples:
                example.research_final_claim = claim
                example.research_run_root = output
        return restored, manifest
    history = historical_operator_index(development['historical_index']['roots'], exclude=[output])
    forbidden = set(history['normalized_operator_digests']) | set(development['historical_index']['normalized_operator_digests'])
    forbidden.update(_digests(development['splits']))
    splits, records, rejected = {}, {}, {}
    for name in HOLDOUT:
        splits[name], records[name], rejected[name] = _generate_split(name, plan['splits'][name], forbidden, rules)
    manifest = dict(version=VERSION, claim_fingerprint=claim.fingerprint, plan_digest=_hash(plan),
                    rules_digest=rules.digest(), splits=records, historical_index=history,
                    rejected_normalized_duplicates=rejected, normalized_operator_disjoint=True,
                    split_use='single frozen evaluation; never model/teacher/KD/controller selection')
    manifest = json_safe(manifest)
    _write_json(final_path, manifest, exclusive=True)
    ledger['final_data_sha256'] = _file_hash(final_path)
    _write_json(output / 'final_claim.json', ledger)
    for examples in splits.values():
        for example in examples:
            example.research_final_claim = claim
            example.research_run_root = output
    return splits, manifest


def complete_final_evaluation(output_dir, claim, result_paths):
    """Seal final use after all requested results are durable; never unconsume it."""
    output = Path(output_dir).resolve()
    ledger = _check_claim(output, claim)
    if not (output / 'final_data_manifest.json').exists() or not result_paths:
        raise ValueError('Final data and complete durable result paths are required')
    ledger.update(status='completed', results={str(Path(p).resolve()): _file_hash(p) for p in result_paths},
                  final_data_sha256=_file_hash(output / 'final_data_manifest.json'))
    _write_json(output / 'final_claim.json', ledger)
    claim.close()
    return ledger
