"""Read-only inventory and evidence-based diagnosis of preserved v6.7 runs.

This module never saves or alters checkpoints. Missing original snapshots remain
missing even when their exact weight signature can be reconstructed from a seed.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from ..provenance import module_signature, write_json
from .models import Components

STAGES = ('smoother', 'transfer', 'joint', 'hierarchical')
COMPONENTS = ('smoother', 'transfer', 'detector', 'controller')


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def assert_new_output(research, output):
    """Resolve symlinks before enforcing the original artifact read-only boundary."""
    research, output = Path(research).resolve(), Path(output).resolve()
    if output == research or research in output.parents:
        raise ValueError('output must not be inside the original research artifacts')
    return output


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preservation_manifest(paths):
    return {'schema_version': 1, 'scope': 'explicit important files only; no recursive artifact scan',
            'files': [{'path': str(path.resolve()), 'size_bytes': path.stat().st_size,
                       'sha256': sha256_file(path)}
                      for path in sorted({Path(p) for p in paths if Path(p).is_file()})]}


def verify_preservation(manifest):
    if isinstance(manifest, (str, Path)):
        manifest = read_json(manifest)
    results = []
    for row in manifest['files']:
        path = Path(row['path'])
        actual = sha256_file(path) if path.is_file() else None
        results.append({'path': str(path), 'unchanged': actual == row['sha256'],
                        'expected_sha256': row['sha256'], 'actual_sha256': actual})
    return {'all_unchanged': all(row['unchanged'] for row in results), 'files': results}


def tensor_weight_signatures(model):
    """Hash real named parameter bytes only, without metadata or architecture attrs."""
    result = {}
    for name in COMPONENTS:
        h = hashlib.sha256()
        for key, value in sorted(getattr(model, name).named_parameters()):
            value = value.detach().cpu().contiguous()
            for part in (key, str(value.dtype), repr(tuple(value.shape))):
                h.update(part.encode())
            h.update(value.numpy().tobytes())
        result[name] = h.hexdigest()
    return result


def component_signatures(model):
    return {name: module_signature(getattr(model, name)) for name in COMPONENTS}


def archived_model_signatures(rows, arm='previous'):
    signatures = set()
    total = 0
    missing = 0
    for row in rows:
        for run in row.get('runs', {}).get(arm, []):
            total += 1
            parts = str(run.get('cache_key', '')).split(':', 2)
            if len(parts) != 3 or not parts[1]:
                missing += 1
            else:
                signatures.add(parts[1])
    return {'signatures': sorted(signatures), 'runs': total, 'missing': missing}


def recover_initial_entry(research):
    """Recreate initial weights only when archived *full* signatures prove equality.

    A reconstructed model is not an original checkpoint file. No final or rejected
    candidate is substituted for a missing entry checkpoint.
    """
    research = Path(research)
    settings = read_json(research / 'configuration.json', {})
    if 'seed' not in settings or 'hidden' not in settings:
        return None, {'verified': False, 'reason': 'missing initialization seed/hidden'}
    with torch.random.fork_rng(devices=[]):
        model = Components.create(hidden=int(settings['hidden']), seed=int(settings['seed']))
    signature = model.signature()
    checks = {}
    for stage in STAGES[:3]:
        raw = read_json(research / stage / 'validation/raw.json', [])
        recorded = archived_model_signatures(raw)
        status = read_json(research / stage / 'status.json', {})
        checks[stage] = dict(recorded,
            every_recorded_entry_matches=bool(recorded['runs'] and not recorded['missing']
                and recorded['signatures'] == [signature]),
            recorded_stage_rejected=status.get('acceptance', {}).get('accepted') is False)
    verified = all(v['every_recorded_entry_matches'] and v['recorded_stage_rejected']
                   for v in checks.values())
    evidence = {'verified': verified, 'role': 'reconstructed_entry', 'original_snapshot_exists': False,
        'path': None, 'seed': int(settings['seed']), 'hidden': int(settings['hidden']),
        'method': 'Components.create; verified against every archived previous-arm full model cache signature',
        'checks': checks, 'component_signatures': component_signatures(model),
        'tensor_weight_signatures': tensor_weight_signatures(model),
        'metadata_limit': 'Exact original metadata is not recovered; tensor and generation-spec signatures are verified.',
        'not_an_original_checkpoint': True}
    return (model if verified else None), evidence


def aggregate_stats(rows, arm):
    totals = {'stats': {}, 'setup_stats': {}}
    runs = 0
    for row in rows:
        for run in row.get('runs', {}).get(arm, []):
            runs += 1
            for namespace in totals:
                for name, value in run.get(namespace, {}).items():
                    if isinstance(value, (int, float)):
                        totals[namespace][name] = totals[namespace].get(name, 0) + value
    return {'measured_runs': runs, **totals}


def diagnose_original_stage(stage, rows, settings, incumbent=None, candidate=None):
    """Unfold the actual original boolean acceptance, without changing its result."""
    common, previous_success, candidate_success, lost, rescued, both = [], [], [], [], [], []
    ratios = []
    neural_cases = 0
    for index, row in enumerate(rows):
        case_id = row.get('example', {}).get('name', str(index))
        previous = row['runs']['previous']; new = row['runs']['candidate']
        ps = bool(previous) and all(r['converged'] for r in previous)
        ns = bool(new) and all(r['converged'] for r in new)
        if ps: previous_success.append(case_id)
        if ns: candidate_success.append(case_id)
        if ps and ns:
            common.append(case_id)
            ratios.append(float(np.median([r['wall_seconds'] for r in previous])) /
                          float(np.median([r['wall_seconds'] for r in new])))
        elif ps: lost.append(case_id)
        elif ns: rescued.append(case_id)
        else: both.append(case_id)
        neural_cases += int(any(r.get('stats', {}).get('accepted_neural_cycles', 0) > 0 for r in new))
    psig = archived_model_signatures(rows, 'previous')
    nsig = archived_model_signatures(rows, 'candidate')
    signatures_known = bool(psig['runs'] and nsig['runs'] and not psig['missing'] and not nsig['missing'])
    signature_changed = (psig['signatures'] != nsig['signatures']) if signatures_known else None
    gm = float(np.exp(np.mean(np.log(ratios)))) if ratios else None
    required = 1 / (1 - float(settings.get('stage_time_margin', .03)))
    checks = {'model_signature_changed': signature_changed, 'actual_neural_use': neural_cases > 0,
              'no_new_failures': len(lost) == 0, 'all_cases_common_success': len(common) == len(rows),
              'geometric_speedup_available': gm is not None,
              'speedup_exceeds_required': gm > required if gm is not None else False}
    changed = None
    if incumbent is not None and candidate is not None:
        trained = {'smoother': ['smoother'], 'transfer': ['transfer'],
                   'joint': ['smoother', 'transfer'],
                   'hierarchical': ['smoother', 'transfer', 'detector']}[stage]
        before, after = tensor_weight_signatures(incumbent), tensor_weight_signatures(candidate)
        changed = {name: {'before': before[name], 'after': after[name],
                          'weights_changed': before[name] != after[name]} for name in trained}
    return {'stage': stage, 'original_checks': checks,
        'original_failed_conditions': [name for name, passed in checks.items() if passed is False],
        'unknown_conditions': [name for name, passed in checks.items() if passed is None],
        'recomputed_original_accepted': all(v is True for v in checks.values()),
        'incumbent_successes': len(previous_success), 'candidate_successes': len(candidate_success),
        'common_successes': len(common), 'total': len(rows), 'new_failure_case_ids': lost,
        'rescued_case_ids': rescued, 'both_failed_case_ids': both,
        'original_minimum_common_samples': len(rows),
        'geometric_speedup_vs_incumbent': gm, 'required_speedup': required,
        'neural_used_cases': neural_cases,
        'neural_use_definition': 'any accepted_neural_cycles > 0 in candidate measured repeats; not a setup/attempt counter',
        'candidate_counters': aggregate_stats(rows, 'candidate'),
        'incumbent_counters': aggregate_stats(rows, 'previous'),
        'actual_trained_weight_comparison': changed,
        'actual_weight_change_was_original_condition': False,
        'original_signature_includes': 'named tensor states and model generation attributes; not metadata',
        'scope': 'diagnosis of original archived timings versus incumbent; no classical speedup or new certification claim'}


def _source_inventory(project):
    wanted = {'stage_acceptance': ('src/adaptive_mg/v67/training.py', 'accept_operator'),
              'stage_checkpoint_flow': ('src/adaptive_mg/v67/training.py', 'pipeline'),
              'final_certificate': ('src/adaptive_mg/v67/evaluation.py', 'certify'),
              'measurement': ('src/adaptive_mg/v67/evaluation.py', 'measured'),
              'solver': ('src/adaptive_mg/v67/solver.py', 'PreparedAdaptiveMG'),
              'bank_cache': ('src/adaptive_mg/v67/solver.py', 'ensure_learned'),
              'bank_generation': ('src/adaptive_mg/v67/banks.py', 'prepare_learned_bank'),
              'checkpoint_save_load': ('src/adaptive_mg/v67/models.py', 'Components')}
    result = {}
    for role, (relative, name) in wanted.items():
        path = project / relative
        if not path.is_file():
            result[role] = {'path': str(path), 'exists': False}; continue
        source = path.read_text(); tree = ast.parse(source)
        found = [node for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
        result[role] = {'path': str(path), 'symbol': name, 'line': found[0].lineno if found else None,
                        'symbol_found': bool(found)}
        if role == 'stage_acceptance' and found:
            assignments = [n for n in ast.walk(found[0]) if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'accepted' for t in n.targets)]
            result[role]['accepted_expression'] = ast.unparse(assignments[0].value) if assignments else None
    return result


def _checkpoint_record(path, stage, role, settings, acceptance, *, resolution=None):
    record = {'checkpoint_id': f'{stage}:{role}', 'stage': stage, 'role': role,
              'path': str(path.resolve()), 'exists': path.is_file(), 'acceptance': acceptance,
              'config': settings.get('solver'), 'config_source': 'research/configuration.json',
              'config_embedded_in_checkpoint': False}
    if resolution: record['resolution'] = resolution
    if not path.is_file():
        record['missing_reason'] = 'original checkpoint file not preserved at expected role path'
        return record
    info = path.stat()
    record.update(size_bytes=info.st_size,
                  mtime_utc=datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
                  created_utc=(datetime.fromtimestamp(info.st_birthtime, timezone.utc).isoformat()
                               if hasattr(info, 'st_birthtime') else None),
                  creation_time_source='filesystem birth time where available; not embedded training timestamp')
    try:
        model = Components.load(path)
        record.update(component_signatures=component_signatures(model),
                      tensor_weight_signatures=tensor_weight_signatures(model),
                      model_signature=model.signature(), metadata=model.metadata)
    except (ValueError, RuntimeError, KeyError) as exc:
        record['load_error'] = str(exc)
    return record


def inspect_candidates(research, output, project_root=None):
    research = Path(research).resolve()
    output = assert_new_output(research, output)
    project = Path(project_root).resolve() if project_root else research.parent.parent
    settings = read_json(research / 'configuration.json')
    if settings is None: raise FileNotFoundError(research / 'configuration.json')
    initial, initial_evidence = recover_initial_entry(research)
    records, stages = [], {}
    important = [research / name for name in ('configuration.json', 'environment.json', 'split_manifest.json',
        'certificate.json', 'pipeline_progress.json', 'audit/raw_results.json', 'audit/summary.json')]
    for stage in STAGES:
        status_path = research / stage / 'status.json'
        status = read_json(status_path, {})
        acceptance = status.get('acceptance', read_json(research / stage / 'validation/acceptance.json', {}))
        entry_path = research / stage / 'entry.pt'
        resolution = None
        if stage == 'hierarchical' and not entry_path.is_file():
            controller_path = research / 'controller/controller.pt'
            raw = read_json(research / stage / 'validation/raw.json', [])
            old = archived_model_signatures(raw)
            if controller_path.is_file() and old['runs'] and not old['missing']:
                controller = Components.load(controller_path)
                if old['signatures'] == [controller.signature()]:
                    entry_path = controller_path
                    resolution = 'preserved controller checkpoint equals every archived hierarchical previous-arm full signature'
        for role, path in [('entry', entry_path), ('candidate', research / stage / 'candidate.pt'),
                           ('promoted', research / stage / 'promoted.pt')]:
            records.append(_checkpoint_record(path, stage, role, settings, acceptance,
                                              resolution=resolution if role == 'entry' else None))
            important.append(path)
        candidate_path = research / stage / 'candidate.pt'
        candidate = Components.load(candidate_path) if candidate_path.is_file() else None
        incumbent = Components.load(entry_path) if entry_path.is_file() else initial if stage != 'hierarchical' else None
        raw_path = research / stage / 'validation/raw.json'
        diagnostic = diagnose_original_stage(stage, read_json(raw_path, []), settings, incumbent, candidate)
        diagnostic.update(recorded_status=status, recorded_acceptance=acceptance,
                          agreement_with_recorded=diagnostic['recomputed_original_accepted'] == acceptance.get('accepted'))
        stages[stage] = diagnostic
        important.extend([status_path, raw_path, research / stage / 'validation/acceptance.json',
                          research / stage / 'resume.pt'])
    for stage, role, relative in [('pipeline', 'final', 'checkpoint_v6_7_final.pt'),
        ('pipeline', 'current', 'current.pt'), ('detector', 'policy', 'detector/detector.pt'),
        ('controller', 'policy', 'controller/controller.pt')]:
        path = research / relative
        records.append(_checkpoint_record(path, stage, role, settings, None)); important.append(path)
    for name in ('initial_v6_7.pt', 'from_v6_6_operators.pt'):
        path = project / 'checkpoints' / name
        records.append(_checkpoint_record(path, 'preexisting', name, settings, None)); important.append(path)
    for name in ('detector', 'controller'):
        important.append(research / name / 'status.json')
    source_inventory = _source_inventory(project)
    source_paths = [Path(v['path']) for v in source_inventory.values()]
    source_paths.extend([project / 'configs/v6_7_research.json', project / 'src/adaptive_mg/v67/config.py',
                         project / 'src/adaptive_mg/v67/data.py', project / 'src/adaptive_mg/data.py',
                         project / 'src/adaptive_mg/provenance.py'])
    audit = read_json(research / 'audit/raw_results.json', [])
    original_import = subprocess.run([sys.executable, '-c',
        'import sys,json,adaptive_mg,adaptive_mg.v67.models; print(json.dumps({"python":sys.executable,"adaptive_mg":adaptive_mg.__file__,"models":adaptive_mg.v67.models.__file__}))'],
        cwd=project, env={**os.environ, 'PYTHONPATH': str(project / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'},
        text=True, capture_output=True, check=True)
    import adaptive_mg
    registry = {'schema_version': 1, 'research_root': str(research), 'project_root': str(project),
                'configuration': settings, 'checkpoints': records,
                'reconstructed_initial_entry': initial_evidence,
                'checkpoint_policy': 'references only; missing roles remain explicit; no candidate duplicated or overwritten'}
    diagnosis = {'schema_version': 1, 'research_root': str(research), 'stages': stages,
        'source_locations': source_inventory,
        'runtime': {'inspection_python': sys.executable, 'inspection_import': adaptive_mg.__file__,
                    'expected_original_python': str(project / '.venv/bin/python'),
                    'original_import_probe': json.loads(original_import.stdout)},
        'classical_baseline_from_saved_config': settings['solver']['mg'],
        'saved_config_matches_project_research_config': settings == read_json(project / 'configs/v6_7_research.json'),
        'detector_status': read_json(research / 'detector/status.json'),
        'controller_status': read_json(research / 'controller/status.json'),
        'final_certificate': read_json(research / 'certificate.json'),
        'final_audit_counters': {arm: aggregate_stats(audit, arm) for arm in ('classical', 'adaptive')},
        'case_reconstruction': {'split_manifest': str(research / 'split_manifest.json'),
            'saved_generation_seed': settings.get('seed'), 'tune_seed_offset': 100001, 'audit_seed_offset': 200002,
            'operator_specs_and_digests': 'saved in split manifest and raw rows',
            'rhs_digest_saved': False, 'x0_saved': False,
            'reconstruction_requirement': 'rebuild with original dataset generator/settings; verify every operator digest; label RHS as source/seed-reconstructed since original RHS digest was not saved',
            'audit_scope': 'diagnostic re-evaluation of previously inspected audit'},
        'artifacts_found': [{'path': str(path), 'exists': path.is_file(),
                             'size_bytes': path.stat().st_size if path.is_file() else None}
                            for path in important],
        'interpretation': 'Stage ratios compare candidate to stage incumbent. Final certificate remains false; new stage promotion is not classical speedup evidence.'}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'checkpoint_registry.json', registry)
    write_json(output / 'original_acceptance_diagnosis.json', diagnosis)
    write_json(output / 'source_baseline_hashes.json', dict(preservation_manifest(source_paths),
        scope='source comparison baseline before authorized edits; source changes are expected'))
    manifest = preservation_manifest(important)
    write_json(output / 'original_preservation_hashes.json', manifest)
    verification = verify_preservation(manifest)
    write_json(output / 'original_preservation_verification.json', verification)
    lines = ['# 원본 v6.7 진단', '', f'원본: `{research}`', '',
        '원본 checkpoint/status/certificate를 수정하지 않았다. 아래 속도는 classical 대비가 아니라 stage incumbent 대비다.', '',
        '| Stage | common/total | speedup vs incumbent | 원래 실패 조건 |', '|---|---:|---:|---|']
    for stage, item in stages.items():
        gm = item['geometric_speedup_vs_incumbent']
        lines.append(f"| {stage} | {item['common_successes']}/{item['total']} | {gm:.9f} | {', '.join(item['original_failed_conditions'])} |" if gm is not None else f'| {stage} | 자료 없음 | — | 자료 없음 |')
    lines += ['', 'smoother/joint의 공통 실패 사례는 JSON의 both_failed_case_ids에 보존했다.',
        '각 stage의 entry/promoted 파일 유무는 registry에 명시했다. 연구 초기화 seed로 재구성한 entry는 원본 파일로 표시하지 않는다.',
        '재구성된 초기 모델은 archived previous 실행의 전체 모델 signature와 대조했다.',
        'audit의 setup/attempted/accepted/rejected/controller/detector 카운터는 JSON에서 각각 합산했다.',
        '최종 certificate 조건과 기존 rejected 결과는 변경하지 않았다.', '']
    (output / 'inspection_summary_ko.md').write_text('\n'.join(lines))
    return {'output': str(output), 'registry': registry, 'diagnosis': diagnosis,
            'preservation': {'file_count': len(manifest['files']), 'all_unchanged': verification['all_unchanged']}}
