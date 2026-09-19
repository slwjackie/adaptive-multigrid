"""Strict final-only assessment, separate from every legacy certificate path."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math

import numpy as np

from ..provenance import hardware_environment, json_safe
from .acceptance import _successful
from .research_data import _check_claim, _file_hash, _hash, _write_json
from .research_policy import _scope
from .research_runtime import is_teacher
from .strong_evaluation import _source_signature, _speed_summary

VERSION = 'strong-research-final-certificate-v1'


def _protocol_matches(run, cfg):
    try:
        actual = dict(run['measurement_config']['mg'])
        expected = dict(cfg.to_dict()['mg'])
        actual.pop('strategy_name', None)
        expected.pop('strategy_name', None)
        return bool(actual == expected and run['numerical_dtype'] == 'float64'
                    and run['inference_dtype'] == cfg.inference_dtype and run['regime'] == 'cold'
                    and run['rhs_count'] == 1 and run['selector_in_wall_time'])
    except (KeyError, TypeError):
        return False


def _successful_final_repeat(run, cfg, expected):
    """Re-check evidence; never trust saved success booleans or aggregate tables."""
    try:
        results = run['rhs_results']
        if len(results) != 1 or not _protocol_matches(run, cfg):
            return False
        record = results[0]
        checked = float(record['recomputed_true_residual'])
        wall = float(run['wall_seconds'])
        return bool(_successful(record, cfg) and math.isfinite(checked)
            and 0 <= checked <= record['stopping_threshold'] and math.isfinite(wall) and wall > 0
            and run['error'] is None and run['operator_digest'] == expected['operator_digest']
            and run['normalized_operator_digest'] == expected['normalized_operator_digest']
            and record['rhs_digest'] == expected['rhs_digest'] and record['exact_digest'] == expected['exact_digest']
            and record['x0_digest'] == hashlib.sha256(np.zeros(expected['case']['n'] ** 2, np.float64).tobytes()).hexdigest()
            and record['measurement_config'] == run['measurement_config']
            and run['rhs_digests'] == [expected['rhs_digest']] and run['exact_digests'] == [expected['exact_digest']])
    except (KeyError, ValueError, TypeError, IndexError, OverflowError):
        return False


def _accepted_neural(run):
    if len(run.get('rhs_results', [])) != 1:
        return False
    counters = run['rhs_results'][0].get('stats', {})
    accepted = sum(counters.get('branch_' + branch + '_cycles', 0) for branch in ('H_S', 'H_P', 'H_SP'))
    numerical_calls = sum(counters.get(name, 0) for name in ('neural_apply_calls', 'learned_transfer_apply_calls', 'learned_operator_applications'))
    return bool(accepted > 0 and numerical_calls > 0)


def _assess_final(rows, expected_cases, cfg, *, adaptive_arm, repeats,
                  min_cases=20, minimum_speedup=None, require_solution_error_not_worse=True):
    minimum_speedup = max(1.03, 1 / (1 - cfg.win_margin), float(minimum_speedup or 0))
    min_cases = max(20, int(min_cases))
    strong_success, neural_success, new_failures, paired, neural_cases, worse = [], [], [], [], [], []
    ratios = []
    protocol = True
    expected_by_name = {record['name']: record for record in expected_cases}
    names = [row['example']['name'] for row in rows]
    complete = bool(len(names) == len(expected_cases) and len(set(names)) == len(names)
                    and set(names) == set(expected_by_name)
                    and all(row.get('research_split') == 'final' for row in rows))
    unique = len({row['example']['normalized_operator_digest'] for row in rows})
    for row in rows:
        name = row['example']['name']
        expected = expected_by_name.get(name)
        if expected is None:
            protocol = False
            continue
        if any(row['example'].get(key) != expected[key] for key in ('operator_digest', 'normalized_operator_digest', 'case', 'seed', 'scale')):
            protocol = False
        try:
            classical = row['runs']['strong_C']['cold']['1']
            neural = row['runs'][adaptive_arm]['cold']['1']
        except KeyError:
            protocol = False
            continue
        if len(classical) != repeats or len(neural) != repeats:
            protocol = False
        protocol &= all(run.get('requested_branch') == 'C' for run in classical)
        protocol &= all(run.get('requested_branch') == 'auto' for run in neural)
        for run in classical + neural:
            protocol &= _protocol_matches(run, cfg) and not run.get('training_only', True)
        cs = len(classical) == repeats and all(_successful_final_repeat(run, cfg, expected) for run in classical)
        ns = len(neural) == repeats and all(_successful_final_repeat(run, cfg, expected) for run in neural)
        if cs:
            strong_success.append(name)
        if ns:
            neural_success.append(name)
        if cs and not ns:
            new_failures.append(name)
        if cs and ns:
            paired.append(name)
            ratios.append(float(np.median([r['wall_seconds'] for r in classical]) / np.median([r['wall_seconds'] for r in neural])))
            if any(_accepted_neural(run) for run in neural):
                neural_cases.append(name)
            if require_solution_error_not_worse:
                for metric in ('relative_solution_error', 'relative_energy_error'):
                    cv = [r['rhs_results'][0].get(metric) for r in classical]
                    nv = [r['rhs_results'][0].get(metric) for r in neural]
                    finite = all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in cv + nv)
                    if not finite or max(nv) > max(cv) * 1.05 + 1e-12:
                        worse.append(name)
                        break
    speed = _speed_summary(ratios)
    checks = dict(complete_precommitted_final_cohort=complete, unique_final_operators=unique == len(rows),
        enough_final_operators=len(rows) >= min_cases, enough_common_successes=len(paired) >= min_cases,
        minimum_five_repeats=repeats >= 5, same_fp64_tolerance=bool(protocol),
        successes_not_fewer=len(neural_success) >= len(strong_success), no_lost_strong_successes=not new_failures,
        actual_accepted_neural_work=bool(neural_cases),
        speedup_exceeds_required_margin=speed['geometric_speedup'] is not None and speed['geometric_speedup'] >= minimum_speedup,
        bootstrap_ci_lower_above_one=speed['ci95'][0] is not None and speed['ci95'][0] > 1.,
        manufactured_errors_not_worse=not worse)
    acceptance = dict(final_cases=len(rows), repeats=repeats, same_fp64_tolerance=bool(protocol),
        strong_successes=len(strong_success), neural_successes=len(neural_success), new_failures=len(new_failures),
        neural_used_cases=len(neural_cases), geometric_speedup=speed['geometric_speedup'], ci95=speed['ci95'],
        minimum_speedup=minimum_speedup, common_successes=len(paired),
        require_solution_error_not_worse=require_solution_error_not_worse,
        accuracy_worse_cases=len(worse), accuracy_slack=dict(relative=.05, absolute=1e-12))
    return dict(validated=all(checks.values()), acceptance=acceptance, checks=checks,
                new_failure_case_ids=new_failures, strong_success_case_ids=strong_success,
                neural_success_case_ids=neural_success, paired_case_ids=paired,
                actual_neural_case_ids=neural_cases, accuracy_worse_case_ids=worse)


def certify_research_final(run_root, claim, result, policy, cfg, rules, *, adaptive_arm='adaptive',
                          min_cases=20, minimum_speedup=None, require_solution_error_not_worse=True):
    """Write a provenance-pinned assessment and return a research-policy certificate.

    Call while the final claim is active, after all final measurements are saved,
    before complete_final_evaluation seals the run.  Untested/failed evidence
    returns validated=False; changed provenance or illegal final use raises.
    This function never changes any policy, checkpoint or legacy certificate.
    """
    root = Path(run_root).resolve()
    ledger = _check_claim(root, claim)
    frozen = json.loads(claim.freeze_path.read_text())
    frozen_config = frozen['config']
    if (frozen_config.get('solver') != cfg.to_dict() or frozen_config.get('policy_digest') != policy.digest()
            or frozen_config.get('rules_digest') != rules.digest()):
        raise ValueError('Frozen solver/policy/rules differ from the final assessment')
    if rules.digest() != frozen['rules_digest'] or policy.rules_digest != rules.digest() or policy.config_scope != _scope(cfg):
        raise ValueError('Policy/classical selector/configuration scope mismatch')
    environment = hardware_environment()
    if policy.hardware != environment:
        raise ValueError('Policy hardware differs from the measured/final hardware')
    if any(is_teacher(module) for model in policy.models.values() for module in (model.smoother, model.transfer)):
        raise ValueError('Teacher cannot be certified as a deployed Student')
    if any(policy.signatures.get(branch) != model.signature() for branch, model in policy.models.items()):
        raise ValueError('Expert/controller weights changed after policy fit')
    out = Path(result['output_dir']).resolve()
    if root not in out.parents:
        raise ValueError('Final evaluation results must belong to this research run')
    manifest_path, raw_path, comparison_path = (out / name for name in ('run_manifest.json', 'raw_results.json', 'comparison.json'))
    manifest = json.loads(manifest_path.read_text())
    rows = json.loads(raw_path.read_text())
    if json_safe(result['manifest']) != manifest or json_safe(result['rows']) != rows:
        raise ValueError('In-memory final result differs from durable raw measurements')
    if json.loads((out / 'progress.json').read_text()).get('status') != 'complete':
        raise ValueError('Final measurement is incomplete')
    if manifest['source_signature'] != _source_signature() or manifest['hardware'] != environment:
        raise ValueError('Final source/hardware differs from the measured manifest')
    if manifest['config'] != cfg.to_dict() or manifest['rules_digest'] != rules.digest():
        raise ValueError('Final numerical protocol/selector changed')
    final_protocol = frozen_config.get('final_evaluation', {})
    for key in ('repeats', 'regimes', 'rhs_counts'):
        if key not in final_protocol or json_safe(final_protocol[key]) != manifest[key]:
            raise ValueError('Final workload was not exactly precommitted before testing')
    required_speedup = max(1.03, 1 / (1 - cfg.win_margin), float(minimum_speedup or 0))
    if (bool(final_protocol.get('require_solution_error_not_worse', True)) != require_solution_error_not_worse
            or max(20, int(min_cases)) != int(final_protocol.get('min_cases', 20))
            or required_speedup != float(final_protocol.get('minimum_speedup', max(1.03, 1 / (1 - cfg.win_margin))))):
        raise ValueError('Final acceptance criteria differ from precommitted freeze settings')
    if 'cold' not in manifest['regimes']:
        raise ValueError('Cold single-RHS time-to-tolerance is required')
    adaptive = manifest['arms'].get(adaptive_arm, {})
    classical = manifest['arms'].get('strong_C', {})
    if (adaptive.get('training_only', True) or adaptive.get('branch') != 'auto'
            or adaptive.get('expected_rhs_mode', 'actual') != 'actual'
            or adaptive.get('policy_signature') != policy.digest() or not adaptive.get('selector')
            or classical.get('branch') != 'C' or not classical.get('selector') or classical.get('model_signature') is not None):
        raise ValueError('Assessment must compare the frozen adaptive Student policy with exact strong_C')
    if any(arm.get('training_only') for arm in manifest['arms'].values()):
        raise ValueError('Final evaluation cannot include training-only Teacher arms')
    final_manifest = json.loads((root / 'final_data_manifest.json').read_text())
    if final_manifest['claim_fingerprint'] != claim.fingerprint:
        raise ValueError('Final data do not belong to the active single-use claim')
    expected = final_manifest['splits']['final']
    evidence = _assess_final(rows, expected, cfg, adaptive_arm=adaptive_arm, repeats=manifest['repeats'],
        min_cases=min_cases, minimum_speedup=minimum_speedup,
        require_solution_error_not_worse=require_solution_error_not_worse)
    assessment = json_safe(dict(schema=1, version=VERSION, **evidence,
        adaptive_arm=adaptive_arm, final_claim_fingerprint=claim.fingerprint,
        freeze_sha256=_file_hash(claim.freeze_path), final_data_sha256=_file_hash(root / 'final_data_manifest.json'),
        raw_evidence={name: dict(path=str(path), sha256=_file_hash(path)) for name, path in
                      (('manifest', manifest_path), ('raw_results', raw_path), ('comparison', comparison_path))},
        policy_digest=policy.digest(), rules_digest=rules.digest(), config_scope=_scope(cfg),
        hardware=environment, source_signature=manifest['source_signature'],
        generation_signatures={branch: model.generation_signature() for branch, model in policy.models.items()},
        statistical_scope='precommitted final operator cohort; paired successful-operator bootstrap; no universal convergence/hardware claim',
        legacy_certificate_modified=False))
    assessment_path = root / 'final_assessment.json'
    if assessment_path.exists():
        if json.loads(assessment_path.read_text()) != assessment:
            raise ValueError('Final assessment already committed; acceptance criteria cannot be changed post-test')
    else:
        _write_json(assessment_path, assessment, exclusive=True)
    certificate = dict(schema=1, version=VERSION, validated=evidence['validated'],
        policy_digest=policy.digest(), rules_digest=rules.digest(), config_scope=_scope(cfg), hardware=environment,
        final_freeze=dict(path=str(claim.freeze_path), sha256=_file_hash(claim.freeze_path)),
        final_result=dict(path=str(assessment_path), sha256=_file_hash(assessment_path)),
        acceptance=evidence['acceptance'], checks=evidence['checks'],
        final_claim_fingerprint=claim.fingerprint, generation_signatures=assessment['generation_signatures'],
        final_time_improvement_certified=evidence['validated'], universal_convergence_guarantee=False,
        reason='untouched_final_pass' if evidence['validated'] else 'untouched_final_nonpassing_or_insufficient_evidence')
    return json_safe(certificate)
