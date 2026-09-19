"""Offline leaf-to-strategy calibration; runtime remains deterministic A-only.

No network fitting, change to feature thresholds, or numerical solves occur in
this module. Only a measured train/tune classical bank may supply evidence.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import math
import numpy as np

from .strong import load_strong_rules
from .config import AdaptiveConfig
from .acceptance import _successful
from .limited import digest_file
from ..provenance import write_json, json_safe


def _accurate(runs, expected):
    if len(runs) != expected or not runs:
        return False
    for run in runs:
        config = run.get('measurement_config')
        if not config or not _successful(run, AdaptiveConfig.from_dict(config)):
            return False
        checked = run.get('recomputed_true_residual')
        if not isinstance(checked, (float, int)) or not math.isfinite(checked) or not 0 <= checked <= run['stopping_threshold']:
            return False
    return True


def _time(runs):
    values = [run.get('wall_seconds') for run in runs]
    if not values or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in values):
        return None
    return float(np.median(values))


def calibrate_leaf_mapping(rows, rules, *, repeats, minimum_leaf_cases=3, tie_fraction=.01):
    """Minimize measured relative cost after failure preservation/success count.

    Candidates losing any fixed-C or incumbent-selector success are excluded. Among remaining
    strategies maximize success count, then minimize geometric relative time on
    the SAME fixed-success cohort. Keep the current leaf choice within a stated
    timing tie. Empty/undersampled leaves keep their predeclared default.
    """
    if repeats < 1 or minimum_leaf_cases < 1 or not 0 <= tie_fraction < 1:
        raise ValueError('invalid repetitions, sample minimum, or tie fraction')
    buckets = {key: [] for key in rules.rule_ids}
    seen = set()
    for row in rows:
        name = row['example']['name']
        if name in seen:
            raise ValueError('duplicate calibration case ID')
        seen.add(name)
        strong = row['runs']['strong_C']
        leaves = {r['selection']['rule_id'] for r in strong}
        hashes = {r['selection']['rules_digest'] for r in strong}
        if len(strong) != repeats or len(leaves) != 1 or hashes != {rules.digest()}:
            raise ValueError('recorded selector evidence differs from frozen input rules')
        leaf = next(iter(leaves))
        if leaf not in buckets:
            raise ValueError('unknown recorded rule leaf')
        buckets[leaf].append(row)
    selected = dict(rules.strategy_by_rule)
    reports = {}
    for leaf, cases in buckets.items():
        prior = selected[leaf]
        entry = dict(samples=len(cases), previous_strategy=prior, selected_strategy=prior,
                     status='unobserved_or_insufficient_samples', candidates=[])
        reports[leaf] = entry
        if len(cases) < minimum_leaf_cases:
            continue
        banks = [{k.removeprefix('oracle::') for k in row['runs'] if k.startswith('oracle::')} for row in cases]
        if not banks[0] or any(bank != banks[0] for bank in banks):
            raise ValueError('complete identical measured oracle bank required for each training case')
        if prior not in banks[0]:
            raise ValueError('current rule strategy missing from measured bank')
        fixed_success = {row['example']['name'] for row in cases if _accurate(row['runs']['fixed_C'], repeats)}
        prior_success = {row['example']['name'] for row in cases
                         if _accurate(row['runs']['oracle::' + prior], repeats)
                         or _accurate(row['runs']['strong_C'], repeats)}
        candidates = []
        for strategy in sorted(banks[0]):
            success, losses, prior_losses, log_costs = [], [], [], []
            valid_time = True
            for row in cases:
                name = row['example']['name']
                runs = row['runs']['oracle::' + strategy]
                if len(runs) != repeats:
                    raise ValueError('partial bank measurement cannot calibrate a rule')
                for run in runs:
                    if run.get('measurement_config', {}).get('mg', {}).get('strategy_name') != strategy:
                        raise ValueError('measured bank strategy/config mismatch')
                ok = _accurate(runs, repeats)
                if ok: success.append(name)
                if name in prior_success and not ok: prior_losses.append(name)
                if name in fixed_success:
                    if not ok:
                        losses.append(name)
                    else:
                        tc, tn = _time(row['runs']['fixed_C']), _time(runs)
                        if tc is None or tn is None: valid_time = False
                        else: log_costs.append(math.log(tn / tc))
            cost = float(np.exp(np.mean(log_costs))) if log_costs and valid_time else None
            candidates.append(dict(strategy=strategy, successes=len(success), success_case_ids=success,
                new_failure_case_ids=losses, common_reference_cases=len(fixed_success),
                new_failure_vs_incumbent_selector_case_ids=prior_losses,
                geometric_relative_time_on_fixed_successes=cost,
                eligible=not losses and not prior_losses and (not fixed_success or (valid_time and len(log_costs) == len(fixed_success)))))
        entry['candidates'] = candidates
        eligible = [c for c in candidates if c['eligible']]
        if not eligible:
            entry['status'] = 'no_candidate_preserves_reference_successes'; continue
        best_count = max(c['successes'] for c in eligible)
        finalists = [c for c in eligible if c['successes'] == best_count]
        if not fixed_success:
            # All fixed-C failures provide no comparable successful baseline time.
            # Rescue count may select a strategy, but never a faster failed run.
            chosen = next((c for c in finalists if c['strategy'] == prior), finalists[0])
            entry['timing_selection'] = 'not_available_no_fixed_success'
        else:
            chosen = min(finalists, key=lambda c: (c['geometric_relative_time_on_fixed_successes'], c['strategy']))
            old = next((c for c in finalists if c['strategy'] == prior), None)
            if old and old['geometric_relative_time_on_fixed_successes'] <= chosen['geometric_relative_time_on_fixed_successes'] / (1-tie_fraction):
                chosen = old
            entry['timing_selection'] = 'same_fixed_success_cohort'
        selected[leaf] = chosen['strategy']
        entry.update(selected_strategy=chosen['strategy'], status='offline_calibrated_on_training_cases',
                     fixed_success_cases=sorted(fixed_success), incumbent_selector_success_cases=sorted(prior_success))
    return selected, reports


def calibrate_from_run(input_run, research, rules, output, *, minimum_leaf_cases=3, tie_fraction=.01):
    run_dir, research, output = Path(input_run).resolve(), Path(research).resolve(), Path(output).resolve()
    if output == run_dir or run_dir in output.parents or output in run_dir.parents:
        raise ValueError('calibration output must be separate from input measurements')
    # Same guard as the benchmark, including staging/scratch workflows. Default
    # CLI results live under artifacts/my_v6_7_strong_baseline.
    for protected in (research, research.parent / 'my_v6_7_limited_fix'):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('calibration output must be separate from original artifacts')
    manifest_path, rows_path = run_dir / 'run_manifest.json', run_dir / 'raw_results.json'
    manifest = json.loads(manifest_path.read_text()); rows = json.loads(rows_path.read_text())
    if manifest.get('rules_digest') != rules.digest():
        raise ValueError('benchmark used different selector rules')
    progress = json.loads((run_dir / 'progress.json').read_text())
    if progress.get('status') != 'complete':
        raise ValueError('finish the bank benchmark before freezing a selector')
    provenance = manifest.get('provenance', {})
    split = provenance.get('split')
    if split not in ('train', 'tune'):
        raise ValueError('only original train/tune bank runs may calibrate; audit/fresh/smoke are forbidden')
    if not rows:
        raise ValueError('no calibration cases')
    saved = json.loads((research / 'split_manifest.json').read_text())['splits']
    settings = json.loads((research / 'configuration.json').read_text())
    baseline_mg = AdaptiveConfig.from_dict(settings['solver']).mg.to_dict()
    baseline_strategy = baseline_mg.pop('strategy_name')
    recorded_inputs = {r['name']:r for r in manifest.get('case_inputs', [])}
    if len(recorded_inputs) != len(rows):
        raise ValueError('complete recorded A/RHS/x0 inputs required for calibration')
    allowed_cases = {e['name']: e for e in saved[split]}
    forbidden = {e['normalized_operator_digest'] for e in saved.get('audit', [])}
    for row in rows:
        e = row['example']
        if json_safe(e) != allowed_cases.get(e['name']):
            raise ValueError('calibration case does not match original train/tune specification')
        if e['normalized_operator_digest'] in forbidden:
            raise ValueError('audit operator leakage into calibration')
        inputs = recorded_inputs.get(e['name'], {})
        if any(not inputs.get(k) for k in ('operator_digest', 'rhs_digest', 'x0_digest')):
            raise ValueError('missing calibration A/RHS/x0 provenance')
        for runs in row['runs'].values():
            if any(r.get(k) != inputs[k] for r in runs for k in ('operator_digest', 'rhs_digest', 'x0_digest')):
                raise ValueError('calibration run A/RHS/x0 digest mismatch')
        for arm, runs in row['runs'].items():
            if arm != 'fixed_C' and not arm.startswith('oracle::'):
                continue
            for run in runs:
                config = run.get('measurement_config', {})
                if arm == 'fixed_C' and config.get('mg', {}).get('strategy_name') != baseline_strategy:
                    raise ValueError('fixed_C must be the original fixed classical strategy')
                observed_mg = dict(config.get('mg', {})); observed_mg.pop('strategy_name', None)
                if observed_mg != baseline_mg or config.get('branch') != 'C':
                    raise ValueError('bank timings must use the unchanged classical stopping/work protocol')
    repeats = int(manifest['repeats'])
    mappings, reports = calibrate_leaf_mapping(rows, rules, repeats=repeats,
        minimum_leaf_cases=minimum_leaf_cases, tie_fraction=tie_fraction)
    evidence = dict(schema_version=1, calibration_kind='offline deterministic leaf-strategy mapping; no NN fitting',
        split=split, original_rules_digest=rules.digest(), input_run=str(run_dir),
        manifest_sha256=digest_file(manifest_path), raw_sha256=digest_file(rows_path),
        split_manifest_sha256=digest_file(research / 'split_manifest.json'),
        case_ids=[r['example']['name'] for r in rows],
        normalized_operator_digests=sorted({r['example']['normalized_operator_digest'] for r in rows}),
        minimum_leaf_cases=minimum_leaf_cases, tie_fraction=tie_fraction,
        objective='preserve every fixed-C and incumbent-selector success; maximize successes; minimize GM time on identical fixed-success cases',
        thresholds_changed=False, audit_used=False, independent_strength_validated=False,
        leaves=reports)
    provenance_text = json.dumps({k:v for k,v in evidence.items() if k != 'leaves'}, sort_keys=True)
    frozen = rules.replace_strategies(mappings, provenance=provenance_text)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a new calibration output directory; frozen rules are not overwritten')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'selector_rules.json', frozen.to_dict())
    write_json(output / 'calibration.json', dict(evidence, selected_rules_digest=frozen.digest()))
    return frozen, evidence
