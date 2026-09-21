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


def measure_classical_portfolio(examples, cfg, rules, *, repeats=3, rhs_count=2, bank='controlled', seed=2209):
    """Cold per-A portfolio costs; all distinct RHS must converge.

    Selection features/labels are recorded once outside candidate timing, since
    their cost is common to every possible leaf choice. Final C/H benchmarking
    separately charges real selector construction to both algorithms.
    """
    from time import perf_counter
    from .strong import classical_bank, select_strong_strategy
    from .solver import PreparedAdaptiveMG
    from .research_evaluation import manufactured_rhs
    from ..provenance import stable_norm
    if repeats < 1 or rhs_count < 1:
        raise ValueError('positive repeats and RHS count required')
    rng=np.random.default_rng(seed)
    strategies=tuple(s.name for s in classical_bank(bank))
    if cfg.mg.strategy_name not in strategies:
        raise ValueError('portfolio must include the exact fixed_C reference')
    rows=[]
    for e in examples:
        selection=select_strong_strategy(e.a,e.n,rules)
        bs,_=manufactured_rhs(e,rhs_count)
        runs={s:[] for s in strategies}
        for _ in range(repeats):
            for strategy in rng.permutation(strategies):
                chosen=replace(cfg,mode='classical',branch='C',
                               mg=replace(cfg.mg,strategy_name=str(strategy)))
                started=perf_counter()
                try:
                    prepared=PreparedAdaptiveMG(e.a,e.n,config=chosen)
                    result=prepared.solve_many(bs)
                    seconds=perf_counter()-started
                    true=[stable_norm(b-e.a@r.x) for b,r in zip(bs,result)]
                    ok=len(result)==rhs_count and all(r.converged and np.isfinite(v)
                        and v<=r.stopping_threshold and r.executed_cycles<=cfg.mg.max_cycles
                        for r,v in zip(result,true))
                    record=dict(seconds=seconds,success=bool(ok),
                                cycles=[r.executed_cycles for r in result],true_residual=true,
                                thresholds=[r.stopping_threshold for r in result])
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    record=dict(seconds=perf_counter()-started,success=False,cycles=[],error=str(exc))
                runs[str(strategy)].append(record)
        rows.append(dict(name=e.name,normalized_operator_digest=e.group_digest,
                         operator_digest=e.digest,n=e.n,rule_id=selection.rule_id,
                         rules_digest=rules.digest(),rhs_count=rhs_count,runs=runs))
    return rows


def calibrate_multisize(train_rows, tune_rows, rules, *, required_sizes,
                        fixed_strategy='line_alt_bilinear_full', minimum_leaf_cases=6,
                        max_cycles=150, cycle_margin=.8, tie_fraction=.01):
    """Fit on train, admit on tune; uncovered leaves use the exact fixed parent.

    Sample counts are independent operators, NOT timing repetitions or RHS.
    A candidate must solve every recorded RHS/repetition with budget headroom
    on both sets. This is empirical development evidence, not a universal
    convergence guarantee and not permission to tune on final/OOD results.
    """
    from .strong import classical_bank
    sizes=tuple(sorted(set(required_sizes)))
    if (not train_rows or not tune_rows or len(sizes)<2 or minimum_leaf_cases<1
            or max_cycles<1 or not 0<cycle_margin<=1 or not 0<=tie_fraction<1):
        raise ValueError('multi-size disjoint train/tune and valid margins required')
    if (any(not isinstance(n,int) or isinstance(n,bool) or n<3 or n&(n+1) for n in sizes)
            or any(b!=2*a+1 for a,b in zip(sizes,sizes[1:]))):
        raise ValueError('coverage needs consecutive 2**L-1 grid sizes')
    if fixed_strategy not in {s.name for s in classical_bank('all')}:
        raise ValueError('unknown fixed strategy')
    seen=set()
    for rows in (train_rows,tune_rows):
        for row in rows:
            if row['normalized_operator_digest'] in seen:
                raise ValueError('calibration operator overlap/duplicate')
            seen.add(row['normalized_operator_digest'])
            if row['rules_digest']!=rules.digest() or row['rule_id'] not in rules.rule_ids:
                raise ValueError('stale selector evidence')
            if fixed_strategy not in row['runs']:
                raise ValueError('exact fixed_C missing from portfolio')
            for runs in row['runs'].values():
                if not runs or any(not np.isfinite(r['seconds']) or r['seconds']<=0 for r in runs):
                    raise ValueError('invalid or incomplete measured timing')
    mapping={leaf:fixed_strategy for leaf in rules.rule_ids}
    reports={};coverage=[]
    def acceptable(rows,strategy):
        for row in rows:
            runs=row['runs'].get(strategy,[])
            if (not runs or any(not r['success'] or len(r['cycles'])!=row['rhs_count']
                 or max(r['cycles'],default=max_cycles+1)>cycle_margin*max_cycles for r in runs)):
                return False
        return True
    def cost(rows,strategy):
        return float(np.mean([np.log(np.median([r['seconds'] for r in row['runs'][strategy]])) for row in rows]))
    for leaf in rules.rule_ids:
        train=[r for r in train_rows if r['rule_id']==leaf]
        tune=[r for r in tune_rows if r['rule_id']==leaf]
        report=dict(train_operators=len(train),tune_operators=len(tune),
                    train_sizes=sorted({r['n'] for r in train}),tune_sizes=sorted({r['n'] for r in tune}),
                    selected_strategy=fixed_strategy,status='fixed_fallback_insufficient_coverage')
        reports[leaf]=report
        if (min(len(train),len(tune))<minimum_leaf_cases
                or not set(sizes).issubset(report['train_sizes'])
                or not set(sizes).issubset(report['tune_sizes'])):
            continue
        candidates=set.intersection(*(set(row['runs']) for row in train+tune))
        candidates=[s for s in candidates if acceptable(train,s)]
        if not candidates:
            report['status']='fixed_fallback_no_train_candidate_with_headroom'
            continue
        best=min(candidates,key=lambda s:(cost(train,s),s))
        if fixed_strategy in candidates and cost(train,fixed_strategy)<=cost(train,best)-np.log1p(-tie_fraction):
            best=fixed_strategy
        # Tune is an independent pass/fail gate, never a re-ranking of timings.
        if not acceptable(tune,best):
            report['status']='fixed_fallback_tune_robustness_rejection'
            continue
        mapping[leaf]=best
        coverage.append((leaf,min(sizes),max(sizes)))
        report.update(selected_strategy=best,status='train_selected_tune_admitted',
                      train_geometric_seconds=float(np.exp(cost(train,best))),
                      tune_geometric_seconds=float(np.exp(cost(tune,best))))
    evidence=dict(version=2,leaves=reports,required_sizes=list(sizes),
                  independent_train_operators=len(train_rows),independent_tune_operators=len(tune_rows),
                  minimum_leaf_cases=minimum_leaf_cases,cycle_margin=cycle_margin,max_cycles=max_cycles,
                  fixed_reference=fixed_strategy,all_rhs_required=True,final_used=False,
                  universal_convergence_guarantee=False)
    frozen=replace(rules,strategy_by_rule=mapping,require_coverage=True,
                   coverage_by_rule=tuple(coverage),fallback_strategy_name=fixed_strategy,
                   provenance='v2: independent multi-size train selection and tune headroom gate; unobserved sizes/leaves use fixed_C; no final evidence')
    return frozen,evidence
