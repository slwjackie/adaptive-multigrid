"""ONE tuned classical plan, selected on fit and admitted on tune.

This is not the per-operator strong selector. No test labels or runtime timings
choose a plan. The 88-plan selector remains a separately labelled audit.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import torch

from ...provenance import hardware_environment, json_safe
from ..config import AdaptiveConfig
from ..strong import StrongRules, load_strong_rules
from ..strong_calibration import measure_classical_portfolio, calibrate_multisize
from ..research_data import FAMILIES, _generate_split, _restore, historical_operator_index
from ..three_pillars import _source_digest
from ..limited import initialize_timing_runtime
from ..world_model.data import digest, file_hash, write_json

VERSION = 'hs-world-thesis-v1'


def read(path):
    return json.loads(Path(path).read_text())


def fixed_rules(plan):
    r = StrongRules()
    return replace(r, strategy_by_rule=tuple((key, plan) for key in r.rule_ids),
                   fallback_strategy_name=plan, require_coverage=False,
                   coverage_by_rule=(), provenance='C_tuned: single fixed plan; not per-A optimality')


def load(output, *, calibrated=True):
    out = Path(output).resolve()
    settings = read(out/'configuration.json')
    if settings.get('thesis', {}).get('version') != VERSION:
        raise ValueError('use a new H_S-only thesis run')
    torch.set_num_threads(int(settings.get('torch_threads', 1)))
    initialize_timing_runtime()
    h = read(out/'thesis_manifest.json')
    if h['source_digest'] != _source_digest() or h['settings_digest'] != digest(settings):
        raise ValueError('source/config changed; create a new run')
    if h['hardware'] != hardware_environment(refresh=True):
        raise ValueError('timing environment changed; create a new run')
    cfg = AdaptiveConfig.from_dict(settings['solver'])
    if calibrated:
        ev = read(out/'tuned_classical.json')
        if ev['measurements_sha256'] != file_hash(out/'classical_measurements.json'):
            raise ValueError('classical tuning evidence changed')
        rules = load_strong_rules(out/'selector_rules.json')
        if rules.digest() != ev['rules_digest'] or rules != fixed_rules(ev['plan']):
            raise ValueError('C_tuned plan/rules changed')
        cfg = replace(cfg, mg=replace(cfg.mg, strategy_name=ev['plan']))
    else:
        rules = fixed_rules(cfg.mg.strategy_name)
    return out, settings, cfg, rules


def assert_open(out):
    if (Path(out)/'freeze.json').exists() or (Path(out)/'test_claim.json').exists():
        raise ValueError('thesis is frozen; development is closed')


def fit_global(rows, tune, candidates, anchor):
    """Choose on fit only, validate once on tune; rejection uses fixed anchor.

    Failure timings are never treated as successful timings. Speed ranking uses
    the SAME anchor-success fit operators. Tune cannot select a runner-up.
    """
    a = {r['normalized_operator_digest'] for r in rows}
    b = {r['normalized_operator_digest'] for r in tune}
    if not a or not b or a & b:
        raise ValueError('nonempty operator-disjoint fit/tune required')
    def ok(row, plan):
        runs = row['runs'][plan]
        return bool(runs) and all(v['success'] for v in runs)
    anchor_ok = [ok(r, anchor) for r in rows]
    scores = {}
    for plan in candidates:
        successes = [ok(r, plan) for r in rows]
        loss = any(c and not h for c,h in zip(anchor_ok, successes))
        cohort = [i for i,c in enumerate(anchor_ok) if c]
        ratios = [np.median([v['seconds'] for v in rows[i]['runs'][plan]]) /
                  np.median([v['seconds'] for v in rows[i]['runs'][anchor]]) for i in cohort]
        if not ratios or not np.isfinite(ratios).all() or min(ratios) <= 0:
            score = None
        else:
            score = float(np.exp(np.log(ratios).mean()))
        scores[plan] = dict(successes=sum(successes), total=len(rows), new_failure_vs_anchor=loss,
                            paired_time_ratio=score)
    eligible = [p for p in candidates if not scores[p]['new_failure_vs_anchor'] and scores[p]['paired_time_ratio'] is not None]
    if not eligible:
        raise ValueError('no validated fit successes; improve classical candidate set, not test-time selection')
    selected = min(eligible, key=lambda p: (-scores[p]['successes'], scores[p]['paired_time_ratio'], p))
    admitted = all(not ok(r, anchor) or ok(r, selected) for r in tune)
    final = selected if admitted else anchor
    return dict(plan=final, fit_winner=selected, tune_admitted=admitted, fit_scores=scores,
                tune_successes=sum(ok(r, final) for r in tune), tune_total=len(tune),
                fit_operator_ids=sorted(a), tune_operator_ids=sorted(b),
                scope='single global plan; fit selection then tune admission; no final data used',
                universal_robustness_guarantee=False)


def measure_split(out, label, examples, cfg, rules, settings, *, candidates=None, bank='controlled', resume=False):
    rows = []
    for i,e in enumerate(examples):
        path = out/'calibration_records'/label/(e.group_digest+'.json')
        request = digest(dict(source=_source_digest(), operator=e.digest, label=label,
                    config=cfg.to_dict(), rules=rules.digest(), settings=settings,
                    candidates=candidates, bank=bank))
        if path.exists():
            entry = read(path)
            if not resume or entry['request'] != request or digest(entry['row']) != entry['row_digest']:
                raise ValueError('use exact --resume; calibration record changed')
            row = entry['row']
        else:
            row = measure_classical_portfolio([e], cfg, rules, repeats=int(settings['repeats']),
                rhs_count=int(settings['rhs_count']), regime='warm_multiple', bank=bank,
                candidate_names=candidates, seed=int(settings['seed'])+i)[0]
            write_json(path, dict(request=request, row=row, row_digest=digest(row)))
        rows.append(row)
        print('[classical]',label,i+1,'/',len(examples),flush=True)
    return rows


def calibrate(config, output, *, resume=False):
    settings = read(config); out = Path(output).resolve()
    if settings.get('thesis',{}).get('version') != VERSION:
        raise ValueError('wrong H_S thesis configuration')
    cfg = AdaptiveConfig.from_dict(settings['solver'])
    if cfg.use_transfer or cfg.branch != 'H_S' or not cfg.use_smoother:
        raise ValueError('H_S-only config required; learned P stays disabled')
    candidates = settings['thesis']['classical_candidates']
    if cfg.mg.strategy_name not in candidates:
        raise ValueError('candidate set must retain the declared classical anchor')
    torch.set_num_threads(int(settings.get('torch_threads',1)));initialize_timing_runtime()
    header = dict(version=VERSION, source_digest=_source_digest(), settings_digest=digest(settings),
                  hardware=hardware_environment(refresh=True))
    if (out/'thesis_manifest.json').exists():
        if not resume or read(out/'thesis_manifest.json') != json_safe(header):
            raise ValueError('new run or exact --resume required')
    else:
        if resume or out.exists() and any(out.iterdir()):raise FileExistsError('new empty run required')
        out.mkdir(parents=True,exist_ok=True);write_json(out/'configuration.json',settings)
        write_json(out/'thesis_manifest.json',header)
    assert_open(out)
    if (out/'tuned_classical.json').exists():
        return load(out)[3]
    initial = fixed_rules(cfg.mg.strategy_name)
    file = out/'calibration_manifest.json'
    if not file.exists():
        history = historical_operator_index(settings.get('historical_roots') or [str(out.parent)], exclude=[out])
        excluded = set(history['normalized_operator_digests']); splits = {}
        for i,name in enumerate(('selector_train','selector_tune')):
            spec = dict(sizes=settings['calibration_sizes'],families=settings.get('families', list(FAMILIES)),
                per_family=settings['calibration_per_family'],seed=settings['seed']+100003*i)
            _,splits[name],_ = _generate_split(name,spec,excluded,initial)
        write_json(file,dict(splits=splits,rules=initial.to_dict(),historical_index=history))
    data = read(file)
    groups = {n:_restore(v,initial) for n,v in data['splits'].items()}
    protocol = dict(repeats=settings['calibration_repeats'],rhs_count=settings['calibration_rhs'],seed=settings['seed'])
    rows = {n:measure_split(out,n,es,cfg,initial,protocol,candidates=candidates,resume=resume) for n,es in groups.items()}
    result = fit_global(rows['selector_train'],rows['selector_tune'],candidates,cfg.mg.strategy_name)
    rules = fixed_rules(result['plan'])
    write_json(out/'classical_measurements.json', rows)
    result.update(rules_digest=rules.digest(), measurements_sha256=file_hash(out/'classical_measurements.json'))
    write_json(out/'selector_rules.json',rules.to_dict());write_json(out/'tuned_classical.json',result)
    print('C_tuned:',result['plan'],'; one plan for every operator',flush=True)
    return rules


def strong_audit(output, *, resume=False):
    out,settings,cfg,_ = load(output); assert_open(out)
    if (out/'expert_selection.json').exists():
        raise ValueError('declare/freeze the optional strong audit before selecting the expert')
    if (out/'strong_audit_rules.json').exists():
        if resume:return load_strong_rules(out/'strong_audit_rules.json')
        raise FileExistsError('strong audit is already calibrated')
    saved = read(out/'calibration_manifest.json');original=StrongRules.from_dict(saved['rules'])
    cfg=replace(cfg,mg=replace(cfg.mg,strategy_name=settings['solver']['mg']['strategy_name']))
    init=StrongRules(require_coverage=True,fallback_strategy_name=cfg.mg.strategy_name)
    protocol=dict(repeats=settings['calibration_repeats'],rhs_count=settings['calibration_rhs'],seed=settings['seed'])
    groups={n:_restore(v,original) for n,v in saved['splits'].items()}
    rows={n:measure_split(out,'strong_'+n,es,cfg,init,protocol,bank='em_schedule',resume=resume) for n,es in groups.items()}
    rules,evidence=calibrate_multisize(rows['selector_train'],rows['selector_tune'],init,
        required_sizes=settings['calibration_sizes'],fixed_strategy=cfg.mg.strategy_name,
        minimum_leaf_cases=settings['minimum_leaf_cases'],max_cycles=cfg.mg.max_cycles,
        cycle_margin=settings['cycle_margin'])
    write_json(out/'strong_audit_evidence.json',dict(evidence=evidence,rows=rows,
        role='robustness_only; does not set C_tuned or select H_S architecture'))
    write_json(out/'strong_audit_rules.json',rules.to_dict())
    return rules
