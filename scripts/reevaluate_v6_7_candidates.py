#!/usr/bin/env python3
"""Re-evaluate preserved candidates. Never calls any S/P training routine."""
from pathlib import Path
import argparse
import json
import sys
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from adaptive_mg.v67.models import Components
from adaptive_mg.v67.acceptance import evaluate_stage_acceptance
from adaptive_mg.v67.inspection import recover_initial_entry
from adaptive_mg.v67.limited import read_settings, safe_output, diagnostic_cases, digest_file, REVISION
from adaptive_mg.v67.training import accept_operator
from adaptive_mg.provenance import write_json


def main(argv=None):
    p = argparse.ArgumentParser(description='NO TRAINING: stage-relative or original strict re-evaluation, never a final certificate.')
    p.add_argument('--research-dir', default='artifacts/my_v6_7_research')
    p.add_argument('--output-dir', default='artifacts/my_v6_7_limited_fix/stages')
    p.add_argument('--measurement', choices=['archived', 'remeasure'], default='archived',
                   help='archived: apply policy to original raw timings; remeasure: reproduce stage arms')
    p.add_argument('--stages', default='smoother,transfer,joint,hierarchical')
    p.add_argument('--policy', choices=['stage_relative_v2', 'strict_v1'], default='stage_relative_v2')
    p.add_argument('--limit', type=int, help='fresh timing subset only; small runs are smoke evidence')
    p.add_argument('--repeats', type=int, help='fresh measurement default: original validation_repeats')
    p.add_argument('--stage-min-common-success', type=int, default=5)
    p.add_argument('--stage-min-common-fraction', type=float, default=.5)
    p.add_argument('--entry-checkpoint', help='explicit original entry, one stage only; archived cache signatures must match')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    research, settings, cfg = read_settings(args.research_dir)
    out = safe_output(args.output_dir, research)
    stages = args.stages.split(',')
    if any(s not in ('smoother', 'transfer', 'joint', 'hierarchical') for s in stages): p.error('unknown stage')
    if args.entry_checkpoint and len(stages) != 1: p.error('--entry-checkpoint requires one stage')
    if args.measurement == 'archived' and (args.limit or args.repeats): p.error('archived evaluation uses original complete runs')
    settings = dict(settings, stage_acceptance_policy=args.policy,
                    stage_min_common_success=args.stage_min_common_success,
                    stage_min_common_fraction=args.stage_min_common_fraction)
    if args.repeats is not None:
        if args.repeats < 1: p.error('repeats must be positive')
        settings['validation_repeats'] = args.repeats
    dependencies = {research / s / name for s in stages for name in ('candidate.pt', 'validation/raw.json')}
    dependencies.update(research / s / name for s in ('smoother', 'transfer', 'joint')
                        for name in ('status.json', 'validation/raw.json'))
    if 'hierarchical' in stages: dependencies.add(research / 'controller/controller.pt')
    spec = dict(revision=REVISION, no_training=True, measurement=args.measurement,
                policy=args.policy, settings=settings, stages=stages, limit=args.limit,
                inputs={str(path): digest_file(path) if path.exists() else None for path in sorted(dependencies)})
    if args.entry_checkpoint: spec['explicit_entry_hash'] = digest_file(args.entry_checkpoint)
    manifest = out / 'run_manifest.json'
    if manifest.exists():
        if not args.resume: raise FileExistsError('Choose a new output directory or --resume')
        if json.loads(manifest.read_text()) != spec: raise ValueError('Resume config/input/policy signature mismatch')
    else:
        if args.resume: raise FileNotFoundError('No run to resume')
        write_json(manifest, spec)
    initial, recovery = recover_initial_entry(research)
    write_json(out / 'entry_reconstruction.json', recovery)
    summaries = {}
    for stage in stages:
        stageout = out / stage; stageout.mkdir(exist_ok=True)
        reportfile = stageout / 'status.json'
        if args.resume and reportfile.exists():
            prior = json.loads(reportfile.read_text())
            if prior.get('accepted'):
                selected = Path(prior['selected_checkpoint'])
                if not selected.exists(): raise FileNotFoundError('Selected checkpoint missing from completed stage')
                saved = Components.load(selected)
                expected = Components.load(research / stage / 'candidate.pt')
                if (saved.signature() != expected.signature() or not saved.metadata.get('policy_stale')
                        or saved.metadata.get('certificate', {}).get('validated') is not False):
                    raise ValueError('Selected checkpoint differs from recorded selection or safety metadata')
            summaries[stage] = prior; continue
        path = research / stage / 'candidate.pt'
        rawpath = research / stage / 'validation/raw.json'
        if not path.exists():
            report = dict(stage=stage, accepted=False, stage_promoted=False, status='missing_candidate', path=str(path))
        else:
            candidate = Components.load(path)
            if args.entry_checkpoint: incumbent = Components.load(args.entry_checkpoint)
            elif stage == 'hierarchical':
                entry = research / 'controller/controller.pt'
                incumbent = Components.load(entry) if entry.exists() else None
            else: incumbent = initial
            if incumbent is None:
                report = dict(stage=stage, accepted=False, stage_promoted=False, status='missing_entry_checkpoint',
                              reason='No exact original entry or verified deterministic reconstruction; no training started')
            else:
                original_rows = json.loads(rawpath.read_text())
                # Cache keys contain the actual complete model tensor/spec signature.
                signatures = {r['cache_key'].split(':', 2)[1] for row in original_rows for r in row['runs']['previous']}
                if signatures != {incumbent.signature()}:
                    raise ValueError(f'{stage}: entry does not match recorded original comparison')
                candidate_signatures = {r['cache_key'].split(':', 2)[1] for row in original_rows for r in row['runs']['candidate']}
                if candidate_signatures != {candidate.signature()}:
                    raise ValueError(f'{stage}: candidate does not match recorded original comparison')
                stagecfg = replace(cfg, mode='research' if stage == 'hierarchical' else 'hybrid',
                    use_smoother=stage != 'transfer', use_transfer=stage != 'smoother',
                    spatial=stage == 'hierarchical', gate_mode='learned' if stage == 'hierarchical' else 'open')
                if args.measurement == 'archived':
                    report = evaluate_stage_acceptance(original_rows, candidate, incumbent, stagecfg, settings, stage=stage)
                    write_json(stageout / 'raw_source.json', dict(path=str(rawpath), sha256=digest_file(rawpath),
                        measurement='unchanged archived raw runs; no fresh timing', run_count=sum(len(r) for row in original_rows for r in row['runs'].values())))
                else:
                    limit = args.limit or int(settings['stage_validation_cases'])
                    examples, provenance = diagnostic_cases(research, settings, 'tune', limit)
                    write_json(stageout / 'case_provenance.json', provenance)
                    accepted, report = accept_operator(candidate, incumbent, examples, stagecfg,
                                                        settings, stageout / 'validation', stage=stage)
                smoke = args.limit is not None and args.limit < int(settings['stage_validation_cases'])
                report.update(stage=stage, measurement=args.measurement, no_training=True, smoke=smoke,
                    candidate_checkpoint=str(path), independent_certification_evidence=False,
                    stage_promoted=bool(report['accepted']), direct_classical_speedup_observed=None,
                    final_time_improvement_certified=False)
                if report['accepted']:
                    candidate.mark_policy_stale('stage_relative_reselection_requires_matching_policy_refit')
                    candidate.metadata['limited_fix_selection'] = dict(revision=REVISION, stage=stage,
                        source_checkpoint=str(path), acceptance_policy_version=args.policy,
                        measurement=args.measurement, stage_promoted=True,
                        direct_classical_speedup_observed=None, final_time_improvement_certified=False)
                    selected = stageout / 'selected.pt'; candidate.save(selected)
                    report['selected_checkpoint'] = str(selected.resolve())
                write_json(stageout / 'acceptance.json', report)
        write_json(reportfile, report); summaries[stage] = report
        print(stage, 'accepted=', report.get('accepted'), 'failed=', report.get('failed_conditions', report.get('status')), flush=True)
    write_json(out / 'summary.json', summaries)
    write_json(out / 'reason_summary.json', {s: dict(accepted=r.get('accepted'), failed_conditions=r.get('failed_conditions'),
        stage_promoted=r.get('stage_promoted'), measurement=r.get('measurement')) for s, r in summaries.items()})
    print('NO TRAINING. Original certificate unchanged. Output:', out)


if __name__ == '__main__': main()
