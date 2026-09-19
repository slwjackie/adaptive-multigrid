#!/usr/bin/env python3
"""Optional existing detector/controller labels and refit, with frozen S/P."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_mg.v67.limited import read_settings, safe_output, digest_file, REVISION
from adaptive_mg.v67.models import Components
from adaptive_mg.v67.data import dataset
from adaptive_mg.v67.training import train_detector, temporal_table, fit_controller
from adaptive_mg.provenance import write_json, hardware_environment, module_signature


def main(argv=None):
    p = argparse.ArgumentParser(description='OPTIONAL potentially long offline policy label regeneration/refit; NO smoother/transfer retraining.')
    p.add_argument('--research-dir', default='artifacts/my_v6_7_research')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--branch', choices=['H_S', 'H_P', 'H_SP'], default='H_SP')
    p.add_argument('--output-dir', default='artifacts/my_v6_7_limited_fix/policy_refit')
    p.add_argument('--limit', type=int, help='small plumbing check only; cannot calibrate production')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    research, settings, cfg = read_settings(args.research_dir)
    out = safe_output(args.output_dir, research)
    cfg = replace(cfg, branch='auto', use_smoother=args.branch in ('H_S', 'H_SP'),
                  use_transfer=args.branch in ('H_P', 'H_SP'))
    spec = dict(revision=REVISION, checkpoint=str(Path(args.checkpoint).resolve()),
                checkpoint_sha256=digest_file(args.checkpoint), branch=args.branch,
                settings=settings, config=cfg.to_dict(), limit=args.limit,
                environment=hardware_environment(), no_operator_training=True)
    manifest = out / 'run_manifest.json'
    if manifest.exists():
        if not args.resume: raise FileExistsError('Use new output or --resume')
        if json.loads(manifest.read_text()) != spec: raise ValueError('Resume config/input/revision mismatch')
    else:
        if args.resume: raise FileNotFoundError('No incomplete refit manifest')
        write_json(manifest, spec)
    if (out / 'status.json').exists() and json.loads((out / 'status.json').read_text()).get('status') == 'complete':
        print('Refit already complete:', out / 'calibrated.pt'); return
    model = Components.load(args.checkpoint)
    original_generation = model.generation_signature()
    model.metadata['controller_trained'] = False
    if not cfg.use_smoother:
        model.metadata['detector_trained'] = False
        model.metadata.pop('detector_expert_signatures', None)
    # Existing label routines explicitly force H-bursts and retain their original losses.
    examples = dataset(settings['train_sizes'], settings['train_per_size'], int(settings['seed']))
    if args.limit is not None:
        if args.limit < 1: raise ValueError('limit must be positive')
        examples = examples[:args.limit]
    print('Frozen S/P. Regenerating detector/controller labels with original training method.', flush=True)
    detector = out / 'detector'
    if cfg.use_smoother:
        if args.resume and (detector / 'detector.pt').exists(): model = Components.load(detector / 'detector.pt')
        else:
            model.metadata['detector_trained'] = False
            model = train_detector(model, examples, cfg, settings, detector)
    # stale deployment policy must not inhibit offline label exploration (burst mode).
    actionfile = out / 'actions/action_table.json'
    actiondone = out / 'actions_complete.json'
    if args.resume and actiondone.exists():
        if json.loads(actiondone.read_text())['sha256'] != digest_file(actionfile):
            raise ValueError('Saved action table checksum mismatch')
        rows = json.loads(actionfile.read_text())
    else:
        rows = temporal_table(model, examples, cfg, settings, out / 'actions')
        write_json(actionfile, rows)
        write_json(actiondone, dict(complete=True, rows=len(rows), sha256=digest_file(actionfile)))
    model.metadata['controller_trained'] = False
    model = fit_controller(model, rows, settings, out / 'controller')
    if model.generation_signature() != original_generation:
        raise RuntimeError('S/P changed during policy refit')
    calibrated = bool(model.metadata.get('controller_trained') and rows and args.limit is None
                      and not model.metadata.get('policy_stale'))
    model.metadata['policy_stale'] = not calibrated
    model.metadata['policy_stale_reason'] = None if calibrated else 'limited_smoke_or_untrained_policy'
    model.metadata['policy_generation_signature'] = model.generation_signature()
    model.metadata['detector_generation_signature'] = model.generation_signature()
    model.metadata['controller_generation_signature'] = model.generation_signature()
    model.metadata['branch_calibration'] = {args.branch: dict(validated=calibrated,
        generation_signature=model.generation_signature(), config_scope=cfg.certification_scope(),
        controller_signature=module_signature(model.controller), detector_signature=module_signature(model.detector),
        hardware=hardware_environment(), evidence=str((out / 'actions').resolve()),
        scope='existing offline cost labels and policy refit; not independent performance certification')}
    model.metadata['limited_refit_config'] = cfg.to_dict()
    model.metadata['certificate'] = dict(validated=False, reason='refitted_policy_requires_unchanged_independent_final_audit')
    model.save(out / 'calibrated.pt')
    write_json(out / 'status.json', dict(status='complete', calibrated=calibrated,
        operator_weights_unchanged=True, examples=len(examples), controller_rows=len(rows),
        final_time_improvement_certified=False, checkpoint=str((out / 'calibrated.pt').resolve())))
    print('Saved:', out / 'calibrated.pt', '; final certificate remains False')


if __name__ == '__main__': main()
