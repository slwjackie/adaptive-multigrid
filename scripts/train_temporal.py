#!/usr/bin/env python3
"""Collect complete standalone solves, train K, audit and save format-12 state."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import json
from time import perf_counter
import torch
from adaptive_mg import MGConfig,TemporalComponents
from adaptive_mg.data import make_examples,split_manifest
from adaptive_mg.train_switch import collect_table,fit_switch,audit_policy
from adaptive_mg.provenance import write_json,hardware_environment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/v6_6_verification.json')
    p.add_argument('--initial-checkpoint',default='checkpoints/initial_temporal.pt')
    p.add_argument('--output-dir',default='artifacts/temporal_training')
    args=p.parse_args(); cfg=json.loads(Path(args.config).read_text()); out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cfg.get('torch_threads',1)); t=perf_counter()
    model=TemporalComponents.load(args.initial_checkpoint)
    solver=MGConfig(**cfg.get('solver',{}))
    seed=cfg.get('seed',73); per=cfg.get('per_size',6)
    train=make_examples(cfg['train_sizes'],per_size=per,seed=seed)
    tune=make_examples(cfg['tune_sizes'],per_size=cfg.get('tune_per_size',6),seed=seed+10000)
    audit=make_examples(cfg['audit_sizes'],per_size=cfg.get('audit_per_size',6),seed=seed+20000)
    write_json(out/'split_manifest.json',split_manifest(train=train,tune=tune,audit=audit))
    write_json(out/'training_config.json',cfg); write_json(out/'environment.json',hardware_environment())
    model.save(out/'initial_temporal.pt')
    # Existing v6.5 learned smoother is intentionally kept by default. Optional
    # prefix-specific pretraining must be requested in config, and all K labels
    # are recollected AFTER any such model change.
    if cfg.get('smoother_epochs',0)>0:
        from adaptive_mg.learning import pretrain_smoother
        history=pretrain_smoother(model,train,solver,epochs=cfg['smoother_epochs'],
            learning_rate=cfg.get('smoother_learning_rate',1e-4),seed=seed,device=cfg.get('training_device','cpu'))
        write_json(out/'smoother_pretraining.json',history)
    model.save(out/'smoother_for_policy.pt')
    regimes=tuple(cfg.get('regimes',['cold']))
    options=dict(repeats=cfg.get('timing_repeats',3),warmups=cfg.get('warmups',1),
        lambda_instability=cfg.get('lambda_instability',2.),seed=seed,regimes=regimes)
    tr=collect_table(train,model,solver,cfg['k_values'],output=out/'train_cost_table.json',**options)
    va=collect_table(tune,model,solver,cfg['k_values'],output=out/'tune_cost_table.json',**options)
    history=fit_switch(model,tr,va,epochs=cfg.get('switch_epochs',600),hidden=cfg.get('switch_hidden',24),
                       learning_rate=cfg.get('switch_learning_rate',.004),seed=seed)
    write_json(out/'switch_training.json',history)
    model.save(out/'candidate_temporal.pt')
    cert,reports=audit_policy(model,tune,audit,solver,repeats=cfg.get('audit_repeats',3),
                              min_speedup=cfg.get('min_speedup',1.01),seed=seed+30000,output_dir=out,regimes=regimes)
    model.metadata['training_config']=cfg
    model.save(out/'checkpoint_v6_6_final.pt')
    summary=dict(seconds=perf_counter()-t,policy_validated=cert['validated'],loss=history['loss'],
        train_oracle_nonzero=history['train_oracle_nonzero'],train_records=len(tr),
        tune={key:{k:v for k,v in record.items() if k!='rows'} for key,record in reports['tune'].items()},
        audit={key:{k:v for k,v in record.items() if k!='rows'} for key,record in reports['audit'].items()},
        final_checkpoint=str(out/'checkpoint_v6_6_final.pt'))
    write_json(out/'training_summary.json',summary); print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
