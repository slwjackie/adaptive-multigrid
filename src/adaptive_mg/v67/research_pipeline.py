"""Ordered strong-aware research phases; final test is a separate single-use command."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import torch

from ..provenance import write_json, json_safe, hardware_environment
from .models import Components
from .config import AdaptiveConfig
from .limited import read_settings, initialize_timing_runtime, digest_file
from .strong import load_strong_rules
from .research_runtime import is_teacher
from .research_training import create_research_components, train_expert, validation_adoption, parameter_counts
from .research_data import (make_research_plan,prepare_development_data,load_development_data,
    freeze_research,claim_final_evaluation,materialize_final_data,complete_final_evaluation)
from .research_evaluation import evaluate_research, evaluate_classical_reference
from .research_policy import ResearchPolicy,refit_research_policy

PHASES=('data','baseline','direct','smoother','p','teacher','kd','joint','policy','evaluate','generation')


def _source_files(project):
    return sorted((project/'src/adaptive_mg').rglob('*.py'))+[project/'scripts/run_v6_7_strong_aware.py']


def _read(path):return json.loads(Path(path).read_text())


def _empty_safe_output(project,output):
    allowed=(project/'artifacts/my_v6_7_strong_aware').resolve();out=Path(output).resolve()
    if allowed not in out.parents:
        raise ValueError('Use a run subdirectory of artifacts/my_v6_7_strong_aware; original artifacts are protected')
    out.mkdir(parents=True,exist_ok=True)
    return out


def _arms(research, cfg):
    arms={'fixed_C':dict(model=None,branch='C',selector=False),
          'strong_C':dict(model=None,branch='C',selector=True)}
    return arms


def _architecture_checkpoint_specs(out, research):
    """Predetermined cold final/OOD controls; no post-final model selection."""
    architectures = {
        'ordinary_CNN_H_S': ('ordinary_direct', 'H_S'),
        'compact_CNN_H_S': ('compact_direct', 'H_S'),
        'D4_CNN_H_S': ('d4_direct', 'H_S'),
        'O2_steerable_H_S': ('o2_direct', 'H_S'),
        'existing_CNN_P_H_P': ('p_cnn_direct', 'H_P'),
        'small_GNN_H_P': ('gnn_direct', 'H_P'),
        'edge_MLP_H_P': ('edge_direct', 'H_P'),
    }
    specs = {name: dict(path=str((out/'checkpoints'/checkpoint/'candidate.pt').resolve()), branch=branch, selector=True)
             for name, (checkpoint, branch) in architectures.items()}
    for name, checkpoint, branch in (('distilled_smoother_H_S', 's_distilled', 'H_S'),
                                     ('distilled_transfer_H_P', 'p_distilled', 'H_P')):
        path = out/'checkpoints'/checkpoint/'candidate.pt'
        if path.is_file():
            specs[name] = dict(path=str(path.resolve()), branch=branch, selector=True)
    for name, stage, branch, selector in (
        ('old_fixed_H_S', 'smoother', 'H_S', False),
        ('old_strong_H_S', 'smoother', 'H_S', True),
        ('old_strong_H_P', 'transfer', 'H_P', True),
        ('old_strong_H_SP', 'joint', 'H_SP', True)):
        specs[name] = dict(path=str((research/stage/'candidate.pt').resolve()), branch=branch, selector=selector)
    if any(not Path(spec['path']).is_file() for spec in specs.values()):
        raise FileNotFoundError('Complete direct architecture checkpoints are required before final freeze')
    return specs


def _case_metrics(result,arm):
    records={}
    for row in result['rows']:
        runs=row['runs'][arm]['cold']['1']
        records[row['example']['name']]=dict(success=all(r['success'] for r in runs),
            seconds=float(np.median([r['wall_seconds'] for r in runs])),
            contraction=float(np.median([r['convergence_factor'] for r in runs])) if all(r.get('convergence_factor') is not None for r in runs) else None)
    return records


def _summary(result,arm):return result['summary']['cold']['1'][arm]


def _accepted_neural_cases(result,arm):
    return {row['example']['name'] for row in result['rows']
            if all(r.get('counters',r.get('stats',{})).get('accepted_neural_cycles',0)>0
                   for r in row['runs'][arm]['cold']['1'])}


def _best(result, names):
    """Choose a research candidate on the SAME validation cohort, never final data.

    This is a candidate for subsequent ablations, not production acceptance.
    The WHICH policy separately requires improvement over strong C.
    """
    metrics={name:_case_metrics(result,name) for name in names}
    anchor_success={k for k,v in metrics[names[0]].items() if v['success']}
    names=tuple(n for n in names if anchor_success.issubset({k for k,v in metrics[n].items() if v['success']}))
    common=[case for case in next(iter(metrics.values())) if all(m[case]['success'] for m in metrics.values())]
    def score(name):
        values=metrics[name]
        successes=sum(v['success'] for v in values.values())
        gm=float(np.exp(np.mean([np.log(values[c]['seconds']) for c in common]))) if common else float('inf')
        return (-successes,gm,name)
    chosen=min(names,key=score)
    return chosen,dict(selected=chosen,common_case_ids=common,preserved_reference_success_ids=sorted(anchor_success),selection='preserve reference successes, then max successes and time on identical common validation operators',production_accepted=False)


class DevelopmentRun:
    def __init__(self,project,settings,out,cfg,rules,research,train,validation,manifest,resume):
        self.project,self.settings,self.out,self.cfg,self.rules=project,settings,out,cfg,rules
        self.research,self.train,self.validation,self.manifest,self.resume=research,train,validation,manifest,resume
        self.models={};self.status={};self.results={}
        self.selected={};self.policy=None
        self.base=_arms(research,cfg)

    def initial(self,s='ordinary_control',p='existing',sh=16,ph=16):
        return create_research_components(smoother=s,transfer=p,smoother_hidden=sh,transfer_hidden=ph,
            seed=int(self.settings['seed']),support=self.settings.get('support','standard'),
            complexity_caps=self.settings['complexity_caps'])

    def job(self,name,branch,initial,teacher=None):
        model,status=train_expert(initial,self.train,self.cfg,self.rules,self.settings['training'],
            self.out/'checkpoints'/name,branch=branch,teacher=teacher,resume=self.resume and (self.out/'checkpoints'/name/'training_manifest.json').exists())
        self.models[name]=model;self.status[name]=status
        return model

    def evaluate(self,name,arm_specs,examples=None,**overrides):
        arguments=dict(repeats=int(self.settings['validation_repeats']),warmups=int(self.settings.get('warmups',1)),
                       rhs_counts=(1,),regimes=('cold',))
        arguments.update(overrides)
        path=self.out/'validation'/name
        result=evaluate_research(self.validation if examples is None else examples,
            dict(self.base,**arm_specs),self.cfg,self.rules,path,
            resume=self.resume and (path/'run_manifest.json').exists(),**arguments)
        self.results[name]=result
        return result

    def phase_direct(self):
        # All students use the same committed training sequence and update budget.
        self.job('ordinary_direct','H_S',self.initial())
        self.job('compact_direct','H_S',self.initial(s='student_cnn'))
        self.job('gnn_direct','H_P',self.initial(p='small_gnn'))
        old=Components.load(self.research/'smoother/candidate.pt')
        oldp=Components.load(self.research/'transfer/candidate.pt')
        oldsp=Components.load(self.research/'joint/candidate.pt')
        result=self.evaluate('direct',{
            'old_fixed_H_S':dict(model=old,branch='H_S',selector=False),
            'old_strong_H_S':dict(model=old,branch='H_S'),
            'old_strong_H_P':dict(model=oldp,branch='H_P'),
            'old_strong_H_SP':dict(model=oldsp,branch='H_SP'),
            **{n:dict(model=self.models[n],branch='H_P' if n=='gnn_direct' else 'H_S')
               for n in ('ordinary_direct','compact_direct','gnn_direct')}})
        return result

    def phase_baseline(self):
        old={stage:Components.load(self.research/stage/'candidate.pt') for stage in ('smoother','transfer','joint')}
        self.evaluate('frozen_baseline_reproduction',{
            'old_fixed_H_S':dict(model=old['smoother'],branch='H_S',selector=False),
            'old_strong_H_S':dict(model=old['smoother'],branch='H_S'),
            'old_strong_H_P':dict(model=old['transfer'],branch='H_P'),
            'old_strong_H_SP':dict(model=old['joint'],branch='H_SP')})
        examples = self.validation
        if self.settings['smoke']:
            examples = [e for e in examples if e.n == min(x.n for x in examples)]
        target = self.out/'validation/offline_classical_reference'
        self.results['offline_classical_reference'] = evaluate_classical_reference(
            examples, self.cfg, self.rules, target, research=self.research,
            bank=self.settings.get('classical_reference_bank', 'controlled'),
            repeats=int(self.settings['validation_repeats']), warmups=int(self.settings.get('warmups', 1)),
            resume=self.resume and (target/'run_manifest.json').exists())

    def phase_smoother(self):
        self.job('d4_direct','H_S',self.initial(s='d4'))
        self.job('o2_direct','H_S',self.initial(s='o2',sh=18))
        names=('ordinary_direct','compact_direct','d4_direct','o2_direct')
        result=self.evaluate('smoother_architectures',{n:dict(model=self.models[n],branch='H_S') for n in names})
        eligible=tuple(n for n in names if self.status[n]['all_updates_completed'])
        if not eligible:raise RuntimeError('No smoother completed the requested training protocol')
        choice,evidence=_best(result,eligible);self.selected['smoother']=choice
        write_json(self.out/'smoother_selection.json',evidence)

    def phase_p(self):
        self.job('p_cnn_direct','H_P',self.initial(p='existing'))
        self.job('edge_direct','H_P',self.initial(p='edge_mlp'))
        names=('p_cnn_direct','gnn_direct','edge_direct')
        result=self.evaluate('p_architectures',{n:dict(model=self.models[n],branch='H_P') for n in names})
        eligible=tuple(n for n in names if self.status[n]['all_updates_completed'])
        if not eligible:raise RuntimeError('No transfer completed the requested training protocol')
        choice,evidence=_best(result,eligible);self.selected['transfer']=choice
        write_json(self.out/'p_selection.json',evidence)

    def phase_teacher(self):
        kind=self.settings.get('smoother_teacher','teacher_cnn')
        self.job('s_teacher','H_S',self.initial(s=kind,sh=48 if kind=='teacher_cnn' else 32))
        self.job('p_teacher','H_P',self.initial(p='gnn_teacher',ph=64))
        result=self.evaluate('teacher_upper_bounds',{
            'compact_direct':dict(model=self.models['compact_direct'],branch='H_S'),
            'gnn_direct':dict(model=self.models['gnn_direct'],branch='H_P'),
            's_teacher':dict(model=self.models['s_teacher'],branch='H_S',training_only=True),
            'p_teacher':dict(model=self.models['p_teacher'],branch='H_P',training_only=True)})
        decisions={}
        for teacher,student in (('s_teacher','compact_direct'),('p_teacher','gnn_direct')):
            decision=validation_adoption(_case_metrics(result,teacher),_case_metrics(result,student),
                task_reduction=float(self.settings['teacher_task_reduction']),min_cases=int(self.settings['adoption_min_cases']))
            decision['teacher_generation_cost_reported_separately']=True
            decision['all_teacher_updates_completed']=self.status[teacher]['all_updates_completed']
            observed=_accepted_neural_cases(result,teacher)
            decision['accepted_neural_case_ids']=sorted(observed)
            decision['teacher_used_on_every_comparable_case']=set(decision['common_case_ids']).issubset(observed)
            decision['adopted'] &= decision['all_teacher_updates_completed'] and decision['teacher_used_on_every_comparable_case']
            decisions[teacher]=decision
        write_json(self.out/'teacher_decisions.json',decisions)

    def phase_kd(self):
        decisions=_read(self.out/'teacher_decisions.json');report={}
        for name,teacher,direct,branch,initial in (
            ('s_distilled','s_teacher','compact_direct','H_S',self.initial(s='student_cnn')),
            ('p_distilled','p_teacher','gnn_direct','H_P',self.initial(p='small_gnn'))):
            if not decisions[teacher]['adopted']:
                report[name]=dict(created=False,adopted=False,reason='teacher_did_not_materially_improve_direct_student')
                continue
            # Start independently from the same initialization, with U total
            # updates: KD+task then task-only. Direct also received exactly U.
            self.job(name,branch,initial,teacher=self.models[teacher])
            result=self.evaluate(name,{direct:dict(model=self.models[direct],branch=branch),name:dict(model=self.models[name],branch=branch)})
            decision=validation_adoption(_case_metrics(result,name),_case_metrics(result,direct),
                time_reduction=float(self.settings['adoption_time_reduction']),min_cases=int(self.settings['adoption_min_cases']))
            decision.update(created=True,same_size_direct_control=direct,same_requested_updates=True,
                            all_student_updates_completed=self.status[name]['all_updates_completed'])
            decision['adopted'] &= decision['all_student_updates_completed']
            decision['actual_student_neural_work']=bool(_accepted_neural_cases(result,name))
            decision['adopted'] &= decision['actual_student_neural_work']
            report[name]=decision
        write_json(self.out/'distillation_decisions.json',report)

    def phase_joint(self):
        s=self.selected['smoother'];p=self.selected['transfer']
        # Compare accepted KD against the existing architecture winner before
        # replacing it. A teacher never becomes a deployment component.
        kd=_read(self.out/'distillation_decisions.json')
        for part,name,old,branch in (('smoother','s_distilled',s,'H_S'),('transfer','p_distilled',p,'H_P')):
            if kd.get(name,{}).get('adopted'):
                result=self.evaluate(name+'_vs_architecture',{name:dict(model=self.models[name],branch=branch),old:dict(model=self.models[old],branch=branch)})
                decision=validation_adoption(_case_metrics(result,name),_case_metrics(result,old),
                    time_reduction=float(self.settings['adoption_time_reduction']),min_cases=int(self.settings['adoption_min_cases']))
                if decision['adopted']:self.selected[part]=name
        s=self.selected['smoother'];p=self.selected['transfer']
        combined=deepcopy(self.models[s]);combined.transfer=deepcopy(self.models[p].transfer)
        combined.metadata.update(transfer_trained=self.models[p].metadata.get('transfer_trained',False),
            component_training_provenance=dict(smoother=s,transfer=p),training_branch='independent_student_pair')
        combined.metadata['role']='student';combined.mark_policy_stale('new_student_pair_requires_refit')
        self.models['independent_pair']=combined
        self.job('joint_student','H_SP',combined)
        result=self.evaluate('joint_rollback',{
            'independent_H_S':dict(model=combined,branch='H_S'),
            'joint_H_S':dict(model=self.models['joint_student'],branch='H_S'),
            'independent_H_SP':dict(model=combined,branch='H_SP'),
            'joint_H_SP':dict(model=self.models['joint_student'],branch='H_SP')})
        s_check=validation_adoption(_case_metrics(result,'joint_H_S'),_case_metrics(result,'independent_H_S'),time_reduction=0.,min_cases=int(self.settings['adoption_min_cases']))
        sp_check=validation_adoption(_case_metrics(result,'joint_H_SP'),_case_metrics(result,'independent_H_SP'),time_reduction=float(self.settings['adoption_time_reduction']),min_cases=int(self.settings['adoption_min_cases']))
        use_joint=s_check['adopted'] and sp_check['adopted'] and self.status['joint_student']['all_updates_completed']
        self.branch_models={'H_S':deepcopy(combined),'H_P':deepcopy(combined),
                            'H_SP':deepcopy(self.models['joint_student'] if use_joint else combined)}
        selected=self.out/'selected_students';selected.mkdir(exist_ok=True)
        for branch,model in self.branch_models.items():
            assert not any(is_teacher(m) for m in model.modules())
            path=selected/(branch+'.pt')
            if path.exists():
                if Components.load(path).signature()!=model.signature():raise ValueError('Selected checkpoint would change on resume')
            else:model.save(path,extra={'role':'validation_selected_student','branch':branch})
        write_json(self.out/'joint_decision.json',dict(adopted=use_joint,smoother_preservation=s_check,joint_improvement=sp_check,forced_P_deployment=False))
        write_json(self.out/'selected_students.json',dict(components=self.selected,branches={b:str((selected/(b+'.pt')).resolve()) for b in self.branch_models},performance_certified=False))

    def phase_policy(self):
        target=self.out/'policy'
        if self.resume and (target/'deployable_policy/policy.json').exists():
            self.policy=ResearchPolicy.load(target/'deployable_policy')
        else:
            self.policy=refit_research_policy(self.branch_models,self.train,self.validation,
                self.cfg,self.rules,self.settings['policy'],target,dataset_manifest=self.manifest)

    def phase_evaluate(self):
        arms={b:dict(model=model,branch=b) for b,model in self.branch_models.items()}
        arms['adaptive']=dict(policy=self.policy,branch='auto')
        self.evaluate('branches',arms)
        # Smoke measures all requested RHS counts on a declared small subset;
        # full research measures them on every validation operator.
        examples=self.validation
        if self.settings['smoke']:
            examples=[next(e for e in self.validation if e.n==n) for n in sorted({e.n for e in self.validation})]
        batch_arms = dict(arms, adaptive_rhs_blind=dict(policy=self.policy, branch='auto',
                         expected_rhs_mode='blind', reference_only=True))
        self.evaluate('cold_warm_multiple_rhs',batch_arms,examples,
            regimes=('cold','warm','multiple'),rhs_counts=(1,4,16,64),
            repeats=int(self.settings.get('multi_repeats',self.settings['validation_repeats'])))

    def phase_generation(self):
        from .research_generation import benchmark_generation, rotation_resampling_diagnostic
        from ..transfer import matrix_feature_array
        target=self.out/'generation'
        if self.resume and (target/'generation.json').exists():
            if _read(target/'generation.json').get('status')=='complete' and (self.out/'validation_rotation_resampling.json').exists():return
            # Generation has no training/selection effect. Preserve incomplete
            # evidence and measure a fresh explicitly labelled diagnostic retry.
            index=1
            while (self.out/f'generation_retry_{index}').exists():index+=1
            target=self.out/f'generation_retry_{index}'
        models={name:self.models[name] for name in ('ordinary_direct','compact_direct','d4_direct','o2_direct','gnn_direct','edge_direct','s_teacher','p_teacher')}
        benchmark_generation(models,self.validation[:1] if self.settings['smoke'] else self.validation,
            self.cfg,self.rules,target,repeats=int(self.settings.get('generation_repeats',3)),
            branches=('H_S','H_P'),warmups=1,split='validation')
        features=torch.tensor(matrix_feature_array(self.validation[0].a,(self.validation[0].n,)*2)).unsqueeze(0).float()
        rotations={name:rotation_resampling_diagnostic(self.models[name].smoother,features)
                   for name in ('ordinary_direct','d4_direct','o2_direct')}
        write_json(self.out/'validation_rotation_resampling.json',rotations)

    def save_progress(self,completed):
        write_json(self.out/'progress.json',dict(status='development_complete' if len(completed)==len(PHASES) else 'phase_complete',
            completed=completed,final_test_consumed=False,models=self.status,selection=self.selected,
            teacher_or_KD_not_automatically_deployed=True))

    def tables(self):
        output=self.out/'tables';output.mkdir(exist_ok=True)
        for name,result in self.results.items():
            write_json(output/(name+'.json'),dict(summary=result['summary'],table=result.get('table',[]),
                amortization_crossover=result.get('amortization_crossover'),
                offline_oracle=result.get('offline_oracle'),reference_only=result.get('reference_only',False)))
            rows=result.get('table',[])
            if rows:
                with (output/(name+'.csv')).open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
                    writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
        write_json(output/'ood_status.json',dict(status='not_evaluated',reason='Final/OOD remain sealed until explicit freeze and final commands'))
        write_json(self.out/'development_report.json',dict(
            smoke=self.settings['smoke'],training_operators=len(self.train),validation_operators=len(self.validation),
            rules_digest=self.rules.digest(),selector_changed=False,models=self.status,selected=self.selected,
            architecture_parameters={k:parameter_counts(m) for k,m in self.models.items()},
            validation_tables={name:str((output/(name+'.json')).resolve()) for name in self.results},
            final_test_consumed=False,final_performance_certified=False,
            teacher_decisions=_read(self.out/'teacher_decisions.json') if (self.out/'teacher_decisions.json').exists() else {},
            kd_decisions=_read(self.out/'distillation_decisions.json') if (self.out/'distillation_decisions.json').exists() else {},
            note='Development results only. Production adopts no unsupported neural branch; cold/warm/multiple RHS are measured separately.'))


def _frozen_final(project,out,settings,cfg,rules,phase,resume):
    policy=ResearchPolicy.load(out/'policy/deployable_policy')
    selected=_read(out/'selected_students.json')['branches']
    checkpoints={b:str(out/'selected_students'/(b+'.pt')) for b in selected}
    original = project/settings['base_research']
    architecture_specs = _architecture_checkpoint_specs(out, original)
    protocol=dict(repeats=int(settings['final_repeats']),warmups=int(settings.get('warmups',1)),
                  regimes=['cold','warm','multiple'],rhs_counts=[1,4,16,64])
    acceptance_protocol=dict(protocol,min_cases=int(settings['final_min_cases']),
        minimum_speedup=max(1.03,1/(1-cfg.win_margin)),require_solution_error_not_worse=True)
    architecture_protocol = dict(repeats=int(settings['final_repeats']), warmups=int(settings.get('warmups', 1)),
                                 regimes=['cold'], rhs_counts=[1])
    reference_protocol = dict(bank=settings.get('classical_reference_bank', 'controlled'),
                              repeats=int(settings['final_repeats']), warmups=int(settings.get('warmups', 1)), order_seed=1007)
    freeze_config=dict(solver=cfg.to_dict(),rules_digest=rules.digest(),policy_digest=policy.digest(),
                       final_evaluation=acceptance_protocol,minimum_cases=int(settings['final_min_cases']),
                       time_reduction=cfg.win_margin,architecture_choices=_read(out/'selected_students.json'),
                       architecture_evaluation=architecture_protocol,architecture_checkpoints=architecture_specs,
                       classical_reference=reference_protocol,expected_rhs_ablation=['actual','blind'])
    if phase=='freeze':
        controllers={p.name:str(p) for p in (out/'policy/deployable_policy').iterdir() if p.is_file()}
        freeze_checkpoints=dict(checkpoints, **{'architecture:'+name: spec['path'] for name, spec in architecture_specs.items()})
        frozen=freeze_research(out,freeze_checkpoints,out/'frozen_rules.json',controllers,_source_files(project),freeze_config)
        print('Frozen; final/OOD remain untouched:',frozen,flush=True)
        return dict(status='frozen',path=str(frozen),final_test_consumed=False)
    if settings['smoke']:raise ValueError('Smoke cannot consume final test')
    if _read(out/'research_freeze.json')['config'] != json_safe(freeze_config):
        raise ValueError('Final/architecture/reference protocol or checkpoint inventory differs from freeze')
    from .research_certification import certify_research_final
    invocation=dict(policy_digest=policy.digest(),rules_digest=rules.digest(),protocol=protocol,
                    architecture_evaluation=architecture_protocol,classical_reference=reference_protocol,
                    expected_rhs_ablation=['actual','blind'])
    results={}
    with claim_final_evaluation(out,out/'research_freeze.json',invocation,resume=resume) as claim:
        splits,final_manifest=materialize_final_data(out,claim,rules)
        arms=dict(fixed_C=dict(model=None,branch='C',selector=False),strong_C=dict(model=None,branch='C'))
        arms.update({b:dict(model=Components.load(path),branch=b) for b,path in checkpoints.items()})
        arms['adaptive']=dict(policy=policy,branch='auto')
        arms['adaptive_rhs_blind']=dict(policy=policy,branch='auto',expected_rhs_mode='blind',reference_only=True)
        # Frozen historical controls are named clearly; no teacher/KD selection.
        arms['old_fixed_aware_H_S']=dict(model=Components.load(original/'smoother/candidate.pt'),branch='H_S')
        architecture_arms = dict(fixed_C=dict(model=None,branch='C',selector=False),strong_C=dict(model=None,branch='C'))
        architecture_arms.update({name: dict(model=Components.load(spec['path']), branch=spec['branch'], selector=spec['selector'])
                                  for name, spec in architecture_specs.items()})
        paths=[]
        for split,examples in splits.items():
            target=out/'final_evaluation'/split
            result=evaluate_research(examples,arms,cfg,rules,target,**protocol,
                                     resume=resume and (target/'run_manifest.json').exists())
            results[split]=result;paths.extend([target/'comparison.json',target/'raw_results.json',target/'run_manifest.json'])
            architecture_target = out/'final_architecture_comparison'/split
            architecture_result = evaluate_research(examples,architecture_arms,cfg,rules,architecture_target,
                **architecture_protocol,resume=resume and (architecture_target/'run_manifest.json').exists())
            paths.extend([architecture_target/'comparison.json',architecture_target/'raw_results.json',architecture_target/'run_manifest.json'])
        reference_target = out/'final_classical_reference'
        reference = evaluate_classical_reference(splits['final'],cfg,rules,reference_target,research=original,
            **reference_protocol,resume=resume and (reference_target/'run_manifest.json').exists())
        paths.extend([reference_target/name for name in ('comparison.json','raw_results.json','run_manifest.json','offline_reference.json')])
        certificate=certify_research_final(out,claim,results['final'],policy,cfg,rules,
            adaptive_arm='adaptive',min_cases=int(settings['final_min_cases']),require_solution_error_not_worse=True)
        write_json(out/'strong_aware_certificate.json',certificate);paths.append(out/'strong_aware_certificate.json')
        paths.append(out/'final_assessment.json')
        if certificate['validated']:
            certified=policy.attach_final_certificate(certificate)
            if (out/'certified_policy/policy.json').exists():
                existing=ResearchPolicy.load(out/'certified_policy')
                if existing.digest()!=certified.digest() or existing.final_certificate!=certificate:
                    raise ValueError('Certified export differs from the frozen final decision')
            else:certified.save(out/'certified_policy')
        complete_final_evaluation(out,claim,paths)
    print('Final consumed once. Strong-aware certificate:',certificate['validated'],flush=True)
    return certificate


def pipeline(config_path,output,*,phase=None,resume=False):
    project=Path(__file__).resolve().parents[3]
    settings=_read(config_path)
    if settings.get('version')!=1:raise ValueError('Unsupported strong-aware configuration')
    phase=phase or settings.get('phase','develop')
    if phase not in PHASES+('develop','freeze','final'):raise ValueError('Unknown research phase')
    out=_empty_safe_output(project,output)
    research,original,cfg=read_settings(project/settings['base_research'])
    cfg=replace(cfg,mode='research',inference_device=settings.get('inference_device',cfg.inference_device))
    if cfg.inference_dtype!='float32':raise ValueError('Research generators require the unchanged FP32 contract')
    torch.set_num_threads(int(original.get('torch_threads',1)))
    initialize_timing_runtime()
    rules_path=project/settings['rules'];rules=load_strong_rules(rules_path)
    manifest_path=out/'run_manifest.json'
    run_manifest=dict(version=1,settings=settings,solver=cfg.to_dict(),rules_digest=rules.digest(),
        selector_source_sha256=digest_file(project/'src/adaptive_mg/v67/strong.py'),
        sources={str(p.relative_to(project)):digest_file(p) for p in _source_files(project)},
        historical_checkpoints={stage:digest_file(research/stage/'candidate.pt') for stage in ('smoother','transfer','joint')},
        hardware=hardware_environment(),config_path=str(Path(config_path).resolve()))
    if manifest_path.exists():
        if not resume and phase not in ('freeze','final'):raise FileExistsError('Use --resume or a new research run directory')
        if _read(manifest_path)!=json_safe(run_manifest):raise ValueError('Research settings/source/rules/hardware changed; start a new run')
    else:
        if resume or phase in ('freeze','final'):raise FileNotFoundError('No original development run to continue')
        if any(out.iterdir()):raise FileExistsError('Research run output must be empty')
        write_json(manifest_path,run_manifest);write_json(out/'configuration.json',settings)
        write_json(out/'frozen_rules.json',rules.to_dict())
        write_json(out/'current_baseline_registry.json',dict(
            fixed_strategy=cfg.mg.strategy_name,strong_rules_digest=rules.digest(),
            checkpoints={stage:dict(path=str(research/stage/'candidate.pt'),sha256=digest_file(research/stage/'candidate.pt')) for stage in ('smoother','transfer','joint')},
            original_checkpoints_copied=False))
    if phase in ('freeze','final'):return _frozen_final(project,out,settings,cfg,rules,phase,resume)
    if (out/'final_claim.json').exists():raise ValueError('Final test has been consumed/claimed; development cannot resume')
    if (out/'development_manifest.json').exists():train,validation,manifest=load_development_data(out,rules)
    else:
        plan=make_research_plan(smoke=bool(settings['smoke']),seed=int(settings['seed']))
        train,validation,manifest=prepare_development_data(out,rules,plan=plan,historical_roots=[research.parent])
    run=DevelopmentRun(project,settings,out,cfg,rules,research,train,validation,manifest,resume)
    completed=['data'];run.save_progress(completed)
    target=len(PHASES)-1 if phase=='develop' else PHASES.index(phase)
    for current in PHASES[1:target+1]:
        print('[PHASE]',current,'No final-test access.',flush=True)
        getattr(run,'phase_'+current)();completed.append(current);run.save_progress(completed);run.tables()
    run.tables()
    return _read(out/'development_report.json')


def main(argv=None):
    parser=argparse.ArgumentParser(description='Strong-aware student research. Full training/final test run ONLY by explicit phase commands.')
    parser.add_argument('--config',default='configs/v6_7_strong_aware_smoke.json')
    parser.add_argument('--output-dir',default='artifacts/my_v6_7_strong_aware/smoke')
    parser.add_argument('--phase',choices=PHASES+('develop','freeze','final'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    result=pipeline(args.config,args.output_dir,phase=args.phase,resume=args.resume)
    print('Results:',Path(args.output_dir).resolve())
    print('Final certified:',result.get('validated',result.get('final_performance_certified',False)))


if __name__=='__main__':main()
