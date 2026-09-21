"""Strong-C-aware direct, teacher, KD and joint full-V-cycle training.

Only development examples accepted by research_data enter these routines.
Architecture/weight selection belongs to a separate measured validation step.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F

from ..provenance import json_safe, write_json, hardware_environment
from .models import Components, TransferNet
from .limited import forced_config, digest_file
from .strong import PreparedStrongMG
from .banks import Stats, hybrid_cycle, selected_level
from .spatial import SpatialState
from ..hierarchy import classical_cycle
from .unroll import make_graph, cycle
from .research_runtime import offline_teacher_generation, research_spec
from .research_smoothers import make_research_smoother
from .research_transfer import make_graph_transfer, differentiable_transfer_cost
from .research_controls import ControlledTransferCNN


def transfer_feasibility(root, model, cfg, reference_root=None):
    """Exact detached runtime caps, with a separate differentiable repair term.

    Infeasible proposals may be optimized toward feasibility, but their task
    contraction must never be reported as the deployable learned objective.
    The repair surrogate is not an exact derivative of discrete sparse nnz.
    """
    from ..transfer import scipy_prolongation_from_weights
    from ..hierarchy import galerkin_coarse_operator
    from .research_transfer import enforce_transfer_complexity, TransferComplexityError
    zero=root.a.values.new_zeros(())
    if not cfg.use_transfer or not hasattr(model.transfer,'complexity_caps'):
        return dict(feasible=True,violations=[],scope='no constrained learned transfer'),zero
    caps=model.transfer.complexity_caps;violations=[];penalties=[];reports=[]
    nnz=0;level=root
    while level is not None:
        nnz+=level.raw_scipy.count_nonzero()
        if level.coarse is not None and selected_level(level.index,cfg):
            w=level.interpolation_weights
            baseline=torch.as_tensor(level.base_weights,dtype=w.dtype,device=w.device)
            # Delta-to-classical repair is finite even at zero new-edge weights;
            # sparse projection STE supplies gradients to pruned candidates.
            scale=baseline.abs().sum(1).clamp_min(1e-8)
            repair=((w-baseline).abs().sum(1)/scale).mean()
            repair=repair+F.relu(w.abs().sum(1)-8.).square().mean()
            penalties.append(repair)
            p=level.p.numpy();a=level.raw_scipy;ac=level.coarse.raw_scipy
            p0=scipy_prolongation_from_weights(level.pattern,level.base_weights)
            if float(w.detach().abs().sum(1).max())>8.:
                violations.append(dict(level=level.index,reason='row_magnitude_safety_limit'))
            try:
                report=enforce_transfer_complexity(a,p,ac,baseline_p=p0,
                    baseline_ac=galerkin_coarse_operator(a,p0),caps=caps)
            except TransferComplexityError as error:
                report=error.report
                violations.append(dict(level=level.index,reason=str(error)))
            reports.append(dict(level=level.index,**report))
        level=level.coarse
    complexity=nnz/max(root.raw_scipy.count_nonzero(),1)
    from .research_transfer import hierarchy_complexity_report
    if reference_root is None:
        reference_root=make_graph(root.raw_scipy,root.shape,model,cfg,learned=False)
    def counts(level):
        result=[]
        while level is not None:
            result.append(level.raw_scipy.count_nonzero());level=level.coarse
        return result
    try:
        aggregate=hierarchy_complexity_report(counts(root),counts(reference_root),caps)
    except TransferComplexityError as error:
        aggregate=error.report
        violations.append(dict(level='all',reason=str(error)))
    # Weight the continuous repair signal by observed discrete budget excess.
    severity=max([1.,aggregate.get('operator_complexity_ratio',1.)/caps.get('max_operator_complexity_ratio',1.15) if caps.get('complexity_reference')=='parent' else complexity/caps['max_operator_complexity']]+[
        max(r.get('p_ratio',1.)/caps['max_p_ratio'],r.get('ac_ratio',1.)/caps['max_ac_ratio']) for r in reports])
    penalty=torch.stack(penalties).mean()*severity if penalties and violations else zero
    return dict(feasible=not violations,violations=violations,levels=reports,
        operator_complexity=complexity,repair_severity=severity,
        scope='actual sparse numerical proposal; discrete feasibility, continuous repair surrogate'),penalty


def create_research_components(*, smoother='ordinary_control', transfer='existing',
                               smoother_hidden=16, transfer_hidden=16, seed=2026091601,
                               support='standard', complexity_caps=None):
    """Independent initialization; old trained weights are historical controls only."""
    model = Components.create(hidden=16, seed=seed)
    model.smoother = make_research_smoother(smoother, hidden=smoother_hidden)
    model.transfer = (ControlledTransferCNN(hidden=transfer_hidden,support=support,complexity_caps=complexity_caps) if transfer == 'existing' else
                      make_graph_transfer(transfer, width=transfer_hidden, support=support,
                                          complexity_caps=complexity_caps))
    model.metadata.update(research_version='strong-aware-v1', initialization_seed=seed,
        role='teacher' if any(getattr(m,'training_only',False) for m in model.modules()) else 'student',
        selector_changed=False, trained_against='existing frozen C*(A)', final_test_seen=False)
    model.mark_policy_stale('new_experts_need_strong_aware_policy_refit')
    return model.eval()


def parameter_counts(model):
    return {n:sum(p.numel() for p in getattr(model,n).parameters())
            for n in ('smoother','transfer','detector','controller')}


def sample_config(example, cfg, rules, branch):
    evidence = getattr(example, 'strong_selection', None)
    if not evidence or evidence.get('rules_digest') != rules.digest():
        raise ValueError('Training requires the saved frozen C*(A) for every example')
    return replace(forced_config(cfg,branch),
                   mg=replace(cfg.mg,strategy_name=evidence['strategy_name']))


def _levels(root):
    while root.coarse is not None:
        yield root
        root=root.coarse


def distillation_loss(student, teacher, branch):
    """Normalized direction/log-gain and same-candidate interpolation KD.

    Teacher is detached. Matching output MSE is auxiliary to a full MG task loss.
    P values retain the sparse support; there is no dense fine/coarse matrix.
    """
    terms=[]
    for s,t in zip(_levels(student),_levels(teacher)):
        if branch in ('H_S','H_SP') and s.dirs is not None:
            if t.dirs is None:raise ValueError('Teacher smoother missing at a learned level')
            terms.append((s.dirs-t.dirs.detach().to(s.dirs)).square().mean())
            terms.append(.25*F.smooth_l1_loss(s.gains.clamp_min(1e-12).log(),
                                              t.gains.detach().to(s.gains).clamp_min(1e-12).log()))
        if branch in ('H_P','H_SP') and s.interpolation_weights.requires_grad:
            if not np.array_equal(s.pattern.columns,t.pattern.columns):
                raise ValueError('KD requires identical ordered candidate interpolation edges')
            sw=s.interpolation_weights;tw=t.interpolation_weights.detach().to(sw)
            valid=torch.tensor(s.pattern.columns>=0)
            scale=tw.abs().sum(1,keepdim=True).clamp_min(1e-8)
            terms.append(((sw-tw)/scale)[valid].square().mean())
    if not terms:raise ValueError('No matching teacher/student numerical operators for KD')
    return torch.stack(terms).mean()


def full_cycle_objective(example, model, cfg, *, prefix=2, tail=1, teacher=None,
                         lambda_kd=0., lambda_mg=1., lambda_stability=.1,
                         lambda_complexity=.01, lambda_compute=.02, compute_ratio=1.,lambda_feasibility=1.):
    if prefix<1 or tail<0:raise ValueError('At least one full learned V-cycle is required')
    learned=make_graph(example.a,(example.n,example.n),model,cfg,learned=True)
    classical=make_graph(example.a,(example.n,example.n),model,cfg,learned=False)
    feasibility,repair=transfer_feasibility(learned,model,cfg,classical)
    b=torch.tensor(example.b,dtype=torch.float64);x=torch.zeros_like(b)
    norm0=torch.linalg.vector_norm(b).clamp_min(1e-100)
    history=[]
    for k in range(prefix+tail):
        root=learned if k<prefix and feasibility['feasible'] else classical
        x=cycle(root,x,b,model,cfg,k)
        history.append(torch.linalg.vector_norm(b-learned.a.apply(x))/norm0)
    h=torch.stack(history)
    weights=torch.arange(1,len(h)+1,dtype=torch.float64);weights/=weights.sum()
    mg=(weights*h.clamp_min(cfg.mg.tolerance).log()).sum()
    previous=torch.cat((h.new_ones(1),h[:-1]))
    stability=F.relu(h/previous.clamp_min(1e-100)-1).square().mean()
    complexity_terms=[]
    if cfg.use_transfer:
        for level in _levels(learned):
            if level.interpolation_weights.requires_grad:
                proxy=differentiable_transfer_cost(level.interpolation_weights,level.pattern,
                                                  baseline=level.base_weights,a=level.a)
                complexity_terms.append(proxy['loss'])
    complexity=torch.stack(complexity_terms).mean() if complexity_terms else mg.new_zeros(())
    kd=mg.new_zeros(());kd_status='not_requested'
    if teacher is not None and lambda_kd>0:
        with torch.no_grad():
            target=make_graph(example.a,(example.n,example.n),teacher,cfg,learned=True)
            teacher_feasibility,_=transfer_feasibility(target,teacher,cfg,classical)
        if teacher_feasibility['feasible']:
            kd=distillation_loss(learned,target,cfg.branch);kd_status='applied'
        else:kd_status='teacher_proposal_infeasible_on_training_operator'
    # Architecture FLOPs/work calibration is constant for a fixed branch.
    # Sparse P soft occupancy is the differentiable work term; no clock gradient.
    compute=mg.new_tensor((prefix*compute_ratio+tail)/(prefix+tail))
    loss=lambda_mg*mg+lambda_kd*kd+lambda_stability*stability+lambda_complexity*complexity+lambda_compute*compute+lambda_feasibility*repair
    return loss,dict(multicycle=mg,kd=kd,stability=stability,complexity_proxy=complexity,
        compute_proxy=compute,feasibility_repair=repair,proposal_feasibility=feasibility,kd_status=kd_status,
        task_branch='learned_prefix' if feasibility['feasible'] else 'classical_fallback',
        history=h,actions=(['H']*prefix+['C']*tail if feasibility['feasible'] else ['C']*(prefix+tail))),x


def calibrate_work(example, model, cfg, rules):
    """Actual one-cycle throughput calibration on TRAIN only, not a timing gradient."""
    try:
        with offline_teacher_generation():
            prepared=PreparedStrongMG(example.a,example.n,model,cfg,rules)
            stats=Stats();t=perf_counter();root=prepared.ensure_branch(cfg.branch,stats);setup=perf_counter()-t
            x=np.zeros_like(example.b);ct=[];ht=[]
            for _ in range(3):
                t=perf_counter();classical_cycle(prepared.classical,x,example.b,cfg.mg,Stats());ct.append(perf_counter()-t)
                t=perf_counter();hybrid_cycle(root,x,example.b,cfg,Stats(),SpatialState(model,cfg),1,True);ht.append(perf_counter()-t)
        return dict(measured=True,cycle_ratio=float(np.median(ht)/max(np.median(ct),1e-12)),
                    generation_setup_seconds=setup,hardware=hardware_environment(),
                    source='training operator; actual cached numerical cycle times',constant_for_fixed_schedule=True)
    except (ValueError,RuntimeError,FloatingPointError) as error:
        return dict(measured=False,cycle_ratio=1.,reason=str(error),
                    source='neutral constant fallback; not claimed as a measured runtime',constant_for_fixed_schedule=True)


def train_expert(initial, examples, cfg, rules, settings, out, *, branch, teacher=None, resume=False):
    """One independently resumable direct/KD/joint candidate; never promotes itself."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not examples or any(getattr(e,'research_split',None) not in ('train',) for e in examples):
        raise ValueError('Only fresh committed TRAIN examples may optimize weights')
    steps=int(settings['updates']);kd_steps=int(settings.get('kd_updates',steps//2)) if teacher is not None else 0
    if steps<1 or not 0<=kd_steps<steps:raise ValueError('KD must leave at least one task-only fine-tuning update')
    teacher_signature=teacher.generation_signature() if teacher else None
    schedule=np.random.default_rng(int(settings['seed'])).permutation(len(examples)).tolist()
    schedule=(schedule*((steps+len(schedule)-1)//len(schedule)))[:steps]
    spec=dict(version=1,initial_signature=initial.generation_signature(),teacher_signature=teacher_signature,
        rules_digest=rules.digest(),branch=branch,config=cfg.to_dict(),settings=settings,
        sample_ids=[e.name for e in examples],sample_digests=[e.group_digest for e in examples],
        sequence=[examples[i].name for i in schedule],kd_updates=kd_steps,
        no_final_test=True,hardware=hardware_environment())
    path=out/'training_manifest.json'
    if path.exists():
        if not resume or json.loads(path.read_text())!=json_safe(spec):raise ValueError('Training resume protocol/inputs differ')
    else:
        if resume:raise FileNotFoundError('No training manifest to resume')
        if any(out.iterdir()):raise FileExistsError('Training output must be fresh')
        write_json(path,spec);initial.save(out/'initial.pt',extra={'role':'independent_initialization'})
    if (out/'status.json').exists():
        old=json.loads((out/'status.json').read_text())
        if old.get('status')=='complete':
            if digest_file(out/'candidate.pt')!=old['checkpoint_sha256']:raise ValueError('Candidate checksum mismatch')
            return Components.load(out/'candidate.pt'),old
    model=deepcopy(initial)
    names=('smoother',) if branch=='H_S' else ('transfer',) if branch=='H_P' else ('smoother','transfer')
    start=0;records=[];payload=None
    if resume and (out/'resume.pt').exists():
        model=Components.load(out/'resume.pt')
        payload=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)['extra']
        start=payload['next_update'];records=payload['records']
    from .banks import resolve_device
    device=resolve_device(cfg)
    for name in ('smoother','transfer','detector','controller'):
        net=getattr(model,name);net.to(device=device if name in ('smoother','transfer') else 'cpu',dtype=torch.float32)
        for p in net.parameters():p.requires_grad_(name in names)
        net.train(name in names)
    if teacher is not None:
        teacher=deepcopy(teacher).eval()
        for module in teacher.modules():
            for p in module.parameters():p.requires_grad_(False)
    parameters=[p for name in names for p in getattr(model,name).parameters()]
    optimizer=torch.optim.Adam(parameters,lr=float(settings.get('learning_rate',1e-4)))
    if payload:optimizer.load_state_dict(payload['optimizer'])
    calibration_path=out/'compute_calibration.json'
    if calibration_path.exists():calibration=json.loads(calibration_path.read_text())
    else:
        calibration=calibrate_work(examples[0],model,sample_config(examples[0],cfg,rules,branch),rules)
        write_json(calibration_path,calibration)
    start_time=perf_counter()
    for step in range(start,steps):
        e=examples[schedule[step]];scfg=sample_config(e,cfg,rules,branch)
        optimizer.zero_grad(set_to_none=True);kd_phase=teacher is not None and step<kd_steps
        t=perf_counter()
        try:
            loss,details,_=full_cycle_objective(e,model,scfg,
                prefix=int(settings.get('prefix_cycles',2)),tail=int(settings.get('tail_cycles',1)),
                teacher=teacher if kd_phase else None,lambda_kd=float(settings.get('lambda_kd',.25)) if kd_phase else 0.,
                lambda_mg=float(settings.get('lambda_mg',1.)),lambda_stability=float(settings.get('lambda_stability',.1)),
                lambda_complexity=float(settings.get('lambda_complexity',.01)),lambda_compute=float(settings.get('lambda_compute',.02)),
                lambda_feasibility=float(settings.get('lambda_feasibility',1.)),
                compute_ratio=calibration['cycle_ratio'])
            if not loss.requires_grad or not torch.isfinite(loss):raise FloatingPointError('nonfinite or inactive objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,float(settings.get('gradient_clip',2.)))
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
            optimizer.step()
            record=dict(step=step,operator=e.name,operator_digest=e.group_digest,selected_strategy=scfg.mg.strategy_name,
                branch=branch,phase='distill_plus_task' if kd_phase else 'full_cycle_task',skipped=False,
                loss=float(loss.detach()),gradient_norm=float(norm),
                **{key:(value.detach().tolist() if isinstance(value,torch.Tensor) else value) for key,value in details.items()})
        except (ValueError,RuntimeError,FloatingPointError) as error:
            record=dict(step=step,operator=e.name,selected_strategy=scfg.mg.strategy_name,branch=branch,
                        phase='distill_plus_task' if kd_phase else 'full_cycle_task',skipped=True,reason=str(error))
        record['training_seconds']=perf_counter()-t;records.append(record)
        model.save(out/'resume.pt',extra=dict(next_update=step+1,optimizer=optimizer.state_dict(),records=records))
        write_json(out/'training.json',records)
        print(f'[TRAIN {out.name}] {step+1}/{steps} {e.name}: '+('SKIP '+record['reason'] if record['skipped'] else f"loss={record['loss']:.5g}"),flush=True)
    updates=sum(not r['skipped'] for r in records)
    for name in names:model.metadata[name+'_trained']=updates>0
    for module in model.modules():
        module.cpu().eval()
        for p in module.parameters():p.requires_grad_(True)
    model.metadata.update(training_rules_digest=rules.digest(),training_operator_digests=[e.group_digest for e in examples],
        training_branch=branch,training_kind='distilled_then_task_finetuned' if teacher else 'direct_full_cycle',
        optimizer_updates=updates,requested_optimizer_updates=steps,teacher_signature=teacher_signature)
    model.mark_policy_stale('strong_aware_experts_changed_requires_new_labels')
    model.save(out/'candidate.pt',extra={'role':'teacher' if model.metadata.get('role')=='teacher' else 'student_candidate',
                                      'not_selected_or_certified':True})
    report=dict(status='complete',branch=branch,updates=updates,requested_updates=steps,skipped=steps-updates,
        checkpoint=str((out/'candidate.pt').resolve()),checkpoint_sha256=digest_file(out/'candidate.pt'),
        parameters=parameter_counts(model),teacher_training_only=model.metadata.get('role')=='teacher',
        teacher_used_for_training=teacher is not None,
        feasible_task_updates=sum(not r['skipped'] and r.get('task_branch')=='learned_prefix' for r in records),
        feasibility_repair_updates=sum(not r['skipped'] and r.get('task_branch')=='classical_fallback' for r in records),
        generation_signature=model.generation_signature(),rules_digest=rules.digest(),
        training_seconds_this_invocation=perf_counter()-start_time,all_updates_completed=updates==steps,
        final_test_seen=False,performance_certified=False,selection_requires_measured_validation=True)
    write_json(out/'status.json',report)
    return model.eval(),report


def validation_adoption(candidate, reference, *, time_reduction=.03, task_reduction=None, min_cases=5):
    """Conservative paired adoption; a success-count swap cannot pass.

    Input rows are per-case validated success + median seconds + contraction.
    A teacher upper bound can use task reduction, with generation cost reported
    separately. Student deployment adoption always uses total measured time.
    """
    if set(candidate)!=set(reference):raise ValueError('Adoption needs identical validation operators')
    key='contraction' if task_reduction is not None else 'seconds'
    for collection in (candidate,reference):
        for row in collection.values():
            if row['success'] and (not isinstance(row[key],(int,float)) or not np.isfinite(row[key]) or row[key]<=0):
                raise ValueError('Successful adoption metrics must be finite and positive')
    common=[k for k in reference if reference[k]['success'] and candidate[k]['success']]
    losses=[k for k in reference if reference[k]['success'] and not candidate[k]['success']]
    gain=None
    threshold=1/(1-(task_reduction if task_reduction is not None else time_reduction))
    if common:
        ratios=[reference[k][key]/max(candidate[k][key],1e-100) for k in common]
        if all(np.isfinite(r) and r>0 for r in ratios):gain=float(np.exp(np.mean(np.log(ratios))))
    passed=not losses and len(common)>=min_cases and gain is not None and gain>=threshold
    return dict(adopted=passed,new_failure_case_ids=losses,common_case_ids=common,
        geometric_improvement=gain,required_improvement=threshold,metric=key,min_cases=min_cases,
        scope='validation architecture selection; never final certification')
