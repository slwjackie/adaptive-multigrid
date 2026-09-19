"""Cold/warm end-to-end measurement, failure-aware summaries, scoped certificates."""
from dataclasses import replace
from time import perf_counter
from pathlib import Path
import csv
import numpy as np
from ..provenance import write_json,hardware_environment,stable_norm,operator_digest
from .solver import PreparedAdaptiveMG


def measured(example,components,cfg,*,regime='cold',x0=None):
    if regime not in {'cold','warm'}:raise ValueError('bad timing regime')
    if regime=='warm':
        p=PreparedAdaptiveMG(example.a,example.n,components,cfg)
        p.solve(example.b,x0)
        t=perf_counter();r=p.solve(example.b,x0);wall=perf_counter()-t
    else:
        t=perf_counter();p=PreparedAdaptiveMG(example.a,example.n,components,cfg)
        r=p.solve(example.b,x0);wall=perf_counter()-t
    out=r.to_dict();out.update(wall_seconds=wall,time_scope=regime,
        measurement_config=cfg.to_dict(),operator_digest=operator_digest(example.a),
        residual_reference_norm=(r.residual_history[0] if cfg.mg.residual_reference=='initial'
                                 else stable_norm(example.b)),
        relative_solution_error=stable_norm(r.x-example.exact)/max(stable_norm(example.exact),1e-300))
    return out


def paired_suite(examples,components,configs,*,repeats=3,warmups=1,seed=7,regime='cold',output=None):
    rng=np.random.default_rng(seed);rows=[]
    for index,e in enumerate(examples):
        modes=list(configs)
        for _ in range(warmups):
            for mode in modes:measured(e,components,configs[mode],regime=regime)
        runs={mode:[] for mode in modes}
        for rep in range(repeats):
            for mode in rng.permutation(modes):
                result=measured(e,components,configs[mode],regime=regime)
                result.update(mode=mode,repeat=rep)
                runs[mode].append(result)
                if output:
                    # Append every run before moving to the next problem.
                    from ..provenance import json_safe
                    import json
                    path=Path(output);path.mkdir(parents=True,exist_ok=True)
                    with (path/'raw_runs.jsonl').open('a') as f:f.write(json.dumps(json_safe(dict(example=e.manifest(),result=result)))+'\n')
        row={'example':e.manifest(),'runs':runs};rows.append(row)
        if output:write_json(Path(output)/'progress.json',{'completed_examples':index+1,'total':len(examples),'status':'running'})
    if output:
        write_json(Path(output)/'raw_results.json',rows)
        write_json(Path(output)/'progress.json',{'completed_examples':len(examples),'status':'complete'})
    return rows


def summarize(rows,baseline='classical',bootstrap_samples=2000):
    if not rows:return {'status':'empty','certified':False}
    modes=list(rows[0]['runs']);summary={};table=[]
    for mode in modes:
        ratios=[];success=0;regressions=0;used=0;c_cycles=[];n_cycles=[]
        for row in rows:
            cr=row['runs'][baseline];nr=row['runs'][mode]
            cs=all(x['converged'] for x in cr);ns=all(x['converged'] for x in nr)
            tc=float(np.median([x['wall_seconds'] for x in cr]));tn=float(np.median([x['wall_seconds'] for x in nr]))
            success+=int(ns);regressions+=int(cs and not ns)
            used+=int(any(x['stats']['accepted_neural_cycles']>0 for x in nr))
            cc=float(np.median([x['cycles'] for x in cr]));nc=float(np.median([x['cycles'] for x in nr]))
            if cs and ns:ratios.append(tc/tn);c_cycles.append(cc);n_cycles.append(nc)
            table.append(dict(case=row['example']['name'],n=row['example']['case']['n'],mode=mode,
                classical_success=cs,success=ns,classical_seconds=tc,seconds=tn,speedup=tc/tn if cs and ns else None,
                classical_cycles=cc,cycles=nc))
        if ratios:
            logs=np.log(ratios);gm=float(np.exp(logs.mean()))
            rng=np.random.default_rng(93);samples=np.exp(rng.choice(logs,(bootstrap_samples,len(logs)),replace=True).mean(1))
            ci=np.quantile(samples,[.025,.975]).tolist()
        else:gm=None;ci=[None,None]
        summary[mode]=dict(successes=success,total=len(rows),common_successes=len(ratios),
            geometric_mean_speedup=gm,speedup_ci95=ci,new_failures=regressions,neural_used_cases=used,
            mean_classical_cycles_common=float(np.mean(c_cycles)) if c_cycles else None,
            mean_cycles_common=float(np.mean(n_cycles)) if n_cycles else None)
    return {'summary':summary,'table':table,'failure_policy':'speed only on jointly successful cases; failures reported separately',
        'ci_scope':'operator bootstrap; not a universal stability or hardware guarantee'}


def save_summary(output,result):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);write_json(output/'summary.json',result)
    table=result.get('table',[])
    if table:
        with (output/'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# Standalone benchmark','', '| Mode | Success | Common success | Speedup | New failures | NN-used cases |', '|---|---:|---:|---:|---:|---:|']
    for m,s in result.get('summary',{}).items():
        speed='not available' if s['geometric_mean_speedup'] is None else f"{s['geometric_mean_speedup']:.4f}x"
        lines.append(f"| {m} | {s['successes']}/{s['total']} | {s['common_successes']} | {speed} | {s['new_failures']} | {s['neural_used_cases']} |")
    lines+=['','Speedup = classical constructor+solve time / selected-mode constructor+solve time.',
        'Not all-success speedups are conditional; inspect failure counts. M2 Pro/A100: not measured unless environment says otherwise.']
    (output/'summary.md').write_text('\n'.join(lines))


def certify(components,cfg,rows,*,min_cases=20,margin=.05):
    s=summarize(rows)['summary']['adaptive']
    # No per-operator cherry picking; an untouched audit with enough independent
    # operators, no regressions and confidence lower bound must pass together.
    valid=bool(len(rows)>=min_cases and s['new_failures']==0 and s['common_successes']==len(rows)
        and s['neural_used_cases']>0 and s['speedup_ci95'][0] is not None
        and s['speedup_ci95'][0]>1/(1-margin))
    return dict(validated=valid,model_signature=components.signature(),scope=cfg.certification_scope(),
        hardware=hardware_environment(),audit_summary=s,audit_min_cases=min_cases,required_time_reduction=margin,
        universal_convergence_guarantee=False,reason='audit_pass' if valid else 'insufficient_or_nonpassing_independent_audit')
