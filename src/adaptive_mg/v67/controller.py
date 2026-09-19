"""Online decisions have no shadow solves. Counterfactual costs are learned offline."""
from dataclasses import dataclass,field
import math
import numpy as np


def controller_features(context,relative,rho,tc,mode,cached,cycle,threshold_ratio,entries,stagnation):
    dynamic=np.array([math.log(max(relative,1e-30)),math.log(max(rho,1e-12)),
        math.log(max(tc,1e-9)),float(mode=='HYBRID'),float(cached),cycle/100.,
        math.log(max(threshold_ratio,1e-30)),entries/4.,stagnation/4.,1.],np.float32)
    return np.concatenate((context,dynamic)).astype(np.float32)


def break_even(remaining_log,rho_c,tc,rate_ratio,time_ratio,setup_seconds,horizon,margin):
    """Estimate C tail vs H burst + C tail. Counts/clock are predictions, not proof."""
    if not 0<rho_c<1 or not np.isfinite(tc) or tc<=0:
        return dict(use_hybrid=False,reason='no_reliable_classical_rate')
    rc=-math.log(rho_c);rh=rc*float(np.clip(rate_ratio,1e-3,1e3))
    th=tc*float(np.clip(time_ratio,1e-3,1e3))
    nc=math.ceil(max(remaining_log,0)/rc)
    nh=min(horizon,math.ceil(max(remaining_log,0)/rh))
    ntail=math.ceil(max(0,remaining_log-nh*rh)/rc)
    t_class=nc*tc;t_hybrid=setup_seconds+nh*th+ntail*tc
    eta_c=rc/tc;eta_h=rh/th
    use=bool(eta_h>eta_c and t_hybrid<(1-margin)*t_class)
    return dict(use_hybrid=use,reason='break_even_pass' if use else 'break_even_fail',
        predicted_classical_remaining=t_class,predicted_hybrid_remaining=t_hybrid,
        estimated_classical_cycles=nc,estimated_hybrid_cycles=nh,estimated_tail_cycles=ntail,
        setup_remaining=setup_seconds,eta_classical=eta_c,eta_hybrid=eta_h)

@dataclass
class OnlineHistory:
    mode: str='CLASSICAL'
    dwell: int=0
    entries: int=0
    bad_efficiency: int=0
    c_samples: list=field(default_factory=list)
    h_samples: list=field(default_factory=list)
    def update(self,path,rho,seconds):
        self.dwell+=1
        target=self.h_samples if path=='HYBRID' else self.c_samples
        if np.isfinite(rho) and rho>0 and seconds>0:
            target.append((rho,seconds));del target[:-3]
    def reference(self):
        if not self.c_samples: return None
        rho=float(np.exp(np.mean(np.log([r for r,t in self.c_samples]))))
        t=float(np.median([t for r,t in self.c_samples]))
        return rho,t
    def select(self,cfg,net,trained,context,norm,target,norm0,cycle,cached):
        if self.mode=='CLASSICAL_LOCK': return self.mode,dict(reason='locked')
        if norm<=cfg.mg.near_tolerance_factor*target:
            return 'CLASSICAL_LOCK',dict(reason='near_tolerance')
        if cfg.mode=='burst' and cycle>cfg.hybrid_horizon:
            return 'CLASSICAL_LOCK',dict(reason='offline_burst_budget_complete')
        if cfg.mode in {'hybrid','burst'}: return 'HYBRID',dict(reason='forced_hybrid_ablation')
        if cfg.gate_mode=='closed' or not (cfg.use_smoother or cfg.use_transfer):
            return 'CLASSICAL',dict(reason='all_learned_work_disabled')
        ref=self.reference()
        if ref is None: return 'CLASSICAL',dict(reason='first_real_classical_cycle')
        rho_c,tc=ref
        if not trained: return 'CLASSICAL',dict(reason='controller_untrained')
        rho=self.h_samples[-1][0] if self.mode=='HYBRID' and self.h_samples else rho_c
        f=controller_features(context,norm/max(norm0,1e-300),rho,tc,self.mode,cached,cycle,
                               target/max(norm0,1e-300),self.entries,int(rho>=cfg.mg.stagnation_rho))
        y=np.asarray(net(f),float)
        if not np.isfinite(y).all(): return 'CLASSICAL_LOCK',dict(reason='invalid_controller_output')
        rate_ratio=float(np.exp(np.clip(y[0],-7,7)));time_ratio=float(np.exp(np.clip(y[1],-7,7)))
        setup=0. if cached else tc*float(np.expm1(np.clip(y[2],0,12)))
        p=1/(1+math.exp(-float(np.clip(y[3],-40,40))))
        # Once H was actually observed, use observed recent H throughput too.
        if self.mode=='HYBRID' and self.h_samples and 0<rho_c<1:
            hr=float(np.exp(np.mean(np.log([r for r,t in self.h_samples]))))
            ht=float(np.median([t for r,t in self.h_samples]))
            if 0<hr<1:
                rate_ratio=(-math.log(hr))/(-math.log(rho_c));time_ratio=ht/tc
        info=break_even(math.log(max(norm/target,1)),rho_c,tc,rate_ratio,time_ratio,setup,
                        cfg.hybrid_horizon,cfg.win_margin)
        info['predicted_hybrid_probability']=p
        candidate=info['use_hybrid'] and p>=.5 if cfg.break_even else p>=.5
        if self.mode=='CLASSICAL':
            if self.entries>=cfg.max_hybrid_entries: return 'CLASSICAL_LOCK',dict(info,reason='entry_budget_exhausted')
            if self.dwell<cfg.min_dwell: return 'CLASSICAL',dict(info,reason='minimum_dwell')
            return ('HYBRID' if candidate else 'CLASSICAL'),info
        # Exit hysteresis is weaker than the entry time-margin condition.
        good=info.get('eta_hybrid',0)>=cfg.exit_efficiency_ratio*info.get('eta_classical',float('inf'))
        self.bad_efficiency=0 if good else self.bad_efficiency+1
        if self.bad_efficiency>=cfg.underperformance_patience:
            return 'CLASSICAL_LOCK',dict(info,reason='repeated_hybrid_underperformance')
        if self.dwell<cfg.min_dwell: return 'HYBRID',dict(info,reason='minimum_dwell')
        return ('HYBRID' if good and (candidate or self.dwell<cfg.hybrid_horizon) else 'CLASSICAL'),info
