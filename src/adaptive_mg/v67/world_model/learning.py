"""Small action-conditioned recurrent dynamics model, not a flame surrogate.

Predicts observed solver-state transitions, costs, contraction and failure for
hierarchy actions. The two-step planner rolls out predicted next observations;
it NEVER reads A[t+1] at decision time. Accuracy is empirical, not a certificate.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .backend import ACTIONS
from .data import digest

FEATURES=('log_N','log_nnz_per_row','log_diag_scale','log_diag_contrast','diag_cv',
          'log_anisotropy','sin2angle','cos2angle','orientation_variation',
          'matrix_change','p_age','has_bank','same_A','compatible','rhs_norm_log',
          'initial_residual_log','log_tolerance','log_max_cycles','last_log_rho',
          'last_log_cycles','last_log_total','last_success','log_mean_density',
          'density_available','mean_temperature_scaled','temperature_available')
TARGETS=('log_setup','log_solve','log_rho','log_cycles')


def feedback(result):
    if result is None:return np.zeros(5,dtype=np.float32)
    rho=(result.residuals[-1]/max(result.residuals[0],1e-100))**(1/max(result.cycles,1))
    return np.array([np.log(max(result.setup_seconds,1e-8)),np.log(max(result.solve_seconds,1e-8)),
                     np.log(np.clip(rho,1e-8,1e3)),np.log1p(result.cycles),float(result.success)],np.float32)


def observation(s,bank,previous,result,selection,cfg,plans):
    diag=s.a.diagonal();scale=float(np.mean(diag));f=selection.features
    change=0. if previous is None else 1.
    if previous is not None and previous.a.shape==s.a.shape:
        change=float(np.linalg.norm((s.a-previous.a).data)/max(np.linalg.norm(previous.a.data),1e-100))
    same=bool(bank and bank.matrix_digest==s.matrix_digest)
    compatible=bool(bank and bank.topology==s.topology_key and bank.plan==selection.strategy_name)
    fb=feedback(result);angle=np.deg2rad(float(f.get('principal_angle_deg',0)))
    ctx=s.context
    density=ctx.get('mean_density');temperature=ctx.get('mean_temperature')
    if density is not None and (not np.isfinite(density) or density<=0):raise ValueError('invalid density summary')
    if temperature is not None and (not np.isfinite(temperature) or temperature<=0):raise ValueError('invalid temperature summary')
    values=[np.log(s.a.shape[0]),np.log(s.a.nnz/s.a.shape[0]),np.log(scale),np.log(diag.max()/diag.min()),
        float(diag.std()/scale),np.log1p(float(f.get('tensor_anisotropy_ratio',1))),
        np.sin(2*angle),np.cos(2*angle),float(f.get('orientation_variation',0)),np.log1p(change),
        np.log1p(bank.p_age if bank else 0),float(bank is not None),float(same),float(compatible),
        np.log(max(np.linalg.norm(s.b),1e-100)),np.log(max(np.linalg.norm(s.b-s.a@s.x0),1e-100)),
        np.log(cfg.mg.tolerance),np.log(cfg.mg.max_cycles),fb[2],fb[3],
        np.log(max(result.total_seconds,1e-8)) if result else 0.,float(result.success) if result else 0.,
        np.log(density) if density is not None else 0.,float(density is not None),
        temperature/1000. if temperature is not None else 0.,float(temperature is not None)]
    values += [float(selection.strategy_name==p) for p in plans]
    out=np.asarray(values,np.float32)
    if not np.isfinite(out).all():raise ValueError('nonfinite observation')
    return out

class WorldNet(nn.Module):
    def __init__(self,obs_dim,hidden=24):
        super().__init__();self.obs_dim=obs_dim;self.hidden=hidden
        self.memory=nn.GRUCell(obs_dim+4+5,hidden)
        self.transition=nn.Sequential(nn.Linear(hidden+4,hidden),nn.SiLU(),nn.Linear(hidden,obs_dim+5))

    def observe(self,obs,previous_action,previous_feedback,h):
        return self.memory(torch.cat([obs,previous_action,previous_feedback],-1),h)

    def predict(self,h):
        # Output each possible action's cost/rho/cycles, failure logit and next obs.
        eye=torch.eye(4,dtype=h.dtype,device=h.device)
        x=torch.cat([h.expand(4,-1),eye],1)
        z=self.transition(x)
        return z[:,:4],z[:,4],z[:,5:]


def training_arrays(episodes):
    obs=np.concatenate([np.asarray(e['obs']) for e in episodes])
    ys=np.concatenate([np.asarray(e['targets'])[np.asarray(e['available'],bool)] for e in episodes])
    return obs,ys


def fit_world(episodes,*,ensemble=3,hidden=24,epochs=60,seed=11,lr=.002):
    if not episodes or min(ensemble,hidden,epochs)<1:raise ValueError('invalid world training settings')
    torch.set_num_threads(1)
    x,y=training_arrays(episodes)
    xm=x.mean(0).astype(np.float32);xs=np.maximum(x.std(0),.1).astype(np.float32)
    ym=y.mean(0).astype(np.float32);ys=np.maximum(y.std(0),.2).astype(np.float32)
    models=[];logs=[]
    for member in range(ensemble):
        torch.manual_seed(seed+member);rng=np.random.default_rng(seed+member)
        net=WorldNet(x.shape[1],hidden);optimizer=torch.optim.Adam(net.parameters(),lr=lr)
        member_log=[]
        # Bootstrap whole trajectories, not individual snapshots/RHS/repeats.
        indices=rng.integers(0,len(episodes),size=len(episodes))
        for epoch in range(epochs):
            total=0.
            for ei in rng.permutation(indices):
                ep=episodes[int(ei)];h=torch.zeros(1,hidden);pa=torch.zeros(1,4);pf=torch.zeros(1,5);loss=0.
                for t,raw in enumerate(ep['obs']):
                    obs=torch.tensor((np.asarray(raw,np.float32)-xm)/xs)[None,:]
                    h=net.observe(obs,pa,pf,h);pred,logit,nxt=net.predict(h)
                    mask=torch.tensor(ep['available'][t],dtype=torch.bool)
                    target=torch.tensor((np.asarray(ep['targets'][t],np.float32)-ym)/ys)
                    success=torch.tensor(ep['success'][t],dtype=torch.float32)
                    loss=loss+F.smooth_l1_loss(pred[mask],target[mask])+F.binary_cross_entropy_with_logits(logit[mask],success[mask])
                    if t+1<len(ep['obs']):
                        next_target=torch.tensor((np.asarray(ep['next_obs'][t],np.float32)-xm)/xs)
                        loss=loss+.2*F.smooth_l1_loss(nxt[mask],next_target[mask])
                    a=int(ep['behavior'][t]);pa=F.one_hot(torch.tensor([a]),4).float()
                    actual=np.asarray(ep['feedback'][t],np.float32).copy();actual[:4]=(actual[:4]-ym)/ys
                    pf=torch.tensor(actual)[None,:]
                loss=loss/len(ep['obs'])
                if not torch.isfinite(loss):raise FloatingPointError('world model nonfinite loss')
                optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),2.);optimizer.step();total+=float(loss.detach())
            member_log.append(total/max(len(indices),1))
        models.append(net.eval());logs.append(member_log)
    return dict(models=models,xmean=xm,xscale=xs,ymean=ym,yscale=ys,hidden=hidden,training_loss=logs)


class Predictor:
    def __init__(self,artifact):
        self.a=artifact;self.reset()

    def reset(self):
        self.h=[torch.zeros(1,self.a['hidden']) for _ in self.a['models']]
        self.pa=torch.zeros(1,4);self.pf=torch.zeros(1,5)

    @torch.no_grad()
    def predict(self,raw,horizon=2):
        x=torch.tensor((np.asarray(raw,np.float32)-self.a['xmean'])/self.a['xscale'])[None,:]
        times=[];risks=[];rolls=[]
        for i,net in enumerate(self.a['models']):
            self.h[i]=net.observe(x,self.pa,self.pf,self.h[i]);p,logit,nextobs=net.predict(self.h[i])
            physical=p.numpy()*self.a['yscale']+self.a['ymean']
            costs=np.exp(np.clip(physical[:,:2],-30,20)).sum(1);success=torch.sigmoid(logit).numpy()
            score=costs.copy()
            if horizon==2:
                for action in range(4):
                    # Learned next observation, not ground-truth future A.
                    nf=np.r_[p[action].numpy(),success[action]][None,:].astype(np.float32)
                    nh=net.observe(nextobs[action:action+1],torch.eye(4)[action:action+1],torch.tensor(nf),self.h[i])
                    ncost,nrisk,_=net.predict(nh)
                    nc=ncost.numpy()*self.a['yscale']+self.a['ymean']
                    ct=np.exp(np.clip(nc[:,:2],-30,20)).sum(1)
                    eligible=torch.sigmoid(nrisk).numpy()>=.9
                    eligible[0]=True
                    if not self.a.get('expert_signature'):eligible[3]=False
                    score[action]+=.8*float(ct[eligible].min())
            times.append(costs);risks.append(success);rolls.append(score)
        return np.asarray(times),np.asarray(risks),np.asarray(rolls)

    def update(self,action,result):
        self.pa=F.one_hot(torch.tensor([int(action)]),4).float()
        f=feedback(result);f[:4]=(f[:4]-self.a['ymean'])/self.a['yscale']
        self.pf=torch.tensor(f)[None,:]


def calibrate(artifact,episodes):
    """Trajectory-wise optimism margins from TUNE; no formal coverage claim."""
    errors=[[] for _ in ACTIONS];counts=[0]*4;failures=[0]*4
    for ep in episodes:
        predictor=Predictor(artifact);per=[[] for _ in ACTIONS]
        for t,raw in enumerate(ep['obs']):
            times,probs,_=predictor.predict(raw,horizon=1)
            means=times.mean(0)
            truth=np.exp(np.asarray(ep['targets'][t])[:,:2]).sum(1)
            for a in range(4):
                if not ep['available'][t][a]:continue
                counts[a]+=1
                if ep['success'][t][0] and not ep['success'][t][a]:failures[a]+=1
                if ep['success'][t][0] and ep['success'][t][a]:
                    # Cost-ratio optimism, calibrated by worst step per trajectory.
                    per[a].append(np.log(means[0]/means[a])-np.log(truth[0]/truth[a]))
            a=int(ep['behavior'][t]);predictor.pa=F.one_hot(torch.tensor([a]),4).float()
            f=np.asarray(ep['feedback'][t],np.float32).copy();f[:4]=(f[:4]-artifact['ymean'])/artifact['yscale']
            predictor.pf=torch.tensor(f)[None,:]
        for a in range(4):
            if per[a]:errors[a].append(max(per[a]))
    return dict(margin=[max(.03,float(np.quantile(v,.95,method='higher'))) if v else 1e6 for v in errors],
                episode_coverage=[len(v) for v in errors],failure_counts=failures,counts=counts,
                meaning='empirical one-step trajectory optimism; not certified multi-step regret or speed')


def choose(predictions,available,calibration,*,minimum_gain=.03,minimum_episodes=2):
    times,probs,rolls=predictions;mean=times.mean(0);selected=[0];details={}
    if not all(np.isfinite(v).all() for v in predictions):return 0,dict(reason='nonfinite_prediction',certified=False)
    for action in range(1,4):
        logratios=np.log(np.maximum(times[:,0],1e-20)/np.maximum(times[:,action],1e-20))
        lower=float(logratios.mean()-2*logratios.std()-calibration['margin'][action])
        ok=bool(available[action] and calibration['episode_coverage'][action]>=minimum_episodes
                and calibration['failure_counts'][action]==0 and probs[:,action].min()>=.90
                and lower>np.log1p(minimum_gain))
        if ok:selected.append(action)
        details[ACTIONS[action]]=dict(eligible=ok,lower_log_gain=lower,predicted_success=float(probs[:,action].mean()))
    # Two-step planning ranks only actions that pass the one-step conservative
    # gate. This deliberately cannot exploit unproven long-horizon savings.
    best=min(selected,key=lambda i:float(rolls[:,i].mean()))
    return best,dict(actions=details,chosen=ACTIONS[best],certified=False)


def save_model(path,artifact):
    data={k:v for k,v in artifact.items() if k!='models'}
    for k in ('xmean','xscale','ymean','yscale'):data[k]=torch.tensor(data[k])
    data['states']=[m.state_dict() for m in artifact['models']]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)

def load_model(path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    for k in ('xmean','xscale','ymean','yscale'):data[k]=data[k].numpy()
    models=[]
    for state in data.pop('states'):
        net=WorldNet(len(data['xmean']),data['hidden']);net.load_state_dict(state);models.append(net.eval())
    data['models']=models;return data
