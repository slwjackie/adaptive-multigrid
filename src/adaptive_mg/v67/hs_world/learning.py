"""Action-conditioned solver dynamics for classical P and learned smoothing.

Prediction/admission margins are empirical, not convergence, speed or regret
certificates. Action semantics and checkpoint version are not interchangeable
with the historical H_P-oriented world model.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import torch
from torch.nn import functional as F
from ..world_model import learning as legacy
from .backend import ACTIONS, levels

VERSION='hs-only-world-dynamics-v1'
WorldNet=legacy.WorldNet
fit_world=legacy.fit_world
feedback=legacy.feedback
save_model=legacy.save_model
FEATURES=legacy.FEATURES+('hs_active','hs_cache_ready')


def observation(s,bank,previous,result,selection,cfg,plans):
    if not selection.features:
        from ..strong import operator_features
        selection=replace(selection,features=operator_features(s.a,s.shape))
    base=legacy.observation(s,bank,previous,result,selection,cfg,plans)
    active=bank is not None and any(l.neural_stencil is not None for l in levels(bank.root))
    ready=bool(bank and bank.matrix_digest==s.matrix_digest and bank.smoother_root is not None)
    return np.r_[base,np.float32(active),np.float32(ready)].astype(np.float32)


def load_model(path):
    artifact=legacy.load_model(path)
    if artifact.get('version')!=VERSION or artifact.get('actions')!=list(ACTIONS):
        raise ValueError('incompatible world checkpoint/action semantics')
    return artifact


class Predictor(legacy.Predictor):
    @torch.no_grad()
    def predict(self,raw,horizon=2):
        if horizon not in (1,2):raise ValueError('only one/two-step planning supported')
        raw=np.asarray(raw,np.float32)
        if raw.shape!=self.a['xmean'].shape or not np.isfinite(raw).all():
            raise ValueError('invalid world observation')
        x=torch.tensor((raw-self.a['xmean'])/self.a['xscale'])[None,:]
        times=[];risks=[];rolls=[]
        for i,net in enumerate(self.a['models']):
            self.h[i]=net.observe(x,self.pa,self.pf,self.h[i]);p,logit,nextobs=net.predict(self.h[i])
            physical=p.numpy()*self.a['yscale']+self.a['ymean']
            costs=np.exp(np.clip(physical[:,:2],-30,20)).sum(1)
            success=torch.sigmoid(logit).numpy();score=costs.copy()
            if horizon==2:
                for action in range(4):
                    nf=np.r_[p[action].numpy(),success[action]][None,:].astype(np.float32)
                    nh=net.observe(nextobs[action:action+1],torch.eye(4)[action:action+1],torch.tensor(nf),self.h[i])
                    nc,nrisk,_=net.predict(nh)
                    numeric=nc.numpy()*self.a['yscale']+self.a['ymean']
                    ct=np.exp(np.clip(numeric[:,:2],-30,20)).sum(1)
                    allowed=torch.sigmoid(nrisk).numpy()>=.9;allowed[0]=True
                    if not self.a['allow_smoother']:allowed[2:]=False
                    # Predicted compatibility informs only imagination. Actual
                    # current compatibility is always checked by the backend.
                    predicted_raw=nextobs[action].numpy()*self.a['xscale']+self.a['xmean']
                    if predicted_raw[13]<.5:allowed[[1,3]]=False
                    score[action]+=.8*float(ct[allowed].min())
            times.append(costs);risks.append(success);rolls.append(score)
        return np.asarray(times),np.asarray(risks),np.asarray(rolls)


def calibrate(artifact,episodes):
    """Pairwise margins: abstention returns a tuned classical reuse heuristic,
    not mandatory rebuilding. One worst optimism error per TUNE trajectory.
    """
    errors=[[[] for _ in ACTIONS] for _ in ACTIONS]
    fails=np.zeros((4,4),int);counts=np.zeros((4,4),int)
    for ep in episodes:
        pred=Predictor(artifact);per=[[[] for _ in ACTIONS] for _ in ACTIONS]
        for t,raw in enumerate(ep['obs']):
            times,_,_=pred.predict(raw,horizon=1);means=times.mean(0)
            truth=np.exp(np.asarray(ep['targets'][t])[:,:2]).sum(1)
            for ref in range(4):
                for a in range(4):
                    if not (ep['available'][t][a] and ep['available'][t][ref]):continue
                    counts[ref,a]+=1
                    if ep['success'][t][ref] and not ep['success'][t][a]:fails[ref,a]+=1
                    if ep['success'][t][ref] and ep['success'][t][a]:
                        per[ref][a].append(float(np.log(means[ref]/means[a])-np.log(truth[ref]/truth[a])))
            pred.pa=F.one_hot(torch.tensor([ep['behavior'][t]]),4).float()
            f=np.asarray(ep['feedback'][t],np.float32).copy();f[:4]=(f[:4]-artifact['ymean'])/artifact['yscale']
            pred.pf=torch.tensor(f)[None,:]
        for ref in range(4):
            for a in range(4):
                if per[ref][a]:errors[ref][a].append(max(per[ref][a]))
    margin=[[max(.02,float(np.quantile(v,.95,method='higher'))) if v else 1e6 for v in row] for row in errors]
    return dict(margin=margin,episode_coverage=[[len(v) for v in row] for row in errors],
                failure_counts=fails.tolist(),counts=counts.tolist(),certified=False)


def choose(predictions,available,calibration,reference,*,minimum_gain=.03,minimum_episodes=2):
    times,probs,rolls=predictions
    if reference not in (0,1) or not available[reference]:raise ValueError('valid classical fallback action required')
    if not all(np.isfinite(v).all() for v in predictions):return reference,dict(reason='nonfinite_prediction',certified=False)
    candidates=[reference];details={}
    for action in range(4):
        if action==reference:continue
        lr=np.log(np.maximum(times[:,reference],1e-20)/np.maximum(times[:,action],1e-20))
        lower=float(lr.mean()-2*lr.std()-calibration['margin'][reference][action])
        ok=bool(available[action] and calibration['episode_coverage'][reference][action]>=minimum_episodes
            and not calibration['failure_counts'][reference][action] and probs[:,action].min()>=.9
            and lower>np.log1p(minimum_gain))
        if ok:candidates.append(action)
        details[ACTIONS[action]]=dict(eligible=ok,lower_log_gain=lower)
    selected=min(candidates,key=lambda i:float(rolls[:,i].mean()))
    return selected,dict(reference=ACTIONS[reference],chosen=ACTIONS[selected],candidates=details,certified=False)
