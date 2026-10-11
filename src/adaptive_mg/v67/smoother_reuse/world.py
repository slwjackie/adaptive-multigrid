"""Small action-conditioned solver world model, with a separately fitted MLP.

The recurrent model predicts log total backend cost and the next observation
under refresh/reuse. Two-step planning is entirely predicted, never supplied
future test states. Uncertainty and benefit are empirical; residual safeguards
remain in FixedPSolver. This model does not predict a physical flow field.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import VERSION, ACTIONS
from .policy import FEATURES


class Dynamics(nn.Module):
    def __init__(self, hidden=16, recurrent=True):
        super().__init__()
        self.hidden, self.recurrent = hidden, recurrent
        self.observe_net = nn.GRUCell(len(FEATURES)+2, hidden) if recurrent else nn.Sequential(
            nn.Linear(len(FEATURES)+2, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 2*(len(FEATURES)+2))

    def observe(self, obs, previous_action, state=None):
        inputs = torch.cat((obs, previous_action), -1)
        if self.recurrent:
            return self.observe_net(inputs, state)
        return self.observe_net(inputs)

    def predict(self, state):
        output = self.head(state).reshape(-1, 2, len(FEATURES)+2)
        return output[..., 0], output[..., 1], output[..., 2:]


def fit(episodes, *, hidden=16, epochs=30, seed=20261011, recurrent=True):
    if not episodes or any(e["split"] != "train" for e in episodes):
        raise ValueError("world fitting accepts training trajectories only")
    raw = np.concatenate([e["obs"] for e in episodes]).astype(np.float32)
    available = [np.asarray(e["available"], bool) for e in episodes]
    costs = np.concatenate([np.asarray(e["log_cost"], np.float32)[m] for e, m in zip(episodes, available)])
    xmean, xscale = raw.mean(0), np.maximum(raw.std(0), .05)
    ymean, yscale = float(costs.mean()), max(float(costs.std()), .1)
    torch.manual_seed(seed)
    net = Dynamics(hidden, recurrent)
    opt = torch.optim.Adam(net.parameters(), lr=.002)
    losses = []
    for epoch in range(epochs):
        accumulated = 0.
        for ep in episodes:
            state = None
            previous = torch.zeros(1, 2)
            chunk = []
            for t, values in enumerate(ep["obs"]):
                x = torch.tensor((np.asarray(values, np.float32)-xmean)/xscale)[None]
                state = net.observe(x, previous, state)
                pred, success, nxt = net.predict(state)
                mask = torch.tensor(ep["available"][t], dtype=torch.bool)
                target = torch.tensor((np.asarray(ep["log_cost"][t], np.float32)-ymean)/yscale)
                okay = torch.tensor(ep["success"][t], dtype=torch.float32)
                loss = F.smooth_l1_loss(pred[0, mask], target[mask])
                loss = loss + .1*F.binary_cross_entropy_with_logits(success[0, mask], okay[mask])
                if t+1 < len(ep["obs"]):
                    yn = torch.tensor((np.asarray(ep["next_obs"][t], np.float32)-xmean)/xscale)
                    loss = loss + .1*F.smooth_l1_loss(nxt[0, mask], yn[mask])
                chunk.append(loss)
                previous = F.one_hot(torch.tensor([ep["behavior"][t]]), 2).float()
                if len(chunk) == 32 or t+1 == len(ep["obs"]):
                    opt.zero_grad()
                    value = torch.stack(chunk).mean()
                    value.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 2.)
                    opt.step()
                    accumulated += float(value.detach())
                    state = state.detach()
                    chunk = []
            # Hidden state resets at physical trajectory boundaries.
        losses.append(accumulated)
    net.eval()
    if not all(torch.isfinite(v).all() for v in net.state_dict().values()):
        raise ValueError("nonfinite world fit")
    return dict(schema=VERSION, actions=list(ACTIONS), features=list(FEATURES),
                hidden=hidden, recurrent=recurrent, state_dict=net.state_dict(),
                xmean=xmean.tolist(), xscale=xscale.tolist(), ymean=ymean, yscale=yscale,
                feature_min=raw.min(0).tolist(), feature_max=raw.max(0).tolist(),
                epochs=epochs, losses=losses, seed=seed)


class Predictor:
    def __init__(self, artifact, *, minimum_gain=.05):
        if (artifact.get("schema") != VERSION or artifact.get("actions") != list(ACTIONS)
                or artifact.get("features") != list(FEATURES)):
            raise ValueError("incompatible stencil-reuse world model")
        self.artifact = artifact
        self.net = Dynamics(artifact["hidden"], artifact["recurrent"])
        self.net.load_state_dict(artifact["state_dict"])
        self.net.eval()
        self.xmean = np.asarray(artifact["xmean"], np.float32)
        self.xscale = np.asarray(artifact["xscale"], np.float32)
        self.minimum_gain = minimum_gain
        self.reset()

    def reset(self):
        self.state = None
        self.previous = torch.zeros(1, 2)

    def accept(self, action):
        self.previous = F.one_hot(torch.tensor([ACTIONS.index(action)]), 2).float()

    @torch.no_grad()
    def choose(self, raw, reference, *, horizon=2, stateless=False):
        if horizon not in (1, 2):
            raise ValueError("world horizon must be one or two")
        raw = np.asarray(raw, np.float32)
        if raw.shape != (len(FEATURES),) or not np.isfinite(raw).all():
            return reference, dict(reason="invalid_observation")
        x = torch.tensor((raw-self.xmean)/self.xscale)[None]
        self.state = self.net.observe(x, self.previous, None if stateless else self.state)
        cost, risk, nxt = self.net.predict(self.state)
        if not all(torch.isfinite(v).all() for v in (cost, risk, nxt)):
            self.reset()
            return reference, dict(reason="nonfinite_prediction")
        def seconds(v):
            return np.exp(np.clip(v.numpy()*self.artifact["yscale"]+self.artifact["ymean"], -25, 15))
        immediate = seconds(cost[0])
        scores = immediate.copy()
        safe = torch.sigmoid(risk[0]).numpy() >= .9
        safe[0] = True
        if raw[-1] < .5:
            safe[1] = False
        if horizon == 2:
            for action in range(2):
                h = self.net.observe(nxt[:, action], torch.eye(2)[action:action+1], self.state)
                c2, r2, _ = self.net.predict(h)
                if not torch.isfinite(c2).all() or not torch.isfinite(r2).all():
                    return reference, dict(reason="nonfinite_rollout")
                allowed = torch.sigmoid(r2[0]).numpy() >= .9
                allowed[0] = True
                scores[action] += .9*float(seconds(c2[0])[allowed].min())
        ref = ACTIONS.index(reference)
        candidate = min(np.flatnonzero(safe), key=lambda i: scores[i])
        # The current matrix size must stay within the fit support. Other
        # extrapolation is reported and must be evaluated on held-out cases.
        trained_size = self.artifact["feature_min"][0]-.01 <= raw[0] <= self.artifact["feature_max"][0]+.01
        margin = float(self.artifact.get("calibration_margin", .1))
        finite = np.isfinite(scores).all()
        change = finite and trained_size and scores[candidate]*(1+self.minimum_gain+margin) < scores[ref]
        chosen = int(candidate) if change else ref
        return ACTIONS[chosen], dict(reason="predicted_gain" if change else "reference_abstention",
            horizon=horizon, predicted_seconds=immediate.tolist(), rollout_seconds=scores.tolist(),
            empirical_margin=margin, certified=False)


def calibrate(artifact, episodes):
    """Tune-only optimism margin; one maximum pairwise error per trajectory."""
    errors = []
    if not episodes or any(e["split"] != "tune" for e in episodes):
        raise ValueError("independent tune episodes required")
    for ep in episodes:
        predictor = Predictor(artifact)
        per = []
        for t, raw in enumerate(ep["obs"]):
            _, detail = predictor.choose(raw, "refresh", horizon=1)
            if "predicted_seconds" not in detail:
                per.append(10.)
            elif all(ep["available"][t]) and all(ep["success"][t]):
                prediction = np.asarray(detail["predicted_seconds"])
                truth = np.exp(ep["log_cost"][t])
                per.append(abs(float(np.log(prediction[0]/prediction[1])-np.log(truth[0]/truth[1]))))
            predictor.accept(ACTIONS[ep["behavior"][t]])
        if per:
            errors.append(max(per))
    return float(np.expm1(min(max(errors, default=10.), 10.)))
