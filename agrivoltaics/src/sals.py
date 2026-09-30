"""SALS: Stress-Aware Light Sharing for agrivoltaics.

A design network maps site descriptors (crop, latitude, climate) to the ground
coverage ratio of the array, and a control network maps the daily crop and soil
state plus the same-day weather forecast to the light-sharing level of the
trackers. Both are trained end-to-end through the differentiable simulator to
maximise electricity subject to a yield-retention floor rho (penalty method),
across sites, years and warming scenarios.
"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import physics as P
import simulate as S

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCEN = ["baseline", "+1.5C", "+2C", "+3C"]
G_MIN, G_MAX = 0.05, 0.60
TRAIN_YEARS = list(range(2001, 2015))
TEST_YEARS = list(range(2015, 2020))


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def load_all():
    return {s: torch.load(f"{ROOT}/data/cache/seasons_{s}.pt") for s in SCEN}


def site_split(seed=0, test_frac=0.3):
    """Spatial split by 10-degree blocks so that test sites are geographically
    separated from training sites."""
    sites = pd.read_csv(f"{ROOT}/data/sites.csv")
    block = (np.floor(sites.lat / 10)).astype(int).astype(str) + "_" + (np.floor(sites.lon / 10)).astype(int).astype(str)
    ub = sorted(block.unique())
    rng = np.random.default_rng(seed)
    test_blocks = set(rng.choice(ub, int(round(test_frac * len(ub))), replace=False))
    return np.array([b in test_blocks for b in block])


def site_features(d):
    """Static descriptors per season row: crop one-hot, latitude, climate."""
    crop = nn.functional.one_hot(d["crop"].long(), 5).float()
    act = d["active"]
    n = act.sum(1).clamp(min=1)
    ghi = d["ghi"].mean(1) / 25.0
    tmax_gs = (d["tmax"] * act).sum(1) / n / 40.0
    rain_gs = (d["rain"] * act).sum(1) / n / 10.0
    vpd = (P.es(d["tmax"]) - P.es(d["tdew"])).clamp(min=0)
    vpd_gs = (vpd * act).sum(1) / n / 3.0
    heat = ((d["tmax"] > d["Theat"][:, None]).float() * act).sum(1) / n
    return torch.cat([crop, torch.stack([d["lat"] / 90.0, ghi, tmax_gs, rain_gs, vpd_gs, heat, d["irr_frac"]], 1)], 1)


def expand_irrigation(d):
    """Duplicate every season into a rainfed and an irrigated row."""
    B = d["lat"].shape[0]
    out = {k: (torch.cat([v, v]) if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B else v) for k, v in d.items()}
    irr = torch.cat([torch.zeros(B), torch.ones(B)])
    w = torch.cat([1 - d["irr_frac"], d["irr_frac"]])
    return out, irr, w


def combine(x, w):
    """Area-weighted combination of the rainfed and irrigated rows."""
    B = x.shape[0] // 2
    return x[:B] * w[:B] + x[B:] * w[B:]


# ----------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------
class Controller(nn.Module):
    def __init__(self, n_in=18, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, f):
        return torch.sigmoid(self.net(f)[:, 0])


class Designer(nn.Module):
    def __init__(self, n_in=12, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, f):
        return G_MIN + (G_MAX - G_MIN) * torch.sigmoid(self.net(f)[:, 0])


# ----------------------------------------------------------------------------
# References
# ----------------------------------------------------------------------------
@torch.no_grad()
def references(d, chunk=4000):
    """Open-field yield (area-weighted rainfed/irrigated) and the electricity of a
    conventional PV plant (GCR 0.4, backtracking, no crop) for every season."""
    B = d["lat"].shape[0]
    y_open, e_pv = torch.zeros(B), torch.zeros(B)
    for i in range(0, B, chunk):
        idx = torch.arange(i, min(i + chunk, B))
        dd, irr, w = expand_irrigation(S._sel(d, idx))
        y, _ = S.rollout(dd, torch.full((len(irr),), 0.4), panels=False, irrigated=irr)
        y_open[idx] = combine(y, w)
        _, e = S.rollout(S._sel(d, idx), torch.full((len(idx),), 0.4))
        e_pv[idx] = e
    return y_open, e_pv


def evaluate(d, g, controller=None, u_rule=None, chunk=3000, kappa=None):
    """Area-weighted yield and electricity for a design g (B,) and controller."""
    B = d["lat"].shape[0]
    y_out, e_out = torch.zeros(B), torch.zeros(B)
    with torch.no_grad():
        for i in range(0, B, chunk):
            idx = torch.arange(i, min(i + chunk, B))
            dd, irr, w = expand_irrigation(S._sel(d, idx))
            gg = torch.cat([g[idx], g[idx]])
            ctrl = controller if controller is not None else u_rule
            y, e = S.rollout(dd, gg, controller=ctrl, irrigated=irr, kappa=S.KAPPA if kappa is None else kappa)
            y_out[idx] = combine(y, w); e_out[idx] = e[: len(idx)]
    return y_out, e_out


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def loss_fn(ret, eratio, rho, mu, mu1=5.0):
    """Exact-penalty (hinge) plus quadratic penalty for the retention floor."""
    short = torch.relu(rho - ret)
    return (-eratio + mu1 * short + mu * short ** 2).mean()


def train(D, refs, train_mask, rho=0.9, mu=30.0, steps=300, batch=384, lr=3e-3, seed=0,
          use_design=True, fixed_g=None, scenarios=SCEN, stress_features=True, log=None,
          checkpoint=None):
    torch.manual_seed(seed); np.random.seed(seed)
    ctrl, des = Controller(), Designer()
    params = list(ctrl.parameters()) + (list(des.parameters()) if use_design else [])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    pools = {s: torch.where(train_mask[s])[0] for s in scenarios}
    feats = {s: site_features(D[s]) for s in scenarios}
    mask_stress = torch.ones(18)
    if not stress_features:  # keep only phenology/time/design/crop information
        mask_stress[[2, 3, 4, 5, 9, 10]] = 0.0
    t0 = time.time()
    n_skipped = 0
    for step in range(steps):
        s = scenarios[step % len(scenarios)]
        idx = pools[s][torch.randint(len(pools[s]), (batch,))]
        d = S._sel(D[s], idx)
        g = des(feats[s][idx]) if use_design else torch.full((batch,), fixed_g if fixed_g else 0.3)
        dd, irr, w = expand_irrigation(d)
        y, e = S.rollout(dd, torch.cat([g, g]), controller=lambda f: ctrl(f * mask_stress), irrigated=irr)
        ret = combine(y, w) / refs[s][0][idx].clamp(min=0.05)
        eratio = e[:batch] / refs[s][1][idx]
        loss = loss_fn(ret, eratio, rho, mu)
        opt.zero_grad(); loss.backward()
        finite = torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in params if q.grad is not None)
        if finite:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        else:
            n_skipped += 1
        sched.step()
        if checkpoint is not None and step + 1 in checkpoint[0]:
            ctrl.mask = mask_stress
            checkpoint[1](step + 1, ctrl, des)
        if log is not None and step % 10 == 0:
            msg = dict(step=step, scen=s, loss=float(loss.detach()), ret=float(ret.detach().mean()),
                       comply=float((ret >= rho).float().mean()), eratio=float(eratio.detach().mean()),
                       g=float(g.detach().mean()), skipped=n_skipped, t=round(time.time() - t0))
            print(json.dumps(msg), flush=True, file=log)
    ctrl.mask = mask_stress
    return ctrl, des


def wrap(ctrl):
    return lambda f: ctrl(f * ctrl.mask)
