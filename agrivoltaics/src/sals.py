"""SALS: Stress-Aware Light Sharing for agrivoltaics.

A density-conditioned control network maps the daily crop and soil state, the
array's ground coverage ratio and a noisy same-day weather forecast to the
light-sharing level of the trackers. It is trained end-to-end through the
differentiable simulator over randomly sampled array densities, sites, years and
warming scenarios, to maximise electricity subject to a yield-retention floor
(penalty method). At deployment, the array density of a site is chosen from the
site's historical seasons only (see evaluate.py), exactly as for the baselines.
"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import physics as P
import simulate as S

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCEN = ["baseline", "+1.5C", "+2C", "+3C"]
G_MIN, G_MAX = 0.05, 0.50   # 0.50 keeps >= 2 m between rows for 2-m-wide modules
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
            # rainfed and irrigated sub-fields each carry their own array:
            # both yield and electricity are area-weighted
            y_out[idx] = combine(y, w); e_out[idx] = combine(e, w)
    return y_out, e_out


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def loss_fn(ret, eratio, rho, mu, mu1=5.0):
    """Exact-penalty (hinge) plus quadratic penalty for the retention floor."""
    short = torch.relu(rho - ret)
    return (-eratio + mu1 * short + mu * short ** 2).mean()


def train(D, refs, train_mask, rho=0.9, mu=30.0, steps=300, batch=512, lr=3e-3, seed=0,
          scenarios=SCEN, stress_features=True, log=None, g_range=(0.10, G_MAX)):
    """Train the density-conditioned controller. Each training season gets a
    random ground coverage ratio in g_range."""
    torch.manual_seed(seed); np.random.seed(seed)
    ctrl = Controller()
    params = list(ctrl.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    pools = {s: torch.where(train_mask[s])[0] for s in scenarios}
    mask_stress = torch.ones(18)
    if not stress_features:  # keep only phenology/time/design/crop information
        mask_stress[[2, 3, 4, 5, 9, 10]] = 0.0
    t0 = time.time()
    n_skipped = 0
    for step in range(steps):
        s = scenarios[step % len(scenarios)]
        idx = pools[s][torch.randint(len(pools[s]), (batch,))]
        d = S._sel(D[s], idx)
        g = g_range[0] + (g_range[1] - g_range[0]) * torch.rand(batch)
        dd, irr, w = expand_irrigation(d)
        y, e = S.rollout(dd, torch.cat([g, g]), controller=lambda f: ctrl(f * mask_stress), irrigated=irr,
                         noise_seed=seed * 100000 + step)
        ret = combine(y, w) / refs[s][0][idx].clamp(min=0.05)
        eratio = combine(e, w) / refs[s][1][idx]
        loss = loss_fn(ret, eratio, rho, mu)
        opt.zero_grad(); loss.backward()
        finite = torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in params if q.grad is not None)
        if finite:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        else:
            n_skipped += 1
        sched.step()
        if log is not None and step % 10 == 0:
            msg = dict(step=step, scen=s, loss=float(loss.detach()), ret=float(ret.detach().mean()),
                       comply=float((ret >= rho).float().mean()), eratio=float(eratio.detach().mean()),
                       skipped=n_skipped, t=round(time.time() - t0))
            print(json.dumps(msg), flush=True, file=log)
    ctrl.mask = mask_stress
    return ctrl


def wrap(ctrl):
    return lambda f: ctrl(f * ctrl.mask)
