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
        # bounded logit keeps a non-vanishing gradient at the extremes (u in [0.007, 0.993])
        return torch.sigmoid(5.0 * torch.tanh(self.net(f)[:, 0] / 5.0))


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
          scenarios=SCEN, stress_features=True, log=None, g_site=None, jitter=(-0.05, 0.10),
          warm_start=True, bc_batches=2, bc_steps=1500):
    """Train the density-conditioned controller. Each training season is simulated
    with a GCR close to the density that is sustainable at its site: g_site[s] is a
    (B,) tensor per scenario (from calibrate_training_gcr), perturbed by a uniform
    jitter so that the network learns to control a range of densities."""
    torch.manual_seed(seed); np.random.seed(seed)
    ctrl = Controller()
    params = list(ctrl.parameters())
    pools = {s: torch.where(train_mask[s])[0] for s in scenarios}
    mask_stress = torch.ones(18)
    if not stress_features:  # keep only phenology/time/design/crop information
        mask_stress[[2, 3, 4, 5, 9, 10]] = 0.0
    if warm_start:
        behaviour_clone(ctrl, D, pools, g_site, jitter, mask_stress, scenarios, batch, bc_batches, bc_steps, seed, log)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    n_skipped = 0
    for step in range(steps):
        s = scenarios[step % len(scenarios)]
        idx = pools[s][torch.randint(len(pools[s]), (batch,))]
        d = S._sel(D[s], idx)
        g = (g_site[s][idx] + jitter[0] + (jitter[1] - jitter[0]) * torch.rand(batch)).clamp(G_MIN, G_MAX)
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


def behaviour_clone(ctrl, D, pools, g_site, jitter, mask_stress, scenarios, batch, n_batches, steps, seed, log):
    """Warm start: imitate the phenology rule (soft targets 0.95 / 0.05) on the
    states visited by that rule, before fine-tuning through the simulator."""
    feats, targets = [], []
    gen = torch.Generator().manual_seed(seed)

    def recorder(f):
        u = u_phenology(f)
        keep = f[:, 8] > 0
        feats.append((f * mask_stress)[keep]); targets.append(u[keep])
        return u
    with torch.no_grad():
        for s in scenarios:
            for b in range(n_batches):
                idx = pools[s][torch.randint(len(pools[s]), (batch,), generator=gen)]
                d = S._sel(D[s], idx)
                g = (g_site[s][idx] + jitter[0] + (jitter[1] - jitter[0]) * torch.rand(batch, generator=gen)).clamp(G_MIN, G_MAX)
                dd, irr, w = expand_irrigation(d)
                S.rollout(dd, torch.cat([g, g]), controller=recorder, irrigated=irr, noise_seed=seed * 7 + b)
    X, Y = torch.cat(feats), torch.cat(targets) * 0.9 + 0.05
    opt = torch.optim.Adam(ctrl.parameters(), lr=3e-3)
    for it in range(steps):
        j = torch.randint(len(X), (8192,), generator=gen)
        loss = nn.functional.binary_cross_entropy(ctrl(X[j]), Y[j])
        opt.zero_grad(); loss.backward(); opt.step()
    if log is not None:
        print(json.dumps(dict(warm_start="phenology rule", samples=len(X), bce=float(loss.detach()))), file=log, flush=True)


def wrap(ctrl):
    return lambda f: ctrl(f * ctrl.mask)


def u_season(f):
    """Seasonal sharing: light-sharing rotation during the whole cropping window."""
    return f[:, 8]


def u_phenology(f):
    """Phenology rule: share light only while the crop is growing (from canopy
    onset to physiological maturity)."""
    tt, active = f[:, 0], f[:, 8]
    return active * ((tt > 0.05) & (tt < 1.0)).float()


def u_stress(f):
    """Expert stress rule: share light in the main growth phase, except on days
    with a heat-stress forecast or drought, when the trackers shade the crop."""
    tt, arid, heat, active = f[:, 0], f[:, 3], f[:, 4], f[:, 8]
    return active * ((tt > 0.1) & (tt < 0.9) & (heat < 0) & (arid < 0.5)).float()


@torch.no_grad()
def calibrate_training_gcr(D, refs, train_mask, rho=0.9, grid=None):
    """Per training site and climate: largest GCR at which seasonal sharing keeps
    the mean retention of the site's training seasons >= rho (0.05 if none)."""
    grid = grid or [0.05 + 0.05 * i for i in range(10)]
    out = {}
    for s, d in D.items():
        idx = torch.where(train_mask[s])[0]
        dt = _sel_local(d, idx)
        site = dt["site"].long()
        best = torch.full((int(d["site"].max()) + 1,), 0.05)
        for g in grid:
            y, _ = evaluate(dt, torch.full((len(idx),), g), u_rule=u_season)
            r = y / refs[s][0][idx].clamp(min=0.05)
            sums = torch.zeros_like(best).index_add_(0, site, r)
            cnt = torch.zeros_like(best).index_add_(0, site, torch.ones_like(r))
            ok = (sums / cnt.clamp(min=1) >= rho) & (cnt > 0)
            best = torch.where(ok, torch.full_like(best, g), best)
        out[s] = best[d["site"].long()]            # per season row of the full set
    return out


def _sel_local(d, idx):
    return S._sel(d, idx)
