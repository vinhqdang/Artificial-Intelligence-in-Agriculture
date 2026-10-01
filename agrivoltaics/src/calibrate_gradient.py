"""Gradient-based Bayesian calibration of the crop shade response.

Per crop, two parameters enter the differentiable simulator: the radiation-use compensation c and a
multiplier h on the flowering-window harvest-index penalty. An ensemble of M members is fitted by
gradient descent through whole simulated seasons to published yield--shading observations (meta-analysis
curves and field trials, each with an observation error) with weak priors. Every member sees the data
perturbed by its error and a prior draw perturbed by its width (randomised maximum a posteriori), so the
ensemble approximates the posterior of the shade response. Seasons: training-block sites, 2001-2014.
Output: data/shade_posterior.json, results/shade_posterior.csv, results/shade_posterior_fit.csv
"""
import os, sys, json, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A, simulate as S, physics as P
from run_train import get_refs

ROOT = A.ROOT
CROPS = ["wheat", "rice", "maize", "soybean", "potato"]
GRID = [0.1, 0.2, 0.3, 0.4, 0.5]
M, NS, STEPS = 12, 40, 40
torch.set_num_threads(4)

# (crop, shading, kind, value, sigma); kind rel = relative yield, hi = harvest-index factor
LAUB = {"wheat": (0.85, 0.62), "rice": (0.85, 0.62), "maize": (0.72, 0.45), "soybean": (0.75, 0.50), "potato": (0.85, 0.60)}
OBS = []
for c, (a, b) in LAUB.items():
    OBS += [(c, 0.2, "rel", a, 0.06), (c, 0.4, "rel", b, 0.06)]
OBS += [("rice", 0.25, "rel", 0.842, 0.06),
        ("wheat", 0.30, "rel", 0.813, 0.08), ("wheat", 0.30, "rel", 1.027, 0.08),
        ("potato", 0.30, "rel", 0.818, 0.08), ("potato", 0.30, "rel", 1.110, 0.08),
        ("maize", 0.35, "rel", 0.84, 0.08), ("maize", 0.35, "rel", 0.71, 0.08), ("maize", 0.35, "rel", 0.95, 0.08),
        ("maize", 0.30, "rel", 0.91, 0.08), ("maize", 0.50, "rel", 0.70, 0.08),
        ("maize", 0.35, "hi", 1.00, 0.05)]       # harvest index of maize unaffected by shade (Ramos-Fuentes 2023)
SUFFIX = ""
if os.environ.get("AV_TARGETS") == "ml":      # targets from the data-driven hierarchical model (shade_ml_fit.py)
    SUFFIX = "_ml"
    ml = pd.read_csv(f"{ROOT}/results/shade_ml_curves.csv")
    tmap = {"wheat": "C3 cereals", "rice": "C3 cereals", "maize": "Maize", "soybean": "Grain legumes", "potato": "Tubers/root crops"}
    OBS = [o for o in OBS if o[3] is None or o[0] == "__none__"]
    OBS = []
    for c, t in tmap.items():
        for s in (0.2, 0.4):
            r = ml[(ml.ctype == t) & (abs(ml.s - s) < 1e-9)].iloc[0]
            OBS.append((c, s, "rel", float(r["median"]), max(0.04, float((r.hi - r.lo) / 2.56))))
    OBS += [("rice", 0.25, "rel", 0.842, 0.06),
            ("wheat", 0.30, "rel", 0.813, 0.08), ("wheat", 0.30, "rel", 1.027, 0.08),
            ("potato", 0.30, "rel", 0.818, 0.08), ("potato", 0.30, "rel", 1.110, 0.08),
            ("maize", 0.35, "rel", 0.84, 0.08), ("maize", 0.35, "rel", 0.71, 0.08), ("maize", 0.35, "rel", 0.95, 0.08),
            ("maize", 0.30, "rel", 0.91, 0.08), ("maize", 0.50, "rel", 0.70, 0.08),
            ("maize", 0.35, "hi", 1.00, 0.05)]
PRIOR_C, PRIOR_CSD, PRIOR_H, PRIOR_HSD = 0.4, 0.5, 1.0, 0.5


def interp(x, xs, ys):
    """Differentiable piecewise-linear interpolation of ys(xs) at scalar x."""
    x = float(np.clip(x, xs[0], xs[-1]))
    j = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    w = (x - xs[j]) / (xs[j + 1] - xs[j])
    return ys[j] * (1 - w) + ys[j + 1] * w


def subset():
    D = A.load_all(); d = D["baseline"]; refs = get_refs(D)
    m = (~A.site_split(0)[d["site"].numpy().astype(int)]) & (d["year"].numpy() <= 2014) & (refs["baseline"][0] >= 0.2).numpy()
    idx = torch.where(torch.tensor(m))[0]
    rng = np.random.default_rng(1); out = {}
    for ci, c in enumerate(CROPS):
        ii = idx[d["crop"][idx] == ci].numpy()
        out[c] = S._sel(d, torch.tensor(sorted(rng.choice(ii, NS, replace=False))))
    return out


def shade_cache(dc):
    """Season shading of static arrays at every grid GCR (independent of the crop response)."""
    n = dc["lat"].shape[0]; out = []
    with torch.no_grad():
        for g in GRID:
            _, _, rec = S.rollout(dc, torch.full((n,), g), irrigated=torch.zeros(n), record=True)
            out.append(rec["shade"])
    return out


def curves(dc, cpar, hpar, cache):
    """Simulated retention at the GCR grid for M members (cpar,hpar: (M,)) on one crop's seasons."""
    n = dc["lat"].shape[0]
    big = {k: (torch.cat([v] * M) if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n else v) for k, v in dc.items()}
    big["theta_c"] = cpar.repeat_interleave(n); big["theta_h"] = hpar.repeat_interleave(n)
    irr = torch.zeros(n * M)
    with torch.no_grad():
        y0, _ = S.rollout(big, torch.full((n * M,), 0.3), panels=False, irrigated=irr)
    ret, hif = [], []
    for gi, g in enumerate(GRID):
        y, _, rec = S.rollout(big, torch.full((n * M,), g), irrigated=irr, record=True,
                              shade_override=cache[gi].repeat(M, 1))
        ret.append((y.view(M, n).sum(1)) / y0.view(M, n).sum(1)); hif.append(rec["hif"].view(M, n).mean(1))
    return torch.stack(ret, 1), torch.stack(hif, 1)          # (M, G)


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    sub = subset(); cache = {c: shade_cache(sub[c]) for c in CROPS}
    shading = pd.read_csv(f"{ROOT}/results/shade_calibration.csv")
    sh = {c: shading[shading.crop == c].sort_values("g").shading.values for c in CROPS}
    cpar = torch.full((5, M), 0.4); hpar = torch.ones(5, M)
    cpar.requires_grad_(); hpar.requires_grad_()
    opt = torch.optim.Adam([cpar, hpar], lr=0.08)
    rng = np.random.default_rng(0)
    noise = {(i): rng.standard_normal(M) for i in range(len(OBS))}
    c0 = torch.tensor(PRIOR_C + PRIOR_CSD * rng.standard_normal((5, M)), dtype=torch.float32)
    h0 = torch.tensor(PRIOR_H + PRIOR_HSD * rng.standard_normal((5, M)), dtype=torch.float32)
    t0 = time.time()
    for step in range(STEPS):
        loss = 0.0
        for ci, c in enumerate(CROPS):
            ret, hif = curves(sub[c], cpar[ci].clamp(0, 2.5), hpar[ci].clamp(0, 2.0), cache[c])
            for i, (oc, s, kind, val, sd) in enumerate(OBS):
                if oc != c: continue
                sim = interp(s, sh[c], ret.T)[...] if kind == "rel" else interp(s, sh[c], hif.T)[...]
                target = torch.tensor(val + sd * noise[i], dtype=torch.float32)
                loss = loss + (((sim - target) / sd) ** 2).sum()
            loss = loss + (((cpar[ci] - c0[ci]) / PRIOR_CSD) ** 2).sum() + (((hpar[ci] - h0[ci]) / PRIOR_HSD) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            cpar.clamp_(0, 2.5); hpar.clamp_(0, 2.0)
        print(f"step {step} loss {float(loss):.1f} t={time.time() - t0:.0f}s c_mean={cpar.mean(1).detach().numpy().round(2)} "
              f"h_mean={hpar.mean(1).detach().numpy().round(2)}", flush=True)
        json.dump(dict(c=cpar.detach().T.tolist(), h=hpar.detach().T.tolist(), crops=CROPS, step=step),
                  open(f"{ROOT}/data/shade_posterior{SUFFIX}.json", "w"))
    cp, hp = cpar.detach(), hpar.detach()
    rows = [dict(crop=c, c_mean=float(cp[i].mean()), c_sd=float(cp[i].std()), h_mean=float(hp[i].mean()), h_sd=float(hp[i].std()))
            for i, c in enumerate(CROPS)]
    pd.DataFrame(rows).to_csv(f"{ROOT}/results/shade_posterior{SUFFIX}.csv", index=False); print(pd.DataFrame(rows).round(2))
    fit = []
    with torch.no_grad():
        for ci, c in enumerate(CROPS):
            ret, hif = curves(sub[c], cp[ci], hp[ci], cache[c])
            for gi, g in enumerate(GRID):
                fit.append(dict(crop=c, g=g, shading=float(sh[c][gi]), rel_mean=float(ret[:, gi].mean()), rel_sd=float(ret[:, gi].std()),
                                hi_mean=float(hif[:, gi].mean())))
    pd.DataFrame(fit).to_csv(f"{ROOT}/results/shade_posterior_fit{SUFFIX}.csv", index=False)
