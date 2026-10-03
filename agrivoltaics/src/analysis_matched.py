"""Matched-retention comparison of controllers, free of the 0.05 density grid of the design step: for every crop-site pair
the test-season mean retention r(g) and land-equivalent ratio LER(g) are computed at each grid density and LER is read off
at mean retention 0.90 by linear interpolation. Pair-level differences are averaged over the splits in which a site was held
out and bootstrapped over distinct pairs. Also reports differences at fixed densities. Output: results/matched_frontier.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO
TEST = 2015; rng = np.random.default_rng(0)
tag = sys.argv[1] if len(sys.argv) > 1 else "ctl"
rows = []
for sp in (0, 1, 2):
    g = pd.read_parquet(f"{RES}/grid_{tag}{sp}.parquet")
    g["method"] = np.where(g.method.str.startswith("sals_"), "SALS#" + g.method.str[-1], g.method)
    g = g[(g.y_open >= 0.2) & (g.year >= TEST)].copy(); g["r"] = g.y / g.y_open; g["ler"] = g.r + g.e / g.e_pv
    m = g.groupby(["method", "site", "g"]).agg(r=("r", "mean"), ler=("ler", "mean")).reset_index(); m["split"] = sp; rows.append(m)
M = pd.concat(rows)
out = []
for (meth, site, sp), x in M.groupby(["method", "site", "split"]):
    x = x.sort_values("g"); r, l = x.r.values, x.ler.values
    if r.min() < RHO < r.max():
        o = np.argsort(r); out.append(dict(method=meth, site=site, split=sp, ler90=float(np.interp(RHO, r[o], l[o]))))
F = pd.DataFrame(out)
sm = F[F.method.str.startswith("SALS#")].groupby(["site", "split"]).ler90.mean().reset_index().assign(method="SALS")   # mean over training seeds
F = pd.concat([F[~F.method.str.startswith("SALS#")], sm]).groupby(["method", "site"]).ler90.mean().reset_index()
S = F.groupby("method").ler90.agg(["mean", "size"]).reset_index(); print(S.round(4).to_string())
res = []
for a, b in [("SALS", "Optimised ramp schedule"), ("SALS", "Tuned phenology rule"), ("SALS", "Phenology rule"), ("Optimised ramp schedule", "Phenology rule")]:
    pa, pb = F[F.method == a].set_index("site").ler90, F[F.method == b].set_index("site").ler90
    ids = pa.index.intersection(pb.index); dd = (pa.loc[ids] - pb.loc[ids]).values
    bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
    res.append(dict(a=a, b=b, diff=dd.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), median=np.median(dd), pairs=len(dd)))
R = pd.DataFrame(res); R.to_csv(f"{RES}/matched_frontier{'' if tag=='ctl' else '_'+tag}.csv", index=False); print(R.round(4).to_string())
# differences at fixed densities
fx = []
for gv in (0.10, 0.15, 0.20, 0.25, 0.30):
    Mx = M.copy(); Mx["method"] = np.where(Mx.method.str.startswith("SALS#"), "SALS", Mx.method)
    z = Mx[np.isclose(Mx.g, gv)].groupby(["method", "site"])[["r", "ler"]].mean().reset_index()
    for b in ("Optimised ramp schedule",):
        pa, pb = z[z.method == "SALS"].set_index("site"), z[z.method == b].set_index("site"); ids = pa.index.intersection(pb.index)
        dd = (pa.loc[ids].ler - pb.loc[ids].ler).values; bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
        fx.append(dict(g=gv, b=b, dLER=dd.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), dRet=(pa.loc[ids].r - pb.loc[ids].r).mean(), pairs=len(ids)))
print(pd.DataFrame(fx).round(4).to_string()); pd.DataFrame(fx).to_csv(f"{RES}/matched_fixed_density.csv", index=False)
