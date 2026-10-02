"""Design under shade-response uncertainty pooled over three spatial splits, with cluster bootstrap over
distinct sites, and a cost-matched comparison with a higher floor under the mean response.
Usage: python analysis_splits.py [prefix=e2s] [out=design_splits]
Outputs: results/<out>.csv, <out>_boot.csv, <out>_frontier.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO, HIST
from analysis_robust import apply
MEMBERS = [0, 3, 5, 8, 10]
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "e2s"
OUT = sys.argv[2] if len(sys.argv) > 2 else "design_splits"
rng = np.random.default_rng(0)


def load(tag):
    g = pd.read_parquet(f"{RES}/grid_{tag}.parquet")
    g["method"] = np.where(g.method.str.startswith("sals_"), "SALS", g.method)
    g = g[g.method.isin(["AV static", "Phenology rule", "SALS"]) & (g.y_open >= 0.2)].copy()
    g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    return g


def feas(g, rho=RHO):
    h = g[g.year <= HIST].groupby(["method", "site", "g"]).r.mean().reset_index()
    h["ok"] = (h.r >= rho).astype(float)
    return h[["method", "site", "g", "ok"]]


def pick(fl, q):
    f = pd.concat(fl).groupby(["method", "site", "g"]).ok.mean().reset_index()
    return f[f.ok >= q].groupby(["method", "site"]).g.max().rename("g_sel").reset_index()


rows = []
for sp in (0, 1, 2):
    mem = {k: load(f"{PREFIX}{sp}_{k}") for k in MEMBERS}; mean_g = load(f"{PREFIX}{sp}_mean")
    for k in MEMBERS:
        others = [feas(mem[j]) for j in MEMBERS if j != k]
        designs = [("oracle", pick([feas(mem[k])], 1.0)), ("ensemble mean", pick([feas(mean_g)], 1.0))]
        designs += [(f"aware q={q}", pick(others, q)) for q in (0.5, 0.75, 0.9)]
        designs += [(f"margin floor={f}", pick([feas(mean_g, f)], 1.0)) for f in (0.92, 0.94, 0.96, 0.98)]
        for design, sel in designs:
            x = apply(sel, mem[k])
            s = x.groupby(["method", "site"]).agg(r=("r", "mean"), ler=("ler", "mean"), gcr=("g", "mean"),
                                                   season_ok=("r", lambda v: (v >= RHO).mean())).reset_index()
            s["split"] = sp; s["member"] = k; s["design"] = design; rows.append(s)
S = pd.concat(rows); S["site_ok"] = (S.r >= RHO).astype(float)
site = S.groupby(["method", "design", "split", "site"])[["r", "ler", "gcr", "season_ok", "site_ok"]].mean().reset_index()
# one record per distinct site (a site can appear in several splits as a held-out site)
dsite = site.groupby(["method", "design", "site"])[["r", "ler", "gcr", "season_ok", "site_ok"]].mean().reset_index()
summ = dsite.groupby(["method", "design"]).agg(sites=("site", "size"), GCR=("gcr", "mean"), retention=("r", "mean"),
                                               seasons_ok=("season_ok", "mean"), sites_ok=("site_ok", "mean"), LER=("ler", "mean")).reset_index()
summ[["seasons_ok", "sites_ok"]] *= 100
summ.to_csv(f"{RES}/{OUT}.csv", index=False)
pd.set_option("display.width", 200); print(summ[summ.method != "AV static"].round(3).to_string())
boot = []
for m in ["Phenology rule", "SALS", "AV static"]:
    P = {d: dsite[(dsite.method == m) & (dsite.design == d)].set_index("site") for d in dsite.design.unique()}
    ids = P["oracle"].index
    for a, b, name in [("aware q=0.75", "ensemble mean", "aware minus mean"), ("oracle", "aware q=0.75", "oracle minus aware"),
                       ("aware q=0.75", "margin floor=0.94", "aware minus margin 0.94")]:
        pa, pb = P[a].loc[ids], P[b].loc[ids]; d = []
        for _ in range(2000):
            i = rng.integers(0, len(ids), len(ids))
            d.append([(pa.site_ok.values[i] - pb.site_ok.values[i]).mean() * 100, (pa.ler.values[i] - pb.ler.values[i]).mean()])
        d = np.array(d)
        boot.append(dict(method=m, contrast=name, compliance_pp=d[:, 0].mean(), c_lo=np.quantile(d[:, 0], .025), c_hi=np.quantile(d[:, 0], .975),
                         ler=d[:, 1].mean(), l_lo=np.quantile(d[:, 1], .025), l_hi=np.quantile(d[:, 1], .975), sites=len(ids)))
B = pd.DataFrame(boot); B.to_csv(f"{RES}/{OUT}_boot.csv", index=False); print(B.round(3).to_string())
fr = summ[summ.method.isin(["Phenology rule", "SALS"])][["method", "design", "GCR", "sites_ok", "seasons_ok", "LER"]]
fr.to_csv(f"{RES}/{OUT}_frontier.csv", index=False)
