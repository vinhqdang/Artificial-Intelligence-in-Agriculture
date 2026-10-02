"""Design under shade-response uncertainty pooled over three spatial splits (ensemble E2), with site-level
cluster bootstrap intervals. Output: results/design_splits.csv, results/design_splits_boot.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO
from analysis_design import feas, pick, oracle, Q
from analysis_robust import apply
MEMBERS = [0, 3, 5, 8, 10]
rng = np.random.default_rng(0)


def load(tag):
    g = pd.read_parquet(f"{RES}/grid_{tag}.parquet")
    g["method"] = np.where(g.method.str.startswith("sals_"), "SALS", g.method)
    g = g[g.method.isin(["AV static", "Phenology rule", "SALS"]) & (g.y_open >= 0.2)].copy()
    g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    return g


rows = []
for sp in (0, 1, 2):
    mem = {k: load(f"e2s{sp}_{k}") for k in MEMBERS}; mean_g = load(f"e2s{sp}_mean")
    for k in MEMBERS:
        others = [mem[j] for j in MEMBERS if j != k]
        for design, sel in [("oracle", oracle(mem[k])), ("ensemble mean", pick([feas(mean_g)], 1.0)),
                            ("uncertainty-aware", pick([feas(o) for o in others], Q))]:
            x = apply(sel, mem[k])
            s = x.groupby(["method", "site"]).agg(r=("r", "mean"), ler=("ler", "mean"), gcr=("g", "mean"),
                                                   season_ok=("r", lambda v: (v >= RHO).mean())).reset_index()
            s["split"] = sp; s["member"] = k; s["design"] = design; rows.append(s)
S = pd.concat(rows)
S["site_ok"] = (S.r >= RHO).astype(float)
site = S.groupby(["method", "design", "split", "site"])[["r", "ler", "gcr", "season_ok", "site_ok"]].mean().reset_index()
summ = site.groupby(["method", "design"]).agg(sites=("site", "size"), GCR=("gcr", "mean"), retention=("r", "mean"),
                                              seasons_ok=("season_ok", "mean"), sites_ok=("site_ok", "mean"), LER=("ler", "mean")).reset_index()
summ[["seasons_ok", "sites_ok"]] *= 100
summ.to_csv(f"{RES}/design_splits.csv", index=False); print(summ.round(3).to_string())
boot = []
for m in ["Phenology rule", "SALS", "AV static"]:
    a = site[(site.method == m) & (site.design == "uncertainty-aware")].set_index(["split", "site"])
    b = site[(site.method == m) & (site.design == "ensemble mean")].set_index(["split", "site"])
    o = site[(site.method == m) & (site.design == "oracle")].set_index(["split", "site"])
    idx = a.index.intersection(b.index).intersection(o.index)
    a, b, o = a.loc[idx], b.loc[idx], o.loc[idx]
    n = len(idx); draws = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        draws.append([(a.site_ok.values[i] - b.site_ok.values[i]).mean() * 100, (a.ler.values[i] - b.ler.values[i]).mean(),
                      (o.site_ok.values[i] - a.site_ok.values[i]).mean() * 100])
    d = np.array(draws)
    for j, name in enumerate(["site compliance, aware minus mean (pp)", "LER, aware minus mean", "site compliance, oracle minus aware (pp)"]):
        boot.append(dict(method=m, contrast=name, est=float(d[:, j].mean()), lo=float(np.quantile(d[:, j], 0.025)),
                         hi=float(np.quantile(d[:, j], 0.975)), sites=n))
B = pd.DataFrame(boot); B.to_csv(f"{RES}/design_splits_boot.csv", index=False); print(B.round(3).to_string())
