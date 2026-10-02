"""Controllers including a tuned phenology rule, pooled over three splits (main response, baseline climate,
history-based design, coarse GCR grid, every third held-out crop-site pair), cluster bootstrap over distinct pairs."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO, summarize
from analysis_robust import apply, design
rng = np.random.default_rng(0)
rows = []
for sp in (0, 1, 2):
    g = pd.read_parquet(f"{RES}/grid_ctl{sp}.parquet")
    g["method"] = np.where(g.method.str.startswith("sals_"), "SALS", g.method)
    g = g[(g.y_open >= 0.2)].copy(); g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    x = apply(design(g, None), g); x["split"] = sp; rows.append(x)
X = pd.concat(rows)
site = X.groupby(["method", "split", "site"]).agg(r=("r", "mean"), ler=("ler", "mean"), gcr=("g", "mean"), erel=("erel", "mean"),
                                                   season_ok=("r", lambda v: (v >= RHO).mean())).reset_index()
d = site.groupby(["method", "site"])[["r", "ler", "gcr", "erel", "season_ok"]].mean().reset_index()
d["site_ok"] = (d.r >= RHO).astype(float)
S = d.groupby("method").agg(pairs=("site", "size"), GCR=("gcr", "mean"), erel=("erel", "mean"), retention=("r", "mean"),
                            seasons_ok=("season_ok", "mean"), sites_ok=("site_ok", "mean"), LER=("ler", "mean")).reset_index()
S[["seasons_ok", "sites_ok"]] *= 100
S.to_csv(f"{RES}/controllers_tuned.csv", index=False); print(S.round(3).to_string())
P = {m: d[d.method == m].set_index("site") for m in d.method.unique()}
ids = P["SALS"].index
for a, b in [("SALS", "Tuned phenology rule"), ("SALS", "Phenology rule"), ("Tuned phenology rule", "Phenology rule")]:
    ids2 = P[a].index.intersection(P[b].index); pa, pb = P[a].loc[ids2], P[b].loc[ids2]; dd = []
    for _ in range(2000):
        i = rng.integers(0, len(ids2), len(ids2)); dd.append((pa.ler.values[i] - pb.ler.values[i]).mean())
    print(a, "-", b, round(float((pa.ler - pb.ler).mean()), 4), np.round(np.quantile(dd, [0.025, 0.975]), 4), len(ids2))
