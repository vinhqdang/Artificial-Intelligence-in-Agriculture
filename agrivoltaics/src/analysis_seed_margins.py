"""LER margin of each SALS training seed over the CMA-ES window (design-based, three splits pooled over distinct pairs, and per split).
Output: results/seed_margins.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO
from analysis_robust import apply, design
rng = np.random.default_rng(0); rows = []
for sp in (0, 1, 2):
    g = pd.read_parquet(f"{RES}/grid_ctl{sp}.parquet"); g["method"] = np.where(g.method.str.startswith("sals_"), "SALS#" + g.method.str[-1], g.method)
    g = g[g.y_open >= 0.2].copy(); g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    x = apply(design(g, None), g); x["split"] = sp; rows.append(x)
X = pd.concat(rows); s = X.groupby(["method", "split", "site"]).ler.mean().reset_index(); d = s.groupby(["method", "site"]).ler.mean().reset_index()
out = []
def add(label, a, b):
    ids = a.index.intersection(b.index); dd = (a[ids] - b[ids]).values; bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
    out.append(dict(scope=label, diff=dd.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), pairs=len(ids)))
r = d[d.method == "Optimised ramp schedule"].set_index("site").ler
for k in "012":
    add(f"seed {k}, pooled", d[d.method == "SALS#" + k].set_index("site").ler, r)
    for sp in (0, 1, 2):
        ss = s[s.split == sp]; add(f"seed {k}, split {sp}", ss[ss.method == "SALS#" + k].set_index("site").ler, ss[ss.method == "Optimised ramp schedule"].set_index("site").ler)
R = pd.DataFrame(out); R.to_csv(f"{RES}/seed_margins.csv", index=False); print(R.round(4).to_string())
