"""Net-value-optimal design: at each electricity price the density of a site is the feasible GCR (historical mean retention >= floor)
that maximises the historical mean net value; outcomes on test years. Pooled over three splits (main response, coarse grid).
Output: results/cost_optimal.csv, manuscript figure cost_optimal.pdf"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO, HIST, TEST
import cost as C
FIG = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
rows = []
for sp in (0, 1, 2):
    g = pd.read_parquet(f"{RES}/grid_ctl{sp}.parquet"); g["method"] = np.where(g.method.str.startswith("sals_"), "SALS", g.method)
    g = g[g.y_open >= 0.2].copy(); g["r"] = g.y / g.y_open
    comp = C.components(g); g["E"] = comp["E"]; g["closs"] = comp["closs"]; g["e_pv"] = comp["e_pv"]; g["split"] = sp
    rows.append(g[["method", "site", "year", "g", "r", "E", "closs", "split"]])
G = pd.concat(rows)
res = []
for price in range(20, 131, 10):
    p = dict(C.BASE, p_e=price)
    G["net"] = C.net_av(G.E.values, G.g.values, G.closs.values, p)
    h = G[G.year <= HIST].groupby(["method", "site", "g"]).agg(r=("r", "mean"), net=("net", "mean")).reset_index()
    h = h[h.r >= RHO]
    # open field (no array, net 0) is feasible and chosen when every feasible density loses money
    best = h.sort_values("net").groupby(["method", "site"]).tail(1)
    t = G[G.year >= TEST].merge(best[["method", "site", "g"]], on=["method", "site", "g"])
    pair = t.groupby(["method", "site"]).agg(net=("net", "mean"), g=("g", "mean"), E=("E", "mean")).reset_index()
    keep = best[best.net > 0][["method", "site"]]
    pair["array"] = pair.merge(keep.assign(a=1), on=["method", "site"], how="left").a.fillna(0).values
    pair["net_eff"] = np.where(pair.array == 1, pair.net, 0.0)       # sites where no density pays stay open field
    if price == 90:
        pair[["method", "site", "net_eff"]].to_csv(f"{RES}/cost_optimal_pairs90.csv", index=False)
    for m, x in pair.groupby("method"):
        res.append(dict(price=price, method=m, net=x.net_eff.mean(), share_array=x.array.mean() * 100, GCR=x[x.array == 1].g.mean() if (x.array == 1).any() else np.nan, pairs=len(x)))
R = pd.DataFrame(res); R.to_csv(f"{RES}/cost_optimal.csv", index=False)
print(R[R.price.isin([30, 60, 90, 120])].round(1).to_string())
fig, ax = plt.subplots(figsize=(3.6, 2.7)); col = {"AV static": "#6b7280", "Phenology rule": "#0369a1", "Tuned phenology rule": "#059669", "SALS": "#c2410c"}
for m, c in col.items():
    x = R[R.method == m]; ax.plot(x.price, x.net / 1000, color=c, label=m)
ax.axhline(0, color="#9ca3af", lw=0.6); ax.set_xlabel("Electricity price (USD MWh$^{-1}$)"); ax.set_ylabel("Net value vs open field (kUSD ha$^{-1}$ yr$^{-1}$)")
ax.legend(fontsize=5.5, frameon=False); fig.tight_layout(); fig.savefig(f"{FIG}/cost_optimal.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/cost_optimal.png", dpi=200)
