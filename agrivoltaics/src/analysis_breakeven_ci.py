"""Cluster-bootstrap (over crop-site pairs) intervals for break-even electricity prices at floor-designed density.
Intervals cover the sampling of pairs only, not the cost parameters (see cost sensitivity)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES
from analysis_robust import apply, design
from analysis_cost import comp_rows, pooled
import cost as C
rows = []
for sp in (0, 1, 2):
    g = pd.read_parquet(f"{RES}/grid_ctl{sp}.parquet"); g = g[~g.method.str.match(r"sals_s\d_seed[12]")].copy(); g["method"] = np.where(g.method.str.startswith("sals_"), "SALS", g.method)
    g = g[g.y_open >= 0.2].copy(); g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    rows.append(comp_rows(apply(design(g, None), g)))
R = pooled(pd.concat(rows)); P = C.BASE
for m in R.method.unique(): R.loc[R.method == m, "cap"] = C.annual_capex(R[R.method == m].g.values, P["capex_av"], P)
rng = np.random.default_rng(0); sites = np.array(sorted(R.site.unique())); out = []
def be(x): return (x.cap.mean() + x.closs.mean()) / x.E.mean()
def plant(x): return (C.annual_capex(0.4, P["capex_pv"], P) + x.copen.mean()) / x.e_pv.mean()
Rs = {m: R[R.method == m].set_index("site") for m in R.method.unique()}
meths = ["AV static", "Phenology rule", "Optimised ramp schedule", "SALS"]
draw = [rng.choice(sites, len(sites)) for _ in range(2000)]
est = {m: be(Rs[m]) for m in meths}; est["plant"] = plant(Rs["SALS"])
bs = {m: [] for m in est}
for s in draw:
    for m in meths:
        x = Rs[m].reindex(s).dropna(); bs[m].append(be(x))
    x = Rs["SALS"].reindex(s).dropna(); bs["plant"].append(plant(x))
for m in est: out.append(dict(quantity=f"break-even vs open field: {m}", value=est[m], lo=np.quantile(bs[m], .025), hi=np.quantile(bs[m], .975)))
for a, b in [("SALS", "AV static"), ("SALS", "Optimised ramp schedule"), ("Optimised ramp schedule", "AV static")]:
    d = np.array(bs[a]) - np.array(bs[b]); out.append(dict(quantity=f"break-even difference: {a} - {b}", value=est[a] - est[b], lo=np.quantile(d, .025), hi=np.quantile(d, .975)))
D = pd.DataFrame(out); D.to_csv(f"{RES}/breakeven_ci.csv", index=False); print(D.round(1).to_string())
