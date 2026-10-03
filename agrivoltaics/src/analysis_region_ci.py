"""Cluster-bootstrap (over sites) intervals for the regional figures of SALS on the main split (baseline climate, calibrated density):
production-weighted retention, share of sites with mean retention >= 0.9, density and LER. Output: results/by_region_ci.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import analysis as AN
from analysis import RES, ROOT, RHO, SALS, RULES, MIN_OPEN, calibrate, region
sites = pd.read_csv(f"{ROOT}/data/sites.csv")
g = pd.read_parquet(f"{RES}/grid_split0.parquet"); g = g[g.method.isin(RULES + ["sals_s0_seed0"])].copy()
g["method"] = g.method.replace({"sals_s0_seed0": SALS})
op = g.drop_duplicates(["site", "year", "scenario"]).pivot_table(index=["site", "year"], columns="scenario", values="y_open")
ok = (op >= MIN_OPEN).all(axis=1); ok = ok[ok].reset_index()[["site", "year"]]
g = g.merge(ok, on=["site", "year"]); g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
cal0 = calibrate(g)
x = cal0[(cal0.kappa == 1.0) & (cal0.scenario == "baseline") & (cal0.method == SALS)].copy()
x = x.join(sites[["lat", "lon"]], on="site"); x["region"] = [region(la, lo) for la, lo in zip(x.lat, x.lon)]
rng = np.random.default_rng(0); rows = []
for reg, d in x.groupby("region"):
    per = d.groupby("site").agg(y=("y", "sum"), yo=("y_open", "sum"), r=("r", "mean"), g=("g", "mean"), ler=("ler", "mean")).reset_index()
    def stat(p): return dict(r_prod=p.y.sum() / p.yo.sum(), site_ok=(p.r >= RHO).mean() * 100, GCR=p.g.mean(), LER=p.ler.mean())
    base = stat(per); bs = [stat(per.iloc[rng.integers(0, len(per), len(per))]) for _ in range(2000)]
    row = dict(region=reg, sites=len(per), pairs=len(d))
    for k, v in base.items(): row[k] = v; row[k + "_lo"] = np.quantile([b[k] for b in bs], .025); row[k + "_hi"] = np.quantile([b[k] for b in bs], .975)
    rows.append(row)
R = pd.DataFrame(rows); R.to_csv(f"{RES}/by_region_ci.csv", index=False)
print(R[["region", "sites", "r_prod", "r_prod_lo", "r_prod_hi", "site_ok", "site_ok_lo", "site_ok_hi", "GCR", "LER"]].round(3).to_string())
