"""Propagation of the PV-model bias at higher ground coverage (Section 'PV output'): the electricity of every method is multiplied by the
ratio c(g) = relative pvlib plane-of-array irradiance / relative simulated irradiance (relative to g = 0.05, interpolated over g; from
results/pv_validation_gcr.csv) or by c(g)^2 (a bound that allows the partial-shading loss of DC energy to exceed that of irradiance).
Retention, hence the density design, does not depend on electricity, so the designs of the main analysis are kept.
Output: results/pvbias.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO
from analysis_robust import apply, design
import cost as C
rng = np.random.default_rng(0)
v = pd.read_csv(f"{RES}/pv_validation_gcr.csv"); m = v.groupby("gcr")[["rel_sim", "rel_pvlib"]].mean(); cg = (m.rel_pvlib / m.rel_sim)
c_of = lambda g: np.interp(g, cg.index.values, cg.values)
print("c(g):", cg.round(4).to_dict())
rows = []
for sp in (0, 1, 2):
    g = pd.read_parquet(f"{RES}/grid_ctl{sp}.parquet"); g = g[~g.method.str.match(r"sals_s\d_seed[12]")].copy()
    g["method"] = np.where(g.method.str.startswith("sals_"), "SALS", g.method)
    g = g[g.y_open >= 0.2].copy(); g["r"] = g.y / g.y_open
    sel = design(g, None); x0 = apply(sel, g)
    for name, power in (("main analysis", 0), ("pvlib-corrected", 1), ("bound (c squared)", 2)):
        x = x0.copy(); x["e"] = x.e * c_of(x.g.values) ** power; x["e_pv"] = x.e_pv * c_of(0.4) ** power   # the reference plant (g = 0.4, backtracking) is corrected in the same way
        x["erel"] = x.e / x.e_pv; x["ler"] = x.r + x.erel; x["scenario"] = name; x["split"] = sp; rows.append(x)
X = pd.concat(rows)
out = []
for sc, x in X.groupby("scenario"):
    d = x.groupby(["method", "site", "split"]).agg(ler=("ler", "mean"), E=("e", "mean"), g=("g", "mean"), y_open=("y_open", "mean"), y=("y", "mean"), crop=("crop", "first"), e_pv=("e_pv", "mean")).reset_index()
    d = d.groupby(["method", "site"]).agg(ler=("ler", "mean"), E=("E", "mean"), g=("g", "mean"), y_open=("y_open", "mean"), y=("y", "mean"), crop=("crop", "first"), e_pv=("e_pv", "mean")).reset_index()
    P = {k: d[d.method == k].set_index("site") for k in d.method.unique()}
    for k in ["AV static", "Phenology rule", "Optimised ramp schedule", "SALS"]:
        p = P[k]; price = np.array([C.BASE["price"][C.CROPS[int(i)]] for i in p.crop]); closs = price * (p.y_open - p.y).values
        cap = C.annual_capex(p.g.values, C.BASE["capex_av"], C.BASE)
        out.append(dict(scenario=sc, quantity=f"LER {k}", value=p.ler.mean()))
        out.append(dict(scenario=sc, quantity=f"break-even {k}", value=(cap.mean() + closs.mean()) / p.E.mean()))
        out.append(dict(scenario=sc, quantity=f"net at 90 {k}", value=(90 * p.E - cap - closs).mean() / 1000))
    ps = P["SALS"]; price = np.array([C.BASE["price"][C.CROPS[int(i)]] for i in ps.crop]); copen = price * ps.y_open.values
    for tag, capx in (("0.75 USD/Wp", C.BASE["capex_pv"]), ("1.0 USD/Wp (equal capex)", C.BASE["capex_av"])):
        capp = C.annual_capex(0.4, capx, C.BASE)
        out.append(dict(scenario=sc, quantity=f"plant break-even vs open field, plant capex {tag}", value=(capp + copen.mean()) / ps.e_pv.mean()))
        for k in ["AV static", "SALS", "Optimised ramp schedule"]:
            p = P[k]; pr = np.array([C.BASE["price"][C.CROPS[int(i)]] for i in p.crop]); closs = pr * (p.y_open - p.y).values; cap = C.annual_capex(p.g.values, C.BASE["capex_av"], C.BASE)
            out.append(dict(scenario=sc, quantity=f"plant overtakes {k} above (USD/MWh), plant capex {tag}", value=((capp + copen.mean()) - (cap.mean() + closs.mean())) / (ps.e_pv.mean() - p.E.mean())))
    for a, b in [("SALS", "Optimised ramp schedule"), ("SALS", "Phenology rule"), ("SALS", "AV static")]:
        ids = P[a].index.intersection(P[b].index); dd = (P[a].ler[ids] - P[b].ler[ids]).values
        bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
        out.append(dict(scenario=sc, quantity=f"dLER {a} - {b}", value=dd.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
R = pd.DataFrame(out); R.to_csv(f"{RES}/pvbias.csv", index=False)
print(R.pivot_table(index="quantity", columns="scenario", values="value").round(4).to_string())
