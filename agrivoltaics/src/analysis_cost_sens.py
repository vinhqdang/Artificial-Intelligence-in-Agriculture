"""One-at-a-time sensitivity of the net value at 90 USD/MWh (LER-designed densities, pooled splits); writes manuscript/tables/costsens.tex"""
import os, sys, copy
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES
from analysis_robust import apply, design
import cost as C
rows = []
for sp in (0, 1, 2):
    g = pd.read_parquet(f"{RES}/grid_ctl{sp}.parquet"); g["method"] = np.where(g.method.str.startswith("sals_"), "SALS", g.method)
    g = g[g.y_open >= 0.2].copy(); g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    x = apply(design(g, None), g); c = C.components(x); d = pd.DataFrame(c); d["site"] = x.site.values; d["method"] = x.method.values; rows.append(d)
R = pd.concat(rows).groupby(["method", "site"])[["E", "g", "e_pv", "closs", "copen"]].mean().reset_index()
def net(m, p):
    x = R[R.method == m]; return C.net_av(x.E.values, x.g.values, x.closs.values, p).mean() / 1000
def plant(p):
    x = R[R.method == "SALS"]; return C.net_pv_plant(x.e_pv.values, x.copen.values, p).mean() / 1000
base = dict(C.BASE, p_e=90.0)
L = [r"\begin{table}[t]", r"\centering\small", r"\caption{One-at-a-time sensitivity of the net value (kUSD\,ha$^{-1}$\,yr$^{-1}$, relative to the open field) at 90\,USD\,MWh$^{-1}$, densities designed for the floor as in the main analysis. In the capex rows the capex of the conventional plant is set to 0.75 times that of the agrivoltaic plant.}\label{tab:costsens}",
     r"\begin{tabular}{llcccc}", r"\toprule", r"Parameter & Value & Static arrays & Optimised window & SALS & Conventional plant\\", r"\midrule"]
def add(label, val, p, p_pv=None):
    pv = plant(p if p_pv is None else p_pv)
    L.append(f"{label} & {val} & {net('AV static', p):.1f} & {net('Optimised ramp schedule', p):.1f} & {net('SALS', p):.1f} & {pv:.1f}\\\\")
add("Base", "", base)
for v in (0.7, 1.5): add("AV capex (USD/Wp)", f"{v}", dict(base, capex_av=v), dict(base, capex_pv=v * 0.75))
for v in (0.04, 0.08): add("Discount rate", f"{int(v*100)}\\%", dict(base, rate=v))
for v in (0.5, 2.0): add("Crop prices", f"{v}$\\times$", dict(base, crop_mult=v))
for v in (60, 120): add("Electricity price", f"{v}", dict(base, p_e=v))
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(os.path.dirname(__file__), "..", "manuscript", "tables", "costsens.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
