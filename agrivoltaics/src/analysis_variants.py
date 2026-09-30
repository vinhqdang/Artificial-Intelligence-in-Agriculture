"""Sensitivity of the light-sharing results to the crop shade response (rule controllers,
split 0, baseline climate, every fourth site). Output: results/shade_variants.csv"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import calibrate, summarize, RES, RULES

rows = []
for name, tag in [("conservative", "split0"), ("calibrated", "var_calibrated"), ("field-fitted", "var_field")]:
    g = pd.read_parquet(f"{RES}/grid_{tag}.parquet")
    g = g[(g.scenario == "baseline") & (g.site % 4 == 0) & g.method.isin(RULES) & (g.kappa == 1.0)]
    g = g[g.y_open >= 0.2].copy()
    g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    cal = calibrate(g)
    for m, x in cal.groupby("method"):
        s = summarize(x); s["variant"] = name; s["method"] = m; rows.append(s)
out = pd.DataFrame(rows)[["variant", "method", "sites", "GCR", "no_array", "erel", "r", "comply", "LER"]]
out.to_csv(f"{RES}/shade_variants.csv", index=False)
print(out.round(3).to_string())
