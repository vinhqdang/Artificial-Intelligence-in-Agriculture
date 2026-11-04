"""Oracle design (density chosen under the true response) on the regenerated grids, for the conformal/learned design comparison. Output: results/conformal_oracle.csv"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import analysis_conformal as AC           # re-runs the conformal analysis on import (deterministic), reuses load/select/outcome
rows = []
for sp in (0, 1, 2):
    for k in AC.MEM:
        t = AC.load(f"e2s{sp}_{k}"); s = AC.outcome(AC.select(t, 0.9), t); s["kind"] = "in-ensemble"; rows.append(s)
    for T in ["conservative", "calibrated", "field", "harsh"]:
        t = AC.load(f"nt{sp}_{T}"); s = AC.outcome(AC.select(t, 0.9), t); s["kind"] = "out-of-ensemble"; rows.append(s)
R = pd.concat(rows); R = R.groupby(["kind", "site"]).agg(ok=("ok", "mean"), ler=("ler", "mean")).reset_index()   # pair level
o = R.groupby("kind").agg(compliance=("ok", "mean"), ler=("ler", "mean")).reset_index(); o["compliance"] *= 100
o.to_csv(f"{AC.RES}/conformal_oracle.csv", index=False); print(o)
