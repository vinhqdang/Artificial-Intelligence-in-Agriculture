"""Layout, rotation-limit and harvest-index-window sensitivity (split 0, every sixth held-out pair, coarse GCR grid, main response,
history-based design). Output: results/sensitivity_physics.csv and manuscript/tables/physsens.tex"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO
from analysis_robust import apply, design
rows = []
def run(tag, label):
    g = pd.read_parquet(f"{RES}/grid_{tag}.parquet"); g["method"] = np.where(g.method.str.startswith("sals_"), "SALS", g.method)
    g = g[g.y_open >= 0.2].copy(); g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    x = apply(design(g, None), g)
    for m, d in x.groupby("method"):
        p = d.groupby("site").agg(r=("r", "mean"), ler=("ler", "mean"), gcr=("g", "mean"), erel=("erel", "mean")).reset_index()
        rows.append(dict(setting=label, method=m, pairs=len(p), GCR=p.gcr.mean(), erel=p.erel.mean(), retention=p.r.mean(), LER=p.ler.mean()))
run("rot60", "rotation limit 60 deg (main)"); run("rot45", "rotation limit 45 deg"); run("rot80", "rotation limit 80 deg")
run("fx", "fixed arrays")
run("win_early", "harvest-index window 0.30-0.50"); run("win_late", "harvest-index window 0.60-0.80"); run("win_h0", "no harvest-index penalty")
D = pd.DataFrame(rows); D.to_csv(f"{RES}/sensitivity_physics.csv", index=False); pd.set_option("display.width", 200); print(D.round(3).to_string())
