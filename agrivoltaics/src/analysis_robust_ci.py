"""Cluster-bootstrap intervals (over crop-site pairs) for LER differences between controllers under the four fixed
shade responses (matched design), split 0, every sixth held-out pair. Output: results/robust_ci.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES
from analysis_robust import load, design, apply, TRUTHS, NAMES
rng = np.random.default_rng(0); rows = []
S = NAMES["sals_s0_seed0"]; R = NAMES["rob_s0"]; A_ = NAMES["adp_s0"]
for T in TRUTHS:
    g = load(T); x = apply(design(g, None), g)
    d = x.groupby(["method", "site"]).ler.mean().unstack(0)
    for a, b in [(S, "Phenology rule"), (R, S), (A_, S), ("Feedback rule", "Phenology rule"), ("SALS-robust, E2", S), ("SALS-robust, E3", S), ("SALS-adaptive, E3", S), ("SALS-robust, E3", "Phenology rule"), (S, "Tuned phenology rule"), (S, "Optimised ramp schedule")]:
        diff = (d[a] - d[b]).dropna().values; bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        rows.append(dict(truth=T, a=a, b=b, diff=diff.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), pairs=len(diff)))
r = pd.DataFrame(rows); r.to_csv(f"{RES}/robust_ci.csv", index=False); print(r.round(3).to_string())
