"""Pair-level cluster-bootstrap intervals for net-value contrasts at 90 USD/MWh, and crossover prices (interpolated)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES
rng = np.random.default_rng(0); out = []
for name, f, col in [("floor-designed density", "cost_ler_pairs90.csv", "net90"), ("net-optimal density", "cost_optimal_pairs90.csv", "net_eff")]:
    d = pd.read_csv(f"{RES}/{f}").pivot(index="site", columns="method", values=col) / 1000
    for a, b in [("SALS", "AV static"), ("SALS", "Optimised ramp schedule"), ("SALS", "Tuned phenology rule"), ("SALS", "Phenology rule"), ("Optimised ramp schedule", "AV static")]:
        x = (d[a] - d[b]).dropna().values; bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        out.append(dict(design=name, a=a, b=b, diff_kusd=x.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), pairs=len(x)))
D = pd.DataFrame(out); D.to_csv(f"{RES}/cost_ci.csv", index=False); print(D.round(2).to_string())
R = pd.read_csv(f"{RES}/cost_optimal.csv").pivot(index="price", columns="method", values="net")
def cross(a, b):
    d = (R[a] - R[b]).values; p = R.index.values
    for i in range(len(p) - 1):
        if d[i] < 0 <= d[i + 1]: return p[i] - d[i] * (p[i + 1] - p[i]) / (d[i + 1] - d[i])
    return np.nan
for a in ("SALS", "Optimised ramp schedule", "Tuned phenology rule", "Stress rule", "Phenology rule"): print("net-optimal crossover with static arrays:", a, round(cross(a, "AV static"), 1))
