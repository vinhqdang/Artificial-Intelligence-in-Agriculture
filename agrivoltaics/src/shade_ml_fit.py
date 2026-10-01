"""Fit the hierarchical shade-response model with covariates to all data, with cluster (study) bootstrap
intervals for the crop-type curves and covariate effects. Output: results/shade_ml_curves.csv, shade_ml_coef.csv"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from shade_ml import load, fit, COV, ROOT
d = load(); G = d.g.nunique(); types = list(d.ctype.astype("category").cat.categories)
S = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
rng = np.random.default_rng(0)
studies = d.ref.unique()


def curves(m):
    out = {}
    for gi, t in enumerate(types):
        x = pd.DataFrame(dict(s=S, g=gi, ghi_z=0.0, sub=0.0, panel=0.0, inter=0.0))
        from shade_ml import predict
        out[t] = predict(m, x)
    return out


m0 = fit(d, G, "hier+cov")
base = curves(m0)
coef_hat = m0.delta.detach().numpy()
B = 150; boots = {t: [] for t in types}; coefs = []
for b in range(B):
    pick = rng.choice(studies, len(studies), replace=True)
    db = pd.concat([d[d.ref == s] for s in pick], ignore_index=True)
    mb = fit(db, G, "hier+cov")
    c = curves(mb)
    for t in types: boots[t].append(c[t])
    coefs.append(mb.delta.detach().numpy())
    if b % 25 == 0: print("boot", b, flush=True)
rows = []
for t in types:
    arr = np.array(boots[t]); n = int((d.ctype == t).sum()); ns = int(d[d.ctype == t].ref.nunique())
    for i, s in enumerate(S):
        rows.append(dict(ctype=t, s=s, n=n, studies=ns, median=float(base[t][i]), lo=float(np.quantile(arr[:, i], 0.1)), hi=float(np.quantile(arr[:, i], 0.9))))
pd.DataFrame(rows).to_csv(f"{ROOT}/results/shade_ml_curves.csv", index=False)
cf = np.array(coefs)
pd.DataFrame(dict(covariate=COV, effect_log_beta=coef_hat, lo=np.quantile(cf, 0.05, axis=0), hi=np.quantile(cf, 0.95, axis=0))
             ).to_csv(f"{ROOT}/results/shade_ml_coef.csv", index=False)
print(pd.read_csv(f"{ROOT}/results/shade_ml_coef.csv").round(3).to_string())
print(pd.DataFrame(rows).query("s in [0.2,0.4]").round(3).to_string())
