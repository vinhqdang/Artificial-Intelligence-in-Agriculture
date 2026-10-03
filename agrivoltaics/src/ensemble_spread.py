"""Simulated relative yield versus shading for one ensemble member (static arrays, baseline climate, held-out sites of
split 0, test years, rainfed), used to compare the spread among members with the between-study error of the data.
Usage: AV_VARIANT=member3 AV_POSTERIOR=shade_posterior_ml.json python src/ensemble_spread.py TAG"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A, simulate as S
from run_train import get_refs
torch.set_num_threads(1)
CROPS = ["wheat", "rice", "maize", "soybean", "potato"]
D = A.load_all(); refs = get_refs(D); d = D["baseline"]
test = A.site_split(0)
m = test[d["site"].numpy().astype(int)] & np.isin(d["year"].numpy(), A.TEST_YEARS) & (refs["baseline"][0] >= 0.2).numpy()
idx = torch.where(torch.tensor(m))[0]
idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(1))[:800]]
dt = S._sel(d, idx); rows = []
for gv in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    with torch.no_grad():
        y, e, rec = S.rollout(dt, torch.full((len(idx),), gv), irrigated=torch.zeros(len(idx)), record=True)
        y0, _ = S.rollout(dt, torch.full((len(idx),), gv), panels=False, irrigated=torch.zeros(len(idx)))
    act = dt["active"]; shade = 1 - (rec["shade"] * act).sum(1) / act.sum(1)
    for c in range(5):
        mm = (dt["crop"] == c) & (y0 > 0.2)
        rows.append(dict(member=sys.argv[1], crop=CROPS[c], g=gv, shading=float(shade[mm].mean()), rel_yield=float((y[mm] / y0[mm]).mean())))
pd.DataFrame(rows).to_csv(f"{A.ROOT}/results/spread_{sys.argv[1]}.csv", index=False)
