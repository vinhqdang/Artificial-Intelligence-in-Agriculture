"""Tune the window of the phenology rule on training-block seasons (baseline climate, main response), with the same
objective as the other rules: electricity at the density where mean retention equals the floor."""
import os, sys, itertools
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A, simulate as S
from run_train import get_refs, train_mask
torch.set_num_threads(2)
ROOT = A.ROOT
SPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
D = A.load_all(); refs = get_refs(D)
tm = train_mask(D, refs, A.site_split(SPLIT))["baseline"]
idx = torch.where(tm)[0]; g_ = torch.Generator().manual_seed(1)
idx = idx[torch.randperm(len(idx), generator=g_)[:500]]
d = S._sel(D["baseline"], idx); yo, ep = refs["baseline"][0][idx], refs["baseline"][1][idx]
G = [0.1, 0.2, 0.3, 0.4, 0.5]; rows = []
for a, b in itertools.product([0.0, 0.05, 0.15, 0.3], [0.6, 0.8, 1.0]):
    f = A.make_window(a, b); ret, er = [], []
    for g in G:
        y, e = A.evaluate(d, torch.full((len(idx),), g), u_rule=f)
        ret.append(float((y / yo).mean())); er.append(float((e / ep).mean()))
    ret, er = np.array(ret), np.array(er)
    # electricity at retention 0.9 (linear interpolation over GCR; retention decreases with GCR)
    e90 = float(np.interp(0.9, ret[::-1], er[::-1])) if ret.min() < 0.9 < ret.max() else float("nan")
    rows.append(dict(a=a, b=b, e_at_floor=e90, ret_at_0p3=ret[2])); print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{ROOT}/results/phenology_tuning.csv" if SPLIT == 0 else f"{ROOT}/results/phenology_tuning_split{SPLIT}.csv", index=False)
best = df.loc[df.e_at_floor.idxmax()]; print("best", best.a, best.b, best.e_at_floor)
