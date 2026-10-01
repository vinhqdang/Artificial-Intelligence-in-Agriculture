"""Choose the target ratio of the feedback rule on training-block seasons (2001-2014, baseline climate),
with the shade response drawn from the calibrated ensemble, so that the rule is tuned like a deployed one."""
import os, sys, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A, simulate as S
from run_train import get_refs, train_mask
torch.set_num_threads(2)
ROOT = A.ROOT
D = A.load_all(); refs = get_refs(D)
tm = train_mask(D, refs, A.site_split(0))["baseline"]
idx = torch.where(tm)[0]
g_ = torch.Generator().manual_seed(0)
idx = idx[torch.randperm(len(idx), generator=g_)[:400]]
post = json.load(open(f"{ROOT}/data/shade_posterior.json"))
theta = dict(c=torch.tensor(post["c"]), h=torch.tensor(post["h"]))
d = A.attach_theta(S._sel(D["baseline"], idx), theta, g_)
yo, ep = refs["baseline"][0][idx], refs["baseline"][1][idx]
rows = []
for g in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y, e = A.evaluate(d, torch.full((len(idx),), g), u_rule=A.u_phenology)
    print('phenology', g, float((y / yo).mean()), float((e / ep).mean()), flush=True)
for rs in [float(x) for x in sys.argv[1].split(',')]:
    f = A.make_feedback(rs)
    for g in [0.1, 0.2, 0.3, 0.4, 0.5]:
        y, e = A.evaluate(d, torch.full((len(idx),), g), controller=f, track_ref=True)
        rows.append(dict(r_star=rs, g=g, ret=float((y / yo).mean()), erel=float((e / ep).mean())))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{ROOT}/results/feedback_tuning_{sys.argv[2]}.csv", index=False)
best = {}
for rs, x in df.groupby("r_star"):
    x = x.sort_values("g")
    ok = x[x.ret >= 0.9]
    best[rs] = float(ok.erel.iloc[-1]) if len(ok) else 0.0
print(best)
