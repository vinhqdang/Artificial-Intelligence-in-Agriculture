"""Held-out evaluation of SALS variants (ablation), without re-running baselines."""
import os, sys, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A
import simulate as S
from evaluate import load_model, rows_for
from run_train import get_refs

if __name__ == "__main__":
    models = json.loads(sys.argv[1])
    D = A.load_all(); refs = get_refs(D); test_site = A.site_split()
    out = []
    for scen in A.SCEN:
        d = D[scen]; site = d["site"].numpy().astype(int)
        idx = torch.where(torch.tensor(test_site[site] & np.isin(d["year"].numpy(), A.TEST_YEARS)))[0]
        dt = S._sel(d, idx); feats = A.site_features(dt)
        for mname, label in models.items():
            ctrl, des, cfg = load_model(mname)
            with torch.no_grad():
                g = des(feats) if cfg["use_design"] else torch.full((len(idx),), cfg["fixed_g"] or 0.3)
            y, e = A.evaluate(dt, g, controller=A.wrap(ctrl))
            out.append(rows_for(label, scen, d, idx, g, y, e, refs[scen]))
            print(scen, label, flush=True)
    pd.concat(out).to_csv(f"{A.ROOT}/results/eval_ablation.csv", index=False)
