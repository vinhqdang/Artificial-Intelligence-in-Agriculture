"""Extra experiments on split 0.

rho   : controllers evaluated for floors rho in {0.8,0.85,0.9,0.95}; the rho-conditioned
        network receives rho as input, the others are the 0.9-trained networks and rules.
noise : robustness to forecast error: error scale 0, 1, 2, 3 x the training errors, at
        fixed densities (matched-density comparison).
Usage: python src/run_extra.py rho|noise
"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A
import simulate as S
from evaluate_grid import load_ctrl, GRID
from run_train import get_refs

ROOT = A.ROOT
torch.set_num_threads(4)
LEVELS = [0.8, 0.85, 0.9, 0.95]

if __name__ == "__main__":
    mode = sys.argv[1]
    D = A.load_all(); refs = get_refs(D); te = A.site_split(0)
    out = []
    for scen in ["baseline", "+3C"]:
        d = D[scen]
        idx = torch.where(torch.tensor(te[d["site"].numpy().astype(int)]))[0]
        dt = S._sel(d, idx)
        base = dict(site=dt["site"].numpy().astype(int), year=dt["year"].numpy().astype(int), crop=dt["crop"].numpy().astype(int),
                    y_open=refs[scen][0][idx].numpy(), e_pv=refs[scen][1][idx].numpy())
        if mode == "rho":
            models = {"SALS rho-conditioned": ("sals_rhocond", True), "SALS chance-trained": ("sals_chance", False)}
            for name, (mdl, cond) in models.items():
                if not os.path.exists(f"{ROOT}/results/models/{mdl}.pt"):
                    continue
                c = load_ctrl(mdl)
                for rho in (LEVELS if cond else [0.9]):
                    for g in GRID:
                        y, e = A.evaluate(dt, torch.full((len(idx),), g), controller=c, rho=rho if cond else None)
                        out.append(pd.DataFrame(dict(base, method=name, scenario=scen, kappa=1.0, g=g, rho=rho, y=y.numpy(), e=e.numpy())))
                    print(scen, name, rho, flush=True)
        else:
            ctrls = {"Phenology rule": A.u_phenology, "Stress rule": A.u_stress, "SALS": load_ctrl("sals_s0_seed0")}
            for ns in [0.0, 1.0, 2.0, 3.0]:
                for name, c in ctrls.items():
                    for g in [0.2, 0.3]:
                        y, e = A.evaluate(dt, torch.full((len(idx),), g), controller=c, noise_scale=ns)
                        out.append(pd.DataFrame(dict(base, method=name, scenario=scen, kappa=1.0, g=g, noise=ns, y=y.numpy(), e=e.numpy())))
                print(scen, "noise", ns, flush=True)
    pd.concat(out).to_parquet(f"{ROOT}/results/extra_{mode}.parquet", index=False)
    print("saved", mode)
