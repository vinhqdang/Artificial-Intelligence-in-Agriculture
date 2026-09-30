"""Evaluate every controller over a grid of ground coverage ratios (GCR) on all
seasons (2001-2019) of the held-out sites of a spatial split.

The grid results support (i) deployable, history-based design selection
(largest GCR whose retention on the site's 2001-2014 seasons meets the floor),
applied identically to all methods, (ii) matched-GCR comparisons and (iii)
sensitivity analyses. Usage:
    python src/evaluate_grid.py SPLIT MODEL[,MODEL...] [kappas=1.0] [scen=all] [rules=1] [tag=...]
"""
import os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A
import simulate as S
from run_train import get_refs

ROOT = A.ROOT
torch.set_num_threads(4)
GRID = [round(0.05 + 0.025 * i, 3) for i in range(19)]      # 0.05 ... 0.50


def u_season(f):
    """Seasonal sharing: light-sharing rotation during the whole cropping season."""
    return f[:, 8]


def u_stress(f):
    """Expert stress rule: share light in the main growth phase, except on days
    with a heat-stress forecast or drought, when the trackers shade the crop."""
    tt, arid, heat, active = f[:, 0], f[:, 3], f[:, 4], f[:, 8]
    return active * ((tt > 0.1) & (tt < 0.9) & (heat < 0) & (arid < 0.5)).float()


def load_ctrl(name):
    m = torch.load(f"{ROOT}/results/models/{name}.pt")
    c = A.Controller(); c.load_state_dict(m["ctrl"]); c.mask = m["mask"]; c.eval()
    return A.wrap(c)


if __name__ == "__main__":
    split = int(sys.argv[1])
    models = [m for m in sys.argv[2].split(",") if m]
    kw = dict(a.split("=") for a in sys.argv[3:])
    kappas = [float(k) for k in kw.get("kappas", "1.0").split(",")]
    scens = kw.get("scen", ",".join(A.SCEN)).split(",")
    tag = kw.get("tag", f"split{split}")
    methods = {}
    if kw.get("rules", "1") == "1":
        methods.update({"AV static": None, "Seasonal sharing": u_season, "Stress rule": u_stress})
    for m in models:
        methods[m] = load_ctrl(m)
    D = A.load_all(); refs = get_refs(D)
    test_site = A.site_split(split)
    out = []
    t0 = time.time()
    for scen in scens:
        d = D[scen]
        idx = torch.where(torch.tensor(test_site[d["site"].numpy().astype(int)]))[0]
        dt = S._sel(d, idx)
        base = dict(site=dt["site"].numpy().astype(int), year=dt["year"].numpy().astype(int),
                    crop=dt["crop"].numpy().astype(int), irr=dt["irr_frac"].numpy(),
                    y_open=refs[scen][0][idx].numpy(), e_pv=refs[scen][1][idx].numpy())
        for kappa in kappas:
            for name, ctrl in methods.items():
                for g in GRID:
                    gg = torch.full((len(idx),), g)
                    if ctrl is None:
                        y, e = A.evaluate(dt, gg, kappa=kappa)
                    else:
                        y, e = A.evaluate(dt, gg, controller=ctrl, kappa=kappa)
                    out.append(pd.DataFrame(dict(base, method=name, scenario=scen, kappa=kappa, g=g,
                                                 y=y.numpy(), e=e.numpy())))
                print(f"{scen} kappa={kappa} {name} done {time.time() - t0:.0f}s", flush=True)
        pd.concat(out).to_parquet(f"{ROOT}/results/grid_{tag}.parquet", index=False)
    print("saved", tag)
