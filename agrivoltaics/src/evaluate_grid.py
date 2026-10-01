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
if os.environ.get("AV_GRID") == "coarse":
    GRID = [round(0.05 * i, 2) for i in range(1, 11)]


from sals import u_season, u_phenology, u_stress


def load_ctrl(name):
    m = torch.load(f"{ROOT}/results/models/{name}.pt")
    c = A.Controller(n_in=len(m["mask"])); c.load_state_dict(m["ctrl"]); c.mask = m["mask"]; c.eval()
    f = A.wrap(c); f.track_ref = bool(m["cfg"].get("track_ref", False)); return f


if __name__ == "__main__":
    split = int(sys.argv[1])
    models = [m for m in sys.argv[2].split(",") if m]
    kw = dict(a.split("=") for a in sys.argv[3:])
    kappas = [float(k) for k in kw.get("kappas", "1.0").split(",")]
    scens = kw.get("scen", ",".join(A.SCEN)).split(",")
    tag = kw.get("tag", f"split{split}")
    methods = {}
    if kw.get("rules", "1") == "1":
        methods.update({"AV static": None, "Seasonal sharing": u_season, "Phenology rule": u_phenology,
                        "Stress rule": u_stress})
    if kw.get("feedback"):
        for rs in kw["feedback"].split(","):
            f = A.make_feedback(float(rs)); f.track_ref = True; methods[f"Feedback rule"] = f
    for m in models:
        methods[m] = load_ctrl(m)
    if kw.get("only"):
        keep = kw["only"].split(",")
        methods = {k: v for k, v in methods.items() if k.replace(' ', '_') in keep}
    D = A.load_all(); refs = get_refs(D)
    test_site = A.site_split(split)
    cache = f"{ROOT}/results/grid_cache/{tag}"          # one file per (scenario, kappa, method): restartable
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()
    for scen in scens:
        d = D[scen]
        sel = test_site[d["site"].numpy().astype(int)]
        sub = int(kw.get("sub", 1))
        if sub > 1:      # thinned site sample (sensitivity runs)
            sel = sel & (d["site"].numpy().astype(int) % sub == 0)
        idx = torch.where(torch.tensor(sel))[0]
        dt = S._sel(d, idx)
        base = dict(site=dt["site"].numpy().astype(int), year=dt["year"].numpy().astype(int),
                    crop=dt["crop"].numpy().astype(int), irr=dt["irr_frac"].numpy(),
                    y_open=refs[scen][0][idx].numpy(), e_pv=refs[scen][1][idx].numpy())
        for kappa in kappas:
            for name, ctrl in methods.items():
                fn = f"{cache}/{scen}_{kappa}_{name.replace(' ', '_')}.parquet"
                if os.path.exists(fn):
                    continue
                parts = []
                for g in GRID:
                    gg = torch.full((len(idx),), g)
                    if ctrl is None:
                        y, e = A.evaluate(dt, gg, kappa=kappa)
                    else:
                        y, e = A.evaluate(dt, gg, controller=ctrl, kappa=kappa, track_ref=getattr(ctrl, 'track_ref', False))
                    parts.append(pd.DataFrame(dict(base, method=name, scenario=scen, kappa=kappa, g=g,
                                                   y=y.numpy(), e=e.numpy())))
                pd.concat(parts).to_parquet(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
                print(f"{scen} kappa={kappa} {name} done {time.time() - t0:.0f}s", flush=True)
    files = sorted(os.listdir(cache))
    pd.concat([pd.read_parquet(f"{cache}/{f}") for f in files if f.endswith(".parquet")]).to_parquet(
        f"{ROOT}/results/grid_{tag}.parquet", index=False)
    print("saved", tag)
