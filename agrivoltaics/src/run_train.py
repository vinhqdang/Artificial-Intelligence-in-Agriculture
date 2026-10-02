"""Train the SALS controller.

Usage: python src/run_train.py NAME [split=0] [seed=0] [steps=300] [stress=1] [scen=baseline,+1.5C,+2C,+3C]
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A

ROOT = A.ROOT
torch.set_num_threads(4)
MIN_OPEN = 0.2   # seasons with open-field yield below this (crop failure) are excluded


def get_refs(D):
    fn = f"{ROOT}/results/refs.pt"
    if os.path.exists(fn):
        return torch.load(fn)
    refs = {s: A.references(D[s]) for s in D}
    torch.save(refs, fn)
    return refs


def train_mask(D, refs, test_site):
    out = {}
    for s, d in D.items():
        site = d["site"].numpy().astype(int)
        ok = (refs[s][0] >= MIN_OPEN).numpy()
        out[s] = torch.tensor((~test_site[site]) & np.isin(d["year"].numpy(), A.TRAIN_YEARS) & ok)
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    kw = dict(a.split("=") for a in sys.argv[2:])
    split = int(kw.get("split", 0))
    cfg = dict(rho=0.9, steps=int(kw.get("steps", 300)), batch=512, seed=int(kw.get("seed", 0)),
               lr=float(kw.get("lr", 1e-3)), warm_start=kw.get("warm", "1") == "1",
               stress_features=kw.get("stress", "1") == "1", scenarios=kw.get("scen", ",".join(A.SCEN)).split(","))
    os.makedirs(f"{ROOT}/results/models", exist_ok=True)
    D = A.load_all()
    refs = get_refs(D)
    tm = train_mask(D, refs, A.site_split(split))
    def gcr_for(rho):
        fn = f"{ROOT}/results/models/train_gcr_split{split}" + ("" if rho == 0.9 else f"_rho{rho}") + ".pt"
        if os.path.exists(fn):
            return torch.load(fn)
        g = A.calibrate_training_gcr(D, refs, tm, rho=rho); torch.save(g, fn); return g
    g_site = gcr_for(0.9)
    if kw.get("theta", "0") == "1":      # robust training: shade response drawn from the calibrated ensemble per season
        post = json.load(open(f"{ROOT}/data/" + kw.get("post", "shade_posterior.json")))
        cfg["theta"] = dict(c=torch.tensor(post["c"]), h=torch.tensor(post["h"]))
    if kw.get("ref", "0") == "1":        # adaptive training: controller sees measured biomass relative to a reference plot
        cfg["track_ref"] = True
    if "jhi" in kw:
        cfg["jitter"] = (-0.05, float(kw["jhi"]))
    extra = {}
    if kw.get("rho_cond", "0") == "1":
        levels = [0.8, 0.85, 0.9, 0.95]
        extra = dict(rho_levels=levels, g_site_by_rho={r: gcr_for(r) for r in levels})
    if kw.get("chance", "0") == "1":
        extra["chance"] = (0.8, 20.0)
    with open(f"{ROOT}/results/models/{name}.log", "w") as log:
        print(json.dumps(dict({k: v for k, v in cfg.items() if k != 'theta'}, split=split, theta='theta' in cfg)), file=log, flush=True)
        ctrl = A.train(D, refs, tm, log=log, g_site=g_site, **cfg, **extra)
    torch.save(dict(ctrl=ctrl.state_dict(), mask=ctrl.mask, cfg=dict({k: v for k, v in cfg.items() if k != "theta"}, split=split, rho_cond=bool(extra.get('rho_levels')), chance=bool(extra.get('chance')), theta=("theta" in cfg))),
               f"{ROOT}/results/models/{name}.pt")
    print("saved", name)
