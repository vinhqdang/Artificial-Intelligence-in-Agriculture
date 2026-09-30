"""Train SALS and its ablations. Usage: python src/run_train.py NAME [options as key=value]"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A

ROOT = A.ROOT
torch.set_num_threads(4)


def get_refs(D):
    fn = f"{ROOT}/results/refs.pt"
    if os.path.exists(fn):
        return torch.load(fn)
    refs = {s: A.references(D[s]) for s in D}
    torch.save(refs, fn)
    return refs


def train_mask(D, test_site):
    out = {}
    for s, d in D.items():
        site = d["site"].numpy().astype(int)
        out[s] = torch.tensor((~test_site[site]) & np.isin(d["year"].numpy(), A.TRAIN_YEARS))
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    kw = dict(a.split("=") for a in sys.argv[2:])
    cfg = dict(rho=float(kw.get("rho", 0.9)), steps=int(kw.get("steps", 700)), batch=int(kw.get("batch", 512)),
               use_design=kw.get("design", "1") == "1", fixed_g=float(kw["fixed_g"]) if "fixed_g" in kw else None,
               stress_features=kw.get("stress", "1") == "1", seed=int(kw.get("seed", 0)),
               scenarios=kw.get("scen", ",".join(A.SCEN)).split(","))
    os.makedirs(f"{ROOT}/results/models", exist_ok=True)
    D = A.load_all()
    refs = get_refs(D)
    test_site = A.site_split()
    tm = train_mask(D, test_site)
    with open(f"{ROOT}/results/models/{name}.log", "w") as log:
        print(json.dumps(cfg), file=log, flush=True)
        def save(step, ctrl, des):
            torch.save(dict(ctrl=ctrl.state_dict(), des=des.state_dict(), mask=ctrl.mask, cfg=cfg),
                       f"{ROOT}/results/models/{name}_step{step}.pt")
        ctrl, des = A.train(D, refs, tm, log=log, checkpoint=({200}, save), **cfg)
    torch.save(dict(ctrl=ctrl.state_dict(), des=des.state_dict(), mask=ctrl.mask, cfg=cfg),
               f"{ROOT}/results/models/{name}.pt")
    print("saved", name)
