"""Global assessment with the trained SALS model: all 400 sites, test years
(2015-2019), four climates; sensitivity to canopy cooling; example trajectories."""
import os, sys, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A
import simulate as S
from evaluate import load_model
from run_train import get_refs

ROOT = A.ROOT


def main(model="sals_main"):
    D = A.load_all(); refs = get_refs(D)
    ctrl, des, cfg = load_model(model)
    rule_g = {int(k): v for k, v in json.load(open(f"{ROOT}/results/rule_g_main.json")).items()}
    rows = []
    for scen in A.SCEN:
        d = D[scen]
        idx = torch.where(torch.tensor(np.isin(d["year"].numpy(), A.TEST_YEARS)))[0]
        dt = S._sel(d, idx)
        with torch.no_grad():
            g = des(A.site_features(dt))
        y, e = A.evaluate(dt, g, controller=A.wrap(ctrl))
        gr = torch.tensor([rule_g[int(c)] for c in dt["crop"]], dtype=torch.float32)
        ys, es = A.evaluate(dt, gr)
        rows.append(pd.DataFrame(dict(scenario=scen, site=dt["site"].numpy().astype(int), year=dt["year"].numpy().astype(int),
                                      crop=dt["crop"].numpy().astype(int), g=g.numpy(), y=y.numpy(), e=e.numpy(),
                                      y_static=ys.numpy(), e_static=es.numpy(), g_static=gr.numpy(),
                                      y_open=refs[scen][0][idx].numpy(), e_pv=refs[scen][1][idx].numpy())))
        print(scen, "global done", flush=True)
    pd.concat(rows).to_csv(f"{ROOT}/results/global_{model}.csv", index=False)
    # sensitivity to the canopy-cooling coefficient (baseline and +3C, test sites and years)
    test_site = A.site_split()
    sens = []
    for scen in ["baseline", "+3C"]:
        d = D[scen]
        site = d["site"].numpy().astype(int)
        idx = torch.where(torch.tensor(test_site[site] & np.isin(d["year"].numpy(), A.TEST_YEARS)))[0]
        dt = S._sel(d, idx)
        with torch.no_grad():
            g = des(A.site_features(dt))
        gr = torch.tensor([rule_g[int(c)] for c in dt["crop"]], dtype=torch.float32)
        for kappa in [0.0, 1.0, 2.0, 3.0]:
            # the open-field reference does not depend on kappa
            y, e = A.evaluate(dt, g, controller=A.wrap(ctrl), kappa=kappa)
            ys, es = A.evaluate(dt, gr, kappa=kappa)
            yo, ep = refs[scen][0][idx], refs[scen][1][idx]
            for name, yy, ee in [("SALS (ours)", y, e), ("AV static (rule)", ys, es)]:
                r = (yy / yo.clamp(min=0.05)).numpy(); ok = yo.numpy() >= 0.2
                sens.append(dict(scenario=scen, kappa=kappa, method=name, ret=r[ok].mean(), comply=(r[ok] >= 0.9).mean(),
                                 e=(ee / ep).numpy().mean(), E=ee.numpy().mean()))
            print(scen, kappa, sens[-2], flush=True)
    pd.DataFrame(sens).to_csv(f"{ROOT}/results/sensitivity_kappa.csv", index=False)
    # example daily trajectories (one hot, dry rainfed maize and one irrigated rice site)
    for scen in ["baseline", "+3C"]:
        d = D[scen]
        idx = torch.where(torch.tensor(np.isin(d["year"].numpy(), [2017])))[0]
        dt = S._sel(d, idx)
        with torch.no_grad():
            g = des(A.site_features(dt))
            y, e, rec = S.rollout(dt, g, controller=A.wrap(ctrl), record=True)
        np.savez_compressed(f"{ROOT}/results/traj_{scen}.npz", u=rec["u"].numpy(), shade=rec["shade"].numpy(),
                            fheat=rec["fheat"].numpy(), fwater=rec["fwater"].numpy(), tmax=dt["tmax"].numpy(),
                            active=dt["active"].numpy(), site=dt["site"].numpy(), crop=dt["crop"].numpy(), g=g.numpy())


if __name__ == "__main__":
    main(*sys.argv[1:])
