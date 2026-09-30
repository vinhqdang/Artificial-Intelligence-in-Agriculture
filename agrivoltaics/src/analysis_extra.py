"""Analysis of the extra experiments: floor-conditioned / chance-trained controllers and forecast noise."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import analysis as A

ROOT, RES = A.ROOT, A.RES
LEVELS = [0.8, 0.85, 0.9, 0.95]


def ok_keys():
    g = pd.read_parquet(f"{RES}/grid_split0.parquet")
    op = g.drop_duplicates(["site", "year", "scenario"]).pivot_table(index=["site", "year"], columns="scenario", values="y_open")
    ok = (op >= A.MIN_OPEN).all(axis=1)
    return ok[ok].reset_index()[["site", "year"]]


def prep(g, keys):
    g = g.merge(keys, on=["site", "year"])
    g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    return g


if __name__ == "__main__":
    keys = ok_keys()
    base = pd.read_parquet(f"{RES}/grid_split0.parquet")
    base = base[base.method.isin(A.RULES + ["sals_s0_seed0"]) & (base.kappa == 1.0)].copy()
    base["method"] = base.method.replace({"sals_s0_seed0": "SALS (fixed-rho net)"})
    base = prep(base, keys)
    ex = prep(pd.read_parquet(f"{RES}/extra_rho.parquet"), keys)
    rows = []
    for rho in LEVELS:
        cond = ex[(ex.method == "SALS rho-conditioned") & np.isclose(ex.rho, rho)]
        allg = pd.concat([base, cond.drop(columns="rho")])
        c = A.calibrate(allg, "mean", rho=rho)
        for scen in ["baseline", "+3C"]:
            for m in ["AV static", "Phenology rule", "Stress rule", "SALS (fixed-rho net)", "SALS rho-conditioned"]:
                x = c[(c.scenario == scen) & (c.method == m)]
                s = A.summarize(x)
                s["comply"] = (x.r >= rho).mean() * 100
                rows.append(dict(rho=rho, scenario=scen, method=m, **s))
    t = pd.DataFrame(rows); t.to_csv(f"{RES}/rho_study.csv", index=False)
    pd.set_option("display.width", 200)
    print(t[t.scenario == "baseline"].pivot(index="method", columns="rho", values="LER").round(3))
    print(t[t.scenario == "+3C"].pivot(index="method", columns="rho", values="LER").round(3))
    print(t[t.scenario == "baseline"].pivot(index="method", columns="rho", values="r").round(3))
    # chance-trained network at rho 0.9
    ch = ex[(ex.method == "SALS chance-trained")].drop(columns="rho")
    allg = pd.concat([base, ch])
    rows = []
    for crit in ["mean", "chance"]:
        c = A.calibrate(allg, crit, rho=0.9)
        for scen in ["baseline", "+3C"]:
            for m in ["Phenology rule", "SALS (fixed-rho net)", "SALS chance-trained"]:
                x = c[(c.scenario == scen) & (c.method == m)]
                rows.append(dict(design=crit, scenario=scen, method=m, **A.summarize(x)))
    ct = pd.DataFrame(rows); ct.to_csv(f"{RES}/chance_trained.csv", index=False)
    print(ct[["design", "scenario", "method", "GCR", "E", "r", "comply", "p10", "below80", "LER"]].round(3).to_string())
    if os.path.exists(f"{RES}/extra_noise.parquet"):
        nz = prep(pd.read_parquet(f"{RES}/extra_noise.parquet"), keys)
        n = nz.groupby(["scenario", "method", "g", "noise"]).apply(lambda d: pd.Series(dict(
            r=d.r.mean(), e=d.erel.mean(), LER=(d.r + d.erel).mean()))).reset_index()
        n.to_csv(f"{RES}/noise_study.csv", index=False)
        print(n[(n.scenario == "baseline")].pivot_table(index=["g", "method"], columns="noise", values="LER").round(3))
        print(n[(n.scenario == "baseline")].pivot_table(index=["g", "method"], columns="noise", values="r").round(3))
