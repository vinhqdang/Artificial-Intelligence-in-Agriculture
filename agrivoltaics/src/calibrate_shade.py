"""Calibrate the crop response to shade (RUE compensation, one parameter per crop) against the
meta-analytic yield-versus-shading curves of Laub et al. (2022) and validate it against
independent field experiments (rice, wheat, potato, maize, Dupraz et al. 2024).

Calibration uses static arrays on training-block sites and 2001-2014 seasons only.
Output: data/rue_comp.json and results/shade_calibration.csv, results/shade_validation.csv
"""
import os, sys, json
import numpy as np, pandas as pd, torch
os.environ["AV_VARIANT"] = "conservative"
sys.path.insert(0, os.path.dirname(__file__))
import sals as A, simulate as S, physics as P
from run_train import get_refs

ROOT = A.ROOT
CROPS = ["wheat", "rice", "maize", "soybean", "potato"]
# Laub et al. (2022): relative yield at 20% and 40% radiation shading (read from their fitted curves)
LAUB = {"wheat": (0.85, 0.62), "rice": (0.85, 0.62), "maize": (0.72, 0.45), "soybean": (0.75, 0.50), "potato": (0.85, 0.60)}
GRID = [0.1, 0.2, 0.3, 0.4, 0.5]
torch.manual_seed(0)


def subset(n=800):
    D = A.load_all(); d = D["baseline"]; refs = get_refs(D)
    test = A.site_split(0)
    m = (~test[d["site"].numpy().astype(int)]) & (d["year"].numpy() <= 2014) & (refs["baseline"][0] >= 0.2).numpy()
    idx = torch.where(torch.tensor(m))[0]
    rng = np.random.default_rng(0); keep = []
    for c in range(5):
        ii = idx[d["crop"][idx] == c].numpy()
        keep += list(rng.choice(ii, min(n, len(ii)), replace=False))
    return S._sel(d, torch.tensor(sorted(keep)))


def curve(dt, comp, hi_scale=1.0):
    P.RUE_COMP.update({c: comp[c] for c in CROPS})
    saved = dict(P.HI_SHADE)
    for c in CROPS: P.HI_SHADE[c] = saved[c] * hi_scale
    rows = []
    n = len(dt["crop"]); irr = torch.zeros(n)
    with torch.no_grad():
        y0, _ = S.rollout(dt, torch.full((n,), 0.3), panels=False, irrigated=irr)
        for g in GRID:
            y, e, rec = S.rollout(dt, torch.full((n,), g), irrigated=irr, record=True)
            act = dt["active"]; sh = 1 - (rec["shade"] * act).sum(1) / act.sum(1)
            for ci, c in enumerate(CROPS):
                mm = (dt["crop"] == ci) & (y0 > 0.2)
                rows.append(dict(crop=c, g=g, shading=float(sh[mm].mean()), rel=float((y[mm] / y0[mm]).mean())))
    P.HI_SHADE.update(saved)
    return pd.DataFrame(rows)


def interp(df, c, s):
    x = df[df.crop == c].sort_values("shading")
    return float(np.interp(s, x.shading, x.rel))


FIELD = {"wheat": (0.30, 0.92), "rice": (0.25, 0.842), "maize": (0.35, 0.833), "potato": (0.30, 0.964),
         "soybean": (0.40, 0.50)}   # soybean: Laub legumes (no field trial supplied)


def fit_field(dt):
    saved = P.HI_SHADE["maize"]; P.HI_SHADE["maize"] = 0.0
    cand = np.round(np.arange(0.0, 2.01, 0.1), 2); out = {}
    allr = pd.concat([curve(dt, {c: v for c in CROPS}).assign(c=v) for v in cand])
    for c in CROPS:
        err = [(interp(allr[allr.c == v], c, FIELD[c][0]) - FIELD[c][1]) ** 2 for v in cand]
        out[c] = float(cand[int(np.argmin(err))])
    P.HI_SHADE["maize"] = saved
    json.dump(out, open(f"{ROOT}/data/rue_comp_field.json", "w"), indent=1); print("field", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "field":
        fit_field(subset()); sys.exit()
    dt = subset()
    cand = np.round(np.arange(0.0, 1.61, 0.1), 2)
    res = {}
    allrows = []
    for cval in cand:
        df = curve(dt, {c: cval for c in CROPS}); df["c"] = cval; allrows.append(df)
    allr = pd.concat(allrows)
    best = {}
    for c in CROPS:
        err = []
        for cval in cand:
            df = allr[allr.c == cval]
            err.append((interp(df, c, 0.2) - LAUB[c][0]) ** 2 + (interp(df, c, 0.4) - LAUB[c][1]) ** 2)
        best[c] = float(cand[int(np.argmin(err))])
        print(c, "best c", best[c], "rmse", np.sqrt(min(err) / 2))
    json.dump(best, open(f"{ROOT}/data/rue_comp.json", "w"), indent=1)
    cal = curve(dt, best); cal.to_csv(f"{ROOT}/results/shade_calibration.csv", index=False)
    cons = curve(dt, {c: 0.0 for c in CROPS})
    # independent validation targets
    val = []
    def add(name, crop, shading, obs, lo=None, hi=None, hi_scale=1.0):
        d_ = cal if hi_scale == 1.0 else curve(dt, best, hi_scale)
        val.append(dict(study=name, crop=crop, shading=shading, observed=obs, obs_lo=lo, obs_hi=hi,
                        calibrated=interp(d_, crop, shading), conservative=interp(cons, crop, shading)))
    add("Gonocruz 2021 (rice, S=0.25)", "rice", 0.25, 1 - 0.63 * 0.25)
    add("Weselek 2021 wheat (S~0.30)", "wheat", 0.30, 1 - 0.187, 1 - 0.187, 1.027)
    add("Weselek 2021 potato (S~0.30)", "potato", 0.30, 1 - 0.182, 1 - 0.182, 1.11)
    for g in [0.2, 0.3, 0.4, 0.5]:
        for c in CROPS:
            pass
    for name, s, o in [("Ramos-Fuentes 2023 maize DAV FI (S~0.35)", 0.35, np.mean([0.84, 0.71, 0.95])),
                       ("Ramos-Fuentes 2023 maize AVhalf FI (S~0.30)", 0.30, 0.91),
                       ("Ramos-Fuentes 2023 maize AVfull FI (S~0.50)", 0.50, 0.70)]:
        add(name, "maize", s, o)
    V = pd.DataFrame(val); V.to_csv(f"{ROOT}/results/shade_validation.csv", index=False)
    print(V.round(3).to_string())
    print(cal.round(3).to_string())
