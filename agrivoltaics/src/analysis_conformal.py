"""Conformal risk control of the design margin (ensemble E2, phenology rule, three splits).
Units: (crop-site pair, held-out member k). For a margin d (floor used in the design under the leave-one-out mean response), the loss is 1 if the mean
retention of the pair under the held-out member is below 0.9 and 0 otherwise. A margin is calibrated on two splits and evaluated on the third
(leave-one-split-out), globally and per crop (Mondrian). Conformal risk control (Angelopoulos et al.): choose the smallest d with
n/(n+1) * risk_hat(d) + 1/(n+1) <= alpha. Evaluation also under four fixed responses outside the ensemble.
Output: results/conformal_units.parquet, conformal_summary.csv, conformal_margins.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO, HIST
from analysis_robust import apply
MEM = [0, 3, 5, 8, 10]; D = np.round(np.arange(0.90, 0.9751, 0.005), 3); CROP = {0: "wheat", 1: "rice", 2: "maize", 3: "soybean", 4: "potato"}


def load(tag):
    g = pd.read_parquet(f"{RES}/grid_{tag}.parquet"); g = g[(g.method == "Phenology rule") & (g.y_open >= 0.2)].copy()
    g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv; return g


def select(belief, floor):
    h = belief[belief.year <= HIST].groupby(["method", "site", "g"]).r.mean().reset_index()
    return h[h.r >= floor].groupby(["method", "site"]).g.max().rename("g_sel").reset_index()


def outcome(sel, truth):
    x = apply(sel, truth); s = x.groupby(["site"]).agg(r=("r", "mean"), ler=("ler", "mean"), crop=("crop", "first")).reset_index()
    s["ok"] = (s.r >= RHO).astype(float); s["short"] = np.maximum(0, RHO - s.r); return s


rows = []
for sp in (0, 1, 2):
    for k in MEM:
        truth, bel = load(f"e2s{sp}_{k}"), load(f"e2x{sp}_{k}")
        for d in D:
            s = outcome(select(bel, d), truth); s["split"], s["member"], s["d"], s["kind"] = sp, k, d, "in-ensemble"; rows.append(s)
    mean_g = load(f"e2s{sp}_mean")
    for T in ["conservative", "calibrated", "field", "harsh"]:
        truth = load(f"nt{sp}_{T}")
        for d in D:
            s = outcome(select(mean_g, d), truth); s["split"], s["member"], s["d"], s["kind"] = sp, T, d, "out-of-ensemble"; rows.append(s)
U = pd.concat(rows, ignore_index=True); U["crop"] = U.crop.map(CROP); U.to_parquet(f"{RES}/conformal_units.parquet")
I = U[U.kind == "in-ensemble"]; O = U[U.kind == "out-of-ensemble"]


def calibrate(cal, alpha):
    """smallest margin d whose conformal-adjusted risk is <= alpha (None: infeasible)"""
    n = cal.groupby("d").size(); r = 1 - cal.groupby("d").ok.mean()
    for d in D:
        if n[d] / (n[d] + 1) * r[d] + 1 / (n[d] + 1) <= alpha + 1e-12: return d
    return D[-1]


out = []; marg = []
for alpha in (0.10, 0.15, 0.20):
    for test_sp in (0, 1, 2):
        cal = I[I.split != test_sp]; ev = I[I.split == test_sp]; evo = O[O.split == test_sp]
        dg = calibrate(cal, alpha); marg.append(dict(alpha=alpha, test_split=test_sp, scope="global", crop="all", d=dg))
        dc = {c: calibrate(cal[cal.crop == c], alpha) for c in CROP.values()}
        for c, v in dc.items(): marg.append(dict(alpha=alpha, test_split=test_sp, scope="per crop", crop=c, d=v))
        for name, pick in (("global conformal", lambda c: dg), ("per-crop conformal", lambda c: dc[c])):
            for kind, E in (("in-ensemble", ev), ("out-of-ensemble", evo)):
                z = E[[np.isclose(d_, pick(c_)) for d_, c_ in zip(E.d, E.crop)]]
                out.append(dict(alpha=alpha, test_split=test_sp, rule=name, kind=kind, pairs=z.site.nunique(), compliance=z.ok.mean() * 100, ler=z.ler.mean(), short=z.short.mean(),
                                **{f"ok_{c}": z[z.crop == c].ok.mean() * 100 for c in CROP.values()}))
for name, dfix in (("mean response (d = 0.90)", 0.90), ("fixed margin 0.92", 0.92), ("fixed margin 0.94", 0.94), ("fixed margin 0.96", 0.96)):
    for test_sp in (0, 1, 2):
        for kind, E in (("in-ensemble", I[I.split == test_sp]), ("out-of-ensemble", O[O.split == test_sp])):
            z = E[np.isclose(E.d, dfix)]
            out.append(dict(alpha=np.nan, test_split=test_sp, rule=name, kind=kind, pairs=z.site.nunique(), compliance=z.ok.mean() * 100, ler=z.ler.mean(), short=z.short.mean(),
                            **{f"ok_{c}": z[z.crop == c].ok.mean() * 100 for c in CROP.values()}))
R = pd.DataFrame(out); R.to_csv(f"{RES}/conformal_by_split.csv", index=False); pd.DataFrame(marg).to_csv(f"{RES}/conformal_margins.csv", index=False)
S = R.groupby(["alpha", "rule", "kind"], dropna=False).agg(compliance=("compliance", "mean"), ler=("ler", "mean"), short=("short", "mean"), **{f"ok_{c}": (f"ok_{c}", "mean") for c in CROP.values()}).reset_index()
S.to_csv(f"{RES}/conformal_summary.csv", index=False)
pd.set_option("display.width", 220); print(S.round(3).to_string()); print(pd.DataFrame(marg).pivot_table(index=["alpha", "scope", "crop"], columns="test_split", values="d").round(3).to_string())
