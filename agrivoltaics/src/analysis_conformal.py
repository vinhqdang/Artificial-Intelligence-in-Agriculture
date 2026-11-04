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
U = pd.concat(rows, ignore_index=True); U["crop"] = U.crop.map(CROP); U["member"] = U.member.astype(str); U.to_parquet(f"{RES}/conformal_units.parquet")
I = U[U.kind == "in-ensemble"]; O = U[U.kind == "out-of-ensemble"]
split_sites = {sp: set(U[U.split == sp].site) for sp in (0, 1, 2)}


def pair_table(X):
    """pair-level outcome: share of units at the floor, mean LER, mean shortfall (mean over members/responses and over splits)"""
    return X.groupby(["site", "d"]).agg(ok=("ok", "mean"), ler=("ler", "mean"), short=("short", "mean"), crop=("crop", "first")).reset_index()


def calibrate(cal, alpha):
    P = pair_table(cal); n = P.site.nunique(); r = 1 - P.groupby("d").ok.mean()
    for d in D:
        if n / (n + 1) * r[d] + 1 / (n + 1) <= alpha + 1e-12: return d, n
    return D[-1], n


out = []; marg = []
for alpha in (0.10, 0.15, 0.20):
    for te in (0, 1, 2):
        cal = I[(I.split != te) & (~I.site.isin(split_sites[te]))]; dg, n = calibrate(cal, alpha); marg.append(dict(alpha=alpha, test_split=te, d=dg, calibration_pairs=n))
        for kind, E in (("in-ensemble", I[I.split == te]), ("out-of-ensemble", O[O.split == te])):
            z = pair_table(E[np.isclose(E.d, dg)])
            out.append(dict(alpha=alpha, test_split=te, rule="conformal margin", kind=kind, pairs=len(z), compliance=z.ok.mean() * 100, ler=z.ler.mean(), short=z.short.mean(),
                            **{f"ok_{c}": z[z.crop == c].ok.mean() * 100 for c in CROP.values()}))
for name, dfix in (("mean response (d = 0.90)", 0.90), ("fixed margin 0.92", 0.92), ("fixed margin 0.94", 0.94), ("fixed margin 0.96", 0.96)):
    for te in (0, 1, 2):
        for kind, E in (("in-ensemble", I[I.split == te]), ("out-of-ensemble", O[O.split == te])):
            z = pair_table(E[np.isclose(E.d, dfix)])
            out.append(dict(alpha=np.nan, test_split=te, rule=name, kind=kind, pairs=len(z), compliance=z.ok.mean() * 100, ler=z.ler.mean(), short=z.short.mean(),
                            **{f"ok_{c}": z[z.crop == c].ok.mean() * 100 for c in CROP.values()}))
R = pd.DataFrame(out); R.to_csv(f"{RES}/conformal_by_split.csv", index=False); pd.DataFrame(marg).to_csv(f"{RES}/conformal_margins.csv", index=False)
# pooled over distinct pairs: the margins chosen in each test split are applied to that split's pairs; a pair held out in several splits is averaged
pool = []
for alpha in (0.10, 0.15, 0.20):
    for kind, X in (("in-ensemble", I), ("out-of-ensemble", O)):
        rows = []
        for te in (0, 1, 2):
            d = [m["d"] for m in marg if m["alpha"] == alpha and m["test_split"] == te][0]; E = X[(X.split == te) & np.isclose(X.d, d)]; rows.append(pair_table(E))
        Z = pd.concat(rows).groupby("site").agg(ok=("ok", "mean"), ler=("ler", "mean"), short=("short", "mean"), crop=("crop", "first")).reset_index(); Z["alpha"], Z["kind"], Z["rule"] = alpha, kind, "conformal margin"; pool.append(Z)
for name, dfix in (("mean response (d = 0.90)", 0.90), ("fixed margin 0.92", 0.92), ("fixed margin 0.94", 0.94), ("fixed margin 0.96", 0.96)):
    for kind, X in (("in-ensemble", I), ("out-of-ensemble", O)):
        Z = pair_table(X[np.isclose(X.d, dfix)]).groupby("site").agg(ok=("ok", "mean"), ler=("ler", "mean"), short=("short", "mean"), crop=("crop", "first")).reset_index(); Z["alpha"], Z["kind"], Z["rule"] = np.nan, kind, name; pool.append(Z)
Pp = pd.concat(pool); Pp.to_csv(f"{RES}/conformal_pairs.csv", index=False)
S = Pp.groupby(["alpha", "rule", "kind"], dropna=False).agg(pairs=("site", "nunique"), compliance=("ok", "mean"), ler=("ler", "mean"), short=("short", "mean"), **{f"ok_{c}": ("ok", lambda v, c=c: np.nan) for c in CROP.values()}).reset_index()
for c in CROP.values():
    sub = Pp[Pp.crop == c].groupby(["alpha", "rule", "kind"], dropna=False).ok.mean().mul(100)
    for k, v in sub.items():
        m = np.ones(len(S), bool)
        for col, val in zip(("alpha", "rule", "kind"), k): m &= (S[col].isna() if (isinstance(val, float) and np.isnan(val)) else (S[col] == val)).values
        S.loc[m, f"ok_{c}"] = v
S["compliance"] *= 100; S.to_csv(f"{RES}/conformal_summary.csv", index=False)
pd.set_option("display.width", 220); print(S.round(3).to_string()); print(pd.DataFrame(marg).round(3).to_string())
print(R[(R.rule == "conformal margin") & (R.kind == "in-ensemble")][["alpha", "test_split", "pairs", "compliance"]].round(1).to_string())
