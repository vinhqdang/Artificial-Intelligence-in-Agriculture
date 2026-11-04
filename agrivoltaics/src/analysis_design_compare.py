"""Pair-level comparison of the learned design network with the fixed-margin frontier (in-ensemble and outside the ensemble), with a cluster bootstrap over
crop-site pairs of the LER difference at matched compliance. A pair held out in several splits is averaged. Output: results/learned_design_pairs.csv,
learned_design_summary.csv (pair-level, per lambda), learned_design_margin_frontier.csv (pair-level, per margin), learned_design_compare.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES
rng = np.random.default_rng(0)
L = pd.read_parquet(f"{RES}/learned_design_units.parquet"); C = pd.read_parquet(f"{RES}/conformal_units.parquet")
L = L.assign(kind=L.kind.map({"in": "in-ensemble", "out": "out-of-ensemble"}))
LP = L.groupby(["lam", "kind", "site"]).agg(ok=("ok", "mean"), ler=("ler", "mean")).reset_index()          # mean over seeds, splits, members/responses
MP = C.groupby(["kind", "d", "site"]).agg(ok=("ok", "mean"), ler=("ler", "mean")).reset_index()
LP.to_csv(f"{RES}/learned_design_pairs.csv", index=False)
S = LP.groupby(["lam", "kind"]).agg(pairs=("site", "nunique"), compliance=("ok", "mean"), ler=("ler", "mean")).reset_index(); S["compliance"] *= 100; S.to_csv(f"{RES}/learned_design_summary.csv", index=False)
F = MP.groupby(["kind", "d"]).agg(pairs=("site", "nunique"), compliance=("ok", "mean"), ler=("ler", "mean")).reset_index(); F["compliance"] *= 100; F.to_csv(f"{RES}/learned_design_margin_frontier.csv", index=False)


def frontier_at(mp, c):
    g = mp.groupby("d").agg(ok=("ok", "mean"), ler=("ler", "mean")).sort_values("ok"); o = g.ok.values; l = g.ler.values
    if c < o.min() or c > o.max(): return np.nan
    return float(np.interp(c, o, l))


rows = []
for kind in ("in-ensemble", "out-of-ensemble"):
    mp = MP[MP.kind == kind]; sites = np.array(sorted(mp.site.unique()))
    for lam in sorted(LP.lam.unique()):
        lp = LP[(LP.kind == kind) & (LP.lam == lam)].set_index("site")
        def stat(ids):
            a = lp.loc[ids]; m = mp[mp.site.isin(ids)]
            # bootstrap resample with replacement: weight pairs by multiplicity
            w = pd.Series(ids).value_counts(); mm = mp.merge(w.rename("w"), left_on="site", right_index=True)
            mm = mm.assign(okw=mm.ok * mm.w, lerw=mm.ler * mm.w); g = mm.groupby("d").agg(o=("okw", "sum"), l=("lerw", "sum"), w=("w", "sum")); g["o"] /= g.w; g["l"] /= g.w
            g = g.sort_values("o"); c = a.ok.mean()
            ref = np.interp(c, g.o.values, g.l.values) if g.o.min() <= c <= g.o.max() else np.nan
            return c * 100, a.ler.mean(), a.ler.mean() - ref
        base = stat(sites); bs = [stat(rng.choice(sites, len(sites))) for _ in range(500)]
        d = np.array([b[2] for b in bs]); d = d[~np.isnan(d)]
        rows.append(dict(kind=kind, lam=lam, compliance=base[0], ler=base[1], ler_minus_margin=base[2], lo=np.quantile(d, .025) if len(d) else np.nan, hi=np.quantile(d, .975) if len(d) else np.nan))
R = pd.DataFrame(rows); R.to_csv(f"{RES}/learned_design_compare.csv", index=False); pd.set_option("display.width", 200); print(R.round(3).to_string())
