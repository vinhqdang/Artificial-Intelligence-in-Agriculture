"""Design under shade-response uncertainty, final version (phenology rule, three splits, distinct crop-site pairs).
(a) leave-one-out: truth = held-out member k; 'mean response' and 'margin' designs use the mean of the other four members
    (grids e?x{split}_{k}); quantile designs use the other four members' histories (q=0.75: three of four; 'all others': four of four).
(b) out of ensemble: truth = one of four fixed responses; designs use the whole ensemble (five members, their mean).
Metrics per pair: mean retention >= 0.9, share of seasons >= 0.9, expected shortfall max(0, 0.9 - mean retention), LER.
Output: results/design_final_{loo,ooe}_E{2,3}.csv and *_boot.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO, HIST
from analysis_robust import apply
MEM = [0, 3, 5, 8, 10]; rng = np.random.default_rng(0)


def load(tag):
    g = pd.read_parquet(f"{RES}/grid_{tag}.parquet"); g = g[(g.method == "Phenology rule") & (g.y_open >= 0.2)].copy()
    g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv; return g


def feas(g, rho=RHO):
    h = g[g.year <= HIST].groupby(["method", "site", "g"]).r.mean().reset_index(); h["ok"] = (h.r >= rho).astype(float); return h[["method", "site", "g", "ok"]]


def pick(fl, q):
    f = pd.concat(fl).groupby(["method", "site", "g"]).ok.mean().reset_index(); return f[f.ok >= q - 1e-9].groupby(["method", "site"]).g.max().rename("g_sel").reset_index()


def pairs(x):
    s = x.groupby(["method", "site"]).agg(r=("r", "mean"), ler=("ler", "mean"), season_ok=("r", lambda v: (v >= RHO).mean())).reset_index()
    s["ok"] = (s.r >= RHO).astype(float); s["short"] = np.maximum(0, RHO - s.r); return s


def summarize(rows, name):
    R = pd.concat(rows)
    d = R.groupby(["design", "site"])[["r", "ler", "season_ok", "ok", "short"]].mean().reset_index()
    S = d.groupby("design").agg(pairs=("site", "nunique"), retention=("r", "mean"), seasons_ok=("season_ok", "mean"), pairs_ok=("ok", "mean"), shortfall=("short", "mean"), LER=("ler", "mean")).reset_index()
    S[["seasons_ok", "pairs_ok"]] *= 100; S.to_csv(f"{RES}/{name}.csv", index=False)
    P = {k: d[d.design == k].set_index("site") for k in d.design.unique()}; B = []
    for a, b in [("aware q=0.75", "mean response"), ("margin 0.92", "mean response"), ("margin 0.92", "aware q=0.75"), ("oracle", "margin 0.92")]:
        if a not in P or b not in P: continue
        ids = P[a].index.intersection(P[b].index); pa, pb = P[a].loc[ids], P[b].loc[ids]; dd = []
        for _ in range(2000):
            i = rng.integers(0, len(ids), len(ids)); dd.append([(pa.ok.values[i] - pb.ok.values[i]).mean() * 100, (pa.ler.values[i] - pb.ler.values[i]).mean(), (pa.short.values[i] - pb.short.values[i]).mean()])
        dd = np.array(dd); B.append(dict(contrast=f"{a} minus {b}", pp=dd[:, 0].mean(), pp_lo=np.quantile(dd[:, 0], .025), pp_hi=np.quantile(dd[:, 0], .975),
                                         ler=dd[:, 1].mean(), ler_lo=np.quantile(dd[:, 1], .025), ler_hi=np.quantile(dd[:, 1], .975), short=dd[:, 2].mean(), pairs=len(ids)))
    pd.DataFrame(B).to_csv(f"{RES}/{name}_boot.csv", index=False)
    pd.set_option("display.width", 200); print(name); print(S.round(3).to_string()); print(pd.DataFrame(B).round(3).to_string())


for ens, tg_s, tg_x in [("E2", "e2s", "e2x"), ("E3", "e3s", "e3x")]:
    rows = []
    for sp in (0, 1, 2):
        for k in MEM:
            truth = load(f"{tg_s}{sp}_{k}"); mx = load(f"{tg_x}{sp}_{k}")
            others = [feas(load(f"{tg_s}{sp}_{j}")) for j in MEM if j != k]
            designs = [("oracle", pick([feas(truth)], 1.0)), ("mean response", pick([feas(mx)], 1.0)), ("margin 0.92", pick([feas(mx, 0.92)], 1.0)),
                       ("margin 0.94", pick([feas(mx, 0.94)], 1.0)), ("margin 0.96", pick([feas(mx, 0.96)], 1.0)),
                       ("aware q=0.5", pick(others, 0.5)), ("aware q=0.75", pick(others, 0.75)), ("aware all others", pick(others, 1.0))]
            for name, sel in designs:
                s = pairs(apply(sel, truth)); s["design"] = name; rows.append(s)
    summarize(rows, f"design_final_loo_{ens}")
    rows = []
    for sp in (0, 1, 2):
        mem = [feas(load(f"{tg_s}{sp}_{k}")) for k in MEM]; mean_g = load(f"{tg_s}{sp}_mean")
        for T in ["conservative", "calibrated", "field", "harsh"]:
            truth = load(f"nt{sp}_{T}")
            for name, sel in [("oracle", pick([feas(truth)], 1.0)), ("mean response", pick([feas(mean_g)], 1.0)), ("margin 0.92", pick([feas(mean_g, 0.92)], 1.0)),
                              ("margin 0.94", pick([feas(mean_g, 0.94)], 1.0)), ("margin 0.96", pick([feas(mean_g, 0.96)], 1.0)),
                              ("aware q=0.75", pick(mem, 0.75)), ("aware all members", pick(mem, 1.0))]:
                s = pairs(apply(sel, truth)); s["design"] = name; s["truth"] = T; rows.append(s)
    R = pd.concat(rows)
    for T, x in R.groupby("truth"):
        summarize([x.drop(columns="truth")], f"design_final_ooe_{ens}_{T}")
