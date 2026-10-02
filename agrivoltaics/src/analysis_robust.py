"""Rules and controllers under five true shade responses, with the array density designed either under the
true response (matched) or under the calibrated-ensemble mean (belief). Output: results/robust_eval.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO, HIST, TEST, summarize

NAMES = {"sals_s0_seed0": "SALS (trained on proportional response)", "rob_s0": "SALS-robust (ensemble-trained)",
         "adp_s0": "SALS-adaptive (ensemble, biomass feedback)",
         "rob2_s0": "SALS-robust, E2", "rob3_s0": "SALS-robust, E3", "adp3_s0": "SALS-adaptive, E3"}
TRUTHS = ["conservative", "calibrated", "field", "harsh"]


def load(v):
    g = pd.read_parquet(f"{RES}/grid_rb_{v}.parquet")
    g["method"] = g.method.replace(NAMES)
    g = g[g.y_open >= 0.2].copy()
    g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    return g


def design(h, methods):
    h = h[h.year <= HIST]
    m = h.groupby(["method", "site", "g"]).r.mean().reset_index()
    ok = m[m.r >= RHO].groupby(["method", "site"]).g.max().rename("g_sel").reset_index()
    return ok


def apply(sel, t):
    t = t[t.year >= TEST]
    keys = t[["method", "site"]].drop_duplicates().merge(sel, how="left", on=["method", "site"])
    x = t.merge(keys, on=["method", "site"])
    chosen = x[x.g == x.g_sel]
    none = x[x.g_sel.isna()].drop_duplicates(["method", "site", "year", "crop", "irr"]).copy()
    none["g"], none["y"], none["e"], none["r"], none["erel"] = 0.0, none.y_open, 0.0, 1.0, 0.0
    out = pd.concat([chosen, none], ignore_index=True)
    out["ler"] = out.r + out.erel
    return out


if __name__ == "__main__":
    belief = load("posterior")
    rows = []
    for T in TRUTHS:
        truth = load(T)
        for how, hist in [("matched", truth), ("belief", belief)]:
            sel = design(hist, None)
            for m, x in apply(sel, truth).groupby("method"):
                s = summarize(x); s["truth"] = T; s["design"] = how; s["method"] = m; rows.append(s)
    out = pd.DataFrame(rows)[["truth", "design", "method", "sites", "GCR", "erel", "r", "comply", "site_comply", "p10", "LER"]]
    out = out[out.design == "matched"]
    out.to_csv(f"{RES}/robust_eval.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())
