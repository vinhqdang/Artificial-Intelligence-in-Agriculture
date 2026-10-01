"""Design under shade-response uncertainty. The array density of a site is the largest GCR whose historical
retention meets the floor in at least 75% of the plausible shade responses (ensemble members), compared with
a design under the ensemble mean and with a design under the true response (oracle).
Held-out member (leave-one-out) and four named responses are used as truths. Output: results/design_uncertainty.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO, HIST, summarize
from analysis_robust import apply

MEMBERS = [0, 3, 5, 8, 10]
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "mb"       # mb: first ensemble, mbml: ensemble calibrated to the data-driven curves
MEAN = sys.argv[2] if len(sys.argv) > 2 else "rb_posterior"
OUT = sys.argv[3] if len(sys.argv) > 3 else "design_uncertainty"
NAME = {"sals_s0_seed0": "SALS", "rob_s0": "SALS"}
Q = 0.75


def load(tag):
    g = pd.read_parquet(f"{RES}/grid_{tag}.parquet")
    g["method"] = g.method.replace(NAME)
    g = g[g.method.isin(["AV static", "Phenology rule", "SALS"]) & (g.y_open >= 0.2)].copy()
    g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    return g


def feas(g):
    h = g[g.year <= HIST].groupby(["method", "site", "g"]).r.mean().reset_index()
    h["ok"] = (h.r >= RHO).astype(float)
    return h[["method", "site", "g", "ok"]]


def pick(feas_list, q):
    f = pd.concat(feas_list).groupby(["method", "site", "g"]).ok.mean().reset_index()
    return f[f.ok >= q].groupby(["method", "site"]).g.max().rename("g_sel").reset_index()


def oracle(g):
    return pick([feas(g)], 1.0)


if __name__ == "__main__":
    mem = {k: load(f"{PREFIX}_{k}") for k in MEMBERS}
    mean_g = load(MEAN)
    named = {t: load(f"rb_{t}") for t in ["conservative", "calibrated", "field", "harsh"]}
    rows = []

    def add(truth_name, truth, design, sel):
        for m, x in apply(sel, truth).groupby("method"):
            s = summarize(x); s["truth"] = truth_name; s["design"] = design; s["method"] = m; rows.append(s)

    for k in MEMBERS:       # leave-one-out over members
        others = [mem[j] for j in MEMBERS if j != k]
        add(f"member{k}", mem[k], "oracle", oracle(mem[k]))
        add(f"member{k}", mem[k], "ensemble mean", pick([feas(mean_g)], 1.0))
        add(f"member{k}", mem[k], "uncertainty-aware (LOO)", pick([feas(o) for o in others], Q))
    for t, g in named.items():
        add(t, g, "oracle", oracle(g))
        add(t, g, "ensemble mean", pick([feas(mean_g)], 1.0))
        add(t, g, "uncertainty-aware", pick([feas(mem[k]) for k in MEMBERS], Q))
    out = pd.DataFrame(rows)[["truth", "design", "method", "sites", "GCR", "erel", "r", "comply", "site_comply", "p10", "LER"]]
    out.to_csv(f"{RES}/{OUT}.csv", index=False)
    out["kind"] = np.where(out.truth.str.startswith("member"), "member (LOO)", out.truth)
    agg = out.groupby(["kind", "design", "method"])[["GCR", "erel", "r", "comply", "site_comply", "LER"]].mean().round(3)
    pd.set_option("display.width", 200); print(agg.to_string())
