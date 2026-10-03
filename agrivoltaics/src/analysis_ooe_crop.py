"""Out-of-ensemble design outcomes by crop (E2 and E3; phenology rule; three splits pooled): share of distinct crop-site pairs with
mean retention >= 0.9 under the proportional (P), calibrated (C), field-fitted (F) and harsh (H) responses, per crop.
Output: results/design_ooe_by_crop.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO
from analysis_final import load, feas, pick, MEM        # (re-runs the main design analysis on import)
from analysis_robust import apply
CROP = {0: "wheat", 1: "rice", 2: "maize", 3: "soybean", 4: "potato"}
rows = []
for ens, tg in (("E2", "e2s"), ("E3", "e3s")):
    for sp in (0, 1, 2):
        mem = [feas(load(f"{tg}{sp}_{k}")) for k in MEM]; mean_g = load(f"{tg}{sp}_mean")
        for T, tl in (("conservative", "P"), ("calibrated", "C"), ("field", "F"), ("harsh", "H")):
            truth = load(f"nt{sp}_{T}")
            for name, sel in [("oracle", pick([feas(truth)], 1.0)), ("mean response", pick([feas(mean_g)], 1.0)), ("floor 0.92", pick([feas(mean_g, 0.92)], 1.0)),
                              ("floor 0.96", pick([feas(mean_g, 0.96)], 1.0)), ("75% of members", pick(mem, 0.75))]:
                x = apply(sel, truth)
                s = x.groupby(["site", "crop"]).agg(r=("r", "mean")).reset_index(); s["ok"] = (s.r >= RHO).astype(float)
                s["ens"], s["truth"], s["design"], s["split"] = ens, tl, name, sp; rows.append(s)
R = pd.concat(rows); R["crop"] = R.crop.map(CROP)
P = R.groupby(["ens", "truth", "design", "crop", "site"]).ok.mean().reset_index()     # mean over splits of a pair held out more than once
S = P.groupby(["ens", "truth", "design", "crop"]).agg(pairs=("site", "nunique"), pairs_ok=("ok", "mean")).reset_index(); S["pairs_ok"] *= 100
S.to_csv(f"{RES}/design_ooe_by_crop.csv", index=False)
print(S[S.ens == "E2"].pivot_table(index=["design", "crop"], columns="truth", values="pairs_ok").round(0).to_string())
print(S.groupby("crop").pairs.max())
