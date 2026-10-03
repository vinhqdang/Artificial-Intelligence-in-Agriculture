"""Figures that show the agronomic data: (1) shade-yield observations of the meta-analysis (Laub et al.) with the learned curves and the simulator's
ensemble, (2) crop calendars of the 400 crop-site pairs (GGCMI) with the shading-sensitive window of the crop model."""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), ".."); FIG = f"{ROOT}/manuscript/figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
# ---- (1) observations
d = pd.read_excel(f"{ROOT}/data/laub/Data_Meta_Analysis_updated.xlsx").iloc[:, :12]
d.columns = ["crop", "ctype", "rsr", "yld", "exp", "zone", "lat", "lon", "ghi", "nets", "shade", "ref"]
d["s"] = d.rsr / 100; d["y"] = d.yld / 100; d = d[d.s > 0]
cur = pd.read_csv(f"{ROOT}/results/shade_ml_curves.csv")
sp = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ROOT}/results/spread_E2_m*.csv")])
panels = [("C3 cereals", ["wheat", "rice"], "Wheat and rice"), ("Maize", ["maize"], "Maize"), ("Grain legumes", ["soybean"], "Soybean"), ("Tubers/root crops", ["potato"], "Potato")]
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.9), sharey=True)
for ax, (ct, crops, title) in zip(axs, panels):
    x = d[d.ctype == ct]; c = cur[cur.ctype == ct].sort_values("s")
    ax.plot([0, 60], [1, 0.4], ":", color="#9ca3af", lw=0.8, label="proportional loss")
    ax.fill_between(c.s * 100, c.lo, c.hi, color="#93c5fd", alpha=0.5, lw=0, label="learned curve, 80% interval")
    ax.plot(c.s * 100, c["median"], color="#1d4ed8", lw=1.2, label="learned curve, median")
    pan = x.shade == "Solar_panel"
    ax.scatter(100 * x.s[~pan], x.y[~pan], s=7, color="#374151", alpha=0.55, lw=0, label="observations (field shade)")
    ax.scatter(100 * x.s[pan], x.y[pan], s=9, color="#c2410c", alpha=0.8, lw=0, label="observations (solar panels)")
    for cr, col in zip(crops, ["#15803d", "#a16207"]):
        for m, z in sp[sp.crop == cr].groupby("member"):
            z = z.sort_values("shading"); ax.plot(100 * z.shading, z.rel_yield, color=col, lw=0.5, alpha=0.6)
    ax.set_title(f"{title}  (n = {len(x)})", fontsize=7.5); ax.set_xlim(0, 62); ax.set_ylim(0, 1.35); ax.set_xlabel("Shade (%)")
axs[0].set_ylabel("Relative yield")
h, l = axs[1].get_legend_handles_labels()
fig.legend(h + [plt.Line2D([], [], color="#15803d", lw=0.8)], l + ["simulator, five ensemble members"], loc="lower center", ncol=3, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(f"{FIG}/data_shade.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/data_shade.png", dpi=200); plt.close(fig)
# ---- (2) crop calendars
s = pd.read_csv(f"{ROOT}/data/sites.csv")
cal = {"wheat": ("sow_wwh_rf", "mat_wwh_rf"), "rice": ("sow_ri1_rf", "mat_ri1_rf"), "maize": ("sow_mai_rf", "mat_mai_rf"), "soybean": ("sow_soy_rf", "mat_soy_rf"), "potato": ("sow_pot_rf", "mat_pot_rf")}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axs[0]
for i, (cr, (a, b)) in enumerate(cal.items()):
    z = s[s.crop == cr].copy(); sa = z[a].where(z[a].notna(), z.get("sow_swh_rf") if cr == "wheat" else np.nan); ma = z[b].where(z[b].notna(), z.get("mat_swh_rf") if cr == "wheat" else np.nan)
    z = z.assign(sow=sa, mat=ma).dropna(subset=["sow", "mat"])
    z["len"] = (z.mat - z.sow) % 365; z = z.sort_values(["sow"]).reset_index(drop=True)
    ys = i + (np.arange(len(z)) / max(len(z) - 1, 1) - 0.5) * 0.8
    for yy, (_, r) in zip(ys, z.iterrows()):
        x0, ln = r.sow, r.len
        ax.plot([x0, min(x0 + ln, 365)], [yy, yy], color="#6b7280", lw=0.3, alpha=0.5)
        if x0 + ln > 365: ax.plot([0, x0 + ln - 365], [yy, yy], color="#6b7280", lw=0.3, alpha=0.5)
        f0, f1 = x0 + 0.45 * ln, x0 + 0.65 * ln     # shading-sensitive window around flowering (0.45-0.65 of the season)
        for u, v in ((f0, f1),):
            if v <= 365: ax.plot([u, v], [yy, yy], color="#c2410c", lw=0.6, alpha=0.8)
            else: ax.plot([u % 365, v % 365] if u > 365 else [u, 365], [yy, yy], color="#c2410c", lw=0.6, alpha=0.8)
ax.set_yticks(range(5)); ax.set_yticklabels([c.capitalize() for c in cal]); ax.invert_yaxis()
ax.set_xticks([1, 60, 121, 182, 244, 305, 365]); ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Dec"]); ax.set_xlim(0, 365)
ax.set_xlabel("Day of year (rainfed calendar, one line per crop--site pair)")
ax.plot([], [], color="#6b7280", lw=1, label="sowing to maturity"); ax.plot([], [], color="#c2410c", lw=1.2, label="shade-sensitive window (0.45--0.65 of season)")
ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax = axs[1]
area = s.groupby("crop").area_ha.sum().reindex(list(cal)) / 1e6
irr = s.groupby("crop").apply(lambda z: np.average(z.irr_frac, weights=z.area_ha)).reindex(list(cal))
ax.barh(range(5), area.values, color="#a7f3d0", label="harvested area of the selected cells"); ax.invert_yaxis()
for i, (a_, f) in enumerate(zip(area.values, irr.values)): ax.text(a_ + 0.3, i, f"{100 * f:.0f}% irrigated", va="center", fontsize=6.5)
ax.set_yticks(range(5)); ax.set_yticklabels([c.capitalize() for c in cal]); ax.set_xlabel("Harvested area of selected cells (Mha)"); ax.set_xlim(0, area.max() * 1.45)
fig.tight_layout(); fig.savefig(f"{FIG}/data_calendar.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/data_calendar.png", dpi=200)
print(area.round(1).to_dict(), irr.round(2).to_dict())
