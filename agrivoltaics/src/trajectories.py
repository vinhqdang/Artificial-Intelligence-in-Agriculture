"""Learned control behaviour and simulated shade response on held-out sites.

* trajectories and heat-day statistics of SALS at the calibrated GCR, recorded on
  the sub-field (rainfed or irrigated) that dominates each site's area;
* simulated relative yield as a function of season-mean shading for static arrays.
"""
import os, sys, json
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import sals as A
import simulate as S
from evaluate_grid import load_ctrl
from run_train import get_refs

ROOT = A.ROOT
FIG = f"{ROOT}/manuscript/figures"
CROPS = ["wheat", "rice", "maize", "soybean", "potato"]
torch.set_num_threads(4)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})


def dominant_rollout(dt, g, ctrl):
    irr = (dt["irr_frac"] > 0.5).float()
    with torch.no_grad():
        y, e, rec = S.rollout(dt, g, controller=ctrl, irrigated=irr, record=True)
    return irr, rec


if __name__ == "__main__":
    sites = pd.read_csv(f"{ROOT}/data/sites.csv")
    D = A.load_all(); refs = get_refs(D)
    calg = json.load(open(f"{ROOT}/results/calibrated_gcr_split0.json"))
    ctrl = load_ctrl("sals_s0_seed0")
    test = A.site_split(0)
    stats = []
    traj = {}
    for scen in ["baseline", "+3C"]:
        d = D[scen]
        m = test[d["site"].numpy().astype(int)] & np.isin(d["year"].numpy(), A.TEST_YEARS) & (refs[scen][0] >= 0.2).numpy()
        idx = torch.where(torch.tensor(m))[0]
        dt = S._sel(d, idx)
        g = torch.tensor([calg.get(f"SALS|{scen}|{int(s)}", 0.0) for s in dt["site"]], dtype=torch.float32)
        keep = g > 0
        idx, g = idx[keep], g[keep]
        dt = S._sel(d, idx)
        irr, rec = dominant_rollout(dt, g, ctrl)
        act = dt["active"] > 0
        heat = (rec["fheat"] < 1) & act
        cool = (rec["fheat"] >= 1) & act
        u = rec["u"]
        per_site = pd.DataFrame(dict(site=dt["site"].numpy().astype(int),
                                     u_heat=[(u[i][heat[i]].mean().item() if heat[i].any() else np.nan) for i in range(len(idx))],
                                     u_cool=[(u[i][cool[i]].mean().item() if cool[i].any() else np.nan) for i in range(len(idx))]))
        ps = per_site.groupby("site").mean().dropna()
        stats.append(dict(scenario=scen, u_heat=float(u[heat].mean()), u_cool=float(u[cool].mean()),
                          within_site_diff=float((ps.u_heat - ps.u_cool).mean()),
                          share_sites_less_on_heat=float((ps.u_heat < ps.u_cool).mean() * 100), n_sites=len(ps)))
        if scen == "+3C":
            score = ((1 - rec["fheat"]) * act).sum(1) + ((1 - rec["fwater"]) * act).sum(1)
            for cname in ["maize", "wheat"]:
                ci = CROPS.index(cname)
                cand = torch.where(dt["crop"] == ci)[0]
                if len(cand) == 0:
                    continue
                i = int(cand[torch.argmax(score[cand])])
                s = sites.iloc[int(dt["site"][i])]
                traj[cname] = dict(u=u[i].numpy(), shade=rec["shade"][i].numpy(), tmax=dt["tmax"][i].numpy(),
                                   active=dt["active"][i].numpy(), g=float(g[i]), irr=bool(irr[i]),
                                   label=f"{cname} ({'irrigated' if irr[i] else 'rainfed'}), {s.country} ({s.lat:.1f}, {s.lon:.1f}), "
                                         f"GCR {float(g[i]):.3f}, +3 °C, season sown {int(dt['year'][i])}")
    pd.DataFrame(stats).to_csv(f"{ROOT}/results/control_stats.csv", index=False)
    print(pd.DataFrame(stats).round(3).to_string())
    fig, axs = plt.subplots(len(traj), 1, figsize=(7.2, 1.8 * len(traj)), sharex=True, squeeze=False)
    for ax, (k, tr) in zip(axs[:, 0], traj.items()):
        days = np.arange(365)
        ax.fill_between(days, 0, (tr["active"] > 0) * 1.0, color="#f3f4f6", step="mid")
        ax.plot(days, tr["u"], color="#c2410c", lw=0.9, label="light sharing $u_t$")
        ax.plot(days, tr["shade"], color="#059669", lw=0.7, label="crop light fraction $s_t$")
        ax2 = ax.twinx(); ax2.plot(days, tr["tmax"], color="#6b7280", lw=0.5, label="$T_{max}$ (right axis)")
        ax2.set_ylabel("$T_{max}$ (°C)"); ax2.spines["top"].set_visible(False)
        ax.set_ylim(0, 1.05); ax.set_ylabel("fraction"); ax.set_title(tr["label"], loc="left", fontsize=7.5)
    h1, l1 = axs[0, 0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axs[0, 0].legend(h1 + h2, l1 + l2, fontsize=6, frameon=False, loc="upper right")
    axs[-1, 0].set_xlabel("Day after sowing")
    fig.tight_layout(); fig.savefig(f"{FIG}/trajectory.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/trajectory.png", dpi=200)

    # simulated shade response of static arrays (baseline climate, held-out sites, test years)
    d = D["baseline"]
    m = test[d["site"].numpy().astype(int)] & np.isin(d["year"].numpy(), A.TEST_YEARS) & (refs["baseline"][0] >= 0.2).numpy()
    idx = torch.where(torch.tensor(m))[0]
    dt = S._sel(d, idx)
    rows = []
    for gv in [0.1, 0.2, 0.3, 0.4, 0.5]:
        with torch.no_grad():
            y, e, rec = S.rollout(dt, torch.full((len(idx),), gv), irrigated=torch.zeros(len(idx)), record=True)
            y0, _ = S.rollout(dt, torch.full((len(idx),), gv), panels=False, irrigated=torch.zeros(len(idx)))
        act = dt["active"]
        shade = 1 - (rec["shade"] * act).sum(1) / act.sum(1)
        for c in range(5):
            mm = (dt["crop"] == c) & (y0 > 0.2)
            rows.append(dict(crop=CROPS[c], g=gv, shading=float(shade[mm].mean()), rel_yield=float((y[mm] / y0[mm]).mean())))
    sr = pd.DataFrame(rows); sr.to_csv(f"{ROOT}/results/shade_response.csv", index=False); print(sr.round(3).to_string())
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for c, col in zip(CROPS, ["#b45309", "#0369a1", "#c2410c", "#15803d", "#7c3aed"]):
        x = sr[sr.crop == c]
        ax.plot(100 * x.shading, 100 * x.rel_yield, "-o", ms=3, color=col, label=c)
    ax.plot([0, 60], [100, 40], ":", color="#9ca3af", lw=0.8, label="proportional loss")
    ax.set_xlabel("Season-mean shading of the crop (%)"); ax.set_ylabel("Relative yield, rainfed (%)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/shade_response.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/shade_response.png", dpi=200)
