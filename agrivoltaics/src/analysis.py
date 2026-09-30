"""Tables, statistics and figures for the manuscript."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG, TAB = f"{ROOT}/results", f"{ROOT}/manuscript/figures", f"{ROOT}/manuscript/tables"
CROPS = ["wheat", "rice", "maize", "soybean", "potato"]
SCEN = ["baseline", "+1.5C", "+2C", "+3C"]
SCEN_TEX = {"baseline": "Baseline", "+1.5C": "+1.5\\,\\textdegree C", "+2C": "+2\\,\\textdegree C", "+3C": "+3\\,\\textdegree C"}
RHO, MIN_OPEN = 0.9, 0.2
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
COL = {"SALS (ours)": "#c2410c", "AV static (rule)": "#6b7280", "AV static (oracle GCR)": "#1d4ed8",
       "Seasonal sharing": "#059669", "Stress rule": "#7c3aed", "Open-loop (foresight)": "#111827"}


def add_metrics(df):
    df = df.copy()
    df["method"] = df.method.replace({"Oracle open-loop": "Open-loop (foresight)"})
    df["ret"] = df.y / df.y_open.clip(lower=0.05)
    df["erel"] = df.e / df.e_pv
    df["ler"] = df.ret + df.erel
    df["ok"] = df.y_open >= MIN_OPEN
    return df


def boot_ci(x, groups, n=1000, seed=0):
    """Bootstrap 95% CI of the mean, resampling sites."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    per = {g: x[groups == g] for g in ug}
    means = []
    for _ in range(n):
        s = rng.choice(ug, len(ug))
        means.append(np.concatenate([per[g] for g in s]).mean())
    return np.percentile(means, [2.5, 97.5])


def main_table(ev, order):
    rows = []
    for scen in SCEN:
        for m in order:
            x = ev[(ev.scenario == scen) & (ev.method == m) & ev.ok]
            if len(x) == 0:
                continue
            lo, hi = boot_ci(x.ler.values, x.site.values)
            rows.append(dict(scenario=scen, method=m, n=len(x), GCR=x.g.mean(), E=x.e.mean(), erel=x.erel.mean(),
                             ret=x.ret.mean(), comply=100 * (x.ret >= RHO).mean(), LER=x.ler.mean(), LER_lo=lo, LER_hi=hi))
    t = pd.DataFrame(rows)
    t.to_csv(f"{RES}/table_main.csv", index=False)
    with open(f"{TAB}/main.tex", "w") as f:
        for scen in SCEN:
            f.write(f"\\multicolumn{{7}}{{l}}{{\\textit{{{SCEN_TEX[scen]}}}}}\\\\\n")
            ts = t[t.scenario == scen]
            best_e = ts[ts.method != "Open-loop (foresight)"].E.max()
            for _, r in ts.iterrows():
                name = r.method.replace("(ours)", "(ours)")
                e_str = f"{r.E:.0f}"
                if r.method != "Open-loop (foresight)" and r.E == best_e:
                    e_str = f"\\textbf{{{e_str}}}"
                f.write(f"{name} & {r.GCR:.2f} & {e_str} & {r.erel:.2f} & {r.ret:.3f} & {r.comply:.1f} & "
                        f"{r.LER:.2f} [{r.LER_lo:.2f}, {r.LER_hi:.2f}]\\\\\n")
            f.write("\\addlinespace\n")
    return t


def stats(ev, ref="SALS (ours)"):
    out = []
    for scen in SCEN:
        a = ev[(ev.scenario == scen) & (ev.method == ref) & ev.ok].set_index(["site", "year"])
        for m in ev.method.unique():
            if m == ref:
                continue
            b = ev[(ev.scenario == scen) & (ev.method == m) & ev.ok].set_index(["site", "year"])
            j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
            if len(j) < 10:
                continue
            for metric in ["ler", "erel"]:
                d = j[f"{metric}_a"] - j[f"{metric}_b"]
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                out.append(dict(scenario=scen, versus=m, metric=metric, mean_diff=d.mean(), p=p, n=len(j)))
    s = pd.DataFrame(out)
    s.to_csv(f"{RES}/stats.csv", index=False)
    return s


def fig_map(gl, sites):
    import cartopy.crs as ccrs, cartopy.feature as cf
    x = gl[gl.scenario == "baseline"].groupby("site").agg(e=("e", "mean"), y=("y", "mean"), yo=("y_open", "mean"),
                                                          ep=("e_pv", "mean"), g=("g", "mean")).join(sites)
    x["ler"] = x.y / x.yo.clip(lower=0.05) + x.e / x.ep
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), subplot_kw=dict(projection=ccrs.Robinson()))
    for ax, col, lab, cmap in [(axs[0], "e", "Electricity (MWh ha$^{-1}$ yr$^{-1}$)", "YlOrRd"),
                               (axs[1], "ler", "Land-equivalent ratio", "viridis")]:
        ax.set_global(); ax.add_feature(cf.LAND, facecolor="#eeeeee"); ax.coastlines(lw=0.25)
        sc = ax.scatter(x.lon, x.lat, c=x[col], s=6, cmap=cmap, transform=ccrs.PlateCarree(), linewidths=0)
        cb = plt.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03); cb.set_label(lab)
    axs[0].set_title("a  SALS electricity (baseline climate)", loc="left", fontsize=8)
    axs[1].set_title("b  SALS land-equivalent ratio", loc="left", fontsize=8)
    fig.savefig(f"{FIG}/map.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/map.png", dpi=200, bbox_inches="tight")


def fig_sites(sites):
    import cartopy.crs as ccrs, cartopy.feature as cf
    split = np.load(f"{RES}/test_sites.npy")
    fig = plt.figure(figsize=(7.2, 3.0)); ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global(); ax.add_feature(cf.LAND, facecolor="#eeeeee"); ax.coastlines(lw=0.25)
    marks = dict(wheat="o", rice="s", maize="^", soybean="D", potato="v")
    for c, mk in marks.items():
        for te, col in [(False, "#1d4ed8"), (True, "#c2410c")]:
            m = (sites.crop == c) & (split == te)
            ax.scatter(sites.lon[m], sites.lat[m], marker=mk, s=9, facecolors="none", edgecolors=col, lw=0.6,
                       transform=ccrs.PlateCarree(), label=f"{c} ({'test' if te else 'train'})")
    ax.legend(ncol=5, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.savefig(f"{FIG}/sites.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/sites.png", dpi=200, bbox_inches="tight")


def fig_warming(gl):
    """Yield change relative to the baseline open field, by crop and warming level."""
    base = gl[gl.scenario == "baseline"].set_index(["site", "year"]).y_open
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 2.0), sharey=True)
    rows = []
    for ci, c in enumerate(CROPS):
        ax = axs[ci]
        for key, lab, col in [("y_open", "Open field", "#6b7280"), ("y_static", "AV static (rule)", "#1d4ed8"),
                              ("y", "SALS", "#c2410c")]:
            vals = []
            for scen in SCEN:
                x = gl[(gl.scenario == scen) & (gl.crop == ci)].set_index(["site", "year"])
                b = base.loc[x.index]
                ok = b >= MIN_OPEN
                v = 100 * (x[key][ok].sum() / b[ok].sum() - 1)
                vals.append(v); rows.append(dict(crop=c, scenario=scen, system=lab, change=v))
            ax.plot(range(4), vals, "-o", ms=3, color=col, label=lab, lw=1)
        ax.axhline(0, color="k", lw=0.4); ax.set_xticks(range(4)); ax.set_xticklabels(["0", "1.5", "2", "3"])
        ax.set_title(c, fontsize=8); ax.set_xlabel("Warming (\u00b0C)")
    axs[0].set_ylabel("Yield vs. baseline open field (%)")
    axs[-1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/warming.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/warming.png", dpi=200)
    pd.DataFrame(rows).to_csv(f"{RES}/warming_yield_change.csv", index=False)


def fig_tradeoff(ev):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, scen in zip(axs, ["baseline", "+3C"]):
        for m, col in COL.items():
            x = ev[(ev.scenario == scen) & (ev.method == m) & ev.ok]
            if len(x) == 0:
                continue
            ax.scatter(x.erel.mean(), x.ret.mean(), color=col, s=30, label=m, zorder=3)
            ax.errorbar(x.erel.mean(), x.ret.mean(), xerr=x.erel.std() / np.sqrt(len(x)) * 1.96,
                        yerr=x.ret.std() / np.sqrt(len(x)) * 1.96, color=col, lw=0.8)
        for L in [1.2, 1.4, 1.6, 1.8]:
            xx = np.linspace(0, 1, 20); ax.plot(xx, L - xx, ":", color="#9ca3af", lw=0.6)
            ax.text(L - 0.815, 0.815, f"LER={L}", fontsize=6, color="#6b7280", rotation=-52)
        ax.axhline(RHO, color="k", lw=0.5, ls="--")
        ax.set_xlim(0.2, 0.95); ax.set_ylim(0.8, 1.0)
        ax.set_xlabel("Electricity relative to PV plant, $e$"); ax.set_ylabel("Yield retention, $r$")
        ax.set_title(f"{'a  Baseline climate' if scen == 'baseline' else 'b  +3 °C'}", loc="left", fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, frameon=False, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{FIG}/tradeoff.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/tradeoff.png", dpi=200)


def fig_trajectory(sites):
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True)
    tb = np.load(f"{RES}/traj_+3C.npz")
    # pick the rainfed site-season with the most heat-stress days among maize sites
    crop = tb["crop"]; heat = ((1 - tb["fheat"]) * tb["active"]).sum(1); dry = ((1 - tb["fwater"]) * tb["active"]).sum(1)
    cands = [("maize", np.where(crop == 2)[0]), ("wheat", np.where(crop == 0)[0])]
    for ax, (name, idx) in zip(axs, cands):
        i = idx[np.argmax(heat[idx] + dry[idx])]
        s = sites.iloc[int(tb["site"][i])]
        act = tb["active"][i] > 0
        days = np.arange(365)
        ax.fill_between(days, 0, act * 1.0, color="#f3f4f6", step="mid")
        ax.plot(days, tb["u"][i], color="#c2410c", lw=0.9, label="light sharing $u_t$")
        ax.plot(days, tb["shade"][i], color="#059669", lw=0.7, label="crop light fraction $s_t$")
        ax2 = ax.twinx(); ax2.plot(days, tb["tmax"][i], color="#6b7280", lw=0.5, label="$T_{max}$ (right axis)")
        ax2.set_ylabel("$T_{max}$ (\u00b0C)"); ax2.spines["top"].set_visible(False)
        ax.set_ylim(0, 1.05); ax.set_ylabel("fraction")
        ax.set_title(f"{name}, {s.country} ({s.lat:.1f}, {s.lon:.1f}), GCR {tb['g'][i]:.2f}, +3 \u00b0C, 2017 season",
                     loc="left", fontsize=8)
    h1, l1 = axs[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axs[0].legend(h1 + h2, l1 + l2, fontsize=6, frameon=False, loc="upper right"); axs[1].set_xlabel("Day after sowing")
    fig.tight_layout(); fig.savefig(f"{FIG}/trajectory.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/trajectory.png", dpi=200)


def upscale(gl, fraction=0.05):
    """Electricity if SALS agrivoltaics covered `fraction` of each crop's harvested area."""
    sp = pd.read_csv(f"/home/user/data/spam/spam_1deg.csv")
    area = {"wheat": sp.whea_a.sum(), "rice": sp.rice_a.sum(), "maize": sp.maiz_a.sum(),
            "soybean": sp.soyb_a.sum(), "potato": sp.pota_a.sum()}
    out = []
    for scen in SCEN:
        tot = 0; tot_s = 0
        for ci, c in enumerate(CROPS):
            x = gl[(gl.scenario == scen) & (gl.crop == ci)]
            twh = area[c] * fraction * x.e.mean() / 1e6
            tot += twh; tot_s += area[c] * fraction * x.e_static.mean() / 1e6
            out.append(dict(scenario=scen, crop=c, area_Mha=area[c] / 1e6, TWh=twh))
        out.append(dict(scenario=scen, crop="total", area_Mha=sum(area.values()) / 1e6, TWh=tot, TWh_static=tot_s))
    u = pd.DataFrame(out); u.to_csv(f"{RES}/upscale.csv", index=False)
    return u


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    sites = pd.read_csv(f"{ROOT}/data/sites.csv")
    sys.path.insert(0, os.path.dirname(__file__))
    import sals as A
    np.save(f"{RES}/test_sites.npy", A.site_split())
    ev = add_metrics(pd.read_csv(f"{RES}/eval_main.csv"))
    order = ["AV static (rule)", "AV static (oracle GCR)", "Seasonal sharing", "Stress rule", "SALS (ours)", "Open-loop (foresight)"]
    print(main_table(ev, order).round(3).to_string())
    print(stats(ev).round(4).to_string())
    fig_sites(sites); fig_tradeoff(ev)
    if os.path.exists(f"{RES}/global_sals_main.csv"):
        gl = pd.read_csv(f"{RES}/global_sals_main.csv")
        fig_map(gl, sites); fig_warming(gl); fig_trajectory(sites)
        print(upscale(gl).round(1).to_string())
