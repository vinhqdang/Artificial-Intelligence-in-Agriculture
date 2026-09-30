"""Tables, statistics and figures of the revised study (from the GCR grids).

Deployable design rule, identical for every method: for each held-out site, the
array GCR is the largest grid value whose retention on the site's 2001-2014
seasons meets the floor ('mean': mean retention >= 0.9; 'chance': >= 80% of
seasons with retention >= 0.9). If no GCR qualifies, no array is built (open
field). Performance is then measured on the 2015-2019 seasons.
"""
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
SCEN_TEX = {"baseline": "Baseline", "+1.5C": "+1.5\\,\\textdegree C", "+2C": "+2\\,\\textdegree C",
            "+3C": "+3\\,\\textdegree C"}
RHO, MIN_OPEN, HIST, TEST = 0.9, 0.2, 2014, 2015
RULES = ["AV static", "Seasonal sharing", "Phenology rule", "Stress rule"]
SALS = "SALS"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"AV static": "#6b7280", "Seasonal sharing": "#059669", "Phenology rule": "#0369a1", "Stress rule": "#7c3aed", SALS: "#c2410c"}


# ----------------------------------------------------------------------------
def calibrate(g, criterion="mean", rho=RHO):
    """Choose one GCR per (method, scenario, kappa, site) from historical seasons."""
    h = g[g.year <= HIST]
    if criterion == "mean":
        feas = h.groupby(["method", "scenario", "kappa", "site", "g"]).r.mean() >= rho
    else:
        feas = h.groupby(["method", "scenario", "kappa", "site", "g"]).r.apply(lambda x: (x >= rho).mean()) >= 0.8
    feas = feas[feas].reset_index()
    best = feas.groupby(["method", "scenario", "kappa", "site"]).g.max().rename("g_sel").reset_index()
    t = g[g.year >= TEST]
    keys = t[["method", "scenario", "kappa", "site"]].drop_duplicates()
    best = keys.merge(best, how="left", on=["method", "scenario", "kappa", "site"])
    sel = t.merge(best, on=["method", "scenario", "kappa", "site"])
    chosen = sel[sel.g == sel.g_sel]
    # sites with no feasible GCR: no array (open field)
    none = sel[sel.g_sel.isna()].drop_duplicates(["method", "scenario", "kappa", "site", "year"]).copy()
    none["g"], none["y"], none["e"], none["r"], none["erel"] = 0.0, none.y_open, 0.0, 1.0, 0.0
    out = pd.concat([chosen, none], ignore_index=True).drop(columns="g_sel")
    out["ler"] = out.r + out.erel
    return out


def summarize(x):
    site_mean = x.groupby("site").r.mean()
    return pd.Series(dict(n=len(x), sites=x.site.nunique(), GCR=x.g.mean(), no_array=(x.groupby("site").g.max() == 0).mean() * 100,
                          E=x.e.mean(), erel=x.erel.mean(), r=x.r.mean(), r_prod=x.y.sum() / x.y_open.sum(),
                          comply=(x.r >= RHO).mean() * 100, site_comply=(site_mean >= RHO).mean() * 100,
                          p10=x.r.quantile(0.1), below80=(x.r < 0.8).mean() * 100, LER=x.ler.mean()))


def boot_diff(a, b, n=2000, seed=0):
    """Cluster (site) bootstrap CI of the mean paired difference of site means."""
    rng = np.random.default_rng(seed)
    d = (a - b).values
    return np.percentile([rng.choice(d, len(d)).mean() for _ in range(n)], [2.5, 97.5])


# ----------------------------------------------------------------------------
def tables_main(cal0, cal_by_split, seeds_cal):
    rows = []
    for scen in SCEN:
        for m in RULES + [SALS]:
            x = cal0[(cal0.scenario == scen) & (cal0.method == m) & (cal0.kappa == 1.0)]
            s = summarize(x); s["scenario"], s["method"] = scen, m
            rows.append(s)
    t = pd.DataFrame(rows)
    t.to_csv(f"{RES}/table_main.csv", index=False)
    with open(f"{TAB}/main.tex", "w") as f:
        for scen in SCEN:
            f.write(f"\\multicolumn{{9}}{{l}}{{\\textit{{{SCEN_TEX[scen]}}}}}\\\\\n")
            ts = t[t.scenario == scen]
            for _, r in ts.iterrows():
                name = "SALS (ours)" if r.method == SALS else r.method
                f.write(f"{name} & {r.GCR:.2f} & {r.E:.0f} & {r.erel:.2f} & {r.r:.3f} & {r.r_prod:.3f} & "
                        f"{r.comply:.0f} & {r.site_comply:.0f} & {r.LER:.2f}\\\\\n")
            if scen != SCEN[-1]:
                f.write("\\addlinespace\n")
    return t


def stats(cal0):
    out = []
    for scen in SCEN:
        a = cal0[(cal0.scenario == scen) & (cal0.method == SALS) & (cal0.kappa == 1.0)].groupby("site")[["ler", "erel", "r"]].mean()
        for m in RULES:
            b = cal0[(cal0.scenario == scen) & (cal0.method == m) & (cal0.kappa == 1.0)].groupby("site")[["ler", "erel", "r"]].mean()
            j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
            for metric in ["ler", "erel", "r"]:
                d = j[f"{metric}_a"] - j[f"{metric}_b"]
                lo, hi = boot_diff(j[f"{metric}_a"], j[f"{metric}_b"])
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                out.append(dict(scenario=scen, versus=m, metric=metric, diff=d.mean(), lo=lo, hi=hi, p=p,
                                n_sites=len(j), share_better=(d > 0).mean() * 100))
    s = pd.DataFrame(out); s.to_csv(f"{RES}/stats.csv", index=False)
    pf = lambda p: "$<10^{-4}$" if p < 1e-4 else f"{p:.3f}"
    with open(f"{TAB}/stats.tex", "w") as f:
        for scen in SCEN:
            for m in RULES:
                a = s[(s.scenario == scen) & (s.versus == m) & (s.metric == "ler")].iloc[0]
                b = s[(s.scenario == scen) & (s.versus == m) & (s.metric == "r")].iloc[0]
                f.write(f"{SCEN_TEX[scen]} & {m} & {a['diff']:+.3f} [{a.lo:+.3f}, {a.hi:+.3f}] & {pf(a.p)} & "
                        f"{a.share_better:.0f} & {b['diff']:+.3f} & {pf(b.p)} & {int(a.n_sites)}\\\\\n")
    return s


def robustness(cal_by_split, seeds_cal):
    rows = []
    for name, c in seeds_cal.items():
        for scen in ["baseline", "+3C"]:
            x = c[(c.scenario == scen) & (c.method == SALS) & (c.kappa == 1.0)]
            s = summarize(x); rows.append(dict(kind="seed", run=name, scenario=scen, **s))
    for sp, c in cal_by_split.items():
        for scen in ["baseline", "+3C"]:
            for m in RULES + [SALS]:
                x = c[(c.scenario == scen) & (c.method == m) & (c.kappa == 1.0)]
                s = summarize(x); rows.append(dict(kind="split", run=f"split{sp}", method=m, scenario=scen, **s))
    r = pd.DataFrame(rows); r.to_csv(f"{RES}/robustness.csv", index=False)
    return r


def kappa_table(cal_k):
    rows = []
    for scen in ["baseline", "+3C"]:
        for k in [0.0, 1.0, 2.0]:
            for m in RULES + [SALS]:
                x = cal_k[(cal_k.scenario == scen) & (cal_k.method == m) & (cal_k.kappa == k)]
                if len(x):
                    s = summarize(x); rows.append(dict(scenario=scen, kappa=k, method=m, **s))
    t = pd.DataFrame(rows); t.to_csv(f"{RES}/kappa_sensitivity.csv", index=False)
    return t


def crop_region(cal0, sites):
    x = cal0[(cal0.kappa == 1.0) & cal0.scenario.isin(["baseline", "+3C"]) & cal0.method.isin(["Stress rule", SALS])].copy()
    x["crop_name"] = x.crop.map(dict(enumerate(CROPS)))
    x = x.join(sites[["lat", "lon"]], on="site")
    x["region"] = [region(la, lo) for la, lo in zip(x.lat, x.lon)]
    a = x.groupby(["scenario", "method", "crop_name"]).apply(summarize).reset_index()
    b = x.groupby(["scenario", "method", "region"]).apply(summarize).reset_index()
    a.to_csv(f"{RES}/by_crop.csv", index=False); b.to_csv(f"{RES}/by_region.csv", index=False)
    return a, b


def region(lat, lon):
    if -30 <= lon <= 60 and lat >= 35: return "Europe & W. Asia"
    if lon >= 60 and lat >= 36: return "Central & E. Asia (north)"
    if 60 <= lon < 95 and lat < 36: return "South Asia"
    if lon >= 95 and lat < 36: return "E. & SE Asia (south)"
    if -30 <= lon < 60 and lat < 35: return "Africa & Middle East"
    if lon < -30 and lat >= 15: return "North America"
    return "Latin America & Oceania" if lon < -30 else "Oceania"


# ----------------------------------------------------------------------------
def fig_frontier(g0):
    t = g0[(g0.year >= TEST) & (g0.kappa == 1.0)]
    m = t.groupby(["scenario", "method", "g"]).agg(erel=("erel", "mean"), r=("r", "mean")).reset_index()
    m.to_csv(f"{RES}/frontier.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, scen in zip(axs, ["baseline", "+3C"]):
        for meth, col in COL.items():
            x = m[(m.scenario == scen) & (m.method == meth)]
            ax.plot(x.erel, x.r, "-o", ms=2.5, lw=1, color=col, label="SALS (ours)" if meth == SALS else meth)
            for gv in [0.2, 0.3, 0.4, 0.5]:
                p = x[np.isclose(x.g, gv)]
                if len(p):
                    ax.annotate(f"{gv:.1f}", (p.erel.iloc[0], p.r.iloc[0]), fontsize=5.5, color=col,
                                xytext=(2, 2), textcoords="offset points")
        ax.axhline(RHO, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Electricity relative to PV plant, $e$"); ax.set_ylabel("Mean yield retention, $r$")
        ax.set_title("a  Baseline climate" if scen == "baseline" else "b  +3 °C", loc="left", fontsize=8)
    axs[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{FIG}/frontier.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/frontier.png", dpi=200)
    return m


def fig_sites(sites):
    import cartopy.crs as ccrs, cartopy.feature as cf
    sys.path.insert(0, os.path.dirname(__file__)); import sals as A
    fig = plt.figure(figsize=(7.2, 3.0)); ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global(); ax.add_feature(cf.LAND, facecolor="#eeeeee"); ax.coastlines(lw=0.25)
    te = A.site_split(0)
    marks = dict(wheat="o", rice="s", maize="^", soybean="D", potato="v")
    for c, mk in marks.items():
        for flag, col in [(False, "#1d4ed8"), (True, "#c2410c")]:
            m = (sites.crop == c) & (te == flag)
            ax.scatter(sites.lon[m], sites.lat[m], marker=mk, s=9, facecolors="none", edgecolors=col, lw=0.6,
                       transform=ccrs.PlateCarree(), label=f"{c} ({'test' if flag else 'train'})")
    ax.legend(ncol=5, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.savefig(f"{FIG}/sites.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/sites.png", dpi=200, bbox_inches="tight")


def pooled_test_sites(cal_by_split):
    """Every site once, evaluated by the split in which it is a held-out site."""
    seen, parts = set(), []
    for sp in sorted(cal_by_split):
        c = cal_by_split[sp]
        c = c[(c.kappa == 1.0) & ~c.site.isin(seen)]
        parts.append(c.assign(split=sp)); seen |= set(c.site.unique())
    return pd.concat(parts)


def upscale(pool, fraction=0.05):
    area = pd.read_csv(f"{ROOT}/data/phys_area_by_crop.csv", index_col=0).iloc[:, 0]
    key = dict(wheat="whea", rice="rice", maize="maiz", soybean="soyb", potato="pota")
    rows = []
    for scen in ["baseline", "+3C"]:
        for m in RULES + [SALS]:
            x = pool[(pool.scenario == scen) & (pool.method == m)]
            tw = cap = prod = prod_open = 0.0
            for ci, c in enumerate(CROPS):
                xc = x[x.crop == ci]
                a = area[key[c]] * fraction
                tw += a * xc.e.mean() / 1e6                                  # TWh
                cap += a * (2.1 * xc.g).mean() / 1e6                         # TWp
                prod += a * xc.y.mean(); prod_open += a * xc.y_open.mean()
            rows.append(dict(scenario=scen, method=m, area_Mha=area.sum() * fraction / 1e6, TWh=tw, TWp=cap,
                             CF=tw / (cap * 8760.0) if cap > 0 else np.nan, retention_prod=prod / prod_open))
    u = pd.DataFrame(rows); u.to_csv(f"{RES}/upscale.csv", index=False)
    return u


def fig_supply(pool):
    area = pd.read_csv(f"{ROOT}/data/phys_area_by_crop.csv", index_col=0).iloc[:, 0]
    key = dict(wheat="whea", rice="rice", maize="maiz", soybean="soyb", potato="pota")
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    out = []
    for m in RULES + [SALS]:
        x = pool[(pool.scenario == "baseline") & (pool.method == m)].groupby(["site", "crop"]).agg(e=("e", "mean")).reset_index()
        n = x.crop.value_counts()
        x["w"] = [area[key[CROPS[c]]] / n[c] for c in x.crop]           # ha represented by each site
        x = x.sort_values("e", ascending=False)
        cum_a = x.w.cumsum() / area.sum() * 100
        cum_e = (x.w * x.e).cumsum() / 1e6
        ax.plot(cum_a, cum_e / 1000, color=COL[m], lw=1.2, label="SALS (ours)" if m == SALS else m)
        out.append(pd.DataFrame(dict(method=m, area_pct=cum_a.values, PWh=(cum_e / 1000).values)))
    ax.set_xlim(0, 10); ax.set_xlabel("Staple cropland equipped, best sites first (%)")
    ax.set_ylabel("Electricity (PWh yr$^{-1}$)")
    ymax = max(o[o.area_pct <= 10].PWh.max() for o in out); ax.set_ylim(0, ymax * 1.05)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/supply.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/supply.png", dpi=200)
    pd.concat(out).to_csv(f"{RES}/supply_curve.csv", index=False)


def fig_map(pool, sites):
    import cartopy.crs as ccrs, cartopy.feature as cf
    x = pool[(pool.scenario == "baseline") & (pool.method == SALS)].groupby("site").agg(
        e=("e", "mean"), ler=("ler", "mean"), g=("g", "mean")).join(sites)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), subplot_kw=dict(projection=ccrs.Robinson()))
    for ax, col, lab, cmap in [(axs[0], "e", "Electricity (MWh ha$^{-1}$ yr$^{-1}$)", "YlOrRd"),
                               (axs[1], "ler", "Land-equivalent ratio", "viridis")]:
        ax.set_global(); ax.add_feature(cf.LAND, facecolor="#eeeeee"); ax.coastlines(lw=0.25)
        sc = ax.scatter(x.lon, x.lat, c=x[col], s=6, cmap=cmap, transform=ccrs.PlateCarree(), linewidths=0)
        cb = plt.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03); cb.set_label(lab)
    axs[0].set_title("a  SALS electricity (baseline climate)", loc="left", fontsize=8)
    axs[1].set_title("b  SALS land-equivalent ratio", loc="left", fontsize=8)
    fig.savefig(f"{FIG}/map.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/map.png", dpi=200, bbox_inches="tight")


def fig_warming(pool):
    """Production relative to the baseline open field (pooled held-out sites)."""
    base = pool[(pool.scenario == "baseline") & (pool.method == SALS)].set_index(["site", "year"]).y_open
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 2.0), sharey=True)
    rows = []
    for ci, c in enumerate(CROPS):
        ax = axs[ci]
        for key, meth, lab, col in [("y_open", SALS, "Open field", "#111827"), ("y", "Stress rule", "Stress rule", COL["Stress rule"]),
                                    ("y", SALS, "SALS", COL[SALS])]:
            vals = []
            for scen in SCEN:
                x = pool[(pool.scenario == scen) & (pool.method == meth) & (pool.crop == ci)].set_index(["site", "year"])
                b = base.loc[x.index]
                v = 100 * (x[key].sum() / b.sum() - 1); vals.append(v)
                rows.append(dict(crop=c, scenario=scen, system=lab, change=v))
            ax.plot(range(4), vals, "-o", ms=3, color=col, label=lab, lw=1)
        ax.axhline(0, color="k", lw=0.4); ax.set_xticks(range(4)); ax.set_xticklabels(["0", "1.5", "2", "3"])
        ax.set_title(c, fontsize=8); ax.set_xlabel("Warming (\u00b0C)")
    axs[0].set_ylabel("Production vs. baseline (%)"); axs[-1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/warming.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/warming.png", dpi=200)
    pd.DataFrame(rows).to_csv(f"{RES}/warming_production.csv", index=False)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    sites = pd.read_csv(f"{ROOT}/data/sites.csv")
    seed_names = {"seed0": "sals_s0_seed0", "seed1": "sals_s0_seed1", "seed2": "sals_s0_seed2"}

    def load_with_model(tag):
        g = pd.read_parquet(f"{RES}/grid_{tag}.parquet")
        g["_model"] = g.method
        g["method"] = g.method.replace({m: SALS for m in g.method.unique() if m.startswith(("sals_", "abl_"))})
        op = g.drop_duplicates(["site", "year", "scenario"]).pivot_table(index=["site", "year"], columns="scenario", values="y_open")
        ok = (op >= MIN_OPEN).all(axis=1)
        ok = ok[ok].reset_index()[["site", "year"]]
        g = g.merge(ok, on=["site", "year"])
        g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
        return g

    g0 = load_with_model("split0")
    g0k = pd.concat([load_with_model("split0_kappa"), g0[g0._model.isin(RULES + ["sals_s0_seed0"])]])
    grids = {0: g0[g0._model.isin(RULES + ["sals_s0_seed0"])], 1: load_with_model("split1"), 2: load_with_model("split2")}
    cal = {sp: calibrate(g.drop(columns="_model")) for sp, g in grids.items()}
    cal0 = cal[0]
    seeds_cal = {k: calibrate(g0[g0._model.isin(RULES + [v])].drop(columns="_model")) for k, v in seed_names.items()}
    abl = {k: calibrate(g0[g0._model.isin([v])].drop(columns="_model"))
           for k, v in {"w/o stress features": "abl_nostress", "baseline climate only": "abl_baseonly"}.items()}
    chance = calibrate(grids[0].drop(columns="_model"), "chance")
    calk = calibrate(g0k.drop(columns="_model").drop_duplicates(["method", "scenario", "kappa", "site", "year", "g"]))

    t = tables_main(cal0, cal, seeds_cal); print(t.round(3).to_string())
    s = stats(cal0); print(s[s.metric == "ler"].round(4).to_string())
    rb = robustness(cal, seeds_cal); print(rb.round(3).to_string())
    kt = kappa_table(calk); print(kt[["scenario", "kappa", "method", "E", "r", "r_prod", "comply", "LER"]].round(3).to_string())
    ch = pd.DataFrame([dict(scenario=sc, method=m, **summarize(chance[(chance.scenario == sc) & (chance.method == m)]))
                       for sc in ["baseline", "+3C"] for m in RULES + [SALS]])
    ch.to_csv(f"{RES}/chance_constraint.csv", index=False); print(ch.round(3).to_string())
    ab = pd.DataFrame([dict(variant=k, scenario=sc, **summarize(c[c.scenario == sc]))
                       for k, c in abl.items() for sc in ["baseline", "+3C"]] +
                      [dict(variant="SALS seed0", scenario=sc, **summarize(seeds_cal["seed0"][(seeds_cal["seed0"].scenario == sc) & (seeds_cal["seed0"].method == SALS)]))
                       for sc in ["baseline", "+3C"]])
    ab.to_csv(f"{RES}/ablation.csv", index=False); print(ab.round(3).to_string())
    a, b = crop_region(cal0, sites); print(a.round(3).to_string()); print(b.round(3).to_string())
    fr = fig_frontier(g0.drop(columns="_model")[g0._model.isin(RULES + ["sals_s0_seed0"])])
    pool = pooled_test_sites(cal); pool.to_csv(f"{RES}/pooled_calibrated.csv", index=False)
    print("pooled sites", pool.site.nunique())
    print(upscale(pool).round(3).to_string())
    fig_sites(sites); fig_supply(pool); fig_map(pool, sites); fig_warming(pool)
    json.dump({f"{r.method}|{r.scenario}|{r.site}": r.g for r in cal0[cal0.kappa == 1.0].drop_duplicates(["method", "scenario", "site"]).itertuples()},
              open(f"{RES}/calibrated_gcr_split0.json", "w"))
