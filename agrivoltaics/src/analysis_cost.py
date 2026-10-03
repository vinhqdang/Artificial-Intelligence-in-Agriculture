"""Economics of light-sharing control and of the design margin (illustrative parameters in cost.py).
Outputs: results/controllers_cost.csv, cost_sensitivity.csv, design_cost.csv, figure cost_price.pdf"""
import os, sys, copy
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO, HIST
from analysis_robust import apply, design
import cost as C
FIG = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def comp_rows(x):
    c = C.components(x); d = pd.DataFrame(c); d["site"] = x.site.values; d["method"] = x.method.values; return d


def pooled(df):
    """mean components per (method, distinct pair), splits pooled"""
    return df.groupby(["method", "site"])[["E", "g", "e_pv", "closs", "copen"]].mean().reset_index()


# ---- controllers (main response, three splits, coarse grid)
rows = []
for sp in (0, 1, 2):
    g = pd.read_parquet(f"{RES}/grid_ctl{sp}.parquet")
    g = g[~g.method.str.match(r"sals_s\d_seed[12]")].copy(); g["method"] = np.where(g.method.str.startswith("sals_"), "SALS", g.method)
    g = g[g.y_open >= 0.2].copy(); g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv
    rows.append(comp_rows(apply(design(g, None), g)))
R = pooled(pd.concat(rows))
order = ["AV static", "Seasonal sharing", "Stress rule", "Phenology rule", "Tuned phenology rule", "Optimised ramp schedule", "SALS"]


def summarize(R, p=C.BASE):
    out = []
    for m in order:
        x = R[R.method == m]
        net = C.net_av(x.E.values, x.g.values, x.closs.values, p)
        n3 = {f"net_{q}": C.net_av(x.E.values, x.g.values, x.closs.values, dict(p, p_e=q)).mean() for q in (60, 90, 120)}
        out.append(dict(method=m, pairs=len(x), E=x.E.mean(), capex_ann=C.annual_capex(x.g.values, p["capex_av"], p).mean(),
                        crop_loss=p["crop_mult"] * x.closs.mean(), net=net.mean(), **n3,
                        breakeven=C.breakeven_price(x.E.mean(), x.g.mean(), x.closs.mean(), p) if False else
                        (C.annual_capex(x.g.values, p["capex_av"], p).mean() + p["crop_mult"] * x.closs.mean()) / x.E.mean()))
    return pd.DataFrame(out)


S = summarize(R)
R.assign(net90=C.net_av(R.E.values, R.g.values, R.closs.values, dict(C.BASE, p_e=90.0)))[["method", "site", "net90"]].to_csv(f"{RES}/cost_ler_pairs90.csv", index=False)
x0 = R[R.method == "SALS"]
pv = C.net_pv_plant(x0.e_pv.values, x0.copen.values).mean()
S.loc[len(S)] = dict(method="Conventional PV plant (GCR 0.4, crop replaced)", pairs=len(x0), E=x0.e_pv.mean(),
                     capex_ann=C.annual_capex(0.4, C.BASE["capex_pv"], C.BASE), crop_loss=x0.copen.mean(), net=pv,
                     **{f"net_{q}": C.net_pv_plant(x0.e_pv.values, x0.copen.values, dict(C.BASE, p_e=q)).mean() for q in (60, 90, 120)},
                     breakeven=(C.annual_capex(0.4, C.BASE["capex_pv"], C.BASE) + x0.copen.mean()) / x0.e_pv.mean())
S.to_csv(f"{RES}/controllers_cost.csv", index=False); print(S.round(1).to_string())

# ---- sensitivity (one at a time)
sens = []
for name, key, vals in [("electricity price (USD/MWh)", "p_e", [30, 60, 90, 120]), ("AV capex (USD/Wp)", "capex_av", [0.7, 1.0, 1.5]),
                        ("discount rate", "rate", [0.04, 0.06, 0.08]), ("crop prices (multiplier)", "crop_mult", [0.5, 1.0, 2.0])]:
    for v in vals:
        p = copy.deepcopy(C.BASE); p[key] = v
        s = summarize(R, p); 
        for _, r in s.iterrows(): sens.append(dict(parameter=name, value=v, method=r.method, net=r.net))
        xx = x0; sens.append(dict(parameter=name, value=v, method="Conventional PV plant", net=C.net_pv_plant(xx.e_pv.values, xx.copen.values, p).mean()))
pd.DataFrame(sens).to_csv(f"{RES}/cost_sensitivity.csv", index=False)
fig, ax = plt.subplots(figsize=(3.6, 2.7))
pr = np.arange(20, 131, 10); col = {"AV static": "#6b7280", "Phenology rule": "#0369a1", "Optimised ramp schedule": "#059669", "SALS": "#c2410c"}
for m, c in col.items():
    x = R[R.method == m]; y = [C.net_av(x.E.values, x.g.values, x.closs.values, dict(C.BASE, p_e=q)).mean() / 1000 for q in pr]; ax.plot(pr, y, color=c, label=m)
y = [C.net_pv_plant(x0.e_pv.values, x0.copen.values, dict(C.BASE, p_e=q)).mean() / 1000 for q in pr]; ax.plot(pr, y, "--", color="k", label="PV plant replacing the crop")
ax.axhline(0, color="#9ca3af", lw=0.6); ax.set_xlabel("Electricity price (USD MWh$^{-1}$)"); ax.set_ylabel("Net value vs open field (kUSD ha$^{-1}$ yr$^{-1}$)")
ax.legend(fontsize=5.5, frameon=False); fig.tight_layout(); fig.savefig(f"{FIG}/cost_price.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/cost_price.png", dpi=200)

# ---- cost of the design margin (phenology rule, symmetric leave-one-member-out, E2 and E3)
def load(tag):
    g = pd.read_parquet(f"{RES}/grid_{tag}.parquet"); g = g[~g.method.str.match(r"sals_s\d_seed[12]")].copy(); g["method"] = np.where(g.method.str.startswith("sals_"), "SALS", g.method)
    g = g[g.method.isin(["Phenology rule"]) & (g.y_open >= 0.2)].copy(); g["r"] = g.y / g.y_open; g["erel"] = g.e / g.e_pv; return g
def feas(g, rho=RHO):
    h = g[g.year <= HIST].groupby(["method", "site", "g"]).r.mean().reset_index(); h["ok"] = (h.r >= rho).astype(float); return h[["method", "site", "g", "ok"]]
def pick(fl, q):
    f = pd.concat(fl).groupby(["method", "site", "g"]).ok.mean().reset_index(); return f[f.ok >= q - 1e-9].groupby(["method", "site"]).g.max().rename("g_sel").reset_index()
out = []
for tagp, tagx, ens in [("e2s", "e2x", "E2"), ("e3s", "e3x", "E3")]:
    acc = {}
    for sp in (0, 1, 2):
        for k in (0, 3, 5, 8, 10):
            truth = load(f"{tagp}{sp}_{k}"); mx = load(f"{tagx}{sp}_{k}"); others = [feas(load(f"{tagp}{sp}_{j}")) for j in (0, 3, 5, 8, 10) if j != k]
            for name, sel in [("oracle", pick([feas(truth)], 1.0)), ("mean response", pick([feas(mx)], 1.0)), ("floor met in 75% of other members", pick(others, 0.75)),
                              ("floor met in all other members", pick(others, 1.0)), ("floor 0.92 under mean response", pick([feas(mx, 0.92)], 1.0)),
                              ("floor 0.94 under mean response", pick([feas(mx, 0.94)], 1.0))]:
                x = apply(sel, truth); d = comp_rows(x); d["r"] = x.r.values
                d["member"] = k; d["split"] = sp; d["design"] = name; acc.setdefault(name, []).append(d)
    for name, lst in acc.items():
        d = pd.concat(lst); s = d.groupby(["design", "split", "member", "site"])[["E", "g", "closs", "r"]].mean().reset_index()
        s["ok"] = (s.r >= RHO).astype(float)
        s = s.groupby(["design", "site"])[["E", "g", "closs", "r", "ok"]].mean().reset_index()
        out.append(dict(ensemble=ens, design=name, pairs=s.site.nunique(), E=s.E.mean(), crop_loss=s.closs.mean(), pairs_ok=s.ok.mean() * 100,
                        **{f"net_{q}": C.net_av(s.E.values, s.g.values, s.closs.values, dict(C.BASE, p_e=q)).mean() for q in (60, 90, 120)}))
D = pd.DataFrame(out); D.to_csv(f"{RES}/design_cost.csv", index=False); print(D.round(1).to_string())
