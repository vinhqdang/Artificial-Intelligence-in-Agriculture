"""Figure and table for the learned design margin: fixed margins, conformal margins and the learned design network (in-ensemble and out-of-ensemble)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), ".."); RES = f"{ROOT}/results"; FIG = f"{ROOT}/manuscript/figures"; TAB = f"{ROOT}/manuscript/tables"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
F = pd.read_csv(f"{RES}/learned_design_margin_frontier.csv"); L = pd.read_csv(f"{RES}/learned_design_summary.csv"); C = pd.read_csv(f"{RES}/conformal_summary.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, kind, fk, lk, ttl in ((axs[0], "in-ensemble", "in-ensemble", "in", "True response inside the ensemble"), (axs[1], "out-of-ensemble", "out-of-ensemble", "out", "True response outside the ensemble")):
    f = F[F.kind == fk].sort_values("d"); ax.plot(f.compliance, f.ler, "-", color="#6b7280", lw=1.2, label="fixed margin, 0.90 to 0.975")
    l = L[(L.kind == lk) & (L.lam >= 0.6)].sort_values("lam"); ax.plot(l.compliance, l.ler, "o-", color="#c2410c", ms=3, lw=0.9, label="learned design network (lambda 0.6 to 3)")
    c = C[(C.kind == kind) & (C.rule == "global conformal")]; ax.plot(c.compliance, c.ler, "s", color="#1d4ed8", ms=4.5, label="conformal margin (targets 90, 85, 80%)")
    c2 = C[(C.kind == kind) & (C.rule == "per-crop conformal")]; ax.plot(c2.compliance, c2.ler, "^", color="#059669", ms=4.5, label="per-crop conformal margin")
    ax.set_xlabel("Pairs at the floor (%)"); ax.set_ylabel("Land-equivalent ratio"); ax.set_title(ttl, fontsize=7.5)
O_ = pd.read_csv(f"{RES}/conformal_oracle.csv").set_index("kind")
for ax, k in zip(axs, ("in-ensemble", "out-of-ensemble")): ax.plot(O_.loc[k].compliance, O_.loc[k].ler, "*", color="black", ms=8, label="oracle (true response)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(f"{FIG}/design_learning.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/design_learning.png", dpi=200)
def row(label, i, o, rngs=""):
    return f"{label} & {i.compliance:.0f} & {i.ler:.3f} & {rngs} & {o.compliance:.0f} & {o.ler:.3f}\\\\"
S = C.set_index(["rule", "alpha", "kind"], drop=False) if False else C
O = pd.read_csv(f"{RES}/conformal_oracle.csv").set_index("kind")
lines = [f"Oracle (design under the true response) & {O.loc['in-ensemble'].compliance:.0f} & {O.loc['in-ensemble'].ler:.3f} & -- & {O.loc['out-of-ensemble'].compliance:.0f} & {O.loc['out-of-ensemble'].ler:.3f}\\\\", "\\addlinespace"]
def pick(rule, alpha, kind):
    x = C[(C.rule == rule) & (C.kind == kind)]
    x = x[x.alpha.isna()] if np.isnan(alpha) else x[np.isclose(x.alpha, alpha)]
    return x.iloc[0]
for name, key, al in (("Design under the mean response", "mean response (d = 0.90)", np.nan), ("Fixed margin 0.92", "fixed margin 0.92", np.nan), ("Fixed margin 0.94", "fixed margin 0.94", np.nan), ("Fixed margin 0.96", "fixed margin 0.96", np.nan)):
    i, o = pick(key, al, "in-ensemble"), pick(key, al, "out-of-ensemble")
    r = [i.ok_wheat, i.ok_rice, i.ok_maize, i.ok_soybean, i.ok_potato]; lines.append(row(name, i, o, f"{min(r):.0f}--{max(r):.0f}"))
lines.append("\\addlinespace")
for name, key in (("Conformal margin", "global conformal"), ("Per-crop conformal margin", "per-crop conformal")):
    for al in (0.10, 0.15, 0.20):
        i, o = pick(key, al, "in-ensemble"), pick(key, al, "out-of-ensemble"); r = [i.ok_wheat, i.ok_rice, i.ok_maize, i.ok_soybean, i.ok_potato]
        lines.append(row(f"{name}, target {100 * (1 - al):.0f}\\%", i, o, f"{min(r):.0f}--{max(r):.0f}"))
lines.append("\\addlinespace")
for lam in (0.8, 1.0, 2.0):
    i = L[(L.kind == "in") & np.isclose(L.lam, lam)].iloc[0]; o = L[(L.kind == "out") & np.isclose(L.lam, lam)].iloc[0]
    lines.append(f"Learned design network, $\\lambda={lam}$ & {i.compliance:.0f} & {i.ler:.3f} & -- & {o.compliance:.0f} & {o.ler:.3f}\\\\")
tex = ("\\begin{table}[t]\n\\centering\\small\n\\caption{Design rules for the density under shade-response uncertainty (ensemble E2, phenology rule, leave-one-split-out, 76 pairs): share of pairs (\\%) with mean retention $\\ge0.9$ and mean LER when the true response is the held-out member (inside) or one of four fixed responses (outside the ensemble); the third column gives the range over crops of the inside share. Margins and the network are fitted on two splits and evaluated on the third; the fixed margins are not fitted.}\\label{tab:learndesign}\n"
       "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{\\begin{tabular}{lccccc}\n\\toprule\n & \\multicolumn{3}{c}{Inside the ensemble} & \\multicolumn{2}{c}{Outside}\\\\\nRule & Pairs at floor & LER & By crop & Pairs at floor & LER\\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
open(f"{TAB}/learndesign.tex", "w").write(tex); print(tex)
