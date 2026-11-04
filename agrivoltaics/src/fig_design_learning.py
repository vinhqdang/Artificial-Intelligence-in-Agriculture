"""Figure and table for the design margin: fixed margins, conformal margins, the learned design network and the oracle (pair level, in-ensemble and outside)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), ".."); RES = f"{ROOT}/results"; FIG = f"{ROOT}/manuscript/figures"; TAB = f"{ROOT}/manuscript/tables"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
F = pd.read_csv(f"{RES}/learned_design_margin_frontier.csv"); L = pd.read_csv(f"{RES}/learned_design_summary.csv"); C = pd.read_csv(f"{RES}/conformal_summary.csv")
O = pd.read_csv(f"{RES}/conformal_oracle.csv").set_index("kind"); K = pd.read_csv(f"{RES}/learned_design_compare.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, kind, ttl in ((axs[0], "in-ensemble", "True response inside the ensemble"), (axs[1], "out-of-ensemble", "True response outside the ensemble")):
    f = F[F.kind == kind].sort_values("d"); ax.plot(f.compliance, f.ler, "-", color="#6b7280", lw=1.2, label="fixed margin, 0.90 to 0.975")
    l = L[(L.kind == kind) & (L.lam >= 0.6)].sort_values("lam"); ax.plot(l.compliance, l.ler, "o-", color="#c2410c", ms=3, lw=0.9, label="learned design network ($\\lambda$ 0.6 to 3)")
    c = C[(C.kind == kind) & (C.rule == "conformal margin")]; ax.plot(c.compliance, c.ler, "s", color="#1d4ed8", ms=4.5, label="conformal margin (targets 90, 85, 80%)")
    ax.plot(O.loc[kind].compliance, O.loc[kind].ler, "*", color="black", ms=8, label="oracle (true response)")
    ax.set_xlabel("Pairs at the floor (%)"); ax.set_ylabel("Land-equivalent ratio"); ax.set_title(ttl, fontsize=7.5)
h, l_ = axs[0].get_legend_handles_labels(); fig.legend(h, l_, loc="lower center", ncol=2, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(f"{FIG}/design_learning.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/design_learning.png", dpi=200)


def pick(rule, alpha, kind):
    x = C[(C.rule == rule) & (C.kind == kind)]
    x = x[x.alpha.isna()] if np.isnan(alpha) else x[np.isclose(x.alpha, alpha)]
    return x.iloc[0]


def crops(r): v = [r.ok_wheat, r.ok_rice, r.ok_maize, r.ok_soybean, r.ok_potato]; return f"{np.nanmin(v):.0f}--{np.nanmax(v):.0f}"


lines = [f"Oracle (design under the true response) & {O.loc['in-ensemble'].compliance:.0f} & {O.loc['in-ensemble'].ler:.3f} & -- & {O.loc['out-of-ensemble'].compliance:.0f} & {O.loc['out-of-ensemble'].ler:.3f}\\\\", "\\addlinespace"]
for name, key in (("Design under the mean response", "mean response (d = 0.90)"), ("Fixed margin 0.92", "fixed margin 0.92"), ("Fixed margin 0.94", "fixed margin 0.94"), ("Fixed margin 0.96", "fixed margin 0.96")):
    i, o = pick(key, np.nan, "in-ensemble"), pick(key, np.nan, "out-of-ensemble")
    lines.append(f"{name} & {i.compliance:.0f} & {i.ler:.3f} & {crops(i)} & {o.compliance:.0f} & {o.ler:.3f}\\\\")
lines.append("\\addlinespace")
for al in (0.10, 0.15, 0.20):
    i, o = pick("conformal margin", al, "in-ensemble"), pick("conformal margin", al, "out-of-ensemble")
    lines.append(f"Conformal margin, target {100 * (1 - al):.0f}\\% & {i.compliance:.0f} & {i.ler:.3f} & {crops(i)} & {o.compliance:.0f} & {o.ler:.3f}\\\\")
lines.append("\\addlinespace")
for lam in (0.8, 1.0, 1.5, 2.0):
    i = L[(L.kind == "in-ensemble") & np.isclose(L.lam, lam)].iloc[0]; o = L[(L.kind == "out-of-ensemble") & np.isclose(L.lam, lam)].iloc[0]
    k = K[(K.kind == "in-ensemble") & np.isclose(K.lam, lam)].iloc[0]
    lines.append(f"Learned design network, $\\lambda={lam}$ & {i.compliance:.0f} & {i.ler:.3f} & -- & {o.compliance:.0f} & {o.ler:.3f}\\\\")
tex = ("\\begin{table}[t]\n\\centering\\small\n\\caption{Design rules for the density under shade-response uncertainty (ensemble E2, phenology rule, 76 distinct crop--site pairs): share of pairs (\\%) at the floor (mean retention $\\ge0.9$, averaged over the held-out members or responses) and mean LER, when the true response is a held-out member (inside) or one of four fixed responses (outside the belief of the ensemble); the third column gives the range over crops of the inside share. Conformal margins and the network are fitted on the pairs of two splits that are not in the third and evaluated on the third (leave-one-split-out); fixed margins and the oracle are not fitted.}\\label{tab:learndesign}\n"
       "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{\\begin{tabular}{lccccc}\n\\toprule\n & \\multicolumn{3}{c}{Inside the ensemble} & \\multicolumn{2}{c}{Outside}\\\\\nRule & Pairs at floor & LER & By crop & Pairs at floor & LER\\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
open(f"{TAB}/learndesign.tex", "w").write(tex); print(tex[tex.index("\\midrule"):])
