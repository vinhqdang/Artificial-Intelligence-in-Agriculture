"""Graphical abstract (13 cm x 6.5 cm; PDF and 300 dpi PNG/TIFF)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
ROOT = os.path.join(os.path.dirname(__file__), ".."); OUT = f"{ROOT}/manuscript/figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
fig = plt.figure(figsize=(13 / 2.54, 6.5 / 2.54))
# panel 1: schematic
a = fig.add_axes([0.01, 0.12, 0.32, 0.72]); a.set_xlim(0, 10); a.set_ylim(0, 7); a.axis("off")
a.add_patch(Rectangle((0, 0), 10, 1.0, color="#a16207", alpha=0.55)); a.text(5, 0.45, "crop rows", ha="center", va="center", fontsize=7, color="white")
for x in (2.5, 7.5):
    a.plot([x - 1.0, x + 1.0], [3.2, 3.9], color="#1d4ed8", lw=4, solid_capstyle="butt"); a.plot([x, x], [1.0, 3.5], color="#6b7280", lw=1.5)
for x in (1.5, 4.5, 6.5, 9.0):
    a.add_patch(Rectangle((x - 0.35, 1.0), 0.7, 0.9, color="#16a34a"))
a.annotate("", xy=(2.0, 4.2), xytext=(0.3, 6.4), arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.5)); a.text(0.2, 6.6, "sun", color="#b45309", fontsize=7)
a.text(5, 5.4, "daily light-sharing\nlevel u per tracker", ha="center", fontsize=6.5, fontweight="bold")
fig.text(0.17, 0.90, "Trackers share light", ha="center", fontsize=7, fontweight="bold")
# panel 2: LER
b = fig.add_axes([0.40, 0.22, 0.25, 0.58]); names = ["static", "rule", "SALS"]; vals = [1.07, 1.24, 1.27]
b.bar(names, vals, color=["#9ca3af", "#60a5fa", "#1d4ed8"], width=0.65); b.set_ylim(1.0, 1.32); b.set_ylabel("Land-equivalent ratio", fontsize=7)
for i, v in enumerate(vals): b.text(i, v + 0.004, f"{v:.2f}", ha="center", fontsize=6.5)
b.spines[["top", "right"]].set_visible(False); b.tick_params(labelsize=7)
fig.text(0.51, 0.90, "Sharing raises LER", ha="center", fontsize=7, fontweight="bold")
fig.text(0.51, 0.03, "learning adds 0.007 LER", ha="center", fontsize=5.8, style="italic")
# panel 3: compliance inside / outside the ensemble
c = fig.add_axes([0.765, 0.22, 0.225, 0.58]); labs = ["mean\nresponse", "conformal\nmargin", "margin\n0.96"]; ins = [70, 94, 99]; out = [27, 37, 73]; x = np.arange(3); w = 0.36
c.bar(x - w / 2, ins, w, color="#c2410c", label="inside ensemble"); c.bar(x + w / 2, out, w, color="#fca5a5", label="outside ensemble")
for xi, v in zip(x - w / 2, ins): c.text(xi, v + 2, f"{v}", ha="center", fontsize=6)
for xi, v in zip(x + w / 2, out): c.text(xi, v + 2, f"{v}", ha="center", fontsize=6)
c.set_xticks(x); c.set_xticklabels(labs, fontsize=5.5); c.set_ylim(0, 128); c.set_yticks([0, 20, 40, 60, 80, 100]); c.set_ylabel("Pairs at floor (%)", fontsize=6.5); c.tick_params(axis="y", labelsize=6)
c.legend(fontsize=5.2, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, 1.12), ncol=1); c.spines[["top", "right"]].set_visible(False)
fig.text(0.86, 0.90, "Shade response limits it", ha="center", fontsize=7, fontweight="bold")
fig.text(0.87, 0.03, "safe only inside the ensemble", ha="center", fontsize=5.8, style="italic")
fig.savefig(f"{OUT}/graphical_abstract.pdf"); fig.savefig(f"{OUT}/graphical_abstract.png", dpi=300); fig.savefig(f"{OUT}/graphical_abstract.tiff", dpi=300)
print(plt.imread(f"{OUT}/graphical_abstract.png").shape)
