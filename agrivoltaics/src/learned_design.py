"""Learned compliance-aware design: a small network maps the history retention curves of the other ensemble members (mean, min, max over members at every
grid density), the crop and the latitude to a choice among the grid densities (or no array), trained to maximise the land-equivalent ratio minus a penalty for
missing the retention floor under the held-out member (loss = sum_g p_g (-LER_g + lambda * 1[retention_g < 0.9])), with leave-one-split-out evaluation.
Compared with the margin rules and the conformal margins on the same units. Output: results/learned_design_*.csv"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from analysis import RES, RHO, HIST, ROOT
torch.set_num_threads(2)
MEM = [0, 3, 5, 8, 10]; G = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]; NG = len(G)
CROP = ["wheat", "rice", "maize", "soybean", "potato"]
sites = pd.read_csv(f"{ROOT}/data/sites.csv")


def tables(tag):
    g = pd.read_parquet(f"{RES}/grid_{tag}.parquet"); g = g[(g.method == "Phenology rule") & (g.y_open >= 0.2)].copy()
    g["r"] = g.y / g.y_open; g["ler"] = g.r + g.e / g.e_pv
    h = g[g.year <= HIST].groupby(["site", "g"]).r.mean().unstack()
    t = g[g.year >= 2015].groupby(["site", "g"]).agg(r=("r", "mean"), ler=("ler", "mean")); crop = g.groupby("site").crop.first()
    return h, t.r.unstack(), t.ler.unstack(), crop


def build(sp):
    """returns per member k: history curves, test retention, test LER (sites x grid), crop, site ids, for the sites present in all members"""
    T = {k: tables(f"e2s{sp}_{k}") for k in MEM}
    ids = sorted(set.intersection(*[set(T[k][0].index) & set(T[k][1].index) for k in MEM]))
    ids = [i for i in ids if all(T[k][0].loc[i].notna().all() and T[k][1].loc[i].notna().all() for k in MEM)]
    return {k: (T[k][0].loc[ids].values, T[k][1].loc[ids].values, T[k][2].loc[ids].values) for k in MEM}, T[MEM[0]][3].loc[ids].values.astype(int), np.array(ids)


def feats(curves, crop, ids):
    st = np.stack(curves)                                        # members x sites x grid
    lat = sites.set_index(sites.index).loc[ids, "lat"].values if len(sites) > ids.max() else np.zeros(len(ids))
    return np.concatenate([st.mean(0), st.min(0), st.max(0), np.eye(5)[crop], np.abs(lat)[:, None] / 60.0], 1).astype(np.float32)


def outcome_tables(rt, lt):
    ok = np.concatenate([(rt >= RHO).astype(np.float32), np.ones((len(rt), 1), np.float32)], 1)       # last column: no array
    ler = np.concatenate([lt, np.ones((len(lt), 1))], 1).astype(np.float32); return ok, ler


units = []
for sp in (0, 1, 2):
    D, crop, ids = build(sp)
    for k in MEM:
        others = [D[j][0] for j in MEM if j != k]
        ok, ler = outcome_tables(D[k][1], D[k][2])
        units.append(dict(split=sp, member=k, kind="in", X=feats(others, crop, ids), ok=ok, ler=ler, site=ids, crop=crop))
    mem_all = [D[j][0] for j in MEM]
    for v in ["conservative", "calibrated", "field", "harsh"]:
        _, rt, lt, c2 = tables(f"nt{sp}_{v}")
        rt, lt = rt.reindex(ids), lt.reindex(ids); good = rt.notna().all(1).values & lt.notna().all(1).values
        ok, ler = outcome_tables(rt.values[good], lt.values[good])
        units.append(dict(split=sp, member=v, kind="out", X=feats(mem_all, crop, ids)[good], ok=ok, ler=ler, site=ids[good], crop=crop[good]))


def pool(sel):
    return {k: np.concatenate([u[k] for u in sel]) for k in ("X", "ok", "ler")}


class Net(torch.nn.Module):
    def __init__(self, n):
        super().__init__(); self.f = torch.nn.Sequential(torch.nn.Linear(n, 32), torch.nn.Tanh(), torch.nn.Linear(32, NG + 1))
    def forward(self, x): return self.f(x)


def train(P, lam, seed, epochs=400):
    torch.manual_seed(seed); X = torch.tensor(P["X"]); mu, sd = X.mean(0), X.std(0) + 1e-3; X = (X - mu) / sd
    cost = torch.tensor(-P["ler"] + lam * (1 - P["ok"])); net = Net(X.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-2, weight_decay=1e-3)
    for _ in range(epochs):
        p = torch.softmax(net(X), 1); loss = (p * cost).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return net, mu, sd


def decide(net, mu, sd, X):
    with torch.no_grad(): return net(((torch.tensor(X) - mu) / sd)).argmax(1).numpy()


rows = []; LAMS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0]
for lam in LAMS:
    for te in (0, 1, 2):
        test_ids = set(np.concatenate([u["site"] for u in units if u["split"] == te and u["kind"] == "in"]).tolist())
        def drop(u):   # pairs that are held out in the test split are removed from the training set (a pair can be a test pair in several splits)
            keep = np.array([int(i) not in test_ids for i in u["site"]]); return dict(X=u["X"][keep], ok=u["ok"][keep], ler=u["ler"][keep])
        tr = pool([drop(u) for u in units if u["kind"] == "in" and u["split"] != te])
        nets = [train(tr, lam, s) for s in range(3)]
        for u in [u for u in units if u["split"] == te]:
            for s, (net, mu, sd) in enumerate(nets):
                a = decide(net, mu, sd, u["X"]); ix = np.arange(len(a))
                for i in ix: rows.append(dict(lam=lam, seed=s, split=te, kind=u["kind"], member=str(u["member"]), site=int(u["site"][i]), crop=CROP[int(u["crop"][i])], ok=u["ok"][i, a[i]], ler=u["ler"][i, a[i]], g=(G + [0.0])[a[i]]))
R = pd.DataFrame(rows); R.to_parquet(f"{RES}/learned_design_units.parquet")
S = R.groupby(["lam", "kind"]).agg(compliance=("ok", "mean"), ler=("ler", "mean"), g=("g", "mean")).reset_index(); S["compliance"] *= 100
S.to_csv(f"{RES}/learned_design_summary.csv", index=False)
C = pd.read_parquet(f"{RES}/conformal_units.parquet")
F = C.groupby(["kind", "d"]).agg(compliance=("ok", "mean"), ler=("ler", "mean")).reset_index(); F["compliance"] *= 100; F.to_csv(f"{RES}/learned_design_margin_frontier.csv", index=False)
pd.set_option("display.width", 200); print(S.round(3).to_string()); print(F[F.kind == "in-ensemble"].round(3).to_string())
