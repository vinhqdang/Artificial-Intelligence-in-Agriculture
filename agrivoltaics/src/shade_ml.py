"""Data-driven model of the yield response to shade, learned from 428 observations of 58 studies
(Pataczek, Laub et al., Zenodo 10.5281/zenodo.5716091, CC BY 4.0).

Relative yield y(s) = exp(-beta * s**gamma) with s the relative shade reduction. Candidates:
  proportional  y = 1 - s (the response of the simulator's main analysis)
  global        one (beta, gamma) for all crops
  type          one curve per crop type, no pooling
  hier          crop-type curves with partial pooling (random effects on log beta, log gamma)
  hier+cov      hier plus covariates: irradiation, climate zone, shade type, intercropping
  gbm           gradient-boosted trees on (s, crop type, covariates)
Evaluation: leave-one-study-out cross-validation (every study held out once).
Output: results/shade_ml_cv.csv, results/shade_ml_curves.csv, results/shade_ml_coef.csv
"""
import os, sys, json
import numpy as np, pandas as pd, torch
from sklearn.ensemble import GradientBoostingRegressor
ROOT = os.path.join(os.path.dirname(__file__), "..")
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
TAU = 0.5


def load():
    d = pd.read_excel(f"{ROOT}/data/laub/Data_Meta_Analysis_updated.xlsx").iloc[:, :12]
    d.columns = ["crop", "ctype", "rsr", "yld", "exp", "zone", "lat", "lon", "ghi", "nets", "shade", "ref"]
    d["s"] = d.rsr / 100.0; d["y"] = d.yld / 100.0
    d["ghi_z"] = (d.ghi - d.ghi.mean()) / d.ghi.std()
    d["sub"] = (d.zone == "Subtropical").astype(float)
    d["panel"] = (d.shade == "Solar_panel").astype(float)
    d["inter"] = (d.exp == "Intercropping").astype(float)
    d["g"] = d.ctype.astype("category").cat.codes
    return d


class Hier(torch.nn.Module):
    def __init__(self, G, ncov, pooled, cov=True, per_type=True):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([-0.5, 0.0]))        # log beta, log gamma (global)
        self.bg = torch.nn.Parameter(torch.zeros(G, 2))
        self.delta = torch.nn.Parameter(torch.zeros(ncov)) if cov else None
        self.logsig = torch.nn.Parameter(torch.tensor(-1.5))
        self.pooled, self.per_type = pooled, per_type

    def curve(self, s, g, X=None):
        bg = self.bg[g] if self.per_type else 0.0
        lb = self.a[0] + (bg[:, 0] if self.per_type else 0.0)
        lg = self.a[1] + (bg[:, 1] if self.per_type else 0.0)
        if self.delta is not None and X is not None:
            lb = lb + X @ self.delta
        return torch.exp(-torch.exp(lb) * torch.clamp(s, min=1e-6) ** torch.exp(lg))

    def nll(self, s, y, g, X):
        mu = self.curve(s, g, X)
        t = (y - mu) / torch.exp(self.logsig)
        nu = 4.0
        ll = -torch.log1p(t ** 2 / nu) * (nu + 1) / 2 - self.logsig
        prior = 0.0
        if self.per_type:
            tau = TAU if self.pooled else 1e3
            prior = (self.bg ** 2).sum() / (2 * tau ** 2)
        prior = prior + (self.a ** 2).sum() / (2 * 2.0 ** 2)
        if self.delta is not None:
            prior = prior + (self.delta ** 2).sum() / (2 * 1.0 ** 2)
        return -ll.sum() + prior


COV = ["ghi_z", "sub", "panel", "inter"]


def fit(df, G, kind, steps=400):
    s = torch.tensor(df.s.values, dtype=torch.float32); y = torch.tensor(df.y.values, dtype=torch.float32)
    g = torch.tensor(df.g.values, dtype=torch.long); X = torch.tensor(df[COV].values, dtype=torch.float32)
    m = Hier(G, len(COV), pooled=(kind != "type"), cov=(kind == "hier+cov"), per_type=(kind in ("type", "hier", "hier+cov")))
    opt = torch.optim.LBFGS(m.parameters(), lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); l = m.nll(s, y, g, X); l.backward(); return l
    opt.step(closure)
    return m


def predict(m, df):
    with torch.no_grad():
        return m.curve(torch.tensor(df.s.values, dtype=torch.float32), torch.tensor(df.g.values, dtype=torch.long),
                       torch.tensor(df[COV].values, dtype=torch.float32)).numpy()


def gbm_feats(df, G):
    return np.column_stack([df.s.values, np.eye(G)[df.g.values], df[COV].values])


if __name__ == "__main__":
    d = load(); G = d.g.nunique()
    kinds = ["proportional", "global", "type", "hier", "hier+cov", "gbm"]
    rows = []
    studies = d.ref.unique()
    pred = {k: np.zeros(len(d)) for k in kinds}
    for j, st in enumerate(studies):
        tr, te = d[d.ref != st], d[d.ref == st]
        ii = np.where(d.ref == st)[0]
        pred["proportional"][ii] = 1 - te.s.values
        for k in ["global", "type", "hier", "hier+cov"]:
            tr2 = tr.assign(g=0) if k == "global" else tr
            te2 = te.assign(g=0) if k == "global" else te
            m = fit(tr2, G if k != "global" else 1, "hier" if k == "global" else k)
            if k == "global":      # one curve for every crop
                m.per_type = False
            pred[k][ii] = predict(m, te2)
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, loss="huber", random_state=0)
        gb.fit(gbm_feats(tr, G), tr.y.values); pred["gbm"][ii] = gb.predict(gbm_feats(te, G))
        if j % 10 == 0: print("study", j, flush=True)
    d["study"] = d.ref
    out = []
    mask = (d.s > 0)
    for k in kinds:
        e = pred[k] - d.y.values
        per_study = pd.DataFrame(dict(study=d.study, e=e, ae=np.abs(e))).groupby("study").agg(mae=("ae", "mean"), mse=("e", lambda x: (x ** 2).mean()))
        out.append(dict(model=k, rmse=float(np.sqrt((e[mask] ** 2).mean())), mae=float(np.abs(e[mask]).mean()),
                        median_ae=float(np.median(np.abs(e[mask]))), study_mae=float(per_study.mae.mean())))
        for g_, name in [(c, n) for n, c in zip(d.ctype.astype("category").cat.categories, range(G))]:
            pass
    cv = pd.DataFrame(out); cv.to_csv(f"{ROOT}/results/shade_ml_cv.csv", index=False); print(cv.round(3).to_string())
    # per crop type (for the five crops used) loss
    d["pred_hier"] = pred["hier"]; d["pred_cov"] = pred["hier+cov"]; d["pred_prop"] = pred["proportional"]; d["pred_gbm"] = pred["gbm"]
    d.to_pickle(f"{ROOT}/results/shade_ml_cvpred.pkl")
