"""Evaluate SALS and baselines on geographically held-out sites and held-out years.

Methods
-------
Open field          no panels (retention 1, no electricity)
PV plant            conventional ground-mounted PV (GCR 0.4, backtracking), no crop
AV static (rule)    backtracking trackers; one GCR per crop, the largest grid value
                    whose mean retention on the training seasons is >= rho
AV static (oracle)  backtracking; per-site largest GCR meeting rho on its test seasons
Seasonal sharing    light-sharing rotation during the whole cropping season, per-site
                    oracle GCR
Stress rule         light sharing during the main growth phase except on heat-stress
                    or drought days (then track to shade the crop), per-site oracle GCR
SALS                learned design (GCR) and learned daily control (ours)
Oracle open-loop    per-season GCR and daily control optimised with perfect foresight
"""
import os, sys, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A
import simulate as S
from run_train import get_refs, train_mask

ROOT = A.ROOT
torch.set_num_threads(4)
G_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def u_season(f):
    return f[:, 8]


def u_stress(f):
    tt, arid, heat, active = f[:, 0], f[:, 3], f[:, 4], f[:, 8]
    return active * ((tt > 0.1) & (tt < 0.9) & (heat < 0) & (arid < 0.5)).float()


def load_model(name):
    m = torch.load(f"{ROOT}/results/models/{name}.pt")
    ctrl, des = A.Controller(), A.Designer()
    ctrl.load_state_dict(m["ctrl"]); des.load_state_dict(m["des"]); ctrl.mask = m["mask"]
    ctrl.eval(); des.eval()
    return ctrl, des, m["cfg"]


def rows_for(method, scen, d, idx, g, y, e, refs):
    return pd.DataFrame(dict(method=method, scenario=scen, site=d["site"][idx].numpy().astype(int),
                             year=d["year"][idx].numpy().astype(int), crop=d["crop"][idx].numpy().astype(int),
                             g=g.numpy(), y=y.numpy(), y_open=refs[0][idx].numpy(), e=e.numpy(),
                             e_pv=refs[1][idx].numpy()))


def per_site_oracle_g(res_by_g, sites, rho):
    """Largest grid GCR whose mean retention over a site's seasons is >= rho."""
    best = {}
    for s in np.unique(sites):
        m = sites == s
        ok = [g for g, (ret, _) in res_by_g.items() if ret[m].mean() >= rho]
        best[s] = max(ok) if ok else min(res_by_g)
    return best


def oracle_openloop(d, refs_s, rho=0.9, steps=150, mu=30.0, lr=0.1):
    """Perfect-foresight optimisation of a per-season GCR and daily light sharing."""
    B = d["lat"].shape[0]
    zu = torch.zeros(B, 365, requires_grad=True)
    zg = torch.zeros(B, requires_grad=True)
    opt = torch.optim.Adam([zu, zg], lr=lr)
    dd, irr, w = A.expand_irrigation(d)
    for it in range(steps):
        g = A.G_MIN + (A.G_MAX - A.G_MIN) * torch.sigmoid(zg)
        u = torch.sigmoid(zu)
        y, e = S.rollout(dd, torch.cat([g, g]), u_seq=torch.cat([u, u]), irrigated=irr)
        ret = A.combine(y, w) / refs_s[0].clamp(min=0.05)
        loss = A.loss_fn(ret, e[:B] / refs_s[1], rho, mu)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        g = A.G_MIN + (A.G_MAX - A.G_MIN) * torch.sigmoid(zg)
        y, e = S.rollout(dd, torch.cat([g, g]), u_seq=torch.cat([torch.sigmoid(zu)] * 2), irrigated=irr)
    return g, A.combine(y, w), e[:B]


def main(models, rho=0.9, do_oracle=True, tag="main"):
    D = A.load_all()
    refs = get_refs(D)
    test_site = A.site_split()
    tm = train_mask(D, test_site)
    out = []
    # --- design rule for static AV, calibrated on training seasons (baseline climate)
    d0 = D["baseline"]; tr = torch.where(tm["baseline"])[0]
    dtr = S._sel(d0, tr)
    rule_g = {}
    ret_tr = {}
    for g in G_GRID:
        y, _ = A.evaluate(dtr, torch.full((len(tr),), g))
        ret_tr[g] = (y / refs["baseline"][0][tr].clamp(min=0.05)).numpy()
    crop_tr = dtr["crop"].numpy()
    for c in range(5):
        ok = [g for g in G_GRID if ret_tr[g][crop_tr == c].mean() >= rho]
        rule_g[c] = max(ok) if ok else G_GRID[0]
    print("static rule GCR per crop", rule_g, flush=True)
    for scen in A.SCEN:
        d = D[scen]
        site = d["site"].numpy().astype(int)
        te = torch.tensor(test_site[site] & np.isin(d["year"].numpy(), A.TEST_YEARS))
        idx = torch.where(te)[0]
        dt = S._sel(d, idx)
        rs = (refs[scen][0][idx], refs[scen][1][idx])
        B = len(idx)
        sites = dt["site"].numpy().astype(int)
        # static AV with the crop rule
        g = torch.tensor([rule_g[int(c)] for c in dt["crop"]], dtype=torch.float32)
        y, e = A.evaluate(dt, g)
        out.append(rows_for("AV static (rule)", scen, d, idx, g, y, e, refs[scen]))
        # grids for the oracle-GCR baselines
        for name, rule in [("AV static (oracle GCR)", None), ("Seasonal sharing", u_season), ("Stress rule", u_stress)]:
            res = {}
            for gv in G_GRID:
                y, e = A.evaluate(dt, torch.full((B,), gv), u_rule=rule)
                res[gv] = ((y / rs[0].clamp(min=0.05)).numpy(), (y, e))
            best = per_site_oracle_g({k: (v[0], None) for k, v in res.items()}, sites, rho)
            gsel = torch.tensor([best[s] for s in sites], dtype=torch.float32)
            y = torch.stack([res[best[s]][1][0][i] for i, s in enumerate(sites)])
            e = torch.stack([res[best[s]][1][1][i] for i, s in enumerate(sites)])
            out.append(rows_for(name, scen, d, idx, gsel, y, e, refs[scen]))
            print(scen, name, "done", flush=True)
        # learned models
        feats = A.site_features(dt)
        for mname, label in models.items():
            ctrl, des, cfg = load_model(mname)
            with torch.no_grad():
                g = des(feats) if cfg["use_design"] else torch.full((B,), cfg["fixed_g"] or 0.3)
            y, e = A.evaluate(dt, g, controller=A.wrap(ctrl))
            out.append(rows_for(label, scen, d, idx, g, y, e, refs[scen]))
            print(scen, label, "done", flush=True)
        if do_oracle and scen in ("baseline", "+3C"):
            g, y, e = oracle_openloop(dt, rs, rho)
            out.append(rows_for("Open-loop (foresight)", scen, d, idx, g.detach(), y.detach(), e.detach(), refs[scen]))
            print(scen, "oracle done", flush=True)
        pd.concat(out).to_csv(f"{ROOT}/results/eval_{tag}.csv", index=False)
    json.dump({int(k): v for k, v in rule_g.items()}, open(f"{ROOT}/results/rule_g_{tag}.json", "w"))


if __name__ == "__main__":
    models = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"sals_main": "SALS (ours)"}
    tag = sys.argv[2] if len(sys.argv) > 2 else "main"
    oracle = (sys.argv[3] == "1") if len(sys.argv) > 3 else True
    rho = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
    main(models, rho=rho, do_oracle=oracle, tag=tag)
