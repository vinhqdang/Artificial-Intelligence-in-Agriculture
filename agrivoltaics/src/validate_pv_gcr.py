"""Benchmark the row-to-row shading and diffuse-transposition part of the PV model against pvlib (infinite sheds, isotropic sky) at
ground coverage ratios of 0.05 to 0.5. Both models use the same hourly irradiance and sun position (taken from the simulator's cache) and
backtracking trackers; the quantity compared is the plane-of-array irradiance on the modules, averaged over the year and, for the ratio to the
sparse-array value, relative to GCR 0.05. Output: results/pv_validation_gcr.csv"""
import os, sys, math
import numpy as np, pandas as pd, torch
import pvlib
from pvlib import tracking
from pvlib.bifacial import infinite_sheds
sys.path.insert(0, os.path.dirname(__file__))
import sals as A, physics as P
ROOT = A.ROOT
d = torch.load(f"{ROOT}/data/cache/seasons_baseline.pt")
sites = pd.read_csv(f"{ROOT}/data/sites.csv")
rng = np.random.default_rng(0)
sel = []
for s in sorted(np.unique(d["site"].numpy())):
    r = int(torch.where(d["site"] == s)[0][0]); sel.append((s, float(d["lat"][r]), r))
sel = sorted(sel, key=lambda x: x[1]); pick = [sel[int(i)] for i in np.linspace(0, len(sel) - 1, 12)]
G = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]; rows = []
for s, lat, r in pick:
    ghi, dhi, bh = [d[k][r].float().reshape(-1) for k in ("ghi_h", "dhi_h", "bh_h")]
    sx, sz, psi = [d[k][r].float().reshape(-1) for k in ("sx", "sz", "psi")]
    cosz = torch.clamp(sz, min=0.0)
    dni = torch.where(cosz > 0.05, bh / torch.clamp(cosz, min=0.05), torch.zeros_like(bh))
    up = (cosz > 0.05) & (ghi > 1.0)
    zen = np.degrees(np.arccos(cosz.numpy()))
    # sun vector: x east, y north (az = atan2(x, y), degrees from north); sy is not cached, so rebuild it from the declination
    doy = np.repeat(d["doy"][r].numpy(), 24); lat_r = math.radians(lat)
    decl = 23.45 * math.pi / 180 * np.sin(2 * math.pi * (284 + doy) / 365.0); om = (np.tile(P.HOURS.numpy(), 365) - 12.0) * 15 * math.pi / 180
    sy = math.cos(lat_r) * np.sin(decl) - math.sin(lat_r) * np.cos(decl) * np.cos(om)
    az = (np.degrees(np.arctan2(sx.numpy(), sy)) + 360) % 360
    m = up.numpy()
    for g in G:
        phi_bt = P.backtracking_angle(psi, torch.full_like(psi, g))
        f_ps = P.panel_shaded_fraction(phi_bt, psi, g)
        cos_aoi = torch.clamp(sx * torch.sin(phi_bt) + sz * torch.cos(phi_bt), min=0.0)
        poa_sim = dni * cos_aoi * (1 - f_ps) + dhi * (1 + torch.cos(phi_bt)) / 2 + P.ALBEDO * ghi * (1 - torch.cos(phi_bt)) / 2
        tr = tracking.singleaxis(pd.Series(zen[m]), pd.Series(az[m]), axis_tilt=0, axis_azimuth=0, max_angle=60, backtrack=True, gcr=g)
        tilt = tr["surface_tilt"].fillna(0).values; saz = tr["surface_azimuth"].fillna(90).values
        out = infinite_sheds.get_irradiance_poa(tilt, saz, zen[m], az[m], g, P.HUB_HEIGHT, 1.0 / g, ghi.numpy()[m], dhi.numpy()[m], dni.numpy()[m], P.ALBEDO, model="isotropic", npoints=40)
        rows.append(dict(site=s, lat=lat, gcr=g, poa_sim=float(poa_sim.numpy()[m].sum()) / 1000, poa_pvlib=float(np.asarray(out["poa_global"]).sum()) / 1000))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
b = df[df.gcr == 0.05].set_index("site")
df["rel_sim"] = df.poa_sim / df.site.map(b.poa_sim); df["rel_pvlib"] = df.poa_pvlib / df.site.map(b.poa_pvlib)
df["ratio"] = df.poa_sim / df.poa_pvlib
df.to_csv(f"{ROOT}/results/pv_validation_gcr.csv", index=False)
print(df.groupby("gcr")[["ratio", "rel_sim", "rel_pvlib"]].agg(["mean", "min", "max"]).round(3).to_string())
