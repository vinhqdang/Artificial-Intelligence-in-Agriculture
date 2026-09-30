"""Benchmark the PV model against PVGIS 5.2 (horizontal N-S single-axis tracker, 'inclined_axis' with 0 deg slope).

The simulator's specific yield (kWh per kWp) is computed for an isolated tracker
row (GCR 0.05, negligible row-to-row shading) and averaged over all seasons of a
site; PVGIS returns the long-term mean for a 1-kWp system with 14% losses."""
import os, sys, json, time, urllib.request
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A
import simulate as S

ROOT = A.ROOT
URL = ("https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?lat={lat}&lon={lon}&peakpower=1&loss=14"
       "&inclined_axis=1&inclinedaxisangle=0&outputformat=json")   # horizontal N-S axis

if __name__ == "__main__":
    torch.set_num_threads(2)
    sites = pd.read_csv(f"{ROOT}/data/sites.csv")
    cells = sites[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    pick = cells.sample(30, random_state=0)
    d = torch.load(f"{ROOT}/data/cache/seasons_baseline.pt")
    rows = []
    for _, c in pick.iterrows():
        sidx = sites.index[(sites.lat == c.lat) & (sites.lon == c.lon)][0]
        idx = torch.where(d["site"] == sidx)[0]
        if len(idx) == 0:
            continue
        with torch.no_grad():
            _, e = S.rollout(S._sel(d, idx), torch.full((len(idx),), 0.05))
        sim = float(e.mean()) * 1000 / (2.1 * 0.05 * 1000)   # MWh/ha -> kWh per kWp
        try:
            j = json.load(urllib.request.urlopen(URL.format(lat=c.lat, lon=c.lon), timeout=60))
            pv = j["outputs"]["totals"]["inclined_axis"]["E_y"]
            db = j["inputs"]["meteo_data"]["radiation_db"]
        except Exception as ex:
            pv, db = np.nan, str(ex)[:40]
        rows.append(dict(lat=c.lat, lon=c.lon, sim_kWh_kWp=sim, pvgis_kWh_kWp=pv, pvgis_db=db))
        print(rows[-1], flush=True); time.sleep(1)
    df = pd.DataFrame(rows); df.to_csv(f"{ROOT}/results/pv_validation.csv", index=False)
    ok = df.dropna()
    r = np.corrcoef(ok.sim_kWh_kWp, ok.pvgis_kWh_kWp)[0, 1]
    bias = (ok.sim_kWh_kWp / ok.pvgis_kWh_kWp - 1).mean() * 100
    print(f"n={len(ok)} r={r:.3f} mean bias={bias:+.1f}% MAPE={(abs(ok.sim_kWh_kWp/ok.pvgis_kWh_kWp-1)).mean()*100:.1f}%")
