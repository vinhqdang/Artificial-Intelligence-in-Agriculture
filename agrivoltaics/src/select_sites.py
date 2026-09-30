"""Select representative cropland sites for five staple crops.

Harvested areas come from SPAM 2010 v2r0 aggregated to 1-degree cells.
For every crop, N_SITES cells with at least MIN_HA hectares of the crop are
sampled without replacement with probability proportional to harvested area,
so that the sample represents where the crop is actually grown. Sowing and
maturity dates come from the GGCMI Phase 3 crop calendar (rainfed).
"""
import os
import numpy as np, pandas as pd, xarray as xr

DATA = os.environ.get("DATA_DIR", "/home/user/data")
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
N_SITES, MIN_HA, SEED = 80, 5000, 0
CROPS = {"maize": ("maiz", ["mai_rf"]), "wheat": ("whea", ["wwh_rf", "swh_rf"]),
         "rice": ("rice", ["ri1_rf"]), "soybean": ("soyb", ["soy_rf"]), "potato": ("pota", ["pot_rf"])}


def calendar(code, lat, lon):
    """Planting and maturity day of the nearest 0.5-degree calendar cell; coastal
    cells without data take the closest valid cell within 2 degrees."""
    d = xr.open_dataset(f"{DATA}/ggcmi/{code}.nc4")
    P, M = d["planting_day"].values, d["maturity_day"].values
    la, lo = d["lat"].values, d["lon"].values
    ok = np.isfinite(P) & np.isfinite(M)
    vi, vj = np.where(ok)
    p_out, m_out = [], []
    for y, x in zip(lat, lon):
        dist = (la[vi] - y) ** 2 + (lo[vj] - x) ** 2
        k = int(np.argmin(dist))
        if dist[k] > 4.0:
            p_out.append(np.nan); m_out.append(np.nan); continue
        p_out.append(P[vi[k], vj[k]]); m_out.append(M[vi[k], vj[k]])
    return np.array(p_out), np.array(m_out)


if __name__ == "__main__":
    g = pd.read_csv(f"{DATA}/spam/spam_1deg.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for crop, (sp, cal) in CROPS.items():
        c = g[g[f"{sp}_a"] >= MIN_HA].reset_index(drop=True)
        w = c[f"{sp}_a"].values / c[f"{sp}_a"].sum()
        idx = rng.choice(len(c), size=min(N_SITES, len(c)), replace=False, p=w)
        s = c.loc[idx].copy()
        s["crop"] = crop
        s["area_ha"] = s[f"{sp}_a"]
        s["irr_frac"] = (s[f"{sp}_i"] / s[f"{sp}_a"]).clip(0, 1)
        s["share_of_global_area"] = s["area_ha"] / g[f"{sp}_a"].sum()
        for code in cal:
            p, m = calendar(code, s.lat1.values + 0.25, s.lon1.values + 0.25)
            s[f"sow_{code}"] = p; s[f"mat_{code}"] = m
        rows.append(s[["crop", "lat1", "lon1", "name_cntr", "area_ha", "irr_frac", "share_of_global_area"] +
                      [k for k in s.columns if k.startswith(("sow_", "mat_"))]])
    sites = pd.concat(rows, ignore_index=True).rename(columns={"lat1": "lat", "lon1": "lon", "name_cntr": "country"})
    os.makedirs(OUT, exist_ok=True)
    sites.to_csv(f"{OUT}/sites.csv", index=False)
    print(sites.groupby("crop").agg(n=("lat", "size"), area_Mha=("area_ha", lambda a: a.sum() / 1e6),
                                    share=("share_of_global_area", "sum"), irr=("irr_frac", "mean")))
    print("unique cells", sites[["lat", "lon"]].drop_duplicates().shape[0])
    print(sites.country.value_counts().head(15).to_dict())
