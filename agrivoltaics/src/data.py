"""Build season tensors (one 365-day year starting at sowing) for every site.

Wheat uses the GGCMI winter-wheat calendar unless the coldest-month mean
temperature is below -8 degC, in which case the spring-wheat calendar is used.
Thermal-time requirements are calibrated per site: T_sum is the mean thermal time
accumulated between the GGCMI sowing and maturity dates in 2001-2019, and the
canopy parameters I50A and I50B are scaled by T_sum / T_sum(reference cultivar),
as is common practice in gridded crop modelling.
"""
import os
import numpy as np, pandas as pd, torch
import physics as P
import simulate as S

WEATHER = os.environ.get("WEATHER_DIR", "/home/user/data/power")
ROOT = os.path.join(os.path.dirname(__file__), "..")
CROPS = list(P.CROP_PARAMS)
YEARS = list(range(2001, 2020))
L = 365
WARMING = {"baseline": (0.0, 390.0), "+1.5C": (1.5, 450.0), "+2C": (2.0, 500.0), "+3C": (3.0, 600.0)}


def load_weather(lat, lon):
    df = pd.read_csv(f"{WEATHER}/{lat:+.2f}_{lon:+.2f}.csv", index_col=0, parse_dates=True)
    return df


def build(warming="baseline"):
    dT, co2 = WARMING[warming]
    sites = pd.read_csv(f"{ROOT}/data/sites.csv")
    seq = {k: [] for k in ["ghi", "dhi", "tmax", "tmin", "tdew", "rain", "u2", "doy", "active"]}
    stat = {k: [] for k in ["site", "year", "lat", "elev", "crop", "irr_frac"] + P.PARAM_NAMES}
    for si, s in sites.iterrows():
        df = load_weather(s.lat, s.lon)
        if s.crop == "wheat":
            cold = df.T2M.groupby(df.index.month).mean().min()
            code = "swh_rf" if cold < -8 else "wwh_rf"
        else:
            code = {"maize": "mai_rf", "rice": "ri1_rf", "soybean": "soy_rf", "potato": "pot_rf"}[s.crop]
        sow, mat = int(s[f"sow_{code}"]), int(s[f"mat_{code}"])
        glen = (mat - sow) % 365
        pars = dict(zip(P.PARAM_NAMES, P.CROP_PARAMS[s.crop]))
        # warming changes temperature; relative humidity is kept constant
        tmax_all, tmin_all = df.T2M_MAX + dT, df.T2M_MIN + dT
        tdew_all = df.T2MDEW + dT
        tt_hist = []
        rows = []
        for y in YEARS:
            start = pd.Timestamp(y, 1, 1) + pd.Timedelta(days=sow - 1)
            w = df.loc[start:start + pd.Timedelta(days=L - 1)]
            if len(w) < L:
                continue
            tmean_hist = ((df.T2M_MAX + df.T2M_MIN) / 2).loc[w.index][:glen]
            tt_hist.append(np.clip(tmean_hist - pars["Tbase"], 0, None).sum())
            rows.append((y, w))
        tsum = float(np.mean(tt_hist))
        scale = tsum / pars["Tsum"]
        for y, w in rows:
            idx = w.index
            seq["ghi"].append(w.ALLSKY_SFC_SW_DWN.values)
            seq["dhi"].append(np.minimum(w.ALLSKY_SFC_SW_DIFF.values, w.ALLSKY_SFC_SW_DWN.values))
            seq["tmax"].append(tmax_all.loc[idx].values)
            seq["tmin"].append(tmin_all.loc[idx].values)
            seq["tdew"].append(tdew_all.loc[idx].values)
            seq["rain"].append(w.PRECTOTCORR.values)
            seq["u2"].append(w.WS2M.values)
            seq["doy"].append(idx.dayofyear.values)
            # crop in the field from sowing until the calendar maturity date + 45 days
            seq["active"].append((np.arange(L) < min(glen + 45, L)).astype(np.float32))
            stat["site"].append(si); stat["year"].append(y); stat["lat"].append(s.lat)
            stat["elev"].append(float(w.elevation.iloc[0])); stat["crop"].append(CROPS.index(s.crop))
            stat["irr_frac"].append(s.irr_frac)
            for k in P.PARAM_NAMES:
                v = pars[k]
                if k == "Tsum":
                    v = tsum
                elif k in ("I50A", "I50B"):
                    v = v * scale
                elif k == "root":
                    v = min(v, 1000.0)
                stat[k].append(v)
    out = {k: torch.tensor(np.stack(v), dtype=torch.float32) for k, v in seq.items()}
    out.update({k: torch.tensor(np.array(v), dtype=torch.float32) for k, v in stat.items()})
    out["co2"] = torch.full_like(out["lat"], co2)
    out["dT"] = dT
    return out


if __name__ == "__main__":
    import time
    os.makedirs(f"{ROOT}/data/cache", exist_ok=True)
    for wname in WARMING:
        t = time.time()
        d = S.precompute(build(wname))
        torch.save(d, f"{ROOT}/data/cache/seasons_{wname}.pt")
        print(wname, d["ghi"].shape, f"{time.time() - t:.0f}s")
