"""Download daily NASA POWER (AG community) weather, 2001-2020, for every site."""
import os, json, time, urllib.request
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

OUT = os.environ.get("WEATHER_DIR", "/home/user/data/power")
PARAMS = "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DIFF,T2M,T2M_MAX,T2M_MIN,T2MDEW,PRECTOTCORR,RH2M,WS2M"
URL = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters={p}&community=AG"
       "&longitude={lon}&latitude={lat}&start=20010101&end=20201231&format=JSON")


def fetch(latlon):
    lat, lon = latlon
    fn = f"{OUT}/{lat:+.2f}_{lon:+.2f}.csv"
    if os.path.exists(fn):
        return fn
    for attempt in range(5):
        try:
            j = json.load(urllib.request.urlopen(URL.format(p=PARAMS, lat=lat, lon=lon), timeout=120))
            p = j["properties"]["parameter"]
            df = pd.DataFrame({k: pd.Series(v) for k, v in p.items()})
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df = df.replace(-999.0, np.nan).interpolate(limit_direction="both")
            df["elevation"] = j["geometry"]["coordinates"][2]
            df.to_csv(fn)
            return fn
        except Exception as e:  # rate limiting or transient errors
            time.sleep(5 * (attempt + 1))
    return None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    s = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "sites.csv"))
    cells = s[["lat", "lon"]].drop_duplicates().values.tolist()
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(fetch, cells))
    print("downloaded", sum(r is not None for r in res), "of", len(cells))
