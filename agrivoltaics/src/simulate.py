"""Day-by-day rollout of an agrivoltaic system for a batch of site-seasons.

A controller chooses, every day, a light-sharing level u in [0, 1]: the tracker
rotation is phi = (1 - u) * phi_backtrack + u * phi_share, where phi_backtrack
maximises electricity and phi_share minimises the shadow on the crop. Under the
panels the crop receives less radiation, has a lower canopy maximum temperature
(KAPPA degC per unit of shading) and a lower reference evapotranspiration.
"""
import math
import torch
import physics as P

KAPPA = 1.0     # canopy Tmax reduction (degC) under full shade (main case)
# forecast errors seen by controllers: Tmax (degC), relative irradiance and ET0
FORECAST_SD = {"tmax": 2.0, "ghi": 0.20, "et0": 0.15}
SVF = None


def get_svf():
    global SVF
    if SVF is None:
        SVF = P.SkyViewTable()
    return SVF


def _sel(d, idx):
    out = {}
    for k, v in d.items():
        out[k] = v[idx] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == d["lat"].shape[0] else v
    return out


def features(st, p, day, t, g, irrigated, crop_onehot, lat):
    """Controller inputs available on day t (current state + same-day forecast)."""
    return torch.stack([
        st.tt / p["Tsum"], st.fsolar, st.wat / (P.AWC * p["root"]), st.arid,
        (day["tmax"] - p["Theat"]) / 10.0, day["ghi"] / 30.0, irrigated, g,
        day["active"], ((day["tmax"] + day["tmin"]) / 2 - p["Topt"]) / 10.0, day["et0"] / 10.0,
        torch.abs(lat) / 90.0, torch.full_like(lat, t / 365.0),
    ] + [crop_onehot[:, i] for i in range(crop_onehot.shape[1])], 1)


def precompute(d):
    """Cache all weather-only hourly quantities of a season set (float16 storage
    for the large hourly tensors)."""
    B, T = d["ghi"].shape
    keys = ["ghi_h", "dhi_h", "bh_h", "psi", "sx", "sz", "tair_h"]
    c = {k: torch.zeros(B, T, 24, dtype=torch.float16) for k in keys}
    c["et0"], c["rso"] = torch.zeros(B, T), torch.zeros(B, T)
    for t in range(T):
        geo = P.solar_geometry(d["lat"], d["doy"][:, t])
        ghi, dhi, bh = P.hourly_irradiance(d["ghi"][:, t], d["dhi"][:, t], geo)
        ra = (1367.0 * torch.clamp(geo["cosz"], min=0)).sum(1) * 3600 / 1e6
        rso = (0.75 + 2e-5 * d["elev"]) * ra
        tx, tn = d["tmax"][:, t:t + 1], d["tmin"][:, t:t + 1]
        vals = dict(ghi_h=ghi, dhi_h=dhi, bh_h=bh, psi=P.sun_projection(geo), sx=geo["sx"], sz=geo["sz"],
                    tair_h=(tx + tn) / 2 + (tx - tn) / 2 * torch.cos(2 * math.pi * (P.HOURS[None] - 14) / 24))
        for k in keys:
            c[k][:, t] = vals[k].half()
        c["rso"][:, t] = rso
        c["et0"][:, t] = P.et0_fao56(d["tmax"][:, t], d["tmin"][:, t], d["tdew"][:, t], d["ghi"][:, t], rso,
                                     d["u2"][:, t], d["elev"])
    d.update(c)
    return d


def rollout(d, g, controller=None, u_seq=None, irrigated=None, panels=True, kappa=KAPPA, record=False,
            forecast_noise=True, noise_seed=0, rho=None, noise_scale=1.0):
    """Simulate a batch. g: (B,) ground-coverage ratio. controller(features)->u (B,),
    or u_seq (B, 365). Returns yield (t ha-1 dry matter), electricity (MWh ha-1),
    and optional daily records."""
    svf = get_svf()
    B = d["lat"].shape[0]
    p = {k: d[k] for k in P.PARAM_NAMES}
    if irrigated is None:
        irrigated = torch.zeros(B)
    wcap = P.AWC * p["root"]
    st = P.CropState(B, wcap)
    crop_onehot = torch.nn.functional.one_hot(d["crop"].long(), 5).float()
    energy = torch.zeros(B)
    hi_shade = torch.tensor([P.HI_SHADE[c] for c in P.CROP_PARAMS])[d["crop"].long()]
    rue_comp = torch.tensor([P.RUE_COMP[c] for c in P.CROP_PARAMS])[d["crop"].long()]
    gen = torch.Generator().manual_seed(noise_seed)
    T = d["ghi"].shape[1]
    if forecast_noise and controller is not None:
        nz = {k: torch.randn(B, T, generator=gen) * sd * noise_scale for k, sd in FORECAST_SD.items()}
    else:
        nz = None
    rec = {"u": [], "shade": [], "fheat": [], "fwater": []} if record else None
    for t in range(d["ghi"].shape[1]):
        day = {k: d[k][:, t] for k in ["ghi", "dhi", "tmax", "tmin", "tdew", "rain", "u2", "doy", "active", "et0", "rso"]}
        rso = day["rso"]
        ghi, dhi, bh = d["ghi_h"][:, t].float(), d["dhi_h"][:, t].float(), d["bh_h"][:, t].float()
        geo = dict(sx=d["sx"][:, t].float(), sz=d["sz"][:, t].float())
        geo["cosz"] = geo["sz"]
        if panels:
            psi = d["psi"][:, t].float()
            gg = g[:, None]
            phi_bt = P.backtracking_angle(psi, gg)
            if controller is not None:
                fday = day
                if nz is not None:  # the controller sees noisy forecasts, the physics the true weather
                    fday = dict(day)
                    fday["tmax"] = day["tmax"] + nz["tmax"][:, t]
                    fday["ghi"] = day["ghi"] * (1 + nz["ghi"][:, t]).clamp(min=0)
                    fday["et0"] = day["et0"] * (1 + nz["et0"][:, t]).clamp(min=0)
                fin = features(st, p, fday, t, g, irrigated, crop_onehot, d["lat"])
                if rho is not None:   # food-security floor as an extra controller input
                    rcol = rho if torch.is_tensor(rho) else torch.full((B,), float(rho))
                    fin = torch.cat([fin, rcol[:, None]], 1)
                u = controller(fin) * day["active"]
            elif u_seq is not None:
                u = u_seq[:, t] * day["active"]
            else:
                u = torch.zeros(B)
            phi = (1 - u[:, None]) * phi_bt + u[:, None] * P.light_sharing_angle(psi)
            tair = d["tair_h"][:, t].float()
            elec, ground = P.pv_and_ground(phi, psi, gg, geo, ghi, dhi, bh, tair, svf)
            energy = energy + elec.sum(1) * 1e-2          # Wh m-2 -> MWh ha-1
            shade = ground.sum(1) / (ghi.sum(1) + 1e-6)
        else:
            u = torch.zeros(B); shade = torch.ones(B)
        rad_c = day["ghi"] * shade
        tmax_c = day["tmax"] - kappa * (1 - shade)
        et0_c = P.et0_fao56(tmax_c, day["tmin"], day["tdew"], rad_c, rso, day["u2"], d["elev"])
        tmean = (day["tmax"] + day["tmin"]) / 2
        fh, fw = P.crop_step(st, p, tmean, tmax_c, rad_c, day["rain"], et0_c, d["co2"], irrigated, day["active"], wcap, shade=shade, rue_comp=rue_comp)
        if record:
            rec["u"].append(u); rec["shade"].append(shade); rec["fheat"].append(fh); rec["fwater"].append(fw)
    matured = torch.sigmoid((st.tt - p["Tsum"]) / 20.0)
    yld = st.biomass * p["HI"] * P.hi_factor(st, hi_shade) * matured / 100.0             # g m-2 -> t ha-1
    if record:
        rec = {k: torch.stack(v, 1) for k, v in rec.items()}
        return yld, energy, rec
    return yld, energy
