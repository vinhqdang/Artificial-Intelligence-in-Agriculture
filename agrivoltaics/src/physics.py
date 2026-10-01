"""Differentiable physics of an agrivoltaic system (PyTorch).

Components
----------
* hourly solar geometry and disaggregation of daily irradiance
  (Collares-Pereira and Rabl, 1979; Liu and Jordan, 1960);
* horizontal single-axis (north-south) trackers with backtracking: plane-of-array
  irradiance, row-to-row shading, cell temperature and PV yield;
* light reaching the crop between and below the rows: beam shading fraction and
  a two-dimensional isotropic sky-view factor (pre-computed table);
* FAO-56 Penman-Monteith reference evapotranspiration;
* the SIMPLE crop model (Zhao et al., 2019) with heat and drought stress and a
  daily soil-water balance for the ARID index (Woli et al., 2012).

All angles are in radians. Irradiance is in W m-2 (hourly means) or MJ m-2 d-1.
"""
import math
import numpy as np
import torch

DEG = math.pi / 180.0
MAX_ROT = 60 * DEG          # mechanical rotation limit of the tracker
HUB_HEIGHT = 1.25           # axis height / module width (elevated agrivoltaic structure)
ALBEDO = 0.20
ETA_STC, GAMMA_T, NOCT, SYS_LOSS = 0.21, -0.0035, 45.0, 0.14
HOURS = torch.arange(24, dtype=torch.float32) + 0.5  # solar time, mid-hour


# ----------------------------------------------------------------------------
# Solar geometry and irradiance disaggregation
# ----------------------------------------------------------------------------
def solar_geometry(lat, doy):
    """lat, doy: (B,) tensors. Returns dict of (B, 24) tensors."""
    lat = lat[:, None] * DEG
    decl = 23.45 * DEG * torch.sin(2 * math.pi * (284 + doy[:, None]) / 365.0)
    omega = (HOURS[None, :] - 12.0) * 15 * DEG
    cosz = torch.sin(lat) * torch.sin(decl) + torch.cos(lat) * torch.cos(decl) * torch.cos(omega)
    # sun unit vector: x east, y north, z up
    sx = -torch.cos(decl) * torch.sin(omega)
    sy = torch.cos(lat) * torch.sin(decl) - torch.sin(lat) * torch.cos(decl) * torch.cos(omega)
    sz = cosz
    ws = torch.arccos(torch.clamp(-torch.tan(lat) * torch.tan(decl), -1.0, 1.0))  # sunset hour angle
    return dict(cosz=cosz, sx=sx, sy=sy, sz=sz, omega=omega, ws=ws)


def hourly_irradiance(ghi_day, dhi_day, geo):
    """Disaggregate daily global/diffuse (MJ m-2 d-1) into hourly means (W m-2)."""
    w, ws = geo["omega"], geo["ws"]
    den = torch.sin(ws) - ws * torch.cos(ws) + 1e-6
    a = 0.409 + 0.5016 * torch.sin(ws - 60 * DEG)
    b = 0.6609 - 0.4767 * torch.sin(ws - 60 * DEG)
    day = (torch.cos(w) > torch.cos(ws)).float() * (geo["cosz"] > 0).float()
    rt = math.pi / 24 * (a + b * torch.cos(w)) * (torch.cos(w) - torch.cos(ws)) / den * day
    rd = math.pi / 24 * (torch.cos(w) - torch.cos(ws)) / den * day
    # renormalise so that hourly values integrate exactly to the daily totals
    rt = rt / (rt.sum(1, keepdim=True) + 1e-9)
    rd = rd / (rd.sum(1, keepdim=True) + 1e-9)
    ghi = ghi_day[:, None] * rt * 1e6 / 3600.0
    dhi = torch.minimum(dhi_day[:, None] * rd * 1e6 / 3600.0, ghi)
    bh = ghi - dhi
    return ghi, dhi, bh


# ----------------------------------------------------------------------------
# Tracker geometry
# ----------------------------------------------------------------------------
def sun_projection(geo):
    """Sun angle projected on the east-west vertical plane (ideal tracker angle,
    positive towards east) and its cosine-like weight."""
    psi = torch.atan2(geo["sx"], torch.clamp(geo["sz"], min=1e-3))
    return psi


def backtracking_angle(psi, gcr):
    """Energy-optimal rotation with backtracking (no row-to-row shading)."""
    phi = torch.clamp(psi, -MAX_ROT, MAX_ROT)
    c = torch.cos(psi) / gcr
    corr = torch.arccos(torch.clamp(c, -1.0, 1.0 - 1e-6))
    need = (c < 1.0).float()
    phi_bt = psi - torch.sign(psi) * corr
    return torch.clamp(need * phi_bt + (1 - need) * phi, -MAX_ROT, MAX_ROT)


def light_sharing_angle(psi):
    """Feasible rotation that minimises the beam shadow cast on the crop:
    as close to edge-on (|phi - psi| = 90 deg) as the rotation limit allows."""
    c1 = torch.clamp(psi - math.pi / 2, -MAX_ROT, MAX_ROT)
    c2 = torch.clamp(psi + math.pi / 2, -MAX_ROT, MAX_ROT)
    better1 = (torch.abs(torch.cos(c1 - psi)) <= torch.abs(torch.cos(c2 - psi))).float()
    return better1 * c1 + (1 - better1) * c2


def ground_beam_fraction(phi, psi, gcr):
    """Fraction of between-row ground (averaged over the pitch) reached by beam light."""
    shade = gcr * torch.abs(torch.cos(phi - psi)) / torch.clamp(torch.cos(psi), min=0.05)
    return 1.0 - torch.clamp(shade, 0.0, 1.0)


def panel_shaded_fraction(phi, psi, gcr):
    """Fraction of the module width shaded by the neighbouring row."""
    return torch.clamp(1.0 - torch.cos(psi) / (gcr * torch.abs(torch.cos(phi - psi)) + 1e-6), 0.0, 1.0)


def _svf_numpy(gcr, phi, height=HUB_HEIGHT, n_x=60, n_a=721, n_rows=15):
    """Pitch-averaged sky-view factor of the ground under infinite 2D rows of width 1."""
    p = 1.0 / gcr
    alphas = np.linspace(-np.pi / 2, np.pi / 2, n_a)
    wts = 0.5 * np.cos(alphas) * (alphas[1] - alphas[0])
    vis = []
    for x in (np.arange(n_x) + 0.5) / n_x * p:
        blocked = np.zeros(n_a, bool)
        for i in range(-n_rows, n_rows + 1):
            cx = i * p
            ex = cx + 0.5 * np.array([-np.cos(phi), np.cos(phi)])
            ez = height + 0.5 * np.array([np.sin(phi), -np.sin(phi)])
            ang = np.arctan2(ex - x, ez)  # angle from zenith, positive towards +x
            lo, hi = ang.min(), ang.max()
            blocked |= (alphas >= lo) & (alphas <= hi)
        vis.append((wts * (~blocked)).sum())
    return float(np.mean(vis))


class SkyViewTable:
    """Bilinear interpolation of the ground sky-view factor over (gcr, |phi|)."""

    def __init__(self, g_grid=np.linspace(0.05, 0.65, 25), p_grid=np.linspace(0, MAX_ROT, 13)):
        self.g = torch.tensor(g_grid, dtype=torch.float32)
        self.p = torch.tensor(p_grid, dtype=torch.float32)
        self.t = torch.tensor([[_svf_numpy(g, ph) for ph in p_grid] for g in g_grid], dtype=torch.float32)

    def __call__(self, gcr, phi):
        g = torch.clamp(gcr, self.g[0], self.g[-1]).expand_as(phi)
        a = torch.clamp(torch.abs(phi), self.p[0], self.p[-1])
        gi = torch.clamp(((g - self.g[0]) / (self.g[1] - self.g[0])), 0, len(self.g) - 1.001)
        pi = torch.clamp(((a - self.p[0]) / (self.p[1] - self.p[0])), 0, len(self.p) - 1.001)
        g0, p0 = gi.floor().long(), pi.floor().long()
        dg, dp = gi - g0, pi - p0
        t = self.t
        v00, v01 = t[g0, p0], t[g0, p0 + 1]
        v10, v11 = t[g0 + 1, p0], t[g0 + 1, p0 + 1]
        return (1 - dg) * ((1 - dp) * v00 + dp * v01) + dg * ((1 - dp) * v10 + dp * v11)


def pv_and_ground(phi, psi, gcr, geo, ghi, dhi, bh, tair_h, svf):
    """Hourly PV electricity (Wh per m2 of land) and ground irradiance (W m-2)."""
    cosz = torch.clamp(geo["cosz"], min=0.0)
    dni = torch.where(cosz > 0.05, bh / torch.clamp(cosz, min=0.05), torch.zeros_like(bh))
    # module normal (sin phi, 0, cos phi) with phi positive towards east
    cos_aoi = torch.clamp(geo["sx"] * torch.sin(phi) + geo["sz"] * torch.cos(phi), min=0.0)
    f_ps = panel_shaded_fraction(phi, psi, gcr)
    poa = dni * cos_aoi * (1 - f_ps) + dhi * (1 + torch.cos(phi)) / 2 + ALBEDO * ghi * (1 - torch.cos(phi)) / 2
    tcell = tair_h + poa * (NOCT - 20.0) / 800.0
    eta = ETA_STC * (1 + GAMMA_T * (tcell - 25.0)) * (1 - SYS_LOSS)
    elec = poa * eta * gcr                      # W per m2 of land (module area = gcr * land)
    ground = bh * ground_beam_fraction(phi, psi, gcr) + dhi * svf(gcr, phi)
    return elec, ground


# ----------------------------------------------------------------------------
# Reference evapotranspiration (FAO-56)
# ----------------------------------------------------------------------------
def es(t):
    return 0.6108 * torch.exp(17.27 * t / (t + 237.3))


def et0_fao56(tmax, tmin, tdew, rs, rs_clear, u2, elev):
    """Daily FAO-56 Penman-Monteith grass reference ET (mm d-1). rs in MJ m-2 d-1."""
    t = (tmax + tmin) / 2
    delta = 4098 * es(t) / (t + 237.3) ** 2
    p = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    gamma = 0.000665 * p
    e_s = (es(tmax) + es(tmin)) / 2
    e_a = torch.minimum(es(tdew), e_s)
    rns = 0.77 * rs
    sigma = 4.903e-9
    ratio = torch.clamp(rs / torch.clamp(rs_clear, min=1e-3), 0.25, 1.0)
    rnl = sigma * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2) * (0.34 - 0.14 * torch.sqrt(e_a)) * (1.35 * ratio - 0.35)
    rn = rns - rnl
    et0 = (0.408 * delta * rn + gamma * 900 / (t + 273) * u2 * (e_s - e_a)) / (delta + gamma * (1 + 0.34 * u2))
    return torch.clamp(et0, min=0.0)


# ----------------------------------------------------------------------------
# SIMPLE crop model (Zhao et al., 2019, Table 1a)
# ----------------------------------------------------------------------------
CROP_PARAMS = {
    #          Tsum   HI    I50A I50B Tbase Topt RUE  I50maxH I50maxW Theat Text  SCO2  Swater rootmm
    "wheat":   (2150, 0.34, 280, 50,  0,  15, 1.24, 100, 25, 34, 45, 0.08, 0.4, 1000),
    "rice":    (2300, 0.47, 850, 200, 9,  26, 1.24, 100, 10, 34, 50, 0.08, 1.0, 400),
    "maize":   (2050, 0.50, 500, 50,  8,  28, 2.10, 100, 12, 34, 50, 0.01, 1.2, 1500),
    "soybean": (2350, 0.40, 600, 200, 6,  27, 0.86, 120, 20, 36, 50, 0.07, 0.9, 1400),
    "potato":  (2400, 0.85, 500, 350, 4,  22, 1.30, 50,  30, 34, 45, 0.10, 0.4, 825),
}
# Sensitivity of the harvest index to shading during the critical window around
# flowering (grain number / tuber set), after Fischer (1985) and Andrade et al.
# (1999): HI is multiplied by 1 - HI_SHADE * (mean relative radiation deficit in
# the window). The open field has no deficit and is unaffected.
# Shade compensation of radiation-use efficiency (diffuse-light RUE, light saturation): the
# crop's biomass growth is multiplied by 1 + RUE_COMP * (1 - s), s being the fraction of
# open-field irradiance reaching the crop. The 'conservative' variant has no compensation
# (proportional response); the 'calibrated' variant fits RUE_COMP for rice, wheat and potato
# to published field results (see calibrate_shade.py and data/rue_comp.json).
import json as _json, os as _os
VARIANT = _os.environ.get("AV_VARIANT", "conservative")
RUE_COMP = {"wheat": 0.0, "rice": 0.0, "maize": 0.0, "soybean": 0.0, "potato": 0.0}
if VARIANT == "calibrated":
    RUE_COMP.update(_json.load(open(_os.path.join(_os.path.dirname(__file__), "..", "data", "rue_comp.json"))))
HI_SHADE = {"wheat": 0.4, "rice": 0.5, "maize": 0.6, "soybean": 0.5, "potato": 0.2}
if VARIANT == "harsh":      # strongest plausible shade penalty: proportional growth response, 1.5x harvest-index penalty
    HI_SHADE.update({k: 1.5 * v for k, v in HI_SHADE.items()})
if VARIANT == "posterior":  # ensemble mean of the gradient-calibrated response (calibrate_gradient.py)
    _post = _json.load(open(_os.path.join(_os.path.dirname(__file__), "..", "data", "shade_posterior.json")))
    _cm = [sum(m[i] for m in _post["c"]) / len(_post["c"]) for i in range(5)]
    _hm = [sum(m[i] for m in _post["h"]) / len(_post["h"]) for i in range(5)]
    for _i, _k in enumerate(_post["crops"]):
        RUE_COMP[_k] = _cm[_i]; HI_SHADE[_k] = HI_SHADE[_k] * _hm[_i]
if VARIANT.startswith("member"):   # one member of the calibrated ensemble (index after 'member')
    _post = _json.load(open(_os.path.join(_os.path.dirname(__file__), "..", "data", "shade_posterior.json")))
    _k = int(VARIANT[6:])
    for _i, _c in enumerate(_post["crops"]):
        RUE_COMP[_c] = _post["c"][_k][_i]; HI_SHADE[_c] = HI_SHADE[_c] * _post["h"][_k][_i]
if VARIANT == "field":      # fitted to the field trials themselves (optimistic bound), maize HI unaffected
    RUE_COMP.update(_json.load(open(_os.path.join(_os.path.dirname(__file__), "..", "data", "rue_comp_field.json"))))
    HI_SHADE["maize"] = 0.0
CRIT_WINDOW = (0.45, 0.65)   # fraction of the thermal-time requirement
PARAM_NAMES = ["Tsum", "HI", "I50A", "I50B", "Tbase", "Topt", "RUE", "I50maxH", "I50maxW",
               "Theat", "Text", "SCO2", "Swater", "root"]
AWC = 0.13      # plant-available water per mm of soil (mm mm-1)
UPTAKE = 0.096  # root water uptake coefficient of the ARID index (Woli et al., 2012)
CN, DRAIN = 65.0, 0.55


class CropState:
    def __init__(self, B, wcap):
        z = torch.zeros(B)
        self.tt, self.biomass, self.i50b_extra = z.clone(), z.clone(), z.clone()
        self.wat = wcap.clone()           # start at field capacity
        self.fsolar = z.clone()
        self.arid = z.clone()
        self.defsum, self.wsum = z.clone(), z.clone()   # shade deficit in the critical window


def hi_factor(st, hi_shade):
    """Harvest-index multiplier from shading in the critical window."""
    mean_def = st.defsum / torch.clamp(st.wsum, min=1e-6)
    return 1 - hi_shade * mean_def * (st.wsum > 0).float()


def crop_step(st, p, tmean, tmax_c, rad_c, rain, et0_c, co2, irrigated, active, wcap, shade=None, rue_comp=None):
    """Advance SIMPLE by one day. All inputs (B,). `p` is a dict of (B,) params.
    `active` is 1 while the crop is in the field (sowing to harvest window)."""
    # soil water balance and ARID index
    s_ret = 25400 / CN - 254
    ro = torch.where(rain > 0.2 * s_ret, (rain - 0.2 * s_ret) ** 2 / (rain + 0.8 * s_ret), torch.zeros_like(rain))
    w1 = st.wat + rain - ro
    dr = DRAIN * torch.clamp(w1 - wcap, min=0.0)
    w1 = w1 - dr
    tr = torch.minimum(UPTAKE * w1, et0_c)
    arid = 1 - tr / torch.clamp(et0_c, min=1e-3)
    arid = torch.where(et0_c > 1e-3, arid, torch.zeros_like(arid)) * (1 - irrigated)
    st.wat = torch.clamp(w1 - tr * active, min=0.0) * (1 - irrigated) + wcap * irrigated
    st.arid = arid
    # thermal time and stress factors
    dtt = torch.clamp(tmean - p["Tbase"], min=0.0)
    growing = active * torch.sigmoid((p["Tsum"] - st.tt) / 20.0)
    st.tt = st.tt + dtt * active
    f_temp = torch.clamp((tmean - p["Tbase"]) / (p["Topt"] - p["Tbase"]), 0.0, 1.0)
    f_heat = torch.clamp(1 - (tmax_c - p["Theat"]) / (p["Text"] - p["Theat"]), 0.0, 1.0)
    f_water = torch.clamp(1 - p["Swater"] * arid, 0.0, 1.0)
    # S_CO2 is interpreted per 100 ppm above 350 ppm (capped at 700 ppm), which
    # gives RUE gains consistent with FACE experiments
    f_co2 = 1 + p["SCO2"] * torch.clamp(co2 - 350.0, 0.0, 350.0) / 100.0
    # accelerated senescence under heat and drought stress
    st.i50b_extra = st.i50b_extra + growing * (p["I50maxH"] * (1 - f_heat) + p["I50maxW"] * (1 - f_water))
    i50b = p["I50B"] + st.i50b_extra
    fs_early = torch.sigmoid(0.01 * (st.tt - p["I50A"]))            # numerically stable logistic
    fs_late = torch.sigmoid(-0.01 * (st.tt - (p["Tsum"] - i50b)))
    f_solar = 0.95 * torch.minimum(fs_early, fs_late)
    f_solar = torch.where(f_water < 0.1, f_solar * (0.9 + f_water), f_solar)
    st.fsolar = f_solar
    growth = rad_c * p["RUE"] * f_solar * f_co2 * f_temp * torch.minimum(f_heat, f_water)  # g m-2 d-1
    if shade is not None and rue_comp is not None:
        growth = growth * (1 + rue_comp * (1 - shade))
    st.biomass = st.biomass + growth * growing
    if shade is not None:
        rel = st.tt / p["Tsum"]
        w = active * torch.sigmoid((rel - CRIT_WINDOW[0]) * 50) * torch.sigmoid((CRIT_WINDOW[1] - rel) * 50)
        st.defsum = st.defsum + w * (1 - shade)
        st.wsum = st.wsum + w
    return f_heat, f_water
