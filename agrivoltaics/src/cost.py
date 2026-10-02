"""Illustrative annual economics per hectare of cropland (USD ha-1 yr-1). All parameters are assumptions with ranges,
not data; the analysis reports break-even prices and one-at-a-time sensitivities."""
import numpy as np
CROPS = ["wheat", "rice", "maize", "soybean", "potato"]
BASE = dict(
    p_e=60.0,            # electricity price, USD per MWh (30-120)
    capex_av=1.0,        # AV single-axis tracker plant, USD per Wp (0.7-1.5; elevated structure)
    capex_pv=0.75,       # conventional ground-mounted tracker plant, USD per Wp (0.6-1.0)
    rate=0.06, life=25,  # discount rate and lifetime
    om=0.015,            # O&M, fraction of capex per year
    kwp_ha=2100.0,       # kWp per hectare at GCR 1 (module area share g of 10 000 m2 at 0.21 kWp m-2)
    price=dict(wheat=290.0, rice=400.0, maize=250.0, soybean=520.0, potato=1000.0),   # USD per tonne dry matter
    crop_mult=1.0,       # multiplier on all crop prices
)


def crf(rate, life):
    return rate * (1 + rate) ** life / ((1 + rate) ** life - 1)


def annual_capex(g, capex_per_wp, p):
    """USD ha-1 yr-1: annuity plus O&M of the installed capacity at ground coverage ratio g."""
    return p["kwp_ha"] * g * 1000.0 * capex_per_wp * (crf(p["rate"], p["life"]) + p["om"])


def components(df):
    """Per-row economic components that are linear in the prices: mean energy (MWh), GCR, crop-revenue loss and open revenue at unit prices."""
    price = np.array([BASE["price"][c] for c in CROPS])[df.crop.astype(int).values]
    return dict(E=df.e.values, g=df.g.values, e_pv=df.e_pv.values, closs=price * (df.y_open.values - df.y.values), copen=price * df.y_open.values)


def net_av(E, g, closs, p=BASE):
    """Net value of the AV system relative to the open field (crop costs unchanged)."""
    return p["p_e"] * E - annual_capex(g, p["capex_av"], p) - p["crop_mult"] * closs


def net_pv_plant(e_pv, copen, p=BASE):
    """Net value of a conventional PV plant (GCR 0.4) that replaces the crop, relative to the open field."""
    return p["p_e"] * e_pv - annual_capex(0.4, p["capex_pv"], p) - p["crop_mult"] * copen


def breakeven_price(E, g, closs, p=BASE):
    return (annual_capex(g, p["capex_av"], p) + p["crop_mult"] * closs) / E
