# SALS: stress-aware light sharing for climate-resilient agrivoltaics

Code, data pipeline and manuscript for the study *"Learning to share sunlight:
differentiable-simulation design and control of climate-resilient agrivoltaics
across the world's staple croplands"* (prepared for *Artificial Intelligence in
Agriculture*).

SALS learns, end-to-end through a differentiable agrivoltaic simulator,

* a **design network** that chooses the ground coverage ratio of single-axis
  tracker rows for a site from its crop and climate, and
* a **control network** that sets the daily light-sharing level of the trackers
  from the crop's phenological stage, soil water and the same-day weather
  forecast,

so that electricity is maximised while at least 90 % of the open-field crop
yield is retained, across 400 cropland sites (wheat, rice, maize, soybean,
potato), 19 seasons and four warming levels (baseline, +1.5, +2, +3 °C).

## Layout

| path | content |
|------|---------|
| `src/select_sites.py` | area-weighted sampling of cropland sites (SPAM 2010) and crop calendars (GGCMI Phase 3) |
| `src/download_weather.py` | NASA POWER daily weather 2001-2020 |
| `src/physics.py` | solar geometry, trackers, PV, crop light, FAO-56 ET0, SIMPLE crop model (PyTorch) |
| `src/data.py` | season tensors for the four climates |
| `src/simulate.py` | differentiable day-by-day rollout |
| `src/sals.py` | SALS networks, training objective and references |
| `src/run_train.py` | training of SALS and ablations |
| `src/evaluate.py` | baselines, oracle and held-out evaluation |
| `src/global_assess.py` | global assessment, canopy-cooling sensitivity, trajectories |
| `src/analysis.py` | tables and figures |
| `data/sites.csv` | the 400 crop-site pairs |
| `results/` | evaluation outputs (CSV) and trained models |
| `manuscript/` | LaTeX source (elsarticle) and figures |

## Reproduction

```bash
pip install torch numpy pandas scipy xarray netCDF4 matplotlib cartopy
# inputs: SPAM 2010 v2r0 harvested area CSV, GGCMI Phase 3 calendars (see select_sites.py)
python src/select_sites.py
python src/download_weather.py
python src/data.py
python src/run_train.py sals_main
bash src/run_queue.sh
python src/analysis.py
cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Everything runs on a 4-core CPU; the main training takes about one hour.
