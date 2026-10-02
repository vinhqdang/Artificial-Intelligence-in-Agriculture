# Learning when to share sunlight: differentiable simulation and learned tracker control for agrivoltaics

Code, data pipeline and manuscript for a study prepared for *Artificial Intelligence in
Agriculture*.

A differentiable agrivoltaic simulator (solar geometry, single-axis trackers with backtracking,
PV output, crop light and microclimate, soil water, SIMPLE crop model with a flowering-stage
shade penalty) is used to train a neural controller (SALS) that sets the daily light-sharing
level of the trackers, maximising electricity under a 90 % yield-retention floor. All
controllers are compared under the same deployable protocol: the array density (GCR) of a site
is chosen from its 2001-2014 seasons only, and performance is measured on geographically
held-out sites and on the years 2015-2019, for 400 crop-site pairs (wheat, rice, maize,
soybean, potato) in baseline, +1.5, +2 and +3 degC climates.

## Layout

| path | content |
|------|---------|
| `src/select_sites.py` | area-weighted sampling of cropland sites (SPAM 2010) and crop calendars (GGCMI Phase 3) |
| `src/download_weather.py` | NASA POWER daily weather 2001-2020 |
| `src/physics.py` | solar geometry, trackers, PV, crop light, FAO-56 ET0, SIMPLE crop model (PyTorch) |
| `src/data.py` | season tensors for the four climates |
| `src/simulate.py` | differentiable day-by-day rollout, forecast noise |
| `src/sals.py` | controller, training objective, rule baselines, behaviour-cloning warm start |
| `src/run_train.py` | training (seeds, spatial splits, ablations, floor-conditioned and chance-constrained variants) |
| `src/evaluate_grid.py` | evaluation of all controllers over a GCR grid on held-out sites (restartable) |
| `src/analysis.py`, `src/trajectories.py`, `src/run_extra.py` | history-based design selection, tables, statistics, figures, extra experiments |
| `src/validate_pv.py` | benchmark of the PV model against PVGIS |
| `src/run_all_training.sh`, `src/run_all_eval.sh`, `src/run_all_extra.sh` | experiment queues |
| `src/shade_ml.py`, `src/shade_ml_fit.py` | hierarchical shade-response model learned from the public meta-analysis data in `data/laub/` (leave-one-study-out validation, bootstrap curves) |
| `src/calibrate_shade.py`, `src/calibrate_gradient.py` | calibration of the simulator's shade response: one-parameter fit to published curves, and gradient-based randomised-MAP ensembles (E1 hand-read curves, E2 learned curves, E3 with between-study spread) |
| `src/tune_feedback.py`, `src/tune_phenology.py` | tuning of the feedback rule and of the phenology-rule window on training seasons |
| `src/analysis_variants.py`, `src/analysis_robust.py`, `src/analysis_robust_ci.py`, `src/analysis_design.py`, `src/analysis_splits.py`, `src/analysis_ctl.py` | sensitivity to the shade response, robust/adaptive controllers, design under uncertainty (margin baseline, three splits), tuned-rule comparison |
| `run_robust_eval.sh`, `run_member_eval*.sh`, `run_split_eval.sh`, `run_review_fixes.sh` | restartable evaluation drivers for those analyses |
| `data/sites.csv` | the 400 crop-site pairs |
| `data/shade_posterior*.json` | shade-response ensembles E1, E2, E3 |
| `results/` | result tables (CSV) |
| `manuscript/` | LaTeX source (elsarticle) and figures |

## Reproduction

```bash
pip install torch numpy pandas scipy pyarrow xarray netCDF4 matplotlib cartopy
# inputs: SPAM 2010 v2r0 harvested / physical area CSV, GGCMI Phase 3 calendars (see select_sites.py)
python src/select_sites.py && python src/download_weather.py && python src/data.py
bash src/run_all_training.sh && bash src/run_all_eval.sh && bash src/run_all_extra.sh
cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The pipeline runs on a 4-core CPU; each training run takes about 20 minutes.
