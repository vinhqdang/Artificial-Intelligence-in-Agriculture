#!/bin/bash
# Grids needed for the conformal design analysis (phenology rule, ensemble E2, three splits); restartable per tag
cd "$(dirname "$0")"
P=shade_posterior_ml.json
for sp in 0 1 2; do
  for k in 0 3 5 8 10; do
    [ -f results/grid_e2s${sp}_$k.parquet ] || AV_POSTERIOR=$P AV_VARIANT=member$k AV_GRID=coarse python3 src/evaluate_grid.py $sp "" scen=baseline tag=e2s${sp}_$k sub=3 only=Phenology_rule > results/e2s${sp}_$k.out 2>&1
    [ -f results/grid_e2x${sp}_$k.parquet ] || AV_POSTERIOR=$P AV_VARIANT=meanex$k AV_GRID=coarse python3 src/evaluate_grid.py $sp "" scen=baseline tag=e2x${sp}_$k sub=3 only=Phenology_rule > results/e2x${sp}_$k.out 2>&1
  done
  [ -f results/grid_e2s${sp}_mean.parquet ] || AV_POSTERIOR=$P AV_VARIANT=posterior AV_GRID=coarse python3 src/evaluate_grid.py $sp "" scen=baseline tag=e2s${sp}_mean sub=3 only=Phenology_rule > results/e2s${sp}_mean.out 2>&1
  for v in conservative calibrated field harsh; do
    [ -f results/grid_nt${sp}_$v.parquet ] || AV_VARIANT=$v AV_GRID=coarse python3 src/evaluate_grid.py $sp "" scen=baseline tag=nt${sp}_$v sub=3 only=Phenology_rule > results/nt${sp}_$v.out 2>&1
  done
done
touch results/conformal_grids.done
