#!/bin/bash
# Design-under-uncertainty evaluation on all three spatial splits (ensemble E2, every third held-out site); restartable
cd "$(dirname "$0")"
for sp in 0 1 2; do
  for k in 0 3 5 8 10; do
    AV_POSTERIOR=shade_posterior_ml.json AV_VARIANT=member$k AV_GRID=coarse python src/evaluate_grid.py $sp sals_s${sp}_seed0 scen=baseline tag=e2s${sp}_$k sub=3 only=AV_static,Phenology_rule,sals_s${sp}_seed0 > results/e2s${sp}_$k.out 2>&1
  done
  AV_POSTERIOR=shade_posterior_ml.json AV_VARIANT=posterior AV_GRID=coarse python src/evaluate_grid.py $sp sals_s${sp}_seed0 scen=baseline tag=e2s${sp}_mean sub=3 only=AV_static,Phenology_rule,sals_s${sp}_seed0 > results/e2s${sp}_mean.out 2>&1
done
touch results/split_eval.done
