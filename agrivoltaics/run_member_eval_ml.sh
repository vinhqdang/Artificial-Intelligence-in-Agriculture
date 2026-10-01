#!/bin/bash
# Members of the ensemble calibrated to the data-driven curves (restartable)
cd "$(dirname "$0")"
for k in 0 3 5 8 10; do
  AV_POSTERIOR=shade_posterior_ml.json AV_VARIANT=member$k AV_GRID=coarse python src/evaluate_grid.py 0 sals_s0_seed0 scen=baseline tag=mbml_$k sub=6 only=AV_static,Phenology_rule,sals_s0_seed0 > results/mbml_$k.out 2>&1
done
# the mean of that ensemble as the design belief
AV_POSTERIOR=shade_posterior_ml.json AV_VARIANT=posterior AV_GRID=coarse python src/evaluate_grid.py 0 sals_s0_seed0 scen=baseline tag=mbml_mean sub=6 only=AV_static,Phenology_rule,sals_s0_seed0 > results/mbml_mean.out 2>&1
touch results/mbml_eval.done
