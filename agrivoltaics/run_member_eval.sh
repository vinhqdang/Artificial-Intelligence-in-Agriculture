#!/bin/bash
# Evaluate three methods under five members of the calibrated shade-response ensemble (restartable)
cd "$(dirname "$0")"
for k in 0 3 5 8 10; do
  AV_VARIANT=member$k AV_GRID=coarse python src/evaluate_grid.py 0 sals_s0_seed0 scen=baseline tag=mb_$k sub=6 only=AV_static,Phenology_rule,sals_s0_seed0 > results/mb_$k.out 2>&1
done
touch results/mb_eval.done
