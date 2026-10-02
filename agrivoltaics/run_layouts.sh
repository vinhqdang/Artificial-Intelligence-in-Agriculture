#!/bin/bash
# Fixed non-tracking arrays and rotation-limit sensitivity (split 0, every sixth held-out pair, main response, coarse GCR grid)
cd "$(dirname "$0")"
for a in 0 30; do
  AV_GRID=coarse python src/evaluate_grid.py 0 "" scen=baseline tag=fx sub=6 rules=0 fixed=$a > results/fx$a.out 2>&1
done
for lim in 45 80; do
  AV_MAXROT=$lim AV_GRID=coarse python src/evaluate_grid.py 0 sals_s0_seed0 scen=baseline tag=rot$lim sub=6 window=0.15,1.0 only=AV_static,Phenology_rule,Tuned_phenology_rule,sals_s0_seed0 > results/rot$lim.out 2>&1
done
AV_GRID=coarse python src/evaluate_grid.py 0 sals_s0_seed0 scen=baseline tag=rot60 sub=6 window=0.15,1.0 only=AV_static,Phenology_rule,Tuned_phenology_rule,sals_s0_seed0 > results/rot60.out 2>&1
touch results/layouts.done
