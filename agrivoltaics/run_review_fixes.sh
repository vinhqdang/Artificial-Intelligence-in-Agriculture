#!/bin/bash
# (1) controllers incl. a tuned phenology rule on three splits (main response); (2) ensemble E3 members on three splits
cd "$(dirname "$0")"
for sp in 0 1 2; do
  AV_GRID=coarse python src/evaluate_grid.py $sp sals_s${sp}_seed0 scen=baseline tag=ctl$sp sub=3 window=0.15,1.0 > results/ctl$sp.out 2>&1
done
touch results/ctl.done
for sp in 0 1 2; do
  for k in 0 3 5 8 10; do
    AV_POSTERIOR=shade_posterior_ml3.json AV_VARIANT=member$k AV_GRID=coarse python src/evaluate_grid.py $sp sals_s${sp}_seed0 scen=baseline tag=e3s${sp}_$k sub=3 only=AV_static,Phenology_rule,sals_s${sp}_seed0 > results/e3s${sp}_$k.out 2>&1
  done
  AV_POSTERIOR=shade_posterior_ml3.json AV_VARIANT=posterior AV_GRID=coarse python src/evaluate_grid.py $sp sals_s${sp}_seed0 scen=baseline tag=e3s${sp}_mean sub=3 only=AV_static,Phenology_rule,sals_s${sp}_seed0 > results/e3s${sp}_mean.out 2>&1
done
touch results/e3.done
