#!/bin/bash
# Leave-one-out design belief (mean of the other four members) for E2 and E3, and named responses on three splits
cd "$(dirname "$0")"
for ens in ml:e2x ml3:e3x; do
  post=${ens%%:*}; tg=${ens##*:}
  for sp in 0 1 2; do for k in 0 3 5 8 10; do
    AV_POSTERIOR=shade_posterior_$post.json AV_VARIANT=meanex$k AV_GRID=coarse python src/evaluate_grid.py $sp "" scen=baseline tag=${tg}${sp}_$k sub=3 only=Phenology_rule > results/${tg}${sp}_$k.out 2>&1
  done; done
done
touch results/review3_loo.done
for v in conservative calibrated field harsh; do for sp in 0 1 2; do
  AV_VARIANT=$v AV_GRID=coarse python src/evaluate_grid.py $sp "" scen=baseline tag=nt${sp}_$v sub=3 only=Phenology_rule > results/nt${sp}_$v.out 2>&1
done; done
touch results/review3_named.done
