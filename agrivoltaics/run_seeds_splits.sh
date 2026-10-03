#!/bin/bash
# Two more training seeds for splits 1 and 2, then evaluation with each split's own tuned CMA-ES schedule untouched (SALS only)
cd "$(dirname "$0")"
for spec in "sals_s1_seed1 split=1 seed=1" "sals_s1_seed2 split=1 seed=2" "sals_s2_seed1 split=2 seed=1" "sals_s2_seed2 split=2 seed=2"; do
  set -- $spec
  [ -f results/models/$1.pt ] || python3 -u src/run_train.py $spec > results/models/$1.out 2>&1
done
for sp in 1 2; do
  AV_GRID=coarse python src/evaluate_grid.py $sp sals_s${sp}_seed1,sals_s${sp}_seed2 scen=baseline tag=ctl$sp sub=3 rules=0 only=sals_s${sp}_seed1,sals_s${sp}_seed2 > results/ctlseeds$sp.out 2>&1
done
touch results/seeds_splits.done
