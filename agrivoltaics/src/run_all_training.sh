#!/bin/bash
# Training runs for the revised study (300 steps each).
cd "$(dirname "$0")/.."
for spec in "sals_s0_seed0 split=0 seed=0" "sals_s0_seed1 split=0 seed=1" "sals_s0_seed2 split=0 seed=2" \
            "sals_s1_seed0 split=1 seed=0" "sals_s2_seed0 split=2 seed=0" \
            "abl_nostress split=0 seed=0 stress=0" "abl_baseonly split=0 seed=0 scen=baseline"; do
  set -- $spec
  [ -f results/models/$1.pt ] || python3 -u src/run_train.py $spec > results/models/$1.out 2>&1
done
echo done > results/models/training.done
