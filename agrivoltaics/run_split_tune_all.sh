#!/bin/bash
# sequential (memory): re-tune on the training block of splits 1 and 2
cd "$(dirname "$0")"
for sp in 1 2; do
  python src/tune_phenology.py $sp > results/phenology_tuning_split$sp.out 2>&1
  for s in 1 2 3; do python src/tune_ramp.py $s $sp > results/ramp_tuning_split${sp}_s$s.out 2>&1; done
done
touch results/split_tune.done
