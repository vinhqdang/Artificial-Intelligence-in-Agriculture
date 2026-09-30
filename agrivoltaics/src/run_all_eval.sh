#!/bin/bash
# Grid evaluations for the revised study; each step is skipped if its output already says "saved".
cd "$(dirname "$0")/.."
while [ ! -f results/models/training.done ]; do sleep 60; done
step() { tag=$1; shift; grep -qs "^saved" results/grid_$tag.out || python3 -u src/evaluate_grid.py "$@" > results/grid_$tag.out 2>&1; }
step split0 0 sals_s0_seed0,sals_s0_seed1,sals_s0_seed2,abl_nostress,abl_baseonly tag=split0
step split0_kappa 0 sals_s0_seed0 kappas=0.0,2.0 scen=baseline,+3C tag=split0_kappa
step split1 1 sals_s1_seed0 tag=split1
step split2 2 sals_s2_seed0 tag=split2
echo done > results/eval.done
python3 -u src/analysis.py > results/analysis.out 2>&1 && python3 -u src/trajectories.py > results/trajectories.out 2>&1
