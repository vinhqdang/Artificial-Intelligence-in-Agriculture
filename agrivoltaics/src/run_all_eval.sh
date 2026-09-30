#!/bin/bash
# Grid evaluations for the revised study (run after run_all_training.sh).
cd "$(dirname "$0")/.."
while [ ! -f results/models/training.done ]; do sleep 60; done
python3 -u src/evaluate_grid.py 0 sals_s0_seed0,sals_s0_seed1,sals_s0_seed2,abl_nostress,abl_baseonly tag=split0 > results/grid_split0.out 2>&1
python3 -u src/evaluate_grid.py 0 sals_s0_seed0 kappas=0.0,2.0 scen=baseline,+3C tag=split0_kappa > results/grid_split0_kappa.out 2>&1
python3 -u src/evaluate_grid.py 1 sals_s1_seed0 tag=split1 > results/grid_split1.out 2>&1
python3 -u src/evaluate_grid.py 2 sals_s2_seed0 tag=split2 > results/grid_split2.out 2>&1
echo done > results/eval.done
python3 -u src/analysis.py > results/analysis.out 2>&1 && python3 -u src/trajectories.py > results/trajectories.out 2>&1
