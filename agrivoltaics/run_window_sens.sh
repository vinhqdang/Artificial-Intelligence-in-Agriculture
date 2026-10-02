#!/bin/bash
# Sensitivity of the controller ranking to the assumed harvest-index window and sensitivity (split 0, every sixth held-out pair)
cd "$(dirname "$0")"
while [ ! -f results/layouts.done ]; do sleep 30; done
AV_CRITWIN=0.30,0.50 AV_GRID=coarse python src/evaluate_grid.py 0 sals_s0_seed0 scen=baseline tag=win_early sub=6 window=0.15,1.0 > results/win_early.out 2>&1
AV_CRITWIN=0.60,0.80 AV_GRID=coarse python src/evaluate_grid.py 0 sals_s0_seed0 scen=baseline tag=win_late sub=6 window=0.15,1.0 > results/win_late.out 2>&1
AV_HISCALE=0 AV_GRID=coarse python src/evaluate_grid.py 0 sals_s0_seed0 scen=baseline tag=win_h0 sub=6 window=0.15,1.0 > results/win_h0.out 2>&1
touch results/window_sens.done
