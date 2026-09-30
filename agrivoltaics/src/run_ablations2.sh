#!/bin/bash
# Remaining ablations (200 training steps each, same budget as the stress-feature ablation).
cd "$(dirname "$0")/.."
python3 -u src/run_train.py sals_fixedg design=0 fixed_g=0.3 steps=200 > results/train_fixedg.out 2>&1
python3 -u src/run_train.py sals_baseonly scen=baseline steps=200 > results/train_baseonly.out 2>&1
python3 -u src/run_train.py sals_rho80 rho=0.8 steps=200 > results/train_rho80.out 2>&1
python3 -u src/eval_ablation.py '{"sals_main": "SALS (400 steps)", "sals_main_step200": "SALS (200 steps)", "sals_nostress": "w/o stress features (200 steps)", "sals_fixedg": "w/o design network, GCR 0.3 (200 steps)", "sals_baseonly": "trained on baseline climate only (200 steps)"}' > results/eval_ablation.out 2>&1
python3 -u src/evaluate.py '{"sals_rho80": "SALS (ours)"}' rho80 0 0.8 > results/eval_rho80.out 2>&1
echo done > results/ablations2.done
