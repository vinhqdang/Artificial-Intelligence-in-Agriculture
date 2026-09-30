#!/bin/bash
# Experiment queue (sequential to stay within memory).
cd "$(dirname "$0")/.."
python3 -u src/run_train.py sals_main steps=400 > results/train_main.out 2>&1
python3 -u src/evaluate.py '{"sals_main": "SALS (ours)"}' main 1 > results/eval_main.out 2>&1
python3 -u src/global_assess.py sals_main > results/global.out 2>&1
python3 -u src/run_train.py sals_nostress stress=0 steps=200 > /dev/null 2>&1
python3 -u src/evaluate.py '{"sals_main_step200": "SALS (200 steps)", "sals_nostress": "w/o stress features (200 steps)"}' ablation 0 > results/eval_ablation.out 2>&1
echo done > results/queue.done
