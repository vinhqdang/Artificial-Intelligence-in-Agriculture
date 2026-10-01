#!/bin/bash
# Evaluate rules and controllers under five true shade responses (restartable; per-method caches)
cd "$(dirname "$0")"
MODELS=${1:-sals_s0_seed0,rob_s0}
RULES=${2:-1}
for v in posterior conservative calibrated field harsh; do
  AV_VARIANT=$v AV_GRID=coarse python src/evaluate_grid.py 0 "$MODELS" scen=baseline tag=rb_$v sub=6 feedback=1.0 rules=$RULES > results/rb_$v.out 2>&1
done
touch results/rb_eval.done
