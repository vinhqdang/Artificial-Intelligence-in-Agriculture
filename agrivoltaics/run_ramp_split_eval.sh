#!/bin/bash
# Evaluate on splits 1 and 2 the CMA-ES schedule tuned on each split's own training block (split 0 keeps its schedule)
cd "$(dirname "$0")"
declare -A F=( [1]=results/ramp_tuning_split1_s2.json [2]=results/ramp_tuning_split2_s1.json )
for sp in 1 2; do
  P=$(python -c "import json;print(','.join(str(x) for x in json.load(open('${F[$sp]}'))['params']))")
  AV_GRID=coarse python src/evaluate_grid.py $sp "" scen=baseline tag=ctl$sp sub=3 rules=0 ramp=$P > results/ctlramp_own$sp.out 2>&1
done
touch results/ramp_split_eval.done
