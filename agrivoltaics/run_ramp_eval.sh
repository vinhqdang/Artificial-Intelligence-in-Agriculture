#!/bin/bash
# Evaluate the CMA-ES ramp schedule on three splits (appended to the controller comparison caches)
cd "$(dirname "$0")"
P=$(python -c "import json;print(','.join(str(x) for x in json.load(open('results/ramp_tuning.json'))['params']))")
for sp in 0 1 2; do
  AV_GRID=coarse python src/evaluate_grid.py $sp "" scen=baseline tag=ctl$sp sub=3 rules=0 ramp=$P > results/ctlramp$sp.out 2>&1
done
touch results/ramp_eval.done
