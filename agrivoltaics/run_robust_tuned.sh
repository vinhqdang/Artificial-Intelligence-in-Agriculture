#!/bin/bash
# tuned window and CMA-ES ramp under the four fixed responses (same 21-pair sample as the robust-controller comparison)
cd "$(dirname "$0")"
P=$(python -c "import json;print(','.join(str(x) for x in json.load(open('results/ramp_tuning.json'))['params']))")
for v in conservative calibrated field harsh; do
  AV_VARIANT=$v AV_GRID=coarse python src/evaluate_grid.py 0 "" scen=baseline tag=rb_$v sub=6 rules=0 window=0.15,1.0 ramp=$P > results/rbt_$v.out 2>&1
done
touch results/rb_tuned.done
