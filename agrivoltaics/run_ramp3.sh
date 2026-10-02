#!/bin/bash
# CMA-ES ramp schedule: three seeds, 200 evaluations each; the best training objective is evaluated on three splits
cd "$(dirname "$0")"
for s in 1 2 3; do python src/tune_ramp.py $s > results/ramp_tuning_s$s.out 2>&1; done
python - <<'PY'
import json
best=max((json.load(open(f"results/ramp_tuning_s{s}.json")) for s in (1,2,3)), key=lambda d: d["e_at_floor"])
json.dump(best, open("results/ramp_tuning.json","w"))
print(best)
PY
rm -rf results/grid_cache/ctl0/baseline_1.0_Optimised_ramp_schedule.parquet results/grid_cache/ctl1/baseline_1.0_Optimised_ramp_schedule.parquet results/grid_cache/ctl2/baseline_1.0_Optimised_ramp_schedule.parquet
rm -rf results/grid_cache/rb_*/baseline_1.0_Optimised_ramp_schedule.parquet
./run_ramp_eval.sh
./run_robust_tuned.sh
touch results/ramp3.done
