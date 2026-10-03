#!/bin/bash
cd "$(dirname "$0")"
for pair in "E2:shade_posterior_ml.json" "E3:shade_posterior_ml3.json"; do
  e=${pair%%:*}; f=${pair##*:}
  for k in 0 3 5 8 10; do
    AV_VARIANT=member$k AV_POSTERIOR=$f python src/ensemble_spread.py ${e}_m$k >> results/spread.out 2>&1
  done
done
touch results/spread.done
