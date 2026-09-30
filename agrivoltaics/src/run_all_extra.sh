#!/bin/bash
# Extra experiments: rho-conditioned and chance-trained controllers, forecast-noise robustness.
cd "$(dirname "$0")/.."
while [ ! -f results/eval.done ]; do sleep 60; done
[ -f results/models/sals_rhocond.pt ] || python3 -u src/run_train.py sals_rhocond split=0 seed=0 rho_cond=1 > results/models/sals_rhocond.out 2>&1
[ -f results/models/sals_chance.pt ] || python3 -u src/run_train.py sals_chance split=0 seed=0 chance=1 > results/models/sals_chance.out 2>&1
python3 -u src/run_extra.py rho > results/extra_rho.out 2>&1
python3 -u src/run_extra.py noise > results/extra_noise.out 2>&1
echo done > results/extra.done
