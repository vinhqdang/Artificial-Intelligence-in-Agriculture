#!/bin/bash
# Retrain the robust and adaptive controllers on the wider ensembles E2 and E3, then evaluate them under the four fixed responses
cd "$(dirname "$0")"
python src/run_train.py rob2_s0 split=0 seed=0 theta=1 post=shade_posterior_ml.json jhi=0.25 > results/models/rob2_s0.out 2>&1
python src/run_train.py rob3_s0 split=0 seed=0 theta=1 post=shade_posterior_ml3.json jhi=0.25 > results/models/rob3_s0.out 2>&1
python src/run_train.py adp3_s0 split=0 seed=0 theta=1 ref=1 post=shade_posterior_ml3.json jhi=0.25 > results/models/adp3_s0.out 2>&1
touch results/retrain.done
./run_robust_eval.sh rob2_s0,rob3_s0,adp3_s0 0
touch results/retrain_eval.done
