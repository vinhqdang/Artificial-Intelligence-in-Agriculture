"""Derivative-free optimisation (CMA-ES) of a five-parameter ramp-plateau light-sharing schedule on training-block seasons,
with the same objective as the other rules: electricity at the density where mean retention equals the floor."""
import os, sys, json, time
import numpy as np, torch, cma
sys.path.insert(0, os.path.dirname(__file__))
import sals as A, simulate as S
from run_train import get_refs, train_mask
torch.set_num_threads(2)
ROOT = A.ROOT
SPLIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
D = A.load_all(); refs = get_refs(D)
tm = train_mask(D, refs, A.site_split(SPLIT))["baseline"]
idx = torch.where(tm)[0]; g_ = torch.Generator().manual_seed(1)
idx = idx[torch.randperm(len(idx), generator=g_)[:300]]
d = S._sel(D["baseline"], idx); yo, ep = refs["baseline"][0][idx], refs["baseline"][1][idx]
G = [0.1, 0.2, 0.3, 0.4, 0.5]


def decode(x):
    a = 0.6 / (1 + np.exp(-x[0])); b = 0.5 + 0.7 / (1 + np.exp(-x[1])); w1 = 0.01 * np.exp(x[2]); w2 = 0.01 * np.exp(x[3]); lvl = 1 / (1 + np.exp(-x[4]))
    return float(a), float(b), float(min(w1, 0.5)), float(min(w2, 0.5)), float(lvl)


def objective(x):
    f = A.make_ramp(*decode(x)); ret, er = [], []
    for g in G:
        y, e = A.evaluate(d, torch.full((len(idx),), g), u_rule=f)
        ret.append(float((y / yo).mean())); er.append(float((e / ep).mean()))
    ret, er = np.array(ret), np.array(er)
    if not (ret.min() < 0.9 < ret.max()):
        return 1.0 + abs(ret.mean() - 0.9)        # penalise schedules that never reach or always exceed the floor
    return -float(np.interp(0.9, ret[::-1], er[::-1]))


SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
x0 = (np.random.default_rng(SEED).standard_normal(5) * 1.5).tolist() if SEED > 1 else [-1.0, 1.0, 1.0, 1.0, 3.0]   # seed 1 starts at the tuned window; seeds 2 and 3 at random points
es = cma.CMAEvolutionStrategy(x0, 1.0, dict(popsize=8, seed=SEED, maxiter=25, verbose=-9))
n = 0
while not es.stop():
    xs = es.ask(); fs = []
    for x in xs:
        t0 = time.time(); fs.append(objective(x)); print('eval', round(time.time() - t0, 1), flush=True)
    es.tell(xs, fs); n += len(xs)
    print(n, round(min(fs), 4), decode(es.result.xbest), flush=True)
best = decode(es.result.xbest); json.dump(dict(params=best, e_at_floor=-es.result.fbest, evaluations=n, seed=SEED, split=SPLIT), open(f"{ROOT}/results/ramp_tuning_s{SEED}.json" if SPLIT == 0 else f"{ROOT}/results/ramp_tuning_split{SPLIT}_s{SEED}.json", "w"))
print("best", best, -es.result.fbest, n)
