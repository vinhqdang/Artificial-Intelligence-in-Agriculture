"""Cache the crop-light time series and electricity of the phenology rule for every baseline season and
a grid of GCRs. The rule depends only on thermal time, so the shading is independent of the crop's shade
response and later simulations over many responses are crop-only (cheap). Restartable per (chunk, GCR)."""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
import sals as A, simulate as S
ROOT = A.ROOT
torch.set_num_threads(4)
G = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
CH = 800
if __name__ == "__main__":
    d = A.load_all()["baseline"]; B = d["lat"].shape[0]
    os.makedirs(f"{ROOT}/data/cache/shade", exist_ok=True); t0 = time.time()
    for i in range(0, B, CH):
        idx = torch.arange(i, min(i + CH, B)); dt = S._sel(d, idx)
        for g in G:
            fn = f"{ROOT}/data/cache/shade/ph_{i}_{g}.pt"
            if os.path.exists(fn): continue
            with torch.no_grad():
                y, e, rec = S.rollout(dt, torch.full((len(idx),), g), controller=A.u_phenology, forecast_noise=False,
                                      irrigated=torch.zeros(len(idx)), record=True)
            torch.save(dict(shade=rec["shade"].half(), e=e), fn + ".tmp"); os.replace(fn + ".tmp", fn)
            print(i, g, f"{time.time() - t0:.0f}s", flush=True)
    open(f"{ROOT}/data/cache/shade/done", "w").write("ok")
