"""Device-independence check: evaluate programs on the laptop CPU with the same protocol as the cluster (fixed harness,
seeds 100-102, 128x2048) on a random subset of search instances, so cluster numbers can be compared with an independent device.

    JAX_PLATFORMS=cpu python scripts/cpu_replication.py --n 16 --programs evolve/baselines/repeat_random.py,runs/evolve_overnight/candidates/cand1_gen037_0.6994.py
"""
import os, sys, json, time, argparse, importlib.util, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=16); ap.add_argument("--seeds", default="100,101,102"); ap.add_argument("--workers", type=int, default=128); ap.add_argument("--steps", type=int, default=2048)
ap.add_argument("--programs", default="evolve/baselines/repeat_random.py,runs/evolve_overnight/candidates/cand1_gen037_0.6994.py"); ap.add_argument("--out", default="results/cpu_replication.json"); ap.add_argument("--split", default="runs/hyak/split.json")
a = ap.parse_args()
from autodev.search_eval import evaluate_hooks
from autodev.substrate import KinetixSubstrate
split = json.load(open(a.split)); rng = np.random.default_rng(0); names = sorted(rng.choice(split["search"], size=a.n, replace=False).tolist())
sub_split = dict(split); sub_split["search"] = names; tmp = f"/tmp/cpu_rep_split_{os.getpid()}.json"; json.dump(sub_split, open(tmp, "w"))
sub = KinetixSubstrate(size="l"); res = {"names": names, "seeds": a.seeds, "workers": a.workers, "steps": a.steps, "programs": {}}
for path in a.programs.split(","):
    spec = importlib.util.spec_from_file_location("prog", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    t0 = time.time(); r = evaluate_hooks(m.init, m.restore, m.act, m.archive, split_path=tmp, which="search", size="l", seeds=tuple(int(x) for x in a.seeds.split(",")), workers=a.workers, steps=a.steps, slots=256, chunk=8, substrate=sub, verify_hits=True)
    res["programs"][os.path.basename(path)] = dict(score=r["score"], detected=r["detected"], first_hit=r["first_hit"], n_hits=r["n_hits"], n_unverified=r["n_unverified"], wall=time.time() - t0)
    print(f"{os.path.basename(path):32s} score {r['score']:.3f} detected {r['detected']:.2f} hits {r['n_hits']} unverified {r['n_unverified']} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
