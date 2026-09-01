"""Mac screening bench for exploration methods: evaluate hooks programs on the stratified subset runs/hyak/split_bench.json
(24 search / 12 held-out-level / 12 held-out-class instances, no unwitnessable or free-hit instances) with the fixed, verified
harness at a laptop-feasible budget. Ranks programs; results/bench/<tag>.json keeps per-instance first hits.

    JAX_PLATFORMS=cpu python scripts/bench_methods.py --tag round0 --programs evolve/baselines/repeat_random.py,evolve/initial.py,...
"""
import os, sys, json, time, argparse, importlib.util, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--programs", required=True); ap.add_argument("--tag", default="bench")
ap.add_argument("--split", default="runs/hyak/split_bench.json"); ap.add_argument("--splits", default="search,heldout_levels_instances,heldout_class_instances")
ap.add_argument("--seeds", default="100,101"); ap.add_argument("--workers", type=int, default=64); ap.add_argument("--steps", type=int, default=1024); ap.add_argument("--slots", type=int, default=256); ap.add_argument("--chunk", type=int, default=8)
a = ap.parse_args()
from autodev.search_eval import evaluate_hooks
from autodev.substrate import KinetixSubstrate
sub = KinetixSubstrate(size="l"); out_path = f"results/bench/{a.tag}.json"; os.makedirs("results/bench", exist_ok=True)
R = json.load(open(out_path)) if os.path.exists(out_path) else {"cfg": vars(a), "programs": {}}
for path in a.programs.split(","):
    key = os.path.basename(path).replace(".py", "")
    if key in R["programs"] and all(s in R["programs"][key] for s in a.splits.split(",")): print("skip (done)", key); continue
    spec = importlib.util.spec_from_file_location("prog", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    R["programs"].setdefault(key, {})
    for spl in a.splits.split(","):
        t0 = time.time()
        try:
            r = evaluate_hooks(m.init, m.restore, m.act, m.archive, split_path=a.split, which=spl, size="l", seeds=tuple(int(x) for x in a.seeds.split(",")), workers=a.workers, steps=a.steps, slots=a.slots, chunk=a.chunk, substrate=sub)
        except Exception as e:
            import traceback; R["programs"][key][spl] = dict(error=traceback.format_exc()[-1500:]); print(f"{key:28s} {spl[:8]:8s} ERROR {type(e).__name__}: {str(e)[:160]}", flush=True); json.dump(R, open(out_path, "w")); continue
        R["programs"][key][spl] = dict(score=r["score"], detected=r["detected"], detected_per_class=r["detected_per_class"], first_hit=r["first_hit"], names=r["names"], budget=r["budget"], n_hits=r["n_hits"], n_unverified=r["n_unverified"], wall=time.time() - t0)
        print(f"{key:28s} {spl[:8]:8s} score {r['score']:.3f} detected {r['detected']:.2f} unverified {r['n_unverified']}/{r['n_hits']} per-class {{{', '.join(f'{k}:{v:.2f}' for k, v in r['detected_per_class'].items())}}} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(R, open(out_path, "w"))
print("\nRANKING (search / held-out levels / held-out class):")
rows = sorted(R["programs"].items(), key=lambda kv: -kv[1].get("search", {}).get("score", 0))
for k, v in rows: print(f"  {k:28s} " + "  ".join(f"{spl[:6]} {v[spl]['score']:.3f}/{100*v[spl]['detected']:.0f}%" if 'score' in v.get(spl, {}) else f"{spl[:6]} ERROR" for spl in a.splits.split(",") if spl in v))
