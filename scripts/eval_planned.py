"""Evaluate an LLM-planned per-level tester assignment: each level's instances are run by the program the plan assigns, and the
merged per-instance first-hit array is scored exactly like a single tester (same protocol, same exclusions downstream).

    python scripts/eval_planned.py --assignment results/planner_assignment.json --split runs/hyak/split.json \
        --which search --seeds 100,101,102 --out runs/hyak/eval_planned.json
"""
import os, sys, json, time, argparse, importlib.util
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--assignment", required=True); ap.add_argument("--split", default="runs/hyak/split.json")
ap.add_argument("--which", default="search"); ap.add_argument("--seeds", default="100,101,102"); ap.add_argument("--workers", type=int, default=128)
ap.add_argument("--steps", type=int, default=2048); ap.add_argument("--slots", type=int, default=256); ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--programs_dir", default="evolve/methods"); ap.add_argument("--out", default="runs/hyak/eval_planned.json"); a = ap.parse_args()
from autodev.search_eval import evaluate_hooks
plan = {k: v["program"] if isinstance(v, dict) else v for k, v in json.load(open(a.assignment)).items()}
split = json.load(open(a.split)); names = split[a.which]
groups = {}
for n in names: groups.setdefault(plan.get(n.split("__")[0], "split_pump"), []).append(n)
print({k: len(v) for k, v in groups.items()}, flush=True)
seeds = tuple(int(x) for x in a.seeds.split(","))
fh_all = {}; budget = None
import tempfile
for prog, sub in groups.items():
    path = os.path.join(a.programs_dir, prog + ".py")
    spec = importlib.util.spec_from_file_location(prog, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    tmp = dict(split); tmp[a.which] = sub
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(tmp, tf); tf.close()
    t0 = time.time()
    r = evaluate_hooks(m.init, m.restore, m.act, m.archive, split_path=tf.name, which=a.which, size="l", seeds=seeds,
                       workers=a.workers, steps=a.steps, slots=a.slots, chunk=a.chunk, log=lambda s: print("  " + s, flush=True))
    fh = np.asarray(r["first_hit"], float); budget = r["budget"]
    for nm, row in zip(r["names"], fh): fh_all[nm] = row
    print(f"{prog}: {len(sub)} instances score {r['score']:.3f} detected {r['detected']:.2f} unverified {r['n_unverified']}/{r['n_hits']} ({time.time()-t0:.0f}s)", flush=True)
fh = np.stack([fh_all[n] for n in names])
score = float(np.clip(1 - np.minimum(fh, budget) / budget, 0, 1).mean()); det = float((fh < budget).mean())
print(f"PLANNED {a.which}: score {score:.3f} detected {det:.2f} over {len(names)} instances", flush=True)
out = {}
if os.path.exists(a.out): out = json.load(open(a.out))
out["planned_" + os.path.basename(a.assignment).replace(".json", "")] = {a.which: dict(score=score, detected=det, first_hit=fh.tolist(), names=names, budget=int(budget))}
json.dump(out, open(a.out, "w")); print("wrote", a.out)
