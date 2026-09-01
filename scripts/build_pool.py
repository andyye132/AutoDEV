"""Build the planted-bug instance pool and probe each instance with the hand-designed tester programs.
Output: runs/pool.json (+ runs/instances/*.json mutated levels). Selection into search/held-out splits is a separate step."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from jax2d.engine import PhysicsEngine
from autodev.kinetix_utils import list_levels, load_level, make_env, save_level_json, level_hash
from autodev.defects import inject, CLASSES
from autodev.oracles import witness
from autodev.evaluate import meta_to_json, stack
from autodev.features import make_feature_fn
from autodev.tester import TesterConfig, make_tester, BUILTIN_PROGRAMS

ap = argparse.ArgumentParser()
ap.add_argument("--tiers", default="s,m"); ap.add_argument("--size", default="m"); ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--classes", default="OOB,TUNNEL,SHORTCUT,ANOMALY,DISARMED", help="defect classes to plant (GOAL = unmodified level, win condition as target)")
ap.add_argument("--levels_index", default="", help="JSON index of generated levels (list of {path}); overrides --tiers")
ap.add_argument("--programs", default="random,repeat_random,goexplore"); ap.add_argument("--workers", type=int, default=64); ap.add_argument("--steps", type=int, default=1024)
ap.add_argument("--slots", type=int, default=256); ap.add_argument("--chunk", type=int, default=8); ap.add_argument("--out", default="runs"); ap.add_argument("--max_levels", type=int, default=0)
args = ap.parse_args()
os.makedirs(f"{args.out}/instances", exist_ok=True)
from autodev.evaluate import load_instance_level
levels = [e["path"] for e in json.load(open(args.levels_index))] if args.levels_index else list_levels(tuple(args.tiers.split(",")))
if args.max_levels: levels = levels[: args.max_levels]
_load = (lambda lv: load_instance_level(lv, args.size)) if args.levels_index else (lambda lv: load_level(lv, args.size))
st0, sp, ep = _load(levels[0])
env = make_env(sp, ep, auto_reset=False); engine = PhysicsEngine(sp)
feat_fn, F, spec, describe = make_feature_fn(sp, ep, engine)
cfg = TesterConfig(num_workers=args.workers, steps=args.steps, slots=args.slots)
oracle = lambda s, r, d, meta: witness(s, r, d, meta, engine)[None]
pool, L, M = [], [], []
for lv in levels:
    st, _, _ = _load(lv)
    seen = set()
    for cls in args.classes.split(','):
        for seed in range(args.seeds):
            out = inject(cls, st, sp, ep, np.random.default_rng(seed))
            if out is None: continue
            mst, meta = out
            key = (cls, bool(meta.target_is_poly), int(meta.target_idx))
            if key in seen: continue
            seen.add(key)
            name = f"{os.path.basename(lv).replace('.json', '')}__{cls}__{seed}" if args.levels_index else f"{lv.replace('/', '_').replace('.json', '')}__{cls}__{seed}"
            path = f"{args.out}/instances/{name}.json"
            save_level_json(path, mst, sp, ep)
            pool.append(dict(name=name, level=lv, cls=cls, seed=seed, path=path, meta=meta_to_json(meta), base_hash=level_hash(st), hash=level_hash(mst), probes={}))
            L.append(mst); M.append(meta)
print(f"{len(pool)} instances from {len(levels)} levels; per class:", {c: sum(1 for p in pool if p['cls'] == c) for c in args.classes.split(",")}, flush=True)
t0 = time.time()
for p in args.programs.split(","):
    run = jax.jit(jax.vmap(make_tester(env, ep, sp, feat_fn, BUILTIN_PROGRAMS[p](), cfg, oracle, 1), in_axes=(0, 0, 0)))
    for c0 in range(0, len(pool), args.chunk):
        idx = list(range(c0, min(len(pool), c0 + args.chunk)))
        out = run(jax.random.split(jax.random.PRNGKey(c0), len(idx)), stack([L[i] for i in idx]), stack([M[i] for i in idx])); jax.block_until_ready(out)
        fh = np.array(out["first_hit"][:, 0])
        for j, i in enumerate(idx):
            pool[i]["probes"][p] = float(fh[j]) if fh[j] < cfg.budget else None
        print(f"[{time.time()-t0:6.0f}s] {p:14s} {idx[0]:3d}-{idx[-1]:3d}: found {(fh < cfg.budget).mean():.2f}", flush=True)
    json.dump(dict(cfg=dict(workers=args.workers, steps=args.steps, slots=args.slots, budget=cfg.budget, size=args.size, programs=args.programs), instances=pool), open(f"{args.out}/pool.json", "w"), indent=1)
# summary per class
for c in CLASSES:
    rows = [q for q in pool if q["cls"] == c]
    if not rows: continue
    line = f"{c:9s} n={len(rows):3d} " + " ".join(f"{p}: found {np.mean([q['probes'][p] is not None for q in rows]):.2f}" for p in args.programs.split(","))
    print(line)
print("pool written to", f"{args.out}/pool.json")
