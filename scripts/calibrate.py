"""Build the defect-instance pool: inject each class into each level with a few seeds, save mutated levels + metas,
and probe each instance with random play and a short task-reward PPO. Output: runs/pool.json + runs/instances/*.json"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from autodev.kinetix_utils import list_levels, load_level, save_level_json, level_hash
from autodev.defects import inject, CLASSES
from autodev.evaluate import Evaluator, meta_to_json, summarize
from autodev.objectives import ZeroObjective, CountObjective
from autodev.ppo import PPOConfig

ap = argparse.ArgumentParser()
ap.add_argument("--tiers", default="s,m"); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--out", default="runs")
ap.add_argument("--probe_updates", type=int, default=30); ap.add_argument("--num_envs", type=int, default=64); ap.add_argument("--rollout", type=int, default=64)
ap.add_argument("--max_levels", type=int, default=0); ap.add_argument("--no_probe", action="store_true")
args = ap.parse_args()
os.makedirs(f"{args.out}/instances", exist_ok=True)
levels = list_levels(tuple(args.tiers.split(",")))
if args.max_levels: levels = levels[: args.max_levels]
cfg = PPOConfig(num_envs=args.num_envs, rollout=args.rollout, num_updates=args.probe_updates, ext_coef=1.0, int_coef=0.0)
ev = Evaluator("m", cfg)
pool, L, M = [], [], []
for lv in levels:
    st, sp, ep = load_level(lv, "m")
    seen = set()
    for cls in CLASSES:
        for seed in range(args.seeds):
            out = inject(cls, st, sp, ep, np.random.default_rng(seed))
            if out is None: continue
            mst, meta = out
            key = (cls, bool(meta.target_is_poly), int(meta.target_idx))
            if key in seen: continue
            seen.add(key)
            name = f"{lv.replace('/', '_').replace('.json', '')}__{cls}__{seed}"
            path = f"{args.out}/instances/{name}.json"
            save_level_json(path, mst, sp, ep)
            pool.append(dict(name=name, level=lv, cls=cls, seed=seed, path=path, meta=meta_to_json(meta), base_hash=level_hash(st), hash=level_hash(mst)))
            L.append(mst); M.append(meta)
print(f"{len(pool)} instances from {len(levels)} levels", flush=True)
if not args.no_probe:
    for name, obj, ext, intc in [("task_ppo", ZeroObjective(), 1.0, 0.0), ("count", CountObjective(), 0.0, 1.0)]:
        ev.cfg = PPOConfig(num_envs=args.num_envs, rollout=args.rollout, num_updates=args.probe_updates, ext_coef=ext, int_coef=intc)
        res = ev.run(obj, L, M, seeds=(0,), chunk=16, log=lambda s: print(s, flush=True))
        for i, p in enumerate(pool):
            p[f"probe_{name}_first"] = float(res["first_witness"][i, 0]); p[f"probe_{name}_solved"] = float(res["solved_frac"][i, 0])
        print(name, summarize(res["first_witness"], res["budget"]), flush=True)
    pool_budget = ev.cfg.budget
else:
    pool_budget = None
json.dump(dict(budget=pool_budget, cfg=dict(num_envs=args.num_envs, rollout=args.rollout, updates=args.probe_updates), instances=pool), open(f"{args.out}/pool.json", "w"), indent=1)
by_cls = {c: sum(1 for p in pool if p["cls"] == c) for c in CLASSES}
print("per class:", by_cls)
