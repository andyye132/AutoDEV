"""Feasibility ceiling: run the oracle-guided tester on a split; report per-instance reachability within budget."""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from jax2d.engine import PhysicsEngine
from autodev.kinetix_utils import make_env, load_level
from autodev.oracles import witness
from autodev.search_eval import load_split, ROOT
from autodev.evaluate import stack
from autodev.features import make_feature_fn
from autodev.tester import TesterConfig, make_tester
from autodev.guided import guide_fn, GuidedProgram
from autodev.defects import CLASSES
ap = argparse.ArgumentParser(); ap.add_argument("--splits", default="search,heldout_levels_instances,heldout_class_instances"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--workers", type=int, default=128); ap.add_argument("--steps", type=int, default=2048); ap.add_argument("--size", default="m"); ap.add_argument("--split_path", default=None)
ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--out", default="runs/guided_ceiling.json")
a = ap.parse_args(); split_path = a.split_path or os.environ.get("AUTODEV_SPLIT", "runs/split.json")
st0, sp, ep = load_level("s/h0_weak_thrust.json", a.size); env = make_env(sp, ep, auto_reset=False); engine = PhysicsEngine(sp)
feat_fn, F, spec, describe = make_feature_fn(sp, ep, engine)
cfg = TesterConfig(num_workers=a.workers, steps=a.steps, slots=256)
oracle = lambda s, r, d, meta: witness(s, r, d, meta, engine)[None]
run = jax.jit(jax.vmap(make_tester(env, ep, sp, feat_fn, GuidedProgram(), cfg, oracle, 1, guide=guide_fn), in_axes=(0, 0, 0)))
seeds = [int(x) for x in a.seeds.split(",")]; res = {}
for spl in a.splits.split(","):
    levels, metas, info = load_split(os.path.join(ROOT, split_path), spl, a.size)
    n = len(levels); fh = np.full((n, len(seeds)), float(cfg.budget)); t0 = time.time()
    for si, seed in enumerate(seeds):
        for c0 in range(0, n, a.chunk):
            idx = list(range(c0, min(n, c0 + a.chunk)))
            out = run(jax.random.split(jax.random.PRNGKey(4242 * seed + c0), len(idx)), stack([levels[i] for i in idx]), stack([metas[i] for i in idx])); jax.block_until_ready(out)
            fh[idx, si] = np.array(out["first_hit"][:, 0])
    reach = (fh < cfg.budget).any(1)
    per_class = {c: float(np.mean([reach[i] for i in range(n) if info[i]["cls"] == c])) for c in CLASSES if any(x["cls"] == c for x in info)}
    res[spl] = dict(n=n, budget=cfg.budget, reachable_any_seed=float(reach.mean()), reachable_per_seed=float((fh < cfg.budget).mean()), per_class=per_class, first_hit=fh.tolist(), names=[x["name"] for x in info])
    print(f"{spl:26s} n={n:3d} reachable(any seed) {reach.mean():.2f} per-seed {(fh < cfg.budget).mean():.2f} per-class {per_class} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
