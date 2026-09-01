"""C-Ach suite: evaluate tester programs on Craftax target-reaching tasks (world key x achievement) under a simulator budget.
Same protocol as scripts/eval_baselines.py: score = 1 - min(steps, budget)/budget averaged over tasks and seeds; detection =
fraction of tasks whose target achievement is unlocked within budget; coverage recorded. Needs the craftax env
(.venv_craftax locally; conda_envs/craftax on Hyak).

    python scripts/eval_craftax.py --programs random,repeat_random,goexplore --program_files runs/evolved/candidates/cand1_gen037_0.6994.py \
        --worlds 4 --achievements 0,1,5,9,11,20,29,61,28,49 --workers 32 --steps 2048 --slots 64 --seeds 0,1,2 --chunk 4 --out runs/hyak/eval_craftax.json
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from autodev.craftax_adapter import CraftaxEnvShim, make_feature_fn, witness, meta_for, make_task_keys, CraftaxMeta
from autodev.tester import TesterConfig, make_tester, BUILTIN_PROGRAMS, SourceProgram

ap = argparse.ArgumentParser()
ap.add_argument("--programs", default="random,repeat_random,goexplore"); ap.add_argument("--program_files", default="")
ap.add_argument("--worlds", type=int, default=4); ap.add_argument("--achievements", default="0,1,5,9,11,20,29,61,28,49"); ap.add_argument("--classic", action="store_true")
ap.add_argument("--workers", type=int, default=32); ap.add_argument("--steps", type=int, default=2048); ap.add_argument("--slots", type=int, default=64)
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--chunk", type=int, default=4); ap.add_argument("--out", default="runs/eval_craftax.json")
a = ap.parse_args()
try:
    from craftax.craftax.constants import Achievement as ACH
    ach_names = {m.value: m.name for m in ACH}
except Exception:
    ach_names = {}
env = CraftaxEnvShim(classic=a.classic); ep = env.default_params; feats = make_feature_fn(a.classic)
cfg = TesterConfig(num_workers=a.workers, steps=a.steps, slots=a.slots)
oracle = lambda s, r, d, m: witness(s, r, d, m)[None]
keys = make_task_keys(a.worlds); achs = [int(x) for x in a.achievements.split(",")]
tasks = [(w, k) for w in range(a.worlds) for k in achs]
names = [f"world{w}__{ach_names.get(k, k)}" for w, k in tasks]
levels = jnp.stack([keys[w] for w, _ in tasks]); metas = jax.tree.map(lambda *x: jnp.stack(x), *[meta_for(k) for _, k in tasks])
seeds = [int(s) for s in a.seeds.split(",")]
progs = {p: BUILTIN_PROGRAMS[p]() for p in a.programs.split(",") if p}
for f in a.program_files.split(","):
    if f: progs[os.path.basename(f)] = SourceProgram(open(f).read())
res = {}
for name, P in progs.items():
    run = jax.jit(jax.vmap(make_tester(env, ep, None, feats, P, cfg, oracle, 1), in_axes=(0, 0, 0)))
    n = len(tasks); fh = np.full((n, len(seeds)), float(cfg.budget)); cov = np.zeros((n, len(seeds))); t0 = time.time()
    for si, seed in enumerate(seeds):
        for c0 in range(0, n, a.chunk):
            idx = list(range(c0, min(n, c0 + a.chunk)))
            ii = jnp.array(idx); out = run(jax.random.split(jax.random.PRNGKey(7919 * seed + c0), len(idx)), levels[ii], jax.tree.map(lambda x: x[ii], metas)); jax.block_until_ready(out)
            fh[idx, si] = np.array(out["first_hit"][:, 0]); cov[idx, si] = np.array(out["coverage"])
    norm = 1.0 - np.minimum(fh, cfg.budget) / cfg.budget
    per_ach = {ach_names.get(k, str(k)): float(np.mean([(fh[i] < cfg.budget).mean() for i, (w, kk) in enumerate(tasks) if kk == k])) for k in achs}
    res[name] = dict(score=float(norm.mean()), detected=float((fh < cfg.budget).mean()), detected_per_achievement=per_ach, first_hit=fh.tolist(), coverage=cov.tolist(),
                     names=names, budget=cfg.budget, wall=time.time() - t0, cfg=dict(workers=a.workers, steps=a.steps, slots=a.slots, worlds=a.worlds, classic=a.classic))
    print(f"{name:26s} score {res[name]['score']:.3f} detected {res[name]['detected']:.2f} | " + ", ".join(f"{k[:14]}:{v:.2f}" for k, v in per_ach.items()) + f" ({res[name]['wall']:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
