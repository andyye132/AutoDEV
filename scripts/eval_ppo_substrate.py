"""Evaluate PPO+intrinsic-motivation RL testers on ANY substrate (Gymnax / MiniGrid / MinAtar / Sokoban / Craftax / Kinetix)
at the same env-step budget as the program testers. The generalization experiment's RL arm. Budget = num_envs*rollout*num_updates.

  python scripts/eval_ppo_substrate.py --substrate xminigrid:MiniGrid-DoorKey-8x8 \
     --split runs/substrates/xminigrid_MiniGrid-DoorKey-8x8_split.json --which search,heldout_levels_instances \
     --objectives count,rnd,icm,curiosity_proximity,zero --seeds 100,101,102 --num_updates 32 --out runs/ppo_doorkey.json
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax
from autodev.substrate import make_substrate
from autodev.objectives import BASELINES
from autodev.ppo import PPOConfig, make_train_substrate
from autodev.evaluate import summarize

ap = argparse.ArgumentParser()
ap.add_argument("--substrate", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--which", default="search,heldout_levels_instances")
ap.add_argument("--objectives", default="count,rnd,icm,curiosity_proximity,zero")
ap.add_argument("--seeds", default="100,101,102")
ap.add_argument("--num_envs", type=int, default=128)
ap.add_argument("--rollout", type=int, default=64)
ap.add_argument("--num_updates", type=int, default=32)   # 32=1x (262144 steps); 128=4x; 512=16x
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()

SUB = make_substrate(a.substrate)
seeds = tuple(int(s) for s in a.seeds.split(","))


def stack(xs):
    return jax.tree.map(lambda *v: np.stack(v) if isinstance(v[0], np.ndarray) else __import__("jax").numpy.stack(v), *xs)


# feature dimension: reset on the first task level and read feat_fn output
_which0 = a.which.split(",")[0]
_levels, _metas, _info = SUB.tasks(a.split, _which0)
_o, _st = SUB.env.reset(jax.random.PRNGKey(0), SUB.ep, _levels[0])
feat_dim = int(SUB.feat_fn(_st).shape[-1])
print(f"substrate={a.substrate} feat_dim={feat_dim} n_act={SUB.env.action_space(SUB.ep).n}", flush=True)

res = {}
for name in a.objectives.split(","):
    ext = 1.0 if name == "zero" else 0.0; intc = 0.0 if name == "zero" else 1.0
    cfg = PPOConfig(num_envs=a.num_envs, rollout=a.rollout, num_updates=a.num_updates, ext_coef=ext, int_coef=intc)
    train = jax.jit(jax.vmap(make_train_substrate(SUB.env, SUB.ep, SUB.oracle, BASELINES[name](), SUB.feat_fn, feat_dim, cfg)))
    label = "ppo_task" if name == "zero" else f"ppo_{name}"
    res[label] = {}
    budget = cfg.budget
    for which in a.which.split(","):
        levels, metas, info = SUB.tasks(a.split, which)
        n = len(levels)
        fw = np.full((n, len(seeds)), float(budget)); wit = np.zeros((n, len(seeds)), bool)
        for si, seed in enumerate(seeds):
            for c0 in range(0, n, a.chunk):
                idx = list(range(c0, min(n, c0 + a.chunk)))
                L = SUB.stack([levels[i] for i in idx]); M = SUB.stack([metas[i] for i in idx])
                rngs = jax.random.split(jax.random.PRNGKey(1000 * seed + c0), len(idx))
                t = time.time(); out = train(rngs, L, M); jax.block_until_ready(out)
                fw[idx, si] = np.array(out["first_witness"]); wit[idx, si] = np.array(out["witnessed"])
                print(f"  [{label}] seed {seed} inst {idx[0]}-{idx[-1]}: {time.time()-t:.1f}s det {wit[idx,si].mean():.2f}", flush=True)
        s = summarize(fw, budget)
        res[label][which] = dict(score=1 - s["norm_steps"], detected=s["detected"], n=n, budget=int(budget),
                                 first_hit=fw.tolist(), names=[x["name"] for x in info])
        print(f"{label:22s} {which:26s} score {1-s['norm_steps']:.3f} detected {s['detected']:.2f}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
print("done ->", a.out)
