"""Evaluate RL testers (PPO + intrinsic-reward objectives) on the planted-bug split with the same env-step budget as the
program testers. Budget = num_envs * rollout * num_updates env steps per instance."""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax
from autodev.evaluate import Evaluator, summarize
from autodev.search_eval import load_split, ROOT
from autodev.objectives import BASELINES
from autodev.ppo import PPOConfig
from autodev.defects import CLASSES
ap = argparse.ArgumentParser()
ap.add_argument("--objectives", default="count,rnd,icm,curiosity_proximity,zero"); ap.add_argument("--splits", default="search,heldout_levels_instances,heldout_class_instances")
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--size", default="m"); ap.add_argument("--split_path", default=None)
ap.add_argument("--num_envs", type=int, default=128); ap.add_argument("--rollout", type=int, default=64); ap.add_argument("--num_updates", type=int, default=32)
ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--out", default="runs/ppo_baselines.json")
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--ent_coef", type=float, default=0.01)
a = ap.parse_args()
split_path = a.split_path or os.environ.get("AUTODEV_SPLIT", "runs/split.json")
seeds = tuple(int(s) for s in a.seeds.split(","))
res = {}
for name in a.objectives.split(","):
    ext = 1.0 if name == "zero" else 0.0; intc = 0.0 if name == "zero" else 1.0
    cfg = PPOConfig(num_envs=a.num_envs, rollout=a.rollout, num_updates=a.num_updates, ext_coef=ext, int_coef=intc, lr=a.lr, ent_coef=a.ent_coef)
    ev = Evaluator(a.size, cfg)
    label = "ppo_task" if name == "zero" else f"ppo_{name}"; res[label] = {}
    for sp in a.splits.split(","):
        levels, metas, info = load_split(os.path.join(ROOT, split_path), sp, a.size)
        r = ev.run(BASELINES[name](), levels, metas, seeds=seeds, chunk=a.chunk, log=None)
        fh = r["first_witness"]; budget = r["budget"]
        per_class = {c: float(np.mean([1 - min(fh[i].mean(), budget) / budget for i in range(len(info)) if info[i]["cls"] == c])) for c in CLASSES if any(x["cls"] == c for x in info)}
        det_class = {c: float(np.mean([(fh[i] < budget).mean() for i in range(len(info)) if info[i]["cls"] == c])) for c in per_class}
        s = summarize(fh, budget)
        res[label][sp] = dict(score=1 - s["norm_steps"], detected=s["detected"], per_class=per_class, detected_per_class=det_class, n=len(info), budget=budget, first_hit=fh.tolist(), names=[x["name"] for x in info])
        print(f"{label:24s} {sp:26s} score {1-s['norm_steps']:.3f} detected {s['detected']:.2f} per-class-detected {{{', '.join(f'{k}:{v:.2f}' for k, v in det_class.items())}}}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
