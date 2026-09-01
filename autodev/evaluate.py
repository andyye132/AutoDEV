"""Run an objective over a set of defect instances and summarize steps-to-first-witness."""
from __future__ import annotations
import json, os, time
import numpy as np, jax, jax.numpy as jnp
from jax2d.engine import PhysicsEngine
from kinetix.util.saving import import_env_state_from_json
from kinetix.environment.utils import static_env_params_from_size
from kinetix.util.saving import expand_env_state
from .kinetix_utils import make_env, load_level
from .defects import DefectMeta, CLASSES
from .features import make_feature_fn
from .ppo import make_train, PPOConfig


def meta_to_json(m: DefectMeta):
    return dict(cls=int(m.cls), target_is_poly=bool(m.target_is_poly), target_idx=int(m.target_idx), box=[float(x) for x in np.array(m.box)],
                speed_thr=float(m.speed_thr), contact_r=float(m.contact_r), goal_p=[bool(x) for x in np.array(m.goal_p)], goal_c=[bool(x) for x in np.array(m.goal_c)])


def meta_from_json(d):
    return DefectMeta(jnp.int32(d["cls"]), jnp.bool_(d["target_is_poly"]), jnp.int32(d["target_idx"]), jnp.array(d["box"], jnp.float32),
                      jnp.float32(d["speed_thr"]), jnp.float32(d["contact_r"]), jnp.array(d["goal_p"], bool), jnp.array(d["goal_c"], bool))


def load_instance_level(path, size="m"):
    with open(path) as f:
        st, sp_l, ep = import_env_state_from_json(json.load(f))
    sp = static_env_params_from_size(size)
    if (sp_l.num_polygons, sp_l.num_circles, sp_l.num_joints, sp_l.num_thrusters) != (sp.num_polygons, sp.num_circles, sp.num_joints, sp.num_thrusters):
        st = expand_env_state(st, sp)
    return st, sp, ep


def stack(pytrees):
    return jax.tree.map(lambda *x: jnp.stack(x), *pytrees)


class Evaluator:
    def __init__(self, size="m", cfg: PPOConfig = PPOConfig()):
        _, self.sp, self.ep = load_level("s/h0_weak_thrust.json", size)
        self.env = make_env(self.sp, self.ep, auto_reset=False)
        self.engine = PhysicsEngine(self.sp)
        self.feat_fn, self.F, self.spec, self.describe = make_feature_fn(self.sp, self.ep, self.engine)
        self.cfg = cfg
        self.size = size

    def train_fn(self, objective):
        return jax.jit(jax.vmap(make_train(self.env, self.ep, self.sp, self.engine, objective, self.feat_fn, self.F, self.cfg)))

    def run(self, objective, levels, metas, seeds=(0,), chunk=32, log=print):
        """levels/metas: lists of per-instance pytrees. Returns dict of arrays [n_inst, n_seeds]."""
        train = self.train_fn(objective)
        n = len(levels)
        fw = np.zeros((n, len(seeds))); wit = np.zeros((n, len(seeds)), bool); solved = np.zeros((n, len(seeds)))
        for si, seed in enumerate(seeds):
            for c0 in range(0, n, chunk):
                idx = list(range(c0, min(n, c0 + chunk)))
                L, M = stack([levels[i] for i in idx]), stack([metas[i] for i in idx])
                rngs = jax.random.split(jax.random.PRNGKey(1000 * seed + c0), len(idx))
                t = time.time(); out = train(rngs, L, M); jax.block_until_ready(out)
                fw[idx, si] = np.array(out["first_witness"]); wit[idx, si] = np.array(out["witnessed"]); solved[idx, si] = np.array(out["solved_frac"])
                if log: log(f"  [{objective.name}] seed {seed} instances {idx[0]}-{idx[-1]}: {time.time()-t:.1f}s, detected {wit[idx, si].mean():.2f}, mean steps {fw[idx, si].mean():.0f}")
        return dict(first_witness=fw, witnessed=wit, solved_frac=solved, budget=self.cfg.budget)


def summarize(first_witness, budget):
    """first_witness: array [..., n_seeds] with budget for undetected. Returns detection fraction, mean normalized
    steps-to-first-witness (undetected = 1.0) and the detection-curve AUC (fraction detected as a function of steps, normalized)."""
    x = np.asarray(first_witness, dtype=float).reshape(-1)
    frac = float((x < budget).mean())
    norm = float(np.minimum(x, budget).mean() / budget)
    grid = np.linspace(0, budget, 201)
    curve = np.array([(x < g).mean() for g in grid])
    auc = float(np.trapezoid(curve, grid) / budget)
    return dict(detected=frac, norm_steps=norm, auc=auc, n=int(x.size))
