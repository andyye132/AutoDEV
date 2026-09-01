"""Substrate interface: everything the harness, evaluator, scans and fitness servers need from an engine, so that the same
frozen tester programs run on Kinetix, Craftax, Gymnax, ... without changes.

A Substrate provides
  env      with reset(rng, ep, level) -> (obs, state) and step(rng, state, action, ep, level) -> (obs, state, reward, done, info),
           pure JAX, pytree state (vmap/scan/stack-able); `level` selects the world/level (a level state or a PRNG key).
  ep       env params (bound at construction, as in Kinetix's make_env)
  feat_fn  state -> (F,) float32; slots [0:2] position, [2:4] velocity, [4] max speed, [5:7] agent position, [7] distance to
           target (1 if unknown), [8] time fraction, [9:11] target position, then engine-specific features
  oracle   (state, reward, done, meta) -> (n_flags,) bool  (witness predicate(s); meta is the task's target description)
  tasks(split_path, which) -> (levels list, metas list, info list of dicts with name/cls)   # stacked by the evaluator

Kinetix is the reference implementation; Craftax and Gymnax adapters implement the same shape.
"""
from __future__ import annotations
import os, json
import jax, jax.numpy as jnp

HEAD_WORLD = 5.0   # head convention: positions live in a [0, HEAD_WORLD]^2 box (Kinetix's 5 m world) and velocity / progress scalars in a
                   # comparable +-5 range, so cell sizes chosen on one engine (0.25-0.5 units) mean the same coarse-graining on every engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Substrate:
    name = "abstract"
    def __init__(self): self.env = None; self.ep = None; self.feat_fn = None; self.oracle = None; self.n_flags = 1
    def tasks(self, split_path, which): raise NotImplementedError
    def stack(self, xs): return jax.tree.map(lambda *a: jnp.stack(a), *xs)


class KinetixSubstrate(Substrate):
    name = "kinetix"
    def __init__(self, size="l"):
        from jax2d.engine import PhysicsEngine
        from .kinetix_utils import load_level, make_env
        from .features import make_feature_fn
        from .oracles import witness
        st0, sp, ep = load_level("s/h0_weak_thrust.json", size)
        self.size = size; self.sp = sp; self.ep = ep
        self.env = make_env(sp, ep, auto_reset=False); engine = PhysicsEngine(sp)
        self.feat_fn, self.F, self.spec, self.describe = make_feature_fn(sp, ep, engine)
        self.oracle = lambda s, r, d, meta: witness(s, r, d, meta, engine)[None]; self.n_flags = 1
    def tasks(self, split_path, which):
        from .search_eval import load_split
        return load_split(os.path.join(ROOT, split_path) if not os.path.isabs(split_path) else split_path, which, self.size)


class CraftaxSubstrate(Substrate):
    name = "craftax"
    def __init__(self, classic=False):
        from .craftax_adapter import CraftaxEnvShim, make_feature_fn, witness
        self.env = CraftaxEnvShim(classic=classic); self.ep = self.env.default_params; self.classic = classic
        self.feat_fn = make_feature_fn(classic); self.oracle = lambda s, r, d, meta: witness(s, r, d, meta)[None]; self.n_flags = 1
    def tasks(self, split_path, which):
        """split json: {"which": [{"name":..., "world": int, "achievement": int}], ...}"""
        from .craftax_adapter import make_task_keys, meta_for
        sp = json.load(open(split_path)); rows = sp[which]; keys = make_task_keys(sp.get("n_worlds", 8), sp.get("base_seed", 0))
        levels = [keys[r["world"]] for r in rows]; metas = [meta_for(r["achievement"], r.get("level", -1)) for r in rows]
        info = [dict(name=r["name"], cls=r.get("cls", f"ach{r['achievement']}")) for r in rows]
        return levels, metas, info


class GymnaxSubstrate(Substrate):
    name = "gymnax"
    def __init__(self, env_name="MountainCar-v0", size=None):
        from .gymnax_adapter import GymnaxEnvShim, make_feature_fn, witness
        kw = {"size": int(size)} if size else {}      # DeepSea-bsuite: grid size N (default 8 is trivial for 64 parallel random workers)
        self.env = GymnaxEnvShim(env_name, **kw); self.ep = self.env.default_params; self.env_name = env_name; self.size = size
        self.feat_fn = make_feature_fn(env_name); self.oracle = lambda s, r, d, meta: witness(s, r, d, meta)[None]; self.n_flags = 1
    def tasks(self, split_path, which):
        """split json: {"which": [{"name":..., "seed": int, "target": [...]}]}"""
        from .gymnax_adapter import meta_for
        sp = json.load(open(split_path)); rows = sp[which]
        levels = [jax.random.PRNGKey(r["seed"]) for r in rows]; metas = [meta_for(self.env_name, r["target"]) for r in rows]
        info = [dict(name=r["name"], cls=r.get("cls", self.env_name)) for r in rows]
        return levels, metas, info


class XMiniGridSubstrate(Substrate):
    name = "xminigrid"
    def __init__(self, env_name="MiniGrid-Empty-8x8"):
        from .xminigrid_adapter import XMiniGridShim, make_feature_fn, witness
        self.env = XMiniGridShim(env_name); self.ep = self.env.default_params; self.env_name = env_name
        self.feat_fn = make_feature_fn(env_name); self.oracle = lambda s, r, d, meta: witness(s, r, d, meta)[None]; self.n_flags = 1
    def tasks(self, split_path, which):
        from .xminigrid_adapter import meta_for
        sp = json.load(open(split_path)); rows = [r for r in sp[which] if r.get("env", self.env_name) == self.env_name]
        levels = [jax.random.PRNGKey(r["seed"]) for r in rows]; metas = [meta_for(self.env_name) for r in rows]
        info = [dict(name=r["name"], cls=r.get("cls", self.env_name)) for r in rows]
        return levels, metas, info


def make_substrate(spec: str):
    """'kinetix' | 'kinetix:m' | 'craftax' | 'craftax:classic' | 'gymnax:MountainCar-v0' | 'xminigrid:MiniGrid-DoorKey-8x8'"""
    kind, _, arg = spec.partition(":")
    if kind == "kinetix": return KinetixSubstrate(size=arg or "l")
    if kind == "craftax": return CraftaxSubstrate(classic=(arg == "classic"))
    if kind == "gymnax":
        name, _, size = (arg or "MountainCar-v0").partition(":")        # 'gymnax:DeepSea-bsuite:24' -> size 24
        return GymnaxSubstrate(env_name=name, size=size or None)
    if kind == "xminigrid": return XMiniGridSubstrate(env_name=arg or "MiniGrid-Empty-8x8")
    raise ValueError(spec)
