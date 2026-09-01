"""Craftax as a second substrate for the exploration harness (C-Ach suite): tasks are (world key, achievement index); the
target predicate is the sticky achievement bit in the state. Wraps the Craftax NoAutoReset env into the reset/step contract
`make_tester` expects (reset(rng, ep, level) / step(rng, state, action, ep, level)) where `level` is the task's world key, and
builds a feature vector whose first 11 slots are semantically compatible with the Kinetix layout that the discovered programs
index (position, "velocity", speed, agent position, distance-to-target, time fraction, target position).

Needs the `craftax` package (separate env: .venv_craftax locally, /mmfs1/gscratch/raivn/andy132/conda_envs/craftax on Hyak).
"""
from __future__ import annotations
import jax, jax.numpy as jnp
from flax import struct

MAP_W = 48.0


@struct.dataclass
class CraftaxMeta:
    """Oracle parameters for one task."""
    achievement: jnp.ndarray     # int32 index into state.achievements (67 in Craftax, 22 in Classic)
    level: jnp.ndarray           # int32: reaching this map level also counts if >= 0 (else ignored)


def meta_for(achievement: int, level: int = -1):
    return CraftaxMeta(jnp.int32(achievement), jnp.int32(level))


class CraftaxEnvShim:
    """Adapts a Craftax NoAutoReset env to the harness contract. `level` is a PRNGKey selecting the world."""
    def __init__(self, classic=False, symbolic=True):
        from craftax.craftax_env import make_craftax_env_from_params
        self.env = make_craftax_env_from_params(classic=classic, symbolic=symbolic, auto_reset=False)
        self.default_params = self.env.default_params
        self.classic = classic
        class _S:  # the harness reads env.static_env_params.num_motor_bindings only for the Kinetix no-op; give it a harmless value
            num_motor_bindings = 0
        self.static_env_params = _S()

    def action_space(self, ep): return self.env.action_space(ep)

    def reset(self, rng, ep, level=None, override_reset_state=None):
        """Same world for every worker: the world key is `level` (falls back to rng)."""
        key = level if level is not None else rng
        return self.env.reset(key, ep)

    def step(self, rng, state, action, ep, level=None):
        return self.env.step(rng, state, action, ep)

    def get_obs(self, state):
        return self.env.get_obs(state)


def _hash_window(m, k=12):
    """Random-projection summary of the local 9x9 block-id window (pure JAX, fixed shapes)."""
    flat = m.reshape(-1).astype(jnp.float32)
    proj = jax.random.normal(jax.random.PRNGKey(1234), (flat.shape[0], k)) / jnp.sqrt(flat.shape[0])
    return jnp.tanh(flat @ proj)


def make_feature_fn(classic=False):
    """feats(state) -> (F,) float32. Slots 0..10 mirror the Kinetix layout used by the discovered programs."""
    n_ach = 22 if classic else 67

    def feats(s):
        pos = s.player_position.astype(jnp.float32) / MAP_W * 5.0                 # (2,) in the [0, 5] head box (substrate.HEAD_WORLD)
        lvl = jnp.float32(s.player_level) / 8.0
        t = jnp.float32(s.timestep) / 10000.0
        head = jnp.concatenate([
            pos,                                    # [0:2]  "dyn_pos"
            jnp.zeros(2),                           # [2:4]  "dyn_vel" (no velocity in a grid world)
            jnp.array([s.achievements.astype(jnp.float32).sum() * 0.25]),   # [4] progress scalar (achievements unlocked x 0.25; "max speed" in physics engines)
            pos,                                    # [5:7]  "green_pos" (the agent)
            jnp.array([5.0]),                       # [7]    "distance to goal" (unknown -> far)
            jnp.array([t]),                         # [8]    time fraction
            jnp.array([2.5, 2.5]),                  # [9:11] "goal position" placeholder (box centre)
        ])
        vit = jnp.array([s.player_health / 9.0, s.player_food / 9.0, s.player_drink / 9.0, s.player_energy / 9.0,
                         getattr(s, "player_mana", 0) / 9.0 if hasattr(s, "player_mana") else 0.0], jnp.float32)
        inv = s.inventory
        inv_v = jnp.array([inv.wood, inv.stone, inv.coal, inv.iron, inv.diamond, inv.sapling, inv.pickaxe, inv.sword], jnp.float32)
        inv_v = jnp.clip(inv_v, 0, 10) / 10.0
        ach = s.achievements.astype(jnp.float32)
        lvl_map = s.map[s.player_level] if s.map.ndim == 3 else s.map
        r, c = s.player_position[0], s.player_position[1]
        win = jax.lax.dynamic_slice(jnp.pad(lvl_map, 4), (r, c), (9, 9))
        win_h = _hash_window(win)
        misc = jnp.array([lvl, jnp.float32(s.player_direction) / 4.0, jnp.float32(s.is_sleeping)], jnp.float32)
        return jnp.concatenate([head, vit, inv_v, ach, win_h, misc]).astype(jnp.float32)
    return feats


def witness(state, reward, done, meta: CraftaxMeta):
    """Target predicate: the achievement bit is set (sticky), or the map level was reached."""
    hit = state.achievements[jnp.clip(meta.achievement, 0, state.achievements.shape[0] - 1)]
    lvl = (meta.level >= 0) & (state.player_level >= meta.level)
    return hit | lvl


def make_task_keys(n_worlds: int, base_seed: int = 0):
    """World keys for the task set (deterministic)."""
    return jax.random.split(jax.random.PRNGKey(base_seed), n_worlds)
