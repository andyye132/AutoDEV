"""XLand-MiniGrid (xminigrid) as the HELD-OUT engine: MiniGrid-style goal-reaching tasks the multi-engine search never
scores. Harness contract: reset(rng, ep, level) -> (obs, state), step(rng, state, action, ep, level) -> (obs, state, reward, done,
info), where our `state` is xminigrid's TimeStep (a pytree) and `level` is the reset key selecting the layout/seed.
"""
from __future__ import annotations
import jax, jax.numpy as jnp
from flax import struct

ENVS = ["MiniGrid-Empty-8x8", "MiniGrid-EmptyRandom-8x8", "MiniGrid-FourRooms", "MiniGrid-DoorKey-8x8", "MiniGrid-Unlock", "MiniGrid-LockedRoom", "MiniGrid-BlockedUnlockPickUp"]


@struct.dataclass
class XMeta:
    kind: jnp.ndarray     # int32 index into ENVS (informational)
    threshold: jnp.ndarray


def meta_for(env_name, target=None):
    return XMeta(jnp.int32(ENVS.index(env_name) if env_name in ENVS else -1), jnp.float32(0.0))


class XMiniGridShim:
    def __init__(self, env_name="MiniGrid-Empty-8x8"):
        import xminigrid
        self.env, self.default_params = xminigrid.make(env_name); self.env_name = env_name
        class _S: num_motor_bindings = 0
        self.static_env_params = _S()
        self._n = int(self.env.num_actions(self.default_params))
    def action_space(self, ep):
        n = self._n
        class _A: pass
        a = _A(); a.n = n; return a
    def reset(self, rng, ep, level=None, override_reset_state=None):
        ts = self.env.reset(ep, level if level is not None else rng)
        return ts.observation, ts
    def step(self, rng, state, action, ep, level=None):
        ts = self.env.step(ep, state, action)
        done = ts.step_type == 2                       # StepType.LAST
        return ts.observation, ts, ts.reward, done, {}
    def get_obs(self, state):
        return state.observation


def _hash_grid(g, k=12):
    flat = g.reshape(-1).astype(jnp.float32)
    proj = jax.random.normal(jax.random.PRNGKey(4321), (flat.shape[0], k)) / jnp.sqrt(flat.shape[0])
    return jnp.tanh(flat @ proj)


def make_feature_fn(env_name):
    def feats(ts):
        s = ts.state
        H = jnp.float32(s.grid.shape[0]); W = jnp.float32(s.grid.shape[1])
        pos = s.agent.position.astype(jnp.float32) / jnp.array([H, W]) * 5.0; d = jnp.float32(s.agent.direction) / 4.0   # [0, 5] box (substrate.HEAD_WORLD)
        t = jnp.float32(s.step_num) / 500.0
        carried = jnp.asarray(s.agent.pocket if hasattr(s.agent, "pocket") else 0).reshape(-1).astype(jnp.float32)[:1]
        head = jnp.concatenate([pos, jnp.zeros(2), (carried > 0).astype(jnp.float32) * 2.5, pos, jnp.array([5.0]), jnp.array([t]), jnp.array([2.5, 2.5])])   # [4] progress: carrying an object
        carry = jnp.asarray(s.agent.pocket if hasattr(s.agent, "pocket") else 0).reshape(-1).astype(jnp.float32)[:2]
        carry = jnp.pad(carry, (0, 2 - carry.shape[0]))
        tail = jnp.concatenate([jnp.array([d]), carry / 12.0, _hash_grid(ts.observation), _hash_grid(s.grid[..., 0] if s.grid.ndim == 3 else s.grid, 8)])
        return jnp.concatenate([head, tail]).astype(jnp.float32)
    return feats


def witness(state, reward, done, meta: XMeta):
    """Goal reached: xminigrid pays a positive reward exactly when the goal condition is met."""
    return reward > 0.0
