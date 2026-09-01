"""Jumanji games as a substrate (Sokoban first): the harness contract (reset(rng, ep, level)/step(rng, state, action, ep, level),
no auto-reset), the 11-slot feature head, and milestone witnesses. `level` is a PRNG key fixing the puzzle instance."""
from __future__ import annotations
import jax, jax.numpy as jnp
from flax import struct

KIND = {"Sokoban-v0": 0}


@struct.dataclass
class JumanjiMeta:
    kind: jnp.ndarray        # int32
    threshold: jnp.ndarray   # float32: boxes-on-target milestone


def meta_for(env_name, target):
    thr = float(target[0]) if isinstance(target, (list, tuple)) and target else float(target)
    return JumanjiMeta(jnp.int32(KIND[env_name]), jnp.float32(thr))


class JumanjiEnvShim:
    def __init__(self, env_name):
        import jumanji
        self.env = jumanji.make(env_name); self.env_name = env_name
        self.default_params = None
        class _S: num_motor_bindings = 0
        self.static_env_params = _S()
        self._n_act = int(self.env.action_spec.num_values)
    def action_space(self, ep):
        class _A: n = self._n_act
        return _A()
    def reset(self, rng, ep, level=None, override_reset_state=None):
        st, ts = self.env.reset(level if level is not None else rng)
        return st.variable_grid, st
    def step(self, rng, state, action, ep, level=None):
        st, ts = self.env.step(state, action)
        done = jnp.asarray(ts.step_type == 2)          # dm_env StepType.LAST
        return st.variable_grid, st, jnp.asarray(ts.reward, jnp.float32), done, {}
    def get_obs(self, state):
        # observation reconstruction is only used after restores; the harness recomputes features from state, so a cheap proxy suffices
        return state.variable_grid


def _boxes_on_targets(s):
    # Jumanji Sokoban constants: fixed_grid TARGET=2, WALL=1; variable_grid BOX=4, AGENT=3
    return jnp.sum((s.fixed_grid == 2) & (s.variable_grid == 4)).astype(jnp.float32)   # TARGET=2 (fixed), BOX=4 (variable)


def make_feature_fn(env_name):
    def feats(s):
        N = jnp.float32(s.fixed_grid.shape[0])
        ay = jnp.float32(s.agent_location[0]) / N * 5.0; ax = jnp.float32(s.agent_location[1]) / N * 5.0
        bot = _boxes_on_targets(s); nb = jnp.sum(s.variable_grid == 4).astype(jnp.float32)
        t = jnp.float32(s.step_count) / 200.0
        head = jnp.array([ax, ay, 0.0, 0.0, bot, ax, ay, (nb - bot), t, 2.5, 5.0])
        tail = jnp.array([bot, nb, jnp.float32(s.step_count)])
        return jnp.concatenate([head, tail]).astype(jnp.float32)
    return feats


def witness(state, reward, done, meta: JumanjiMeta):
    return jnp.where(meta.kind == 0, _boxes_on_targets(state) >= meta.threshold, False)
