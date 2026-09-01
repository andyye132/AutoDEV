"""Gymnax hard-exploration classics as a substrate (MountainCar-v0, Acrobot-v1, DeepSea-bsuite): the harness contract
(reset(rng, ep, level) / step(rng, state, action, ep, level), no auto-reset), the 11-slot semantic feature head, and
target predicates as witness oracles. `level` is a PRNG key that fixes the initial state for all workers of a task.
"""
from __future__ import annotations
import jax, jax.numpy as jnp
from flax import struct

KIND = {"MountainCar-v0": 0, "Acrobot-v1": 1, "DeepSea-bsuite": 2, "CartPole-v1": 3, "Breakout-MinAtar": 4}


@struct.dataclass
class GymnaxMeta:
    kind: jnp.ndarray        # int32: which environment's predicate applies
    threshold: jnp.ndarray   # float32: target threshold (MountainCar: position; Acrobot: tip height; DeepSea: unused)


def meta_for(env_name, target):
    thr = float(target[0]) if isinstance(target, (list, tuple)) and target else float(target) if isinstance(target, (int, float)) else 0.0
    return GymnaxMeta(jnp.int32(KIND[env_name]), jnp.float32(thr))


class GymnaxEnvShim:
    def __init__(self, env_name, **env_kwargs):
        import gymnax
        self.env, self.default_params = gymnax.make(env_name, **env_kwargs); self.env_name = env_name   # e.g. DeepSea-bsuite size=24
        class _S: num_motor_bindings = 0
        self.static_env_params = _S()
    def action_space(self, ep): return self.env.action_space(ep)
    def reset(self, rng, ep, level=None, override_reset_state=None):
        return self.env.reset_env(level if level is not None else rng, ep)      # same start state for every worker of a task
    def step(self, rng, state, action, ep, level=None):
        return self.env.step_env(rng, state, action, ep)                          # no auto-reset; the harness resets on done
    def get_obs(self, state):
        return self.env.get_obs(state)


def make_feature_fn(env_name):
    kind = KIND[env_name]
    def feats(s):
        if kind == 0:      # MountainCar: position in [-1.2, 0.6], velocity in [-0.07, 0.07], goal at 0.5
            pos = (s.position + 1.2) / 1.8 * 5.0; vel = s.velocity / 0.07 * 5.0; t = s.time / 200.0     # [0, 5] box / +-5 (substrate.HEAD_WORLD)
            head = jnp.array([pos, 2.5, vel, 0.0, jnp.abs(vel), pos, 2.5, (0.5 - s.position) / 1.8 * 5.0, t, (0.5 + 1.2) / 1.8 * 5.0, 2.5])
            tail = jnp.array([s.position, s.velocity, jnp.sin(3 * s.position)])
        elif kind == 1:    # Acrobot: tip position from the two joint angles; goal: tip height > 1
            x = jnp.sin(s.joint_angle1) + jnp.sin(s.joint_angle1 + s.joint_angle2); y = -jnp.cos(s.joint_angle1) - jnp.cos(s.joint_angle1 + s.joint_angle2)
            v1 = s.velocity_1 / (4 * jnp.pi); v2 = s.velocity_2 / (9 * jnp.pi); t = s.time / 500.0
            px, py = (x / 2 + 1) * 2.5, (y / 2 + 1) * 2.5                  # tip in the [0, 5] box; velocities in +-5
            head = jnp.array([px, py, v1 * 5, v2 * 5, jnp.maximum(jnp.abs(v1), jnp.abs(v2)) * 5, px, py, (1.0 - y) / 3.0 * 5, t, 2.5, 5.0])
            tail = jnp.array([jnp.cos(s.joint_angle1), jnp.sin(s.joint_angle1), jnp.cos(s.joint_angle2), jnp.sin(s.joint_angle2), v1, v2])
        elif kind == 2:    # DeepSea: (row, column) on an N x N grid; treasure at the bottom-right
            N = jnp.float32(s.action_mapping.shape[0]); r = jnp.float32(s.row) / N; c = jnp.float32(s.column) / N; t = jnp.float32(s.time) / 2000.0
            head = jnp.array([c * 5, r * 5, 0.0, 0.0, r * 5, c * 5, r * 5, (1.0 - c) * 5, t, 5.0, 5.0])   # [4] = depth as the progress scalar
            tail = jnp.array([jnp.float32(s.row), jnp.float32(s.column), jnp.float32(s.bad_episode)])
        elif kind == 4:    # MinAtar Breakout: 10x10 grid; paddle x, ball (x, y), direction, bricks cleared as progress
            bx = jnp.float32(s.ball_x) / 9.0 * 5.0; by = jnp.float32(s.ball_y) / 9.0 * 5.0; px = jnp.float32(s.pos) / 9.0 * 5.0
            cleared = 30.0 - jnp.float32(s.brick_map.sum()); t = jnp.float32(s.time) / 1000.0
            head = jnp.array([bx, by, jnp.float32(s.ball_dir) / 3.0 * 5.0 - 2.5, 0.0, cleared / 30.0 * 5.0, px, 5.0, (30.0 - cleared) / 30.0 * 5.0, t, 2.5, 5.0])
            tail = jnp.array([jnp.float32(s.ball_x), jnp.float32(s.ball_y), jnp.float32(s.pos), jnp.float32(s.ball_dir), cleared, jnp.float32(s.strike)])
        else:              # CartPole (not a target task; kept for completeness)
            head = jnp.array([(s.x / 2.4 + 1) * 2.5, 2.5, s.x_dot / 3 * 5, 0.0, jnp.abs(s.x_dot) / 3 * 5, (s.x / 2.4 + 1) * 2.5, 2.5, 5.0, s.time / 500.0, 2.5, 2.5])
            tail = jnp.array([s.theta, s.theta_dot])
        return jnp.concatenate([head, tail]).astype(jnp.float32)
    return feats


def witness(state, reward, done, meta: GymnaxMeta):
    k = meta.kind
    mc = jnp.where(k == 0, getattr(state, "position", jnp.float32(-9.0)) >= meta.threshold, False)
    if hasattr(state, "joint_angle1"):
        y = -jnp.cos(state.joint_angle1) - jnp.cos(state.joint_angle1 + state.joint_angle2)
        ac = jnp.where(k == 1, y > meta.threshold, False)
    else:
        ac = False
    ds = jnp.where(k == 2, reward > 0.5, False)          # DeepSea: the treasure pays +1 (minus the small cost)
    if hasattr(state, "brick_map"):
        bk = jnp.where(k == 4, (30.0 - jnp.float32(state.brick_map.sum())) >= meta.threshold, False)   # Breakout: bricks cleared milestone
    else:
        bk = False
    return mc | ac | ds | bk
