"""Generic engine-bug oracles for CLEAN levels (no planting). Flags per step:
   0 OOB        a moving shape left the playable box
   1 PENETRATE  a moving shape's centroid is inside a static solid
   2 JOINT_TEAR a joint's two anchors separated beyond tolerance
   3 BLOWUP     speed / angular speed beyond calibrated thresholds, or NaN
Thresholds are calibrated per level on clean random + fixed-action play (see calibrate_anomaly_thresholds)."""
from __future__ import annotations
import jax, jax.numpy as jnp, numpy as np
from flax import struct
from .kinetix_utils import (dynamic_masks, any_dynamic_outside, point_in_polygon, dyn_centroid)

ANOMALIES = ["OOB", "PENETRATE", "JOINT_TEAR", "BLOWUP"]
ANOMALY_MIN_STREAK = (1, 3, 2, 3)   # persistence (consecutive steps) required to count a hit, per anomaly


@struct.dataclass
class AnomalyThr:
    box: jnp.ndarray        # (4,)
    joint_tol: jnp.ndarray  # float
    speed_thr: jnp.ndarray  # float
    angvel_thr: jnp.ndarray # float


def joint_separation(state, sp):
    npoly = sp.num_polygons
    def body(idx):
        is_p = idx < npoly
        pos = jnp.where(is_p, state.polygon.position[jnp.minimum(idx, npoly - 1)], state.circle.position[jnp.maximum(idx - npoly, 0)])
        rot = jnp.where(is_p, state.polygon.rotation[jnp.minimum(idx, npoly - 1)], state.circle.rotation[jnp.maximum(idx - npoly, 0)])
        return pos, rot
    def world(pos, rot, rel):
        c, s = jnp.cos(rot), jnp.sin(rot)
        return pos + jnp.stack([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
    pa, ra = jax.vmap(body)(state.joint.a_index); pb, rb = jax.vmap(body)(state.joint.b_index)
    wa = jax.vmap(world)(pa, ra, state.joint.a_relative_pos); wb = jax.vmap(world)(pb, rb, state.joint.b_relative_pos)
    return jnp.linalg.norm(wa - wb, axis=-1) * state.joint.active


def penetration(state, sp):
    """Any dynamic shape centroid strictly inside a static, solid (collision_mode != 0) polygon/circle that the engine would
    collide it with (state.collision_matrix; jointed pairs are excluded)."""
    pm, cm = dynamic_masks(state)
    static_p = (state.polygon.inverse_mass == 0) & state.polygon.active & (state.polygon.collision_mode != 0)
    static_c = (state.circle.inverse_mass == 0) & state.circle.active & (state.circle.collision_mode != 0)
    npoly = sp.num_polygons
    P = jnp.concatenate([state.polygon.position, state.circle.position]); D = jnp.concatenate([pm, cm])
    G = jnp.arange(P.shape[0])   # global indices
    def inside_any(p, g):
        cm_row = state.collision_matrix[g]
        ip = jax.vmap(lambda j: point_in_polygon(p, state, j))(jnp.arange(npoly)) & static_p & (cm_row[:npoly] > 0)
        ic = (jnp.linalg.norm(p[None] - state.circle.position, axis=-1) < state.circle.radius) & static_c & (cm_row[npoly:] > 0)
        return ip.any() | ic.any()
    return (jax.vmap(inside_any)(P, G) & D).any()


def anomaly_flags(state, sp, thr: AnomalyThr):
    oob = any_dynamic_outside(state, thr.box)
    pen = penetration(state, sp)
    tear = (joint_separation(state, sp) > thr.joint_tol).any()
    pm, cm = dynamic_masks(state)
    speed = jnp.maximum((jnp.linalg.norm(state.polygon.velocity, axis=-1) * pm).max(), (jnp.linalg.norm(state.circle.velocity, axis=-1) * cm).max())
    ang = jnp.maximum((jnp.abs(state.polygon.angular_velocity) * pm).max(), (jnp.abs(state.circle.angular_velocity) * cm).max())
    blow = (speed > thr.speed_thr) | (ang > thr.angvel_thr) | jnp.isnan(dyn_centroid(state)).any()
    return jnp.stack([oob, pen, tear, blow])


def raw_stats(state, sp):
    """Per-step raw quantities used for calibration: (max joint sep, max speed, max ang speed)."""
    pm, cm = dynamic_masks(state)
    speed = jnp.maximum((jnp.linalg.norm(state.polygon.velocity, axis=-1) * pm).max(), (jnp.linalg.norm(state.circle.velocity, axis=-1) * cm).max())
    ang = jnp.maximum((jnp.abs(state.polygon.angular_velocity) * pm).max(), (jnp.abs(state.circle.angular_velocity) * cm).max())
    return jnp.stack([joint_separation(state, sp).max(), speed, ang])


def make_thresholds(box, stats_max, joint_floor=0.2, speed_mult=2.0, speed_floor=12.0, ang_mult=2.0, ang_floor=45.0):
    """stats_max: (3,) max observed on the clean level under scripted play. Thresholds sit well above normal play."""
    return AnomalyThr(jnp.array(box, jnp.float32), jnp.float32(max(joint_floor, 3.0 * float(stats_max[0]))),
                      jnp.float32(max(speed_floor, speed_mult * float(stats_max[1]))), jnp.float32(max(ang_floor, ang_mult * float(stats_max[2]))))
