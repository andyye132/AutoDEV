"""Witness-state oracles: a defect is 'found' the first time its witness predicate is true.
All functions are pure JAX and vmappable over batched (state, meta)."""
from __future__ import annotations
import jax, jax.numpy as jnp
from .kinetix_utils import (dyn_max_speed, dyn_centroid, any_dynamic_inside, any_dynamic_outside, role_contact_with,
                            min_gap_green_to, shape_position, role_centroid, ROLE_AGENT)
from .defects import DefectMeta


def witness(state, reward, done, meta: DefectMeta, engine):
    """Evaluate the witness predicate for meta.cls on the post-step (pre-reset) state. Returns a bool scalar."""
    # OOB: any moving shape leaves the playable box
    oob = any_dynamic_outside(state, meta.box)
    # TUNNEL: any moving shape's centroid is inside the (now non-solid) static obstacle
    tunnel = any_dynamic_inside(state, meta.target_is_poly, meta.target_idx)
    # SHORTCUT: the episode was won by touching the injected goal, not the intended one
    near_target = jnp.linalg.norm(role_centroid(state, ROLE_AGENT) - shape_position(state, meta.target_is_poly, meta.target_idx)) < meta.contact_r
    shortcut = (reward > 0.5) & near_target & (min_gap_green_to(state, meta.goal_p, meta.goal_c) > 0.05)
    # ANOMALY: runaway speed or numerical blow-up
    anomaly = (dyn_max_speed(state) > meta.speed_thr) | jnp.isnan(dyn_centroid(state)).any()
    # DISARMED: a green shape touches the former hazard and the episode does not end
    disarmed = role_contact_with(state, engine, ROLE_AGENT, meta.target_is_poly, meta.target_idx) & (~done)
    # GOAL: the level's own win condition on an unmodified level
    goal = reward > 0.5
    preds = jnp.stack([oob, tunnel, shortcut, anomaly, disarmed, goal])
    return jnp.where(meta.cls >= 0, preds[jnp.clip(meta.cls, 0, 5)], False)
