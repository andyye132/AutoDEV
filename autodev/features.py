"""Oracle-blind features an exploration objective may read. Everything here is visible to a player (positions,
velocities, colors/roles, contacts); nothing about which shape was mutated."""
from __future__ import annotations
import jax, jax.numpy as jnp
from .kinetix_utils import (dyn_centroid, dyn_velocity, dyn_max_speed, role_centroid, green_blue_distance, contact_flags, ROLE_AGENT, ROLE_GOAL)


def make_feature_fn(sp, ep, engine):
    Np, Nc = sp.num_polygons, sp.num_circles
    nrr, ncr, ncc = len(engine.poly_poly_pairs), len(engine.circle_poly_pairs), len(engine.circle_circle_pairs)
    spec = []
    def add(name, n):
        start = sum(s for _, s in spec)
        spec.append((name, n))
        return start
    add("dyn_pos", 2); add("dyn_vel", 2); add("dyn_max_speed", 1); add("green_pos", 2); add("green_blue_dist", 1); add("time_frac", 1); add("blue_pos", 2)
    add("poly_pos", 2 * Np); add("poly_vel", 2 * Np); add("poly_rot", Np); add("poly_active", Np); add("poly_role", Np)
    add("circle_pos", 2 * Nc); add("circle_vel", 2 * Nc); add("circle_active", Nc); add("circle_role", Nc)
    add("contact_pp", nrr); add("contact_cp", ncr); add("contact_cc", ncc)
    dim = sum(n for _, n in spec)

    def feats(state):
        c = dyn_centroid(state); v = dyn_velocity(state); g = role_centroid(state, ROLE_AGENT); b = role_centroid(state, ROLE_GOAL)
        rr, cr, cc = contact_flags(state, engine)
        parts = [c, v, dyn_max_speed(state)[None], g, green_blue_distance(state)[None], (state.timestep / ep.max_timesteps)[None].astype(jnp.float32), b,
                 state.polygon.position.reshape(-1), state.polygon.velocity.reshape(-1), state.polygon.rotation,
                 state.polygon.active.astype(jnp.float32), state.polygon_shape_roles.astype(jnp.float32),
                 state.circle.position.reshape(-1), state.circle.velocity.reshape(-1), state.circle.active.astype(jnp.float32),
                 state.circle_shape_roles.astype(jnp.float32), rr.astype(jnp.float32), cr.astype(jnp.float32), cc.astype(jnp.float32)]
        return jnp.concatenate([p.astype(jnp.float32) for p in parts])

    def describe():
        lines, start = [], 0
        for name, n in spec:
            lines.append(f"  feats[{start}:{start+n}]  {name}  ({n})")
            start += n
        return "\n".join(lines)
    return feats, dim, spec, describe
