"""Thin utilities over Kinetix/jax2d: level loading, padding to one static config, features, contacts, geometry."""
from __future__ import annotations
import os, glob, hashlib, json
import numpy as np
import jax, jax.numpy as jnp
import kinetix
from kinetix.environment.env import make_kinetix_env
from kinetix.environment.env_state import EnvState, EnvParams, StaticEnvParams
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.environment.utils import static_env_params_from_size
from kinetix.util.saving import load_from_json_file, expand_env_state, export_env_state_to_json
from jax2d.engine import PhysicsEngine

WORLD = 5.0
ROLE_NONE, ROLE_AGENT, ROLE_GOAL, ROLE_HAZARD = 0, 1, 2, 3
LEVELS_DIR = os.path.join(os.path.dirname(kinetix.__file__), "levels")


def list_levels(tiers=("s", "m")) -> list[str]:
    out = []
    for t in tiers:
        out += sorted(f"{t}/{os.path.basename(p)}" for p in glob.glob(os.path.join(LEVELS_DIR, t, "*.json")))
    return out


def load_level(rel: str, size: str = "m"):
    """Load a bundled level and pad it to the static params of `size` ('s'|'m'|'l'). Returns (state, static_params, env_params)."""
    state, sp_level, ep = load_from_json_file(rel)
    sp = static_env_params_from_size(size)
    if (sp_level.num_polygons, sp_level.num_circles, sp_level.num_joints, sp_level.num_thrusters) != (
        sp.num_polygons, sp.num_circles, sp.num_joints, sp.num_thrusters):
        state = expand_env_state(state, sp)
    return state, sp, ep


def make_env(sp: StaticEnvParams, ep: EnvParams, action_type=ActionType.DISCRETE,
             obs_type=ObservationType.SYMBOLIC_FLAT, auto_reset: bool = False):
    """One env per static config; the level is supplied per call via override_reset_state."""
    return make_kinetix_env(action_type, obs_type, reset_fn=None, env_params=ep, static_env_params=sp, auto_reset=auto_reset)


def level_hash(state: EnvState) -> str:
    leaves = jax.tree_util.tree_leaves(jax.tree.map(lambda x: np.asarray(x), state))
    h = hashlib.sha256()
    for l in leaves:
        h.update(np.ascontiguousarray(l).tobytes())
    return h.hexdigest()[:16]


# ---------------- roles / masks ----------------
def role_masks(state: EnvState, role: int):
    pm = (state.polygon_shape_roles == role) & state.polygon.active
    cm = (state.circle_shape_roles == role) & state.circle.active
    return pm, cm


def _masked_mean(pos_p, pos_c, pm, cm):
    w = jnp.concatenate([pm, cm]).astype(jnp.float32)
    x = jnp.concatenate([pos_p, pos_c], axis=0)
    return (x * w[:, None]).sum(0) / jnp.maximum(w.sum(), 1.0)


def dynamic_masks(state: EnvState):
    """Shapes that can move (finite mass) and are active."""
    pm = (state.polygon.inverse_mass > 0) & state.polygon.active
    cm = (state.circle.inverse_mass > 0) & state.circle.active
    return pm, cm


def dyn_centroid(state: EnvState):
    pm, cm = dynamic_masks(state)
    return _masked_mean(state.polygon.position, state.circle.position, pm, cm)


def dyn_velocity(state: EnvState):
    pm, cm = dynamic_masks(state)
    return _masked_mean(state.polygon.velocity, state.circle.velocity, pm, cm)


def dyn_max_speed(state: EnvState):
    pm, cm = dynamic_masks(state)
    sp = jnp.linalg.norm(state.polygon.velocity, axis=-1) * pm
    sc = jnp.linalg.norm(state.circle.velocity, axis=-1) * cm
    return jnp.maximum(sp.max(), sc.max())


def role_centroid(state: EnvState, role: int):
    pm, cm = role_masks(state, role)
    return _masked_mean(state.polygon.position, state.circle.position, pm, cm)


def green_blue_distance(state: EnvState):
    """Distance between the green (role 1) and blue (role 2) centroids — the quantity the engine's dense reward tracks."""
    return jnp.linalg.norm(role_centroid(state, ROLE_AGENT) - role_centroid(state, ROLE_GOAL))


# backwards-compatible names
agent_centroid, agent_velocity, agent_max_speed = dyn_centroid, dyn_velocity, dyn_max_speed
def goal_centroid(state: EnvState): return role_centroid(state, ROLE_GOAL)


# ---------------- geometry ----------------
def point_in_polygon(p, state: EnvState, idx):
    """Is world point p strictly inside (convex) polygon idx? Handles rotation and n_vertices<=max."""
    pos = state.polygon.position[idx]
    rot = state.polygon.rotation[idx]
    verts = state.polygon.vertices[idx]
    n = state.polygon.n_vertices[idx]
    c, s = jnp.cos(rot), jnp.sin(rot)
    d = p - pos
    local = jnp.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    k = jnp.arange(verts.shape[0])
    valid = k < n
    nxt = (k + 1) % jnp.maximum(n, 1)
    v0 = verts
    v1 = verts[nxt]
    e = v1 - v0
    w = local[None, :] - v0
    cross = e[:, 0] * w[:, 1] - e[:, 1] * w[:, 0]
    inside_cw = jnp.all(jnp.where(valid, cross < 0, True))
    inside_ccw = jnp.all(jnp.where(valid, cross > 0, True))
    return (inside_cw | inside_ccw) & state.polygon.active[idx]


def polygon_world_extent(state: EnvState, idx):
    """Axis-aligned bounding box (xmin, xmax, ymin, ymax) of polygon idx in world space (numpy, host side)."""
    pos = np.asarray(state.polygon.position[idx]); rot = float(state.polygon.rotation[idx])
    verts = np.asarray(state.polygon.vertices[idx])[: int(state.polygon.n_vertices[idx])]
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    w = verts @ R.T + pos
    return float(w[:, 0].min()), float(w[:, 0].max()), float(w[:, 1].min()), float(w[:, 1].max())


def classify_wall(state: EnvState, idx: int):
    """'floor' | 'ceiling' | 'left' | 'right' | None for the boundary polygon idx, by where its extent center sits."""
    a, b, c, d = polygon_world_extent(state, idx)
    cx, cy = (a + b) / 2, (c + d) / 2
    if cy < 0.0:
        return "floor"
    if cy > WORLD:
        return "ceiling"
    if cx < 0.0:
        return "left"
    if cx > WORLD:
        return "right"
    return None


def world_box(state: EnvState, sp: StaticEnvParams):
    """Inner faces of the 4 fixated boundary polygons -> (xmin, xmax, ymin, ymax). Missing walls fall back to [0,WORLD]."""
    xmin, xmax, ymin, ymax = 0.0, WORLD, 0.0, WORLD
    for i in range(sp.num_static_fixated_polys):
        if not bool(state.polygon.active[i]):
            continue
        kind = classify_wall(state, i)
        a, b, c, d = polygon_world_extent(state, i)
        if kind == "floor":
            ymin = max(ymin, d)
        elif kind == "ceiling":
            ymax = min(ymax, c)
        elif kind == "left":
            xmin = max(xmin, b)
        elif kind == "right":
            xmax = min(xmax, a)
    return xmin, xmax, ymin, ymax


# ---------------- contacts ----------------
def contact_flags(state: EnvState, engine: PhysicsEngine):
    """Per-pair contact flags from the accumulated manifolds in the state (set by the last physics substep)."""
    rr = state.acc_rr_manifolds.active.any(axis=-1)   # (nrr,)
    cr = state.acc_cr_manifolds.active                 # (ncr,)
    cc = state.acc_cc_manifolds.active                 # (ncc,)
    return rr, cr, cc


def agent_contact_with(state: EnvState, engine: PhysicsEngine, target_is_poly, target_idx):
    """True if any agent shape is in contact with the target shape (poly or circle) this step."""
    rr, cr, cc = contact_flags(state, engine)
    pm, cm = role_masks(state, ROLE_AGENT)
    pp, cp, ccp = engine.poly_poly_pairs, engine.circle_poly_pairs, engine.circle_circle_pairs
    # poly-poly: agent poly with target poly
    hit_pp = (rr & ((pm[pp[:, 0]] & (pp[:, 1] == target_idx)) | (pm[pp[:, 1]] & (pp[:, 0] == target_idx)))).any()
    # circle-poly: agent circle with target poly, or agent poly with target circle
    hit_cp_tp = (cr & cm[cp[:, 0]] & (cp[:, 1] == target_idx)).any()
    hit_cp_tc = (cr & pm[cp[:, 1]] & (cp[:, 0] == target_idx)).any()
    # circle-circle
    hit_cc = (cc & ((cm[ccp[:, 0]] & (ccp[:, 1] == target_idx)) | (cm[ccp[:, 1]] & (ccp[:, 0] == target_idx)))).any()
    poly_hit = hit_pp | hit_cp_tp
    circ_hit = hit_cp_tc | hit_cc
    return jnp.where(target_is_poly, poly_hit, circ_hit)


def role_contact_with(state: EnvState, engine: PhysicsEngine, role: int, target_is_poly, target_idx):
    """True if any active shape of `role` is in contact with the target shape this step."""
    rr, cr, cc = contact_flags(state, engine)
    pm, cm = role_masks(state, role)
    pp, cp, ccp = engine.poly_poly_pairs, engine.circle_poly_pairs, engine.circle_circle_pairs
    hit_pp = (rr & ((pm[pp[:, 0]] & (pp[:, 1] == target_idx)) | (pm[pp[:, 1]] & (pp[:, 0] == target_idx)))).any()
    hit_cp_tp = (cr & cm[cp[:, 0]] & (cp[:, 1] == target_idx)).any()
    hit_cp_tc = (cr & pm[cp[:, 1]] & (cp[:, 0] == target_idx)).any()
    hit_cc = (cc & ((cm[ccp[:, 0]] & (ccp[:, 1] == target_idx)) | (cm[ccp[:, 1]] & (ccp[:, 0] == target_idx)))).any()
    return jnp.where(target_is_poly, hit_pp | hit_cp_tp, hit_cp_tc | hit_cc)


def green_blue_contact_excluding(state: EnvState, engine: PhysicsEngine, target_is_poly, target_idx):
    """Any green(1)-blue(2) contact this step that does NOT involve the target shape (i.e. a legitimate win)."""
    rr, cr, cc = contact_flags(state, engine)
    rp, rc = state.polygon_shape_roles * state.polygon.active, state.circle_shape_roles * state.circle.active
    pp, cp, ccp = engine.poly_poly_pairs, engine.circle_poly_pairs, engine.circle_circle_pairs
    def excl_p(idx): return ~(target_is_poly & (idx == target_idx))
    def excl_c(idx): return ~((~target_is_poly) & (idx == target_idx))
    gb_pp = (rr & (rp[pp[:, 0]] * rp[pp[:, 1]] == 2) & excl_p(pp[:, 0]) & excl_p(pp[:, 1])).any()
    gb_cp = (cr & (rc[cp[:, 0]] * rp[cp[:, 1]] == 2) & excl_c(cp[:, 0]) & excl_p(cp[:, 1])).any()
    gb_cc = (cc & (rc[ccp[:, 0]] * rc[ccp[:, 1]] == 2) & excl_c(ccp[:, 0]) & excl_c(ccp[:, 1])).any()
    return gb_pp | gb_cp | gb_cc


def can_collide(state: EnvState, a_is_poly, a_idx, b_is_poly, b_idx):
    """Engine's own collision mask (jointed shapes do not collide). Global index = polygon idx, or num_polygons + circle idx."""
    npoly = state.polygon.active.shape[0]
    ga = jnp.where(a_is_poly, a_idx, npoly + a_idx); gb = jnp.where(b_is_poly, b_idx, npoly + b_idx)
    return state.collision_matrix[ga, gb] > 0


def any_dynamic_inside(state: EnvState, target_is_poly, target_idx):
    """Any dynamic shape's centroid strictly inside the target polygon / circle, counting only shapes the engine would
    collide with the target (jointed / collision-disabled pairs are excluded)."""
    pm, cm = dynamic_masks(state)
    npoly = state.polygon.active.shape[0]
    def inside(p):
        in_poly = point_in_polygon(p, state, target_idx)
        in_circ = (jnp.linalg.norm(p - state.circle.position[target_idx]) < state.circle.radius[target_idx]) & state.circle.active[target_idx]
        return jnp.where(target_is_poly, in_poly, in_circ)
    cp = jax.vmap(lambda i: can_collide(state, True, i, target_is_poly, target_idx))(jnp.arange(npoly))
    cc = jax.vmap(lambda i: can_collide(state, False, i, target_is_poly, target_idx))(jnp.arange(state.circle.active.shape[0]))
    ip = jax.vmap(inside)(state.polygon.position) & pm & cp
    ic = jax.vmap(inside)(state.circle.position) & cm & cc
    return ip.any() | ic.any()


def bounding_radius_p(state: EnvState):
    return jnp.linalg.norm(state.polygon.vertices, axis=-1).max(-1)


def min_gap_green_to(state: EnvState, mask_p, mask_c):
    """Approximate minimum surface gap between any green (role 1) shape and any shape in the masks (bounding radii)."""
    gp, gc = role_masks(state, ROLE_AGENT)
    rp = bounding_radius_p(state); rc = state.circle.radius
    P = jnp.concatenate([state.polygon.position, state.circle.position]); R = jnp.concatenate([rp, rc])
    G = jnp.concatenate([gp, gc]); Mk = jnp.concatenate([mask_p, mask_c])
    d = jnp.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1) - R[:, None] - R[None, :]
    d = jnp.where(G[:, None] & Mk[None, :], d, jnp.inf)
    return d.min()


def any_dynamic_outside(state: EnvState, box, margin=0.1):
    pm, cm = dynamic_masks(state)
    def out(p): return (p[0] < box[0] - margin) | (p[0] > box[1] + margin) | (p[1] < box[2] - margin) | (p[1] > box[3] + margin)
    return (jax.vmap(out)(state.polygon.position) & pm).any() | (jax.vmap(out)(state.circle.position) & cm).any()


def shape_position(state: EnvState, is_poly, idx):
    return jnp.where(is_poly, state.polygon.position[idx], state.circle.position[idx])


def save_level_json(path, state, sp, ep):
    export_env_state_to_json(path, state, sp, ep)
