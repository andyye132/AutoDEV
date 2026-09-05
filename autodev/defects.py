"""Defect injection on Kinetix levels. Each injector returns (mutated_state, DefectMeta) or None if not applicable.
Classes: OOB (boundary removed), TUNNEL (obstacle made non-solid), SHORTCUT (an extra, easy goal),
ANOMALY (runaway physics via restitution), DISARMED (a hazard made harmless)."""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp
from flax import struct
from .kinetix_utils import (ROLE_NONE, ROLE_AGENT, ROLE_GOAL, ROLE_HAZARD, WORLD, world_box, polygon_world_extent, classify_wall)

CLASSES = ["OOB", "TUNNEL", "SHORTCUT", "ANOMALY", "DISARMED", "GOAL"]   # GOAL = unmodified level, target = the level's own win
CLS = {c: i for i, c in enumerate(CLASSES)}


@struct.dataclass
class DefectMeta:
    cls: jnp.ndarray            # int32 class id (index into CLASSES), -1 for a clean level
    target_is_poly: jnp.ndarray  # bool
    target_idx: jnp.ndarray     # int32 index of the mutated shape
    box: jnp.ndarray            # (4,) xmin,xmax,ymin,ymax of the playable area (for OOB)
    speed_thr: jnp.ndarray      # float32 speed threshold for ANOMALY
    contact_r: jnp.ndarray      # float32 distance threshold for SHORTCUT geometric check
    goal_p: jnp.ndarray         # (Np,) bool: original goal polygons
    goal_c: jnp.ndarray         # (Nc,) bool: original goal circles

    @staticmethod
    def clean(sp=None, box=(0.0, WORLD, 0.0, WORLD)):
        Np = sp.num_polygons if sp is not None else 12; Nc = sp.num_circles if sp is not None else 4
        return DefectMeta(jnp.int32(-1), jnp.bool_(True), jnp.int32(0), jnp.array(box, jnp.float32), jnp.float32(1e9), jnp.float32(0.0),
                          jnp.zeros(Np, bool), jnp.zeros(Nc, bool))


def _walls(state, sp):
    return list(range(sp.num_static_fixated_polys))


def _shape_extent_radius(state, is_poly, idx):
    if is_poly:
        a, b, c, d = polygon_world_extent(state, idx)
        return 0.5 * max(b - a, d - c)
    return float(state.circle.radius[idx])


def _agent_radius(state, sp):
    r = 0.0
    for i in range(sp.num_polygons):
        if bool(state.polygon.active[i]) and int(state.polygon_shape_roles[i]) == ROLE_AGENT:
            r = max(r, _shape_extent_radius(state, True, i))
    for i in range(sp.num_circles):
        if bool(state.circle.active[i]) and int(state.circle_shape_roles[i]) == ROLE_AGENT:
            r = max(r, float(state.circle.radius[i]))
    return r


def _candidates(state, sp, role, static_only=None, exclude_walls=True):
    out = []
    for i in range(sp.num_polygons):
        if exclude_walls and i < sp.num_static_fixated_polys:
            continue
        if bool(state.polygon.active[i]) and int(state.polygon_shape_roles[i]) == role:
            if static_only is None or (float(state.polygon.inverse_mass[i]) == 0.0) == static_only:
                out.append((True, i))
    for i in range(sp.num_circles):
        if bool(state.circle.active[i]) and int(state.circle_shape_roles[i]) == role:
            if static_only is None or (float(state.circle.inverse_mass[i]) == 0.0) == static_only:
                out.append((False, i))
    return out


def _meta(cls, is_poly, idx, box, state, speed_thr=1e9, contact_r=0.0):
    gp = (state.polygon_shape_roles == ROLE_GOAL) & state.polygon.active
    gc = (state.circle_shape_roles == ROLE_GOAL) & state.circle.active
    return DefectMeta(jnp.int32(CLS[cls]), jnp.bool_(is_poly), jnp.int32(idx), jnp.array(box, jnp.float32),
                      jnp.float32(speed_thr), jnp.float32(contact_r), gp, gc)


# ---------------- injectors ----------------
def inject_oob(state, sp, ep, rng, wall=None):
    """Remove one boundary wall (never the floor, which makes the bug trivial). Witness: agent centroid leaves the box."""
    box = world_box(state, sp)
    cands = [i for i in _walls(state, sp) if bool(state.polygon.active[i]) and classify_wall(state, i) in ("left", "right", "ceiling")]
    if not cands:
        return None
    i = cands[int(rng.integers(len(cands)))] if wall is None else wall
    new = state.replace(polygon=state.polygon.replace(active=state.polygon.active.at[i].set(False)))
    return new, _meta("OOB", True, i, box, state)


def inject_tunnel(state, sp, ep, rng):
    """Make a static, neutral obstacle non-solid (collision_mode 0 ignores mode-1 shapes). Witness: agent centroid inside it."""
    box = world_box(state, sp)
    from .kinetix_utils import any_dynamic_inside, dynamic_masks, can_collide
    pm, cm = dynamic_masks(state)
    def _has_collidable_dynamic_body(ip, i):
        # The witness counts only bodies the collision matrix allows to hit the target (jointed chains are excluded), so a
        # target that no dynamic body may collide with is unwitnessable by construction (5 of 337 in the first size-l pool).
        cp = jnp.array([bool(can_collide(state, True, j, ip, i)) for j in range(state.polygon.active.shape[0])])
        cc = jnp.array([bool(can_collide(state, False, j, ip, i)) for j in range(state.circle.active.shape[0])])
        return bool((cp & pm).any() | (cc & cm).any())
    cands = [(ip, i) for (ip, i) in _candidates(state, sp, ROLE_NONE, static_only=True)
             if int((state.polygon.collision_mode if ip else state.circle.collision_mode)[i]) != 0
             and not bool(any_dynamic_inside(state, ip, i)) and _has_collidable_dynamic_body(ip, i)]
    if not cands:
        return None
    is_poly, i = cands[int(rng.integers(len(cands)))]
    if is_poly:
        new = state.replace(polygon=state.polygon.replace(collision_mode=state.polygon.collision_mode.at[i].set(0)))
    else:
        new = state.replace(circle=state.circle.replace(collision_mode=state.circle.collision_mode.at[i].set(0)))
    return new, _meta("TUNNEL", is_poly, i, box, state)


def inject_shortcut(state, sp, ep, rng, dist_rank=None):
    """Give a neutral shape the goal role: touching it wins without reaching the intended goal.
    Witness: reward > 0 while the agent is within contact range of the injected shape.
    dist_rank (0.0=nearest .. 1.0=farthest) grades NAVIGATION difficulty: pick the injected shape at that
    percentile of distance from the agent start, instead of at random (farther = needs coordinated exploration)."""
    box = world_box(state, sp)
    from .kinetix_utils import min_gap_green_to, role_centroid, shape_position
    cands = []
    for (ip, i) in _candidates(state, sp, ROLE_NONE, static_only=None):
        mp = jnp.zeros(sp.num_polygons, bool).at[i].set(True) if ip else jnp.zeros(sp.num_polygons, bool)
        mc = jnp.zeros(sp.num_circles, bool).at[i].set(True) if not ip else jnp.zeros(sp.num_circles, bool)
        if float(min_gap_green_to(state, mp, mc)) > 0.3:   # not already touching / adjacent to a green shape at reset
            cands.append((ip, i))
    if not cands:
        return None
    if dist_rank is None:
        is_poly, i = cands[int(rng.integers(len(cands)))]
    else:
        ap = role_centroid(state, ROLE_AGENT)
        dists = [float(jnp.linalg.norm(ap - shape_position(state, ip, ci))) for (ip, ci) in cands]
        order = sorted(range(len(cands)), key=lambda j: dists[j])
        is_poly, i = cands[order[min(len(order) - 1, int(round(float(dist_rank) * (len(order) - 1))))]]
    r = _shape_extent_radius(state, is_poly, i) + _agent_radius(state, sp) + 0.15
    if is_poly:
        new = state.replace(polygon_shape_roles=state.polygon_shape_roles.at[i].set(ROLE_GOAL))
    else:
        new = state.replace(circle_shape_roles=state.circle_shape_roles.at[i].set(ROLE_GOAL))
    return new, _meta("SHORTCUT", is_poly, i, box, state, contact_r=r)


def inject_anomaly(state, sp, ep, rng, speed_thr=None, restitution=1.6):
    """Runaway physics: agent shapes and all static shapes get restitution so that bounces gain energy.
    Witness: agent speed exceeds speed_thr (calibrated on the clean level)."""
    box = world_box(state, sp)
    pm = (state.polygon_shape_roles == ROLE_AGENT) & state.polygon.active
    cm = (state.circle_shape_roles == ROLE_AGENT) & state.circle.active
    static_p = (state.polygon.inverse_mass == 0.0) & state.polygon.active
    static_c = (state.circle.inverse_mass == 0.0) & state.circle.active
    if not (bool(pm.any()) or bool(cm.any())):
        return None
    new_rp = jnp.where(pm | static_p, restitution, state.polygon.restitution)
    new_rc = jnp.where(cm | static_c, restitution, state.circle.restitution)
    new = state.replace(polygon=state.polygon.replace(restitution=new_rp), circle=state.circle.replace(restitution=new_rc))
    thr = 12.0 if speed_thr is None else float(speed_thr)
    return new, _meta("ANOMALY", True, 0, box, state, speed_thr=thr)


def inject_disarmed(state, sp, ep, rng):
    """A hazard (red) shape becomes harmless. Witness: agent contacts it and the episode does not end."""
    box = world_box(state, sp)
    cands = _candidates(state, sp, ROLE_HAZARD, static_only=None, exclude_walls=True)
    if not cands:
        return None
    is_poly, i = cands[int(rng.integers(len(cands)))]
    if is_poly:
        new = state.replace(polygon_shape_roles=state.polygon_shape_roles.at[i].set(ROLE_NONE))
    else:
        new = state.replace(circle_shape_roles=state.circle_shape_roles.at[i].set(ROLE_NONE))
    return new, _meta("DISARMED", is_poly, i, box, state)


def inject_goal(state, sp, ep, rng):
    """No modification: the target is the level's own win condition (reward > 0.5). Used for the K-Goal suite."""
    box = world_box(state, sp)
    return state, _meta("GOAL", True, 0, box, state)


INJECTORS = {"OOB": inject_oob, "TUNNEL": inject_tunnel, "SHORTCUT": inject_shortcut, "ANOMALY": inject_anomaly, "DISARMED": inject_disarmed, "GOAL": inject_goal}


def inject(cls, state, sp, ep, rng, **kw):
    return INJECTORS[cls](state, sp, ep, rng, **kw)
