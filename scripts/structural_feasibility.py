"""Structural feasibility of planted defects: can the witness predicate fire at all, given the level's collision matrix
and collision modes? (Opus round-4 diagnostic: on some TUNNEL instances every dynamic body is joint-excluded from the
de-solidified target, so `any_dynamic_inside` can never be true.) Checks, per class:
  TUNNEL   - some active dynamic body may collide with the target: the witness `any_dynamic_inside` counts only bodies the
             collision matrix allows (jointed chains are excluded), so without such a body it can never be true
  SHORTCUT / DISARMED - contact-based witnesses. jax2d generates contact manifolds geometrically (collision.py: active =
             is_colliding) and applies the collision matrix only when resolving impulses (engine.py resolve_collision), so
             Kinetix's own goal reward and the harness's contact flags fire for jointed pairs that overlap; the matrix is
             NOT a structural obstacle for these classes (verified: 6 matrix-excluded SHORTCUT instances were reached).
             Structural requirement: an active green shape exists.
  OOB/ANOMALY - some active dynamic body exists
Writes results/structural.json (per-instance flags + per-class counts) and compares with the empirical reachable set.

    python scripts/structural_feasibility.py --pool runs/hyak/pool.json --size l --reachable runs/hyak/split_reachable.json
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np
from autodev.evaluate import meta_from_json, load_instance_level
from autodev.kinetix_utils import dynamic_masks, role_masks, can_collide, ROLE_AGENT, make_env, dyn_max_speed, dyn_centroid
def dyn_centroid_y(s): return dyn_centroid(s)[1]
from autodev.oracles import witness
from jax2d.engine import PhysicsEngine
_ENV = {}
_ENV2 = {}
def envelope_probe(st, sp, ep, meta, steps):
    """Hold each of the n_act actions from reset for `steps` steps (frozen at the first done). Returns (any_action_witnesses,
    max_climb_of_moving_bodies, max_speed) — the reachable set of ONE constant input, the boundary the miss diagnosis found."""
    if "roll" not in _ENV2:
        env = _ENV["env"] if "env" in _ENV else make_env(sp, ep, auto_reset=False); engine = _ENV.get("engine", PhysicsEngine(sp))
        n_act = int(env.action_space(ep).n)
        def roll(level, m):
            def one(a):
                obs, s0 = env.reset(jax.random.PRNGKey(0), ep, level)
                y0 = dyn_centroid_y(s0)
                def body(c, _):
                    s, alive, w, climb, vmax = c
                    obs, s2, r, d, i = env.step(jax.random.PRNGKey(0), s, jnp.int32(a), ep, level)
                    s2 = jax.tree.map(lambda x, y: jnp.where(alive, x, y), s2, s)          # freeze after done
                    w2 = w | (alive & witness(s2, r, d, m, engine)); alive2 = alive & ~d
                    return (s2, alive2, w2, jnp.maximum(climb, jnp.where(alive, dyn_centroid_y(s2) - y0, climb)), jnp.maximum(vmax, jnp.where(alive, dyn_max_speed(s2), vmax))), None
                (s, alive, w, climb, vmax), _ = jax.lax.scan(body, (s0, jnp.bool_(True), jnp.bool_(False), jnp.float32(0.0), jnp.float32(0.0)), None, length=steps)
                return w, climb, vmax
            ws, cl, vm = jax.vmap(one)(jnp.arange(n_act))
            return ws.any(), cl.max(), vm.max()
        _ENV2["roll"] = jax.jit(roll)
    w, cl, vm = _ENV2["roll"](st, meta); return bool(w), float(cl), float(vm)
def noop_probe(st, sp, ep, meta, steps):
    """Steps the level with the no-op action; returns the first step (0 = at reset) at which the witness fires, else -1."""
    if "env" not in _ENV:
        _ENV["env"] = make_env(sp, ep, auto_reset=False); _ENV["engine"] = PhysicsEngine(sp)
        _ENV["noop"] = int(sp.num_motor_bindings) * 2
        env = _ENV["env"]
        def roll(level, m):
            obs, s0 = env.reset(jax.random.PRNGKey(0), ep, level)
            w0 = witness(s0, jnp.float32(0.0), jnp.bool_(False), m, _ENV["engine"])
            def body(c, _):
                s, _r = c; obs, s2, r, d, i = env.step(jax.random.PRNGKey(0), s, jnp.int32(_ENV["noop"]), ep, level)
                return (s2, r), witness(s2, r, d, m, _ENV["engine"])
            _, ws = jax.lax.scan(body, (s0, jnp.float32(0.0)), None, length=steps)
            return jnp.concatenate([w0[None], ws])
        _ENV["roll"] = jax.jit(roll)
    ws = np.array(_ENV["roll"](st, meta)); hit = np.where(ws)[0]
    return int(hit[0]) if len(hit) else -1

ap = argparse.ArgumentParser(); ap.add_argument("--pool", default="runs/hyak/pool.json"); ap.add_argument("--size", default="l")
ap.add_argument("--reachable", default="runs/hyak/split_reachable.json"); ap.add_argument("--split", default="runs/hyak/split.json")
ap.add_argument("--out", default="results/structural.json"); ap.add_argument("--noop_steps", type=int, default=8, help="free-hit probe: witness fires within this many no-op steps from reset (gravity alone)")
ap.add_argument("--envelope", type=int, default=0, help="constant-input envelope: hold each action for this many steps (respecting done); records whether a single held input witnesses the defect and the max climb/speed reached")
a = ap.parse_args()
pool = json.load(open(a.pool))["instances"]

def partner_mask(state, is_poly_bodies, tip, tidx):
    n = state.polygon.active.shape[0] if is_poly_bodies else state.circle.active.shape[0]
    return jax.vmap(lambda i: can_collide(state, is_poly_bodies, i, tip, tidx))(jnp.arange(n))

rows = {}
for p in pool:
    path = p["path"] if os.path.exists(p["path"]) else f"runs/hyak/instances/{p['name']}.json"
    st, sp, ep = load_instance_level(path, a.size); m = meta_from_json(p["meta"]); cls = p["cls"]
    pm, cm = dynamic_masks(st); gp, gc = role_masks(st, ROLE_AGENT)
    tip, tidx = bool(m.target_is_poly), int(m.target_idx)
    cp, cc = partner_mask(st, True, tip, tidx), partner_mask(st, False, tip, tidx)
    if tip: cp = cp.at[tidx].set(False)
    else: cc = cc.at[tidx].set(False)
    dyn_partner = bool((cp & pm).any() | (cc & cm).any()); green_partner = bool((cp & gp).any() | (cc & gc).any())
    any_dyn = bool(pm.any() | cm.any())
    any_green = bool(gp.any() | gc.any())
    ok = {"TUNNEL": dyn_partner, "DISARMED": any_green, "SHORTCUT": any_green, "OOB": any_dyn, "ANOMALY": any_dyn, "GOAL": True}[cls]
    free = noop_probe(st, sp, ep, m, a.noop_steps) if a.noop_steps > 0 else -1
    env_w, env_climb, env_vmax = envelope_probe(st, sp, ep, m, a.envelope) if a.envelope > 0 else (None, None, None)
    rows[p["name"]] = dict(cls=cls, level=p["level"], feasible=ok, free_hit_step=free, constant_input_witness=env_w, envelope_climb=env_climb, envelope_vmax=env_vmax, dyn_partner=dyn_partner, green_partner=green_partner,
                           n_dyn_partners=int((cp & pm).sum() + (cc & cm).sum()), n_green_partners=int((cp & gp).sum() + (cc & gc).sum()))
by_cls = {}
for n, r in rows.items(): by_cls.setdefault(r["cls"], []).append(r["feasible"])
summary = {c: dict(n=len(v), structurally_infeasible=int(len(v) - sum(v)), free_hits=sum(1 for n, r in rows.items() if r["cls"] == c and r["free_hit_step"] >= 0)) for c, v in by_cls.items()}
print("per class:", summary)
print("free hits (witness fires within", a.noop_steps, "no-op steps):", sorted((n, r["free_hit_step"]) for n, r in rows.items() if r["free_hit_step"] >= 0))
if a.envelope > 0: print("constant-input witnesses:", sum(1 for r in rows.values() if r["constant_input_witness"]), "of", len(rows))
out = dict(size=a.size, pool=a.pool, per_class=summary, instances=rows)
if os.path.exists(a.split) and os.path.exists(a.reachable):
    split = json.load(open(a.split)); reach = json.load(open(a.reachable))
    for which in ("search", "heldout_levels_instances", "heldout_class_instances"):
        names = split.get(which, []); infeas = [n for n in names if not rows[n]["feasible"]]; free = [n for n in names if rows[n]["free_hit_step"] >= 0]
        rn = set(reach.get(which, [])) if isinstance(reach.get(which), list) else None
        line = f"{which:26s} n={len(names)} structurally infeasible={len(infeas)} free-hits={len(free)}"
        if rn is not None:
            unreached = [n for n in names if n not in rn]; both = [n for n in infeas if n not in rn]
            line += f"  empirically unreached={len(unreached)}  (structural ⊂ unreached: {len(both)}/{len(infeas)})"
            out.setdefault("splits", {})[which] = dict(n=len(names), structurally_infeasible=infeas, free_hits=free, empirically_unreached=unreached, overlap=both)
        else:
            out.setdefault("splits", {})[which] = dict(n=len(names), structurally_infeasible=infeas, free_hits=free)
        print(line)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)
