"""For a flagged PENETRATE hit: replay the chain and report, at the first flagged frame, every (dynamic, static) pair with the
dynamic centroid inside the static shape, the engine's collision-matrix entry, penetration depth proxy, and whether the two shapes
are connected through the joint graph (mechanism-internal overlap) — plus how long the overlap persists."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from autodev.kinetix_utils import load_level, make_env, dynamic_masks, point_in_polygon, polygon_world_extent
from autodev.evaluate import load_instance_level
from autodev.anomalies import anomaly_flags, AnomalyThr
ap = argparse.ArgumentParser(); ap.add_argument("--scan", required=True); ap.add_argument("--level", required=True); ap.add_argument("--program", default=None); ap.add_argument("--size", default="l"); ap.add_argument("--index", action="store_true")
a = ap.parse_args()
scan = json.load(open(a.scan)); rec = next(r for r in scan["levels"] if r["level"] == a.level)
prog = a.program or next(p for p, hs in rec["programs"].items() if any(h["found"] and h["flag"] == "PENETRATE" for h in hs))
hit = next(h for h in rec["programs"][prog] if h["found"] and h["flag"] == "PENETRATE")
st, sp, ep = (load_instance_level(a.level, a.size) if a.index else load_level(a.level, a.size)); env = make_env(sp, ep, auto_reset=False)
thr = AnomalyThr(jnp.array(rec["box"], jnp.float32), jnp.float32(rec["thr"]["joint_tol"]), jnp.float32(rec["thr"]["speed_thr"]), jnp.float32(rec["thr"]["angvel_thr"]))
npoly = sp.num_polygons
# joint graph connectivity (global indices)
J = st.joint; adj = {}
for k in range(J.active.shape[0]):
    if bool(J.active[k]):
        a_, b_ = int(J.a_index[k]), int(J.b_index[k]); adj.setdefault(a_, set()).add(b_); adj.setdefault(b_, set()).add(a_)
def connected(a_, b_):
    seen, stack = set(), [a_]
    while stack:
        x = stack.pop()
        if x == b_: return True
        if x in seen: continue
        seen.add(x); stack.extend(adj.get(x, ()))
    return False
rng = jax.random.PRNGKey(0); obs, s = env.reset(rng, ep, st); states = [s]
for act in hit["actions"]:
    obs, s, r, d, info = env.step(rng, s, jnp.int32(act), ep, st); states.append(s)
    if bool(d): obs, s = env.reset(rng, ep, st)
def pairs(s):
    pm, cm = dynamic_masks(s); out = []
    static_p = np.array((s.polygon.inverse_mass == 0) & s.polygon.active & (s.polygon.collision_mode != 0)); static_c = np.array((s.circle.inverse_mass == 0) & s.circle.active & (s.circle.collision_mode != 0))
    P = [(True, i, s.polygon.position[i]) for i in range(npoly) if bool(pm[i])] + [(False, i, s.circle.position[i]) for i in range(sp.num_circles) if bool(cm[i])]
    for is_p, i, pos in P:
        gi = i if is_p else npoly + i
        for j in range(npoly):
            if static_p[j] and bool(point_in_polygon(pos, s, j)) and int(s.collision_matrix[gi, j]) > 0:
                a_, b_, c_, d_ = polygon_world_extent(s, j); depth = min(pos[0] - a_, b_ - pos[0], pos[1] - c_, d_ - pos[1])
                out.append(("poly" if is_p else "circ", i, "static poly", j, float(depth), connected(gi, j)))
        for j in range(sp.num_circles):
            gj = npoly + j
            if static_c[j] and float(jnp.linalg.norm(pos - s.circle.position[j])) < float(s.circle.radius[j]) and int(s.collision_matrix[gi, gj]) > 0:
                out.append(("poly" if is_p else "circ", i, "static circ", j, float(s.circle.radius[j]) - float(jnp.linalg.norm(pos - s.circle.position[j])), connected(gi, gj)))
    return out
flags = [np.array(anomaly_flags(x, sp, thr))[1] for x in states]; t0 = int(np.argmax(flags))
print(f"{a.level} / {prog}: PENETRATE first at replay step {t0}; persists {int(np.sum(flags[t0:t0+40]))} of next 40 steps")
for t in range(t0, min(len(states), t0 + 6)):
    for kind, i, skind, j, depth, conn in pairs(states[t]):
        v = states[t].polygon.velocity[i] if kind == "poly" else states[t].circle.velocity[i]
        print(f"  t={t}: dynamic {kind} {i} (role {int((states[t].polygon_shape_roles if kind=='poly' else states[t].circle_shape_roles)[i])}, |v|={float(jnp.linalg.norm(v)):.2f}) inside {skind} {j}: depth {depth:.3f} m; joint-connected={conn}")
