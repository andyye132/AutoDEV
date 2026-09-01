"""Post-hoc severity classification of replay-verified scan hits (no rerun needed): replays each chain, measures depth/duration.
Severity: A = OOB (leaves arena) / JOINT_TEAR;  B = PENETRATE with max centroid depth >= 0.05 m or >= 10 flagged steps, or BLOWUP
lasting >= 10 steps;  C = minor transient (everything else). Writes <scan>.classified.json and prints a table."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from autodev.kinetix_utils import load_level, make_env, dynamic_masks, point_in_polygon, polygon_world_extent
from autodev.evaluate import load_instance_level
from autodev.anomalies import anomaly_flags, AnomalyThr, ANOMALIES, joint_separation
from autodev.tester import replay_states
ap = argparse.ArgumentParser(); ap.add_argument("--scan", required=True); ap.add_argument("--size", default="l"); ap.add_argument("--index", action="store_true"); ap.add_argument("--window", type=int, default=40)
a = ap.parse_args()
scan = json.load(open(a.scan)); out_rows = []
cache = {}
def get_env(lv):
    if lv not in cache:
        st, sp, ep = (load_instance_level(lv, a.size) if a.index else load_level(lv, a.size)); cache[lv] = (st, sp, ep, make_env(sp, ep, auto_reset=False))
    return cache[lv]
def pen_depth(s, sp):
    """max centroid depth of any dynamic shape inside a static solid it can collide with (m)."""
    pm, cm = dynamic_masks(s); npoly = sp.num_polygons; best = 0.0
    static_p = np.array((s.polygon.inverse_mass == 0) & s.polygon.active & (s.polygon.collision_mode != 0)); static_c = np.array((s.circle.inverse_mass == 0) & s.circle.active & (s.circle.collision_mode != 0))
    P = [(i, s.polygon.position[i]) for i in range(npoly) if bool(pm[i])] + [(npoly + i, s.circle.position[i]) for i in range(sp.num_circles) if bool(cm[i])]
    for gi, pos in P:
        for j in range(npoly):
            if static_p[j] and int(s.collision_matrix[gi, j]) > 0 and bool(point_in_polygon(pos, s, j)):
                a_, b_, c_, d_ = polygon_world_extent(s, j); best = max(best, min(pos[0] - a_, b_ - pos[0], pos[1] - c_, d_ - pos[1]))
        for j in range(sp.num_circles):
            if static_c[j] and int(s.collision_matrix[gi, npoly + j]) > 0:
                dd = float(s.circle.radius[j]) - float(jnp.linalg.norm(pos - s.circle.position[j]))
                if dd > 0: best = max(best, dd)
    return float(best)
for r in scan["levels"]:
    for p, hs in r["programs"].items():
        for h in hs:
            if not (h["found"] and h.get("replay_verified")): continue
            st, sp, ep, env = get_env(r["level"])
            thr = AnomalyThr(jnp.array(r["box"], jnp.float32), jnp.float32(r["thr"]["joint_tol"]), jnp.float32(r["thr"]["speed_thr"]), jnp.float32(r["thr"]["angvel_thr"]))
            rng = jax.random.PRNGKey(0); obs, s0 = env.reset(rng, ep, st)
            traj, dones = replay_states(env, ep, st, h["actions"], window=a.window)      # jitted; identical to op-by-op stepping
            states = [s0] + traj[:len(h["actions"])]
            fl = np.array([np.array(anomaly_flags(x, sp, thr)) for x in states]); fi = ANOMALIES.index(h["flag"]); t0 = int(np.argmax(fl[:, fi]))
            # continue with no-op after the chain to measure persistence (stop at the first terminal step, as before)
            cont = []
            for k in range(a.window):
                cont.append(traj[len(h["actions"]) + k])
                if bool(dones[len(h["actions"]) + k]): break
            seq = states[t0:] + cont
            flagged = [bool(np.array(anomaly_flags(x, sp, thr))[fi]) for x in seq]
            dur = 0
            for f in flagged:
                if f: dur += 1
                else: break
            depth = max((pen_depth(x, sp) for x in seq[:max(dur, 1)]), default=0.0) if h["flag"] == "PENETRATE" else None
            oob_later = any(np.array(anomaly_flags(x, sp, thr))[0] for x in seq)
            # A: sustained escape (centroid outside the arena for >=10 steps) or joint tear; B: momentary wall-face crossing (the
            # solver ejects the body within <10 steps), deep or long penetration, long runaway; C: transient; X: not reproduced.
            if dur == 0: sev = "X"
            elif h["flag"] == "JOINT_TEAR" or (h["flag"] == "OOB" and dur >= 10): sev = "A"
            elif h["flag"] == "OOB" or (h["flag"] == "PENETRATE" and (depth >= 0.05 or dur >= 10)) or (h["flag"] == "BLOWUP" and dur >= 10): sev = "B"
            else: sev = "C"
            row = dict(level=r["level"], program=p, seed=h["seed"], flag=h["flag"], first_hit_steps=int(h["first_hit"][h["flag"]]), t=t0, duration=dur, depth=depth, oob_later=bool(oob_later), severity=sev, chain_len=len(h["actions"]), spawn=bool(t0 <= 2))
            out_rows.append(row); print(f"{r['level']:34s} {p:14s} s{h['seed']} {h['flag']:9s} t={t0:4d} dur={dur:3d} depth={depth if depth is None else round(depth,3)} oob_later={bool(oob_later)} -> {sev}", flush=True)
json.dump(out_rows, open(a.scan.replace(".json", ".classified.json"), "w"), indent=1)
from collections import Counter
print("severity counts:", dict(Counter(x["severity"] for x in out_rows)), "| distinct (level, flag) with A/B:", len({(x["level"], x["flag"]) for x in out_rows if x["severity"] in "AB"}))
