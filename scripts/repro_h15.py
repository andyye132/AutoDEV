"""Reproduce the m/h15_thrustshoot ceiling-tunnelling escape from a recorded action chain (deterministic replay).
Usage: python scripts/repro_h15.py [--chain runs/kinetix_issue/h15_escape_actions.json] [--size l]
Prints, per step, the highest dynamic-shape y and whether any dynamic shape's centroid is outside the arena; the level's
ceiling is the top wall polygon. Also exports the chain from the scan file when --export is given."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from autodev.kinetix_utils import load_level, make_env, dynamic_masks, world_box, any_dynamic_outside
ap = argparse.ArgumentParser(); ap.add_argument("--chain", default="runs/kinetix_issue/h15_escape_actions.json"); ap.add_argument("--size", default="l")
ap.add_argument("--export", default="", help="scan json to export the shortest replay-verified OOB chain from"); a = ap.parse_args()
if a.export:
    S = json.load(open(a.export)); best = None
    for r in S["levels"]:
        if "h15_thrustshoot" not in r["level"]: continue
        for p, hs in r["programs"].items():
            for h in hs:
                if h["found"] and h.get("replay_verified") and h["flag"] == "OOB" and (best is None or len(h["actions"]) < len(best["actions"])):
                    best = dict(level=r["level"], program=p, seed=h["seed"], actions=[int(x) for x in h["actions"]], size=S["cfg"]["size"], reset_key=0)
    json.dump(best, open(a.chain, "w"), indent=0); print("exported", best["program"], "seed", best["seed"], "len", len(best["actions"]))
C = json.load(open(a.chain))
st, sp, ep = load_level(C["level"], a.size); env = make_env(sp, ep, auto_reset=False)
box = np.array(world_box(st, sp)); rng = jax.random.PRNGKey(C.get("reset_key", 0)); obs, s = env.reset(rng, ep, st)
pm, cm = dynamic_masks(st)
print(f"level {C['level']} size {a.size}; arena box x[{box[0]:.2f},{box[1]:.2f}] y[{box[2]:.2f},{box[3]:.2f}]; {len(C['actions'])} actions")
first_out = None
for t, act in enumerate(C["actions"], 1):
    obs, s, rr, d, info = env.step(rng, s, jnp.int32(act), ep, st)
    ys = np.concatenate([np.array(s.polygon.position[:, 1])[np.array(pm)], np.array(s.circle.position[:, 1])[np.array(cm)]])
    speeds = np.concatenate([np.linalg.norm(np.array(s.polygon.velocity)[np.array(pm)], axis=-1), np.linalg.norm(np.array(s.circle.velocity)[np.array(cm)], axis=-1)])
    out = bool(any_dynamic_outside(s, jnp.array(box)))
    if out and first_out is None: first_out = t
    if t % 10 == 0 or out or t == len(C["actions"]):
        print(f"t={t:3d} action={act:2d} max_y={ys.max():.3f} max_speed={speeds.max():5.2f} outside_arena={out}")
print("RESULT:", f"a dynamic shape leaves the arena at step {first_out}" if first_out else "no escape in this replay")
