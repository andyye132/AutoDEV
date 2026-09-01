"""Replay a flagged chain from runs/scan_clean*.json and render frames around the hit (PNG strip + per-frame diagnostics)."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from autodev.kinetix_utils import load_level, make_env, dynamic_masks, point_in_polygon
from kinetix.render import make_render_pixels
from autodev.anomalies import anomaly_flags, joint_separation, AnomalyThr, ANOMALIES
ap = argparse.ArgumentParser(); ap.add_argument("--scan", default="runs/scan_clean.json"); ap.add_argument("--level", required=True); ap.add_argument("--program", default=None)
ap.add_argument("--size", default="m"); ap.add_argument("--out", default="runs/bugs"); ap.add_argument("--window", type=int, default=6); ap.add_argument("--index", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
scan = json.load(open(a.scan)); rec = next(r for r in scan["levels"] if r["level"] == a.level)
prog = a.program or next(p for p, hs in rec["programs"].items() if hs[0]["found"])
hit = next(h for h in rec["programs"][prog] if h["found"])
from autodev.evaluate import load_instance_level
st, sp, ep = (load_instance_level(a.level, a.size) if (a.index or a.level.startswith('runs/')) else load_level(a.level, a.size)); env = make_env(sp, ep, auto_reset=False)
thr = AnomalyThr(jnp.array(rec["box"], jnp.float32), jnp.float32(rec["thr"]["joint_tol"]), jnp.float32(rec["thr"]["speed_thr"]), jnp.float32(rec["thr"]["angvel_thr"]))
render = make_render_pixels(ep, sp)
rng = jax.random.PRNGKey(0); obs, s = env.reset(rng, ep, st)
states, flags = [s], [np.array(anomaly_flags(s, sp, thr))]
for act in hit["actions"]:
    obs, s, r, d, info = env.step(rng, s, jnp.int32(act), ep, st)
    states.append(s); flags.append(np.array(anomaly_flags(s, sp, thr)))
    if bool(d): obs, s = env.reset(rng, ep, st)
flags = np.array(flags); t_hit = int(np.argmax(flags.any(1)))
print(f"{a.level} / {prog}: chain {len(hit['actions'])} actions; oracle {hit['flag']} first fires at replay step {t_hit} (recorded t={hit['t']})")
# diagnostics at the hit
s = states[t_hit]; pm, cm = dynamic_masks(s)
static_p = np.array((s.polygon.inverse_mass == 0) & s.polygon.active & (s.polygon.collision_mode != 0))
for i in range(sp.num_polygons):
    if bool(pm[i]):
        inside = [j for j in range(sp.num_polygons) if static_p[j] and bool(point_in_polygon(s.polygon.position[i], s, j))]
        if inside: print(f"  polygon {i} (role {int(s.polygon_shape_roles[i])}) centroid {np.round(np.array(s.polygon.position[i]),3)} vel {np.round(np.array(s.polygon.velocity[i]),2)} inside static polygon(s) {inside}")
for i in range(sp.num_circles):
    if bool(cm[i]):
        inside = [j for j in range(sp.num_polygons) if static_p[j] and bool(point_in_polygon(s.circle.position[i], s, j))]
        if inside: print(f"  circle {i} (role {int(s.circle_shape_roles[i])}) centroid {np.round(np.array(s.circle.position[i]),3)} r {float(s.circle.radius[i]):.2f} vel {np.round(np.array(s.circle.velocity[i]),2)} inside static polygon(s) {inside}")
# continue with no-op actions after the hit to see whether the anomaly resolves, persists, or escalates
noop = sp.num_motor_bindings * 2
s2 = states[t_hit]; trail = []
for k in range(40):
    obs, s2, r, d, info = env.step(rng, s2, jnp.int32(noop), ep, st)
    f = np.array(anomaly_flags(s2, sp, thr)); pm2, cm2 = dynamic_masks(s2)
    spd = float(max((np.linalg.norm(np.array(s2.polygon.velocity), axis=-1) * np.array(pm2)).max(), (np.linalg.norm(np.array(s2.circle.velocity), axis=-1) * np.array(cm2)).max()))
    trail.append((k + 1, dict(zip(ANOMALIES, f.tolist())), round(spd, 2), bool(d)))
    if bool(d): break
persist = sum(1 for _, f, _, _ in trail if f[hit["flag"]])
print(f"  after the hit (no-op actions): flag '{hit['flag']}' stays on for {persist} of {len(trail)} steps; OOB later: {any(f['OOB'] for _, f, _, _ in trail)}; BLOWUP later: {any(f['BLOWUP'] for _, f, _, _ in trail)}; episode ended: {trail[-1][3]}")
print("  max dyn speed over the trail:", max(t[2] for t in trail), "| thresholds:", rec["thr"])
print("  joint separation at hit:", np.round(np.array(joint_separation(s, sp)), 3).tolist(), "| flags:", dict(zip(ANOMALIES, flags[t_hit].tolist())))
# render a strip of frames around the hit
from PIL import Image
idx = list(range(max(0, t_hit - a.window), min(len(states), t_hit + 3)))
frames = []
for t in idx:
    px = np.array(render(states[t])).astype(np.uint8).transpose(1, 0, 2)[::-1]
    frames.append(px)
strip = np.concatenate(frames, axis=1)
name = f"{a.level.replace('/', '_').replace('.json','')}__{prog}__{hit['flag']}"
Image.fromarray(strip).save(f"{a.out}/{name}_strip.png")
for t, f in zip(idx, frames): Image.fromarray(f).save(f"{a.out}/{name}_t{t:03d}.png")
print(f"  wrote {a.out}/{name}_strip.png (frames {idx[0]}..{idx[-1]}, hit at {t_hit})")
