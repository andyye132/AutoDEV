"""Build a NAVIGATION-difficulty graded bug benchmark. Each instance = a base Kinetix level with a planted SHORTCUT
defect (a neutral shape given the goal role: touching it wins). Difficulty knob = the DISTANCE the agent must travel to
reach the injected shape, graded by `dist_rank` (0.0=nearest candidate .. 1.0=farthest). Far shortcuts need coordinated
spatial exploration -- exactly where structured/evolved testers should beat random and action-repeat, giving a MARGIN
(unlike the speed-axis bench where action-repeat suffices). Reachable-but-hard is the point: the headroom the saturated
search benchmark lacks.

  python scripts/make_graded_nav_bench.py   # -> runs/graded_nav/pool.json + split.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodev.kinetix_utils import load_level, save_level_json, level_hash, role_centroid, shape_position, ROLE_AGENT
from autodev.defects import inject
from autodev.evaluate import meta_to_json
import jax.numpy as jnp

SIZE = "m"
# motion-capable base levels (mobile agent can traverse to a distant shape)
LEVELS = [
    "m/h10_thrust_right_easy.json", "m/h11_thrust_left_easy.json", "m/h12_thrustfall_left.json",
    "m/h13_thrustfall_right.json", "m/h14_thrustblock.json", "m/h15_thrustshoot.json",
    "m/h16_thrustcontrol_right.json", "m/h17_thrustcontrol_left.json", "m/h1_car_left.json",
    "m/h2_car_right.json", "m/h3_car_thrust.json", "m/h4_thrust_the_needle.json",
    "m/h5_angry_birds.json", "m/h0_unicycle.json", "s/h0_weak_thrust.json", "s/h4_thrust_aim.json",
    "m/h6_thrust_over.json", "m/h7_thrust_up.json", "m/h8_thrust_left_hard.json", "m/h9_thrust_right_hard.json",
]
DIST_RANKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # navigation difficulty: nearest -> farthest injected shortcut
SEED = 0

os.makedirs("runs/graded_nav/instances", exist_ok=True)
pool = []
skipped = 0
for lv in LEVELS:
    try:
        st, sp, ep = load_level(lv, SIZE)
    except Exception as e:
        skipped += 1; continue
    ap = np.asarray(role_centroid(st, ROLE_AGENT))
    for rank in DIST_RANKS:
        out = inject("SHORTCUT", st, sp, ep, np.random.default_rng(SEED), dist_rank=rank)
        if out is None:
            skipped += 1; continue
        mst, meta = out
        # distance from agent start to the injected shortcut shape (the true difficulty measure)
        try:
            tp = np.asarray(shape_position(mst, bool(meta.target_is_poly), int(meta.target_idx)))
            dist = float(np.linalg.norm(ap - tp))
        except Exception:
            dist = float("nan")
        base = os.path.splitext(os.path.basename(lv))[0]
        name = f"{base}__SHORTCUT__r{rank:g}__{SEED}"
        path = f"runs/graded_nav/instances/{name}.json"
        save_level_json(path, mst, sp, ep)
        pool.append(dict(name=name, level=lv, cls="SHORTCUT", seed=SEED, path=path,
                         meta=meta_to_json(meta), base_hash=level_hash(st), hash=level_hash(mst),
                         difficulty=dict(dist_rank=rank, distance=dist), probes={}))

json.dump(dict(cfg=dict(size=SIZE, dist_ranks=DIST_RANKS, levels=LEVELS, note="graded SHORTCUT nav bench (distance axis)"),
               instances=pool), open("runs/graded_nav/pool.json", "w"), indent=1)
names = [p["name"] for p in pool]
json.dump(dict(pool="runs/graded_nav/pool.json", budget=128 * 2048, size=SIZE,
               search=names, heldout_levels_instances=[], heldout_class_instances=[], excluded=[]),
          open("runs/graded_nav/split.json", "w"), indent=1)
print(f"built {len(pool)} instances ({len(LEVELS)} levels x {len(DIST_RANKS)} distance ranks), {skipped} skipped.")
by_r = {r: [p["difficulty"]["distance"] for p in pool if p["difficulty"]["dist_rank"] == r] for r in DIST_RANKS}
for r in DIST_RANKS:
    ds = [d for d in by_r[r] if d == d]
    print(f"  dist_rank {r:g}: n={len(ds)} median-distance={np.median(ds):.2f}" if ds else f"  dist_rank {r:g}: n=0")
