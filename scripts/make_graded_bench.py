"""Build a DIFFICULTY-GRADED synthetic bug-finding benchmark. Each instance = a base Kinetix level with a planted ANOMALY
(runaway-physics) bug whose witness threshold `speed_thr` is the graded difficulty knob: higher speed = harder to trigger
(a tester must drive a body faster to expose the bug). Optional 2nd axis: restitution (lower = less energy = harder).
Build and score at ONE size (l). Then filter to physically-reachable cells with the oracle-guided probe (scripts/eval_guided.py).

  python scripts/make_graded_bench.py   # -> runs/graded/pool.json + runs/graded/split.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodev.kinetix_utils import load_level, save_level_json, level_hash
from autodev.defects import inject
from autodev.evaluate import meta_to_json

SIZE = "m"   # smaller/faster on CPU; base levels are m-tier
# motion-capable base levels (thrust / car / ball) — bodies can accelerate, so a speed anomaly is gradeable
LEVELS = [
    "m/h10_thrust_right_easy.json", "m/h11_thrust_left_easy.json", "m/h12_thrustfall_left.json",
    "m/h13_thrustfall_right.json", "m/h14_thrustblock.json", "m/h15_thrustshoot.json",
    "m/h16_thrustcontrol_right.json", "m/h17_thrustcontrol_left.json", "m/h1_car_left.json",
    "m/h2_car_right.json", "m/h3_car_thrust.json", "m/h4_thrust_the_needle.json",
    "m/h5_angry_birds.json", "m/h0_unicycle.json", "s/h0_weak_thrust.json", "s/h4_thrust_aim.json",
]
SPEED_THR = 14.0                                   # fixed challenging threshold
RESTITUTION = [1.6, 1.4, 1.2, 1.1, 1.05, 1.0]      # PRIMARY difficulty axis: high=physics helps (easy) -> 1.0=elastic, no free energy (hard, must actively drive)
SEED = 0

os.makedirs("runs/graded/instances", exist_ok=True)
pool = []
skipped = 0
for lv in LEVELS:
    try:
        st, sp, ep = load_level(lv, SIZE)
    except Exception as e:
        print(f"  load fail {lv}: {e}"); skipped += 1; continue
    for rest in RESTITUTION:
        out = inject("ANOMALY", st, sp, ep, np.random.default_rng(SEED), speed_thr=SPEED_THR, restitution=rest)
        if out is None:
            skipped += 1; continue
        mst, meta = out
        base = os.path.splitext(os.path.basename(lv))[0]
        name = f"{base}__ANOMALY__rest{rest:g}__{SEED}"
        path = f"runs/graded/instances/{name}.json"
        save_level_json(path, mst, sp, ep)
        pool.append(dict(name=name, level=lv, cls="ANOMALY", seed=SEED, path=path,
                         meta=meta_to_json(meta), base_hash=level_hash(st), hash=level_hash(mst),
                         difficulty=dict(speed_thr=SPEED_THR, restitution=rest), probes={}))

json.dump(dict(cfg=dict(size=SIZE, speed_thr=SPEED_THR, restitution=RESTITUTION, levels=LEVELS, note="graded ANOMALY bench (restitution axis)"),
               instances=pool), open("runs/graded/pool.json", "w"), indent=1)
names = [p["name"] for p in pool]
json.dump(dict(pool="runs/graded/pool.json", budget=128 * 2048, size=SIZE,
               search=names, heldout_levels_instances=[], heldout_class_instances=[], excluded=[]),
          open("runs/graded/split.json", "w"), indent=1)
print(f"built {len(pool)} instances ({len(LEVELS)} levels x {len(RESTITUTION)} difficulties), {skipped} skipped. "
      f"-> runs/graded/pool.json + split.json")
by_r = {r: sum(1 for p in pool if p['difficulty']['restitution'] == r) for r in RESTITUTION}
print("per-difficulty (restitution) counts:", by_r)
