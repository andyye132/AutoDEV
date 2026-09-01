"""Create runs/split.json from runs/pool.json: hold out a fraction of LEVELS (by hash order) and one defect CLASS.
Instances that are trivially found (any hand-designed program finds them within `--trivial` steps) or found by no program
within budget are excluded from the search set (kept, marked, in held-out reporting)."""
import sys, os, json, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodev.defects import CLASSES
ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="runs/pool.json"); ap.add_argument("--out", default="runs/split.json")
ap.add_argument("--heldout_class", default="OOB"); ap.add_argument("--heldout_frac", type=float, default=0.3)
ap.add_argument("--trivial", type=float, default=2048); ap.add_argument("--keep_unfound", action="store_true")
args = ap.parse_args()
pool = json.load(open(args.pool)); inst = pool["instances"]; budget = pool["cfg"]["budget"]; programs = pool["cfg"]["programs"].split(",")
levels = sorted({p["level"] for p in inst}, key=lambda l: hashlib.sha256(l.encode()).hexdigest())
n_held = max(1, int(round(args.heldout_frac * len(levels))))
held_levels = set(levels[:n_held]); search_levels = set(levels[n_held:])
def status(p):
    firsts = [p["probes"].get(q) for q in programs]
    if any(f is not None and f <= args.trivial for f in firsts): return "trivial"
    if all(f is None for f in firsts): return "unfound"
    return "ok"
search, heldout_levels, heldout_class, excluded = [], [], [], []
for p in inst:
    s = status(p)
    if args.heldout_class != "none" and p["cls"] == args.heldout_class: heldout_class.append(p["name"])
    elif p["level"] in held_levels: heldout_levels.append(p["name"])
    elif s == "ok" or (s == "unfound" and args.keep_unfound): search.append(p["name"])
    else: excluded.append((p["name"], s))
split = dict(pool=args.pool, budget=budget, heldout_class=args.heldout_class, search_levels=sorted(search_levels), heldout_levels=sorted(held_levels),
             search=search, heldout_levels_instances=heldout_levels, heldout_class_instances=heldout_class, excluded=excluded)
json.dump(split, open(args.out, "w"), indent=1)
print(f"levels: {len(search_levels)} search / {len(held_levels)} held-out; instances: search {len(search)}, held-out-levels {len(heldout_levels)}, held-out-class({args.heldout_class}) {len(heldout_class)}, excluded {len(excluded)} ({sum(1 for _, s in excluded if s=='trivial')} trivial, {sum(1 for _, s in excluded if s=='unfound')} unfound)")
from collections import Counter
print("search per class:", Counter(next(p['cls'] for p in inst if p['name'] == n) for n in search))
