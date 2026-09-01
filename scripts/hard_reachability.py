"""For every target a bench's programs ALL missed: has anything, at any budget, ever witnessed it, and what does the miss
diagnosis say blocks it? Walks every JSON with (names, first_hit, budget) records under runs/hyak and results/bench.

    python scripts/hard_reachability.py --bench results/bench/gpu_hard0.json
"""
import json, os, glob, argparse, collections
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--bench", required=True)
ap.add_argument("--sources", default="runs/hyak/*.json,results/bench/*.json"); a = ap.parse_args()

def walk(o, path, out):                       # collect every (names, first_hit, budget) record with its dotted path
    if isinstance(o, dict):
        if "names" in o and "first_hit" in o and "budget" in o: out.append((path, o))
        for k, v in o.items(): walk(v, f"{path}.{k}" if path else k, out)
    elif isinstance(o, list):
        for i, v in enumerate(o[:50]): walk(v, f"{path}[{i}]", out)

ever = collections.defaultdict(list)          # target -> [(source, budget, min first hit)]
for pat in a.sources.split(","):
    for f in sorted(glob.glob(pat)):
        if os.path.abspath(f) == os.path.abspath(a.bench): continue
        try: d = json.load(open(f))
        except Exception: continue
        recs = []; walk(d, "", recs)
        for path, r in recs:
            fh = np.asarray(r["first_hit"], dtype=float); names = r["names"]; B = r["budget"]
            if fh.ndim == 2: fh = fh.min(1 if fh.shape[0] == len(names) else 0)
            if fh.shape[0] != len(names): continue
            for nm, t in zip(names, fh):
                if t < B: ever[nm].append((os.path.basename(f) + ":" + path.split(".")[0], int(B), int(t)))

bench = json.load(open(a.bench))["programs"]
diag = json.load(open("results/miss_diagnosis.json")) if os.path.exists("results/miss_diagnosis.json") else {}
blk = {}
def find_blockers(o):
    if isinstance(o, dict):
        if "instance" in o and "blocker" in o: blk[o["instance"]] = o["blocker"]
        elif "name" in o and "blocker" in o: blk[o["name"]] = o["blocker"]
        for v in o.values(): find_blockers(v)
    elif isinstance(o, list):
        for v in o: find_blockers(v)
find_blockers(diag)
if not blk:                                     # fall back: print the structure of one missed entry so it can be wired
    try: print("diagnosis missed-entry sample:", str(diag["splits"]["search"]["missed"][:1])[:400])
    except Exception: pass

for split in ("search", "heldout_levels_instances", "heldout_class_instances"):
    rows = [(p, r[split]) for p, r in bench.items() if split in r]
    if not rows: continue
    names = rows[0][1]["names"]; B = rows[0][1]["budget"]
    H = np.stack([(np.asarray(r["first_hit"], float).min(1) < B) if np.asarray(r["first_hit"]).ndim == 2 else (np.asarray(r["first_hit"], float) < B) for _, r in rows])
    never = [names[j] for j in range(len(names)) if not H[:, j].any()]
    cats = collections.Counter()
    print(f"\n=== {split}: {len(never)}/{len(names)} hard targets missed by every bench program ===")
    for nm in never:
        e = sorted(ever.get(nm, []), key=lambda x: x[2])
        tag = "NEVER witnessed anywhere" if not e else f"witnessed: {e[0][0]} @ step {e[0][2]} (budget {e[0][1]}); {len(e)} records"
        cats["never" if not e else ("only >1x budget" if all(x[2] >= 131072 for x in e) else "within 128x1024 elsewhere")] += 1
        print(f"  {nm:<42} {tag}   blocker={blk.get(nm, '?')}")
    print("  summary:", dict(cats))
