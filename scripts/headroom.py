"""Headroom decomposition: for each split, classify every instance against a chosen leader using ALL valid first-hit records:
  - found-by-leader (and its mean first-hit) -> speed headroom = budget - t
  - stealable: leader misses it but some other valid program finds it (who, how fast)
  - frontier: nobody finds it
Uses per-instance arrays in the given eval files (any program whose record has names/first_hit/budget).
    python scripts/headroom.py --leader run11_best_gen004_0.7917.py --evals runs/hyak/eval_fixed_merged.json,...
"""
import json, argparse, numpy as np, collections
ap = argparse.ArgumentParser(); ap.add_argument("--leader", required=True)
ap.add_argument("--evals", default="runs/hyak/eval_fixed_merged.json,runs/hyak/eval_refix3.json,runs/hyak/eval_restarts4.json,runs/hyak/eval_adaptive.json,runs/hyak/eval_schedule3.json,runs/hyak/eval_run9_best.json,runs/hyak/eval_run11_best.json,runs/hyak/eval_run10_best.json,runs/hyak/eval_proposals4b.json")
ap.add_argument("--exclusions", default="results/structural.json"); a = ap.parse_args()
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodev.exclusions import load_exclusions
EXCL = load_exclusions(a.exclusions)[0]
E = {}
for f in a.evals.split(","):
    try: E.update(json.load(open(f)))
    except Exception: pass
print("programs pooled:", len(E))
for split in ("search", "heldout_levels_instances", "heldout_class_instances"):
    per = {}   # name -> {prog: mean_fh}
    B = None
    for prog, sp in E.items():
        r = sp.get(split)
        if not isinstance(r, dict) or "first_hit" not in r: continue
        fh = np.asarray(r["first_hit"], float); names = r["names"]; B = r["budget"]
        if fh.ndim == 1: fh = fh[:, None]
        if fh.shape[0] != len(names): fh = fh.T
        for n, row in zip(names, fh):
            if n in EXCL: continue
            per.setdefault(n, {})[prog] = float(row.mean())        # seed-mean first hit (budget if never)
    if not per: continue
    n_all = len(per); lead_found = []; steal = []; frontier = []
    for n, d in per.items():
        lt = d.get(a.leader, B)
        others = {p: t for p, t in d.items() if p != a.leader and t < B * 0.999}
        if lt < B * 0.999: lead_found.append((n, lt, min(others.values()) if others else None))
        elif others: steal.append((n, min(others, key=others.get), min(others.values())))
        else: frontier.append(n)
    print(f"\n=== {split}: {n_all} instances (leader {a.leader}) ===")
    lf = np.array([t for _, t, _ in lead_found])
    print(f"  leader finds {len(lead_found)} ({len(lead_found)/n_all:.0%}); mean first-hit {lf.mean():,.0f} of {B:,} budget -> speed headroom {(lf.mean()/B):.0%} of budget still spent")
    fastest = [x for x in lead_found if x[2] is not None and x[2] < x[1] * 0.5]
    print(f"     (someone is >=2x faster than the leader on {len(fastest)} of them)")
    print(f"  STEALABLE (others find, leader misses): {len(steal)} ({len(steal)/n_all:.0%})")
    cnt = collections.Counter(p for _, p, _ in steal)
    for p, c in cnt.most_common(6): print(f"       found by {p}: {c}")
    for n, p, t in sorted(steal, key=lambda x: x[2])[:10]: print(f"       {n:<44} {p} @ {t:,.0f}")
    print(f"  FRONTIER (nobody in the pool finds): {len(frontier)} ({len(frontier)/n_all:.0%})")
    cls = collections.Counter(x.split("__")[1] for x in frontier if "__" in x)
    print(f"       by class: {dict(cls)}")
