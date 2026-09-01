"""Union coverage of a method bench: which targets are hit by which programs.

For each split: best single program, union over all programs, greedy portfolio (which programs add coverage),
targets hit by nobody (a common barrier) and targets hit by exactly one program (specialists). Uses the per-instance
first-hit arrays written by scripts/bench_methods.py (results/bench/*.json). A target counts as hit by a program if it
was witnessed within budget on ANY of the bench seeds (union over seeds; --all-seeds requires every seed).

    python scripts/union_coverage.py --bench results/bench/gpu_hard0.json [--splits search,...] [--top 5]
"""
import json, argparse, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--splits", default="search,heldout_levels_instances,heldout_class_instances")
ap.add_argument("--top", type=int, default=5, help="greedy portfolio size to report")
ap.add_argument("--all-seeds", action="store_true", help="require a hit on every seed (default: any seed)")
ap.add_argument("--exclude", default="", help="comma-separated programs to leave out (e.g. exploit-era ones)")
a = ap.parse_args()
d = json.load(open(a.bench)); P = d["programs"]; excl = set(x for x in a.exclude.split(",") if x)
progs = [p for p in P if p not in excl]

def hit_matrix(split):
    names = None; rows = []
    for p in progs:
        r = P[p].get(split)
        if r is None: continue
        fh = np.asarray(r["first_hit"], dtype=float)         # (n,) or (seeds, n)
        budget = r["budget"]
        h = fh < budget
        if names is None: names = r["names"]
        if h.ndim == 2:                                          # (n, seeds) as written by bench_methods.py, or (seeds, n)
            ax = 1 if h.shape[0] == len(names) else 0
            h = h.all(ax) if a.all_seeds else h.any(ax)
        rows.append((p, h))
    return names, rows

for split in a.splits.split(","):
    names, rows = hit_matrix(split)
    if not rows: continue
    H = np.stack([h for _, h in rows]); ps = [p for p, _ in rows]; n = H.shape[1]
    cover = H.sum(1); best = int(cover.argmax())
    union = H.any(0)
    cls = [nm.split("__")[1] if "__" in nm else nm for nm in names]   # l_<level>__<CLASS>__<k>
    print(f"\n=== {split}: {n} targets, {len(ps)} programs ===")
    print(f"best single: {ps[best]} {cover[best]}/{n} = {cover[best]/n:.0%}   union of all: {union.sum()}/{n} = {union.mean():.0%}")
    # greedy portfolio
    chosen = []; covered = np.zeros(n, bool)
    for _ in range(min(a.top, len(ps))):
        gain = [(H[i] & ~covered).sum() for i in range(len(ps))]
        i = int(np.argmax(gain))
        if gain[i] == 0: break
        chosen.append(ps[i]); covered |= H[i]
        print(f"  + {ps[i]:<24} adds {gain[i]:>2}  -> {covered.sum()}/{n} = {covered.mean():.0%}")
    # never / rarely hit
    cnt = H.sum(0)
    never = [names[j] for j in range(n) if cnt[j] == 0]
    once = [(names[j], ps[int(np.argmax(H[:, j]))]) for j in range(n) if cnt[j] == 1]
    print(f"  hit by nobody: {len(never)}  {dict(collections.Counter(x.split('__')[1] if '__' in x else x for x in never))}")
    for nm in never: print("     ", nm)
    print(f"  hit by exactly one program: {len(once)}")
    for nm, p in once: print(f"      {nm:<40} only {p}")
    # per-class union vs best single
    for c in sorted(set(cls)):
        idx = [j for j in range(n) if cls[j] == c]
        if not idx: continue
        sub = H[:, idx]; b = int(sub.sum(1).argmax())
        print(f"  class {c:<10} n={len(idx):>2}  best single {ps[b]} {sub[b].mean():.0%}   union {sub.any(0).mean():.0%}")

# ---- ranking restricted to targets reached by at least one program ("reachable-hard" subset) ----
print("\n=== ranking on targets reached by at least one bench program (union over seeds for the mask; per-seed for the scores) ===")
for split in a.splits.split(","):
    names, rows = hit_matrix(split)
    if not rows: continue
    H = np.stack([h for _, h in rows]); union = H.any(0); n = int(union.sum())
    out = []
    for p, _ in rows:
        r = P[p][split]; fh = np.asarray(r["first_hit"], float); B = r["budget"]
        if fh.ndim == 1: fh = fh[:, None]
        if fh.shape[0] != len(names): fh = fh.T
        sub = fh[union]                                   # (n_reachable, seeds)
        det = (sub < B).mean(); score = np.clip(1 - np.minimum(sub, B) / B, 0, 1).mean()
        out.append((score, det, p))
    print(f"  {split}: {n}/{len(names)} targets")
    for score, det, p in sorted(out, reverse=True):
        print(f"    {p:<24} score {score:.3f}  detected {det:.0%}")
