"""Paired per-instance comparison of two testers from an evaluation file (per-instance first-hit arrays, seeds as columns):
mean paired gain in score, 95% CI over instances (t-interval; bootstrap optional), and the fraction of instances better / worse
beyond a margin. Exclusions are applied like make_tables.py.

    python scripts/paired_ci.py --evals runs/hyak/eval_fixed_merged.json --a pairladder.py --b repeat_random [--margin 0.02]
"""
import json, argparse, os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--evals", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", default="repeat_random")
ap.add_argument("--splits", default="search,heldout_levels_instances,heldout_class_instances"); ap.add_argument("--margin", type=float, default=0.02)
ap.add_argument("--exclusions", default="results/structural.json"); ap.add_argument("--boot", type=int, default=0); a = ap.parse_args()
E = {}
for f in a.evals.split(","): E.update(json.load(open(f)))
try:
    from autodev.exclusions import load_exclusions
    EXCL = load_exclusions(a.exclusions)[0] if a.exclusions else set()
except Exception: EXCL = set()
def scores(r):
    fh = np.asarray(r["first_hit"], float); B = r["budget"]; names = r["names"]
    if fh.ndim == 1: fh = fh[:, None]
    if fh.shape[0] != len(names): fh = fh.T
    s = 1 - np.minimum(fh, B) / B                        # (n, seeds)
    keep = np.array([n not in EXCL for n in names])
    return {n: s[i].mean() for i, n in enumerate(names) if keep[i]}, {n: (fh[i] < B).mean() for i, n in enumerate(names) if keep[i]}
for sp in a.splits.split(","):
    if sp not in E.get(a.a, {}) or sp not in E.get(a.b, {}): print(sp, "missing"); continue
    sa, da = scores(E[a.a][sp]); sb, db = scores(E[a.b][sp]); names = [n for n in sa if n in sb]
    d = np.array([sa[n] - sb[n] for n in names]); n = len(d)
    se = d.std(ddof=1) / np.sqrt(n); ci = 1.96 * se
    if a.boot:
        rng = np.random.default_rng(0); bs = [rng.choice(d, n).mean() for _ in range(a.boot)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    better = (d > a.margin).mean(); worse = (d < -a.margin).mean()
    dd = np.array([da[n] - db[n] for n in names])
    print(f"{sp:26s} n={n:3d}  {a.a} {np.mean([sa[x] for x in names]):.3f} vs {a.b} {np.mean([sb[x] for x in names]):.3f}  paired gain {d.mean():+.3f} ± {ci:.3f}"
          + (f" (boot [{lo:+.3f},{hi:+.3f}])" if a.boot else "") + f"  better {better:.0%} / worse {worse:.0%} (|Δ|>{a.margin})  detection Δ {dd.mean():+.3f}")
