"""Portfolio / union analysis for the 'bugs found' push.  Reads a multi-program eval json (from eval_baselines.py,
which stores per-instance `first_hit` and `budget` per program/split) and computes:
  * per-program detection (fraction of instances witnessed within budget),
  * the UNION detection (instance found by ANY program) -- the ceiling a portfolio could reach,
  * a GREEDY portfolio order (repeatedly add the program that covers the most still-missed instances),
  * the marginal contribution of each program (bugs it finds that NO other program finds).
This answers: does running a portfolio of testers find more bugs than the single best tester?

  python scripts/portfolio_union.py runs/portfolio/kinetix_multi.json [--split search]
"""
import json, sys, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--split", default="search")
a = ap.parse_args()

data = json.load(open(a.path))
# build {program: detected_bool_vector over instances} on the requested split (collapse seeds -> "detected if any seed hit")
det = {}
budget = None
names = None
for prog, splits in data.items():
    s = splits.get(a.split)
    if not s or not s.get("first_hit"):
        continue
    fh = np.array(s["first_hit"], float)
    B = s["budget"]; budget = B
    names = s.get("names", names)
    # first_hit may be [inst] or [inst, seed]; detected on an instance = hit on ANY seed
    if fh.ndim == 2:
        d = (fh < B).any(1)
    else:
        d = (fh < B)
    det[prog] = d.astype(bool)

if not det:
    print(f"no per-instance first_hit found on split '{a.split}' in {a.path}"); sys.exit(1)

progs = list(det)
M = np.stack([det[p] for p in progs])          # (P, N)
N = M.shape[1]
short = {p: p.replace(".py", "").replace("run7_best_gen035_0.5888", "run7") for p in progs}

print(f"{N} instances, {len(progs)} programs, budget={budget}, split={a.split}\n")
print("per-program detection (sorted):")
order = np.argsort(-M.mean(1))
for i in order:
    print(f"  {short[progs[i]]:22s} {M[i].mean():.3f}  ({M[i].sum()}/{N})")

union = M.any(0)
best = M.mean(1).max()
print(f"\nBEST single program : {best:.3f} ({int(best*N)}/{N})")
print(f"UNION (any program) : {union.mean():.3f} ({union.sum()}/{N})   <- portfolio ceiling")
print(f"gain over best single: +{union.mean()-best:.3f} ({int(union.sum()-best*N)} more bugs)")

# marginal-unique contribution: bugs a program finds that NO other program finds
print("\nunique contribution (bugs found ONLY by this program):")
for i in order:
    others = np.delete(M, i, 0).any(0) if len(progs) > 1 else np.zeros(N, bool)
    uniq = M[i] & ~others
    if uniq.sum():
        print(f"  {short[progs[i]]:22s} +{uniq.sum()} unique")

# greedy portfolio: repeatedly add the program covering the most still-missed instances
print("\ngreedy portfolio (max marginal coverage):")
covered = np.zeros(N, bool); chosen = []
while True:
    gains = [(M[i] & ~covered).sum() if progs[i] not in chosen else -1 for i in range(len(progs))]
    j = int(np.argmax(gains))
    if gains[j] <= 0:
        break
    covered |= M[j]; chosen.append(progs[j])
    print(f"  + {short[progs[j]]:22s} -> cumulative {covered.mean():.3f} ({covered.sum()}/{N})  (+{gains[j]})")
print(f"\n{len(chosen)} programs reach the union {covered.mean():.3f}; "
      f"a {min(3,len(chosen))}-program portfolio reaches "
      f"{np.stack([M[progs.index(c)] for c in chosen[:3]]).any(0).mean():.3f}.")
