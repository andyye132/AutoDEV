"""Evaluate hand-designed tester programs (and optionally an evolved program file) on search / held-out-levels / held-out-class splits."""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from autodev.search_eval import evaluate_hooks
from autodev.tester import BUILTIN_PROGRAMS
assert os.environ.get("AUTODEV_REMOTE_EVAL") != "1", "evaluation must not run through the remote fitness queue (it ignores the split)"   # Opus audit H3
ap = argparse.ArgumentParser()
ap.add_argument("--programs", default="random,repeat_random,goexplore"); ap.add_argument("--program_files", default="")
ap.add_argument("--splits", default="search,heldout_levels_instances,heldout_class_instances"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--workers", type=int, default=64); ap.add_argument("--steps", type=int, default=1024); ap.add_argument("--size", default="m"); ap.add_argument("--out", default="runs/baselines.json")
ap.add_argument("--substrate", default="", help="engine spec for autodev.substrate.make_substrate (default: Kinetix at --size)"); ap.add_argument("--slots", type=int, default=256); ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--restarts", type=int, default=1, help="k sequential restarts of steps/k each at the same total budget; the first restart that witnesses an instance sets first_hit = (i * budget/k) + its own hit")
a = ap.parse_args()
SUB = None
if a.substrate:
    from autodev.substrate import make_substrate
    SUB = make_substrate(a.substrate)


def evaluate_with_restarts(hooks, sp, seeds, k):
    """Same total budget as one run, spent as k independent restarts (different RNG, steps/k each)."""
    from autodev.search_eval import load_split
    from autodev.defects import CLASSES
    sub = a.steps // k; parts = []
    for i in range(k):
        parts.append(evaluate_hooks(*hooks, which=sp, size=a.size, seeds=tuple(1000 * s + 17 * i + 1 for s in seeds), workers=a.workers, steps=sub, slots=a.slots, chunk=a.chunk, substrate=SUB, log=None))
    bsub = parts[0]["budget"]; B = bsub * k
    fh = np.full(np.array(parts[0]["first_hit"]).shape, float(B))
    for i in reversed(range(k)):                       # earlier restarts take precedence (smaller total)
        f = np.array(parts[i]["first_hit"]); fh = np.where(f < bsub, i * bsub + f, fh)
    _, _, info = (SUB.tasks(os.environ.get("AUTODEV_SPLIT", "runs/split.json"), sp) if SUB else load_split(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ.get("AUTODEV_SPLIT", "runs/split.json")), sp, a.size))
    norm = 1.0 - np.minimum(fh, B) / B; n = fh.shape[0]
    per_class = {c: float(np.mean([norm[j].mean() for j in range(n) if info[j]["cls"] == c])) for c in CLASSES if any(x["cls"] == c for x in info)}
    det_class = {c: float(np.mean([(fh[j] < B).mean() for j in range(n) if info[j]["cls"] == c])) for c in per_class}
    return dict(score=float(norm.mean()), detected=float((fh < B).mean()), per_class=per_class, detected_per_class=det_class, first_hit=fh.tolist(),
                names=parts[0]["names"], budget=B, wall=sum(p["wall"] for p in parts), restarts=k)
seeds = tuple(int(s) for s in a.seeds.split(","))
progs = {}
for p in a.programs.split(","):
    if p: progs[p] = BUILTIN_PROGRAMS[p]()
for f in a.program_files.split(","):
    if f:
        ns = {}; exec(open(f).read(), ns)
        progs[os.path.basename(f)] = ns
res = json.load(open(a.out)) if os.path.exists(a.out) else {}   # merge: never clobber other programs' rows
for name, P in progs.items():
    hooks = (P["init"], P["restore"], P["act"], P["archive"]) if isinstance(P, dict) else (P.init, P.restore, P.act, P.archive)
    res[name] = {}
    for sp in a.splits.split(","):
        try:
            r = evaluate_with_restarts(hooks, sp, seeds, a.restarts) if a.restarts > 1 else evaluate_hooks(*hooks, which=sp, size=a.size, seeds=seeds, workers=a.workers, steps=a.steps, slots=a.slots, chunk=a.chunk, substrate=SUB, log=None)
            res[name][sp] = dict(score=r["score"], detected=r["detected"], per_class=r["per_class"], detected_per_class=r["detected_per_class"], wall=r["wall"], n=len(r["names"]), budget=r["budget"], first_hit=r["first_hit"], names=r["names"], restarts=r.get("restarts", 1), coverage=r.get("coverage"))
            print(f"{name:20s} {sp:26s} score {r['score']:.3f} detected {r['detected']:.2f} per-class-detected {{{', '.join(f'{k}:{v:.2f}' for k, v in r['detected_per_class'].items())}}} ({r['wall']:.0f}s)", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        except Exception as e:
            import traceback; res[name][sp] = dict(error=traceback.format_exc()[-1200:]); print(f"{name:20s} {sp:26s} ERROR {type(e).__name__}: {str(e)[:140]}", flush=True); json.dump(res, open(a.out, "w"), indent=1)

