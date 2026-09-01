"""Tuning-vs-structure ablation: random search over the numeric module-level constants of a tester program (same fitness as the
LLM search). Reports the best tuned variant and the distribution of fitness over perturbations."""
import sys, os, re, json, argparse, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodev.search_eval import evaluate_hooks
ap = argparse.ArgumentParser(); ap.add_argument("--program", required=True); ap.add_argument("--trials", type=int, default=30); ap.add_argument("--scale", type=float, default=0.5)
ap.add_argument("--seeds", default="0,1"); ap.add_argument("--workers", type=int, default=64); ap.add_argument("--steps", type=int, default=1024); ap.add_argument("--size", default="m")
ap.add_argument("--split_path", default=None); ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); random.seed(a.seed)
src = open(a.program).read()
block = src[src.index("# EVOLVE-BLOCK-START"):src.index("# EVOLVE-BLOCK-END")]
consts = [c for c in re.findall(r"^([A-Z_][A-Z0-9_]*)\s*=\s*([0-9]*\.?[0-9]+)\s*(#.*)?$", block, flags=re.M) if c[0] not in ("NM", "NT", "BITS")]
print(f"{len(consts)} numeric constants:", [c[0] for c in consts])
def make_variant(perturb):
    s = src
    for name, val, _ in consts:
        if name not in perturb: continue
        s = re.sub(rf"^({name}\s*=\s*)([0-9]*\.?[0-9]+)", lambda m: m.group(1) + repr(perturb[name]), s, count=1, flags=re.M)
    return s
def fitness(source):
    ns = {}; exec(source, ns)
    r = evaluate_hooks(ns["init"], ns["restore"], ns["act"], ns["archive"], split_path=a.split_path, which="search", size=a.size,
                       seeds=tuple(int(x) for x in a.seeds.split(",")), workers=a.workers, steps=a.steps, log=None)
    return r["score"], r["detected"]
results = []
base = fitness(src); results.append(dict(trial=0, perturb={}, score=base[0], detected=base[1])); print(f"base: {base[0]:.4f} (detected {base[1]:.2f})", flush=True)
best = base[0]
for t in range(1, a.trials + 1):
    perturb = {}
    for name, val, _ in consts:
        v = float(val); is_int = "." not in val
        if random.random() < 0.5:
            nv = v * (1 + random.gauss(0, a.scale)); nv = max(nv, 1) if is_int else max(nv, 1e-3)
            perturb[name] = int(round(nv)) if is_int else round(nv, 4)
    if not perturb: continue
    try:
        sc, det = fitness(make_variant(perturb))
    except Exception as e:
        print(f"trial {t}: error {str(e)[:80]}", flush=True); continue
    results.append(dict(trial=t, perturb=perturb, score=sc, detected=det)); best = max(best, sc)
    print(f"trial {t}: {sc:.4f} (best {best:.4f}) {perturb}", flush=True)
    json.dump(dict(program=a.program, base=base[0], results=results), open(a.out or a.program.replace(".py", ".tuning.json"), "w"), indent=1)
print(f"done: base {base[0]:.4f}, best tuned {best:.4f}, mean over trials {sum(r['score'] for r in results[1:])/max(1,len(results)-1):.4f}")
