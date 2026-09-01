"""Chimera audit (Aug 29): does the pre-fix harness race create physically inconsistent archived states, and how much of a
restoring program's detection depends on them?  Evaluates a program on a subset of planted instances under the fixed harness
and under the legacy race (AUTODEV_LEGACY_RACE=1), and counts archived slots whose stored feature vector disagrees with the
features recomputed from the stored state (leaves from different workers).

    JAX_PLATFORMS=cpu python scripts/chimera_audit.py --n 8 --workers 64 --steps 1024 --out results/chimera_audit.json
"""
import os, sys, json, time, argparse, importlib.util, subprocess, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="runs/hyak/split.json"); ap.add_argument("--which", default="search"); ap.add_argument("--size", default="l")
ap.add_argument("--program", default="runs/evolve_overnight/candidates/cand1_gen037_0.6994.py"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=64); ap.add_argument("--steps", type=int, default=1024); ap.add_argument("--slots", type=int, default=256); ap.add_argument("--out", default="results/chimera_audit.json")
ap.add_argument("--mode", default="", help="internal: run one mode (fixed|legacy) in a child process and print RESULT json")
a = ap.parse_args()

def load_hooks(path):
    spec = importlib.util.spec_from_file_location("prog", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.init, m.restore, m.act, m.archive

if a.mode:   # child: one harness mode per JAX process (the race flag is read at trace time)
    import jax, jax.numpy as jnp
    from autodev.search_eval import evaluate_hooks
    from autodev.substrate import KinetixSubstrate
    from autodev.tester import TesterConfig, make_tester
    from autodev.search_eval import HookProgram
    sub = KinetixSubstrate(size=a.size)
    split = json.load(open(a.split)); rng = np.random.default_rng(a.seed)
    names = sorted(rng.choice(split[a.which], size=min(a.n, len(split[a.which])), replace=False).tolist())
    sub_split = dict(split); sub_split[a.which] = names; tmp = f"/tmp/chimera_split_{os.getpid()}.json"; json.dump(sub_split, open(tmp, "w"))
    hooks = load_hooks(a.program)
    r = evaluate_hooks(*hooks, split_path=tmp, which=a.which, size=a.size, seeds=(0,), workers=a.workers, steps=a.steps, slots=a.slots, chunk=8, substrate=sub)
    levels, metas, info = sub.tasks(tmp, a.which)
    cfg = TesterConfig(num_workers=a.workers, steps=a.steps, slots=a.slots, return_archive=True)
    run = jax.jit(make_tester(sub.env, sub.ep, sub.sp, sub.feat_fn, HookProgram(*hooks), cfg, sub.oracle, 1))
    chim = []
    for i in range(min(2, len(levels))):
        out = run(jax.random.PRNGKey(11 + i), levels[i], metas[i]); A = out["archive"]
        valid = np.array(A["valid"][:a.slots]); fe = np.array(A["feats"][:a.slots]); re = np.array(jax.vmap(sub.feat_fn)(jax.tree.map(lambda x: x[:a.slots], A["state"])))
        bad = valid & (np.abs(fe - re).max(1) > 1e-3); chim.append(dict(instance=info[i]["name"], valid=int(valid.sum()), chimeras=int(bad.sum())))
    print("RESULT " + json.dumps(dict(mode=a.mode, score=r["score"], detected=r["detected"], detected_per_class=r["detected_per_class"], names=names, chimera=chim)), flush=True)
    sys.exit(0)

res = {}
for mode in ("fixed", "legacy"):
    env = dict(os.environ); env["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cpu"); env["AUTODEV_LEGACY_RACE"] = "1" if mode == "legacy" else "0"
    t0 = time.time(); p = subprocess.run([sys.executable, __file__] + sys.argv[1:] + ["--mode", mode], env=env, capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line: print(mode, "FAILED", p.stderr[-2000:], flush=True); continue
    res[mode] = json.loads(line[0][7:]); res[mode]["wall"] = time.time() - t0
    print(f"{mode:7s} score {res[mode]['score']:.3f} detected {res[mode]['detected']:.2f} per-class {{{', '.join(f'{k}:{v:.2f}' for k, v in res[mode]['detected_per_class'].items())}}} chimeras {res[mode]['chimera']} ({res[mode]['wall']:.0f}s)", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(dict(program=a.program, n=a.n, workers=a.workers, steps=a.steps, results=res), open(a.out, "w"), indent=1); print("wrote", a.out, flush=True)
