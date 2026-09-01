"""Audit of unverified hits on the CPU: re-run one hooks program on one bench split with AUTODEV_UNVERIFIED_LOG set, then replay every
logged chain through the harness at several lane counts (the run's own B, 128, 1). A chain that replays at B=1 but not at the run's B
(or vice versa) is a lane-count dependence (XLA fusion drift); a chain that replays nowhere is a lineage defect or a genuine
non-determinism. Prints one line per chain.

    JAX_PLATFORMS=cpu python scripts/cpu_unverified_audit.py --program evolve/methods/deathaware.py --which heldout_class_instances
"""
import os, sys, json, time, argparse, importlib.util, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--program", required=True); ap.add_argument("--split", default="runs/hyak/split_bench.json")
ap.add_argument("--which", default="heldout_class_instances"); ap.add_argument("--seeds", default="100,101"); ap.add_argument("--workers", type=int, default=64)
ap.add_argument("--steps", type=int, default=1024); ap.add_argument("--slots", type=int, default=256); ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--log", default="results/unverified_cpu.jsonl"); a = ap.parse_args()
if os.path.exists(a.log): os.remove(a.log)
os.environ["AUTODEV_UNVERIFIED_LOG"] = a.log
from autodev.search_eval import evaluate_hooks
from autodev.substrate import KinetixSubstrate
from autodev.tester import verify_chain_harness
spec = importlib.util.spec_from_file_location("prog", a.program); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t0 = time.time()
r = evaluate_hooks(m.init, m.restore, m.act, m.archive, split_path=a.split, which=a.which, size="l", seeds=tuple(int(x) for x in a.seeds.split(",")),
                   workers=a.workers, steps=a.steps, slots=a.slots, chunk=a.chunk, log=lambda s: print(s, flush=True))
print(f"RUN {os.path.basename(a.program)} {a.which}: score {r['score']:.3f} detected {r['detected']:.2f} hits {r['n_hits']} unverified {r['n_unverified']} ({time.time()-t0:.0f}s)", flush=True)
recs = [json.loads(l) for l in open(a.log)] if os.path.exists(a.log) else []
sub = KinetixSubstrate(size="l"); levels, metas, info = sub.tasks(a.split, a.which); idx = {x["name"]: i for i, x in enumerate(info)}
for rec in recs:
    i = idx[rec["name"]]; chain = rec["chain"]
    res = {}
    for B in sorted({a.workers, 128, 1}):
        try: res[B] = bool(verify_chain_harness(sub.env, sub.ep, sub.sp, sub.feat_fn, levels[i], chain, sub.oracle, metas[i], B=B))
        except Exception as e: res[B] = f"ERR {type(e).__name__}"
    print(f"CHAIN {rec['name']} seed {rec['seed']} t={rec['t']} worker={rec['worker']} origin={rec['origin']} len={len(chain)} buf_len={rec.get('buf_len')} replay: " + ", ".join(f"B={B}:{v}" for B, v in res.items()), flush=True)
print("AUDIT_DONE", len(recs), "unverified chains", flush=True)
