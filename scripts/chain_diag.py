"""Diagnose one unverified hit: re-run the exact evaluation chunk (same key, same vmap structure), take the hit record, reconstruct its
chain, replay it step by step from reset and compare the replayed state with the state stored in the hit record (max |diff| over all
float leaves). Near-zero drift = the chain is right and the witness is a threshold flip under different fusion; large drift = lineage error.

    JAX_PLATFORMS=cpu python scripts/chain_diag.py --program evolve/methods/deathaware.py --which heldout_class_instances --name l_grasp_easy__OOB__1 --seed 100
"""
import os, sys, json, time, argparse, importlib.util, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--program", required=True); ap.add_argument("--split", default="runs/hyak/split_bench.json")
ap.add_argument("--which", default="heldout_class_instances"); ap.add_argument("--name", required=True); ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--workers", type=int, default=64); ap.add_argument("--steps", type=int, default=1024); ap.add_argument("--slots", type=int, default=256); ap.add_argument("--chunk", type=int, default=8)
a = ap.parse_args()
import jax, jax.numpy as jnp
from autodev.search_eval import stack, HookProgram
from autodev.substrate import KinetixSubstrate
from autodev.tester import make_tester, TesterConfig, chain_actions, replay_chain, verify_chain_harness
spec = importlib.util.spec_from_file_location("prog", a.program); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sub = KinetixSubstrate(size="l"); levels, metas, info = sub.tasks(a.split, a.which); names = [x["name"] for x in info]
i = names.index(a.name); c0 = (i // a.chunk) * a.chunk; idx = list(range(c0, min(len(levels), c0 + a.chunk))); j = i - c0
cfg = TesterConfig(num_workers=a.workers, steps=a.steps, slots=a.slots, chain_log=16384, invariants=True)
run = jax.jit(jax.vmap(make_tester(sub.env, sub.ep, sub.sp, sub.feat_fn, HookProgram(m.init, m.restore, m.act, m.archive), cfg, sub.oracle, 1), in_axes=(0, 0, 0)))
t0 = time.time()
out = run(jax.random.split(jax.random.PRNGKey(7919 * a.seed + c0), len(idx)), stack([levels[k] for k in idx]), stack([metas[k] for k in idx])); jax.block_until_ready(out)
hit = jax.tree.map(lambda x: np.array(x[j]), out["hit"]); found = bool(np.array(out["found"])[j]); inv = np.array(out["invariants"])[j]
print(f"chunk run {time.time()-t0:.0f}s: found={found} first_hit={float(np.array(out['first_hit'])[j,0]):.0f} hit t={int(hit['t'])} worker={int(hit['worker'])} origin={int(hit['origin'])} len={int(hit['len'])} invariants(chimera,inherited,dupwriters)={inv.tolist()}")
if not found: sys.exit(0)
chain = chain_actions(hit); print("chain length", len(chain))
ok64 = bool(verify_chain_harness(sub.env, sub.ep, sub.sp, sub.feat_fn, levels[i], chain, sub.oracle, metas[i], B=a.workers)); print("verify_chain_harness B=%d:" % a.workers, ok64)
flags, st = replay_chain(sub.env, sub.ep, levels[i], chain, sub.oracle, metas[i], jit=False)
print("step-loop replay: any flag =", bool(np.array(flags).any()), "| flags in last 10 steps:", np.array(flags)[-10:].ravel().astype(int).tolist())
# compare replayed final state with the stored hit state
stored = hit["state"]; leaves_r = jax.tree.leaves(st); leaves_s = jax.tree.leaves(stored)
diffs = []
for lr, ls in zip(leaves_r, leaves_s):
    lr = np.asarray(lr); ls = np.asarray(ls)
    if lr.shape != ls.shape: diffs.append(("shape", lr.shape, ls.shape)); continue
    if np.issubdtype(lr.dtype, np.floating): diffs.append(float(np.abs(lr - ls).max()))
    else: diffs.append(float((lr != ls).sum()))
fl = [d for d in diffs if isinstance(d, float)]
print(f"replayed-vs-stored state: {len(fl)} leaves, max |diff| = {max(fl):.4g}, leaves with diff>1e-3: {sum(d > 1e-3 for d in fl)}")
# where along the chain does it diverge? replay prefixes and compare positions with nothing stored... instead report the stored state's polygon positions vs replayed
try:
    pr = np.asarray(st.polygon.position); ps = np.asarray(stored.polygon.position); act = np.asarray(stored.polygon.active)
    d = np.linalg.norm(pr - ps, axis=-1); print("polygon position drift (active shapes):", np.round(d[act], 4).tolist())
    print("stored max |x|,|y|:", np.abs(ps[act]).max(0).round(3).tolist(), "replayed:", np.abs(pr[act]).max(0).round(3).tolist())
except Exception as e: print("position compare skipped:", e)
