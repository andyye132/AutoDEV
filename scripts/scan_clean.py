"""Scan CLEAN levels for real engine anomalies with tester programs, batched across levels (vmapped) for GPU efficiency.
For each level: calibrate anomaly thresholds under random play, run each program (seeds), verify every first hit by
deterministic replay, save results. Supports evolved program files and generated-level indexes."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from jax2d.engine import PhysicsEngine
from autodev.kinetix_utils import list_levels, load_level, make_env, world_box, level_hash
from autodev.anomalies import anomaly_flags, raw_stats, make_thresholds, ANOMALIES, ANOMALY_MIN_STREAK, AnomalyThr
from autodev.features import make_feature_fn
from autodev.tester import TesterConfig, make_tester, BUILTIN_PROGRAMS, SourceProgram, chain_actions, replay_chain, replay_chains
from autodev.evaluate import load_instance_level, stack

ap = argparse.ArgumentParser()
ap.add_argument("--tiers", default="s,m"); ap.add_argument("--size", default="m"); ap.add_argument("--programs", default="random,repeat_random,goexplore")
ap.add_argument("--workers", type=int, default=64); ap.add_argument("--steps", type=int, default=1024); ap.add_argument("--slots", type=int, default=256)
ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--out", default="runs/scan_clean.json"); ap.add_argument("--max_levels", type=int, default=0)
ap.add_argument("--program_files", default=""); ap.add_argument("--levels_index", default=""); ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--chain_log", type=int, default=0, help="append-only lineage log size (entries); 0 = slot-pointer lineage")
args = ap.parse_args()
if args.levels_index:
    index = json.load(open(args.levels_index)); levels = [e["path"] for e in index]
else:
    levels = list_levels(tuple(args.tiers.split(",")))
if args.max_levels: levels = levels[: args.max_levels]
def _load(lv): return load_instance_level(lv, args.size) if args.levels_index else load_level(lv, args.size)
st0, sp, ep = _load(levels[0])
env = make_env(sp, ep, auto_reset=False); engine = PhysicsEngine(sp)
feat_fn, F, spec, describe = make_feature_fn(sp, ep, engine)
cfg = TesterConfig(num_workers=args.workers, steps=args.steps, slots=args.slots, min_streak=ANOMALY_MIN_STREAK, chain_log=args.chain_log)
n_act = env.action_space(ep).n
oracle = lambda s, r, d, thr: anomaly_flags(s, sp, thr)

def roll_stats(rng, st):
    obs, s = env.reset(rng, ep, st)
    def body(carry, _):
        rng, s = carry; rng, k1, k2 = jax.random.split(rng, 3)
        obs, s2, r, d, i = env.step(k2, s, jax.random.randint(k1, (), 0, n_act), ep, st)
        obs_r, s_r = env.reset(k2, ep, st)
        s_n = jax.tree.map(lambda a, b: jax.lax.select(d, a, b), s_r, s2)
        return (rng, s_n), raw_stats(s2, sp)
    return jax.lax.scan(body, (rng, s), None, length=256)[1].max(0)
stats_levels = jax.jit(jax.vmap(lambda st, rngs: jax.vmap(roll_stats, in_axes=(0, None))(rngs, st).max(0), in_axes=(0, None)))
programs = {p: BUILTIN_PROGRAMS[p]() for p in args.programs.split(",") if p}
for f in args.program_files.split(","):
    if f: programs[os.path.basename(f).replace(".py", "")] = SourceProgram(open(f).read())
runners = {p: jax.jit(jax.vmap(make_tester(env, ep, sp, feat_fn, P, cfg, oracle, len(ANOMALIES)), in_axes=(0, 0, 0))) for p, P in programs.items()}
t0 = time.time()
states = [_load(lv)[0] for lv in levels]
records = [dict(level=lv, hash=level_hash(st), box=list(world_box(st, sp)), programs={}) for lv, st in zip(levels, states)]
rngs_cal = jax.random.split(jax.random.PRNGKey(0), 64)
thrs = []
for c0 in range(0, len(levels), args.chunk):
    idx = list(range(c0, min(len(levels), c0 + args.chunk)))
    mx = np.array(stats_levels(stack([states[i] for i in idx]), rngs_cal))
    for j, i in enumerate(idx):
        thr = make_thresholds(records[i]["box"], mx[j]); thrs.append(thr)
        records[i]["calib_max"] = mx[j].tolist(); records[i]["thr"] = dict(joint_tol=float(thr.joint_tol), speed_thr=float(thr.speed_thr), angvel_thr=float(thr.angvel_thr))
print(f"[{time.time()-t0:6.0f}s] calibrated {len(levels)} levels", flush=True)
for p, run in runners.items():
    for seed in range(args.seeds):
        for c0 in range(0, len(levels), args.chunk):
            idx = list(range(c0, min(len(levels), c0 + args.chunk)))
            out = run(jax.random.split(jax.random.PRNGKey(1000 * seed + c0), len(idx)), stack([states[i] for i in idx]), stack([thrs[i] for i in idx])); jax.block_until_ready(out)
            fh = np.array(out["first_hit"]); found = np.array(out["found"])
            hits = {j: jax.tree.map(lambda x: np.array(x[j]), out["hit"]) for j in range(len(idx)) if found[j]}
            chains = {j: chain_actions(h) for j, h in hits.items()}
            # replay-verify every hit of the chunk in one batched, jitted call (vmap over chains)
            MAX_CHAIN = int(os.environ.get("AUTODEV_MAX_CHAIN", "131072"))
            js = sorted(j for j in chains if len(chains[j]) <= MAX_CHAIN); skipped = sorted(j for j in chains if len(chains[j]) > MAX_CHAIN)
            t_r = time.time()
            # exact verification: replay each chain through the harness itself (same scan body and lane count as the scan)
            from autodev.tester import verify_chain_harness
            flags_by_j = {j: np.array(verify_chain_harness(env, ep, sp, feat_fn, states[idx[j]], chains[j], oracle, thrs[idx[j]], B=args.workers, n_flags=len(ANOMALIES), min_streak=tuple(ANOMALY_MIN_STREAK)))[None, :] for j in js}
            if chains: print(f"[{time.time()-t0:6.0f}s]   replay: {len(js)} chains, lengths {sorted(len(chains[j]) for j in js)[-5:]} (max), {len(skipped)} skipped as too long ({[len(chains[j]) for j in skipped][:5]}), {time.time()-t_r:.0f}s", flush=True)
            for j, i in enumerate(idx):
                entry = dict(seed=seed, first_hit={a: (float(x) if x < cfg.budget else None) for a, x in zip(ANOMALIES, fh[j])}, found=bool(found[j]), slots=int(out["n_valid_slots"][j]))
                if found[j]:
                    hit = hits[j]; acts = chains[j]
                    if j in flags_by_j:
                        flags = flags_by_j[j]
                        fi = int(hit["flag"]); same = flags[:, fi]     # strict: the replay must raise the SAME flag (any step); any-flag kept for reference
                        entry.update(flag=ANOMALIES[fi], t=int(hit["t"]), chain_len=len(acts), actions=acts,
                                     replay_verified=bool(same.any()), replay_first=int(np.argmax(same)) if same.any() else None,
                                     replay_any_flag=bool(flags.any()), replay_flags=[ANOMALIES[k] for k in np.where(flags.any(0))[0]])
                    else:   # chain longer than AUTODEV_MAX_CHAIN: recorded but not replay-verified (counted as unverified)
                        entry.update(flag=ANOMALIES[int(hit["flag"])], t=int(hit["t"]), chain_len=len(acts), actions=acts, replay_verified=False, replay_skipped_len=len(acts))
                records[i]["programs"].setdefault(p, []).append(entry)
            nf = int(found.sum())
            print(f"[{time.time()-t0:6.0f}s] {p:14s} seed {seed} levels {idx[0]:3d}-{idx[-1]:3d}: {nf} flagged" + (" | " + ", ".join(f"{levels[i].split('/')[-1][:22]}:{ANOMALIES[int(out['hit']['flag'][j])]}" for j, i in enumerate(idx) if found[j]) if nf else ""), flush=True)
        json.dump(dict(cfg=dict(workers=args.workers, steps=args.steps, slots=args.slots, budget=cfg.budget, size=args.size, programs=list(programs)), levels=records), open(args.out, "w"), indent=1)
n_found = sum(1 for r in records for p, hs in r["programs"].items() for h in hs if h["found"])
print(f"done: {len(records)} levels; {n_found} (program,seed) runs flagged an anomaly; results in {args.out}")
