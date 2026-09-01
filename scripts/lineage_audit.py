"""Lineage audit: run the harness (with the lineage log) on one level, then for every archived state reconstruct its action chain
from the log, replay it from reset and compare the replayed state with the stored one. This is the test that caught the
archive-writer race (Aug 29): before the fix 13 of 64 archived states mismatched, after it 0 of 64 / 0 of 96.

    JAX_PLATFORMS=cpu LEVEL=0084 W=32 T=192 K=64 python scripts/lineage_audit.py
"""
import os, sys, json, time, numpy as np
ROOT='/Users/andy132/projects/research/ACT/AutoDev'; sys.path.insert(0, ROOT); os.chdir(ROOT); sys.argv=['x']
src = open('scripts/scan_clean.py').read().split('args = ap.parse_args()')[0]; exec(src)
import jax, jax.numpy as jnp
from autodev.tester import chain_actions
LEVEL = os.environ.get('LEVEL', '0084')
S = json.load(open(os.environ.get('SCAN', 'runs/hyak/scan_fresh_evolved.json'))); rec = [r for r in S['levels'] if os.path.basename(r['level']) == f'{LEVEL}.json' or r['level'].endswith(f'_{LEVEL}.json')][0]
st, sp, ep = (load_instance_level(rec['level'], 'l') if rec['level'].startswith('runs/') else load_level(rec['level'], 'l')); env = make_env(sp, ep, auto_reset=False); engine = PhysicsEngine(sp)
feat_fn, F, spec, describe = make_feature_fn(sp, ep, engine); thr = make_thresholds(rec['box'], np.array(rec['calib_max']))
oracle = lambda s, r, d, th: anomaly_flags(s, sp, th)
prog = SourceProgram(open('runs/evolve_overnight/candidates/cand1_gen037_0.6994.py').read())
W, T, K = int(os.environ.get('W', 32)), int(os.environ.get('T', 192)), int(os.environ.get('K', 64))
cfg = TesterConfig(num_workers=W, steps=T, slots=K, min_streak=ANOMALY_MIN_STREAK, chain_log=int(os.environ.get('LOG', 8192)), return_archive=True)
run = jax.jit(make_tester(env, ep, sp, feat_fn, prog, cfg, oracle, len(ANOMALIES)))
t0 = time.time(); out = run(jax.random.PRNGKey(int(os.environ.get('SEED', 0))), st, thr); jax.block_until_ready(out); print(f'run: {time.time()-t0:.0f}s found={bool(out["found"])} slots={int(out["n_valid_slots"])}')
A = jax.tree.map(np.array, out['archive'])
def chain_of(origin):
    segs = []; s = int(origin); g = 0
    while s >= 0 and g < 100000: segs.append(list(A['log_seg'][s][:A['log_seglen'][s]])); s = int(A['log_parent'][s]); g += 1
    return [int(a) for sg in reversed(segs) for a in sg], len(segs)
step = jax.jit(lambda s, a: env.step(jax.random.PRNGKey(0), s, jnp.int32(a), ep, st)[1:4])
def replay(acts):
    obs, s = env.reset(jax.random.PRNGKey(0), ep, st); dones = []
    for a in acts:
        s, r, d = step(s, a); dones.append(bool(d))
    return s, dones
def same(s1, s2, tol=1e-3):   # float drift between batched and single-lane physics is ~1e-4; lineage errors are O(0.1-1)
    return bool(np.allclose(np.array(s1.polygon.position), np.array(s2.polygon.position), atol=tol) and np.allclose(np.array(s1.circle.position), np.array(s2.circle.position), atol=tol))
valid = A['valid'][:K]; slots = [k for k in range(K) if valid[k]]
rows = []
for k in slots:
    acts, nseg = chain_of(A['logidx'][k]); arch = jax.tree.map(lambda x: x[k], A['state'])
    if not acts: rows.append((k, nseg, 0, int(arch.timestep), None, 0)); continue
    s, dones = replay(acts); d = float(max(np.abs(np.array(s.polygon.position) - np.array(arch.polygon.position)).max(), np.abs(np.array(s.circle.position) - np.array(arch.circle.position)).max()))
    rows.append((k, nseg, len(acts), int(arch.timestep), same(s, arch), sum(dones), round(d, 4)))
ok = [r for r in rows if r[4]]; bad = [r for r in rows if r[4] is False]
drift = [r for r in bad if r[6] <= 0.25]; err = [r for r in bad if r[6] > 0.25]   # lineage errors put bodies elsewhere (>0.25 m); drift is chaotic amplification of rounding
print(f'archived states: {len(rows)}  match {len(ok)}  mismatch {len(bad)}  (drift<=0.25m: {len(drift)}, lineage errors>0.25m: {len(err)})')
print('match    (nseg, len, timestep):', sorted((r[1], r[2], r[3]) for r in ok)[:20])
print('mismatch (nseg, len, timestep, dones_in_replay, max_pos_diff):', sorted((r[1], r[2], r[3], r[5], r[6]) for r in bad)[:25])
print('match max_pos_diff quantiles:', np.quantile([r[6] for r in ok], [0.5, 0.9, 1.0]) if ok else None)
if bool(out['found']):
    hit = jax.tree.map(np.array, out['hit']); acts = chain_actions(hit); s, dones = replay(acts)
    print(f'hit: flag {ANOMALIES[int(hit["flag"])]} t={int(hit["t"])} chain_len={len(acts)} origin={int(hit["origin"])} len={int(hit["len"])} match={same(s, jax.tree.map(lambda x: x, hit["state"]))} hit.timestep={int(hit["state"].timestep)} dones={sum(dones)}')
