import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from jax2d.engine import PhysicsEngine
from autodev.kinetix_utils import load_level, make_env, world_box
from autodev.defects import inject, DefectMeta
from autodev.oracles import witness
from autodev.anomalies import anomaly_flags, raw_stats, make_thresholds, ANOMALIES
from autodev.features import make_feature_fn
from autodev.tester import TesterConfig, make_tester, BUILTIN_PROGRAMS, chain_actions, replay_chain

SIZE = "m"
st, sp, ep = load_level("m/h5_angry_birds.json", SIZE)
env = make_env(sp, ep, auto_reset=False); engine = PhysicsEngine(sp)
feat_fn, F, spec, describe = make_feature_fn(sp, ep, engine)
cfg = TesterConfig(num_workers=32, steps=256, slots=128)
rng = np.random.default_rng(0)
# planted instances on h5 + a hard OOB on h1
inst = [inject("OOB", st, sp, ep, rng), inject("TUNNEL", st, sp, ep, rng), inject("SHORTCUT", st, sp, ep, rng)]
st1, _, _ = load_level("s/h1_thrust_over_ball.json", SIZE); inst.append(inject("OOB", st1, sp, ep, rng))
names = ["h5/OOB", "h5/TUNNEL", "h5/SHORTCUT", "h1/OOB"]
oracle_planted = lambda s, r, d, meta: witness(s, r, d, meta, engine)[None]
for pname, P in BUILTIN_PROGRAMS.items():
    run = jax.jit(jax.vmap(make_tester(env, ep, sp, feat_fn, P(), cfg, oracle_planted, 1), in_axes=(0, 0, 0)))
    L = jax.tree.map(lambda *x: jnp.stack(x), *[i[0] for i in inst]); M = jax.tree.map(lambda *x: jnp.stack(x), *[i[1] for i in inst])
    t = time.time(); out = run(jax.random.split(jax.random.PRNGKey(0), len(inst)), L, M); jax.block_until_ready(out); t1 = time.time() - t
    t = time.time(); out = run(jax.random.split(jax.random.PRNGKey(1), len(inst)), L, M); jax.block_until_ready(out); t2 = time.time() - t
    fh = np.array(out["first_hit"][:, 0]); print(f"{pname:14s} compile {t1:.1f}s run {t2:.1f}s ({len(inst)*cfg.budget/t2:.0f} steps/s) | first_hit " + ", ".join(f"{n}={int(x) if x < cfg.budget else 'none'}" for n, x in zip(names, fh)) + f" | slots used {np.array(out['n_valid_slots']).tolist()}")
    # verify one found bug by deterministic replay
    for i in range(len(inst)):
        if bool(out["found"][i]):
            hit = jax.tree.map(lambda x: x[i], out["hit"]); acts = chain_actions(hit)
            flags, _ = replay_chain(env, ep, inst[i][0], acts, oracle_planted, inst[i][1])
            print(f"   replay {names[i]}: chain length {len(acts)} (origin slot {int(hit['origin'])}, flagged at t={int(hit['t'])}) -> oracle fires at step {int(np.argmax(flags[:,0])) if flags.any() else 'NEVER'} of replay")
            break
# clean level anomaly oracles: calibrate thresholds under repeat-random play, then run goexplore and see if anything is flagged
box = world_box(st, sp)
def stats_oracle(s, r, d, p): return raw_stats(s, sp)
run_stats = jax.jit(make_tester(env, ep, sp, feat_fn, BUILTIN_PROGRAMS["repeat_random"](), cfg, lambda s, r, d, p: jnp.zeros(1, bool), 1))
# compute raw stats separately with a quick vmapped rollout
def roll(rng):
    obs, s = env.reset(rng, ep, st)
    def body(carry, _):
        rng, s = carry; rng, k1, k2 = jax.random.split(rng, 3)
        obs, s, r, d, i = env.step(k2, s, jax.random.randint(k1, (), 0, env.action_space(ep).n), ep, st)
        return (rng, s), raw_stats(s, sp)
    return jax.lax.scan(body, (rng, s), None, length=256)[1].max(0)
mx = jax.jit(jax.vmap(roll))(jax.random.split(jax.random.PRNGKey(0), 64)).max(0)
thr = make_thresholds(box, np.array(mx)); print("clean-level calibration max (joint sep, speed, ang):", np.round(np.array(mx), 3).tolist(), "-> thresholds", float(thr.joint_tol), float(thr.speed_thr), float(thr.angvel_thr))
oracle_anom = lambda s, r, d, p: anomaly_flags(s, sp, p)
run = jax.jit(make_tester(env, ep, sp, feat_fn, BUILTIN_PROGRAMS["goexplore"](), TesterConfig(num_workers=32, steps=512, slots=128), oracle_anom, 4))
out = run(jax.random.PRNGKey(0), st, thr); jax.block_until_ready(out)
print("clean h5_angry_birds with goexplore: first_hit per anomaly", {a: (int(x) if x < 32*512 else None) for a, x in zip(ANOMALIES, np.array(out["first_hit"]))})
