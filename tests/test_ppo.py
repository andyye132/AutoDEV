import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from jax2d.engine import PhysicsEngine
from autodev.kinetix_utils import load_level, make_env
from autodev.defects import inject
from autodev.features import make_feature_fn
from autodev.objectives import BASELINES
from autodev.ppo import PPOConfig, make_train

st, sp, ep = load_level("s/h1_thrust_over_ball.json", "m")
env = make_env(sp, ep, auto_reset=False); engine = PhysicsEngine(sp)
feat_fn, F, spec, describe = make_feature_fn(sp, ep, engine)
print("feature dim", F); print(describe())
rng = np.random.default_rng(0)
inst = [inject("ANOMALY", st, sp, ep, rng), inject("OOB", st, sp, ep, rng), inject("DISARMED", st, sp, ep, rng)]
levels = jax.tree.map(lambda *x: jnp.stack(x), *[i[0] for i in inst]); metas = jax.tree.map(lambda *x: jnp.stack(x), *[i[1] for i in inst])
cfg = PPOConfig(num_envs=16, rollout=32, num_updates=6)
for name in ["count", "zero", "rnd", "icm", "curiosity_proximity"]:
    obj = BASELINES[name]()
    train = jax.jit(jax.vmap(make_train(env, ep, sp, engine, obj, feat_fn, F, cfg)))
    t = time.time(); out = train(jax.random.split(jax.random.PRNGKey(0), 3), levels, metas); jax.block_until_ready(out); t1 = time.time() - t
    t = time.time(); out = train(jax.random.split(jax.random.PRNGKey(1), 3), levels, metas); jax.block_until_ready(out); t2 = time.time() - t
    m = out["metrics"]
    print(f"{name:20s} compile+run {t1:.1f}s run {t2:.1f}s ({3*cfg.budget/t2:.0f} steps/s) | first_witness {np.array(out['first_witness']).tolist()} witnessed {np.array(out['witnessed']).tolist()} | solved {np.round(np.array(out['solved_frac']),2).tolist()} | r_int/update {np.round(np.array(m['r_int'])[0],3).tolist()} | loss finite {bool(np.isfinite(np.array(m['loss'])).all())}")
