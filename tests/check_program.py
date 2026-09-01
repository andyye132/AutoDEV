"""Validate a tester-program file: loads it as a SourceProgram, runs it on 3 planted instances (CPU, small budget) and a clean
level, and prints first-hit steps. Usage: python tests/check_program.py path/to/program.py [--size m] [--steps 256]"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from jax2d.engine import PhysicsEngine
from autodev.kinetix_utils import load_level, make_env
from autodev.defects import inject
from autodev.oracles import witness
from autodev.features import make_feature_fn
from autodev.tester import TesterConfig, make_tester, SourceProgram
ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--size", default="m"); ap.add_argument("--steps", type=int, default=256); ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()
prog = SourceProgram(open(a.path).read())
st, sp, ep = load_level("m/h5_angry_birds.json", a.size); env = make_env(sp, ep, auto_reset=False); engine = PhysicsEngine(sp)
feat_fn, F, spec, describe = make_feature_fn(sp, ep, engine)
rng = np.random.default_rng(0)
inst = [inject("TUNNEL", st, sp, ep, rng), inject("SHORTCUT", st, sp, ep, rng)]
st1, _, _ = load_level("s/h1_thrust_over_ball.json", a.size); inst.append(inject("ANOMALY", st1, sp, ep, rng))
names = ["h5/TUNNEL", "h5/SHORTCUT", "h1/ANOMALY"]
cfg = TesterConfig(num_workers=a.workers, steps=a.steps, slots=128)
oracle = lambda s, r, d, meta: witness(s, r, d, meta, engine)[None]
run = jax.jit(jax.vmap(make_tester(env, ep, sp, feat_fn, prog, cfg, oracle, 1), in_axes=(0, 0, 0)))
L = jax.tree.map(lambda *x: jnp.stack(x), *[i[0] for i in inst]); M = jax.tree.map(lambda *x: jnp.stack(x), *[i[1] for i in inst])
t = time.time(); out = run(jax.random.split(jax.random.PRNGKey(0), len(inst)), L, M); jax.block_until_ready(out)
fh = np.array(out["first_hit"][:, 0])
print(f"OK: compiled and ran in {time.time()-t:.1f}s | first_hit " + ", ".join(f"{n}={int(x) if x < cfg.budget else 'none'}" for n, x in zip(names, fh)) + f" | archive slots used {np.array(out['n_valid_slots']).tolist()} | done_rate {np.array(out['done_rate']).round(3).tolist()}")
