"""Generate fresh levels with Kinetix's procedural sampler (never-before-seen games), filter out ones that random play
solves or loses immediately, and save them as JSON for scanning. Provenance: generation date + hash."""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from jax2d.engine import PhysicsEngine
from kinetix.environment.ued.distributions import sample_kinetix_level
from kinetix.environment.ued.ued_state import UEDParams
from kinetix.environment.utils import static_env_params_from_size
from kinetix.environment.env_state import EnvParams
from autodev.kinetix_utils import make_env, save_level_json, level_hash
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=64); ap.add_argument("--size", default="m"); ap.add_argument("--out", default="runs/fresh_levels"); ap.add_argument("--seed", type=int, default=20260828)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sp = static_env_params_from_size(a.size); ep = EnvParams(); engine = PhysicsEngine(sp); ued = UEDParams()
env = make_env(sp, ep, auto_reset=False); n_act = env.action_space(ep).n; noop = sp.num_motor_bindings * 2
sample = jax.jit(lambda rng: sample_kinetix_level(rng, engine, ep, sp, ued))
def probe(rng, st, policy):
    obs, s = env.reset(rng, ep, st)
    def body(carry, t):
        rng, s, ended, outcome = carry
        rng, k1, k2 = jax.random.split(rng, 3)
        act = jnp.where(policy == 0, noop, jax.random.randint(k1, (), 0, n_act))
        obs, s2, r, d, info = env.step(k2, s, act, ep, st)
        newly = d & ~ended
        outcome = jnp.where(newly, jnp.where(r > 0.5, 1, jnp.where(r < -0.5, -1, 0)), outcome)
        s = jax.tree.map(lambda x, y: jax.lax.select(ended | d, x, y), s, s2)
        return (rng, s, ended | d, outcome), None
    return jax.lax.scan(body, (rng, s, False, jnp.int32(0)), jnp.arange(ep.max_timesteps))[0][3]
vprobe = jax.jit(jax.vmap(probe, in_axes=(0, None, None)), static_argnums=(2,))
kept, meta = 0, []
rng = jax.random.PRNGKey(a.seed); i = 0
while kept < a.n and i < 20 * a.n:
    rng, k = jax.random.split(rng); i += 1
    st = sample(k)
    rngs = jax.random.split(jax.random.PRNGKey(i), 64)
    o0 = np.array(vprobe(rngs, st, 0)); o1 = np.array(vprobe(rngs, st, 1))
    if (o0 == 1).mean() > 0.2 or (o0 == -1).mean() > 0.5 or (o1 == 1).mean() > 0.5: continue   # trivial or unfair
    name = f"fresh_{a.size}_{a.seed}_{i:04d}"; path = f"{a.out}/{name}.json"
    save_level_json(path, st, sp, ep)
    meta.append(dict(name=name, path=path, hash=level_hash(st), sampler_seed=a.seed, index=i, noop_win=float((o0==1).mean()), noop_death=float((o0==-1).mean()), random_win=float((o1==1).mean()), generated=time.strftime("%Y-%m-%d %H:%M")))
    kept += 1
json.dump(meta, open(f"{a.out}/index.json", "w"), indent=1)
print(f"kept {kept} of {i} sampled levels -> {a.out}/index.json")
