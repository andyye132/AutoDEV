"""Automatic design-flaw checks on clean levels: NOOP_WIN (won with no input), NOOP_DEATH (lost with no input),
TRIVIAL_WIN (random play wins fast), plus win/death statistics. Reported separately from engine anomalies."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from autodev.kinetix_utils import list_levels, load_level, make_env
ap = argparse.ArgumentParser(); ap.add_argument("--tiers", default="s,m"); ap.add_argument("--size", default="m"); ap.add_argument("--out", default="runs/design_checks.json"); ap.add_argument("--envs", type=int, default=128)
a = ap.parse_args()
levels = list_levels(tuple(a.tiers.split(",")))
st0, sp, ep = load_level(levels[0], a.size); env = make_env(sp, ep, auto_reset=False); n_act = env.action_space(ep).n; noop = sp.num_motor_bindings * 2
def episode(rng, st, policy):
    obs, s = env.reset(rng, ep, st)
    def body(carry, t):
        rng, s, ended, outcome, t_end = carry
        rng, k1, k2 = jax.random.split(rng, 3)
        act = jnp.where(policy == 0, noop, jax.random.randint(k1, (), 0, n_act))
        obs, s2, r, d, info = env.step(k2, s, act, ep, st)
        newly = d & ~ended
        outcome = jnp.where(newly, jnp.where(r > 0.5, 1, jnp.where(r < -0.5, -1, 0)), outcome)
        t_end = jnp.where(newly, t, t_end)
        s = jax.tree.map(lambda x, y: jax.lax.select(ended | d, x, y), s, s2)
        return (rng, s, ended | d, outcome, t_end), None
    (rng, s, ended, outcome, t_end), _ = jax.lax.scan(body, (rng, s, False, jnp.int32(0), jnp.int32(ep.max_timesteps)), jnp.arange(ep.max_timesteps))
    return outcome, t_end
run = jax.jit(jax.vmap(episode, in_axes=(0, None, None)), static_argnums=(2,))
out = []
for lv in levels:
    st, _, _ = load_level(lv, a.size)
    rngs = jax.random.split(jax.random.PRNGKey(0), a.envs)
    o0, t0 = run(rngs, st, 0); o1, t1 = run(rngs, st, 1)
    o0, t0, o1, t1 = map(np.array, (o0, t0, o1, t1))
    rec = dict(level=lv, noop_win=float((o0 == 1).mean()), noop_death=float((o0 == -1).mean()), noop_t=float(t0.mean()),
               random_win=float((o1 == 1).mean()), random_death=float((o1 == -1).mean()), random_win_median_t=float(np.median(t1[o1 == 1])) if (o1 == 1).any() else None)
    flags = []
    if rec["noop_win"] > 0.5: flags.append("NOOP_WIN")
    if rec["noop_death"] > 0.5: flags.append("NOOP_DEATH")
    if rec["random_win"] > 0.5 and rec["random_win_median_t"] is not None and rec["random_win_median_t"] < 40: flags.append("TRIVIAL_WIN")
    rec["flags"] = flags; out.append(rec)
    print(f"{lv:34s} noop: win {rec['noop_win']:.2f} death {rec['noop_death']:.2f} | random: win {rec['random_win']:.2f} death {rec['random_death']:.2f} median_t {rec['random_win_median_t']} | {flags}", flush=True)
json.dump(out, open(a.out, "w"), indent=1)
print("flagged:", [(r["level"], r["flags"]) for r in out if r["flags"]])
