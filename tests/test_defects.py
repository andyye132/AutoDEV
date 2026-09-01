"""Sanity: oracles never fire on clean levels under random play; each injected defect can be witnessed by some behavior."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from autodev.kinetix_utils import list_levels, load_level, make_env, world_box
from autodev.defects import inject, CLASSES, DefectMeta
from autodev.oracles import witness
from jax2d.engine import PhysicsEngine

SIZE = "m"
levels = list_levels(("s", "m"))
levels = levels[: int(sys.argv[1])] if len(sys.argv) > 1 else levels
st0, sp, ep = load_level(levels[0], SIZE)
env = make_env(sp, ep, auto_reset=False)
engine = PhysicsEngine(sp)
n_act = env.action_space(ep).n
B, T = 64, 400


def rollout(rng, level, meta, policy):
    """policy: 'random' | int fixed action. Returns first-witness step per env (T if none), max speed, min/max centroid."""
    obs, s = env.reset(rng, ep, override_reset_state=level)
    def body(carry, t):
        rng, s = carry
        rng, r1, r2 = jax.random.split(rng, 3)
        a = jax.random.randint(r1, (), 0, n_act) if policy == "random" else jnp.int32(policy)
        obs, s2, rew, done, info = env.step(r2, s, a, ep)
        w = witness(s2, rew, done, meta, engine)
        # manual reset on done
        obs_r, s_r = env.reset(r2, ep, override_reset_state=level)
        s_next = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), s_r, s2)
        return (rng, s_next), (w, done)
    (_, _), (ws, dones) = jax.lax.scan(body, (rng, s), jnp.arange(T))
    first = jnp.where(ws.any(), jnp.argmax(ws), T)
    return first, ws.any(), dones.sum()

vroll = jax.jit(jax.vmap(rollout, in_axes=(0, None, None, None)), static_argnums=(3,))
rng = jax.random.PRNGKey(0)
fp = 0; results = []
for lv in levels:
    st, _, _ = load_level(lv, SIZE)
    box = world_box(st, sp)
    clean_meta_by_cls = {}
    line = f"{lv:28s} box=({box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}) "
    for cls in CLASSES:
        out = inject(cls, st, sp, ep, np.random.default_rng(0))
        if out is None:
            line += f"| {cls}: n/a "
            continue
        mst, meta = out
        # false-positive check: witness of this meta on the CLEAN level under random play
        rngs = jax.random.split(rng, B)
        f_clean, any_clean, _ = vroll(rngs, st, meta, "random")
        fpr = float(any_clean.mean())
        fp += fpr > 0
        # detection under random play and under each fixed action, on the MUTATED level
        best = T; best_pol = None
        for pol in ["random"] + list(range(n_act)):
            f, anyw, ndone = vroll(rngs, mst, meta, pol)
            med = float(jnp.where(anyw.any(), jnp.min(f), T))
            if med < best: best, best_pol = med, pol
        results.append((lv, cls, fpr, best, best_pol))
        line += f"| {cls}: fp={fpr:.2f} first={int(best) if best < T else 'none'}({best_pol}) "
    print(line, flush=True)
print("\nlevels with any false positive:", fp)
found = sum(1 for r in results if r[3] < T); print(f"injected instances witnessable by random/fixed-action play: {found}/{len(results)}")
