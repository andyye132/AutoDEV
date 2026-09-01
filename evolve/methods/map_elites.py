"""M7 — quality-diversity archive (MAP-Elites, Mouret & Clune 2015; Go-Explore's cell archive is the special case with no
quality). Behaviour descriptor = (dyn_pos x, dyn_pos y, speed band, agent-to-goal distance band); each bin keeps the elite
state with the highest class-agnostic importance (travel + speed + boundary/goal proximity). Restores are uniform over
occupied bins (QD's standard selection), so every region of behaviour space keeps being expanded from its most extreme
representative, and elites are replaced when a better state lands in the bin. Actions: heavy-tailed random holds."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
NX, NY, NS, NG = 10, 10, 4, 3           # descriptor bins: x, y, speed band, goal-distance band  (1200 bins, K slots hashed)
P_RESTORE = 0.05
MU = 2.0; N_MAX = 120
WORLD = 5.0
W_DIST, W_SPEED, W_BOUND, W_GOAL = 1.0, 0.6, 0.8, 0.4
def _cdf():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def _importance(feats, start):
    d_travel = jnp.linalg.norm(feats[:, 0:2] - start[:, 0:2], axis=-1) / WORLD
    speed = jnp.minimum(feats[:, 4] / 8.0, 1.5)
    edge = jnp.minimum(jnp.minimum(feats[:, 0], WORLD - feats[:, 0]), jnp.minimum(feats[:, 1], WORLD - feats[:, 1]))
    near_bound = 1.0 - jnp.clip(edge / (0.5 * WORLD), 0.0, 1.0)
    goal = 1.0 - jnp.clip(feats[:, 7] / WORLD, 0.0, 1.0)
    return W_DIST * d_travel + W_SPEED * speed + W_BOUND * near_bound + W_GOAL * goal
def _bin(feats, K):
    bx = jnp.clip((feats[:, 0] / WORLD * NX).astype(jnp.int32), 0, NX - 1); by = jnp.clip((feats[:, 1] / WORLD * NY).astype(jnp.int32), 0, NY - 1)
    bs = jnp.clip((feats[:, 4] / 8.0 * NS).astype(jnp.int32), 0, NS - 1); bg = jnp.clip((feats[:, 7] / WORLD * NG).astype(jnp.int32), 0, NG - 1)
    b = ((bx * NY + by) * NS + bs) * NG + bg
    return b % K                                    # bins are mapped onto archive slots (collisions share a slot)
def init(rng, F, n_act, K, B):
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(),
            "start": jnp.zeros((B, F)), "have_start": jnp.bool_(False), "quality": jnp.full(K, -jnp.inf)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    start = jnp.where(mem["have_start"], mem["start"], feats)
    p = arch_valid.astype(jnp.float32); p = p / jnp.maximum(p.sum(), 1.0)          # uniform over occupied bins
    slot = jax.random.choice(k1, K, (B,), p=p)
    do = (jax.random.uniform(k2, (B,)) < P_RESTORE) & arch_valid.any()
    return do, slot, {**mem, "start": start, "have_start": jnp.bool_(True), "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng); new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k2, (B,)))
    left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    return cur, {**mem, "cur": cur, "left": left}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    imp = _importance(feats, mem["start"]); slot = _bin(feats, K)
    better = (imp > mem["quality"][slot]) & ~done
    # one writer per bin per step: the best importance among the workers landing in the same bin
    best_in_bin = jnp.full(K, -jnp.inf).at[jnp.where(better, slot, 0)].max(jnp.where(better, imp, -jnp.inf))
    win = better & (imp >= best_in_bin[slot])
    first = jnp.full(K, B, jnp.int32).at[jnp.where(win, slot, 0)].min(jnp.where(win, jnp.arange(B), B))
    win = win & (first[slot] == jnp.arange(B))
    quality = mem["quality"].at[jnp.where(win, slot, 0)].set(jnp.where(win, imp, mem["quality"][0]))
    return win, jnp.where(win, slot, 0), {**mem, "quality": quality}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
