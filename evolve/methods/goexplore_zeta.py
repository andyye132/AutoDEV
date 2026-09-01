"""Hybrid H1 — the Go-Explore-style coverage archive of the seed program with heavy-tailed (zeta) hold durations instead of uniform 1..12 holds."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12          # hash bits for archive cells
P_RESTORE = 0.05   # per-step probability that a worker jumps to an archived state
MU = 2.0           # zeta exponent of the hold-duration distribution (heavy-tailed; Dabney et al. 2021)
N_MAX = 120        # truncation of hold durations
CELL_DIMS = (0, 1, 2, 3, 5, 6, 9, 10)   # feature dims used to define cells
CELL_RES = 0.5     # coarse grid resolution for cells


def init(rng, F, n_act, K, B):
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32),
            "proj": jax.random.normal(rng, (len(CELL_DIMS), BITS)), "cell_slot": jnp.full(2 ** BITS, -1, jnp.int32), "next_free": jnp.int32(0),
            "cdf": jnp.cumsum((jnp.arange(1, N_MAX + 1, dtype=jnp.float32) ** (-MU)) / (jnp.arange(1, N_MAX + 1, dtype=jnp.float32) ** (-MU)).sum())}


def _cell(mem, feats):
    x = jnp.round(feats[:, jnp.array(CELL_DIMS)] / CELL_RES) * CELL_RES
    return ((x @ mem["proj"] > 0) * (2 ** jnp.arange(BITS))).sum(-1).astype(jnp.int32)


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng)
    do = jax.random.uniform(k1, (B,)) < P_RESTORE
    logits = jnp.where(arch_valid, -0.5 * jnp.log1p(arch_visits.astype(jnp.float32)), -jnp.inf)   # prefer rarely-visited cells
    slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k2, (B, arch_valid.shape[0])), axis=1)
    return do & arch_valid.any(), slot, mem


def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng)
    new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k2, (B,)))            # n ~ zeta(MU)
    left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    return cur, {**mem, "cur": cur, "left": left}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    cell = _cell(mem, feats)
    is_new = (mem["cell_slot"][cell] < 0) & ~done
    order = jnp.cumsum(is_new) - 1
    slot = (mem["next_free"] + order) % mem["K"]
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, slot, mem["cell_slot"][0]))
    return is_new, slot, {**mem, "cell_slot": cell_slot, "next_free": (mem["next_free"] + is_new.sum()) % mem["K"]}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
