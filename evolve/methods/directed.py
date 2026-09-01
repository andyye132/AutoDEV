"""M4 — directed exploration with a power schedule (AFLGo, Bohme et al. CCS 2017; Hawkeye): archived states are weighted by
their distance to "interesting structures" — here class-agnostic: the arena boundary, the goal shape and any contact — and
the restore schedule spends more expansions on states that are closer (simulated-annealing style: the temperature drops over
time so early exploration is broad and late exploration is directed). Archive: coverage cells (one representative per cell)
plus the distance score per slot. Actions: heavy-tailed random holds."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12
CELL_DIMS = (0, 1, 2, 3, 5, 6); CELL_RES = (0.5, 0.5, 1.0, 1.0, 0.5, 0.5)
P_RESTORE = 0.05
MU = 2.0; N_MAX = 120
WORLD = 5.0
T_START, T_END = 2.0, 0.3     # annealing temperature over the run (in units of the distance score)
W_BOUND, W_GOAL, W_SPEED = 1.0, 0.7, 0.5
def _cdf():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def _distance(feats):
    """Smaller = more interesting: distance to the nearest arena face, distance to the goal, and slowness."""
    edge = jnp.minimum(jnp.minimum(feats[:, 0], WORLD - feats[:, 0]), jnp.minimum(feats[:, 1], WORLD - feats[:, 1]))
    d_bound = jnp.clip(edge / (0.5 * WORLD), 0.0, 1.0)
    d_goal = jnp.clip(feats[:, 7] / WORLD, 0.0, 1.0)
    slow = 1.0 - jnp.minimum(feats[:, 4] / 8.0, 1.0)
    return W_BOUND * d_bound + W_GOAL * d_goal + W_SPEED * slow
def _cell(mem, feats):
    x = jnp.floor(feats[:, jnp.array(CELL_DIMS)] / jnp.array(CELL_RES)).astype(jnp.int32)
    return ((x * mem["mult"]).sum(-1)) & ((1 << BITS) - 1)
def init(rng, F, n_act, K, B):
    mult = jnp.array([73856093, 19349663, 83492791, 49979687, 100000007, 15485863], jnp.int32)[: len(CELL_DIMS)]
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(), "mult": mult,
            "t": jnp.int32(0), "T": jnp.int32(2048), "cell_slot": jnp.full(1 << BITS, -1, jnp.int32), "next_free": jnp.int32(0)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    frac = jnp.clip(mem["t"].astype(jnp.float32) / jnp.maximum(mem["T"].astype(jnp.float32), 1.0), 0.0, 1.0)
    temp = T_START * (T_END / T_START) ** frac
    d = _distance(arch_feats)
    logits = jnp.where(arch_valid, -d / temp - 0.5 * jnp.log1p(arch_visits.astype(jnp.float32)), -jnp.inf)
    slot = jax.random.categorical(k1, logits[None, :].repeat(B, 0), axis=1)
    do = (jax.random.uniform(k2, (B,)) < P_RESTORE) & arch_valid.any()
    return do, slot, {**mem, "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng); new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k2, (B,)))
    left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    return cur, {**mem, "cur": cur, "left": left, "t": mem["t"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    cell = _cell(mem, feats); live = ~done
    is_new = (mem["cell_slot"][cell] < 0) & live
    first = jnp.full(1 << BITS, B, jnp.int32).at[jnp.where(is_new, cell, 0)].min(jnp.where(is_new, jnp.arange(B), B))
    is_new = is_new & (first[cell] == jnp.arange(B))
    order = jnp.cumsum(is_new) - 1; slot = (mem["next_free"] + order) % K
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, slot, mem["cell_slot"][0]))
    return is_new, jnp.where(is_new, slot, 0), {**mem, "cell_slot": cell_slot, "next_free": (mem["next_free"] + is_new.sum()) % K}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
