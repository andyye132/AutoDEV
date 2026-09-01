"""M6 — MCTS-style expansion over archived states with UCB (Ariyurek et al., "Automated video game testing using synthetic
and human-like agents", IEEE ToG 2021, use MCTS testers; Kocsis & Szepesvari 2006). Every archived state is a tree node with a
value = the best class-agnostic importance reached by trajectories expanded from it, and a visit count; expansion restores a
worker to the node with the highest UCB score and applies a random macro-action (heavy-tailed hold). Back-up: when a worker's
running importance improves, the node it was expanded from receives the value. Nodes are created for novel coverage cells
and for states that set a new importance record (so the tree grows toward extreme states)."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12
CELL_DIMS = (0, 1, 2, 3, 5, 6); CELL_RES = (0.5, 0.5, 1.0, 1.0, 0.5, 0.5)
P_EXPAND = 0.06          # per-step probability that a worker becomes a new expansion
C_UCB = 0.7
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
def _cell(mem, feats):
    x = jnp.floor(feats[:, jnp.array(CELL_DIMS)] / jnp.array(CELL_RES)).astype(jnp.int32)
    return ((x * mem["mult"]).sum(-1)) & ((1 << BITS) - 1)
def init(rng, F, n_act, K, B):
    mult = jnp.array([73856093, 19349663, 83492791, 49979687, 100000007, 15485863], jnp.int32)[: len(CELL_DIMS)]
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(), "mult": mult,
            "start": jnp.zeros((B, F)), "have_start": jnp.bool_(False), "best": jnp.zeros(B), "parent": jnp.full(B, -1, jnp.int32),
            "value": jnp.zeros(K), "visits": jnp.zeros(K), "record": jnp.float32(0.0),
            "cell_slot": jnp.full(1 << BITS, -1, jnp.int32), "next_free": jnp.int32(0)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    start = jnp.where(mem["have_start"], mem["start"], feats)
    total = jnp.maximum(mem["visits"].sum(), 1.0)
    ucb = mem["value"] + C_UCB * jnp.sqrt(jnp.log(total + 1.0) / (mem["visits"] + 1.0))
    logits = jnp.where(arch_valid, ucb, -jnp.inf)
    slot = jnp.argmax(logits[None, :] + 0.05 * jax.random.gumbel(k1, (B, K)), axis=1)      # UCB with a little noise across workers
    do = (jax.random.uniform(k2, (B,)) < P_EXPAND) & arch_valid.any()
    visits = mem["visits"].at[jnp.where(do, slot, 0)].add(do.astype(jnp.float32))
    parent = jnp.where(do, slot, mem["parent"]); best = jnp.where(do, 0.0, mem["best"])
    return do, slot, {**mem, "start": start, "have_start": jnp.bool_(True), "visits": visits, "parent": parent, "best": best, "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng); new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k2, (B,)))
    left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    return cur, {**mem, "cur": cur, "left": left}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    imp = _importance(feats, mem["start"]); improved = imp > mem["best"]; best = jnp.maximum(mem["best"], imp)
    # back-up: a worker's parent node takes the max importance reached from it
    has_parent = mem["parent"] >= 0
    value = mem["value"].at[jnp.where(has_parent, mem["parent"], 0)].max(jnp.where(has_parent, imp, -jnp.inf))
    # node creation: novel cell, or a new global importance record
    cell = _cell(mem, feats); live = ~done
    rec = imp > mem["record"]
    is_new = ((mem["cell_slot"][cell] < 0) | rec) & live
    first = jnp.full(1 << BITS, B, jnp.int32).at[jnp.where(is_new, cell, 0)].min(jnp.where(is_new, jnp.arange(B), B))
    is_new = is_new & (first[cell] == jnp.arange(B))
    order = jnp.cumsum(is_new) - 1; slot = (mem["next_free"] + order) % K
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, slot, mem["cell_slot"][0]))
    value = value.at[jnp.where(is_new, slot, 0)].set(jnp.where(is_new, imp, value[0]))     # a new node starts at its own importance
    visits = mem["visits"].at[jnp.where(is_new, slot, 0)].set(jnp.where(is_new, 0.0, mem["visits"][0]))
    record = jnp.maximum(mem["record"], imp.max())
    return is_new, jnp.where(is_new, slot, 0), {**mem, "best": best, "value": value, "visits": visits, "record": record, "cell_slot": cell_slot, "next_free": (mem["next_free"] + is_new.sum()) % K}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
