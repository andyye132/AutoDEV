"""M13 — velocity-aligned drive (miss diagnosis M3): like M8 but the credit of an action is the change of velocity projected on the
current direction of motion, so a worker keeps pushing the way it is already moving (escapes through a missing wall, drives off
platforms) rather than maximising speed in any direction. Original M8 text follows.
 (human QA / speedrunning heuristic; Sestini-style "human-like probes"): at each hold
boundary a worker keeps, with probability P_GREEDY, the action that produced the largest speed gain for it so far (per-worker
running estimate of the speed delta each action produced), otherwise it draws a random action; hold durations are
heavy-tailed. A coverage archive with 1/sqrt(visits) restores keeps breadth. Targets: runaway speed, leaving the arena,
tunnelling — everything that needs sustained energy build-up."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12; CELL_DIMS = (0, 1, 2, 3, 5, 6); CELL_RES = (0.5, 0.5, 1.0, 1.0, 0.5, 0.5)
P_RESTORE = 0.04
P_GREEDY = 0.6
LR = 0.3               # update rate of the per-action speed-gain estimate
MU = 2.0; N_MAX = 120
def _cdf():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def _cell(mem, feats):
    x = jnp.floor(feats[:, jnp.array(CELL_DIMS)] / jnp.array(CELL_RES)).astype(jnp.int32)
    return ((x * mem["mult"]).sum(-1)) & ((1 << BITS) - 1)
def init(rng, F, n_act, K, B):
    mult = jnp.array([73856093, 19349663, 83492791, 49979687, 100000007, 15485863], jnp.int32)[: len(CELL_DIMS)]
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(), "mult": mult,
            "gain": jnp.zeros((B, n_act)), "speed0": jnp.zeros(B), "v0": jnp.zeros((B, 2)), "cell_slot": jnp.full(1 << BITS, -1, jnp.int32), "next_free": jnp.int32(0)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    logits = jnp.where(arch_valid, -0.5 * jnp.log1p(arch_visits.astype(jnp.float32)), -jnp.inf)
    slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k1, (B, K)), axis=1)
    do = (jax.random.uniform(k2, (B,)) < P_RESTORE) & arch_valid.any()
    return do, slot, {**mem, "left": jnp.where(do, 0, mem["left"])}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3 = jax.random.split(rng, 3); new = mem["left"] <= 0
    # credit the finished hold: speed delta over the hold goes to the action that was held
    v = feats[:, 2:4]; vhat = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6); delta = ((v - mem["v0"]) * vhat).sum(-1); speed = feats[:, 4]
    gain = mem["gain"].at[jnp.arange(B), mem["cur"]].set(jnp.where(new, (1 - LR) * mem["gain"][jnp.arange(B), mem["cur"]] + LR * delta, mem["gain"][jnp.arange(B), mem["cur"]]))
    greedy = jnp.argmax(gain + 1e-3 * jax.random.gumbel(k3, gain.shape), axis=1)
    rand = jax.random.randint(k1, (B,), 0, mem["n_act"])
    pick = jnp.where(jax.random.uniform(k2, (B,)) < P_GREEDY, greedy, rand)
    cur = jnp.where(new, pick, mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k3, (B,)))
    left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    speed0 = jnp.where(new, speed, mem["speed0"]); v0 = jnp.where(new[:, None], v, mem["v0"])
    return cur, {**mem, "cur": cur, "left": left, "gain": gain, "speed0": speed0, "v0": v0}
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
