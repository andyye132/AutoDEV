"""Tester program (hand-designed): an EXTREMES archive ("elites"). Escapes, tunnelling and runaway physics are witnessed at
extreme states, so the archive keeps, for each of D feature dimensions, the states with the largest and smallest value seen
(2*D frontier slots) plus a coverage pool keyed by coarse cells. Restores are biased toward the frontier slots of a randomly
chosen dimension (uniform over dimensions, so every extreme gets pushed), with a slow-decaying preference for frontier slots
whose extreme was extended recently. Actions: long single-direction holds (energy pumping) mixed with short random holds.

Feature vector `feats` (per worker, F floats): [0:2] dyn_pos, [2:4] dyn_vel, [4] dyn_max_speed, [5:7] green_pos,
[7] green_blue_dist, [8] time_frac, [9:11] blue_pos, ... Actions: 0..2*NM-1 motors, 2*NM no-op, 2*NM+1.. thrusters.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
EXT_DIMS = (0, 1, 2, 3, 4, 5, 6, 7)   # feature dims whose extremes are tracked: positions, velocities, max speed, green pos, goal dist
P_RESTORE = 0.08                      # per-step probability that a worker jumps to an archived state
P_FRONTIER = 0.75                     # of those, fraction that go to a frontier slot (rest: coverage pool)
BITS = 12                             # coverage-pool hash bits
CELL_DIMS = (0, 1, 2, 3, 5, 6)
CELL_RES = 0.5
MAX_SHORT = 6                         # short random holds
MAX_LONG = 40                         # long single-direction holds
P_LONG = 0.5                          # probability a new hold is long
RECENT_DECAY = 0.98                   # decay of the "recently extended" bonus per step


def init(rng, F, n_act, K, B):
    D = len(EXT_DIMS); nf = 2 * D
    return {"K": jnp.int32(K), "B": jnp.int32(B), "n_act": jnp.int32(n_act), "nf": jnp.int32(nf),
            "ext_val": jnp.concatenate([jnp.full(D, -jnp.inf), jnp.full(D, jnp.inf)]),   # max then min per dim
            "ext_recent": jnp.zeros(nf), "cell_slot": jnp.full(1 << BITS, -1, jnp.int32), "next_free": jnp.int32(nf),
            "hold": jnp.zeros(B, jnp.int32), "act": jnp.zeros(B, jnp.int32), "proj": jax.random.normal(rng, (len(CELL_DIMS), BITS))}


def _cell(mem, feats):
    x = feats[:, jnp.array(CELL_DIMS)] / CELL_RES
    bits = (x @ mem["proj"]) > 0
    return (bits * (1 << jnp.arange(BITS))).sum(-1).astype(jnp.int32)


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; nf = 2 * len(EXT_DIMS); K = arch_valid.shape[0]
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    do = jax.random.uniform(k1, (B,)) < P_RESTORE
    # frontier choice: uniform over dimensions/extremes, weighted by (1 + recent bonus), masked by validity
    w = (1.0 + mem["ext_recent"]) * arch_valid[:nf]
    w = jnp.where(w.sum() > 0, w / jnp.maximum(w.sum(), 1e-6), jnp.ones(nf) / nf)
    fslot = jax.random.choice(k2, nf, (B,), p=w)
    # coverage pool choice: least-visited valid slot among the pool, with noise
    pool_valid = arch_valid.at[:nf].set(False)
    score = jnp.where(pool_valid, -jnp.log1p(arch_visits.astype(jnp.float32)) + 0.3 * jax.random.gumbel(k3, (K,)), -jnp.inf)
    pslot = jnp.argmax(score)
    use_f = (jax.random.uniform(k4, (B,)) < P_FRONTIER) & arch_valid[fslot]
    slot = jnp.where(use_f, fslot, pslot)
    do = do & arch_valid[slot]
    hold = jnp.where(do, 0, mem["hold"])
    return do, slot.astype(jnp.int32), {**mem, "hold": hold}


def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4)
    n_act = mem["n_act"]
    new = mem["hold"] <= 0
    long = jax.random.uniform(k1, (B,)) < P_LONG
    length = jnp.where(long, jax.random.randint(k2, (B,), 8, MAX_LONG + 1), jax.random.randint(k2, (B,), 1, MAX_SHORT + 1))
    # long holds avoid the no-op (2*NM) so they pump energy; short holds are uniform
    nm2 = (n_act - 1) // 1
    a_new = jax.random.randint(k3, (B,), 0, n_act)
    noop = (n_act - 3)  # 2*NM index when the action set is 2*NM + 1 + 2 thrusters (Kinetix: 8 motors, no-op, 2 thrusters)
    a_new = jnp.where(long & (a_new == noop), (a_new + 1) % n_act, a_new)
    action = jnp.where(new, a_new, mem["act"]); hold = jnp.where(new, length, mem["hold"]) - 1
    return action.astype(jnp.int32), {**mem, "hold": hold, "act": action.astype(jnp.int32)}


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; D = len(EXT_DIMS); nf = 2 * D; K = arch_valid.shape[0]
    dims = jnp.array(EXT_DIMS)
    vals = feats[:, dims]                                                   # (B, D)
    # frontier: does any worker extend a max / min?
    cur_max, cur_min = mem["ext_val"][:D], mem["ext_val"][D:]
    best_hi = jnp.argmax(vals, 0); best_lo = jnp.argmin(vals, 0)               # (D,)
    hi_val = vals[best_hi, jnp.arange(D)]; lo_val = vals[best_lo, jnp.arange(D)]
    ext_hi = (hi_val > cur_max) & ~done[best_hi]; ext_lo = (lo_val < cur_min) & ~done[best_lo]
    new_ext = jnp.concatenate([jnp.where(ext_hi, hi_val, cur_max), jnp.where(ext_lo, lo_val, cur_min)])
    recent = mem["ext_recent"] * RECENT_DECAY + jnp.concatenate([ext_hi, ext_lo]).astype(jnp.float32)
    # one store per worker: the worker that extended an extreme stores into that frontier slot (first dim wins if several)
    slot_of_worker = jnp.full(B, -1, jnp.int32)
    for d in range(D):   # static loop over dims (D small)
        slot_of_worker = jnp.where(ext_hi[d] & (slot_of_worker < 0), jnp.where(jnp.arange(B) == best_hi[d], d, slot_of_worker), slot_of_worker)
        slot_of_worker = jnp.where(ext_lo[d] & (slot_of_worker < 0), jnp.where(jnp.arange(B) == best_lo[d], D + d, slot_of_worker), slot_of_worker)
    do_f = slot_of_worker >= 0
    # coverage pool: new cells go to free slots in [nf, K)
    cell = _cell(mem, feats); is_new = (mem["cell_slot"][cell] < 0) & ~done & ~do_f
    order = jnp.cumsum(is_new) - 1
    pslot = nf + ((mem["next_free"] - nf + order) % jnp.maximum(K - nf, 1))
    cell_slot = mem["cell_slot"].at[jnp.where(is_new, cell, 0)].set(jnp.where(is_new, pslot, mem["cell_slot"][0]))
    do = do_f | is_new
    slot = jnp.where(do_f, slot_of_worker, pslot).astype(jnp.int32)
    return do, slot, {**mem, "ext_val": new_ext, "ext_recent": recent, "cell_slot": cell_slot,
                      "next_free": nf + ((mem["next_free"] - nf + is_new.sum()) % jnp.maximum(K - nf, 1))}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
