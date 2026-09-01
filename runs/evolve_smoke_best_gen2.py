"""Tester program (evolvable). The four hooks below define how a batch of B workers explores a Kinetix level.
Kinetix states are save/restore-able: the harness keeps K archive slots; `restore` may jump a worker to a stored
state, `archive` may store the worker's current state. Everything must be pure JAX (jit/vmap-compatible, fixed shapes).

Feature vector `feats` (per worker, F floats, oracle-blind):
  [0:2] dyn_pos (centroid of moving shapes), [2:4] dyn_vel, [4] dyn_max_speed, [5:7] green_pos, [7] green_blue_dist,
  [8] time_frac, [9:11] blue_pos, then per-polygon pos/vel/rot/active/role blocks, per-circle pos/vel/active/role blocks,
  then contact flags. `obs` is Kinetix's symbolic observation. Actions: 0..2*NM-1 motor +/- (NM=4 motor bindings),
  2*NM = no-op, 2*NM+1.. thrusters (2 bindings).
Goal of the search: reach states that expose planted defects as fast as possible (steps-to-first-witness), and do so
in a way that generalizes to defect classes the search never saw."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12          # hash bits for archive cells
P_RESTORE = 0.05   # per-step probability that a worker jumps to an archived state
MAX_REP = 14       # max action repeat
CELL_DIMS = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)   # feature dims used to define cells; includes dyn_max_speed(4) and green_blue_dist(7)
CELL_RES = 0.5     # coarse grid resolution for cells
NM = 4             # number of motor bindings (see action layout in docstring)
P_NOOP_ALLOWED = 0.15   # fraction of steps allowed to sample the full action range (incl. no-op)
def init(rng, F, n_act, K, B):
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32),
            "proj": jax.random.normal(rng, (len(CELL_DIMS), BITS)), "cell_slot": jnp.full(2 ** BITS, -1, jnp.int32), "next_free": jnp.int32(0)}
def _cell(mem, feats):
    x = jnp.round(feats[:, jnp.array(CELL_DIMS)] / CELL_RES) * CELL_RES
    return ((x @ mem["proj"] > 0) * (2 ** jnp.arange(BITS))).sum(-1).astype(jnp.int32)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng)
    do = jax.random.uniform(k1, (B,)) < P_RESTORE
    speed = arch_feats[:, 4]
    speed_norm = speed / (jnp.max(jnp.where(arch_valid, speed, 0.0)) + 1e-6)
    dist = arch_feats[:, 7]
    max_dist = jnp.max(jnp.where(arch_valid, dist, 0.0)) + 1e-6
    closeness = 1.0 - jnp.where(arch_valid, dist, max_dist) / max_dist   # closer to goal -> higher
    # prefer rarely-visited, high-energy, AND goal-proximate cells (helps TUNNEL/SHORTCUT which cluster near the goal)
    logits = jnp.where(arch_valid, -0.5 * jnp.log1p(arch_visits.astype(jnp.float32)) + 0.6 * speed_norm + 0.6 * closeness, -jnp.inf)
    slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k2, (B, arch_valid.shape[0])), axis=1)
    return do & arch_valid.any(), slot, mem
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4)
    new = mem["left"] <= 0
    n_act = mem["n_act"]
    noop_idx = jnp.minimum(jnp.int32(2 * NM), n_act - 1)
    # sample uniformly over all actions EXCEPT no-op (via index-skip trick), to keep workers energetic
    m = jax.random.randint(k1, (B,), 0, jnp.maximum(n_act - 1, 1))
    non_noop = jnp.where(m < noop_idx, m, m + 1)
    full_range = jax.random.randint(k2, (B,), 0, n_act)
    use_full = jax.random.uniform(k3, (B,)) < P_NOOP_ALLOWED
    sampled = jnp.where(use_full, full_range, non_noop)
    cur = jnp.where(new, sampled, mem["cur"])
    left = jnp.where(new, jax.random.randint(k4, (B,), 1, MAX_REP + 1), mem["left"]) - 1
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