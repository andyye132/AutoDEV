"""M3 — rare-event splitting (adaptive multilevel splitting / RESTART, Cerou & Guyader 2007; Villen-Altamirano 1994) as a tester.
Reaching a witness is a rare event. A class-agnostic importance function I(state) measures how "extreme" a state is
(distance travelled by the moving bodies, kinetic energy, proximity to the arena boundary and to any solid shape). Every
PERIOD steps the population is ranked by the running maximum of I along each worker's trajectory; the top ELITE_FRAC workers'
current states are stored in the archive (the "level"), and the bottom SPLIT_FRAC workers are restored onto random elites
(splitting), which multiplies trajectories that have crossed the current importance level. Actions: heavy-tailed random holds
(zeta), so the dynamics between splits are the same as the best non-restoring baseline. Restores are periodic, not per-step,
so every trajectory segment is long enough to build energy."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
PERIOD = 96          # steps between splitting rounds
ELITE_FRAC = 0.25    # fraction of workers whose states define the current level
SPLIT_FRAC = 0.50    # fraction of (worst) workers restored onto elites each round
MU = 2.0             # zeta exponent for hold durations
N_MAX = 160
W_DIST, W_SPEED, W_BOUND, W_GOAL = 1.0, 0.6, 0.8, 0.4   # importance weights
WORLD = 5.0
def _cdf():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def _importance(feats, start):
    # head slots: [0:2] dyn_pos, [2:4] dyn_vel, [4] max speed, [5:7] agent pos, [7] dist-to-goal, [9:11] goal pos
    d_travel = jnp.linalg.norm(feats[:, 0:2] - start[:, 0:2], axis=-1) / WORLD
    speed = jnp.minimum(feats[:, 4] / 8.0, 1.5)
    edge = jnp.minimum(jnp.minimum(feats[:, 0], WORLD - feats[:, 0]), jnp.minimum(feats[:, 1], WORLD - feats[:, 1]))
    near_bound = 1.0 - jnp.clip(edge / (0.5 * WORLD), 0.0, 1.0)
    goal = 1.0 - jnp.clip(feats[:, 7] / WORLD, 0.0, 1.0)
    return W_DIST * d_travel + W_SPEED * speed + W_BOUND * near_bound + W_GOAL * goal
def init(rng, F, n_act, K, B):
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(),
            "t": jnp.int32(0), "start": jnp.zeros((B, F)), "best": jnp.zeros(B), "have_start": jnp.bool_(False),
            "n_elite": jnp.int32(max(1, int(ELITE_FRAC * K))), "next_slot": jnp.int32(0)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    start = jnp.where(mem["have_start"], mem["start"], feats)
    imp = _importance(feats, start); best = jnp.maximum(mem["best"], imp)
    round_now = (mem["t"] % PERIOD) == (PERIOD - 1)
    # split: the worst SPLIT_FRAC workers (by running-max importance) jump to a random valid archived elite
    order = jnp.argsort(best); rank = jnp.zeros(B, jnp.int32).at[order].set(jnp.arange(B))
    worst = rank < jnp.int32(SPLIT_FRAC * B)
    valid_p = arch_valid.astype(jnp.float32); valid_p = valid_p / jnp.maximum(valid_p.sum(), 1.0)
    slot = jax.random.choice(rng, K, (B,), p=valid_p) if True else jnp.zeros(B, jnp.int32)
    do = round_now & worst & arch_valid.any()
    best = jnp.where(do, 0.0, best)                       # a split trajectory restarts its own record
    return do, slot, {**mem, "start": start, "have_start": jnp.bool_(True), "best": best}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng); new = mem["left"] <= 0
    cur = jnp.where(new, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k2, (B,)))
    left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    return cur, {**mem, "cur": cur, "left": left}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    imp = _importance(feats, mem["start"]); best = jnp.maximum(mem["best"], imp)
    round_now = (mem["t"] % PERIOD) == (PERIOD - 2)       # store elites one step before the split
    order = jnp.argsort(-best); rank = jnp.zeros(B, jnp.int32).at[order].set(jnp.arange(B))
    elite = (rank < mem["n_elite"]) & ~done & round_now
    # elites overwrite a rolling window of slots (the archive is the current level, not a history)
    idx = jnp.cumsum(elite) - 1
    slot = (mem["next_slot"] + idx) % K
    nxt = (mem["next_slot"] + elite.sum()) % K
    return elite, jnp.where(elite, slot, 0), {**mem, "best": best, "t": mem["t"] + 1, "next_slot": nxt}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
