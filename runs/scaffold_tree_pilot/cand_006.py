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
BITS = 12            # hash bits for coverage-pool archive cells
P_RESTORE = 0.09      # per-step probability that a worker jumps to an archived state
MAX_SHORT = 5         # max repeat length for short random-exploration holds
MAX_MED = 14          # max repeat length for medium thruster-biased holds
MAX_LONG = 42         # max repeat length for long single-direction / thruster holds (energy pumping)
MAX_PUMP = 60          # max repeat length for resonant motor-oscillation holds (swing-pump energy buildup)
CELL_DIMS = (0, 1, 2, 3, 5, 6, 9, 10)   # feature dims used to define coverage cells
CELL_RES = 0.5        # coarse grid resolution for coverage cells
POOL_FRAC = (0.35, 0.14, 0.14, 0.17, 0.20)   # coverage / speed / green-proximity / green-closing-vel / blue-closing-vel
EXTREME_BONUS = 0.38  # restore-logit bonus for speed/proximity/closing pool slots
NM = 4                # motor bindings (fixed by env spec)
EPS = 1e-6
def _split(K):
    fracs = POOL_FRAC
    cum = [0.0]
    for f in fracs:
        cum.append(cum[-1] + f)
    bounds = [int(round(c * K)) for c in cum]
    bounds[0] = 0
    bounds[-1] = K
    for i in range(1, len(bounds) - 1):
        bounds[i] = max(bounds[i], bounds[i - 1] + 1)
    for i in range(len(bounds) - 2, 0, -1):
        bounds[i] = min(bounds[i], bounds[i + 1] - 1)
    sizes = [bounds[i + 1] - bounds[i] for i in range(5)]
    return tuple(sizes)
def init(rng, F, n_act, K, B):
    K1, K2, K3, K4, K5 = _split(K)
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "cur": jnp.zeros(B, jnp.int32),
            "left": jnp.zeros(B, jnp.int32), "mode": jnp.zeros(B, jnp.int32),
            "osc_motor": jnp.zeros(B, jnp.int32), "osc_period": jnp.ones(B, jnp.int32),
            "proj": jax.random.normal(rng, (len(CELL_DIMS), BITS)),
            "cell_slot": jnp.full(2 ** BITS, -1, jnp.int32), "next_free": jnp.int32(0)}
def _cell(mem, feats):
    x = jnp.round(feats[:, jnp.array(CELL_DIMS)] / CELL_RES) * CELL_RES
    return ((x @ mem["proj"] > 0) * (2 ** jnp.arange(BITS))).sum(-1).astype(jnp.int32)
def _closing_vel(feats, target_lo, target_hi):
    d = feats[:, target_lo:target_hi] - feats[:, 0:2]
    dist = jnp.linalg.norm(d, axis=-1) + EPS
    return (d * feats[:, 2:4]).sum(-1) / dist
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    K1, K2, K3, K4, K5 = _split(K)
    k1, k2 = jax.random.split(rng)
    do = jax.random.uniform(k1, (B,)) < P_RESTORE
    idx = jnp.arange(K)
    bonus = jnp.where(idx >= K1, EXTREME_BONUS, 0.0)
    logits = jnp.where(arch_valid, -0.5 * jnp.log1p(arch_visits.astype(jnp.float32)) + bonus, -jnp.inf)
    slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k2, (B, K)), axis=1)
    return do & arch_valid.any(), slot, mem
def act(mem, feats, obs, rng):
    B = feats.shape[0]; n_act = mem["n_act"]
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 7)
    new = mem["left"] <= 0
    # 0: short random exploration, 1: long single-motor energy pump, 2: medium thruster hold,
    # 3: long thruster burst, 4: resonant motor oscillation (swing-pump for extra kinetic energy)
    mode_probs = jnp.array([0.20, 0.18, 0.19, 0.18, 0.25])   # slightly more thruster/pump weight for high-energy contact witnesses
    mode = jnp.where(new, jax.random.choice(k1, 5, (B,), p=mode_probs), mem["mode"])
    rand_action = jax.random.randint(k2, (B,), 0, n_act)
    thr_lo = 2 * NM + 1
    thr_span = jnp.maximum(n_act - thr_lo, 1)
    thr_action = thr_lo + jax.random.randint(k3, (B,), 0, thr_span)
    thr_action = jnp.where(n_act > thr_lo, thr_action, rand_action)
    is_thruster_mode = (mode == 2) | (mode == 3)
    new_action = jnp.where(is_thruster_mode, thr_action, rand_action)
    short_len = jax.random.randint(k4, (B,), 1, MAX_SHORT + 1)
    med_len = jax.random.randint(k4, (B,), 6, MAX_MED + 1)
    long_len = jax.random.randint(k5, (B,), 15, MAX_LONG + 1)
    pump_len = jax.random.randint(k5, (B,), 30, MAX_PUMP + 1)
    rep_len = jnp.where(mode == 4, pump_len,
                         jnp.where((mode == 1) | (mode == 3), long_len, jnp.where(mode == 2, med_len, short_len)))
    osc_motor = jnp.where(new & (mode == 4), jax.random.randint(k6, (B,), 0, NM), mem["osc_motor"])
    osc_period = jnp.where(new & (mode == 4), jax.random.randint(k7, (B,), 2, 17), mem["osc_period"])
    left = jnp.where(new, rep_len, mem["left"]) - 1
    osc_sign = jnp.mod(left // jnp.maximum(osc_period, 1), 2)
    pump_action = 2 * osc_motor + osc_sign
    cur = jnp.where(new, new_action, mem["cur"])
    cur = jnp.where(mode == 4, pump_action, cur)
    return cur, {**mem, "cur": cur, "left": left, "mode": mode, "osc_motor": osc_motor, "osc_period": osc_period}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    K = arch_valid.shape[0]
    K1, K2, K3, K4, K5 = _split(K)
    live = ~done
    # coverage pool: hash-grid novelty over first K1 slots
    cell = _cell(mem, feats)
    is_new_cell = (mem["cell_slot"][cell] < 0) & live
    order = jnp.cumsum(is_new_cell) - 1
    cov_slot = (mem["next_free"] + order) % K1
    # speed pool: keep the K2 highest dyn_max_speed states seen
    sub_speed = arch_feats[K1:K1 + K2, 4]
    sub_speed_valid = arch_valid[K1:K1 + K2]
    masked_speed = jnp.where(sub_speed_valid, sub_speed, -jnp.inf)
    worst_speed_idx = jnp.argmin(masked_speed)
    worst_speed_val = masked_speed[worst_speed_idx]
    speed_beats = (feats[:, 4] > worst_speed_val) & live
    speed_slot = K1 + worst_speed_idx
    # proximity pool: keep the K3 states with smallest green_blue_dist
    dist_base = K1 + K2
    sub_dist = arch_feats[dist_base:dist_base + K3, 7]
    sub_dist_valid = arch_valid[dist_base:dist_base + K3]
    masked_dist = jnp.where(sub_dist_valid, sub_dist, jnp.inf)
    worst_dist_idx = jnp.argmax(masked_dist)
    worst_dist_val = masked_dist[worst_dist_idx]
    dist_beats = (feats[:, 7] < worst_dist_val) & live
    dist_slot = dist_base + worst_dist_idx
    # green-closing-velocity pool: states approaching the green goal fastest (goal-rush precursors)
    close_g_base = dist_base + K3
    close_g_vel = _closing_vel(feats, 5, 7)
    sub_close_g_feats = arch_feats[close_g_base:close_g_base + K4]
    sub_close_g_valid = arch_valid[close_g_base:close_g_base + K4]
    sub_close_g_vel = _closing_vel(sub_close_g_feats, 5, 7)
    masked_close_g = jnp.where(sub_close_g_valid, sub_close_g_vel, -jnp.inf)
    worst_close_g_idx = jnp.argmin(masked_close_g)
    worst_close_g_val = masked_close_g[worst_close_g_idx]
    close_g_beats = (close_g_vel > worst_close_g_val) & live
    close_g_slot = close_g_base + worst_close_g_idx
    # blue-closing-velocity pool: states rushing toward the secondary/unintended-goal point
    # (rewards SHORTCUT-style trajectories the green-only pools above never prioritized)
    close_b_base = close_g_base + K4
    close_b_vel = _closing_vel(feats, 9, 11)
    sub_close_b_feats = arch_feats[close_b_base:K]
    sub_close_b_valid = arch_valid[close_b_base:K]
    sub_close_b_vel = _closing_vel(sub_close_b_feats, 9, 11)
    masked_close_b = jnp.where(sub_close_b_valid, sub_close_b_vel, -jnp.inf)
    worst_close_b_idx = jnp.argmin(masked_close_b)
    worst_close_b_val = masked_close_b[worst_close_b_idx]
    close_b_beats = (close_b_vel > worst_close_b_val) & live
    close_b_slot = close_b_base + worst_close_b_idx
    use_speed = speed_beats & ~is_new_cell
    use_dist = dist_beats & ~is_new_cell & ~use_speed
    use_close_g = close_g_beats & ~is_new_cell & ~use_speed & ~use_dist
    use_close_b = close_b_beats & ~is_new_cell & ~use_speed & ~use_dist & ~use_close_g
    do = is_new_cell | use_speed | use_dist | use_close_g | use_close_b
    slot = jnp.where(is_new_cell, cov_slot,
                      jnp.where(use_speed, speed_slot,
                                jnp.where(use_dist, dist_slot,
                                          jnp.where(use_close_g, close_g_slot, close_b_slot))))
    cell_slot = mem["cell_slot"].at[jnp.where(is_new_cell, cell, 0)].set(
        jnp.where(is_new_cell, cov_slot, mem["cell_slot"][0]))
    return do, slot, {**mem, "cell_slot": cell_slot, "next_free": (mem["next_free"] + is_new_cell.sum()) % K1}
# EVOLVE-BLOCK-END
def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
