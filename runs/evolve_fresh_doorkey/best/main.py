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
NBINS = 12            # position bins per axis over the [0,5] scaled box -> archive grid cells
DIST_BINS = 5         # coarse buckets of distance-to-goal (feats[7]) crossed into the cell key
P_RESTORE = 0.06      # per-step probability that a worker jumps to an archived state
MAX_REP = 10          # max action repeat during EXPLORE
STUCK_ESCAPE_1 = 2    # steps without moving before a turn-burst escape
STUCK_ESCAPE_2 = 5    # steps without moving before a full random re-roll escape
INTERACT_LEN = 2      # steps to hold an interact action once triggered
CLOSER_FRAC = 0.5     # fraction of the batch assigned the "closer" role (rest are "scouts")
CLOSER_PERIOD = 10    # closers force an interact attempt this often
SCOUT_PERIOD = 30     # scouts force an interact attempt this often (mostly explore instead)
NOVELTY_W = 1.0        # restore logit weight on archive novelty
PROGRESS_W_CLOSER = 2.5  # restore logit weight on progress, for closers
PROGRESS_W_SCOUT = 0.6   # restore logit weight on progress, for scouts
# phase codes
_EXPLORE, _INTERACT, _RECOVER = 0, 1, 2
def init(rng, F, n_act, K, B):
    is_minigrid_like = n_act >= 7
    scout_w = jnp.ones(n_act)
    scout_w = jnp.where(is_minigrid_like, scout_w.at[2].set(4.0), scout_w)   # forward: coverage
    scout_w = jnp.where(is_minigrid_like, scout_w.at[0].set(1.5), scout_w)   # left turn
    scout_w = jnp.where(is_minigrid_like, scout_w.at[1].set(1.5), scout_w)   # right turn
    scout_w = jnp.where(is_minigrid_like, scout_w.at[6].set(0.05), scout_w)  # done: avoid premature end
    closer_w = jnp.ones(n_act)
    closer_w = jnp.where(is_minigrid_like, closer_w.at[2].set(2.0), closer_w)
    closer_w = jnp.where(is_minigrid_like, closer_w.at[3].set(1.5), closer_w)  # pickup
    closer_w = jnp.where(is_minigrid_like, closer_w.at[5].set(1.5), closer_w)  # toggle
    closer_w = jnp.where(is_minigrid_like, closer_w.at[6].set(0.05), closer_w)
    role = (jnp.arange(B) % 100) < int(CLOSER_FRAC * 100)  # True = closer
    period = jnp.where(role, jnp.int32(CLOSER_PERIOD), jnp.int32(SCOUT_PERIOD))
    n_cells = NBINS * NBINS * DIST_BINS
    return {
        "n_act": jnp.int32(n_act), "K": jnp.int32(K), "role": role, "period": period,
        "scout_logits": jnp.log(scout_w), "closer_logits": jnp.log(closer_w),
        "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32),
        "phase": jnp.full((B,), _EXPLORE, jnp.int32),
        "step": jnp.zeros(B, jnp.int32),
        "prev_pos": jnp.zeros((B, 2), jnp.float32), "stuck": jnp.zeros(B, jnp.int32),
        "cell_slot": jnp.full(n_cells, -1, jnp.int32), "next_free": jnp.int32(0),
    }
def _cell(feats):
    pos = jnp.clip(feats[:, 5:7], 0.0, 5.0 - 1e-4)
    bx = jnp.clip(jnp.floor(pos[:, 0] / (5.0 / NBINS)).astype(jnp.int32), 0, NBINS - 1)
    by = jnp.clip(jnp.floor(pos[:, 1] / (5.0 / NBINS)).astype(jnp.int32), 0, NBINS - 1)
    dist = jnp.clip(feats[:, 7], 0.0, 1.0 - 1e-4)
    bd = jnp.clip(jnp.floor(dist * DIST_BINS).astype(jnp.int32), 0, DIST_BINS - 1)
    return (bx * NBINS + by) * DIST_BINS + bd
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; k1, k2 = jax.random.split(rng)
    do = jax.random.uniform(k1, (B,)) < P_RESTORE
    progress = -arch_feats[:, 7]   # feats[7] is distance-to-goal (lower = more progress)
    novelty = 1.0 / jnp.sqrt(arch_visits.astype(jnp.float32) + 1.0)
    base = jnp.where(arch_valid, NOVELTY_W * novelty, -jnp.inf)
    closer_logits = base + PROGRESS_W_CLOSER * jnp.where(arch_valid, progress, 0.0)
    scout_logits = base + PROGRESS_W_SCOUT * jnp.where(arch_valid, progress, 0.0)
    role_logits = jnp.where(mem["role"][:, None], closer_logits[None, :], scout_logits[None, :])
    slot = jnp.argmax(role_logits + jax.random.gumbel(k2, (B, arch_valid.shape[0])), axis=1)
    do = do & arch_valid.any()
    phase = jnp.where(do, jnp.full_like(mem["phase"], _EXPLORE), mem["phase"])
    stuck = jnp.where(do, jnp.zeros_like(mem["stuck"]), mem["stuck"])
    step = jnp.where(do, jnp.zeros_like(mem["step"]), mem["step"])
    return do, slot, {**mem, "phase": phase, "stuck": stuck, "step": step}
def _explore_action(mem, rng, B):
    role_logits = jnp.where(mem["role"][:, None], mem["closer_logits"][None, :], mem["scout_logits"][None, :])
    return jax.random.categorical(rng, role_logits)
def _interact_action(mem, rng, B):
    toggle_pair = jnp.where(jax.random.randint(rng, (B,), 0, 2) == 0, 3, 5)   # pickup / toggle
    return jnp.where(mem["n_act"] >= 7, toggle_pair, jax.random.randint(rng, (B,), 0, mem["n_act"]))
def _recover_action(mem, rng, stuck, B):
    turn_burst = jax.random.randint(rng, (B,), 0, 2)     # left/right turn
    full_reroll = jax.random.randint(rng, (B,), 0, mem["n_act"])
    return jnp.where(stuck >= STUCK_ESCAPE_2, full_reroll, turn_burst)
def act(mem, feats, obs, rng):
    B = feats.shape[0]; k1, k2, k3, k4 = jax.random.split(rng, 4)
    pos = feats[:, 5:7]
    moved = jnp.any(jnp.abs(pos - mem["prev_pos"]) > 1e-4, axis=-1)
    stuck = jnp.where(moved, jnp.zeros_like(mem["stuck"]), mem["stuck"] + 1)
    step = mem["step"] + 1
    new = mem["left"] <= 0
    due_interact = (step % mem["period"] < INTERACT_LEN) & (mem["n_act"] >= 7)
    trigger_recover2 = (stuck >= STUCK_ESCAPE_2) & new
    trigger_recover1 = (stuck >= STUCK_ESCAPE_1) & (stuck < STUCK_ESCAPE_2) & new
    trigger_recover = trigger_recover1 | trigger_recover2
    phase = jnp.where(new & trigger_recover, jnp.full_like(mem["phase"], _RECOVER), mem["phase"])
    phase = jnp.where(new & ~trigger_recover & due_interact, jnp.full_like(mem["phase"], _INTERACT), phase)
    phase = jnp.where(new & ~trigger_recover & ~due_interact, jnp.full_like(mem["phase"], _EXPLORE), phase)
    explore_act = _explore_action(mem, k2, B)
    interact_act = _interact_action(mem, k3, B)
    recover_act = _recover_action(mem, k3, stuck, B)
    sampled = jnp.where(phase == _INTERACT, interact_act, explore_act)
    sampled = jnp.where(phase == _RECOVER, recover_act, sampled)
    cur = jnp.where(new, sampled, mem["cur"])
    rep_explore = jax.random.randint(k4, (B,), 1, MAX_REP + 1)
    rep_interact = jnp.full((B,), INTERACT_LEN, jnp.int32)
    rep_recover = jnp.ones((B,), jnp.int32)
    rep = jnp.where(phase == _INTERACT, rep_interact, jnp.where(phase == _RECOVER, rep_recover, rep_explore))
    left = jnp.where(new, rep, mem["left"]) - 1
    new_stuck = jnp.where(new, jnp.zeros_like(stuck), stuck)
    return cur, {**mem, "cur": cur, "left": left, "prev_pos": pos, "stuck": new_stuck, "phase": phase, "step": step}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    cell = _cell(feats)
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