"""M12 — death-aware checkpoint chaining with goal shunning (from the miss diagnosis, results/miss_diagnosis.md):
episodes end on hazard contact or on winning, so on mazes every held input dies within ~15 steps and on driving levels a
held input wins at ~200 steps — the budget degenerates into thousands of short episodes that never get deep. Mechanisms:
(1) depth-keyed checkpoints: a worker stores its state every CKPT steps while alive; each slot remembers its depth
    (steps since the level reset along its lineage) and a coarse cell; restores prefer DEEP slots (the survivors) with
    few visits, so exploration resumes where the previous episode got furthest instead of from the reset;
(2) banned actions: when an episode ends within FAST_DEATH steps of a restore, the action held right after that restore
    is banned for that slot (a per-slot action bitmask); the same rule bans the action that won the level from a slot
    (a win is a termination too), which is goal shunning: rollouts from that checkpoint are pushed past the goal;
(3) long coherent holds: zeta-distributed hold lengths (heavy tail up to 256 steps)."""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
BITS = 12; CELL_DIMS = (0, 1, 5, 6); CELL_RES = (0.5, 0.5, 0.5, 0.5)
CKPT = 24                 # steps between checkpoints of a live worker
P_RESTORE = 0.06          # per-step restore probability (also applied right after a reset, see below)
P_AFTER_RESET = 0.9       # a freshly reset worker jumps to a checkpoint with this probability
FAST_DEATH = 40           # a termination within this many steps of a restore bans the first action taken there
DEPTH_W = 1.0             # restore-logit weight of log(1 + depth)
MU = 2.0; N_MAX = 256
def _cdf():
    n = jnp.arange(1, N_MAX + 1, dtype=jnp.float32); p = n ** (-MU); return jnp.cumsum(p / p.sum())
def _cell(mem, feats):
    x = jnp.floor(feats[:, jnp.array(CELL_DIMS)] / jnp.array(CELL_RES)).astype(jnp.int32)
    return ((x * mem["mult"]).sum(-1)) & ((1 << BITS) - 1)
def init(rng, F, n_act, K, B):
    mult = jnp.array([73856093, 19349663, 83492791, 49979687], jnp.int32)
    return {"n_act": jnp.int32(n_act), "cur": jnp.zeros(B, jnp.int32), "left": jnp.zeros(B, jnp.int32), "cdf": _cdf(), "mult": mult,
            "depth": jnp.zeros(B, jnp.int32), "since_ckpt": jnp.zeros(B, jnp.int32), "just_reset": jnp.ones(B, bool),
            "src": jnp.full(B, -1, jnp.int32), "since_restore": jnp.full(B, 10 ** 6, jnp.int32), "first_act": jnp.zeros(B, jnp.int32),
            "slot_depth": jnp.zeros(K, jnp.int32), "slot_cell": jnp.zeros(K, jnp.int32), "banned": jnp.zeros((K, n_act), bool), "visits": jnp.zeros(K),
            "cell_count": jnp.zeros(1 << BITS, jnp.int32), "next_free": jnp.int32(0)}
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    novelty = -0.5 * jnp.log1p(mem["cell_count"][mem["slot_cell"]].astype(jnp.float32))
    logits = jnp.where(arch_valid, DEPTH_W * jnp.log1p(mem["slot_depth"].astype(jnp.float32)) + novelty - 0.5 * jnp.log1p(mem["visits"]), -jnp.inf)
    slot = jnp.argmax(logits[None, :] + jax.random.gumbel(k1, (B, K)), axis=1)
    p = jnp.where(mem["just_reset"], P_AFTER_RESET, P_RESTORE)
    do = (jax.random.uniform(k2, (B,)) < p) & arch_valid.any()
    visits = mem["visits"].at[jnp.where(do, slot, 0)].add(do.astype(jnp.float32))
    depth = jnp.where(do, mem["slot_depth"][slot], mem["depth"])
    return do, slot, {**mem, "visits": visits, "depth": depth, "src": jnp.where(do, slot, mem["src"]), "since_restore": jnp.where(do, 0, mem["since_restore"]),
                      "left": jnp.where(do, 0, mem["left"]), "since_ckpt": jnp.where(do, 0, mem["since_ckpt"]), "just_reset": jnp.zeros(B, bool)}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; K = mem["banned"].shape[0]; k1, k2, k3 = jax.random.split(rng, 3); new = mem["left"] <= 0
    # sample an action that is not banned at the worker's source slot (if any)
    src = jnp.maximum(mem["src"], 0); ban = jnp.where((mem["src"] >= 0)[:, None], mem["banned"][src], False)
    logits = jnp.where(ban, -jnp.inf, 0.0) + jax.random.gumbel(k1, ban.shape)
    pick = jnp.argmax(logits, axis=1); pick = jnp.where(jnp.isfinite(logits.max(1)), pick, jax.random.randint(k2, (B,), 0, mem["n_act"]))
    cur = jnp.where(new, pick, mem["cur"])
    dur = 1 + jnp.searchsorted(mem["cdf"], jax.random.uniform(k3, (B,))); left = jnp.where(new, dur.astype(jnp.int32), mem["left"]) - 1
    first_act = jnp.where(mem["since_restore"] == 0, cur, mem["first_act"])
    return cur, {**mem, "cur": cur, "left": left, "first_act": first_act, "since_restore": mem["since_restore"] + 1, "depth": mem["depth"] + 1, "since_ckpt": mem["since_ckpt"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    # bans: a termination soon after a restore bans the first action taken from that slot
    fast = done & (mem["src"] >= 0) & (mem["since_restore"] <= FAST_DEATH)
    banned = mem["banned"].at[jnp.where(fast, jnp.maximum(mem["src"], 0), 0), jnp.where(fast, mem["first_act"], 0)].set(jnp.where(fast, True, mem["banned"][0, 0]))
    # checkpoints: live workers every CKPT steps, one writer per slot (rolling), plus the cell coverage count
    cell = _cell(mem, feats); live = ~done
    cell_count = mem["cell_count"].at[jnp.where(live, cell, 0)].add(live.astype(jnp.int32))
    do = live & (mem["since_ckpt"] >= CKPT)
    order = jnp.cumsum(do) - 1; slot = (mem["next_free"] + order) % K
    slot_depth = mem["slot_depth"].at[jnp.where(do, slot, 0)].set(jnp.where(do, mem["depth"], mem["slot_depth"][0]))
    slot_cell = mem["slot_cell"].at[jnp.where(do, slot, 0)].set(jnp.where(do, cell, mem["slot_cell"][0]))
    visits = mem["visits"].at[jnp.where(do, slot, 0)].set(jnp.where(do, 0.0, mem["visits"][0]))
    banned = banned.at[jnp.where(do, slot, 0)].set(jnp.where(do[:, None], False, banned[0]))
    # resets: depth and source restart
    depth = jnp.where(done, 0, mem["depth"]); src = jnp.where(done, -1, mem["src"])
    return do, jnp.where(do, slot, 0), {**mem, "banned": banned, "cell_count": cell_count, "slot_depth": slot_depth, "slot_cell": slot_cell, "visits": visits,
                                         "since_ckpt": jnp.where(do | done, 0, mem["since_ckpt"]), "depth": depth, "src": src, "just_reset": done,
                                         "since_restore": jnp.where(done, 10 ** 6, mem["since_restore"]), "next_free": (mem["next_free"] + do.sum()) % K}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)
