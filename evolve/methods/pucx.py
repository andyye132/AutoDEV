"""PUCX -- Portfolio-UCB eXplorer.  A zero-training tester that makes run7's two heuristic decisions *principled*:

  * restore = UCB over the archive slots (Monte Carlo Elites, arXiv 2104.08781):
        score(slot) = yield_rate + UCB_BETA*sqrt(log(total)/visits) + NOVW * e3b_novelty(slot)
    where yield_rate = (new archive entries produced after restoring this slot) / (times restored).
  * novelty  = E3B elliptical episodic bonus (arXiv 2210.05805) over the CORE 11-slot feature head:
        nov(x) = x_c^T (Sigma + lam I)^-1 x_c   with Sigma a running feature covariance.
    Used both as the store gate and as the restore prior.
  * act      = a small emitter PORTFOLIO under a sliding-window UCB (Multi-Emitter MAP-Elites 2007.05352 /
    Agent57 2003.13350): {bang-bang-reactive, hold, action-sweep, REWARD-FOLLOW}.  The reward-follow arm
    (greedy on the per-action progress signal) targets the reward-dense tasks where run7 ties.

Pure JAX, fixed shapes, jit/vmap-safe.  Honours the init/restore/act/archive hook contract.
"""
import jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
FC = 11                # core feature dims shared across every engine (pos/vel/energy/agent/dist/time/goal)
LAM = 1.0              # E3B ridge on the covariance
NOVW = 0.6             # weight of the E3B novelty prior in the restore UCB
UCB_BETA = 1.5         # archive-restore UCB exploration coefficient
EMIT_BETA = 1.0        # emitter UCB exploration coefficient
P_RESTORE_BASE = 0.03  # base per-step restore probability
P_RESTORE_TIME = 0.12  # extra restore probability scaled by time_frac (branch more as the episode ages)
VISIT_W = 0.5          # (kept for parity; visits enter via the UCB term)
STORE_Q = 1.0          # store when E3B novelty exceeds this * running-mean novelty (adaptive gate)
N_EMIT = 4             # reactive(0), hold(1), sweep(2), reward-follow(3)
DUR = (64, 32, 24, 48) # per-emitter segment length (steps committed before re-selecting)
N_POL = 8              # pool of sparse linear/bang-bang controllers
POL_KEEP = 0.35        # fraction of feature dims each controller reads
POL_NOISE = 0.25       # exploration noise on the reactive readout
SWEEP_HOLD = 8         # dwell per action inside a systematic sweep
WIN = 0.99             # sliding-window (EWMA) decay for the emitter bandit + per-action progress values
PROG_WIN = 0.98        # EWMA decay of the per-action progress deltas (reward-follow arm)
FREE_EVERY = 5         # every FREE_EVERY-th worker never teleports (keeps long committed trajectories)
FIXED_EVERY = 4        # every FIXED_EVERY-th worker keeps one emitter all episode (guaranteed arm coverage)


def _norm(mem, x):
    """Per-dim min/max normalisation to [-1,1] from the running record range -- scale-free across substrates."""
    span = jnp.maximum(mem["hi"] - mem["lo"], 1e-6)
    z = jnp.clip((x - mem["lo"]) / span, 0.0, 1.0)
    return 2.0 * jnp.where(mem["hi"] > mem["lo"], z, 0.5) - 1.0


def _core(x):
    """First FC dims = the engine-agnostic head; pad if an engine somehow exposes fewer."""
    F = x.shape[-1]
    if F >= FC:
        return x[..., :FC]
    return jnp.concatenate([x, jnp.zeros(x.shape[:-1] + (FC - F,), x.dtype)], -1)


def _sinv(mem):
    return jnp.linalg.inv(mem["sig"] + LAM * jnp.eye(FC))


def _nov(mem, feats, sinv):
    """E3B elliptical novelty of each row of `feats` (uses the normalised core view)."""
    zc = _core(_norm(mem, feats))
    return jnp.einsum("bi,ij,bj->b", zc, sinv, zc)


def init(rng, F, n_act, K, B):
    F, n_act, K, B = int(F), int(n_act), int(K), int(B)
    k = jax.random.split(rng, 8)
    W1 = jax.random.normal(k[1], (N_POL, F, n_act), jnp.float32) / jnp.sqrt(jnp.float32(F))
    W1 = W1 * (jax.random.uniform(k[2], (N_POL, F, 1)) < POL_KEEP)      # sparse linear controllers (reactive arm)
    w2 = jax.random.normal(k[3], (N_POL, F), jnp.float32)
    w2 = w2 * (jax.random.uniform(k[4], (N_POL, F)) < POL_KEEP)         # sparse bang-bang directions
    dur = jnp.array(DUR, jnp.int32)
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "F": jnp.int32(F),
            # normaliser
            "lo": jnp.full(F, jnp.inf, jnp.float32), "hi": jnp.full(F, -jnp.inf, jnp.float32),
            # E3B running covariance over the core dims + running mean novelty (for the adaptive store gate)
            "sig": jnp.eye(FC, dtype=jnp.float32) * LAM, "novbar": jnp.float32(1.0),
            # archive yield bookkeeping (Monte Carlo Elites): den = times restored, num = new entries credited
            "ynum": jnp.zeros(K, jnp.float32), "yden": jnp.zeros(K, jnp.float32),
            "total": jnp.float32(1.0),
            # per-worker segment state
            "W1": W1, "w2": w2, "dur": dur,
            "emit": jnp.arange(B, dtype=jnp.int32) % N_EMIT,
            "fixed": (jnp.arange(B) % FIXED_EVERY == 0), "free": (jnp.arange(B) % FREE_EVERY == 0),
            "left": jnp.zeros(B, jnp.int32), "tick": jnp.zeros(B, jnp.int32),
            "pol": jax.random.randint(k[5], (B,), 0, N_POL), "a0": jnp.zeros(B, jnp.int32),
            # origin slot each worker was last restored from (for yield credit); -1 = fresh
            "origin": jnp.full(B, -1, jnp.int32),
            # did this worker store a new entry during the current segment (emitter reward)?
            "seg_hit": jnp.zeros(B, jnp.float32),
            # progress tracking for the reward-follow arm
            "aprev": jnp.zeros(B, jnp.int32), "pprev": jnp.zeros(B, jnp.float32),
            "pval": jnp.zeros(n_act, jnp.float32),
            # emitter bandit (sliding-window)
            "enum": jnp.zeros(N_EMIT, jnp.float32), "eden": jnp.ones(N_EMIT, jnp.float32) * 1e-3,
            "t": jnp.int32(0)}


def _progress(mem, feats):
    """Scalar the reward-follow arm ascends: closeness to goal (-normalised dist, slot 7) + a little energy (slot 4)."""
    z = _norm(mem, feats)
    return -z[:, 7] + 0.25 * z[:, 4]


def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]
    K = arch_feats.shape[0]
    k1, k2, k3 = jax.random.split(rng, 3)
    # update the normaliser from what we currently see (records widen the min/max box)
    mem = {**mem, "lo": jnp.minimum(mem["lo"], feats.min(0)), "hi": jnp.maximum(mem["hi"], feats.max(0))}
    sinv = _sinv(mem)
    # --- UCB score per archive slot (Monte Carlo Elites) ---
    yield_rate = mem["ynum"] / (mem["yden"] + 1.0)
    ucb = UCB_BETA * jnp.sqrt(jnp.log(mem["total"] + 1.0) / (arch_visits.astype(jnp.float32) + 1.0))
    nov = _nov(mem, arch_feats, sinv)                       # E3B novelty prior per slot
    score = yield_rate + ucb + NOVW * nov
    score = jnp.where(arch_valid, score, -jnp.inf)
    any_valid = arch_valid.any()
    # per-worker stochastic argmax over slots
    slot = jnp.argmax(score[None, :] + jax.random.gumbel(k1, (B, K)), axis=1).astype(jnp.int32)
    tfrac = jnp.clip(feats[:, 8], 0.0, 1.0)
    p = P_RESTORE_BASE + P_RESTORE_TIME * tfrac
    do = (jax.random.uniform(k2, (B,)) < p) & (~mem["free"]) & any_valid
    # bookkeeping: credit den to restored slots, remember origin, count restores
    add_den = jax.ops.segment_sum(do.astype(jnp.float32), jnp.where(do, slot, K), num_segments=K + 1)[:K]
    origin = jnp.where(do, slot, mem["origin"])
    mem = {**mem, "yden": mem["yden"] + add_den, "origin": origin,
           "total": mem["total"] + do.sum()}
    return do, slot, mem


def _emit_action(mem, feats, obs, rng):
    """Vectorised emitter bank -> one action per worker given each worker's current emitter id."""
    B = feats.shape[0]
    k1, k2, k3 = jax.random.split(rng, 3)
    z = _norm(mem, feats)
    # (0) reactive bang-bang: sign of a sparse linear readout of the normalised features
    w2 = mem["w2"][mem["pol"]]                                     # (B,F)
    s = (w2 * z).sum(-1) + POL_NOISE * jax.random.normal(k1, (B,))
    a_react = jnp.where(s > 0, 0, jnp.minimum(1, mem["n_act"] - 1)).astype(jnp.int32)
    # (1) hold: keep the segment's committed action a0
    a_hold = mem["a0"]
    # (2) sweep: cycle through the action set, dwelling SWEEP_HOLD steps each
    a_sweep = ((mem["tick"] // SWEEP_HOLD) % mem["n_act"]).astype(jnp.int32)
    # (3) reward-follow: greedy on the global per-action progress value (with exploration)
    pv = mem["pval"][None, :] + 0.3 * jax.random.gumbel(k2, (B, mem["pval"].shape[0]))
    a_follow = jnp.argmax(pv, axis=1).astype(jnp.int32)
    a = jnp.where(mem["emit"] == 0, a_react,
        jnp.where(mem["emit"] == 1, a_hold,
        jnp.where(mem["emit"] == 2, a_sweep, a_follow)))
    return jnp.clip(a, 0, mem["n_act"] - 1).astype(jnp.int32)


def act(mem, feats, obs, rng):
    B = feats.shape[0]
    k1, k2 = jax.random.split(rng, 2)
    a = _emit_action(mem, feats, obs, rng)
    # --- reward-follow bookkeeping: credit the progress delta to the action just taken ---
    prog = _progress(mem, feats)
    dprog = prog - mem["pprev"]
    # scatter EWMA of progress delta onto the previous action
    tgt = jax.ops.segment_sum(dprog, mem["aprev"], num_segments=mem["pval"].shape[0])
    cnt = jax.ops.segment_sum(jnp.ones(B), mem["aprev"], num_segments=mem["pval"].shape[0])
    upd = jnp.where(cnt > 0, tgt / jnp.maximum(cnt, 1.0), 0.0)
    pval = jnp.where(cnt > 0, PROG_WIN * mem["pval"] + (1 - PROG_WIN) * upd, mem["pval"])
    mem = {**mem, "aprev": a, "pprev": prog, "pval": pval, "tick": mem["tick"] + 1, "t": mem["t"] + 1}
    return a, mem


def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]
    K = arch_feats.shape[0]
    k1, k2 = jax.random.split(rng, 2)
    sinv = _sinv(mem)
    nov = _nov(mem, feats, sinv)                                   # E3B novelty of each worker's state
    # adaptive gate: store the genuinely novel, non-terminal states
    store = (~done) & (nov > STORE_Q * mem["novbar"])
    # choose a target slot per worker: prefer an INVALID (free) slot, else the lowest-yield valid slot.
    # arch_visits is NOT passed to archive(); rank valid slots purely by low yield (invalid slots win first)
    freeness = jnp.where(arch_valid, -(mem["ynum"] / (mem["yden"] + 1.0)), 1e6)
    jitter = jax.random.uniform(k1, (B, K)) * 1e-3
    slot = jnp.argmax(freeness[None, :] + jitter, axis=1).astype(jnp.int32)
    slot = jnp.where(store, slot, K).astype(jnp.int32)             # K = harness trash slot when not storing
    new_entry = store & (~arch_valid[jnp.clip(slot, 0, K - 1)])    # storing into a previously-free slot = a new entry
    # --- yield credit: a new entry credits the ORIGIN slot the worker was restored from ---
    credit = jax.ops.segment_sum((new_entry & (mem["origin"] >= 0)).astype(jnp.float32),
                                 jnp.where(mem["origin"] >= 0, mem["origin"], K), num_segments=K + 1)[:K]
    # --- update running novelty mean + E3B covariance from stored states ---
    novbar = 0.99 * mem["novbar"] + 0.01 * nov.mean()
    zc = _core(_norm(mem, feats))
    sig = 0.999 * mem["sig"] + (zc.T @ zc) / B
    # --- emitter segment accounting: mark hits, roll finished segments, re-select via UCB ---
    seg_hit = jnp.maximum(mem["seg_hit"], store.astype(jnp.float32))
    left = mem["left"] - 1
    finished = left <= 0
    # emitter bandit update on finished segments
    enum_add = jax.ops.segment_sum(jnp.where(finished, seg_hit, 0.0), mem["emit"], num_segments=N_EMIT)
    eden_add = jax.ops.segment_sum(finished.astype(jnp.float32), mem["emit"], num_segments=N_EMIT)
    enum = WIN * mem["enum"] + enum_add
    eden = WIN * mem["eden"] + eden_add
    # UCB over emitters for workers whose segment finished (fixed workers keep their emitter)
    er = enum / (eden + 1e-3)
    eucb = er + EMIT_BETA * jnp.sqrt(jnp.log(eden.sum() + 1.0) / (eden + 1e-3))
    pick = jnp.argmax(eucb[None, :] + jax.random.gumbel(k2, (B, N_EMIT)), axis=1).astype(jnp.int32)
    new_emit = jnp.where(finished & (~mem["fixed"]), pick, mem["emit"])
    new_left = jnp.where(finished, mem["dur"][new_emit], left)
    new_a0 = jnp.where(finished, jax.random.randint(k1, (B,), 0, mem["n_act"]), mem["a0"])
    new_tick = jnp.where(finished, 0, mem["tick"])
    new_seg_hit = jnp.where(finished, 0.0, seg_hit)
    mem = {**mem, "ynum": mem["ynum"] + credit, "novbar": novbar, "sig": sig,
           "emit": new_emit, "left": new_left, "a0": new_a0, "tick": new_tick, "seg_hit": new_seg_hit}
    return store, slot, mem
# EVOLVE-BLOCK-END
