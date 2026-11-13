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
# ---- archive cells ----
BITS = 14                                            # hash bits for archive cells
CELL_DIMS = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)          # feature dims defining a cell
CELL_RES = (0.35, 0.35, 0.7, 0.7, 1.0, 0.5, 0.5, 0.35, 0.35, 0.35)   # per-dim grid resolution
PRIMES = (73856093, 19349663, 83492791, 15485863, 32452843, 49979687, 67867979, 86028121, 2038074743, 15487469)
REC_DIMS = (0, 1, 2, 3, 4, 5, 6, 7)                  # dims tracked for running extreme records (incl. green pos)
REC_EPS = 0.05                                       # margin required to count as a new record
TAIL = 12                                            # trailing feature dims (contact / active flags) = interaction signature
# ---- roles: 0 ENERGY, 1 GOAL, 2 SPREAD, 3 CHURN ----
NROLES, NMODES, NSEL = 4, 6, 6
# modes: 0 jitter, 1 ballistic, 2 thruster-hold, 3 reactive bang-bang, 4 square wave, 5 goal-seek
ROLE_MODE_W = ((0.08, 0.14, 0.10, 0.40, 0.24, 0.04),
               (0.08, 0.10, 0.04, 0.34, 0.10, 0.34),
               (0.10, 0.26, 0.14, 0.34, 0.12, 0.04),
               (0.30, 0.14, 0.06, 0.28, 0.18, 0.04))
# reactive signals: vx, vy, goal dx, goal dy, dir-proj(pos), dir-proj(vel)
ROLE_SEL_W = ((0.20, 0.20, 0.02, 0.02, 0.06, 0.50),
              (0.05, 0.05, 0.35, 0.35, 0.10, 0.10),
              (0.08, 0.08, 0.04, 0.04, 0.60, 0.16),
              (0.30, 0.30, 0.10, 0.10, 0.10, 0.10))
ROLE_PMIN = (2, 4, 8, 1)        # min half-period of the oscillators, per role
ROLE_PMAX = (12, 22, 40, 7)     # max half-period
ROLE_WIN = (20, 24, 34, 14)     # hill-climb window length (steps) per role
# ---- hill climb over the control configuration ----
OBJ_EPS = 1e-3       # improvement margin needed to keep the current configuration
P_FLIP = 0.45        # on a failed window: just invert the control polarity (cheap 1-bit repair)
P_DEEP = 0.25        # of the remaining failures, also resample the mode and reactive signal
# ---- restore policy ----
P_BASE = 0.02        # baseline per-step restore probability
P_STALL = 0.10       # restore probability once a worker is stalled
STALL_T = 45         # steps without a new archive entry before a worker counts as stalled
PURE_EVERY = 5       # every PURE_EVERY-th worker never restores (long free rollouts)
W_VISIT = 0.8        # penalty on already-visited archive cells
W_SPEED, W_DIST, W_GOAL, W_TOUCH = 0.35, 0.30, 0.20, 0.20   # general (role-independent) frontier bonuses
W_ROLE = 0.9         # weight on the worker's own role objective when picking a slot
P_GREEDY, GREEDY_T = 0.65, 1.5                       # fraction of greedy (vs uniform) slot picks, and its temperature
# ---- action policy ----
REP_SHORT, MAX_REP, HOLD = 4, 22, 25      # action-repeat lengths for the open-loop modes
P_THRPAIR = 0.30                          # chance a closed-loop worker toggles (thruster, no-op) instead of a motor pair
P_BURST = 0.22                            # chance an open-loop segment commits to a held high-index (thrust/no-op) burst
BURST_SPAN = 3                            # how many top action indices the burst draws from
GOLDEN = 0.6180339887498949               # stratifies the per-worker headings over the circle
def init(rng, F, n_act, K, B):
    n_act = jnp.int32(n_act)
    z = jnp.zeros(B, jnp.int32)
    nt = max(min(TAIL, int(F)), 1)
    kt, km, ks = jax.random.split(rng, 3)
    role = jnp.arange(B, dtype=jnp.int32) % NROLES
    mode_logits = jnp.log(jnp.array(ROLE_MODE_W, jnp.float32))[role]      # (B, NMODES)
    sel_logits = jnp.log(jnp.array(ROLE_SEL_W, jnp.float32))[role]        # (B, NSEL)
    ang = 2.0 * jnp.pi * jnp.mod(jnp.arange(B, dtype=jnp.float32) * GOLDEN, 1.0)
    dirv = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=1)                # stratified per-worker headings
    pmin = jnp.array(ROLE_PMIN, jnp.int32)[role]
    pmax = jnp.array(ROLE_PMAX, jnp.int32)[role]
    win_len = jnp.array(ROLE_WIN, jnp.int32)[role]
    return {"n_act": n_act, "K": jnp.int32(K),
            "role": role, "dir": dirv, "mode_logits": mode_logits, "sel_logits": sel_logits,
            "pmin": pmin, "pmax": pmax, "win_len": win_len,
            "tail_idx": jnp.arange(int(F) - nt, int(F), dtype=jnp.int32),
            "tail_p": (jax.random.randint(kt, (nt,), 1, 1 << 16) * 2 + 1).astype(jnp.int32),
            "cur": z, "left": z, "pm": z, "par": z, "per": pmin, "usethr": z,
            "thr": z + (n_act - 1), "t": z, "stall": z,
            "mode": jax.random.categorical(km, mode_logits, axis=-1).astype(jnp.int32),
            "sel": jax.random.categorical(ks, sel_logits, axis=-1).astype(jnp.int32),
            "win": z, "obj0": jnp.full(B, -1e30, jnp.float32),
            "restore_ok": (jnp.arange(B) % PURE_EVERY) != 0,
            "primes": jnp.array(PRIMES[:len(CELL_DIMS)], jnp.int32),
            "res": jnp.array(CELL_RES, jnp.float32),
            "dims": jnp.array(CELL_DIMS, jnp.int32),
            "rdims": jnp.array(REC_DIMS, jnp.int32),
            "rec_max": jnp.full(len(REC_DIMS), 1e30, jnp.float32),
            "rec_min": jnp.full(len(REC_DIMS), -1e30, jnp.float32),
            "rec_on": jnp.int32(0),
            "cell_slot": jnp.full(2 ** BITS + 1, -1, jnp.int32),   # last row = dummy sink for masked writes
            "slot_cell": jnp.full(K + 1, -1, jnp.int32),           # inverse map, last row = dummy sink
            "next_free": jnp.int32(0)}
def _cell(mem, feats):
    x = feats[:, mem["dims"]] / mem["res"]
    c = jnp.floor(jnp.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)).astype(jnp.int32)
    b = (feats[:, mem["tail_idx"]] > 0.5).astype(jnp.int32)      # contact / active-flag signature
    h = (c * mem["primes"]).sum(-1) + (b * mem["tail_p"]).sum(-1)
    return jnp.bitwise_and(h, jnp.int32(2 ** BITS - 1))
def _obj(mem, feats):
    """Per-worker scalar objective, one per role; the hill climb only ever compares a worker to its own past."""
    pos = jnp.nan_to_num(feats[:, 0:2], nan=0.0, posinf=1e3, neginf=-1e3)
    vel = jnp.nan_to_num(feats[:, 2:4], nan=0.0, posinf=1e3, neginf=-1e3)
    energy = jnp.abs(feats[:, 4]) + jnp.linalg.norm(vel, axis=1)                       # runaway / tunnelling speed
    goal = -feats[:, 7]                                                                # close the green-blue gap
    spread = jnp.sum(mem["dir"] * pos, axis=1) + 0.25 * jnp.linalg.norm(pos, axis=1)   # sweep an owned heading outward
    churn = ((feats[:, mem["tail_idx"]] > 0.5).sum(-1).astype(jnp.float32)
             + 0.1 * jnp.linalg.norm(vel, axis=1))                                     # stay in contact with things
    cand = jnp.stack([energy, goal, spread, churn], axis=1)
    return jnp.nan_to_num(jnp.take_along_axis(cand, mem["role"][:, None], axis=1)[:, 0], nan=-1e6)
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]
    k1, k2, k3 = jax.random.split(rng, 3)
    vf = arch_valid.astype(jnp.float32)
    n = vf.sum() + 1e-6
    pos = arch_feats[:, 0:2]
    mu = (pos * vf[:, None]).sum(0) / n
    d2 = ((pos - mu) ** 2).sum(-1)
    scale = (d2 * vf).sum() / n + 1e-6
    z_dist = jnp.minimum(jnp.sqrt(d2 / scale), 3.0)                       # far-from-centroid archive states
    sp = jnp.abs(arch_feats[:, 4])
    z_sp = jnp.log1p(sp / ((sp * vf).sum() / n + 1e-6))                   # fast states
    gd = arch_feats[:, 7]
    mgd = (gd * vf).sum() / n
    z_goal = jnp.clip((mgd - gd) / (jnp.abs(mgd) + 1e-6), -2.0, 2.0)      # states close to the goal
    touch = (arch_feats[:, mem["tail_idx"]] > 0.5).astype(jnp.float32).mean(-1)
    tm = (touch * vf).sum() / n
    z_touch = jnp.clip((touch - tm) / (tm + 1e-3), -2.0, 3.0)             # states already in contact with something
    base = (-W_VISIT * jnp.log1p(jnp.maximum(arch_visits.astype(jnp.float32), 0.0))
            + W_SPEED * z_sp + W_DIST * z_dist + W_GOAL * z_goal + W_TOUCH * z_touch)
    role_cols = jnp.stack([z_sp, z_goal, z_dist, z_touch], axis=1)        # (K, NROLES)
    role_term = jnp.take(role_cols, mem["role"], axis=1).T                # (B, K): each worker's own column
    beta = jnp.where(jax.random.uniform(k3, (B,)) < P_GREEDY, GREEDY_T, 0.0)[:, None]
    logits = jnp.where(arch_valid[None, :], beta * (base[None, :] + W_ROLE * role_term), -1e9)
    slot = jnp.argmax(logits + jax.random.gumbel(k2, (B, K)), axis=1).astype(jnp.int32)
    p = jnp.where(mem["stall"] > STALL_T, P_STALL, P_BASE)
    do = (jax.random.uniform(k1, (B,)) < p) & mem["restore_ok"] & arch_valid.any()
    mem = {**mem,
           "left": jnp.where(do, 0, mem["left"]),
           "win": jnp.where(do, 0, mem["win"]),               # re-baseline the hill climb after a teleport
           "obj0": jnp.where(do, -1e30, mem["obj0"]),
           "stall": jnp.where(do, 0, mem["stall"])}
    return do, slot, mem
def act(mem, feats, obs, rng):
    B = feats.shape[0]
    k = jax.random.split(rng, 14)
    n_act = mem["n_act"]
    n_mot = jnp.maximum(n_act - 3, jnp.int32(2))          # 2*NM motor actions occupy [0, n_mot)
    n_pair = jnp.maximum(n_mot // 2, jnp.int32(1))
    noop = jnp.clip(n_mot, 0, n_act - 1)                  # index 2*NM
    t = mem["t"] + 1
    # --- hill climb: keep the configuration while it improves this worker's objective, else repair it ---
    obj = _obj(mem, feats)
    end = mem["win"] <= 0
    improved = obj > mem["obj0"] + OBJ_EPS
    u = jax.random.uniform(k[0], (B,))
    fail = end & ~improved
    flip = fail & (u < P_FLIP)                            # wrong control polarity is the single most likely cause
    resample = fail & (u >= P_FLIP)
    par = jnp.where(flip, 1 - mem["par"], mem["par"])
    par = jnp.where(resample, jax.random.randint(k[1], (B,), 0, 2), par)
    pm = jnp.where(resample, 2 * jax.random.randint(k[2], (B,), 0, n_pair), mem["pm"])
    usethr = jnp.where(resample, (jax.random.uniform(k[3], (B,)) < P_THRPAIR).astype(jnp.int32), mem["usethr"])
    span_per = jnp.maximum(mem["pmax"] - mem["pmin"], 1)
    per = jnp.where(resample, mem["pmin"] + jax.random.randint(k[4], (B,), 0, span_per), mem["per"])
    thr = jnp.where(resample, n_act - 1 - jax.random.randint(k[5], (B,), 0, jnp.minimum(n_act, jnp.int32(2))), mem["thr"])
    deep = resample & (jax.random.uniform(k[6], (B,)) < P_DEEP)
    mode = jnp.where(deep, jax.random.categorical(k[7], mem["mode_logits"], axis=-1).astype(jnp.int32), mem["mode"])
    sel = jnp.where(deep, jax.random.categorical(k[8], mem["sel_logits"], axis=-1).astype(jnp.int32), mem["sel"])
    win = jnp.where(end, mem["win_len"], mem["win"]) - 1
    obj0 = jnp.where(end, obj, mem["obj0"])               # ratchet: the bar rises every kept window
    # --- open-loop modes 0/1/2: repeated actions (0/1 may commit to a held high-index burst) ---
    newseg = (mem["left"] <= 0) | deep
    a_rand = jax.random.randint(k[9], (B,), 0, n_act)
    span = jnp.minimum(n_act, jnp.int32(BURST_SPAN))
    a_burst = n_act - 1 - jax.random.randint(k[10], (B,), 0, span)
    burst = (jax.random.uniform(k[11], (B,)) < P_BURST) & (mode <= 1)
    a_open = jnp.where(burst, a_burst, a_rand)
    rep = jnp.where(mode == 0, jax.random.randint(k[12], (B,), 1, REP_SHORT + 1),
                    jnp.where(mode == 1, jax.random.randint(k[13], (B,), REP_SHORT, MAX_REP + 1), jnp.int32(HOLD)))
    rep = jnp.where(burst, jnp.int32(HOLD), rep)
    a_rep = jnp.where(newseg, jnp.where(mode == 2, thr, a_open), mem["cur"])
    left = jnp.where(newseg, rep, mem["left"]) - 1
    # --- toggle pair used by the closed-loop modes ---
    lo = jnp.where(usethr > 0, thr, pm)
    hi = jnp.where(usethr > 0, noop, jnp.minimum(pm + 1, n_mot - 1))
    # --- mode 3: reactive bang-bang on this worker's role signal (velocity pump / goal gap / heading sweep) ---
    dir_pos = jnp.sum(mem["dir"] * feats[:, 0:2], axis=1)
    dir_vel = jnp.sum(mem["dir"] * feats[:, 2:4], axis=1)
    sig = jnp.stack([feats[:, 2], feats[:, 3], feats[:, 5] - feats[:, 9], feats[:, 6] - feats[:, 10],
                     dir_pos, dir_vel], axis=1)
    s = jnp.take_along_axis(sig, sel[:, None], axis=1)[:, 0]
    a_react = jnp.where(((s > 0).astype(jnp.int32) ^ par) > 0, hi, lo)
    # --- mode 4: open-loop square wave at the role's timescale ---
    phase = (t // jnp.maximum(per, 1))
    a_sq = jnp.where(((phase % 2).astype(jnp.int32) ^ par) > 0, hi, lo)
    # --- mode 5: goal-seek — alternate control between the x-gap and y-gap so one segment closes both axes ---
    dxg = feats[:, 5] - feats[:, 9]
    dyg = feats[:, 6] - feats[:, 10]
    s_goal = jnp.where(((phase // 2) % 2) == 0, dxg, dyg)
    a_goal = jnp.where(((s_goal > 0).astype(jnp.int32) ^ par) > 0, hi, lo)
    cur = jnp.where(mode <= 2, a_rep, jnp.where(mode == 3, a_react, jnp.where(mode == 4, a_sq, a_goal)))
    cur = jnp.clip(jnp.nan_to_num(cur), 0, n_act - 1).astype(jnp.int32)
    return cur, {**mem, "cur": cur, "left": left, "mode": mode, "sel": sel, "pm": pm, "par": par,
                 "per": per, "thr": thr, "usethr": usethr, "win": win, "obj0": obj0,
                 "t": t, "stall": mem["stall"] + 1}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    cell = _cell(mem, feats)
    known = mem["cell_slot"][cell] >= 0
    r = feats[:, mem["rdims"]]
    rec = ((r > mem["rec_max"] + REC_EPS) | (r < mem["rec_min"] - REC_EPS)).any(-1) & (mem["rec_on"] > 0)
    do = (~known | rec) & ~done
    order = jnp.cumsum(do.astype(jnp.int32)) - 1
    slot = jnp.mod(mem["next_free"] + order, mem["K"]).astype(jnp.int32)
    dummy_c = jnp.int32(2 ** BITS)
    dummy_s = mem["K"]
    old = mem["slot_cell"][slot]
    cs = mem["cell_slot"].at[jnp.where(do & (old >= 0), old, dummy_c)].set(-1)      # free the evicted slot's cell
    cs = cs.at[jnp.where(do, cell, dummy_c)].set(slot)
    sc = mem["slot_cell"].at[jnp.where(do, slot, dummy_s)].set(jnp.where(do, cell, -1))
    bmax, bmin = r.max(0), r.min(0)
    on = mem["rec_on"] > 0
    rec_max = jnp.where(on, jnp.maximum(mem["rec_max"], bmax), bmax)
    rec_min = jnp.where(on, jnp.minimum(mem["rec_min"], bmin), bmin)
    mem = {**mem, "cell_slot": cs, "slot_cell": sc,
           "next_free": jnp.mod(mem["next_free"] + do.sum(), mem["K"]).astype(jnp.int32),
           "rec_max": rec_max, "rec_min": rec_min, "rec_on": jnp.int32(1),
           "stall": jnp.where(do, 0, mem["stall"])}
    return do, slot, mem
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)