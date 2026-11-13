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
# ---------------------------------------------------------------- archive: coverage (hash-cell) pool
CELL_BITS = 14          # hash bits for coverage cells
CELL_DIMS = (0, 1, 2, 3, 5, 6, 9, 10)   # candidate feature dims defining a cell (filtered against F at init)
CELL_RES = 0.4          # coarse grid resolution before hashing
ENERGY_DIM = 4          # dyn_max_speed (normalised) -- proxy for "interesting / high-energy" state
ENERGY_W = 0.5          # restore bias toward high normalised-energy states
ENERGY_MARGIN = 0.05    # min normalised-energy gain needed to overwrite an occupied cell
# ---------------------------------------------------------------- archive: frontier (per-dim record) pool
REC_FRAC = 0.35         # fraction of archive slots reserved for the frontier pool
REC_TOL = 0.05          # a record must beat the old extreme by this fraction of the observed span
REC_SKIP = 8            # dim excluded from records (time_frac: monotone on every substrate)
PRIO_REC = 1.0          # restore preference for the frontier pool over the coverage pool
# ---------------------------------------------------------------- ratchet (monotone-progress) detector
PROG_W = 1.0            # restore bias toward states deep along one-way ("ratchet") feature dims
PROG_CELL_W = 0.5       # weight of progress in the coverage-cell overwrite value
RAT_DECAY = 0.995       # EWMA decay of the per-dim up/down move tallies
RAT_EPS = 0.02          # a move counts only if it exceeds this fraction of the dim's observed span
RAT_MIN = 0.8           # up-fraction above which a dim starts counting as a ratchet
# ---------------------------------------------------------------- rare-dim detector (discrete progress dims)
CHG_DECAY = 0.999       # EWMA decay of the per-dim "did this dim move at all" tallies (slow: global statistic)
RARE_MOVE_EPS = 0.02    # a dim counts as moved if it shifts by this fraction of its observed span
RARE_SMOOTH = 0.02      # smoothing added to a dim's change rate before inverting it into a rarity
RARE_CAP = 20.0         # cap on the inverted rate; also the normaliser, so rarity lands in [0, 1]
RARE_WARM = 30.0        # valid samples a dim needs before its rarity is trusted
RARE_DEPTH_W = 0.5      # how much a rare-but-not-monotone dim counts toward the progress axis
SIG_MIN = 0.3           # rarity above which a dim enters the exact cell signature
SIG_LEVELS = 4          # quantisation levels per signature dim (0..SIG_LEVELS)
R_RARE = 1.0            # credit for making a seldom-changing feature dim move
DIVE_W = 2.5            # progress weight for "diver" workers (relaunch from the deepest state)
DIVE_EVERY = 3          # every DIVE_EVERY-th worker is a diver
# ---------------------------------------------------------------- restore policy
P_RESTORE_BASE = 0.03   # base per-step restore probability
P_RESTORE_TIME = 0.12   # extra restore probability scaled by time_frac (branch more as the episode ages)
VISIT_W = 0.5           # penalty on already-visited archive slots
STAG_TOL = 2e-3         # per-step normalised feature motion below which a worker counts as stuck
STAG_LIMIT = 40         # consecutive stuck steps that force a teleport
FREE_EVERY = 5          # every FREE_EVERY-th worker never teleports (preserves long committed trajectories)
# ---------------------------------------------------------------- behaviour bank
M_RAND, M_REP, M_HOLD, M_SWEEP, M_REACT, M_LIN = 0, 1, 2, 3, 4, 5
N_MODE = 6
DUR_MAX = (6, 14, 45, 48, 80, 64)   # per-mode max segment length (open-loop holds short, closed-loop long)
N_POL = 8               # pool of random controllers
POL_KEEP = 0.35         # fraction of feature dims each controller reads (sparse -> clean, interpretable directions)
POL_NOISE = 0.25        # exploration noise on linear-controller scores
SWEEP_HOLD = 8          # max per-action dwell inside a systematic action sweep
FIXED_EVERY = 4         # every FIXED_EVERY-th worker keeps one mode for the whole episode (guaranteed primitive coverage)
# ---------------------------------------------------------------- mode bandit
BANDIT_BETA = 3.0       # softmax sharpness over per-mode reward rates
BANDIT_EPS = 0.25       # uniform floor: every primitive stays alive (transfer to unseen defect classes)
BANDIT_DECAY = 0.99     # EWMA decay of the reward-rate estimate (~100-step window)
R_REC = 1.0             # credit for pushing a per-dim frontier record
R_CELL = 0.25           # credit for occupying a fresh novelty cell
R_PROG = 1.0            # credit per unit of dense ratchet-progress gained this step (steady achievement chains)
# ---------------------------------------------------------------- rare-record bias (favour hard, seldom-hit dims)
REC_RATE_DECAY = 0.995  # EWMA decay of the per-dim record-hit rate (how often a dim's frontier gets pushed)
RARE_EPS = 0.05         # smoothing added to the hit-rate before inverting into a rarity score
RARE_CLIP = 8.0         # cap on rarity so a never-hit dim cannot dominate the score with an unbounded value
REC_RARE_W = 0.6        # weight of the rarity bonus in the frontier-slot restore priority
R_REC_RARE_W = 1.5      # weight of the rarity bonus on the frontier-record bandit reward
def init(rng, F, n_act, K, B):
    F, n_act, K, B = int(F), int(n_act), int(K), int(B)
    k = jax.random.split(rng, 10)
    n_rec = max(1, min(K - 1, int(round(K * REC_FRAC))))
    dims = tuple(d for d in CELL_DIMS if d < F) or tuple(range(min(F, 4)))
    W1 = jax.random.normal(k[1], (N_POL, F, n_act), jnp.float32) / jnp.sqrt(jnp.float32(F))
    W1 = W1 * (jax.random.uniform(k[2], (N_POL, F, 1)) < POL_KEEP)          # sparse linear controllers
    w2 = jax.random.normal(k[3], (N_POL, F), jnp.float32)
    w2 = w2 * (jax.random.uniform(k[4], (N_POL, F)) < POL_KEEP)             # sparse bang-bang directions
    return {"n_act": jnp.int32(n_act), "K": jnp.int32(K), "KR": jnp.int32(n_rec),
            # --- normaliser (shared scale-free view)
            "lo": jnp.full(F, jnp.inf, jnp.float32), "hi": jnp.full(F, -jnp.inf, jnp.float32),
            "rmask": (jnp.arange(F) != REC_SKIP),
            # --- coverage pool
            "cdims": jnp.array(dims, jnp.int32),
            "proj": jax.random.normal(k[0], (len(dims), CELL_BITS)), "cell_slot": jnp.full(2 ** CELL_BITS, -1, jnp.int32),
            "next_free": jnp.int32(0),
            # --- behaviour bank parameters
            "W1": W1, "w2": w2,
            # --- per-worker segment state
            "mode": jnp.arange(B, dtype=jnp.int32) % N_MODE,
            "fixed": (jnp.arange(B) % FIXED_EVERY == 0), "free": (jnp.arange(B) % FREE_EVERY == 0),
            "left": jnp.zeros(B, jnp.int32), "tick": jnp.zeros(B, jnp.int32),
            "a0": jnp.zeros(B, jnp.int32), "aE": jnp.zeros(B, jnp.int32),
            "aA": jnp.zeros(B, jnp.int32), "aB": jnp.zeros(B, jnp.int32),
            "pol": jax.random.randint(k[5], (B,), 0, N_POL), "hold": jnp.ones(B, jnp.int32),
            # --- stagnation detector
            "prevz": jnp.zeros((B, F), jnp.float32), "stag": jnp.zeros(B, jnp.int32),
            # --- ratchet (monotone-progress) detector
            "inc": jnp.zeros(F, jnp.float32), "dec": jnp.zeros(F, jnp.float32),
            "pf": jnp.zeros((B, F), jnp.float32), "res": jnp.zeros(B, bool), "t": jnp.int32(0),
            # --- rare-record bias
            "rec_rate": jnp.zeros(F, jnp.float32), "rec_val": jnp.zeros(n_rec, jnp.float32),
            # --- rare-dim detector: how often each dim moves at all, and the rarity weight derived from it
            "chg": jnp.zeros(F, jnp.float32), "cok": jnp.zeros(F, jnp.float32),
            "rarew": jnp.zeros(F, jnp.float32),
            "sigmul": jax.random.randint(k[6], (F,), 1, 2 ** CELL_BITS).astype(jnp.int32),
            # --- per-worker depth appetite (divers always relaunch from the deepest launch point)
            "dboost": jnp.where(jnp.arange(B) % DIVE_EVERY == 0, DIVE_W, PROG_W).astype(jnp.float32),
            # --- bandit statistics
            "num": jnp.zeros(N_MODE, jnp.float32), "den": jnp.zeros(N_MODE, jnp.float32)}
# ---------------------------------------------------------------- shared helpers
def _cell(mem, feats):
    """Coverage address = coarse projection of the continuous dims XOR an EXACT signature of the discrete ones.
    The projection half is quantised on the NORMALISED view, so cell granularity is the same on every substrate
    whatever the physical units. The signature half is the important one: dims that seldom change (inventory
    counters, achievement ticks, block/contact flags) get their own exact bins, so every distinct progress
    combination occupies its own cell instead of colliding with the same position at zero inventory."""
    z = _norm(mem, feats)
    q = jnp.round(z[:, mem["cdims"]] / CELL_RES) * CELL_RES
    base = ((q @ mem["proj"] > 0) * (2 ** jnp.arange(CELL_BITS))).sum(-1).astype(jnp.int32)
    m = ((mem["rarew"] > SIG_MIN) & mem["rmask"]).astype(jnp.int32)          # all-zero -> pure projection hash
    lvl = jnp.round(jnp.clip(0.5 * (z + 1.0), 0.0, 1.0) * SIG_LEVELS).astype(jnp.int32)
    sig = ((lvl * m) * mem["sigmul"][None, :]).sum(-1)
    return jnp.bitwise_xor(base, jnp.bitwise_and(sig, 2 ** CELL_BITS - 1)).astype(jnp.int32)
def _norm(mem, x):
    """Per-dim min-max normalisation to [-1, 1] from the running record range: scale-free across substrates,
    and dims that never varied map to exactly 0 so they cannot bias any controller."""
    span = jnp.maximum(mem["hi"] - mem["lo"], 1e-6)
    z = jnp.clip((x - mem["lo"]) / span, 0.0, 1.0)
    return 2.0 * jnp.where(mem["hi"] > mem["lo"], z, 0.5) - 1.0
def _energy(mem, x):
    if x.shape[-1] > ENERGY_DIM:
        return _norm(mem, x)[:, ENERGY_DIM]
    return jnp.zeros(x.shape[0], jnp.float32)
def _ratchet(mem):
    """Per-dim weight in [0, 1] measuring how one-way a feature dim is. Inventory counts, achievement ticks
    and depth counters only ever go up and score ~1; positions/velocities/contacts oscillate and score 0.
    A substrate with no one-way dims yields an all-zero vector, so every consumer degrades to a no-op."""
    tot = mem["inc"] + mem["dec"]
    frac = mem["inc"] / jnp.maximum(tot, 1e-6)
    w = jnp.clip((frac - RAT_MIN) / (1.0 - RAT_MIN), 0.0, 1.0) * (tot > 1.0)
    return w * mem["rmask"].astype(jnp.float32)
def _progress(mem, x):
    """How deep a state is along the progress dims: scale-free, in [0, 1], exactly 0 when none exist.
    Strictly one-way ratchet dims count fully; dims that merely change *rarely* count partially. The second
    family is the one that matters on crafting substrates: an inventory counter goes up when gathered and back
    down when spent, so it never passes the ratchet test, yet it is exactly the axis a chain moves along."""
    var = (mem["hi"] > mem["lo"]).astype(jnp.float32) * mem["rmask"].astype(jnp.float32)
    w = jnp.maximum(_ratchet(mem), RARE_DEPTH_W * mem["rarew"] * var)
    return (0.5 * (_norm(mem, x) + 1.0) * w).sum(-1) / jnp.maximum(w.sum(), 1e-3)
def _mode_logits(mem):
    """Bandit policy over the behaviour bank: softmax on decayed reward rates with a uniform exploration floor."""
    val = mem["num"] / (mem["den"] + 1e-3)
    p = jax.nn.softmax(BANDIT_BETA * val)
    return jnp.log((1.0 - BANDIT_EPS) * p + BANDIT_EPS / N_MODE)
# ---------------------------------------------------------------- hooks
def restore(mem, feats, arch_feats, arch_valid, arch_visits, rng):
    B = feats.shape[0]; K = arch_valid.shape[0]; k1, k2 = jax.random.split(rng)
    tf = feats[:, 8] if feats.shape[1] > 8 else jnp.zeros(B, jnp.float32)
    p = P_RESTORE_BASE + P_RESTORE_TIME * tf
    trig = (jax.random.uniform(k1, (B,)) < p) | (mem["stag"] > STAG_LIMIT)   # timed OR stuck-triggered
    do = trig & arch_valid.any() & (~mem["free"])
    KR_static = mem["rec_val"].shape[0]
    rec_bonus = jnp.pad(mem["rec_val"], (0, K - KR_static))                  # rarity of each frontier slot's dim
    prio = jnp.where(jnp.arange(K) < KR_static, PRIO_REC + REC_RARE_W * rec_bonus, 0.0)  # prefer rare frontier slots
    score = (prio - VISIT_W * jnp.log1p(arch_visits.astype(jnp.float32))
             + ENERGY_W * _energy(mem, arch_feats))
    logits = jnp.where(arch_valid, score, -jnp.inf)
    dep = _progress(mem, arch_feats)          # divers weight depth hard; the rest keep breadth
    slot = jnp.argmax(logits[None, :] + mem["dboost"][:, None] * dep[None, :]
                      + jax.random.gumbel(k2, (B, K)), axis=1)
    slot = jnp.clip(slot, 0, K - 1).astype(jnp.int32)
    left = jnp.where(do, 0, mem["left"])                 # a restored worker starts a freshly sampled behaviour
    return do, slot, {**mem, "left": left, "tick": jnp.where(do, 0, mem["tick"]),
                       "stag": jnp.where(do, 0, mem["stag"]), "res": do | mem["res"]}
def act(mem, feats, obs, rng):
    B = feats.shape[0]; n_act = mem["n_act"]; k = jax.random.split(rng, 11)
    new = mem["left"] <= 0                               # segment boundary -> resample mode + its parameters
    samp = jax.random.categorical(k[0], _mode_logits(mem), shape=(B,)).astype(jnp.int32)
    mode = jnp.where(new & (~mem["fixed"]), samp, mem["mode"])   # fixed specialists never leave their assigned mode
    a0 = jnp.where(new, jax.random.randint(k[1], (B,), 0, n_act), mem["a0"])
    aE = jnp.where(new, jnp.where(jax.random.uniform(k[2], (B,)) < 0.5, 0, n_act - 1), mem["aE"])
    aA = jnp.where(new, jax.random.randint(k[3], (B,), 0, n_act), mem["aA"])
    aB = jnp.where(new, (aA + 1 + jax.random.randint(k[4], (B,), 0, jnp.maximum(n_act - 1, 1))) % n_act, mem["aB"])
    pol = jnp.where(new, jax.random.randint(k[5], (B,), 0, N_POL), mem["pol"])
    hold = jnp.where(new, 1 + jax.random.randint(k[6], (B,), 0, SWEEP_HOLD), mem["hold"])
    dmax = jnp.take(jnp.array(DUR_MAX, jnp.int32), mode).astype(jnp.float32)
    dur = 1 + (jax.random.uniform(k[7], (B,)) * dmax).astype(jnp.int32)
    left = jnp.where(new, dur, mem["left"]) - 1
    tick = jnp.where(new, 0, mem["tick"] + 1)
    z = _norm(mem, feats)                                # scale-free view shared by both closed-loop primitives
    a_rand = jax.random.randint(k[8], (B,), 0, n_act)                                  # M_RAND : per-step noise
    a_sweep = (a0 + tick // hold) % n_act                                              # M_SWEEP: systematic coverage
    a_react = jnp.where((z * mem["w2"][pol]).sum(-1) > 0, aA, aB)                      # M_REACT: bang-bang resonance
    scores = jnp.einsum("bf,bfa->ba", z, mem["W1"][pol]) + POL_NOISE * jax.random.gumbel(k[9], (B, mem["W1"].shape[-1]))
    a_lin = jnp.argmax(scores, -1).astype(jnp.int32)                                   # M_LIN  : linear controller
    cand = jnp.stack([a_rand, a0, aE, a_sweep, a_react, a_lin], axis=1)                # (B, N_MODE) behaviour bank
    action = jnp.take_along_axis(cand, mode[:, None], axis=1)[:, 0].astype(jnp.int32)
    # ---- stagnation detector (only dims that have ever varied count toward motion)
    w = (mem["hi"] > mem["lo"]).astype(jnp.float32)
    dz = (jnp.abs(z - mem["prevz"]) * w).sum(-1) / jnp.maximum(w.sum(), 1.0)
    stag = jnp.where(dz < STAG_TOL, mem["stag"] + 1, 0).astype(jnp.int32)
    return action, {**mem, "mode": mode, "a0": a0, "aE": aE, "aA": aA, "aB": aB,
                    "pol": pol, "hold": hold, "left": left, "tick": tick, "prevz": z, "stag": stag}
def archive(mem, feats, done, arch_feats, arch_valid, rng):
    B = feats.shape[0]
    lo, hi, KR = mem["lo"], mem["hi"], mem["KR"]
    # --- frontier pool: per-dim records, content-addressed so each axis keeps its own persistent slot
    tol = REC_TOL * jnp.maximum(hi - lo, 1e-3) + 1e-6            # margin relative to span: monotone dims fire ~1/REC_TOL times
    idx = jnp.arange(B)[:, None]
    top = jnp.argmax(feats, 0)[None, :]; bot = jnp.argmin(feats, 0)[None, :]   # one credited worker per dim
    hit = (((feats > hi + tol) & (top == idx)) | ((feats < lo - tol) & (bot == idx))) & mem["rmask"][None, :]
    rec = hit.any(-1) & ~done
    rec_dim = jnp.argmax(hit, -1).astype(jnp.int32)
    is_max = (feats[jnp.arange(B), rec_dim] > hi[rec_dim]).astype(jnp.int32)
    rec_slot = (2 * rec_dim + is_max) % KR                       # slots [0, KR): max and min of a dim live apart
    # --- cell pool: hash-novelty with normalised-energy+progress overwrite (disjoint from the frontier pool)
    n_cell = mem["K"] - KR
    cell = _cell(mem, feats)
    cur_slot = mem["cell_slot"][cell]
    is_empty = cur_slot < 0
    safe = jnp.clip(cur_slot, KR, arch_feats.shape[0] - 1)
    val = _energy(mem, feats) + PROG_CELL_W * _progress(mem, feats)
    val_a = _energy(mem, arch_feats) + PROG_CELL_W * _progress(mem, arch_feats)
    better = (~is_empty) & (val > val_a[safe] + ENERGY_MARGIN)   # a cell keeps its deepest/hottest occupant
    fresh = is_empty & ~done & ~rec                              # record workers do not also claim a cell
    cell_store = (is_empty | better) & ~done & ~rec
    new_slot = KR + (mem["next_free"] + jnp.cumsum(fresh) - 1) % n_cell
    cslot = jnp.where(is_empty, new_slot, safe)
    store = rec | cell_store
    slot = jnp.clip(jnp.where(rec, rec_slot, cslot), 0, mem["K"] - 1).astype(jnp.int32)
    cell_slot = mem["cell_slot"].at[jnp.where(fresh, cell, 0)].set(jnp.where(fresh, new_slot, mem["cell_slot"][0]))
    # --- ratchet detector: tally up- vs down-moves per dim (first step, teleports and resets excluded)
    nlo, nhi = jnp.minimum(lo, feats.min(0)), jnp.maximum(hi, feats.max(0))
    d = (feats - mem["pf"]) / jnp.maximum(nhi - nlo, 1e-6)
    valid_step = (~(mem["res"] | done))[:, None] & (mem["t"] > 0) & (nhi > nlo)[None, :]
    inc = RAT_DECAY * mem["inc"] + ((d > RAT_EPS) & valid_step).sum(0).astype(jnp.float32)
    dec = RAT_DECAY * mem["dec"] + ((d < -RAT_EPS) & valid_step).sum(0).astype(jnp.float32)
    # --- dense progress credit: reward steady climbs along ratchet dims, not just brand-new global records
    prog_gain = jnp.where(~(mem["res"] | done) & (mem["t"] > 0),
                           jnp.clip(_progress(mem, feats) - _progress(mem, mem["pf"]), 0.0, None), 0.0)
    # --- rare-dim detector: per dim, how often does it move at all? Positions/velocities/time move every step
    # (rate ~ 1 -> rarity ~ 0); inventory ticks, achievement flags and contact flags move on a handful of steps
    # (rate << 1 -> rarity ~ 1). Scored against the PREVIOUS tallies so a dim's first-ever move pays out.
    moved = (jnp.abs(d) > RARE_MOVE_EPS) & valid_step
    surp = jnp.clip(1.0 / (mem["chg"] / jnp.maximum(mem["cok"], 1.0) + RARE_SMOOTH), 0.0, RARE_CAP) / RARE_CAP
    warm = (mem["cok"] > RARE_WARM).astype(jnp.float32)
    # max, not sum: one crafting step flips several correlated dims and should be credited once
    rare_gain = (moved.astype(jnp.float32) * surp * warm).max(-1)
    chg = CHG_DECAY * mem["chg"] + moved.sum(0).astype(jnp.float32)
    cok = CHG_DECAY * mem["cok"] + valid_step.sum(0).astype(jnp.float32)
    nrate = chg / jnp.maximum(cok, 1.0)
    rarew = (jnp.clip(1.0 / (nrate + RARE_SMOOTH), 0.0, RARE_CAP) / RARE_CAP) * (cok > RARE_WARM)
    # --- rare-record bias: dims whose frontier is rarely pushed get a bigger reward and restore bonus
    rec_rate = REC_RATE_DECAY * mem["rec_rate"] + hit.astype(jnp.float32).sum(0)
    rarity = jnp.clip(1.0 / (rec_rate + RARE_EPS), 0.0, RARE_CLIP)
    rarity_g = jnp.take(rarity, rec_dim)
    rec_val = mem["rec_val"].at[jnp.where(rec, rec_slot, 0)].set(
        jnp.where(rec, rarity_g, mem["rec_val"][jnp.where(rec, rec_slot, 0)]))
    # --- bandit credit assignment: which behaviour produced this step's novelty?
    r = R_REC * (1.0 + R_REC_RARE_W * rarity_g) * rec.astype(jnp.float32) \
        + R_CELL * fresh.astype(jnp.float32) + R_PROG * prog_gain + R_RARE * rare_gain
    num = BANDIT_DECAY * mem["num"] + jnp.zeros(N_MODE, jnp.float32).at[mem["mode"]].add(r)
    den = BANDIT_DECAY * mem["den"] + jnp.zeros(N_MODE, jnp.float32).at[mem["mode"]].add(jnp.ones(B, jnp.float32))
    return store, slot, {**mem, "cell_slot": cell_slot, "num": num, "den": den,
                         "inc": inc, "dec": dec, "pf": feats, "res": done, "t": mem["t"] + 1,
                         "rec_rate": rec_rate, "rec_val": rec_val,
                         "chg": chg, "cok": cok, "rarew": rarew,
                         "lo": nlo, "hi": nhi, "next_free": (mem["next_free"] + fresh.sum()) % n_cell,
                         "left": jnp.where(done, 0, mem["left"]),           # episode end -> fresh behaviour
                         "tick": jnp.where(done, 0, mem["tick"]),
                         "stag": jnp.where(done, 0, mem["stag"])}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """Called by the evaluation harness; returns the search-split fitness. Do not edit."""
    import sys, os
    sys.path.insert(0, os.environ.get("AUTODEV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from autodev.search_eval import evaluate_hooks
    return evaluate_hooks(init, restore, act, archive, **kwargs)